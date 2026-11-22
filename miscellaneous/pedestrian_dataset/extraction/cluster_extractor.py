"""Export detection-matched LiDAR samples, camera images, and Stage-1 metadata."""

from __future__ import annotations

import csv
import logging
import os
import shutil
from copy import copy
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any

import numpy as np
import yaml

from bag_reader import decode_message, iter_serialized_messages, make_parser
from extraction.calibration_extractor import extract_calibration
from extraction.camera_extractor import build_camera_output_path, write_camera_frame
from extraction.detection_matching import (
    BagIndex,
    CameraMatches,
    CloudMatches,
    DetectionMatches,
    find_detection_slice,
    index_bag_messages,
    match_cameras,
    match_requested_clouds,
    match_sampled_clouds,
)
from extraction.ego_pose_extractor import find_ego_pose_at
from extraction.lidar_extractor import _write_pcd, build_lidar_output_path, read_cloud_points
from file_writer.csv_writer import write_csv
from file_writer.detection_confidence_csv_writer import write_detection_confidence_csv
from file_writer.detection_label_writer import write_detection_labels
from file_writer.detection_point_count_csv_writer import write_detection_point_count_csv
from file_writer.ego_pose_csv_writer import write_ego_pose_csv
from timestamp_matching.matcher import SampleGroup
from utils.config_loader import CameraConfig, ClusterConfig
from utils.pcd_reader import read_pcd
from utils.timestamp_utils import message_timestamp_ns, ns_to_filename

logger = logging.getLogger(__name__)

DetectionBoxes = list[tuple[Any, np.ndarray]]


@dataclass
class FrameMetadata:
    confidences: dict[int, dict[int, float]] = field(default_factory=dict)
    point_counts: dict[int, dict[int, int]] = field(default_factory=dict)


def run_cluster_extraction(config: ClusterConfig, files: list[str]) -> str:
    """Extract clusters using the shared detection-driven Stage-1 pipeline."""
    return run_detection_extraction(config, files, cluster=True)


def run_detection_extraction(
    config: ClusterConfig,
    files: list[str],
    cluster: bool = False,
    timestamp_groups: list[SampleGroup] | None = None,
) -> str:
    """Export matched samples from a bag or reuse samples from a prior export."""
    if config.reuse_dataset:
        if timestamp_groups is None:
            raise ValueError("Sample reuse requires timestamps.csv")
        return _run_detection_reuse(config, files, timestamp_groups)

    logger.info(">>> STAGE 1: FROM DETECTIONS (%s)", "clusters" if cluster else "full clouds")
    logger.info("BAG input   : %s (%d files)", config.bag_dir, len(files))
    logger.info("Output dir  : %s", config.output_base_dir)

    index = index_bag_messages(config, files)
    if not index.clouds or (not index.detections and timestamp_groups is None):
        raise ValueError("The bag must contain messages on both the LiDAR and detection topics")
    if not len(index.ego_timestamps):
        logger.warning("No ego poses found; ego_poses.csv will have empty numeric cells.")

    # Sampled runs start from detections. CSV runs start from the requested LiDAR frames.
    requested_groups = None
    if timestamp_groups is None:
        cloud_matches, unmatched = match_sampled_clouds(index, config)
    else:
        requested_groups = {
            group.lidar_timestamps[config.lidar_key]: group for group in timestamp_groups
        }
        cloud_matches, unmatched = match_requested_clouds(
            index, requested_groups, config.matching.tolerance_ms,
        )

    camera_matches = match_cameras(index, config, cloud_matches, requested_groups)
    _export_bag_samples(config, files, cluster, index, cloud_matches, camera_matches, unmatched)
    return config.output_base_dir


def _export_bag_samples(
    config: ClusterConfig,
    files: list[str],
    cluster: bool,
    index: BagIndex,
    cloud_matches: CloudMatches,
    camera_matches: CameraMatches,
    unmatched: int,
) -> None:
    metadata = FrameMetadata()
    exported_images = {camera.key: set() for camera in config.cameras}
    written_clouds = 0
    written_images = 0
    lidar_frame = None
    lidar_to_reference = None

    # Stream each selected topic once per split and decode only matched payloads.
    for bag_path in files:
        file_clouds = cloud_matches.get(bag_path, {})
        file_images = camera_matches.by_file.get(bag_path, {})
        if not file_clouds and not file_images:
            continue

        topics = {topic for topic, _ in file_images}
        if file_clouds:
            topics.add(config.lidar_topic)
        message_numbers = dict.fromkeys(topics, 0)
        parser = make_parser(bag_path)
        logger.info(
            "Pass 2: extracting %d matched clouds and %d images from %s ...",
            len(file_clouds), len(file_images), os.path.basename(bag_path),
        )
        try:
            for topic, _, type_name, payload in iter_serialized_messages(parser, sorted(topics)):
                message_number = message_numbers[topic]
                message_numbers[topic] += 1

                if topic == config.lidar_topic:
                    matches = file_clouds.pop(message_number, None)
                    if matches is None:
                        continue
                    cloud = decode_message(parser, topic, type_name, payload)
                    cloud_frame = cloud.header.frame_id.lstrip("/")

                    # The first exported cloud establishes the calibration for the dataset.
                    if lidar_frame is None:
                        lidar_frame = config.lidar_frame_id or cloud.header.frame_id
                        if lidar_frame.lstrip("/") != cloud_frame:
                            raise ValueError("Configured LiDAR frame_id differs from the exported cloud frame")
                        lidar_to_reference = _export_calibration(config, index, lidar_frame, camera_matches.frame_ids)
                    elif lidar_frame.lstrip("/") != cloud_frame:
                        raise ValueError("Exported point clouds must share the configured LiDAR frame")

                    written_clouds += _export_bag_cloud(
                        config, cloud, matches, cluster, lidar_to_reference, metadata,
                    )
                    del cloud
                else:
                    matches = file_images.pop((topic, message_number), None)
                    if matches is None:
                        continue
                    image = decode_message(parser, topic, type_name, payload)
                    written_images += _export_bag_image(config, image, matches, exported_images)
                    del image

                if not file_clouds and not file_images:
                    break

            if file_clouds or file_images:
                raise ValueError(f"Matched point clouds or images missing from {bag_path}")
        finally:
            del parser

    _write_bag_metadata(config, index, metadata, camera_matches, exported_images)
    logger.info(
        "Detection extraction complete: %d PCD files, %d images written, %d unmatched messages",
        written_clouds, written_images, unmatched,
    )


def _export_calibration(
    config: ClusterConfig,
    index: BagIndex,
    lidar_frame: str,
    camera_frames: list[str],
):
    # Preserve the order of sensor links and request each link only once.
    sensor_links = list(dict.fromkeys([lidar_frame, config.ego_pose.child_frame, *camera_frames]))
    calibration = extract_calibration(
        urdf_path=config.urdf_path,
        intrinsics_yaml_path=config.calibration_yaml,
        stage1_out=config.output_base_dir,
        reference_frame=config.reference_frame or config.ego_pose.child_frame,
        sensor_links=sensor_links,
        camera_frame_ids=camera_frames,
        tf_static=index.static_transforms,
        camera_info=index.camera_info,
    )
    return calibration["extrinsics"].get(lidar_frame)


def _export_bag_image(
    config: ClusterConfig,
    image,
    matches: dict[str, tuple[CameraConfig, int]],
    exported_images: dict[str, set[int]],
) -> int:
    written = 0
    for camera, timestamp_ns in matches.values():
        path = build_camera_output_path(config.output_base_dir, "", camera.key, timestamp_ns)
        written += write_camera_frame(image, path, camera.compressed, config.on_existing)

        # Files kept by the skip policy still belong in the timestamps CSV.
        if os.path.isfile(path):
            exported_images[camera.key].add(timestamp_ns)
    return written


def _export_bag_cloud(
    config: ClusterConfig,
    cloud,
    matches: DetectionMatches,
    cluster: bool,
    lidar_to_reference,
    metadata: FrameMetadata,
) -> int:
    cloud_frame = cloud.header.frame_id.lstrip("/")
    for _, message in matches:
        if message.header.frame_id.lstrip("/") != cloud_frame:
            raise ValueError("Point cloud and detections must be in the same frame")

    timestamp_ns = message_timestamp_ns(cloud)
    detections = _frame_detections(matches, timestamp_ns, config.target_classes)
    xyz = None
    written = 0

    if cluster:
        if detections:
            xyz = _xyz_coordinates(read_cloud_points(cloud))
            written = _write_bag_clusters(
                cloud, detections, xyz, timestamp_ns, config.output_base_dir, config.on_existing,
            )
    else:
        written = _write_lidar_sample(
            cloud, timestamp_ns, config.output_base_dir, config.on_existing, config.lidar_key,
        )

    _write_frame_labels(config, timestamp_ns, detections, lidar_to_reference, metadata)
    if detections:
        if xyz is None:
            xyz = _xyz_coordinates(read_cloud_points(cloud))
        metadata.point_counts[timestamp_ns] = _count_points(detections, xyz)
    else:
        metadata.point_counts[timestamp_ns] = {}
    return written


def _write_bag_metadata(
    config: ClusterConfig,
    index: BagIndex,
    metadata: FrameMetadata,
    camera_matches: CameraMatches,
    exported_images: dict[str, set[int]],
) -> None:
    groups = []
    ego_poses = []
    for timestamp_ns in sorted(metadata.confidences):
        # Leave the camera cell empty if the selected image could not be written.
        cameras = {}
        for key, image_timestamp in camera_matches.timestamps_by_cloud[timestamp_ns].items():
            cameras[key] = image_timestamp if image_timestamp in exported_images[key] else None
        groups.append(SampleGroup({config.lidar_key: timestamp_ns}, cameras))
        ego_poses.append(find_ego_pose_at(
            timestamp_ns, index.ego_timestamps, index.ego_poses, config.matching.ego_pose_tolerance_ms,
        ))

    output_dir = config.output_base_dir
    write_csv(groups, os.path.join(output_dir, "timestamps.csv"))
    write_ego_pose_csv(groups, ego_poses, config.lidar_key, os.path.join(output_dir, "ego_poses.csv"))
    _write_detection_csvs(output_dir, metadata)


def _run_detection_reuse(
    config: ClusterConfig,
    files: list[str],
    timestamp_groups: list[SampleGroup],
) -> str:
    """Read detections from the current bags and reuse sensors from a prior export."""
    detections = _read_reuse_detections(config.detection_topic, files)
    detection_timestamps = np.array([timestamp for timestamp, _ in detections], dtype=np.int64)
    source_root = config.reuse_dataset
    output_root = config.output_base_dir
    source_poses, lidar_frame, lidar_to_reference = _prepare_reuse_calibration(config, source_root)
    metadata = FrameMetadata()
    groups_by_timestamp = {}

    for group in timestamp_groups:
        timestamp_ns = group.lidar_timestamps[config.lidar_key]
        source = build_lidar_output_path(source_root, "", config.lidar_key, timestamp_ns)
        if not os.path.isfile(source):
            logger.warning("Missing reused LiDAR sample at ts=%d", timestamp_ns)
            continue

        detection_slice = find_detection_slice(
            timestamp_ns, detection_timestamps, config.matching.tolerance_ms,
        )
        if detection_slice is None:
            logger.warning("No matching detections for reused LiDAR sample at ts=%d", timestamp_ns)
            continue

        matches = detections[detection_slice]
        detection_frame = matches[0][1].header.frame_id.lstrip("/")
        if lidar_frame and detection_frame != lidar_frame:
            raise ValueError(
                f"Detection frame {detection_frame} does not match configured LiDAR frame {lidar_frame}",
            )

        selected = _frame_detections(matches, timestamp_ns, config.target_classes)
        points = read_pcd(source)
        xyz = _xyz_coordinates(points)
        metadata.point_counts[timestamp_ns] = _count_points(selected, xyz)
        if config.cluster:
            _write_reused_clusters(config, points, xyz, selected, timestamp_ns)
        else:
            destination = build_lidar_output_path(output_root, "", config.lidar_key, timestamp_ns)
            _place_sample(source, destination, config)

        camera_timestamps = _reuse_camera_samples(config, source_root, group)
        _write_frame_labels(config, timestamp_ns, selected, lidar_to_reference, metadata)
        # With repeated CSV timestamps, the last processed row supplies the metadata.
        groups_by_timestamp[timestamp_ns] = SampleGroup(
            {config.lidar_key: timestamp_ns}, camera_timestamps,
        )

    ordered_timestamps = sorted(groups_by_timestamp)
    groups = [groups_by_timestamp[timestamp_ns] for timestamp_ns in ordered_timestamps]
    write_csv(groups, os.path.join(output_root, "timestamps.csv"))
    _copy_ego_pose_rows(source_poses, os.path.join(output_root, "ego_poses.csv"), ordered_timestamps)
    _write_detection_csvs(output_root, metadata)
    logger.info("Detection reuse complete: %d matched frames", len(groups))
    return output_root


def _read_reuse_detections(topic: str, files: list[str]) -> DetectionMatches:
    detections = []
    for bag_path in files:
        parser = make_parser(bag_path)
        try:
            for topic_name, _, type_name, payload in iter_serialized_messages(parser, [topic]):
                message = decode_message(parser, topic_name, type_name, payload)
                detections.append((message_timestamp_ns(message), message))
        finally:
            del parser

    # Equal header stamps retain the bag order in which they were read.
    detections.sort(key=itemgetter(0))
    return detections


def _prepare_reuse_calibration(config: ClusterConfig, source_root: str) -> tuple[str, str, np.ndarray | None]:
    output_root = config.output_base_dir
    source_path = os.path.abspath(source_root)
    output_path = os.path.abspath(output_root)
    if os.path.commonpath((source_path, output_path)) == source_path:
        raise ValueError("Reuse output.base_dir must be outside the source dataset")

    source_calibration = os.path.join(source_root, "calibration.yaml")
    if not os.path.isfile(source_calibration):
        raise FileNotFoundError(f"Reuse dataset has no calibration.yaml: {source_root}")
    source_poses = os.path.join(source_root, "ego_poses.csv")
    if not os.path.isfile(source_poses):
        raise FileNotFoundError(f"Reuse dataset has no ego_poses.csv: {source_root}")

    os.makedirs(output_root, exist_ok=True)
    output_calibration = os.path.join(output_root, "calibration.yaml")
    if os.path.lexists(output_calibration):
        if config.on_existing == "abort":
            raise FileExistsError(output_calibration)
        if config.on_existing == "overwrite":
            os.unlink(output_calibration)
    if not os.path.exists(output_calibration):
        shutil.copy2(source_calibration, output_calibration)

    # Labels use the transform from the source dataset even if output calibration is kept.
    with open(source_calibration) as stream:
        calibration = yaml.safe_load(stream) or {}
    lidar_frame = (config.lidar_frame_id or "").lstrip("/")
    lidar_transform = None
    for frame, transform in calibration.get("extrinsics", {}).items():
        if frame.lstrip("/") == lidar_frame:
            lidar_transform = transform
            break

    if lidar_frame and lidar_transform is None:
        raise ValueError(f"Reuse calibration has no extrinsic for LiDAR frame '{lidar_frame}'")
    if lidar_transform is not None:
        lidar_transform = np.asarray(lidar_transform)
    return source_poses, lidar_frame, lidar_transform


def _reuse_camera_samples(
    config: ClusterConfig,
    source_root: str,
    group: SampleGroup,
) -> dict[str, int | None]:
    timestamps = {}
    for camera in config.cameras:
        timestamps[camera.key] = None
        timestamp_ns = group.camera_timestamps.get(camera.key)
        if timestamp_ns is None:
            continue

        source = build_camera_output_path(source_root, "", camera.key, timestamp_ns)
        if not os.path.isfile(source):
            logger.warning("Missing reused camera %s image at ts=%d", camera.key, timestamp_ns)
            continue

        destination = build_camera_output_path(config.output_base_dir, "", camera.key, timestamp_ns)
        _place_sample(source, destination, config)
        timestamps[camera.key] = timestamp_ns
    return timestamps


def _place_sample(source: str, destination: str, config: ClusterConfig) -> bool:
    # Unlink the destination first so overwrite cannot write through an old symlink.
    if os.path.lexists(destination):
        if config.on_existing == "skip":
            return False
        if config.on_existing == "abort":
            raise FileExistsError(destination)
        os.unlink(destination)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if config.copy_samples:
        shutil.copy2(source, destination)
    else:
        os.symlink(os.path.abspath(source), destination)
    return True


def _copy_ego_pose_rows(source: str, destination: str, timestamps: list[int]) -> None:
    # Keep the source CSV schema and row order when filtering to exported frames.
    wanted = {ns_to_filename(timestamp_ns) for timestamp_ns in timestamps}
    with open(source, newline="") as src, open(destination, "w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row.get("stamp") in wanted:
                writer.writerow(row)


def detection_to_bbox(detection) -> np.ndarray:
    """Convert a measurement to [cx, cy, cz, length, width, height, yaw]."""
    bbox = np.array(
        [
            detection.x_cog,
            detection.y_cog,
            detection.z_cog,
            detection.length,
            detection.width,
            detection.height,
            detection.orientation,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(bbox).all() or np.any(bbox[3:6] <= 0):
        raise ValueError("Bounding box must have finite values and positive dimensions")
    return bbox


def points_in_bbox(xyz: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Include box boundaries and exclude nonfinite points. Yaw is in radians."""
    cosine = np.cos(bbox[6])
    sine = np.sin(bbox[6])
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])

    # Work in box-local coordinates and leave invalid input points masked out.
    mask = np.isfinite(xyz).all(axis=1)
    local_points = (xyz[mask] - bbox[:3]) @ rotation
    half_dimensions = bbox[3:6] / 2
    mask[mask] = (np.abs(local_points) <= half_dimensions + 1e-9).all(axis=1)
    return mask


def extract_detection_clusters(
    cloud_msg,
    detection_msg,
    timestamp_ns: int,
    output_dir: str,
    on_existing: str = "skip",
    target_classes: list[int] | None = None,
) -> int:
    """Write one binary PCD per valid detection, preserving all point fields."""
    if cloud_msg.header.frame_id.lstrip("/") != detection_msg.header.frame_id.lstrip("/"):
        raise ValueError("Point cloud and detections must be in the same frame")

    detections = _selected_detections(detection_msg, target_classes)
    if not detections:
        return 0

    xyz = _xyz_coordinates(read_cloud_points(cloud_msg))
    written = 0
    for detection in detections:
        path = _cluster_path(output_dir, timestamp_ns, detection)
        if not _should_write_pcd(path, on_existing):
            continue

        # Honor the output policy before validating boxes for the standalone caller.
        try:
            bbox = detection_to_bbox(detection)
        except ValueError as exc:
            logger.warning("Skipping detection %d at ts=%d: %s", detection.box_id, timestamp_ns, exc)
            continue
        _write_cloud_subset(cloud_msg, path, points_in_bbox(xyz, bbox))
        written += 1
    return written


def _active_detections(message):
    if not 0 <= message.count <= len(message.meas_array):
        raise ValueError("Detection count exceeds measurement array length")
    return message.meas_array[:message.count]


def _has_duplicate_ids(detections) -> bool:
    return len({int(detection.box_id) for detection in detections}) != len(detections)


def _selected_detections(message, target_classes: list[int] | None):
    active = _active_detections(message)
    use_local_ids = _has_duplicate_ids(active)
    detections = []
    for index, detection in enumerate(active):
        if target_classes is not None and detection.meas_class not in target_classes:
            continue
        if use_local_ids:
            # Index before filtering and copy before setting IDs on the exported objects.
            detection = copy(detection)
            detection.box_id = index
        detections.append(detection)
    return detections


def _frame_detections(
    matches: DetectionMatches,
    cloud_timestamp: int,
    target_classes: list[int] | None,
) -> DetectionBoxes:
    """Merge unique IDs or use the nearest message when IDs are local to a message."""
    has_local_ids = any(_has_duplicate_ids(_active_detections(message)) for _, message in matches)
    if has_local_ids:
        # Array indices cannot associate objects across detection messages.
        _, _, closest_index = min(
            (abs(timestamp - cloud_timestamp), timestamp, index)
            for index, (timestamp, _) in enumerate(matches)
        )
        matches = [matches[closest_index]]

    selected = {}
    for detection_timestamp, message in matches:
        rank = (abs(detection_timestamp - cloud_timestamp), detection_timestamp)
        for detection in _selected_detections(message, target_classes):
            try:
                bbox = detection_to_bbox(detection)
            except ValueError as exc:
                logger.warning("Skipping detection %d at ts=%d: %s", detection.box_id, detection_timestamp, exc)
                continue
            detection_id = int(detection.box_id)
            previous = selected.get(detection_id)
            if previous is None or rank < previous[0]:
                selected[detection_id] = (rank, detection, bbox)

    boxes = []
    for detection_id in sorted(selected):
        _, detection, bbox = selected[detection_id]
        boxes.append((detection, bbox))
    return boxes


def _xyz_coordinates(points: np.ndarray) -> np.ndarray:
    return np.column_stack((points["x"], points["y"], points["z"])).astype(np.float64)


def _count_points(detections: DetectionBoxes, xyz: np.ndarray) -> dict[int, int]:
    counts = {}
    for detection, bbox in detections:
        counts[int(detection.box_id)] = int(np.count_nonzero(points_in_bbox(xyz, bbox)))
    return counts


def _write_frame_labels(
    config: ClusterConfig,
    timestamp_ns: int,
    detections: DetectionBoxes,
    transform,
    metadata: FrameMetadata,
) -> None:
    # Keep confidence values in the same detection ID space as the exported labels.
    write_detection_labels(
        detections, timestamp_ns, config.output_base_dir, config.on_existing, transform=transform,
    )
    metadata.confidences[timestamp_ns] = {
        int(detection.box_id): float(detection.score) for detection, _ in detections
    }


def _write_detection_csvs(output_dir: str, metadata: FrameMetadata) -> None:
    write_detection_confidence_csv(
        metadata.confidences, os.path.join(output_dir, "detection_confidences.csv"),
    )
    write_detection_point_count_csv(
        metadata.point_counts, os.path.join(output_dir, "detection_point_counts.csv"),
    )


def _cluster_path(output_dir: str, timestamp_ns: int, detection) -> str:
    class_id = str(int(detection.meas_class))
    filename = f"{ns_to_filename(timestamp_ns)}_{int(detection.box_id)}.pcd"
    return os.path.join(output_dir, "clusters", class_id, filename)


def _should_write_pcd(path: str, on_existing: str) -> bool:
    if os.path.exists(path):
        if on_existing == "skip":
            return False
        if on_existing == "abort":
            raise FileExistsError(f"Output file already exists: {path}")
    return True


def _write_lidar_sample(
    cloud_msg, timestamp_ns: int, output_dir: str, on_existing: str, lidar_key: str = "lidar",
) -> bool:
    path = build_lidar_output_path(output_dir, "", lidar_key, timestamp_ns)
    if not _should_write_pcd(path, on_existing):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_pcd(cloud_msg, path)
    return True


def _write_bag_clusters(
    cloud,
    detections: DetectionBoxes,
    xyz: np.ndarray,
    timestamp_ns: int,
    output_dir: str,
    on_existing: str,
) -> int:
    written = 0
    for detection, bbox in detections:
        path = _cluster_path(output_dir, timestamp_ns, detection)
        if not _should_write_pcd(path, on_existing):
            continue
        _write_cloud_subset(cloud, path, points_in_bbox(xyz, bbox))
        written += 1
    return written


def _write_cloud_subset(cloud, path: str, mask: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_pcd(cloud, path, point_indices=mask)


def _write_reused_clusters(
    config: ClusterConfig,
    points: np.ndarray,
    xyz: np.ndarray,
    detections: DetectionBoxes,
    timestamp_ns: int,
) -> None:
    for detection, bbox in detections:
        path = _cluster_path(config.output_base_dir, timestamp_ns, detection)
        if os.path.lexists(path):
            if config.on_existing == "skip":
                continue
            if config.on_existing == "abort":
                raise FileExistsError(path)
        _write_pcd_subset(path, points[points_in_bbox(xyz, bbox)])


def _write_pcd_subset(destination: str, points: np.ndarray) -> None:
    # Preserve every field's scalar type and element count in the cropped PCD.
    fields = points.dtype.names
    sizes = []
    counts = []
    types = []
    type_codes = {"f": "F", "i": "I"}
    for name in fields:
        dtype = points.dtype.fields[name][0]
        if dtype.subdtype is None:
            base_dtype = dtype
            count = 1
        else:
            base_dtype, shape = dtype.subdtype
            count = int(np.prod(shape))
        sizes.append(str(base_dtype.itemsize))
        counts.append(str(count))
        types.append(type_codes.get(base_dtype.kind, "U"))

    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS " + " ".join(fields),
        "SIZE " + " ".join(sizes),
        "TYPE " + " ".join(types),
        "COUNT " + " ".join(counts),
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA binary",
    ]
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        stream.write(points.tobytes())
