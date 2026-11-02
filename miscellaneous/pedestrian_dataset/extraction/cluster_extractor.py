"""
Extract detection-matched full clouds or clusters, with Stage-1 3-D labels.
Output paths: samples/<lidar_key>/<timestamp>.pcd or clusters/<class_id>/<timestamp>_<detection_id>.pcd.

Matching uses message header.stamp; exported frames use cloud header.stamp. The first pass
indexes clouds and retains detections and ego poses; the second streams clouds.
"""

from __future__ import annotations

import logging
import os
from copy import copy

import numpy as np
from rclpy.serialization import deserialize_message

from bag_reader import make_parser
from extraction.ego_pose_extractor import iter_ego_poses, find_ego_pose_at
from extraction.lidar_extractor import _write_pcd, read_cloud_points, build_lidar_output_path
from file_writer.csv_writer import write_csv
from file_writer.detection_label_writer import write_detection_labels
from file_writer.detection_confidence_csv_writer import write_detection_confidence_csv
from file_writer.ego_pose_csv_writer import write_ego_pose_csv
from timestamp_matching.matcher import SampleGroup, _find_closest_within
from utils.timestamp_utils import ns_to_filename

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detection_to_bbox(detection) -> np.ndarray:
    """Convert a measurement to [cx, cy, cz, length, width, height, yaw]."""
    bbox = np.array([
        detection.x_cog, detection.y_cog, detection.z_cog,
        detection.length, detection.width, detection.height, detection.orientation,
    ], dtype=np.float64)
    if not np.isfinite(bbox).all() or np.any(bbox[3:6] <= 0):
        raise ValueError("Bounding box must have finite values and positive dimensions")
    return bbox


def points_in_bbox(xyz: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Include box boundaries and exclude nonfinite points; yaw is in radians."""
    c, s = np.cos(bbox[6]), np.sin(bbox[6])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mask = np.isfinite(xyz).all(axis=1)
    local = (xyz[mask] - bbox[:3]) @ rotation
    mask[mask] = (np.abs(local) <= bbox[3:6] / 2 + 1e-9).all(axis=1)
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

    points = read_cloud_points(cloud_msg)
    xyz = np.column_stack([points[name] for name in ("x", "y", "z")]).astype(np.float64)
    written = 0
    for detection in detections:
        detection_id = int(detection.box_id)
        cluster_dir = os.path.join(output_dir, "clusters", str(int(detection.meas_class)))
        out_path = os.path.join(cluster_dir, f"{ns_to_filename(timestamp_ns)}_{detection_id}.pcd")
        if os.path.exists(out_path):
            if on_existing == "skip":
                continue
            elif on_existing == "abort":
                raise FileExistsError(f"Output file already exists: {out_path}")

        try:
            bbox = detection_to_bbox(detection)
        except ValueError as exc:
            logger.warning("Skipping detection %d at ts=%d: %s", detection_id, timestamp_ns, exc)
            continue
        mask = points_in_bbox(xyz, bbox)
        os.makedirs(cluster_dir, exist_ok=True)
        _write_pcd(cloud_msg, out_path, point_indices=mask)
        written += 1
    return written


def run_cluster_extraction(config, files: list[str]) -> str:
    """Extract clusters using the shared detection-driven Stage-1 pipeline."""
    return run_detection_extraction(config, files, cluster=True)


def run_detection_extraction(config, files: list[str], cluster: bool = False) -> str:
    """Match detections to clouds and export samples or clusters plus frame labels."""
    stage1_out = config.output_base_dir

    logger.info(">>> STAGE 1: FROM DETECTIONS (%s)", "clusters" if cluster else "full clouds")
    logger.info("BAG input   : %s (%d files)", config.bag_dir, len(files))
    logger.info("Output dir  : %s", stage1_out)

    lidar_index = []
    detection_index = []
    ego_ts = []
    ego_poses = []
    topics = sorted({config.lidar_topic, config.detection_topic, config.ego_pose.topic})

    # Pass 1: collect all topics together and retain only the small messages.
    for fpath in files:
        parser = make_parser(fpath)
        logger.info("Pass 1: indexing headers, detections and ego poses from %s ...", os.path.basename(fpath))
        cloud_number = 0
        try:
            for topic, recording_timestamp, type_name, payload in _iter_serialized_messages(parser, topics):
                message = _decode_message(parser, topic, type_name, payload)
                if topic == config.ego_pose.topic:
                    for ts, pose in iter_ego_poses(message, config.ego_pose.parent_frame, config.ego_pose.child_frame):
                        ego_ts.append(ts)
                        ego_poses.append(pose)
                else:
                    stamp = message.header.stamp
                    timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                    if topic == config.lidar_topic:
                        lidar_index.append((timestamp_ns, recording_timestamp, fpath, cloud_number))
                        cloud_number += 1
                    else:
                        detection_index.append((timestamp_ns, recording_timestamp, fpath, message))
                del message
        finally:
            del parser

    if not lidar_index or not detection_index:
        raise ValueError("The bag must contain messages on both the LiDAR and detection topics")
    lidar_index.sort()
    detection_index.sort(key=lambda entry: entry[:3])
    lidar_ts = np.array([entry[0] for entry in lidar_index], dtype=np.int64)

    if ego_ts:
        all_ego_ts = np.array(ego_ts, dtype=np.int64)
        all_ego_data = np.array(ego_poses, dtype=np.float64)
        order = np.argsort(all_ego_ts)
        all_ego_ts = all_ego_ts[order]
        all_ego_data = all_ego_data[order]
    else:
        all_ego_ts = np.array([], dtype=np.int64)
        all_ego_data = np.zeros((0, 7), dtype=np.float64)
        logger.warning("No ego poses found; ego_poses.csv will have empty numeric cells.")

    # Match headers globally before extracting, including across bag boundaries.
    cloud_detections = {}
    unmatched = 0
    for timestamp_ns, _, _, detection_msg in detection_index:
        idx = _find_closest_within(timestamp_ns, lidar_ts, config.tolerance_ms)
        if idx is None:
            unmatched += 1
            logger.warning("No matching point cloud for detections at ts=%d", timestamp_ns)
            continue
        _, _, cloud_path, cloud_number = lidar_index[idx]
        file_matches = cloud_detections.setdefault(cloud_path, {})
        file_matches.setdefault(cloud_number, []).append((timestamp_ns, detection_msg))

    # Pass 2: stream clouds and finish every associated detection before advancing.
    written = 0
    frame_confidences = {}
    for fpath in files:
        file_matches = cloud_detections.get(fpath)
        if not file_matches:
            continue
        parser = make_parser(fpath)
        logger.info("Pass 2: extracting from %d matched clouds in %s ...", len(file_matches), os.path.basename(fpath))
        try:
            for cloud_number, (topic, _, type_name, payload) in enumerate(
                _iter_serialized_messages(parser, [config.lidar_topic])
            ):
                matches = file_matches.pop(cloud_number, None)
                if matches is None:
                    continue
                cloud_msg = _decode_message(parser, topic, type_name, payload)
                stamp = cloud_msg.header.stamp
                timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                for _, detection_msg in matches:
                    if cloud_msg.header.frame_id.lstrip("/") != detection_msg.header.frame_id.lstrip("/"):
                        raise ValueError("Point cloud and detections must be in the same frame")
                detections = _frame_detections(matches, timestamp_ns, config.target_classes)
                if cluster:
                    detection_msg = copy(matches[0][1])
                    detection_msg.meas_array = [detection for detection, _ in detections]
                    detection_msg.count = len(detections)
                    written += extract_detection_clusters(cloud_msg, detection_msg, timestamp_ns, stage1_out, config.on_existing)
                else:
                    written += _write_lidar_sample(cloud_msg, timestamp_ns, stage1_out, config.on_existing, config.lidar_key)
                write_detection_labels(detections, timestamp_ns, stage1_out, config.on_existing)
                frame_confidences[timestamp_ns] = {int(detection.box_id): float(detection.score) for detection, _ in detections}
                del cloud_msg
                if not file_matches:
                    break
            if file_matches:
                raise ValueError(f"Matched point clouds missing from {fpath}")
        finally:
            del parser

    ref_key = config.lidar_key
    groups = [SampleGroup(lidar_timestamps={ref_key: ts}, camera_timestamps={}) for ts in sorted(frame_confidences)]
    ego_pose_results = [
        find_ego_pose_at(ts, all_ego_ts, all_ego_data, config.ego_pose_tolerance_ms)
        for ts in sorted(frame_confidences)
    ]
    write_csv(groups, os.path.join(stage1_out, "timestamps.csv"))
    write_ego_pose_csv(
        groups, ego_pose_results, ref_key, os.path.join(stage1_out, "ego_poses.csv"),
    )
    write_detection_confidence_csv(frame_confidences, os.path.join(stage1_out, "detection_confidences.csv"))
    logger.info("Detection extraction complete: %d PCD files written, %d unmatched messages", written, unmatched)
    return stage1_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _selected_detections(detection_msg, target_classes):
    if not 0 <= detection_msg.count <= len(detection_msg.meas_array):
        raise ValueError("Detection count exceeds measurement array length")
    detections = []
    detection_ids = set()
    for detection in detection_msg.meas_array[:detection_msg.count]:
        if target_classes is not None and detection.meas_class not in target_classes:
            continue
        detection_id = int(detection.box_id)
        if detection_id in detection_ids:
            raise ValueError(f"Duplicate detection ID {detection_id} in detection message")
        detection_ids.add(detection_id)
        detections.append(detection)
    return detections


def _frame_detections(matches, cloud_timestamp: int, target_classes):
    """Keep the nearest observation of each ID when several messages match one cloud."""
    selected = {}
    for detection_timestamp, detection_msg in matches:
        for detection in _selected_detections(detection_msg, target_classes):
            try:
                bbox = detection_to_bbox(detection)
            except ValueError as exc:
                logger.warning("Skipping detection %d at ts=%d: %s", detection.box_id, detection_timestamp, exc)
                continue
            rank = (abs(detection_timestamp - cloud_timestamp), detection_timestamp)
            detection_id = int(detection.box_id)
            if detection_id not in selected or rank < selected[detection_id][0]:
                selected[detection_id] = (rank, detection, bbox)
    return [(selected[detection_id][1], selected[detection_id][2]) for detection_id in sorted(selected)]


def _write_lidar_sample(cloud_msg, timestamp_ns: int, output_dir: str, on_existing: str, lidar_key: str = "lidar") -> bool:
    out_path = build_lidar_output_path(output_dir, "", lidar_key, timestamp_ns)
    if os.path.exists(out_path):
        if on_existing == "skip":
            return False
        elif on_existing == "abort":
            raise FileExistsError(f"Output file already exists: {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_pcd(cloud_msg, out_path)
    return True


def _iter_serialized_messages(parser, topics: list[str]):
    """Stream selected topics together without loading a bag into memory."""
    if hasattr(parser, "reader"):
        for schema, channel, record in parser.reader.iter_messages(topics=topics):
            yield channel.topic, record.log_time, schema.name, record.data
    else:
        placeholders = ",".join("?" for _ in topics)
        rows = parser.conn.execute(
            "SELECT topics.name,messages.timestamp,topics.type,messages.data "
            "FROM messages JOIN topics ON messages.topic_id=topics.id "
            f"WHERE topics.name IN ({placeholders}) ORDER BY messages.timestamp,messages.rowid",
            tuple(topics),
        )
        yield from rows


def _decode_message(parser, topic: str, type_name: str, payload):
    if hasattr(parser, "reader"):
        message = parser._decode(topic, type_name, payload)
        if message is None:
            raise ValueError(f"Cannot decode message for topic {topic}")
        return message
    return deserialize_message(payload, parser.topic_msg_message[topic])
