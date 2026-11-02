"""
Extract points inside Detector3DMeasList boxes in the PointCloud2 frame.
Output path: <output_base_dir>/<YYYYMMDD_HHMMSS_ns>_<detection_id>.pcd

Matching and filenames use message header.stamp timestamps. Bag recording
timestamps are retained only to fetch the corresponding messages from storage.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from bag_reader import make_parser
from extraction.ego_pose_extractor import load_ego_poses_from_parser, find_ego_pose_at
from extraction.lidar_extractor import _write_pcd, read_cloud_points
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
    if not 0 <= detection_msg.count <= len(detection_msg.meas_array):
        raise ValueError("Detection count exceeds measurement array length")

    detections = [
        detection for detection in detection_msg.meas_array[:detection_msg.count]
        if target_classes is None or detection.meas_class in target_classes
    ]
    if not detections:
        return 0

    points = read_cloud_points(cloud_msg)
    xyz = np.column_stack([points[name] for name in ("x", "y", "z")]).astype(np.float64)
    written = 0
    detection_ids = set()
    for detection in detections:
        detection_id = int(detection.box_id)
        if detection_id in detection_ids:
            raise ValueError(f"Duplicate detection ID {detection_id} at ts={timestamp_ns}")
        detection_ids.add(detection_id)
        out_path = os.path.join(output_dir, f"{ns_to_filename(timestamp_ns)}_{detection_id}.pcd")
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
        os.makedirs(output_dir, exist_ok=True)
        _write_pcd(cloud_msg, out_path, point_indices=mask)
        written += 1
    return written


def run_cluster_extraction(config, files: list[str]) -> str:
    """Match each detection message to the nearest cloud across all bag splits."""
    logger.info(">>> CLUSTER EXTRACTION")
    logger.info("BAG input   : %s (%d files)", config.bag_dir, len(files))
    logger.info("Output dir  : %s", config.output_base_dir)

    lidar_index = []
    detection_index = []
    ego_ts_chunks = []
    ego_pose_chunks = []
    for fpath in files:
        parser = make_parser(fpath)
        logger.info("Reading header timestamps and ego poses from %s ...", os.path.basename(fpath))
        lidar_index.extend((ts, record_ts, fpath) for ts, record_ts in _header_timestamps(parser, config.lidar_topic))
        detection_index.extend((ts, record_ts, fpath) for ts, record_ts in _header_timestamps(parser, config.detection_topic))
        file_ego_ts, file_ego_poses = load_ego_poses_from_parser(
            parser, config.ego_pose.topic, config.ego_pose.parent_frame, config.ego_pose.child_frame,
        )
        if len(file_ego_ts) > 0:
            ego_ts_chunks.append(file_ego_ts)
            ego_pose_chunks.append(file_ego_poses)
        del parser

    if not lidar_index or not detection_index:
        raise ValueError("The bag must contain messages on both the LiDAR and detection topics")
    lidar_index.sort()
    detection_index.sort()
    lidar_ts = np.array([ts for ts, _, _ in lidar_index], dtype=np.int64)

    if ego_ts_chunks:
        all_ego_ts = np.concatenate(ego_ts_chunks)
        all_ego_data = np.vstack(ego_pose_chunks)
        order = np.argsort(all_ego_ts)
        all_ego_ts = all_ego_ts[order]
        all_ego_data = all_ego_data[order]
    else:
        all_ego_ts = np.array([], dtype=np.int64)
        all_ego_data = np.zeros((0, 7), dtype=np.float64)
        logger.warning("No ego poses found; ego_poses.csv will have empty numeric cells.")

    # Keep only the active detection/cloud split open, including at split boundaries.
    detection_parser = cloud_parser = None
    detection_file = cloud_file = None
    cached_cloud_key = None
    cached_cloud = None
    written = 0
    unmatched = 0
    groups = []
    ego_pose_results = []
    ref_key = "detections"
    for timestamp_ns, recording_timestamp, fpath in detection_index:
        idx = _find_closest_within(timestamp_ns, lidar_ts, config.tolerance_ms)
        if idx is None:
            unmatched += 1
            logger.warning("No matching point cloud for detections at ts=%d", timestamp_ns)
            continue
        if detection_file != fpath:
            detection_parser = make_parser(fpath)
            detection_file = fpath
        detection_msg = _read_message(detection_parser, config.detection_topic, recording_timestamp)
        _, cloud_recording_timestamp, cloud_path = lidar_index[idx]
        cloud_key = (cloud_path, cloud_recording_timestamp)
        if cloud_key != cached_cloud_key:
            if cloud_file != cloud_path:
                cloud_parser = make_parser(cloud_path)
                cloud_file = cloud_path
            cached_cloud = _read_message(cloud_parser, config.lidar_topic, cloud_recording_timestamp)
            cached_cloud_key = cloud_key
        written += extract_detection_clusters(
            cached_cloud, detection_msg, timestamp_ns, config.output_base_dir, config.on_existing,
            target_classes=config.target_classes,
        )
        groups.append(SampleGroup(lidar_timestamps={ref_key: timestamp_ns}, camera_timestamps={}))
        ego_pose_results.append(find_ego_pose_at(
            ref_ts=timestamp_ns, ts_arr=all_ego_ts, pose_arr=all_ego_data,
            tolerance_ms=config.ego_pose_tolerance_ms,
        ))

    write_ego_pose_csv(
        groups, ego_pose_results, ref_key, os.path.join(config.output_base_dir, "ego_poses.csv"),
    )
    logger.info("Cluster extraction complete: %d PCD files written, %d unmatched messages", written, unmatched)
    return config.output_base_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _header_timestamps(parser, topic: str) -> list[tuple[int, int]]:
    """Index (header timestamp, recording timestamp) without retaining clouds."""
    if hasattr(parser, "reader"):
        # Stream MCAP once with either backend. Query windows use log_time,
        # which can differ from both header.stamp and publish_time.
        messages = (
            (record.log_time, parser._decode(channel.topic, getattr(schema, "name", None), record.data))
            for schema, channel, record in parser.reader.iter_messages(topics=[topic])
        )
    else:
        if topic not in parser.topic_id:
            return []
        messages = (
            (ts, _read_message(parser, topic, ts))
            for ts in parser.get_message_timestamps(topic)
        )

    timestamps = []
    for recording_timestamp, message in messages:
        if message is None:
            raise ValueError(f"Cannot decode message for topic {topic} at ts={recording_timestamp}")
        stamp = message.header.stamp
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        timestamps.append((timestamp_ns, recording_timestamp))
    return timestamps


def _read_message(parser, topic: str, timestamp_ns: int):
    # MCAP end_time is exclusive; SQLite's end bound is inclusive.
    end_time = timestamp_ns + 1 if hasattr(parser, "reader") else timestamp_ns
    messages = parser.get_messages(topics=topic, start_time=timestamp_ns, end_time=end_time)
    if len(messages) != 1:
        raise ValueError(f"Expected one message for topic {topic} at ts={timestamp_ns}, got {len(messages)}")
    return messages[0][1]
