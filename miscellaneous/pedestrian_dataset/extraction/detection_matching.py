"""Index bag messages and match sensor frames to detections by header timestamp."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from bag_reader import decode_message, iter_serialized_messages, make_parser
from extraction.ego_pose_extractor import iter_ego_poses
from timestamp_matching.matcher import SampleGroup, _find_closest_within
from utils.camera_info_parser import CameraInfoParser
from utils.config_loader import CameraConfig, ClusterConfig
from utils.tf_static_parser import TFStaticParser
from utils.timestamp_utils import message_timestamp_ns

logger = logging.getLogger(__name__)


class SensorMessage(NamedTuple):
    timestamp_ns: int
    recording_timestamp: int
    bag_path: str
    message_number: int


class DetectionMessage(NamedTuple):
    timestamp_ns: int
    recording_timestamp: int
    bag_path: str
    message: Any


DetectionMatches = list[tuple[int, Any]]
CloudMatches = dict[str, dict[int, DetectionMatches]]
CameraMatchesByFile = dict[str, dict[tuple[str, int], dict[str, tuple[CameraConfig, int]]]]


@dataclass
class BagIndex:
    clouds: list[SensorMessage]
    detections: list[DetectionMessage]
    images_by_topic: dict[str, list[SensorMessage]]
    ego_timestamps: np.ndarray
    ego_poses: np.ndarray
    static_transforms: TFStaticParser | None
    camera_info: CameraInfoParser | None


@dataclass
class CameraMatches:
    by_file: CameraMatchesByFile
    timestamps_by_cloud: dict[int, dict[str, int | None]]
    frame_ids: list[str]


def index_bag_messages(config: ClusterConfig, files: list[str]) -> BagIndex:
    clouds = []
    detections = []
    images_by_topic = {camera.topic: [] for camera in config.cameras}
    ego_timestamps = []
    ego_poses = []

    topics = {config.lidar_topic, config.detection_topic, config.ego_pose.topic}
    topics.update(images_by_topic)
    static_transforms = None
    if config.urdf_path == "from_tf_static":
        static_transforms = TFStaticParser()
        topics.add("/tf_static")
    camera_info = None
    if config.calibration_yaml == "from_camera_info":
        camera_info = CameraInfoParser(config.cameras)
        topics.update(camera_info.topics)

    # Keep the locations of large sensor messages and retain small messages for matching.
    indexed_topics = sorted(topics)
    for bag_path in files:
        parser = make_parser(bag_path)
        logger.info(
            "Pass 1: indexing headers, detections and ego poses from %s ...",
            os.path.basename(bag_path),
        )
        cloud_number = 0
        camera_numbers = dict.fromkeys(images_by_topic, 0)
        try:
            for topic, recording_timestamp, type_name, payload in iter_serialized_messages(parser, indexed_topics):
                message = decode_message(parser, topic, type_name, payload)
                if topic == "/tf_static" and static_transforms is not None:
                    static_transforms.add_message(message)

                if camera_info is not None and topic in camera_info.topics:
                    camera_info.add_message(topic, message)
                elif topic == config.ego_pose.topic:
                    for timestamp_ns, pose in iter_ego_poses(
                        message, config.ego_pose.parent_frame, config.ego_pose.child_frame,
                    ):
                        ego_timestamps.append(timestamp_ns)
                        ego_poses.append(pose)
                elif topic != "/tf_static":
                    timestamp_ns = message_timestamp_ns(message)
                    if topic == config.lidar_topic:
                        clouds.append(SensorMessage(timestamp_ns, recording_timestamp, bag_path, cloud_number))
                        cloud_number += 1
                    elif topic in images_by_topic:
                        image_number = camera_numbers[topic]
                        images_by_topic[topic].append(
                            SensorMessage(timestamp_ns, recording_timestamp, bag_path, image_number),
                        )
                        camera_numbers[topic] += 1
                    else:
                        detections.append(DetectionMessage(timestamp_ns, recording_timestamp, bag_path, message))
                del message
        finally:
            del parser

    # Sort across all splits so matching can cross bag boundaries.
    clouds.sort()
    detections.sort(key=_detection_sort_key)
    ego_times = np.array(ego_timestamps, dtype=np.int64)
    pose_data = np.array(ego_poses, dtype=np.float64).reshape(-1, 7)
    order = np.argsort(ego_times)
    return BagIndex(
        clouds=clouds,
        detections=detections,
        images_by_topic=images_by_topic,
        ego_timestamps=ego_times[order],
        ego_poses=pose_data[order],
        static_transforms=static_transforms,
        camera_info=camera_info,
    )


def _detection_sort_key(entry: DetectionMessage) -> tuple[int, int, str]:
    # The decoded messages have no meaningful ordering.
    return entry.timestamp_ns, entry.recording_timestamp, entry.bag_path


def match_sampled_clouds(index: BagIndex, config: ClusterConfig) -> tuple[CloudMatches, int]:
    cloud_timestamps = np.array([cloud.timestamp_ns for cloud in index.clouds], dtype=np.int64)
    matches: CloudMatches = {}
    unmatched = 0

    # Several detection messages may match the same cloud.
    for detection in index.detections:
        cloud_index = _find_closest_within(
            detection.timestamp_ns, cloud_timestamps, config.matching.tolerance_ms,
        )
        if cloud_index is None:
            unmatched += 1
            logger.warning("No matching point cloud for detections at ts=%d", detection.timestamp_ns)
            continue
        cloud = index.clouds[cloud_index]
        file_matches = matches.setdefault(cloud.bag_path, {})
        file_matches.setdefault(cloud.message_number, []).append((detection.timestamp_ns, detection.message))

    # Apply the sampling interval globally in LiDAR header order.
    sampling_interval_ns = int(config.matching.sampling_interval_ms * 1_000_000)
    next_timestamp = None
    for cloud in index.clouds:
        file_matches = matches.get(cloud.bag_path, {})
        if cloud.message_number not in file_matches:
            continue
        if next_timestamp is not None and cloud.timestamp_ns < next_timestamp:
            del file_matches[cloud.message_number]
            continue
        next_timestamp = cloud.timestamp_ns + sampling_interval_ns

    return matches, unmatched


def match_requested_clouds(
    index: BagIndex,
    requested_groups: dict[int, SampleGroup],
    tolerance_ms: float,
) -> tuple[CloudMatches, int]:
    cloud_timestamps = np.array([cloud.timestamp_ns for cloud in index.clouds], dtype=np.int64)
    detection_timestamps = np.array([entry.timestamp_ns for entry in index.detections], dtype=np.int64)
    matches: CloudMatches = {}
    unmatched = 0

    # CSV rows require exact sensor timestamps. Detections still use the tolerance.
    for timestamp_ns in requested_groups:
        cloud_index = _find_exact_timestamp(timestamp_ns, cloud_timestamps)
        if cloud_index is None:
            logger.warning("No exact LiDAR frame at CSV ts=%d", timestamp_ns)
            continue
        detection_slice = find_detection_slice(timestamp_ns, detection_timestamps, tolerance_ms)
        if detection_slice is None:
            unmatched += 1
            logger.warning("No matching detections for CSV ts=%d", timestamp_ns)
            continue

        cloud = index.clouds[cloud_index]
        file_matches = matches.setdefault(cloud.bag_path, {})
        file_matches[cloud.message_number] = [
            (entry.timestamp_ns, entry.message) for entry in index.detections[detection_slice]
        ]

    return matches, unmatched


def match_cameras(
    index: BagIndex,
    config: ClusterConfig,
    cloud_matches: CloudMatches,
    requested_groups: dict[int, SampleGroup] | None,
) -> CameraMatches:
    cloud_timestamps = sorted({
        cloud.timestamp_ns
        for cloud in index.clouds
        if cloud.message_number in cloud_matches.get(cloud.bag_path, {})
    })
    timestamps_by_cloud = {timestamp_ns: {} for timestamp_ns in cloud_timestamps}
    by_file: CameraMatchesByFile = {}
    frame_ids = []

    for camera in config.cameras:
        images = sorted(index.images_by_topic[camera.topic])
        image_timestamps = np.array([image.timestamp_ns for image in images], dtype=np.int64)
        has_match = False
        for timestamp_ns in cloud_timestamps:
            camera_timestamps = timestamps_by_cloud[timestamp_ns]
            camera_timestamps[camera.key] = None
            if requested_groups is None:
                image_index = _find_closest_within(timestamp_ns, image_timestamps, config.matching.tolerance_ms)
                if image_index is None:
                    logger.warning("No matching camera %s for cloud at ts=%d", camera.key, timestamp_ns)
                    continue
            else:
                requested_timestamp = requested_groups[timestamp_ns].camera_timestamps.get(camera.key)
                if requested_timestamp is None:
                    continue
                image_index = _find_exact_timestamp(requested_timestamp, image_timestamps)
                if image_index is None:
                    logger.warning("No exact camera %s frame at CSV ts=%d", camera.key, requested_timestamp)
                    continue

            image = images[image_index]
            camera_timestamps[camera.key] = image.timestamp_ns
            # A single bag message can serve several camera keys on the same topic.
            file_matches = by_file.setdefault(image.bag_path, {})
            image_matches = file_matches.setdefault((camera.topic, image.message_number), {})
            image_matches[camera.key] = (camera, image.timestamp_ns)
            has_match = True

        if has_match:
            frame_ids.append(camera.frame_id)

    return CameraMatches(by_file, timestamps_by_cloud, frame_ids)


def find_detection_slice(timestamp_ns: int, sorted_timestamps: np.ndarray, tolerance_ms: float) -> slice | None:
    match_index = _find_closest_within(timestamp_ns, sorted_timestamps, tolerance_ms)
    if match_index is None:
        return None

    # Include every detection message with the winning timestamp.
    matched_timestamp = sorted_timestamps[match_index]
    start = int(np.searchsorted(sorted_timestamps, matched_timestamp, side="left"))
    end = int(np.searchsorted(sorted_timestamps, matched_timestamp, side="right"))
    return slice(start, end)


def _find_exact_timestamp(timestamp_ns: int, sorted_timestamps: np.ndarray) -> int | None:
    index = int(np.searchsorted(sorted_timestamps, timestamp_ns))
    if index == len(sorted_timestamps) or sorted_timestamps[index] != timestamp_ns:
        return None
    return index
