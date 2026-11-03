"""Convert recorded CameraInfo messages to the existing camera-intrinsics schema."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _camera_info_topic(camera) -> str:
    if camera.camera_info_topic:
        return camera.camera_info_topic
    image_topic = camera.topic
    if image_topic.rsplit("/", 1)[-1] in ("compressed", "compressedDepth", "theora"):
        image_topic = image_topic.rsplit("/", 1)[0]
    return image_topic.rpartition("/")[0] + "/camera_info"


def _parse_camera_info(message, camera) -> dict:
    k = np.asarray(message.k, dtype=np.float64)
    distortion = np.asarray(message.d, dtype=np.float64)
    if k.shape != (9,) or not np.isfinite(k).all() or k[0] <= 0 or k[4] <= 0:
        raise ValueError(f"Invalid CameraInfo intrinsics for camera '{camera.key}'")
    if message.width <= 0 or message.height <= 0 or not np.isfinite(distortion).all():
        raise ValueError(f"Invalid CameraInfo dimensions or distortion for camera '{camera.key}'")

    entry = {
        "frame_id": camera.frame_id, "topic": camera.topic,
        "model_type": "PINHOLE", "fx": float(k[0]), "fy": float(k[4]),
        "cx": float(k[2]), "cy": float(k[5]),
        "width": int(message.width), "height": int(message.height),
    }
    model = message.distortion_model.lower()
    if model == "equidistant":
        if len(distortion) != 4:
            raise ValueError(f"CameraInfo equidistant model requires four coefficients for '{camera.key}'")
        entry["model_type"] = "FISHEYE"
        entry["fisheye_dist"] = distortion.tolist()
    elif model in ("plumb_bob", ""):
        if len(distortion) not in (0, 4, 5) or (not model and len(distortion)):
            raise ValueError(f"Unsupported CameraInfo distortion coefficients for '{camera.key}'")
        d = np.zeros(5, dtype=np.float64)
        d[:len(distortion)] = distortion
        entry["radial_dist"] = d[[0, 1, 4]].tolist()
        entry["tangential_dist"] = d[[2, 3]].tolist()
    else:
        raise ValueError(f"Unsupported CameraInfo distortion model '{model}' for '{camera.key}'")
    return entry


class CameraInfoParser:
    """Keep the first calibrated CameraInfo per topic for a single output calibration."""

    def __init__(self, cameras):
        self.cameras = cameras
        self.topics = {_camera_info_topic(camera) for camera in cameras}
        self.messages = {}

    def add_message(self, topic: str, message) -> None:
        if topic not in self.topics or topic in self.messages:
            return
        # ROS uses K[0] == 0 to indicate an uncalibrated camera. A later message
        # or bag split may contain the actual calibration.
        if not message.k[0]:
            return
        self.messages[topic] = message
        logger.info("Loaded camera calibration from %s", topic)

    def get_intrinsics(self, camera_frame_ids=None) -> dict:
        intrinsics = {}
        for camera in self.cameras:
            if camera_frame_ids is not None and camera.frame_id not in camera_frame_ids:
                continue
            topic = _camera_info_topic(camera)
            if topic not in self.messages:
                raise ValueError(f"No calibrated CameraInfo found for camera '{camera.key}' on topic '{topic}'")
            intrinsics[camera.name] = _parse_camera_info(self.messages[topic], camera)
        return intrinsics


def load_camera_info_from_parser(parser, camera_info: CameraInfoParser) -> None:
    """Collect camera calibration from one bag split using the existing reader API."""
    for topic in sorted(camera_info.topics):
        if not hasattr(parser, "reader") and topic not in parser.topic_id:
            continue
        messages = parser.get_messages(topics=topic, start_time=0, end_time=int(np.iinfo(np.int64).max))
        for _, message in messages:
            camera_info.add_message(topic, message)
