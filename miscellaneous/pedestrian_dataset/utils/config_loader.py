"""Load detection dataset settings from YAML."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class CameraConfig:
    key: str
    topic: str
    frame_id: str
    name: str
    compressed: bool = True
    undistorted: bool = False
    is_optical: bool = False
    camera_info_topic: Optional[str] = None


@dataclass
class MatchingConfig:
    tolerance_ms: float
    sampling_interval_ms: float
    ego_pose_tolerance_ms: float


@dataclass
class EgoPoseConfig:
    topic: str
    parent_frame: str
    child_frame: str


@dataclass
class ClusterConfig:
    bag_dir: str
    lidar_topic: str
    detection_topic: str
    output_base_dir: str
    on_existing: str
    ego_pose: EgoPoseConfig
    matching: MatchingConfig
    target_classes: Optional[List[int]] = None
    lidar_key: str = "lidar"
    urdf_path: str = "from_tf_static"
    reference_frame: Optional[str] = None
    lidar_frame_id: Optional[str] = None
    cameras: List[CameraConfig] = field(default_factory=list)
    calibration_yaml: Optional[str] = None
    timestamps_csv: Optional[str] = None
    reuse_dataset: Optional[str] = None
    cluster: bool = False
    copy_samples: bool = False
    camera_selection: Optional[List[str]] = None


def _load_yaml(path: str) -> dict:
    with open(path, "r") as stream:
        return yaml.safe_load(stream) or {}


def _path(value: Optional[str], config_dir: str) -> Optional[str]:
    # Calibration keywords are interpreted by the extractor.
    if value is None or value in ("from_tf_static", "from_camera_info"):
        return value

    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = os.path.join(config_dir, expanded)
    return os.path.abspath(expanded)


def _select_cameras(camera_settings: dict, selection) -> tuple[list[CameraConfig], list[str]]:
    available = {
        key: CameraConfig(key=key, **entry)
        for key, entry in sorted(camera_settings.items())
    }

    # Selecting all cameras uses settings-key order. An explicit list keeps its own order.
    if selection is True:
        camera_keys = list(available)
    elif selection is False or selection is None:
        camera_keys = []
    elif isinstance(selection, list):
        camera_keys = selection
    else:
        raise ValueError("output.cameras must be true, false, or a list of camera keys")

    unknown_keys = [key for key in camera_keys if key not in available]
    if unknown_keys:
        raise ValueError(f"Unknown camera keys {unknown_keys}. Available: {sorted(available)}")
    return [available[key] for key in camera_keys], camera_keys


def _load_matching_config(matching: dict) -> MatchingConfig:
    tolerance_ms = matching.get("tolerance_ms", 200.0)
    sampling_ms = matching.get("sampling_ms", 750.0)
    # Default the ego-pose tolerance to the effective detection-matching tolerance.
    ego_tolerance_ms = matching.get("ego_pose_tolerance_ms", tolerance_ms)

    for value in (tolerance_ms, sampling_ms, ego_tolerance_ms):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Matching tolerances and sampling interval must be positive numbers")

    return MatchingConfig(
        tolerance_ms=tolerance_ms,
        sampling_interval_ms=sampling_ms,
        ego_pose_tolerance_ms=ego_tolerance_ms,
    )


def load_cluster_config(yaml_path: str) -> ClusterConfig:
    config_path = os.path.abspath(os.path.expanduser(yaml_path))
    config_dir = os.path.dirname(config_path)
    raw = _load_yaml(config_path)
    input_settings = raw.get("input") or {}
    output_settings = raw.get("output") or {}
    vehicle = raw.get("vehicle") or {}

    settings_path = _path(vehicle.get("settings"), config_dir)
    settings = _load_yaml(settings_path) if settings_path else {}
    lidars = settings.get("lidars") or {}
    default_lidar_key = next(iter(lidars), "lidar")
    lidar_key = input_settings.get("lidar_key", default_lidar_key)
    lidar = lidars.get(lidar_key) or {}

    # The run's ego-pose settings take precedence over the vehicle defaults.
    ego_pose = dict(settings.get("ego_pose") or {})
    ego_pose.update(raw.get("ego_pose") or {})
    cameras, camera_keys = _select_cameras(
        settings.get("cameras") or {},
        output_settings.get("cameras", False),
    )

    target_classes = output_settings.get("target_classes")
    if target_classes is not None:
        if not isinstance(target_classes, list) or any(
            type(class_id) is not int or class_id < 0 for class_id in target_classes
        ):
            raise ValueError("output.target_classes must be a list of non-negative integer class IDs or null")
    matching = _load_matching_config(raw.get("matching") or {})

    reuse_settings = input_settings.get("reuse_dataset")
    if isinstance(reuse_settings, dict):
        timestamps_csv = reuse_settings.get("timestamps_csv")
        reuse_dataset = reuse_settings.get("samples")
    else:
        # Accept the flat form in older configuration files.
        timestamps_csv = input_settings.get("timestamps_csv")
        reuse_dataset = reuse_settings

    # Normalize empty paths before path resolution turns them into the YAML directory.
    timestamps_csv = timestamps_csv or None
    reuse_dataset = reuse_dataset or None
    copy_samples = output_settings.get("copy_samples", False)
    if reuse_dataset and not timestamps_csv:
        raise ValueError("input.reuse_dataset.samples requires input.reuse_dataset.timestamps_csv")

    if not all(ego_pose.get(key) for key in ("topic", "parent_frame", "child_frame")):
        raise ValueError("Configure ego_pose.topic, parent_frame, and child_frame")
    if not input_settings.get("bag_dir"):
        raise ValueError("Configure input.bag_dir")
    if not input_settings.get("detection_topic"):
        raise ValueError("Configure input.detection_topic")
    if not (input_settings.get("lidar_topic") or lidar.get("topic")):
        raise ValueError("Configure input.lidar_topic or define it in vehicle.settings")
    if not output_settings.get("base_dir"):
        raise ValueError("Configure output.base_dir")

    # Resolve YAML paths relative to the config file.
    bag_dir = input_settings["bag_dir"]
    base_dir = output_settings["base_dir"]
    urdf_path = input_settings.get("urdf_path", "from_tf_static")
    calibration_yaml = input_settings.get("calibration_yaml")
    reference_frame = settings.get("reference_frame", ego_pose.get("child_frame"))

    return ClusterConfig(
        bag_dir=_path(bag_dir, config_dir),
        lidar_topic=input_settings.get("lidar_topic", lidar.get("topic")),
        detection_topic=input_settings["detection_topic"],
        output_base_dir=_path(base_dir, config_dir),
        on_existing=output_settings.get("on_existing", "skip"),
        ego_pose=EgoPoseConfig(
            topic=ego_pose["topic"],
            parent_frame=ego_pose["parent_frame"],
            child_frame=ego_pose["child_frame"],
        ),
        matching=matching,
        target_classes=target_classes,
        lidar_key=lidar_key,
        urdf_path=_path(urdf_path, config_dir),
        reference_frame=reference_frame,
        lidar_frame_id=lidar.get("frame_id"),
        cameras=cameras,
        calibration_yaml=_path(calibration_yaml, config_dir),
        timestamps_csv=_path(timestamps_csv, config_dir),
        reuse_dataset=_path(reuse_dataset, config_dir),
        cluster=output_settings.get("cluster", False),
        copy_samples=copy_samples,
        camera_selection=camera_keys,
    )
