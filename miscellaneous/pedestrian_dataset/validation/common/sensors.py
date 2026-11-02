"""
SensorModel: the sensor topology the validators need to relate IR <-> KITTI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


@dataclass
class SensorModel:
    vehicle: Optional[str]
    lidar_keys: List[str]                       # ordered logical keys (CSV column order)
    camera_keys: List[str]                      # ordered logical keys (CSV column order)
    _cam_name: Dict[str, str]                   # intrinsics key in calibration.yaml
    _cam_frame: Dict[str, str]                  # extrinsics key in calibration.yaml
    _lidar_frame: Dict[str, str]                # extrinsics key for a LiDAR
    reference_frame: Optional[str]              # frame all extrinsics are expressed in
    ego_child_frame: Optional[str]              # annotation anchor frame (center_of_gravity)
    tolerance_ms: Optional[float]
    sampling_interval_ms: Optional[float]
    ego_pose_tolerance_ms: Optional[float]
    num_cameras: Optional[int]
    kitti_camera_selection: Optional[List[str]] # list[str] | None (kitti.cameras from config)

    # -- accessors -------------------------------------------------------
    def camera_name(self, key: str) -> str:
        return self._cam_name.get(key, key)

    def camera_frame_id(self, key: str) -> Optional[str]:
        return self._cam_frame.get(key)

    def lidar_frame_id(self, key: str) -> Optional[str]:
        return self._lidar_frame.get(key)

    def camera_names(self) -> List[str]:
        return [self.camera_name(k) for k in self.camera_keys]

    # -- KITTI camera selection (mirrors converter precedence) -----------
    def resolve_kitti_cameras(self, cli_cameras: Optional[List[str]] = None) -> List[str]:
        """
        Ordered list of camera keys the KITTI converter would emit:
        CLI override > config kitti.cameras > all cameras (sorted).
        Raises ValueError on an unknown key.
        """
        selection = cli_cameras or self.kitti_camera_selection
        if not selection:
            return sorted(self.camera_keys)
        valid = set(self.camera_keys)
        for k in selection:
            if k not in valid:
                raise ValueError(
                    f"Unknown camera key '{k}'. Available: {sorted(valid)}."
                )
        return list(selection)


def load_sensor_model(config_path: str) -> SensorModel:
    """Build a SensorModel from a pipeline config.yaml path."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    vehicle_blk = cfg.get("vehicle", {}) or {}
    settings_name = vehicle_blk.get("settings")
    if not settings_name:
        raise ValueError(f"config '{config_path}' has no vehicle.settings entry.")

    # Settings file is resolved relative to the config file's directory.
    settings_path = settings_name
    if not os.path.isabs(settings_path):
        settings_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), settings_name)
    if not os.path.isfile(settings_path):
        raise FileNotFoundError(f"vehicle settings file not found: {settings_path}")

    with open(settings_path) as fh:
        st = yaml.safe_load(fh) or {}

    lidars = st.get("lidars", {}) or {}
    cameras = st.get("cameras", {}) or {}
    ego = st.get("ego_pose", {}) or {}
    matching = cfg.get("matching", {}) or {}
    kitti_blk = cfg.get("kitti", {}) or {}

    lidar_keys = list(lidars.keys())
    camera_keys = list(cameras.keys())

    return SensorModel(
        vehicle=vehicle_blk.get("name"),
        lidar_keys=lidar_keys,
        camera_keys=camera_keys,
        _cam_name={k: (v.get("name", k)) for k, v in cameras.items()},
        _cam_frame={k: v.get("frame_id") for k, v in cameras.items()},
        _lidar_frame={k: v.get("frame_id", k) for k, v in lidars.items()},
        reference_frame=st.get("reference_frame"),
        ego_child_frame=ego.get("child_frame"),
        tolerance_ms=matching.get("tolerance_ms"),
        sampling_interval_ms=matching.get("sampling_interval_ms"),
        ego_pose_tolerance_ms=matching.get("ego_pose_tolerance_ms"),
        num_cameras=vehicle_blk.get("num_cameras"),
        kitti_camera_selection=kitti_blk.get("cameras"),
    )