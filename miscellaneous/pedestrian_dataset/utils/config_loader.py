"""
Loads and validates the pipeline YAML configuration.

Two files are involved:
  1. config.yaml; contains all parameters that are expected to change between runs.
  2. settings_<vehicle>.yaml; contains only LiDAR and camera topic/frame_id definitions,
     plus the reference_frame used for all extrinsic transforms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Typed dataclasses (one per config section)
# ---------------------------------------------------------------------------

@dataclass
class LidarConfig:
    key: str            # logical name from settings, used in CSV headers and samples/
    topic: str
    frame_id: str


@dataclass
class CameraConfig:
    key: str            # logical name used in CSV headers: camera_0 ... camera_6
    topic: str
    frame_id: str
    name: str           # human-readable sensor name (e.g. "camera_cl")
    compressed: bool    # True -> sensor_msgs/CompressedImage, False -> sensor_msgs/Image
    undistorted: bool   # True -> image already undistorted, False -> distorted
    is_optical: bool    # True -> (x right, y down, z forward), False -> (x forward, y left, z up)
    camera_info_topic: Optional[str] = None  # inferred from the image namespace when omitted


@dataclass
class MatchingConfig:
    tolerance_ms: float           # (+-)N ms tolerance window for sensor matching
    sampling_interval_ms: float   # milliseconds between consecutive sample groups
    ego_pose_tolerance_ms: float  # (+-)N ms tolerance window for ego pose lookup


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


@dataclass
class EgoPoseConfig:
    topic: str          # ROS topic carrying tf2_msgs/msg/TFMessage (/tf)
    parent_frame: str   # frame_id of the world/map frame (map)
    child_frame: str    # child_frame_id of the vehicle frame (center_of_gravity)

@dataclass
class MergeDatasetEntry:
    """
    One additional, already-generated Stage-1 directory to merge into the
    nuScenes output as an extra `log`. Stage 1 is NOT re-run for these.
    """
    stage1_dir: str
    timestamps_file: str = "timestamps.csv"   # timestamps.csv | timestamps_curated.csv | timestamps_balanced.csv | ...
    location: Optional[str] = None            # None -> falls back to top-level scene.location
    description: Optional[str] = None         # None -> falls back to top-level scene.description


@dataclass
class NuscenesConfig:
    scene_duration: float   # [s] scene duration in nuScenes format (default 20.0 s)
    n_opponents: int        # [-] number of opponents (instances) in the dataset
    relative: bool          # [T/F] T set ego_pose to zero: labels are relative

    # Which Stage-1 timestamps file to use for the PRIMARY bag (input.bag_dir).
    timestamps_file: str = "timestamps.csv"

    # Additional pre-existing Stage-1 directories to merge in as extra logs.
    merge_datasets: List["MergeDatasetEntry"] = field(default_factory=list)

@dataclass
class SceneConfig:
    location: str               # geographic location of the recording
    description: str            # optional free-text description of the scene (may be empty)
    map_png_path: Optional[str] # absolute path to the map.png to copy into maps/ (None = no map)


@dataclass
class PipelineConfig:
    # I/O
    bag_dir: str
    calibration_yaml: Optional[str]
    urdf_path: str
    output_base_dir: str
    on_existing: str            # "skip" | "overwrite" | "abort"

    # Vehicle
    vehicle_name: str
    num_cameras: int

    # Calibration
    reference_frame: str        # URDF/TF frame used as the common extrinsics reference frame

    # Sensors (ordered lists) (index 0 is always the reference)
    lidars: List[LidarConfig]
    cameras: List[CameraConfig]

    # Matching
    matching: MatchingConfig

    # Ego pose
    ego_pose: EgoPoseConfig

    # Scene / log metadata (consumed by Stage-2 converters)
    scene: SceneConfig

    # Stage 2
    output_formats: List[str]

    # Nuscenes-specific
    nuscenes: NuscenesConfig

    # KITTI camera selection: list of CameraConfig.key entries
    kitti_camera_selection: Optional[List[str]] = None

    # Runtime-resolved ordered list of CameraConfig objects to emit for KITTI.
    kitti_selected_cameras: Optional[List[CameraConfig]] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cluster_config(yaml_path: str, include_cameras: bool = False) -> ClusterConfig:
    """Load the detection-driven Stage-1 config used by main.py --from_det."""
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)

    inp = _require(raw, "input", "root")
    out = _require(raw, "output", "root")
    
    # timestamp matching
    match_raw = _require(raw, "matching", "root")
    match_raw.update(raw.get("matching", {}))
    tolerance_ms = match_raw.get("tolerance_ms", "-1.0")
    if not 0 < tolerance_ms < float('inf'):
        raise ValueError("Config: matching.tolerance_ms must be in [0, inf]")
    sampling_ms = match_raw.get("sampling_ms", "-1.0")
    if not 0 < sampling_ms < float('inf'):
        raise ValueError("Config: matching.sampling_ms must be in [0, inf]")
    ego_pose_tolerance_ms = match_raw.get("ego_pose_tolerance_ms", "-1.0")
    if not 0 < ego_pose_tolerance_ms < float('inf'):
        raise ValueError("Config: matching.ego_pose_tolerance_ms must be in [0, inf]")
    sampling_ms = match_raw.get(tolerance_ms, "-1.0")

    # Target classes
    target_classes = out.get("target_classes")
    if target_classes is not None:
        if not isinstance(target_classes, list):
            raise ValueError("Config: out.target_classes must be a list")
        elif len(target_classes)>1 and any(
            type(class_id) is not int or not 0 <= class_id <= 255
            for class_id in target_classes
        ):
            raise ValueError("Config: out.target_classes must be a list of integer class IDs (0-255) or [-1]")
    settings = {}
    if "vehicle" in raw:
        settings, _ = _load_vehicle_settings(raw, yaml_path)

    lidar_topic = inp.get("lidar_topic")
    lidar_key = inp.get("lidar_key", "lidar")
    lidar_frame_id = None
    if settings:
        lidars = _require(settings, "lidars", "settings")
        if not lidars:
            raise ValueError("Settings: at least one lidar must be defined")
        lidar_key = inp.get("lidar_key", next(iter(lidars)))
        if lidar_key not in lidars:
            raise ValueError(f"Config: unknown input.lidar_key '{lidar_key}'. Available: {sorted(lidars)}")
        if lidar_topic is None:
            lidar_topic = _require(lidars[lidar_key], "topic", f"lidars.{lidar_key}")
        lidar_frame_id = lidars[lidar_key].get("frame_id")
    if lidar_topic is None:
        lidar_topic = _require(inp, "lidar_topic", "input")

    # Ego pose 
    ego_pose_raw = dict(settings.get("ego_pose", {}))
    ego_pose_raw.update(raw.get("ego_pose", {}))

    policy = out.get("on_existing", "skip")
    if policy not in ("skip", "overwrite", "abort"):
        raise ValueError(f"Config: output.on_existing must be skip|overwrite|abort, got '{policy}'")

    cameras = _parse_cameras(settings.get("cameras", {})) if include_cameras else []
    if include_cameras and not cameras:
        raise ValueError("Config: --cameras requires cameras defined in vehicle.settings")

    return ClusterConfig(
        bag_dir=os.path.expanduser(_require(inp, "bag_dir", "input")),
        lidar_topic=lidar_topic,
        detection_topic=_require(inp, "detection_topic", "input"),
        output_base_dir=os.path.expanduser(_require(out, "base_dir", "output")),
        on_existing=policy,
        ego_pose=EgoPoseConfig(
            topic=_require(ego_pose_raw, "topic", "ego_pose"),
            parent_frame=_require(ego_pose_raw, "parent_frame", "ego_pose"),
            child_frame=_require(ego_pose_raw, "child_frame", "ego_pose"),
        ),
        matching=MatchingConfig(
            tolerance_ms=tolerance_ms,
            sampling_interval_ms=sampling_ms,
            ego_pose_tolerance_ms=ego_pose_tolerance_ms
        ),
        target_classes=target_classes,
        lidar_key=lidar_key,
        urdf_path=_resolve_calibration_path(inp.get("urdf_path") or "from_tf_static", yaml_path),
        reference_frame=settings.get("reference_frame", ego_pose_raw.get("child_frame")),
        lidar_frame_id=lidar_frame_id,
        cameras=cameras,
        calibration_yaml=_resolve_camera_calibration(inp, yaml_path) if include_cameras else None,
    )


def load_config(yaml_path: str) -> PipelineConfig:
    """
    Parse the main config YAML at `yaml_path`, resolve the vehicle settings file
    referenced within it, and return a validated PipelineConfig.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)

    settings, settings_path = _load_vehicle_settings(raw, yaml_path)
    return _parse(raw, settings, yaml_path, settings_path)


def resolve_kitti_cameras(
    cameras: List[CameraConfig],
    selection: Optional[List[str]],
) -> List[CameraConfig]:
    """
    Resolve the ordered list of cameras to emit in the KITTI output.
    """
    if not selection:
        return sorted(cameras, key=lambda c: c.key)

    by_key = {c.key: c for c in cameras}
    resolved: List[CameraConfig] = []
    for key in selection:
        cam = by_key.get(key)
        if cam is None:
            raise ValueError(
                f"Unknown camera key '{key}'. Available: {sorted(by_key)}."
            )
        resolved.append(cam)
    return resolved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require(d: dict, key: str, section: str):
    if key not in d:
        raise ValueError(f"Config: missing required key '{key}' in section '{section}'")
    return d[key]


def _resolve_calibration_path(path: str, yaml_path: str) -> str:
    if path in ("from_tf_static", "from_camera_info"):
        return path
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(yaml_path)), path)


def _resolve_camera_calibration(inp: dict, yaml_path: str) -> Optional[str]:
    source = inp.get("calibration_yaml")
    return _resolve_calibration_path(source, yaml_path) if source else None


def _parse_cameras(cameras_raw: dict) -> List[CameraConfig]:
    cameras: List[CameraConfig] = []
    for key in sorted(cameras_raw.keys()):
        entry = cameras_raw[key]
        cameras.append(CameraConfig(
            key=key,
            topic=_require(entry, "topic", f"cameras.{key}"),
            frame_id=_require(entry, "frame_id", f"cameras.{key}"),
            name=_require(entry, "name", f"cameras.{key}"),
            compressed=entry.get("compressed", True),
            undistorted=entry.get("undistorted", False),
            is_optical=entry.get("is_optical", False),
            camera_info_topic=entry.get("camera_info_topic"),
        ))
    return cameras


def _load_vehicle_settings(raw: dict, yaml_path: str) -> tuple[dict, str]:
    """Resolve vehicle.settings relative to the main config and read it."""
    config_dir = os.path.dirname(os.path.abspath(yaml_path))
    veh_section = _require(raw, "vehicle", "root")
    settings_rel = os.path.expanduser(_require(veh_section, "settings", "vehicle"))
    settings_path = (
        settings_rel if os.path.isabs(settings_rel)
        else os.path.join(config_dir, settings_rel)
    )
    if not os.path.isfile(settings_path):
        raise FileNotFoundError(
            f"Vehicle settings file not found: {settings_path}\n"
            f"  (referenced by vehicle.settings in {yaml_path})"
        )
    with open(settings_path, "r") as fh:
        return yaml.safe_load(fh), settings_path


def _parse(raw: dict, settings: dict, yaml_path: str, settings_path: str) -> PipelineConfig:
    inp   = _require(raw, "input", "root")
    out   = _require(raw, "output", "root")
    veh   = _require(raw, "vehicle", "root")
    match = _require(raw, "matching", "root")
    fmts  = raw.get("output_formats", [])

    ## reference_frame (required in the vehicle settings file)
    reference_frame = _require(
        settings, "reference_frame",
        f"settings ({settings_path})"
    )

    ## LiDARs
    lidars_raw = _require(settings, "lidars", f"settings ({settings_path})")
    lidars: List[LidarConfig] = []
    for key in lidars_raw:
        entry = lidars_raw[key]
        lidars.append(LidarConfig(
            key=key,
            topic=_require(entry, "topic", f"lidars.{key}"),
            frame_id=_require(entry, "frame_id", f"lidars.{key}"),
        ))

    if not lidars:
        raise ValueError(f"Settings file '{settings_path}': at least one lidar must be defined.")

    ## Cameras
    cameras_raw = _require(settings, "cameras", f"settings ({settings_path})")
    cameras = _parse_cameras(cameras_raw)

    ## Ego pose
    ego_pose_raw = _require(settings, "ego_pose", f"settings ({settings_path})")
    ego_pose_cfg = EgoPoseConfig(
        topic = _require(ego_pose_raw, "topic", "ego_pose"),
        parent_frame = _require(ego_pose_raw, "parent_frame", "ego_pose"),
        child_frame = _require(ego_pose_raw, "child_frame", "ego_pose"),
    )

    ## Scene / log metadata (optional section)
    scene_raw = raw.get("scene", {})
    scene_cfg = SceneConfig(
        location     = scene_raw.get("location", "unknown"),
        description  = scene_raw.get("description", ""),
        map_png_path = scene_raw.get("map_png_path", None),
    )

    ## on_existing policy
    policy = out.get("on_existing", "skip")
    if policy not in ("skip", "overwrite", "abort"):
        raise ValueError(f"Config: output.on_existing must be skip|overwrite|abort, got '{policy}'")

    ## output_formats validation
    supported = {"kitti", "nuscenes", "waymo"}
    for fmt in fmts:
        if fmt not in supported:
            raise ValueError(f"Config: unsupported output format '{fmt}'. Supported: {supported}")

    ## KITTI camera selection (optional). List of camera keys; None = all cameras.
    kitti_raw = raw.get("kitti", {}) or {}
    kitti_camera_selection = kitti_raw.get("cameras")
    if kitti_camera_selection is not None:
        if not isinstance(kitti_camera_selection, list) or not kitti_camera_selection:
            raise ValueError(
                "Config: kitti.cameras must be a non-empty list of camera keys "
                "(or omitted/null to include all cameras)."
            )
        camera_keys = {cam.key for cam in cameras}
        for key in kitti_camera_selection:
            if key not in camera_keys:
                raise ValueError(
                    f"Config: kitti.cameras entry '{key}' does not match any "
                    f"defined camera key {sorted(camera_keys)}."
                )
            
    ## NUSCENES-specific configuration
    nuscenes_raw = raw.get("nuscenes", {}) or {}

    merge_raw = nuscenes_raw.get("merge_datasets", []) or []
    if not isinstance(merge_raw, list):
        raise ValueError(
            "Config: nuscenes.merge_datasets must be a list of "
            "{stage1_dir, timestamps_file, location, description} entries."
        )
    merge_datasets: List[MergeDatasetEntry] = []
    for idx, entry in enumerate(merge_raw):
        merge_datasets.append(MergeDatasetEntry(
            stage1_dir      = _require(entry, "stage1_dir", f"nuscenes.merge_datasets[{idx}]"),
            timestamps_file = entry.get("timestamps_file", "timestamps.csv"),
            location        = entry.get("location"),
            description     = entry.get("description"),
        ))

    nuscenes_cfg = NuscenesConfig(
        scene_duration  = nuscenes_raw.get("scene_duration", 20.0),
        n_opponents     = nuscenes_raw.get("n_opponents", 1),
        relative        = nuscenes_raw.get("relative", False),
        timestamps_file = nuscenes_raw.get("timestamps_file", "timestamps.csv"),
        merge_datasets  = merge_datasets,
    )

    return PipelineConfig(
        bag_dir              = _require(inp, "bag_dir",          "input"),
        calibration_yaml     = _resolve_camera_calibration(inp, yaml_path),
        urdf_path            = _resolve_calibration_path(inp.get("urdf_path") or "from_tf_static", yaml_path),
        output_base_dir      = _require(out, "base_dir",         "output"),
        on_existing          = policy,
        vehicle_name         = _require(veh, "name",             "vehicle"),
        num_cameras          = _require(veh, "num_cameras",      "vehicle"),
        reference_frame      = reference_frame,
        lidars               = lidars,
        cameras              = cameras,
        matching             = MatchingConfig(
            tolerance_ms          = float(_require(match, "tolerance_ms",         "matching")),
            sampling_interval_ms  = float(_require(match, "sampling_interval_ms", "matching")),
            # Defaults to tolerance_ms when not explicitly set.
            ego_pose_tolerance_ms = float(match.get("ego_pose_tolerance_ms", match["tolerance_ms"])),
        ),
        ego_pose             = ego_pose_cfg,
        scene                = scene_cfg,
        output_formats       = fmts,
        kitti_camera_selection = kitti_camera_selection,
        nuscenes               = nuscenes_cfg
    )
