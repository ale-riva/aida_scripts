"""
Stage-2 converter: intermediate format -> nuScenes dataset layout.

Output structure
<output_dir>/
├── v1.0-trainval
|   ├── maps/
|   ├── v1.0-trainval/
|   │   ├── attribute.json           (built once for the whole dataset)
|   │   ├── calibrated_sensor.json   (one entry per active channel, per merged log)
|   │   ├── category.json            (built once for the whole dataset)
|   │   ├── ego_pose.json            (one entry per sample)
|   │   ├── instance.json            (one entry per opponent per scene)
|   │   ├── log.json                 (one entry per merged Stage-1 dataset)
|   │   ├── map.json                 (one entry per merged log - placeholder, no dedup)
|   │   ├── sample.json              (one entry per matched group from Stage 1)
|   │   ├── sample_annotation.json   (empty list unless labels are present)
|   │   ├── sample_data.json         (one entry per sensor per sample)
|   │   ├── scene.json               (one entry per scene, across all merged logs)
|   │   ├── sensor.json              (built once for the whole dataset)
|   │   └── visibility.json          (built once for the whole dataset)
|   ├── samples/
|   │   ├── LIDAR_TOP/               merged point cloud (Nx5 float32: x y z intensity ring_index)
|   │   │   └── <scene>__LIDAR_TOP__<timestamp_us>.pcd.bin
|   │   ├── CAM_FRONT/
|   │   │   └── <scene>__CAM_FRONT__<timestamp_us>.jpg
|   │   └── ...
|   └── sweeps/
        └── <channel>/               (created but empty - every frame is a key frame)

LiDAR
All LiDAR sources listed in the channel map are merged into a single cloud
expressed in the reference frame (ring_center / center_of_gravity).
The merged cloud uses the nuScenes Nx5 float32 layout: x, y, z, intensity,
ring_index (read from the 'ring', 'ring_index', 'channel', or 'beam_id' PCD
field; falls back to 0 only for unstructured clouds that carry no ring field).

Camera
Every camera key present in the channel map is included regardless of lens
model (pinhole or fisheye).  Images are converted from PNG to JPEG.

Timestamp convention
nuScenes uses UNIX microseconds for all timestamps.
Stage-1 filenames encode nanoseconds; conversion: ts_us = ts_ns // 1000.

Timestamps file selection
Each Stage-1 directory can contain several files sharing the same schema:
timestamps.csv (every extracted frame), timestamps_curated.csv (manually
filtered), timestamps_balanced.csv (automatically balanced). Which one is
read for the primary bag is controlled by `nuscenes.timestamps_file` in
config.yaml (default: "timestamps.csv").

Multi-log merge
`nuscenes.merge_datasets` in config.yaml may list additional, already
generated Stage-1 directories (Stage 1 is NOT re-run for these). Each listed
Stage-1 directory - and the primary `input.bag_dir` - becomes one nuScenes
`log`; every log's scenes/samples/sample_data are merged into a single
dataset (shared v1.0-trainval JSON tables). Each merged Stage-1 directory
may specify its own `timestamps_file`. See config.yaml for the schema.

Channel mapping at: config/nuscenes_channel_map_<vehicle_name>.yaml
"""

from __future__ import annotations

import collections
import csv
import datetime
import json
import logging
import math
import os
import shutil
import uuid
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import yaml

from .base_converter import BaseConverter
from utils.pcd_reader import (
    to_nuscenes_bin,
    merge_and_transform as _merge_and_transform
)
from utils.timestamp_utils import filename_to_ns
from utils.image_utils import (
    camera_matrix         as _camera_matrix,
    optimal_undistorted_K as _optimal_undistorted_K,
    undistort_image       as _undistort_image,
    _CAM_BODY_TO_OPT,
)
from utils.shared_loader import _load_calibration, _load_timestamps_csv, _load_ego_poses_csv

from .nuscenes_label_converter import convert_label_to_nus

import cv2
from scipy.spatial.transform import Rotation as _R
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# nuScenes visibility bins
_VIS_BINS = [[0,40],[40, 60], [60,80], [80,100]]

# nuScenes metadata sub-directory name written inside the output folder.
_META_DIR = "v1.0-trainval"

_CONVERTERS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR     = os.path.normpath(os.path.join(_CONVERTERS_DIR, "..", "config"))

# Tables built exactly once for the whole merged dataset.
_GLOBAL_TABLE_NAMES = ("sensor", "category", "attribute", "visibility")

# Tables that are per-recording-session and must be (re)built for merging.
_PER_LOG_TABLE_NAMES = (
    "log", "map", "calibrated_sensor", "scene", "sample", "ego_pose",
    "sample_data", "sample_annotation", "instance",
)

# All tables, in the order convert() writes them out.
_TABLE_NAMES = _GLOBAL_TABLE_NAMES + _PER_LOG_TABLE_NAMES


class _JobSpec(NamedTuple):
    """One Stage-1 dataset to be converted into a single nuScenes `log`."""
    stage1_dir: str
    timestamps_file: str
    location: str
    description: str


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------

def _token() -> str:
    """Generate a 32-character hex token (UUID4 without hyphens)."""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _stem_to_us(stem: str) -> int:
    """
    Convert a Stage-1 filename stem (YYYYMMDD_HHMMSS_<ns>) to UNIX microseconds.
    """
    return filename_to_ns(stem) // 1_000


def _stem_to_date(stem: str) -> str:
    """Return a 'YYYY-MM-DD' string derived from a Stage-1 filename stem."""
    ts_ns = filename_to_ns(stem)
    dt    = datetime.datetime.utcfromtimestamp(ts_ns / 1e9)
    return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotation_from_matrix(R: np.ndarray) -> List[float]:
    """
    Convert a 3x3 rotation matrix to a nuScenes quaternion [w, x, y, z].
    """
    q = _R.from_matrix(R).as_quat()   # scipy returns [x, y, z, w]
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> List[float]:
    """
    Convert intrinsic XYZ Euler angles (radians) to a nuScenes quaternion [w, x, y, z].
    """
    q = _R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_channel_map(vehicle_name: str) -> dict:
    """
    Load the per-vehicle nuScenes channel mapping YAML.
    Location: config/nuscenes_channel_map_<vehicle_name>.yaml
    """
    filename = f"nuscenes_channel_map_{vehicle_name}.yaml"
    path     = os.path.join(_CONFIG_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"nuScenes channel map not found: {path}\n"
            f"Create 'config/{filename}' to define how Stage-1 sensor keys map to nuScenes channel names.\n"
        )
    with open(path) as fh:
        return yaml.safe_load(fh)

# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class NuScenesConverter(BaseConverter):
    """
    Reads one or more Stage-1 intermediate folders and writes a single,
    merged nuScenes dataset (one `log` per Stage-1 folder).
    """

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def convert(self) -> None:
        logger.info("[nuScenes] Primary Stage-1 dir:      %s", self.stage1_dir)
        logger.info("[nuScenes] Output directory:         %s", self.output_dir)
        logger.info("[nuScenes] Global Labels:            %s", "False" if self.config.nuscenes.relative else "True")

        channel_map = _load_channel_map(self.config.vehicle_name)

        ## Parse channel map (invariant across every merged log - same vehicle).
        lidar_cfg = channel_map.get("lidar", {})
        lidar_merge = bool(lidar_cfg.get("merge", True))
        merged_ch = str(lidar_cfg.get("merged_channel", "LIDAR_TOP"))
        lidar_sources = list(lidar_cfg.get("sources", [l.key for l in self.config.lidars]))

        cam_map: Dict[str, str] = channel_map.get("cameras", {})   # camera_N -> CHANNEL_NAME
        cameras_sorted = sorted(self.config.cameras, key=lambda c: c.key)
        active_cameras = [c for c in cameras_sorted if c.key in cam_map]

        # Skip cameras that map to the same nuScenes channel (e.g. one camera_front of EVA).
        _seen_channels: Dict[str, str] = {}   # channel_name -> first camera key that claimed it
        _dedup: List = []
        for _cam in active_cameras:
            _ch = cam_map[_cam.key]
            if _ch in _seen_channels:
                logger.warning(
                    "[nuScenes] Camera '%s' maps to channel '%s', which is already claimed by camera '%s'. Skipping '%s'.",
                    _cam.key, _ch, _seen_channels[_ch], _cam.key,
                )
            else:
                _seen_channels[_ch] = _cam.key
                _dedup.append(_cam)
        active_cameras = _dedup # The cameras that will be processed

        # TODO: support unmerged LiDAR channels
        if not lidar_merge:
            logger.warning(
                "[nuScenes] lidar.merge=false is not fully supported yet; "
                "falling back to merged LIDAR_TOP."
            )
            lidar_merge = True

        if not active_cameras:
            logger.warning("[nuScenes] No cameras found in channel map - no images will be written.")

        ## Create output directory structure (shared by every merged log).
        self._make_dirs(merged_ch, active_cameras, cam_map)

        ## Build the ordered list of Stage-1 datasets to convert.
        jobs = self._build_job_list()
        multi_log = len(jobs) > 1

        self._check_unique_base_names(jobs)

        if multi_log:
            logger.info("[nuScenes] Merging %d Stage-1 datasets into one nuScenes dataset.", len(jobs))

        ## Build every dataset-wide table exactly once.
        (
            sensor_table, sensor_tokens,
            category_table, cat_tok,
            attribute_table, attr_tok,
            visibility_table, default_visibility_token,
        ) = self._build_shared_metadata(merged_ch, active_cameras, cam_map)

        merged_tables: Dict[str, List[dict]] = {
            "sensor":    sensor_table,
            "category":  category_table,
            "attribute": attribute_table,
            "visibility": visibility_table,
        }
        for name in _PER_LOG_TABLE_NAMES:
            merged_tables[name] = []

        converted_logs = 0
        for log_idx, job in enumerate(jobs):
            logger.info(
                "[nuScenes] --- Building log %d/%d from: %s (timestamps: %s) ---",
                log_idx + 1, len(jobs), job.stage1_dir, job.timestamps_file,
            )
            try:
                log_tables = self._build_log(
                    job=job,
                    merged_ch=merged_ch,
                    lidar_sources=lidar_sources,
                    cam_map=cam_map,
                    active_cameras=active_cameras,
                    sensor_tokens=sensor_tokens,
                    cat_tok=cat_tok,
                    attr_tok=attr_tok,
                    default_visibility_token=default_visibility_token,
                )
            except FileNotFoundError as exc:
                if multi_log:
                    logger.warning(
                        "[nuScenes] Skipping log from '%s': %s", job.stage1_dir, exc,
                    )
                    continue
                raise

            if log_tables is None:
                # Empty timestamps file for this dataset.
                if multi_log:
                    logger.warning(
                        "[nuScenes] '%s' produced no samples (empty %s) - skipping this log.",
                        job.stage1_dir, job.timestamps_file,
                    )
                    continue
                logger.warning("[nuScenes] %s is empty - nothing to convert.", job.timestamps_file)
                return

            for name in _PER_LOG_TABLE_NAMES:
                merged_tables[name].extend(log_tables[name])
            converted_logs += 1

        if converted_logs == 0:
            logger.warning("[nuScenes] No logs were converted - no output written.")
            return

        ## Write all JSON tables (merged across every log).
        meta_dir = os.path.join(self.output_dir, _META_DIR, _META_DIR)
        self._write_json(meta_dir, "scene.json",             merged_tables["scene"])
        self._write_json(meta_dir, "log.json",               merged_tables["log"])
        self._write_json(meta_dir, "map.json",               merged_tables["map"])
        self._write_json(meta_dir, "sensor.json",            merged_tables["sensor"])
        self._write_json(meta_dir, "calibrated_sensor.json", merged_tables["calibrated_sensor"])
        self._write_json(meta_dir, "sample.json",            merged_tables["sample"])
        self._write_json(meta_dir, "ego_pose.json",          merged_tables["ego_pose"])
        self._write_json(meta_dir, "sample_data.json",       merged_tables["sample_data"])
        self._write_json(meta_dir, "category.json",          merged_tables["category"])
        self._write_json(meta_dir, "attribute.json",         merged_tables["attribute"])
        self._write_json(meta_dir, "visibility.json",        merged_tables["visibility"])
        self._write_json(meta_dir, "instance.json",          merged_tables["instance"])
        self._write_json(meta_dir, "sample_annotation.json", merged_tables["sample_annotation"])

        logger.info(
            "[nuScenes] Conversion complete: %d log(s), %d scenes, %d samples, %d sample_data entries.",
            converted_logs, len(merged_tables["scene"]), len(merged_tables["sample"]), len(merged_tables["sample_data"]),
        )

    # ------------------------------------------------------------------
    # Multi-log job list
    # ------------------------------------------------------------------

    def _build_job_list(self) -> List[_JobSpec]:
        """
        Build the ordered list of Stage-1 datasets to convert: the primary
        bag (self.stage1_dir) first, followed by every entry configured in
        `nuscenes.merge_datasets` (Stage 1 is NOT re-run for those).
        """
        jobs = [_JobSpec(
            stage1_dir      = self.stage1_dir,
            timestamps_file = self.config.nuscenes.timestamps_file,
            location        = self.config.scene.location,
            description     = self.config.scene.description,
        )]

        for entry in self.config.nuscenes.merge_datasets:
            jobs.append(_JobSpec(
                stage1_dir      = entry.stage1_dir,
                timestamps_file = entry.timestamps_file,
                location        = entry.location if entry.location is not None else self.config.scene.location,
                description     = entry.description if entry.description is not None else self.config.scene.description,
            ))

        return jobs

    @staticmethod
    def _check_unique_base_names(jobs: List[_JobSpec]) -> None:
        """
        Every Stage-1 directory's basename becomes that log's scene-name
        prefix; two logs sharing a basename would collide (identical scene
        names, identical output file paths). Fail fast with a clear error
        instead of silently overwriting files.
        """
        base_names = [os.path.basename(os.path.normpath(j.stage1_dir)) for j in jobs]
        seen: Dict[str, str] = {}
        dupes: List[str] = []
        for name, job in zip(base_names, jobs):
            if name in seen:
                dupes.append(f"'{name}' ({seen[name]} vs {job.stage1_dir})")
            else:
                seen[name] = job.stage1_dir
        if dupes:
            raise ValueError(
                "[nuScenes] Duplicate Stage-1 directory name(s) across merged datasets: "
                + "; ".join(dupes)
                + ". Each Stage-1 directory must have a unique folder name, since it is "
                  "used as the log/scene name prefix and the output file-name prefix."
            )

    # ------------------------------------------------------------------
    # Dataset-wide metadata (built exactly once, shared by every log)
    # ------------------------------------------------------------------

    def _build_shared_metadata(
        self,
        merged_ch: str,
        active_cameras: list,
        cam_map: Dict[str, str],
    ) -> Tuple[List[dict], Dict[str, str], List[dict], str, List[dict], str, List[dict], str]:
        """
        Build every table that is identical across all merged logs:
        sensor.json, category.json, attribute.json, visibility.json.

        These depend only on the vehicle's channel map and fixed constants
        (never on any Stage-1 dataset's own files), so they are built once
        for the whole run rather than once per log.

        Returns
            sensor_table, sensor_tokens (channel -> token),
            category_table, cat_tok,
            attribute_table, attr_tok,
            visibility_table, default_visibility_token (80-100% bin)
        """
        sensor_table: List[dict] = []
        sensor_tokens: Dict[str, str] = {}   # channel -> sensor token

        self._build_lidar_sensor(merged_ch, sensor_table, sensor_tokens)
        self._build_camera_sensors(active_cameras, cam_map, sensor_table, sensor_tokens)

        cat_tok = _token()
        category_table = [{
            "token": cat_tok,
            "name":  "vehicle.car",
            "description":  "Racecar",
            "index": 0,
        }]

        attr_tok = _token()
        attribute_table = [{
            "token": attr_tok,
            "name": "vehicle.moving",
            "description": "vehicle is moving"
        }]

        visibility_table: List[dict] = []
        visibility_tokens: List[str] = []
        for lo, hi in _VIS_BINS:
            vis_tok = _token()
            visibility_tokens.append(vis_tok)
            visibility_table.append({
                "description": f"visibility of whole object is between {lo} and {hi}%",
                "token": vis_tok,
                "level": f"v{lo}-{hi}",
            })
        default_visibility_token = visibility_tokens[-1]   # 80-100% bin

        return (
            sensor_table, sensor_tokens,
            category_table, cat_tok,
            attribute_table, attr_tok,
            visibility_table, default_visibility_token,
        )

    # ------------------------------------------------------------------
    # Per-log conversion
    # ------------------------------------------------------------------

    def _build_log(
        self,
        job: _JobSpec,
        merged_ch: str,
        lidar_sources: List[str],
        cam_map: Dict[str, str],
        active_cameras: list,
        sensor_tokens: Dict[str, str],
        cat_tok: str,
        attr_tok: str,
        default_visibility_token: str,
    ) -> Optional[Dict[str, List[dict]]]:
        """
        Convert a single Stage-1 directory into one nuScenes `log` (its own
        `calibrated_sensor` entries, `map` entry, and one or more `scene`s).
        Returns a dict of table-name -> list-of-rows, or None if this
        dataset has no rows to convert.
        """
        stage1_dir = job.stage1_dir

        ## Load Stage-1 data for this dataset.
        calibration = _load_calibration(stage1_dir)
        ego_poses_map = _load_ego_poses_csv(stage1_dir)
        _, rows = _load_timestamps_csv(stage1_dir, job.timestamps_file)

        if not rows:
            return None

        extrinsics = calibration.get("extrinsics", {})
        intrinsics = calibration.get("intrinsics", {})
        is_label_relative = self.config.nuscenes.relative

        # Derive base scene name from the Stage-1 directory name (= bag folder name).
        base_scene_name = os.path.basename(os.path.normpath(stage1_dir))

        # Reference LiDAR key (lidar_front is always index 0).
        ref_lidar_key = self.config.lidars[0].key

        ## Read scene duration from config (global across every merged log).
        scene_duration = self.config.nuscenes.scene_duration
        location = job.location
        description = job.description
        scene_duration_us = int(scene_duration * 1_000_000) if scene_duration >= 0.0 else None
        if scene_duration_us is None:
            logger.info("[nuScenes] Scene splitting disabled: all samples will be stored in one scene.")
        else:
            logger.info("[nuScenes] Scene length: %.3f s", scene_duration)

        ## Build calibrated_sensor.json for this log only (own calibration.yaml).
        calibrated_sensor_table : List[dict] = []
        cal_sensor_tokens       : Dict[str, str] = {}   # channel -> cal_sensor token
        instance_table          : List[dict] = []

        self._build_lidar_calibrated_sensor(
            merged_ch, sensor_tokens,
            calibrated_sensor_table, cal_sensor_tokens,
        )
        self._build_camera_calibrated_sensors(
            active_cameras, cam_map, extrinsics, intrinsics, sensor_tokens,
            calibrated_sensor_table, cal_sensor_tokens,
        )

        ## Split valid rows into scenes by duration.
        ##
        ## A new scene starts when:
        ##   - the elapsed time from the current scene start is >= scene_length_s,
        ##   - or timestamps go backwards.
        ##   - if scene_duration == 0 (create one scene per each sample)

        current_group: List[Tuple[int, dict, str, int]] = []        # (sample_idx, row, ref_stem, ref_ts_us).
        scene_groups: List[List[Tuple[int, dict, str, int]]] = []   # List of current groups
        current_scene_start_us: Optional[int] = None                # current ref_ts_us
        previous_ts_us: Optional[int] = None                        # previous ref_ts_us (to check potential backawrd jumps)

        for i, row in enumerate(rows):
            ref_stem  = row.get(ref_lidar_key, "").strip()        # lidar key sample name
            ref_ts_us = _stem_to_us(ref_stem) if ref_stem else 0  # lidar key ts (us)

            start_new_scene = False

            if not current_group:
                start_new_scene = True

            elif( scene_duration_us == 0.0):
                # Create one scene PER EACH SAMPLE
                start_new_scene = True

            elif previous_ts_us is not None and ref_ts_us < previous_ts_us:
                logger.warning(
                    "[nuScenes] Timestamp went backwards: previous=%d us, current=%d us. Starting a new scene.",
                    previous_ts_us, ref_ts_us,
                )
                start_new_scene = True

            elif ( scene_duration_us is not None and 
                   current_scene_start_us is not None and 
                   ref_ts_us - current_scene_start_us >= scene_duration_us ):
                    # start new scene if current duration exceeds the fixed duration
                    start_new_scene = True

            item = [i, row, ref_stem, ref_ts_us]
            if start_new_scene:
                if current_group:
                    # save current grop
                    scene_groups.append(current_group)
                current_group = [item]
                current_scene_start_us = ref_ts_us
            else:
                current_group.append(item)

            previous_ts_us = ref_ts_us

        if current_group:
            scene_groups.append(current_group)

        logger.info(
            "[nuScenes] Built %d scenes from %d valid samples.",
            len(scene_groups), len(rows),
        )

        ## Derive dataset-level metadata from the first valid row.
        first_stem = scene_groups[0][0][2]
        date_captured = _stem_to_date(first_stem) if first_stem else "2000-01-01"

        ## Build log.json and map.json (per-log).
        log_tok = _token()
        map_tok = _token()

        log_table = [{
            "token":         log_tok,
            "logfile":       base_scene_name,
            "vehicle":       self.config.vehicle_name,
            "date_captured": date_captured,
            "location":      location,
        }]
        map_filename = "maps/map.png"
        map_table = [{
            "token":      map_tok,
            "log_tokens": [log_tok],
            "category":   location,
            "filename":   map_filename,
        }]

        ## Build sample, ego_pose, sample_data, and scene tables.
        sample_table: List[dict] = []          # list of samples in a scene
        ego_pose_table: List[dict] = []        # list of poses in a scene
        scene_table: List[dict] = []           # list of scenes in a log

        # sample_data dict:
        #   - key: (scene_token, channel_name): [str, str]
        #   - value: [sample_data]: List[dict]
        # sd_by_scene_channel(scene_token, channel_name)-> List(dict)
        sd_by_scene_channel: Dict[Tuple[str, str], List[dict]] = collections.defaultdict(list)

        # annotation data dict:
        #   - key: (sample_token, instance_token): [str, str]
        #   - value: [sample_data]: List[dict]
        # an_by_scene_channel(sample_token, instance_token)-> List(dict)
        an_by_scene_instance: Dict[Tuple[str, str], List[dict]] = collections.defaultdict(list)

        instance_by_token: Dict[str, dict] = {}

        for scene_idx, scene_samples in enumerate(scene_groups):
            scene_tok = _token()
            # base_scene_name is unique across every merged log (enforced by
            # _check_unique_base_names), so this scene name is globally
            # unique without needing a separate log index.
            scene_name = f"{base_scene_name}_{scene_idx:04d}"

            sample_tokens: List[str] = [_token() for _ in scene_samples]

            logger.info(
                "[nuScenes] Building scene %s with %d samples.",
                scene_name, len(scene_samples),
            )

            ## instance table (one entry for each opponent, not tracked across scenes)
            if self.config.nuscenes.n_opponents == 1:
                instance_tok = _token()
                instance_by_token[instance_tok] = {
                    "token":  instance_tok,
                    "category_token":  cat_tok,
                    "nbr_annotations": "",            # to be filled later
                    "first_annotation_token":  "",    # to be filled later
                    "last_annotation_token":   "",    # to be filled later
                }

            else:
                raise NotImplementedError("[nuScenes] - multiple opponents management not implemented")


            for local_idx, item in enumerate(scene_samples):
                sample_idx, row, ref_stem, ref_ts_us = item

                # get sample tokens ("" if they are boundaries of a scene)
                sample_tok = sample_tokens[local_idx]
                prev_tok = sample_tokens[local_idx - 1] if local_idx > 0 else ""
                next_tok = sample_tokens[local_idx + 1] if local_idx < len(scene_samples) - 1 else ""

                ## ego_pose
                ep_tok  = _token()
                pose    = ego_poses_map.get(ref_stem)
                if pose and not is_label_relative:
                    ep_translation = [pose["x"], pose["y"], pose["z"]]
                    ep_rotation    = _quaternion_from_rpy(pose["roll"], pose["pitch"], pose["yaw"])
                else:
                    ep_translation = [0.0, 0.0, 0.0]
                    ep_rotation    = [1.0, 0.0, 0.0, 0.0]
                    logger.debug(
                        "[nuScenes] No ego pose for row %d (%s) - using identity.",
                        sample_idx, ref_stem,
                    )
                ego_pose_table.append({
                    "token":       ep_tok,
                    "timestamp":   ref_ts_us,
                    "rotation":    ep_rotation,
                    "translation": ep_translation,
                })

                ## sample
                sample_table.append({
                    "token":       sample_tok,
                    "timestamp":   ref_ts_us,
                    "scene_token": scene_tok,
                    "prev":        prev_tok,
                    "next":        next_tok,
                })

                ## sample_data: merged LiDAR
                lidar_ts_us = ref_ts_us
                lidar_fname = (
                    f"samples/{merged_ch}"
                    f"/{scene_name}__{merged_ch}__{lidar_ts_us}.pcd.bin"
                )
                out_lidar = os.path.join(self.output_dir, _META_DIR,  lidar_fname)
                self._write_lidar(row, lidar_sources, extrinsics, out_lidar, stage1_dir=stage1_dir)

                sd_by_scene_channel[(scene_tok, merged_ch)].append({
                    "token":                   _token(),
                    "sample_token":            sample_tok,
                    "ego_pose_token":          ep_tok,
                    "calibrated_sensor_token": cal_sensor_tokens[merged_ch],
                    "timestamp":               lidar_ts_us,
                    "fileformat":              "pcd",
                    "is_key_frame":            True,
                    "height":                  0,
                    "width":                   0,
                    "filename":                lidar_fname,
                    "prev":                    "",   # filled in second pass
                    "next":                    "",   # filled in second pass
                })

                ## sample_data: cameras
                for cam in active_cameras:
                    ch_name = cam_map[cam.key]
                    cam_stem = row.get(cam.key, "").strip()

                    if not cam_stem:
                        # Camera absent for this sample.
                        continue

                    try:
                        cam_ts_us = _stem_to_us(cam_stem)
                    except Exception as exc:
                        logger.warning(
                            "[nuScenes] Invalid camera timestamp for row %d, camera '%s': '%s': %s - skipping camera sample_data.",
                            sample_idx, cam.key, cam_stem, exc,
                        )
                        continue

                    cam_fname = (
                        f"samples/{ch_name}"
                        f"/{scene_name}__{ch_name}__{cam_ts_us}.jpg"
                    )
                    out_cam = os.path.join(self.output_dir, _META_DIR, cam_fname)
                    cam_intr = intrinsics.get(cam.name, {})
                    w, h = self._write_camera(cam, cam_stem, out_cam, cam_intr, stage1_dir=stage1_dir)

                    sd_by_scene_channel[(scene_tok, ch_name)].append({
                        "token":                   _token(),
                        "sample_token":            sample_tok,
                        "ego_pose_token":          ep_tok,
                        "calibrated_sensor_token": cal_sensor_tokens[ch_name],
                        "timestamp":               cam_ts_us,
                        "fileformat":              "jpg",
                        "is_key_frame":            True,
                        "height":                  h,
                        "width":                   w,
                        "filename":                cam_fname,
                        "prev":                    "",   # filled in second pass
                        "next":                    "",   # filled in second pass
                    })

                ## sample annotation
                inter_label_uri = os.path.join(stage1_dir, "labels", "lidar", ref_stem + ".json")
                anns = convert_label_to_nus(
                    tok=_token(),
                    sample_token=sample_tok,
                    instance_token=instance_tok,
                    attribute_tokens=[attr_tok],
                    visibility_token=default_visibility_token,   # default visibility: 80-100%
                    label_uri=inter_label_uri,
                    lidar_sample_uri=out_lidar,
                    ego_translation=ep_translation,
                    ego_rotation=ep_rotation
                )
                if anns is not None:
                    for ann in anns:
                        an_by_scene_instance[(scene_tok,instance_tok)].append(ann)

            ## scene entry
            scene_table.append({
                "token":              scene_tok,
                "log_token":          log_tok,
                "nbr_samples":        len(scene_samples),
                "first_sample_token": sample_tokens[0] if sample_tokens else "",
                "last_sample_token":  sample_tokens[-1] if sample_tokens else "",
                "name":               scene_name,
                "description":        description,
            })

        ## Fill prev / next links within each scene and channel.
        sample_data_table: List[dict] = []
        for (_scene_tok, _ch_name), sd_list in sd_by_scene_channel.items():
            # The list is already built in temporal order because scene_rows is temporal.
            for j, sd in enumerate(sd_list):
                sd["prev"] = sd_list[j - 1]["token"] if j > 0 else ""
                sd["next"] = sd_list[j + 1]["token"] if j < len(sd_list) - 1 else ""
            sample_data_table.extend(sd_list)

        ## Fill prev / next links within each scene and instance.
        sample_annotation_table: List[dict] = []
        for (_scene_tok, _inst_tok), sa_list in an_by_scene_instance.items():
            # The list is already built in temporal order because scene_rows is temporal.
            for j, sa in enumerate(sa_list):
                sa["prev"] = sa_list[j - 1]["token"] if j > 0 else ""
                sa["next"] = sa_list[j + 1]["token"] if j < len(sa_list) - 1 else ""
            sample_annotation_table.extend(sa_list)

            if _inst_tok in instance_by_token:
                ins = instance_by_token[_inst_tok]
                ins["nbr_annotations"] = len(sa_list)
                ins["first_annotation_token"] = sa_list[0]["token"] if sa_list else ""
                ins["last_annotation_token"] = sa_list[-1]["token"] if sa_list else ""

        instance_table = [
            ins for ins in instance_by_token.values()
            if isinstance(ins.get("nbr_annotations"), int) and ins["nbr_annotations"] > 0
        ]

        logger.info(
            "[nuScenes] Log '%s' complete: %d scenes, %d samples, %d sample_data entries.",
            base_scene_name, len(scene_table), len(sample_table), len(sample_data_table),
        )

        return {
            "log":               log_table,
            "map":               map_table,
            "calibrated_sensor": calibrated_sensor_table,
            "scene":             scene_table,
            "sample":            sample_table,
            "ego_pose":          ego_pose_table,
            "sample_data":       sample_data_table,
            "sample_annotation": sample_annotation_table,
            "instance":          instance_table,
        }

    # ------------------------------------------------------------------
    # Calibration table builders
    # ------------------------------------------------------------------

    def _build_lidar_sensor(
        self,
        merged_ch: str,
        sensor_table: List[dict],
        sensor_tokens: Dict[str, str],
    ) -> None:
        """
        Add the merged LiDAR channel definition to the (dataset-wide) sensor
        table. Built once for the whole run - see _build_shared_metadata.
        """
        lidar_sensor_tok = _token()
        sensor_tokens[merged_ch] = lidar_sensor_tok

        sensor_table.append({
            "token":    lidar_sensor_tok,
            "channel":  merged_ch,
            "modality": "lidar",
        })

    def _build_lidar_calibrated_sensor(
        self,
        merged_ch: str,
        sensor_tokens: Dict[str, str],
        calibrated_sensor_table: List[dict],
        cal_sensor_tokens: Dict[str, str],
    ) -> None:
        """
        Add this log's merged-LiDAR calibrated_sensor entry.

        The calibrated_sensor pose for the merged channel is always identity
        (translation=[0,0,0], rotation=[1,0,0,0]).  All individual LiDAR
        clouds are transformed into the reference frame (ring_center /
        center_of_gravity) by _merge_and_transform before the .pcd.bin is
        written, so the stored points are already expressed in the ego frame.
        """
        lidar_cal_tok = _token()
        cal_sensor_tokens[merged_ch] = lidar_cal_tok

        translation = [0.0, 0.0, 0.0]
        rotation    = [1.0, 0.0, 0.0, 0.0]

        calibrated_sensor_table.append({
            "token":            lidar_cal_tok,
            "sensor_token":     sensor_tokens[merged_ch],
            "translation":      translation,
            "rotation":         rotation,
            "camera_intrinsic": [],   # empty for LiDAR
        })

    def _build_camera_sensors(
        self,
        active_cameras: list,
        cam_map: Dict[str, str],
        sensor_table: List[dict],
        sensor_tokens: Dict[str, str],
    ) -> None:
        """
        Add one channel definition per active camera to the (dataset-wide)
        sensor table. Built once for the whole run - see
        _build_shared_metadata.
        """
        for cam in active_cameras:
            ch_name = cam_map[cam.key]
            cam_sensor_tok = _token()
            sensor_tokens[ch_name] = cam_sensor_tok

            sensor_table.append({
                "token":    cam_sensor_tok,
                "channel":  ch_name,
                "modality": "camera",
            })

    def _build_camera_calibrated_sensors(
        self,
        active_cameras: list,
        cam_map: Dict[str, str],
        extrinsics: dict,
        intrinsics: dict,
        sensor_tokens: Dict[str, str],
        calibrated_sensor_table: List[dict],
        cal_sensor_tokens: Dict[str, str],
    ) -> None:
        """
        Add one calibrated_sensor entry per active camera, for this log,
        using this log's own extrinsics/intrinsics (calibration.yaml).
        """
        for cam in active_cameras:
            ch_name = cam_map[cam.key]
            cam_cal_tok = _token()
            cal_sensor_tokens[ch_name] = cam_cal_tok

            T_raw = extrinsics.get(cam.frame_id)
            if T_raw is not None:
                T = np.array(T_raw, dtype=np.float64)   # T_cam_body -> reference
                R_opt_in_ref = T[:3, :3] @ _CAM_BODY_TO_OPT.T
                translation  = T[:3, 3].tolist()
                rotation     = _rotation_from_matrix(R_opt_in_ref)
            else:
                logger.warning(
                    "[nuScenes] No extrinsic for camera '%s' (%s); using identity.",
                    cam.key, cam.frame_id,
                )
                translation = [0.0, 0.0, 0.0]
                rotation    = [1.0, 0.0, 0.0, 0.0]

            cam_intr = intrinsics.get(cam.name, {})
            if cam_intr:
                K_new = _optimal_undistorted_K(cam_intr)
                camera_intrinsic = K_new.tolist()
            else:
                logger.warning(
                    "[nuScenes] No intrinsics for camera '%s'; camera_intrinsic will be empty.",
                    cam.name,
                )
                camera_intrinsic = []

            calibrated_sensor_table.append({
                "token":            cam_cal_tok,
                "sensor_token":     sensor_tokens[ch_name],
                "translation":      translation,
                "rotation":         rotation,
                "camera_intrinsic": camera_intrinsic,
            })

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def _write_lidar(
        self, row: dict, lidar_keys: List[str], extrinsics: dict, out_path: str,
        stage1_dir: Optional[str] = None,
    ) -> None:
        """
        Merge all LiDAR sources into one Nx5 float32 cloud and write it as a
        nuScenes .pcd.bin file (binary, row-major, no header).

        `stage1_dir` defaults to self.stage1_dir (the primary bag) but is
        overridden with the owning Stage-1 directory when converting a
        merged log.
        """
        stage1_dir = stage1_dir if stage1_dir is not None else self.stage1_dir
        merged = _merge_and_transform(row, lidar_keys, stage1_dir, extrinsics, point_reader=to_nuscenes_bin)
        if len(merged) == 0:
            logger.warning("[nuScenes] Empty merged point cloud written to: %s", out_path)
        merged.tofile(out_path)

    def _write_camera(
        self, cam, stem: str, out_path: str, intr: dict,
        stage1_dir: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Read the Stage-1 PNG, undistort it according to the camera model stored in "intr", and save as JPEG.

        Undistortion is applied model-aware:
          - PINHOLE : cv2.undistort               (radial k1,k2,k3 + tangential p1,p2)
          - FISHEYE : cv2.fisheye.undistortImage  (equidistant k1,k2,k3,k4)

        The optimal new camera matrix (alpha=0 / balance=0, no black border
        pixels) is used for rectification; the same matrix is written into
        calibrated_sensor.json by _build_camera_calibration so the two remain
        consistent.

        `stage1_dir` defaults to self.stage1_dir (the primary bag) but is
        overridden with the owning Stage-1 directory when converting a
        merged log.
        """
        stage1_dir = stage1_dir if stage1_dir is not None else self.stage1_dir
        src = os.path.join(stage1_dir, "samples", cam.key, f"{stem}.png")
        if not os.path.isfile(src):
            logger.warning(
                "[nuScenes] Camera image not found for %s at %s - skipping.", cam.key, src
            )
            return 0, 0

        img = Image.open(src).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        img_np = _undistort_image(img_np, intr)
        img = Image.fromarray(img_np)

        w, h = img.size
        img.save(out_path, "JPEG", quality=95) # Because nuScenes uses JPEGs, 95 to avoid size explosion
        return w, h

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _make_dirs(self, lidar_channel: str, active_cameras: list, cam_map: dict) -> None:
        """
        Create the full nuScenes output directory tree.
        """
        os.makedirs(os.path.join(self.output_dir, _META_DIR, _META_DIR), exist_ok=True)

        for ch in [lidar_channel] + [cam_map[c.key] for c in active_cameras]:
            os.makedirs(os.path.join(self.output_dir, _META_DIR, "samples", ch), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, _META_DIR, "sweeps",  ch), exist_ok=True)

        maps_dir = os.path.join(self.output_dir, _META_DIR, "maps")
        os.makedirs(maps_dir, exist_ok=True)

        map_src = getattr(getattr(self.config, "scene", None), "map_png_path", None)
        if map_src:
            map_src = os.path.expanduser(map_src)
            if os.path.isfile(map_src):
                shutil.copy2(map_src, os.path.join(maps_dir, "map.png"))
                logger.info("[nuScenes] Copied map image: %s -> %s/map.png", map_src, maps_dir)
            else:
                logger.warning("[nuScenes] map_png_path is set but file not found: %s", map_src)
        else:
            logger.warning(
                "[nuScenes] scene.map_png_path is not configured - maps/ folder will be empty. "
                "Set scene.map_png_path in config.yaml to include a map image."
            )

    @staticmethod
    def _write_json(directory: str, filename: str, data: list) -> None:
        """
        Write a JSON file with 4-space indentation into "directory".
        """
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=4)
        logger.debug("[nuScenes] Written: %s  (%d entries)", path, len(data))