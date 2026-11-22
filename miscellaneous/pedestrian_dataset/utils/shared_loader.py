import csv
import logging
import os
from typing import Dict, Optional

import yaml

from timestamp_matching.matcher import SampleGroup
from utils.timestamp_utils import filename_to_ns

logger = logging.getLogger(__name__)


def _load_calibration(stage1_dir: str) -> dict:
    """Load the unified calibration.yaml."""
    cal_path = os.path.join(stage1_dir, "calibration.yaml")
    if not os.path.isfile(cal_path):
        raise FileNotFoundError(
            f"calibration.yaml not found in Stage-1 directory: {stage1_dir}\n"
            "Run Stage 1 first (or re-run it to regenerate the calibration file)."
        )
    with open(cal_path, "r") as fh:
        return yaml.safe_load(fh)
    

def _load_timestamps_csv(stage1_dir: str, filename: str = "timestamps.csv") -> tuple[list[str], list[dict]]:
    """
    Read a Stage-1 timestamps file (timestamps.csv, timestamps_curated.csv,
    timestamps_balanced.csv, or any other file sharing the same schema).

    Parameters
        stage1_dir : path to the Stage-1 intermediate folder
        filename   : which timestamps file to read (default: "timestamps.csv")

    Returns: (header, rows)
        header : list of column names
        rows   : list of dicts  {col_name -> timestamp_str or ""}
    """
    csv_path = os.path.join(stage1_dir, filename)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{filename} not found in: {stage1_dir}")
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def load_sample_groups(
    csv_path: str,
    lidar_keys: list[str],
    camera_keys: list[str],
) -> list[SampleGroup]:
    """Read exact sensor header timestamps from a Stage-1 timestamps CSV."""
    directory, filename = os.path.split(csv_path)
    header, rows = _load_timestamps_csv(directory, filename)
    missing = [key for key in lidar_keys if key not in header]
    if missing:
        raise ValueError(f"Timestamps CSV is missing LiDAR columns: {missing}")

    groups = []
    for row_number, row in enumerate(rows, start=2):
        lidar_cells = {key: (row.get(key) or "").strip() for key in lidar_keys}
        if not all(lidar_cells.values()):
            raise ValueError(f"Timestamps CSV row {row_number} has an empty LiDAR timestamp")
        lidar_timestamps = {key: filename_to_ns(cell) for key, cell in lidar_cells.items()}

        # Camera columns are optional, and blank cells represent missing images.
        camera_timestamps = {}
        for key in camera_keys:
            cell = (row.get(key) or "").strip()
            camera_timestamps[key] = filename_to_ns(cell) if cell else None
        groups.append(SampleGroup(lidar_timestamps, camera_timestamps))
    return groups


def _load_ego_poses_csv(stage1_dir: str) -> Dict[str, Optional[dict]]:
    """
    Load ego_poses.csv produced by Stage 1.
    """
    path = os.path.join(stage1_dir, "ego_poses.csv")
    if not os.path.isfile(path):
        logger.warning("[Loaders] ego_poses.csv not found in %s - all ego poses will be set to identity.", stage1_dir)
        return {}

    result: Dict[str, Optional[dict]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stamp = row.get("stamp", "").strip()
            if not stamp:
                continue
            x_str = row.get("x", "").strip()
            if not x_str:
                result[stamp] = None
                continue
            try:
                result[stamp] = {
                    "x":     float(row["x"]),
                    "y":     float(row["y"]),
                    "z":     float(row["z"]),
                    "roll":  float(row["roll"]),
                    "pitch": float(row["pitch"]),
                    "yaw":   float(row["yaw"]),
                }
            except (KeyError, ValueError) as exc:
                logger.warning("[Loaders] Could not parse ego pose row (%s): %s", stamp, exc)
                result[stamp] = None

    return result
