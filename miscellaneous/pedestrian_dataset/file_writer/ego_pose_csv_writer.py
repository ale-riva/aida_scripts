"""
Writes the Stage-1 ego_poses.csv file.

Schema
header, x, y, z, roll, pitch, yaw

header           : canonical filename timestamp of the reference LiDAR frame
                   (<YYYYMMDD_HHMMSS_ns> identical to the .pcd stem for lidar_front)
x, y, z          : ego translation in metres, expressed in the world frame
roll, pitch, yaw : intrinsic XYZ Euler angles in radians, derived from the pose quaternion
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional

from timestamp_matching.matcher import SampleGroup
from utils.timestamp_utils import ns_to_filename

logger = logging.getLogger(__name__)

_COLUMNS = ["stamp", "x", "y", "z", "roll", "pitch", "yaw"]


def write_ego_pose_csv(
    groups: List[SampleGroup],
    ego_poses: List[Optional[Dict[str, float]]],
    ref_key: str,
    out_path: str,
) -> None:
    """
    Write "ego_poses.csv" to "out_path".

    Parameters
    groups     : matched sample groups (one per row)
    ego_poses  : one pose dict (x, y, z, roll, pitch, yaw) or "None" per group
    ref_key    : logical key of the reference LiDAR ("lidar_front")
    out_path   : absolute path for the output "ego_poses.csv" file
    """
    if not groups:
        logger.warning("No sample groups provided - ego_poses.csv will not be created.")
        return

    if len(groups) != len(ego_poses):
        raise ValueError(
            f"groups ({len(groups)}) and ego_poses ({len(ego_poses)}) must have "
            f"the same length."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    none_count = sum(1 for p in ego_poses if p is None)
    if none_count:
        logger.warning(
            "%d / %d ego poses are None (no data available) - affected rows will be written with empty numeric cells.",
            none_count, len(ego_poses),
        )

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_COLUMNS)

        for group, pose in zip(groups, ego_poses):
            ref_ts = group.lidar_timestamps[ref_key]
            header = ns_to_filename(ref_ts)

            if pose is None:
                writer.writerow([header, "", "", "", "", "", ""])
            else:
                writer.writerow([
                    header,
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["roll"],
                    pose["pitch"],
                    pose["yaw"],
                ])

    logger.info("Ego pose CSV written: %s  (%d rows)", out_path, len(groups))