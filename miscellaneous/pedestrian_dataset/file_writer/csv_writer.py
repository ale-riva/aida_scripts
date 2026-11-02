"""
Writes the Stage-1 CSV file: one row per sample group, one column per sensor.
Format:
    lidar_front,lidar_left,lidar_right,camera_0,camera_1,...,camera_6
    <ts>,       <ts>,      <ts>,       <ts>,    <ts>,    ...,<ts>      <- absent cameras = empty cell

LiDAR columns preserve settings order; camera columns keep their sorted order.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import List

from timestamp_matching.matcher import SampleGroup
from utils.timestamp_utils import ns_to_filename

logger = logging.getLogger(__name__)


def write_csv(groups: List[SampleGroup], out_path: str) -> None:
    """
    Write "groups" to the CSV at "out_path".
    """
    if not groups:
        logger.warning("No sample groups to write, CSV will not be created.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lidar_cols  = list(groups[0].lidar_timestamps)
    camera_cols = sorted(groups[0].camera_timestamps.keys())
    header = lidar_cols + camera_cols

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for group in groups:
            row = []

            for col in lidar_cols:
                ts = group.lidar_timestamps.get(col)
                row.append(ns_to_filename(ts) if ts is not None else "")

            for col in camera_cols:
                ts = group.camera_timestamps.get(col)
                row.append(ns_to_filename(ts) if ts is not None else "")

            writer.writerow(row)

    logger.info("CSV written: %s  (%d rows)", out_path, len(groups))