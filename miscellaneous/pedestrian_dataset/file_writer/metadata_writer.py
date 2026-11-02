"""
Writes the Stage-1 metadata YAML file alongside the extracted data.

Contents:
  - total number of sample groups
  - sampling interval used
  - tolerance window used
  - bag file name(s) processed
  - date of extraction
  - date of capture (derived from the first sample group timestamp)
  - geographic location of the recording session
  - optional scene description
  - list of topics used
  - vehicle name
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import List, Optional

import yaml

from utils.config_loader import PipelineConfig

logger = logging.getLogger(__name__)

_NS_PER_SEC = 1_000_000_000


def write_metadata(
    config: PipelineConfig,
    bag_files: List[str],
    num_groups: int,
    out_path: str,
    first_timestamp_ns: Optional[int] = None,
) -> None:
    """
    Write a YAML metadata file to "out_path".

    Parameters
    config             : loaded PipelineConfig
    bag_files          : list of .mcap/.db3 filenames that were processed
    num_groups         : total number of sample groups extracted
    out_path           : full path for the output .yaml file
    first_timestamp_ns : nanosecond timestamp of the first matched sample group
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lidar_topics  = [l.topic for l in config.lidars]
    camera_topics = [c.topic for c in config.cameras]

    # Derive the actual recording date from the first sample group timestamp.
    if first_timestamp_ns is not None:
        sec = first_timestamp_ns // _NS_PER_SEC
        date_captured = datetime.datetime.utcfromtimestamp(sec).strftime("%Y-%m-%d")
    else:
        date_captured = None

    metadata = {
        "extraction_date":       datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_captured":         date_captured,
        "vehicle":               config.vehicle_name,
        "location":              config.scene.location,
        "scene_description":     config.scene.description,
        "bag_files":             bag_files,
        "num_sample_groups":     num_groups,
        "sampling_interval_ms":  config.matching.sampling_interval_ms,
        "tolerance_window_ms":   config.matching.tolerance_ms,
        "ego_pose_tolerance_ms": config.matching.ego_pose_tolerance_ms,
        "lidar_topics":          lidar_topics,
        "camera_topics":         camera_topics,
    }

    with open(out_path, "w") as fh:
        yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)

    logger.info("Metadata written: %s", out_path)