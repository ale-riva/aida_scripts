"""
Writes inspection_config.yaml: the small amount of metadata the frame
inspection tool (inspect_frames.py) needs and that is not otherwise
recoverable from timestamps.csv / calibration.yaml / info_extraction.yaml
alone.
"""

from __future__ import annotations

import logging
import os

import yaml

logger = logging.getLogger(__name__)


def write_inspection_config(config, stage1_out: str) -> dict:
    """
    Write <stage1_out>/inspection_config.yaml.

    Parameters
    config     : PipelineConfig (same object already used to call
                 extract_calibration() in main.py::run_stage1)
    stage1_out : Stage-1 output directory for this bag

    Returns the dict that was written.
    """
    sensors: dict = {}

    for lidar in config.lidars:
        sensors[lidar.key] = {
            "type": "lidar",
            "frame_id": lidar.frame_id,
        }

    for camera in config.cameras:
        sensors[camera.key] = {
            "type": "camera",
            "frame_id": camera.frame_id,
            "name": camera.name,
        }

    inspection_config = {
        # frame_id that labels/lidar/*.json PSR values are anchored to.
        "annotation_anchor_frame": config.ego_pose.child_frame,
        # pipeline key -> calibration.yaml lookup keys
        "sensors": sensors,
    }

    os.makedirs(stage1_out, exist_ok=True)
    out_path = os.path.join(stage1_out, "inspection_config.yaml")
    with open(out_path, "w") as fh:
        yaml.dump(inspection_config, fh, default_flow_style=False, sort_keys=False)

    logger.info(
        "Inspection config written: %s  (%d sensors, anchor_frame=%s)",
        out_path, len(sensors), config.ego_pose.child_frame,
    )
    return inspection_config