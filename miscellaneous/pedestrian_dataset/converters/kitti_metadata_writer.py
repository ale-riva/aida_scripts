"""
Stage-2 helper: writes the KITTI dataset's metadata/ folder.

Output structure
<output_dir>/metadata/
├── sample_mapping.csv   # one row per KITTI sample, traceable back to the
│                        #   Stage-1 timestamps.csv row and source sensor files
├── kitti_metadata.yaml  # dataset-level summary + sample_mapping.csv schema
├── labeled.txt          # KITTI ids whose label_2/<id>.txt has >=1 object
└── unlabeled.txt        # KITTI ids whose label_2/<id>.txt is empty
"""

from __future__ import annotations

import csv
import datetime
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample record (one row of sample_mapping.csv)
# ---------------------------------------------------------------------------

@dataclass
class SampleRecord:
    """One KITTI sample's traceability record, built by KITTIConverter.convert()."""
    kitti_id:           str             # 6-digit KITTI id, e.g. "000000"
    source_frame_index: int             # row index in Stage-1 timestamps.csv
    is_labeled:         bool            # True if label_2/<kitti_id>.txt has >=1 object
    camera_key:         str             # pipeline camera slot, e.g. "camera_0"
    camera_name:        str             # human-readable camera name (settings yaml)
    camera_frame_id:    str             # URDF frame_id of the camera
    camera_timestamp:   str             # timestamp stem of the source image
    lidar_timestamps:   Dict[str, str]  # {lidar_key -> timestamp stem}
    velodyne_source_id: str             # kitti_id owning the real velodyne .bin


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_date_captured(stage1_dir: str) -> Optional[str]:
    """Read 'date_captured' from the Stage-1 info_extraction.yaml, if present."""
    meta_path = os.path.join(stage1_dir, "info_extraction.yaml")
    if not os.path.isfile(meta_path):
        logger.warning(
            "[KITTI metadata] info_extraction.yaml not found in %s - "
            "date_captured will be null.", stage1_dir,
        )
        return None
    with open(meta_path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("date_captured")


def _write_sample_mapping_csv(
    records: List[SampleRecord], lidar_keys: List[str], out_path: str,
) -> None:
    """Write metadata/sample_mapping.csv."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lidar_cols = [f"{key}_timestamp" for key in lidar_keys]
    header = (
        ["kitti_id", "source_frame_index", "is_labeled",
         "camera_key", "camera_name", "camera_frame_id", "camera_timestamp"]
        + lidar_cols
        + ["velodyne_source_id"]
    )

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for rec in records:
            row = [
                rec.kitti_id, rec.source_frame_index, int(rec.is_labeled),
                rec.camera_key, rec.camera_name, rec.camera_frame_id, rec.camera_timestamp,
            ]
            row += [rec.lidar_timestamps.get(key, "") for key in lidar_keys]
            row.append(rec.velodyne_source_id)
            writer.writerow(row)

    logger.info("[KITTI metadata] sample_mapping.csv written: %s (%d rows)", out_path, len(records))


def _write_id_list(path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        if ids:
            fh.write("\n".join(ids) + "\n")


def _write_label_lists(records: List[SampleRecord], labeled_path: str, unlabeled_path: str) -> None:
    """Write metadata/labeled.txt and metadata/unlabeled.txt."""
    labeled_ids   = sorted(rec.kitti_id for rec in records if rec.is_labeled)
    unlabeled_ids = sorted(rec.kitti_id for rec in records if not rec.is_labeled)
    _write_id_list(labeled_path, labeled_ids)
    _write_id_list(unlabeled_path, unlabeled_ids)
    logger.info(
        "[KITTI metadata] labeled.txt / unlabeled.txt written: %d labeled, %d unlabeled.",
        len(labeled_ids), len(unlabeled_ids),
    )


def _write_metadata_yaml(
    config,
    stage1_dir: str,
    cameras: list,
    lidar_keys: List[str],
    records: List[SampleRecord],
    num_unique_lidar_frames: int,
    out_path: str,
) -> None:
    """Write metadata/kitti_metadata.yaml."""
    labeled_samples = sum(1 for rec in records if rec.is_labeled)

    metadata = {
        "provenance": {
            "generated_by":          "bag_to_fusion_dataset",
            "kitti_conversion_date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "vehicle":               config.vehicle_name,
            "location":              config.scene.location,
            "date_captured":         _load_date_captured(stage1_dir),
        },
        "cameras": {
            "selection_order": [cam.key for cam in cameras],
            "count":           len(cameras),
        },
        "lidars": {
            "merged_keys": lidar_keys,
        },
        "samples": {
            "total_samples":             len(records),
            "total_unique_lidar_frames": num_unique_lidar_frames,
            "labeled_samples":           labeled_samples,
            "unlabeled_samples":         len(records) - labeled_samples,
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)

    logger.info("[KITTI metadata] kitti_metadata.yaml written: %s", out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_kitti_metadata(
    config,
    stage1_dir: str,
    output_dir: str,
    cameras: list,
    lidar_keys: List[str],
    records: List[SampleRecord],
    num_unique_lidar_frames: int,
) -> None:
    """
    Write the KITTI dataset's metadata/ folder: sample_mapping.csv,
    kitti_metadata.yaml, labeled.txt and unlabeled.txt.

    Parameters
    config                   : PipelineConfig
    stage1_dir               : Stage-1 intermediate folder (read-only; used to recover date_captured from info_extraction.yaml)
    output_dir               : KITTI output directory (the <bag>_kitti folder)
    cameras                  : resolved, ordered list of CameraConfig used for KITTI interleaving (= kitti_converter's "cameras")
    lidar_keys               : sorted list of configured LiDAR keys merged into velodyne/*.bin
    records                  : one SampleRecord per generated KITTI sample, in kitti_id order
    num_unique_lidar_frames  : number of real (non-symlinked) velodyne/*.bin files written
    """
    if not records:
        logger.warning("[KITTI metadata] No samples to describe - metadata/ will not be written.")
        return

    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    _write_sample_mapping_csv(records, lidar_keys, os.path.join(meta_dir, "sample_mapping.csv"))
    _write_label_lists(records, os.path.join(meta_dir, "labeled.txt"), os.path.join(meta_dir, "unlabeled.txt"))
    _write_metadata_yaml(
        config, stage1_dir, cameras, lidar_keys, records, num_unique_lidar_frames,
        os.path.join(meta_dir, "kitti_metadata.yaml"),
    )