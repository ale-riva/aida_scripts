#!/usr/bin/env python3
"""
Merge two or more KITTI-format datasets (as produced by kitti_converter.py)
into a single KITTI-format dataset.

Assumption
    All input datasets were generated from the same vehicle and the same
    LiDAR key set (e.g. all EVA). This is validated against each input's
    metadata/kitti_metadata.yaml; datasets with a different vehicle or LiDAR
    configuration are rejected.

Output structure (same layout as a single Stage-2 KITTI dataset)
<output_dir>/
├── ImageSets/                   # empty - run generate_splits.py afterwards
├── training/
│   ├── image_2/                 # 000000.png ... in --inputs order
│   ├── velodyne/                # merged; shared-frame symlinks are rebuilt
│   ├── calib/
│   └── label_2/
└── metadata/
    ├── sample_mapping.csv       # remapped to new ids + a source_dataset column
    ├── kitti_metadata.yaml      # merged summary + per-source provenance
    ├── labeled.txt
    └── unlabeled.txt

Usage
    python merge_kitti_datasets.py --inputs ds1_kitti ds2_kitti ds3_kitti --output merged_kitti
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_REQUIRED_TRAINING_SUBDIRS = ("image_2", "velodyne", "calib", "label_2")
_REQUIRED_METADATA_FILES = ("sample_mapping.csv", "kitti_metadata.yaml", "labeled.txt", "unlabeled.txt")
_MAX_KITTI_ID = 999_999  # inclusive - KITTI ids are fixed-width 6 digits


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class _SourceRow:
    """One sample_mapping.csv row from an input dataset, before renumbering."""
    old_id:             str
    source_dataset:     str             # input dataset's folder name
    source_frame_index: str
    is_labeled:         str
    camera_key:         str
    camera_name:        str
    camera_frame_id:    str
    camera_timestamp:   str
    lidar_timestamps:   Dict[str, str]
    velodyne_source_id: str             # old id (within the same dataset) owning the real .bin


@dataclass
class _MergedRow:
    """One output sample_mapping.csv row, after id remapping."""
    new_id:             str
    source_dataset:     str
    source_frame_index: str
    is_labeled:         str
    camera_key:         str
    camera_name:        str
    camera_frame_id:    str
    camera_timestamp:   str
    lidar_timestamps:   Dict[str, str]
    velodyne_source_id: str             # new id owning the real .bin


# ---------------------------------------------------------------------------
# Input validation / loading
# ---------------------------------------------------------------------------

def _check_dataset_layout(dataset_dir: str) -> None:
    """Verify a KITTI dataset has the expected training/ and metadata/ files."""
    training_dir = os.path.join(dataset_dir, "training")
    for sub in _REQUIRED_TRAINING_SUBDIRS:
        if not os.path.isdir(os.path.join(training_dir, sub)):
            raise FileNotFoundError(f"Missing training/{sub} in dataset: {dataset_dir}")

    meta_dir = os.path.join(dataset_dir, "metadata")
    for fname in _REQUIRED_METADATA_FILES:
        if not os.path.isfile(os.path.join(meta_dir, fname)):
            raise FileNotFoundError(f"Missing metadata/{fname} in dataset: {dataset_dir}")


def _load_kitti_metadata(dataset_dir: str) -> dict:
    with open(os.path.join(dataset_dir, "metadata", "kitti_metadata.yaml"), "r") as fh:
        return yaml.safe_load(fh) or {}


def _validate_common_lidar_keys(dataset_dirs: List[str], metadatas: List[dict]) -> list:
    """
    Verify all input datasets share the same LiDAR key set (still a hard
    requirement: it governs the fixed {lidar_key}_timestamp columns written
    into the merged sample_mapping.csv). Returns the common LiDAR key list.
    """
    ref_lidars = metadatas[0].get("lidars", {}).get("merged_keys", [])

    for dataset_dir, meta in zip(dataset_dirs, metadatas):
        lidars = meta.get("lidars", {}).get("merged_keys", [])
        if lidars != ref_lidars:
            raise ValueError(
                f"LiDAR configuration mismatch: '{dataset_dirs[0]}' has LiDAR keys "
                f"{ref_lidars}, '{dataset_dir}' has {lidars}. "
                "Merging datasets with different LiDAR configurations is not supported."
            )

    return ref_lidars


def _collect_camera_layouts(dataset_dirs: List[str], metadatas: List[dict]) -> tuple[list, dict]:
    """
    Collect each input's camera selection/order. Unlike LiDAR keys, camera
    layout is allowed to differ between inputs - every sample is already
    self-describing via its own calib.txt / label_2.txt - so a mismatch is
    only logged, not rejected.

    Returns (common_cameras, layouts_by_dataset), where common_cameras is the
    shared selection/order if all inputs agree, or None if they differ.
    """
    layouts_by_dataset = {}
    for dataset_dir, meta in zip(dataset_dirs, metadatas):
        name = os.path.basename(os.path.normpath(dataset_dir))
        layouts_by_dataset[name] = meta.get("cameras", {}).get("selection_order", [])

    distinct_layouts = {tuple(layout) for layout in layouts_by_dataset.values()}
    if len(distinct_layouts) > 1:
        details = "; ".join(f"{name}: {layout}" for name, layout in layouts_by_dataset.items())
        logger.warning(
            "[merge] Input datasets have different camera layouts (%s). "
            "Each sample remains self-describing via its own calib.txt/label_2.txt, "
            "but camera_key/camera_name in sample_mapping.csv are only comparable "
            "within the same source_dataset; kitti_metadata.yaml cameras.selection_order "
            "will be set to null - see source_datasets for the per-source layout.",
            details,
        )
        return None, layouts_by_dataset

    return next(iter(distinct_layouts)), layouts_by_dataset


def _validate_common_vehicle(metadatas: List[dict]) -> str:
    vehicles = {m.get("provenance", {}).get("vehicle") for m in metadatas}
    if len(vehicles) > 1:
        raise ValueError(
            f"Input datasets report different vehicles: {sorted(v for v in vehicles if v)}. "
            "Merging datasets captured on different vehicles is not supported."
        )
    return vehicles.pop()


def _read_sample_mapping(dataset_dir: str, lidar_keys: List[str]) -> List[_SourceRow]:
    """Read metadata/sample_mapping.csv, sorted by (original) kitti_id."""
    path = os.path.join(dataset_dir, "metadata", "sample_mapping.csv")
    name = os.path.basename(os.path.normpath(dataset_dir))

    rows = []
    with open(path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(_SourceRow(
                old_id             = row["kitti_id"],
                source_dataset     = name,
                source_frame_index = row["source_frame_index"],
                is_labeled         = row["is_labeled"],
                camera_key         = row["camera_key"],
                camera_name        = row["camera_name"],
                camera_frame_id    = row["camera_frame_id"],
                camera_timestamp   = row["camera_timestamp"],
                lidar_timestamps   = {key: row.get(f"{key}_timestamp", "") for key in lidar_keys},
                velodyne_source_id = row["velodyne_source_id"],
            ))

    rows.sort(key=lambda r: r.old_id)
    return rows


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _make_output_dirs(output_dir: str) -> None:
    os.makedirs(os.path.join(output_dir, "ImageSets"), exist_ok=True)
    for sub in _REQUIRED_TRAINING_SUBDIRS:
        os.makedirs(os.path.join(output_dir, "training", sub), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)


def _copy_sample_files(dataset_dir: str, old_id: str, output_dir: str, new_id: str) -> None:
    """Copy image_2/calib/label_2 for one sample; velodyne is handled separately."""
    shutil.copy2(
        os.path.join(dataset_dir, "training", "image_2", f"{old_id}.png"),
        os.path.join(output_dir, "training", "image_2", f"{new_id}.png"),
    )
    shutil.copy2(
        os.path.join(dataset_dir, "training", "calib", f"{old_id}.txt"),
        os.path.join(output_dir, "training", "calib", f"{new_id}.txt"),
    )
    shutil.copy2(
        os.path.join(dataset_dir, "training", "label_2", f"{old_id}.txt"),
        os.path.join(output_dir, "training", "label_2", f"{new_id}.txt"),
    )


def _merge_velodyne(dataset_dir: str, rows: List[_SourceRow], id_map: Dict[str, str], output_dir: str) -> None:
    """
    Rebuild training/velodyne/ for one input dataset under its new ids.

    A row owns the real .bin when its old_id == velodyne_source_id; every
    other row is a relative symlink to its (remapped) owner, matching the
    real/symlink structure kitti_converter.py originally produced. Owners are
    resolved from sample_mapping.csv rather than by inspecting the input's
    actual symlinks on disk, since the CSV is the authoritative source.
    """
    velo_out = os.path.join(output_dir, "training", "velodyne")

    owners = [r for r in rows if r.old_id == r.velodyne_source_id]
    others = [r for r in rows if r.old_id != r.velodyne_source_id]

    # Owners must be materialized before symlinks that point at them.
    for r in owners:
        shutil.copy2(
            os.path.join(dataset_dir, "training", "velodyne", f"{r.old_id}.bin"),
            os.path.join(velo_out, f"{id_map[r.old_id]}.bin"),
        )

    for r in others:
        new_id       = id_map[r.old_id]
        new_owner_id = id_map[r.velodyne_source_id]
        os.symlink(f"{new_owner_id}.bin", os.path.join(velo_out, f"{new_id}.bin"))


def _write_sample_mapping_csv(rows: List[_MergedRow], lidar_keys: List[str], out_path: str) -> None:
    header = (
        ["kitti_id", "source_dataset", "source_frame_index", "is_labeled",
         "camera_key", "camera_name", "camera_frame_id", "camera_timestamp"]
        + [f"{key}_timestamp" for key in lidar_keys]
        + ["velodyne_source_id"]
    )
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            row = [
                r.new_id, r.source_dataset, r.source_frame_index, r.is_labeled,
                r.camera_key, r.camera_name, r.camera_frame_id, r.camera_timestamp,
            ]
            row += [r.lidar_timestamps.get(key, "") for key in lidar_keys]
            row.append(r.velodyne_source_id)
            writer.writerow(row)

    logger.info("[merge] sample_mapping.csv written: %s (%d rows)", out_path, len(rows))


def _write_label_lists(rows: List[_MergedRow], labeled_path: str, unlabeled_path: str) -> None:
    labeled_ids   = sorted(r.new_id for r in rows if int(r.is_labeled))
    unlabeled_ids = sorted(r.new_id for r in rows if not int(r.is_labeled))
    for path, ids in ((labeled_path, labeled_ids), (unlabeled_path, unlabeled_ids)):
        with open(path, "w") as fh:
            if ids:
                fh.write("\n".join(ids) + "\n")

    logger.info(
        "[merge] labeled.txt / unlabeled.txt written: %d labeled, %d unlabeled.",
        len(labeled_ids), len(unlabeled_ids),
    )


def _write_merged_metadata_yaml(
    common_cameras,
    lidar_keys: list,
    merged_rows: List[_MergedRow],
    source_summaries: List[dict],
    common_vehicle,
    out_path: str,
) -> None:
    labeled_samples = sum(1 for r in merged_rows if int(r.is_labeled))
    num_unique_lidar_frames = sum(1 for r in merged_rows if r.new_id == r.velodyne_source_id)

    locations = {s["location"] for s in source_summaries if s["location"] is not None}
    common_location = locations.pop() if len(locations) == 1 else None
    if common_location is None and len(locations) > 1:
        logger.warning(
            "[merge] Input datasets have different 'location' values - "
            "kitti_metadata.yaml provenance.location set to null; see source_datasets."
        )

    metadata = {
        "provenance": {
            "generated_by":  "merge_kitti_datasets",
            "merge_date":    datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "vehicle":       common_vehicle,
            "location":      common_location,
            "date_captured": None,  # spans multiple sources - see source_datasets below
        },
        "cameras": {
            # null when inputs have different camera layouts - see
            # source_datasets[].cameras for the per-source selection/order.
            "selection_order": common_cameras,
            "count":           len(common_cameras) if common_cameras is not None else None,
        },
        "lidars": {
            "merged_keys": lidar_keys,
        },
        "samples": {
            "total_samples":             len(merged_rows),
            "total_unique_lidar_frames": num_unique_lidar_frames,
            "labeled_samples":           labeled_samples,
            "unlabeled_samples":         len(merged_rows) - labeled_samples,
        },
        "source_datasets": source_summaries,
    }

    with open(out_path, "w") as fh:
        yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)

    logger.info("[merge] kitti_metadata.yaml written: %s", out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def merge_kitti_datasets(input_dirs: List[str], output_dir: str, overwrite: bool = False) -> None:
    if len(input_dirs) < 2:
        raise ValueError("At least two input datasets are required for a merge.")

    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir} "
                "(pass --overwrite to merge into it anyway)."
            )
        logger.warning("[merge] Output directory already exists and --overwrite was set: %s", output_dir)

    for d in input_dirs:
        _check_dataset_layout(d)

    metadatas = [_load_kitti_metadata(d) for d in input_dirs]
    lidar_keys = _validate_common_lidar_keys(input_dirs, metadatas)
    common_cameras, camera_layouts_by_name = _collect_camera_layouts(input_dirs, metadatas)
    common_vehicle = _validate_common_vehicle(metadatas)

    _make_output_dirs(output_dir)

    merged_rows: List[_MergedRow] = []
    source_summaries: List[dict] = []
    counter = 0

    for dataset_dir, meta in zip(input_dirs, metadatas):
        name = os.path.basename(os.path.normpath(dataset_dir))
        rows = _read_sample_mapping(dataset_dir, lidar_keys)
        if not rows:
            logger.warning("[merge] No samples found in %s - skipping.", dataset_dir)
            continue

        id_map: Dict[str, str] = {}
        for r in rows:
            if counter > _MAX_KITTI_ID:
                raise ValueError(
                    f"Merged dataset exceeds the 6-digit KITTI id range "
                    f"(max {_MAX_KITTI_ID + 1} samples)."
                )
            id_map[r.old_id] = f"{counter:06d}"
            counter += 1

        logger.info(
            "[merge] %s: %d samples -> ids %s..%s",
            name, len(rows), id_map[rows[0].old_id], id_map[rows[-1].old_id],
        )

        for r in rows:
            new_id = id_map[r.old_id]
            _copy_sample_files(dataset_dir, r.old_id, output_dir, new_id)
            merged_rows.append(_MergedRow(
                new_id             = new_id,
                source_dataset     = r.source_dataset,
                source_frame_index = r.source_frame_index,
                is_labeled         = r.is_labeled,
                camera_key         = r.camera_key,
                camera_name        = r.camera_name,
                camera_frame_id    = r.camera_frame_id,
                camera_timestamp   = r.camera_timestamp,
                lidar_timestamps   = r.lidar_timestamps,
                velodyne_source_id = id_map[r.velodyne_source_id],
            ))

        _merge_velodyne(dataset_dir, rows, id_map, output_dir)

        provenance = meta.get("provenance", {})
        source_summaries.append({
            "name":                  name,
            "path":                  os.path.abspath(dataset_dir),
            "kitti_id_range":        [id_map[rows[0].old_id], id_map[rows[-1].old_id]],
            "num_samples":           len(rows),
            "location":              provenance.get("location"),
            "date_captured":         provenance.get("date_captured"),
            "kitti_conversion_date": provenance.get("kitti_conversion_date"),
            "cameras":               camera_layouts_by_name[name],
        })

    if not merged_rows:
        raise ValueError("No samples found in any input dataset - nothing to merge.")

    meta_dir = os.path.join(output_dir, "metadata")
    _write_sample_mapping_csv(merged_rows, lidar_keys, os.path.join(meta_dir, "sample_mapping.csv"))
    _write_label_lists(merged_rows, os.path.join(meta_dir, "labeled.txt"), os.path.join(meta_dir, "unlabeled.txt"))
    _write_merged_metadata_yaml(
        common_cameras, lidar_keys, merged_rows, source_summaries, common_vehicle,
        os.path.join(meta_dir, "kitti_metadata.yaml"),
    )

    logger.info(
        "[merge] Complete: %d samples from %d datasets -> %s "
        "(ImageSets/ is empty - run generate_splits.py next).",
        len(merged_rows), len(input_dirs), output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two or more KITTI-format datasets into a single KITTI-format dataset."
    )
    parser.add_argument(
        "--inputs", required=True, nargs="+", metavar="DATASET_DIR",
        help="Paths to the input KITTI dataset roots, in the order they should "
             "appear in the merged dataset (e.g. --inputs ds1_kitti ds2_kitti).",
    )
    parser.add_argument(
        "--output", required=True, metavar="OUTPUT_DIR",
        help="Path to the merged KITTI dataset root (created if missing).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow writing into a non-empty --output directory.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        merge_kitti_datasets(args.inputs, args.output, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()