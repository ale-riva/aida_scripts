"""
IR structure and metadata checks.
"""

from __future__ import annotations

import os
from typing import Dict, List

from ..common import loaders
from ..common.results import SectionResult
from ..common.sensors import SensorModel


def _check_required_files(stage1: str, r: SectionResult) -> None:
    required = ["timestamps.csv", "calibration.yaml", "info_extraction.yaml"]
    optional = ["ego_poses.csv"]
    for f in required:
        p = os.path.join(stage1, f)
        if os.path.isfile(p):
            r.ok(f"file_{f}", f"{f} present.")
        else:
            r.fail(f"file_{f}", f"required file missing: {f}")
    for f in optional:
        p = os.path.join(stage1, f)
        (r.ok if os.path.isfile(p) else r.warn)(
            f"file_{f}", f"{f} {'present' if os.path.isfile(p) else 'missing (ego poses unavailable)'}.")


def _check_sample_consistency(stage1: str, header: List[str], rows: List[Dict[str, str]],
                              sensor: SensorModel, r: SectionResult) -> None:
    """Cross-check timestamps.csv cells against files in samples/<sensor>/."""
    samples_dir = os.path.join(stage1, "samples")
    if not os.path.isdir(samples_dir):
        r.fail("samples_dir", f"samples/ directory missing: {samples_dir}")
        return
    r.ok("samples_dir", "samples/ directory present.")

    lidar_keys = set(sensor.lidar_keys) if sensor else set()
    for col in header:
        sensor_dir = os.path.join(samples_dir, col)
        ext = ".pcd" if col in lidar_keys else (".png" if col.startswith("camera_") else None)
        # Without a sensor model, fall back to extension-by-name heuristic.
        if ext is None:
            ext = ".pcd" if "lidar" in col else ".png"

        csv_stamps = {row[col].strip() for row in rows if row.get(col, "").strip()}
        if not os.path.isdir(sensor_dir):
            if csv_stamps:
                r.fail(f"sample_dir_{col}",
                       f"{col}: {len(csv_stamps)} timestamps in CSV but samples/{col}/ is missing.")
            else:
                r.skip(f"sample_dir_{col}", f"{col}: no timestamps and no directory (absent sensor).")
            continue

        disk = {os.path.splitext(f)[0] for f in os.listdir(sensor_dir) if f.endswith(ext)}
        missing = csv_stamps - disk          # referenced by CSV, absent on disk
        orphan = disk - csv_stamps           # on disk, not referenced by CSV

        if missing:
            r.fail(f"sample_files_{col}",
                   f"{col}: {len(missing)}/{len(csv_stamps)} CSV timestamps have no "
                   f"{ext} file (e.g. {sorted(missing)[:2]}).", value=float(len(missing)))
        elif orphan:
            r.warn(f"sample_files_{col}",
                   f"{col}: {len(orphan)} extra {ext} file(s) on disk not referenced by CSV.",
                   value=float(len(orphan)))
        else:
            r.ok(f"sample_files_{col}",
                 f"{col}: all {len(csv_stamps)} sample files present and referenced.",
                 value=float(len(csv_stamps)))


def _check_metadata(stage1: str, rows: List[Dict[str, str]], sensor: SensorModel,
                    r: SectionResult) -> None:
    meta_path = os.path.join(stage1, "info_extraction.yaml")
    if not os.path.isfile(meta_path):
        return
    meta = loaders.load_metadata(meta_path)

    declared = meta.get("num_sample_groups")
    if declared is None:
        r.warn("meta_group_count", "info_extraction.yaml has no num_sample_groups field.")
    elif int(declared) != len(rows):
        r.fail("meta_group_count",
               f"num_sample_groups={declared} but timestamps.csv has {len(rows)} rows.",
               value=float(declared))
    else:
        r.ok("meta_group_count", f"num_sample_groups matches CSV rows ({len(rows)}).",
             value=float(len(rows)))

    # Cross-check matching parameters against the config (when available).
    if sensor is not None:
        for meta_key, cfg_val, label in [
            ("sampling_interval_ms", sensor.sampling_interval_ms, "sampling_interval_ms"),
            ("tolerance_window_ms", sensor.tolerance_ms, "tolerance_ms"),
        ]:
            mv = meta.get(meta_key)
            if mv is None or cfg_val is None:
                continue
            if abs(float(mv) - float(cfg_val)) > 1e-6:
                r.warn(f"meta_{meta_key}",
                       f"{meta_key}={mv} in metadata differs from config {label}={cfg_val}.")
            else:
                r.ok(f"meta_{meta_key}", f"{meta_key} matches config ({mv}).")

    # Camera topic count vs vehicle expectation.
    cam_topics = meta.get("camera_topics") or []
    if sensor is not None and sensor.num_cameras is not None and cam_topics:
        if len(cam_topics) != sensor.num_cameras:
            r.warn("meta_camera_count",
                   f"metadata lists {len(cam_topics)} camera topics, vehicle declares "
                   f"{sensor.num_cameras}.")
        else:
            r.ok("meta_camera_count",
                 f"camera topic count matches vehicle ({sensor.num_cameras}).")


def run(stage1: str, header: List[str], rows: List[Dict[str, str]],
        sensor: SensorModel) -> SectionResult:
    r = SectionResult("structure")
    _check_required_files(stage1, r)
    _check_sample_consistency(stage1, header, rows, sensor, r)
    _check_metadata(stage1, rows, sensor, r)
    return r