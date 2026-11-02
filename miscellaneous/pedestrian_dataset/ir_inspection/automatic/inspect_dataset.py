#!/usr/bin/env python3
"""
Dataset Inspection and Balancing Tool.

Analyzes the 3D LiDAR annotations of a Stage-1 dataset, displays statistics about
their spatial and class distribution, and - if the user opts in - writes a
timestamps_balanced.csv with frames removed to satisfy configurable balancing
criteria.

Usage
    python inspect_dataset.py <stage1_dir> --input timestamps.csv
    python inspect_dataset.py <stage1_dir> --input timestamps_curated.csv --config balancing_config.yaml

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import plots

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

_WARNED: Set[str] = set()

ANNOTATION_COLUMNS = [
    "row_idx", "obj_type", "x", "y", "z", "distance", "yaw", "length", "width", "height",
]

DEFAULT_DISTANCE_BIN_EDGES_M = [0, 20, 40, 60, 80]
DEFAULT_RANDOM_SEED = 42


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_timestamps(path: str) -> pd.DataFrame:
    """Read a timestamps CSV. Empty cells stay as '' (not NaN), matching the
    Stage-1 writer's convention for absent sensors."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Timestamps file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_annotations(stage1_dir: str, timestamps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a flat table with one row per annotated object, read from
    labels/lidar/<lidar_front_ts>.json for every row of timestamps_df.

    A missing label file, an empty file, or a missing lidar_front cell all mean
    "zero annotations for this row" - not an error (mirrors the pipeline's own
    "missing labels -> skipped" convention).
    """
    labels_dir = os.path.join(stage1_dir, "labels", "lidar")
    records: List[dict] = []

    if "lidar_front" not in timestamps_df.columns:
        raise ValueError("Timestamps file has no 'lidar_front' column - not a valid Stage-1 timestamps file.")

    for row_idx, ts in timestamps_df["lidar_front"].items():
        ts = str(ts).strip()
        if not ts:
            continue

        label_path = os.path.join(labels_dir, f"{ts}.json")
        if not os.path.isfile(label_path):
            continue

        try:
            with open(label_path) as fh:
                anns = json.load(fh)
        except Exception as exc:
            _warn_once(f"bad_json_{ts}", f"Failed to read label file {label_path}: {exc}")
            continue

        if not isinstance(anns, list):
            continue

        for ann in anns:
            psr = ann.get("psr", {}) or {}
            pos = psr.get("position", {}) or {}
            rot = psr.get("rotation", {}) or {}
            scale = psr.get("scale", {}) or {}

            x, y, z = float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))
            obj_type = str(ann.get("obj_type") or "").strip() or "<missing>"

            records.append({
                "row_idx": row_idx,
                "obj_type": obj_type,
                "x": x, "y": y, "z": z,
                "distance": math.hypot(x, y),
                "yaw": float(rot.get("z", 0.0)),
                "length": float(scale.get("x", 0.0)),
                "width": float(scale.get("y", 0.0)),
                "height": float(scale.get("z", 0.0)),
            })

    return pd.DataFrame.from_records(records, columns=ANNOTATION_COLUMNS)


def load_balancing_config(path: str) -> dict:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = dict(raw.get("balancing", {}))
    cfg.setdefault("random_seed", DEFAULT_RANDOM_SEED)
    cfg.setdefault("distance_bin_edges_m", DEFAULT_DISTANCE_BIN_EDGES_M)
    cfg.setdefault("rules", [])
    return cfg


def load_calibration(stage1_dir: str) -> Dict[str, np.ndarray]:
    """Read <stage1_dir>/calibration.yaml's per-sensor extrinsic (4x4) matrices.
    Only used by the min_points_in_bbox rule. Missing file -> empty dict (points
    are then left in their raw sensor frame, with a one-time warning)."""
    path = os.path.join(stage1_dir, "calibration.yaml")
    if not os.path.isfile(path):
        _warn_once("no_calibration", f"calibration.yaml not found at {path} - "
                                      "min_points_in_bbox will use raw sensor-frame points.")
        return {}
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return {key: np.array(matrix, dtype=np.float64) for key, matrix in (raw.get("extrinsics") or {}).items()}


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "The min_points_in_bbox rule requires the 'open3d' package (already used "
            "elsewhere in this pipeline, e.g. utils/pcd_reader.py). Install it, or disable "
            "this rule in the balancing config."
        ) from exc
    return o3d


def _merge_lidar_points(
    stage1_dir: str, row: pd.Series, lidar_keys: List[str], extrinsics: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Read one frame's raw per-sensor PCDs (samples/<lidar_key>/<ts>.pcd) and
    transform each into the reference frame via calibration.yaml's extrinsics,
    mirroring utils/pcd_reader.py's merge_and_transform (re-implemented locally
    so this tool doesn't import pipeline source code).
    """
    o3d = _require_open3d()

    parts = []
    for key in lidar_keys:
        ts = str(row.get(key, "")).strip()
        if not ts:
            continue

        pcd_path = os.path.join(stage1_dir, "samples", key, f"{ts}.pcd")
        if not os.path.isfile(pcd_path):
            continue

        xyz = np.asarray(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float64)
        if xyz.size == 0:
            continue

        transform = extrinsics.get(key)
        if transform is None:
            _warn_once(f"no_extrinsic_{key}",
                       f"No extrinsic for '{key}' in calibration.yaml - using raw sensor-frame points.")
        else:
            xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])
            xyz = (transform @ xyz_h.T).T[:, :3]

        parts.append(xyz)

    return np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.float64)


def _count_points_in_bbox(points_xyz: np.ndarray, center: List[float], extent: List[float], yaw: float) -> int:
    """Count points inside a yaw-only oriented 3D box, mirroring
    utils/pcd_reader.py's get_num_points_in_bbox (re-implemented locally so this
    tool doesn't import pipeline source code)."""
    if len(points_xyz) == 0:
        return 0

    o3d = _require_open3d()

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xyz)
    rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0.0, 0.0, yaw]))
    obb = o3d.geometry.OrientedBoundingBox(np.array(center, dtype=np.float64), rotation, np.array(extent, dtype=np.float64))
    return len(obb.get_point_indices_within_bounding_box(cloud.points))


def _annotation_point_counts(stage1_dir: str, timestamps_df: pd.DataFrame, annotations: pd.DataFrame) -> pd.Series:
    """Number of LiDAR points inside each annotation's 3D box (aligned to
    annotations.index). Each frame's point clouds are merged once and reused
    for every annotation in that frame."""
    extrinsics = load_calibration(stage1_dir)
    lidar_keys = [c for c in timestamps_df.columns if c.startswith("lidar_")]
    counts = pd.Series(0, index=annotations.index, dtype=int)

    for row_idx, group in annotations.groupby("row_idx"):
        points = _merge_lidar_points(stage1_dir, timestamps_df.loc[row_idx], lidar_keys, extrinsics)
        for idx, ann in group.iterrows():
            counts.loc[idx] = _count_points_in_bbox(
                points, [ann["x"], ann["y"], ann["z"]], [ann["length"], ann["width"], ann["height"]], ann["yaw"],
            )

    return counts


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def frame_counts(timestamps_df: pd.DataFrame, annotations: pd.DataFrame) -> pd.Series:
    """Annotation count per row, including rows with zero (indexed like timestamps_df)."""
    if annotations.empty:
        counts = pd.Series(dtype=int)
    else:
        counts = annotations.groupby("row_idx").size()
    return counts.reindex(timestamps_df.index, fill_value=0)


def frame_count_histogram(counts: pd.Series) -> pd.Series:
    """Frame count per annotations-per-frame bucket: 0, 1, 2, 3, 3+."""
    capped = counts.clip(upper=3)
    hist = capped.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    hist.index = ["0", "1", "2", "3+"]
    return hist


def point_count_histogram(point_counts: pd.Series) -> pd.Series:
    """Label (annotation) count per exact number of LiDAR points inside its 3D box."""
    if point_counts.empty:
        return pd.Series(dtype=int)
    return point_counts.value_counts().sort_index()


def _distance_bin_labels(edges: List[float]) -> Tuple[List[float], List[str]]:
    all_edges = list(edges) + [np.inf]
    labels = []
    for lo, hi in zip(all_edges[:-1], all_edges[1:]):
        labels.append(f"{int(lo)}+" if not np.isfinite(hi) else f"{int(lo)}-{int(hi)}")
    return all_edges, labels


def _bin_distances(annotations: pd.DataFrame, edges: List[float]) -> pd.Series:
    all_edges, labels = _distance_bin_labels(edges)
    return pd.cut(annotations["distance"], bins=all_edges, labels=labels, right=False, include_lowest=True)


def distance_histogram(annotations: pd.DataFrame, edges: List[float]) -> pd.DataFrame:
    """Per distance bin: annotation count and distinct-frame count."""
    _, labels = _distance_bin_labels(edges)
    if annotations.empty:
        return pd.DataFrame({"annotation_count": 0, "frame_count": 0}, index=labels)

    bins = _bin_distances(annotations, edges)
    grouped = annotations.groupby(bins, observed=False)
    hist = pd.DataFrame({
        "annotation_count": grouped.size(),
        "frame_count": grouped["row_idx"].nunique(),
    })
    return hist.reindex(labels, fill_value=0)


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def _bev_quadrants(annotations: pd.DataFrame) -> pd.Series:
    """Label each annotation by its BEV quadrant relative to the CoG: forward (x)
    and left (y) each split at 0, giving top-left/top-right/bottom-left/bottom-right
    ("top" = forward, "left" = left), matching the BEV plot's axes."""
    forward_positive = annotations["x"] >= 0
    left_positive = annotations["y"] >= 0
    labels = np.select(
        [forward_positive & left_positive, forward_positive & ~left_positive, ~forward_positive & left_positive],
        ["top-left", "top-right", "bottom-left"],
        default="bottom-right",
    )
    return pd.Series(labels, index=annotations.index)


def _apply_group_cap(
    kept: pd.Index, annotations: pd.DataFrame, group_fn: Callable[[pd.DataFrame], pd.Series],
    cap: Union[int, Dict[str, Optional[int]], None], rng: np.random.Generator, rule_label: str,
) -> pd.Index:
    """
    Generic frame-drop loop shared by every "cap the annotation count of each
    group" rule (distance bin, BEV quadrant, ...): group_fn assigns each active
    annotation a group label, and frames are dropped one at a time from whichever
    group is currently most over its cap until every group is at or below it.

    cap can be a single number, applied to every group, or a {group_label: cap}
    mapping for independent per-group caps - groups missing from the mapping (or
    explicitly mapped to null) are left uncapped. cap=None disables the rule.
    """
    if cap is None:
        return kept

    def group_cap(group_label: str) -> float:
        limit = cap.get(group_label) if isinstance(cap, dict) else cap
        return math.inf if limit is None else limit

    while True:
        active = annotations[annotations["row_idx"].isin(kept)]
        if active.empty:
            break

        groups = group_fn(active)
        counts = active.groupby(groups, observed=False).size()
        caps = counts.index.map(group_cap)
        excess = (counts - caps)[counts > caps]
        if excess.empty:
            break

        target_group = excess.idxmax()
        candidates = active.loc[groups == target_group, "row_idx"].unique()
        if len(candidates) == 0:
            _warn_once(f"{rule_label}_{target_group}",
                       f"{rule_label}: group '{target_group}' exceeds cap but has no removable frames.")
            break

        kept = kept.drop(rng.choice(candidates))

    return kept


def _apply_empty_frame_ratio(
    kept: pd.Index, annotations: pd.DataFrame,
    max_ratio: Optional[float], rng: np.random.Generator,
) -> pd.Index:
    if max_ratio is None:
        return kept

    while len(kept) > 0:
        labeled = set(annotations.loc[annotations["row_idx"].isin(kept), "row_idx"])
        empty = np.array([idx for idx in kept if idx not in labeled])
        if len(empty) == 0 or len(empty) / len(kept) <= max_ratio:
            break
        kept = kept.drop(rng.choice(empty))

    return kept


def _apply_min_points_filter(
    kept: pd.Index, timestamps_df: pd.DataFrame, annotations: pd.DataFrame,
    min_points: Optional[int], stage1_dir: str,
) -> pd.Index:
    """
    Drop frames whose annotations all fall short of min_points LiDAR points
    inside their box - i.e. keep a frame if at least one of its annotations is
    well-observed. Frames with zero annotations to begin with are left alone
    (that's empty_frame_ratio's concern, not this rule's).
    """
    if min_points is None or annotations.empty:
        return kept

    point_counts = _annotation_point_counts(stage1_dir, timestamps_df, annotations)
    qualifies = point_counts >= min_points
    has_qualifying_annotation = qualifies.groupby(annotations["row_idx"]).any()
    frames_to_drop = has_qualifying_annotation.index[~has_qualifying_annotation]

    return kept.difference(frames_to_drop)


def apply_balancing(
    timestamps_df: pd.DataFrame, annotations: pd.DataFrame, config: dict, stage1_dir: str,
) -> Tuple[pd.Index, Dict[str, int]]:
    """
    Run the configured rules in the order they appear in config["rules"], each
    narrowing the same running set of kept row indices. Returns the final kept
    index (in timestamps_df's original order) and a per-rule drop count summary.
    """
    rng = np.random.default_rng(config["random_seed"])
    edges = config["distance_bin_edges_m"]
    kept = timestamps_df.index
    summary: Dict[str, int] = {}

    for rule in config["rules"]:
        rule_type = rule.get("type")
        before = len(kept)

        if rule_type == "distance_bin_cap":
            kept = _apply_group_cap(kept, annotations, lambda a: _bin_distances(a, edges),
                                     rule.get("max_annotations_per_bin"), rng, "distance_bin_cap")
        elif rule_type == "quadrant_cap":
            kept = _apply_group_cap(kept, annotations, _bev_quadrants,
                                     rule.get("max_annotations_per_quadrant"), rng, "quadrant_cap")
        elif rule_type == "empty_frame_ratio":
            kept = _apply_empty_frame_ratio(kept, annotations, rule.get("max_ratio"), rng)
        elif rule_type == "min_points_in_bbox":
            kept = _apply_min_points_filter(kept, timestamps_df, annotations, rule.get("min_points"), stage1_dir)
        else:
            logger.warning("Unknown balancing rule type '%s' - skipping.", rule_type)
            continue

        summary[rule_type] = before - len(kept)

    return kept, summary


def write_balanced_csv(timestamps_df: pd.DataFrame, kept: pd.Index, out_path: str) -> None:
    """Same header/column order/row order as timestamps_df, minus dropped rows.
    Never touches the input file. Atomic write (temp file + os.replace)."""
    tmp_path = out_path + ".tmp"
    timestamps_df.loc[timestamps_df.index.isin(kept)].to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and optionally balance a Stage-1 dataset.")
    parser.add_argument("stage1_dir", help="Path to a Stage-1 intermediate output directory")
    parser.add_argument("--input", required=True,
                         help="Timestamps file to analyze: a filename relative to stage1_dir "
                              "(e.g. timestamps.csv, timestamps_curated.csv, timestamps_balanced.csv) "
                              "or an absolute path")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "balancing_config.yaml"),
                         help="Path to the balancing configuration YAML (default: balancing_config.yaml next to this script)")
    args = parser.parse_args()

    stage1_dir = os.path.abspath(args.stage1_dir)
    input_path = args.input if os.path.isabs(args.input) else os.path.join(stage1_dir, args.input)

    timestamps_df = read_timestamps(input_path)
    if timestamps_df.empty:
        print(f"{input_path} has no rows - nothing to inspect.")
        return

    annotations = load_annotations(stage1_dir, timestamps_df)
    if annotations.empty:
        logger.warning(
            "No LiDAR annotations found under %s - statistics will be empty and balancing would have nothing to act on.",
            os.path.join(stage1_dir, "labels", "lidar"),
        )

    config = load_balancing_config(args.config)

    # --- statistics ---
    f_counts = frame_counts(timestamps_df, annotations)
    f_hist = frame_count_histogram(f_counts)
    d_hist = distance_histogram(annotations, config["distance_bin_edges_m"])

    if annotations.empty:
        point_counts = pd.Series(dtype=int)
    else:
        try:
            point_counts = _annotation_point_counts(stage1_dir, timestamps_df, annotations)
        except Exception as exc:
            logger.warning("Skipping points-in-bbox statistic: %s", exc)
            point_counts = pd.Series(dtype=int)
    p_hist = point_count_histogram(point_counts)

    plots.bev_density_figure(annotations)
    plots.distance_distribution_figure(d_hist)
    plots.frame_count_figure(f_hist)
    plots.point_count_figure(p_hist)
    plt.show()

    answer = input("Generate a balanced dataset? [y/N] ").strip().lower()
    if answer != "y":
        return

    kept, summary = apply_balancing(timestamps_df, annotations, config, stage1_dir)
    dropped = len(timestamps_df) - len(kept)

    print("Balancing complete.")
    print(f"  Rows: {len(timestamps_df)} -> {len(kept)} ({dropped} removed)")
    for rule_type, n in summary.items():
        print(f"  {rule_type}: {n} removed")

    out_path = os.path.join(stage1_dir, "timestamps_balanced.csv")
    write_balanced_csv(timestamps_df, kept, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()