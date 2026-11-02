"""
Multi-Dataset Comparison Tool (view-only).

Loads several Stage-1 intermediate representations - each identified by its own
stage1_dir and its own timestamps file (e.g. timestamps.csv, timestamps_curated.csv,
timestamps_balanced.csv) - and displays their combined statistics in the same four
plots produced by inspect_dataset.py, as if they were a single unified dataset.

Usage
    python compare_datasets.py \\
        --dataset lidar_camera_bag_eva timestamps.csv \\
        --dataset lidar_camera_bag_eva timestamps_curated.csv \\
        --dataset lidar_camera_bag_minerva timestamps_balanced.csv \\
        --config balancing_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import inspect_dataset as insp
import plots

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-dataset loading
# ---------------------------------------------------------------------------

def _resolve_input_path(stage1_dir: str, input_arg: str) -> str:
    """Same relative/absolute resolution rule as inspect_dataset.py's --input."""
    return input_arg if os.path.isabs(input_arg) else os.path.join(stage1_dir, input_arg)


def _load_one_dataset(
    stage1_dir_arg: str, input_arg: str,
) -> Optional[Tuple[str, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Load a single dataset using inspect_dataset.py's own functions, unchanged.

    Returns (label, annotations, frame_counts, point_counts) with row_idx still
    local to this dataset (0-based) - the caller is responsible for making
    row_idx globally unique before merging across datasets. Returns None if the
    dataset has no rows (skipped with a warning, not a hard failure).
    """
    stage1_dir = os.path.abspath(stage1_dir_arg)
    input_path = _resolve_input_path(stage1_dir, input_arg)
    label = f"{os.path.basename(stage1_dir)}:{os.path.basename(input_path)}"

    timestamps_df = insp.read_timestamps(input_path)
    if timestamps_df.empty:
        logger.warning("%s has no rows - skipping this dataset.", label)
        return None

    annotations = insp.load_annotations(stage1_dir, timestamps_df)
    if annotations.empty:
        logger.warning("%s: no LiDAR annotations found - this dataset will contribute empty frames only.", label)

    frame_counts = insp.frame_counts(timestamps_df, annotations)

    if annotations.empty:
        point_counts = pd.Series(dtype=int)
    else:
        try:
            point_counts = insp._annotation_point_counts(stage1_dir, timestamps_df, annotations)
        except Exception as exc:
            logger.warning("%s: skipping points-in-bbox statistic: %s", label, exc)
            point_counts = pd.Series(dtype=int)

    print(f"  {label}: {len(timestamps_df)} frames, {len(annotations)} annotations")
    return label, annotations, frame_counts, point_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple Stage-1 datasets by displaying their combined statistics "
                    "in the same plots, as if they were a single unified dataset. View-only: "
                    "no balancing, no output files.",
    )
    parser.add_argument(
        "--dataset", nargs=2, action="append", required=True, metavar=("STAGE1_DIR", "TIMESTAMPS_INPUT"),
        help="A Stage-1 directory and its timestamps file to include in the comparison "
             "(filename relative to STAGE1_DIR, e.g. timestamps.csv, timestamps_curated.csv, "
             "timestamps_balanced.csv, or an absolute path). Repeat this flag for each dataset.",
    )
    parser.add_argument(
        "--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "balancing_config.yaml"),
        help="Path to the balancing configuration YAML (default: balancing_config.yaml next to this "
             "script). Only its distance_bin_edges_m is used here - the shared bin edges for the "
             "distance distribution plot. Its balancing rules are never read or applied.",
    )
    args = parser.parse_args()

    print(f"Loading {len(args.dataset)} dataset(s)...")
    annotations_list: List[pd.DataFrame] = []
    frame_counts_list: List[pd.Series] = []
    point_counts_list: List[pd.Series] = []
    frame_offset = 0

    for stage1_dir_arg, input_arg in args.dataset:
        loaded = _load_one_dataset(stage1_dir_arg, input_arg)
        if loaded is None:
            continue
        _label, annotations, frame_counts, point_counts = loaded

        # Make row_idx globally unique across datasets before merging.
        annotations = annotations.copy()
        annotations["row_idx"] = annotations["row_idx"] + frame_offset
        frame_offset += len(frame_counts)

        annotations_list.append(annotations)
        frame_counts_list.append(frame_counts)
        point_counts_list.append(point_counts)

    if not annotations_list:
        print("No datasets loaded - nothing to compare.")
        return

    annotations_all = pd.concat(annotations_list, ignore_index=True)
    frame_counts_all = pd.concat(frame_counts_list, ignore_index=True)
    point_counts_all = pd.concat(point_counts_list, ignore_index=True)

    config = insp.load_balancing_config(args.config)
    edges = config["distance_bin_edges_m"]

    # --- statistics (identical calls to inspect_dataset.py's main(), against
    # the merged tables) ---
    f_hist = insp.frame_count_histogram(frame_counts_all)
    d_hist = insp.distance_histogram(annotations_all, edges)
    p_hist = insp.point_count_histogram(point_counts_all)

    plots.bev_density_figure(annotations_all)
    plots.distance_distribution_figure(d_hist)
    plots.frame_count_figure(f_hist)
    plots.point_count_figure(p_hist)
    plt.show()


if __name__ == "__main__":
    main()