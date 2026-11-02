#!/usr/bin/env python3
"""
Generate KITTI dataset splits:
    <dataset>/ImageSets/train.txt
    <dataset>/ImageSets/val.txt
    <dataset>/ImageSets/trainval.txt

Notes
  * test.txt and the testing/ directory are intentionally NOT generated.
  * trainval.txt is the union of all ids (train + val).
  * The split is per-sample (per image id). Two samples of the same frame may land in different splits.

Usage
    python generate_splits.py --dataset /path/to/<bag>_kitti --val-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys


def _collect_frame_ids(dataset_dir: str) -> list[str]:
    """Return sorted 6-digit frame ids from training/velodyne/*.bin."""
    velo_dir = os.path.join(dataset_dir, "training", "velodyne")
    if not os.path.isdir(velo_dir):
        raise FileNotFoundError(
            f"velodyne directory not found: {velo_dir}\n"
            "Run the KITTI conversion (Stage 2) first."
        )
    ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(velo_dir)
        if f.endswith(".bin")
    )
    return ids


def _write_list(path: str, ids: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        if ids:
            fh.write("\n".join(ids) + "\n")


def generate_splits(dataset_dir: str, val_ratio: float, seed: int) -> None:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1], got {val_ratio}")

    frame_ids = _collect_frame_ids(dataset_dir)
    if not frame_ids:
        raise ValueError(
            f"No .bin frames found under {dataset_dir}/training/velodyne"
        )

    shuffled = list(frame_ids)
    random.Random(seed).shuffle(shuffled)

    n_val = int(round(len(shuffled) * val_ratio))
    val_ids      = sorted(shuffled[:n_val])
    train_ids    = sorted(shuffled[n_val:])
    trainval_ids = sorted(frame_ids)

    image_sets = os.path.join(dataset_dir, "ImageSets")
    _write_list(os.path.join(image_sets, "train.txt"),    train_ids)
    _write_list(os.path.join(image_sets, "val.txt"),      val_ids)
    _write_list(os.path.join(image_sets, "trainval.txt"), trainval_ids)

    print(
        f"Wrote splits to {image_sets}: "
        f"train={len(train_ids)}, val={len(val_ids)}, trainval={len(trainval_ids)}"
    )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate KITTI ImageSets splits (train/val/trainval)."
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to the KITTI dataset root (the folder containing training/).",
    )
    p.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Fraction of frames assigned to val (default: 0.2).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for a reproducible split (default: 0).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        generate_splits(args.dataset, args.val_ratio, args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()