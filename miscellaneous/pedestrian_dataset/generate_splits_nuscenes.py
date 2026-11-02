#!/usr/bin/env python3
"""
Generate NUSCENES dataset splits:
    <dataset>/<version>/splits.json

Notes:
    the custom splits are named "custom_train" and "custom_val"

Usage
    python generate_splits.py --dataset /path/to/<bag>_nuscenes --val-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import json
from pathlib import Path


def _collect_scene_names(dataset_dir: str) -> list[str]:
    """Return scene names from v1.0-trainval/scene.json."""
    scenes_json = Path(dataset_dir) / "v1.0-trainval" / "scene.json"

    if not scenes_json.is_file():
        raise FileNotFoundError(
            f"Scene json not found: {scenes_json}\n"
            "Run the nuScenes conversion (Stage 2) first."
        )

    with scenes_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scene_names = [item["name"] for item in data]

    return scene_names

def generate_splits(dataset_dir: str, val_ratio: float, seed: int) -> None:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1], got {val_ratio}")

    scenes_names = _collect_scene_names(dataset_dir)
    if not scenes_names:
        raise ValueError(
            f"No .bin frames found under {dataset_dir}/training/velodyne"
        )

    shuffled = list(scenes_names)
    random.Random(seed).shuffle(shuffled)

    n_val = int(round(len(shuffled) * val_ratio))
    val_ids      = sorted(shuffled[:n_val])
    train_ids    = sorted(shuffled[n_val:])
    trainval_ids = sorted(scenes_names)

    with open(os.path.join(dataset_dir, "v1.0-trainval", "splits.json"), "w", encoding="utf-8") as fh:
        json.dump({"custom_train": train_ids, "custom_val": val_ids}, fh, indent=2)
        fh.write("\n")

    print(
        f"Wrote splits to {dataset_dir}: "
        f"train={len(train_ids)}, val={len(val_ids)}, trainval={len(trainval_ids)}"
    )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate nuScenes splits (trainval)."
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to the nuScenes dataset root (the folder containing sweeps/).",
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