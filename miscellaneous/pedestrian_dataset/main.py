"""Detection-driven pedestrian dataset extraction."""

from __future__ import annotations

import argparse
import logging
import os
import re
from operator import itemgetter

from utils.config_loader import load_cluster_config
from utils.shared_loader import load_sample_groups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)


def sort_bag_files(bag_dir: str, extension: str) -> list[str]:
    """Sort numbered bag splits numerically, then list the other files alphabetically."""
    pattern = re.compile(rf".*_(\d+)\.{extension}$")
    numbered_files = []
    other_files = []

    for name in os.listdir(bag_dir):
        match = pattern.search(name)
        if match:
            numbered_files.append((int(match.group(1)), name))
        elif name.endswith(f".{extension}"):
            other_files.append(name)

    numbered_files.sort(key=itemgetter(0))
    return [name for _, name in numbered_files] + sorted(other_files)


def bag_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]

    # Prefer MCAP when a directory contains both storage formats.
    for extension in ("mcap", "db3"):
        names = sort_bag_files(path, extension)
        if names:
            return [os.path.join(path, name) for name in names]

    raise ValueError(f"No .mcap or .db3 files found in: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pedestrian data matched to 3-D detections",
    )
    parser.add_argument("config", help="YAML configuration file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cluster_config(args.config)

    # A CSV requests exact sensor timestamps, even when it contains zero rows.
    timestamp_groups = None
    if config.timestamps_csv:
        camera_keys = [camera.key for camera in config.cameras]
        timestamp_groups = load_sample_groups(
            config.timestamps_csv,
            [config.lidar_key],
            camera_keys,
        )
    if config.reuse_dataset and not timestamp_groups:
        raise ValueError("Reuse workflow needs a non-empty timestamps CSV")

    # Keep CLI help and configuration errors available without the ROS bag libraries.
    from extraction.cluster_extractor import run_detection_extraction

    files = bag_files(config.bag_dir)
    run_detection_extraction(
        config,
        files,
        cluster=config.cluster,
        timestamp_groups=timestamp_groups,
    )


if __name__ == "__main__":
    main()
