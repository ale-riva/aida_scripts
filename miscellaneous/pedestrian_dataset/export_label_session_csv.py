#!/usr/bin/env python3
"""Export edited SUSTechPOINTS classes as a timestamp-by-detection matrix.

Usage: python export_label_session_csv.py /path/to/labelling_session
Output: labelling_session/detection_labels.csv (stamp is Unix seconds,
with nine decimal places and no scientific notation).
Missing detections have empty cells; empty frames are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from utils.timestamp_utils import filename_to_ns


CLASS_IDS = {
    "Pedestrian": 0,
    "Car": 1,
    "Truck": 2,
    "Bus": 3,
    "Motorcycle": 4,
    "Bicycle": 5,
    "Unknown1": 6,
    "Unknown2": 7,
}


def export_labels(session_dir: Path, output_path: Path | None = None) -> Path:
    """Read session_dir/label without modifying the source annotations."""
    label_dir = session_dir / "label"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    label_paths = sorted(label_dir.glob("*.json"))
    if not label_paths:
        raise ValueError(f"No JSON labels found in {label_dir}")

    frames: dict[int, dict[int, int]] = {}
    for path in label_paths:
        timestamp = filename_to_ns(path.stem)
        labels = json.loads(path.read_text())
        if not isinstance(labels, list):
            raise ValueError(f"{path}: expected a list of label objects")
        classes: dict[int, int] = {}
        for label in labels:
            if not isinstance(label, dict):
                raise ValueError(f"{path}: expected each label to be an object")
            raw_id = str(label.get("obj_id", ""))
            if not raw_id.isascii() or not raw_id.isdecimal():
                raise ValueError(f"{path}: invalid detection ID {raw_id!r}")
            detection_id = int(raw_id)
            if detection_id in classes:
                raise ValueError(f"{path}: duplicate detection ID {detection_id}")
            class_name = label.get("obj_type")
            if not isinstance(class_name, str) or class_name not in CLASS_IDS:
                raise ValueError(f"{path}: unsupported label class {class_name!r}")
            classes[detection_id] = CLASS_IDS[class_name]
        frames[timestamp] = classes

    detection_ids = sorted({key for classes in frames.values() for key in classes})
    output_path = output_path if output_path is not None else session_dir / "detection_labels.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stamp", *[f"detection_{key}" for key in detection_ids]])
        for timestamp, classes in sorted(frames.items()):
            seconds, nanoseconds = divmod(abs(timestamp), 1_000_000_000)
            stamp = f"{'-' if timestamp < 0 else ''}{seconds}.{nanoseconds:09d}"
            writer.writerow([stamp, *[classes.get(key, "") for key in detection_ids]])
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, help="Labelling session containing label/*.json")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV path")
    args = parser.parse_args()
    try:
        output = export_labels(args.session_dir, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Labels exported to {output}")


if __name__ == "__main__":
    main()
