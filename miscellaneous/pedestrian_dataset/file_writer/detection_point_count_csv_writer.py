"""Write one point-count row per frame, with one column per detection ID."""

from __future__ import annotations

import csv
import logging
import os


logger = logging.getLogger(__name__)


def write_detection_point_count_csv(frame_point_counts: dict[int, dict[int, int]], out_path: str) -> None:
    """Write stamp in seconds with nine decimal places; missing detections have empty cells."""
    if not frame_point_counts:
        logger.warning("No matched frames - detection_point_counts.csv will not be created.")
        return
    detection_ids = sorted({detection_id for counts in frame_point_counts.values() for detection_id in counts})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stamp"] + [f"detection_{detection_id}" for detection_id in detection_ids])
        for timestamp_ns in sorted(frame_point_counts):
            seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
            stamp = f"{seconds}.{nanoseconds:09d}"
            counts = frame_point_counts[timestamp_ns]
            writer.writerow([stamp] + [counts.get(detection_id, "") for detection_id in detection_ids])
    logger.info("Detection point-count CSV written: %s (%d rows)", out_path, len(frame_point_counts))
