"""Write one confidence row per frame, with one column per detection ID."""

from __future__ import annotations

import csv
import logging
import os

from utils.timestamp_utils import ns_to_filename

logger = logging.getLogger(__name__)


def write_detection_confidence_csv(frame_confidences: dict[int, dict[int, float]], out_path: str) -> None:
    """Write stamp,detection_<id>,...; missing detections have empty cells."""
    if not frame_confidences:
        logger.warning("No matched frames - detection_confidences.csv will not be created.")
        return
    detection_ids = sorted({detection_id for scores in frame_confidences.values() for detection_id in scores})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stamp"] + [f"detection_{detection_id}" for detection_id in detection_ids])
        for timestamp_ns in sorted(frame_confidences):
            scores = frame_confidences[timestamp_ns]
            writer.writerow([ns_to_filename(timestamp_ns)] + [scores.get(detection_id, "") for detection_id in detection_ids])
    logger.info("Detection confidence CSV written: %s (%d rows)", out_path, len(frame_confidences))
