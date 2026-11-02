"""
Intermediate-representation (Stage-1) validation pipeline. Runs the
IR check modules and writes a console + YAML report and diagnostic figures.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from ..common import loaders, report
from ..common.results import SectionResult
from ..common.sensors import SensorModel, load_sensor_model
from . import structure, timestamps, calibration, ego_pose, labels

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("validate_intermediate")


def _default_frames(n_rows: int, k: int = 3) -> List[int]:
    """Evenly spaced sample indices across the sequence."""
    if n_rows <= 0:
        return []
    if n_rows <= k:
        return list(range(n_rows))
    return [round(i * (n_rows - 1) / (k - 1)) for i in range(k)]


def validate(stage1: str, config: Optional[str], frames: Optional[List[int]],
             out_dir: Optional[str]) -> List[SectionResult]:
    if not os.path.isdir(stage1):
        raise FileNotFoundError(f"stage1 directory not found: {stage1}")

    sensor: Optional[SensorModel] = None
    if config:
        try:
            sensor = load_sensor_model(config)
        except Exception as exc:
            logger.warning("Could not load sensor model from %s: %s", config, exc)

    csv_path = os.path.join(stage1, "timestamps.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"timestamps.csv not found in {stage1}")
    header, rows = loaders.load_timestamps_csv(csv_path)

    if frames is None:
        frames = _default_frames(len(rows))
    frames = [f for f in frames if 0 <= f < len(rows)]

    figdir = os.path.join(out_dir or stage1, "validation_figures_intermediate")
    os.makedirs(figdir, exist_ok=True)

    def _safe(name, fn):
        try:
            return fn()
        except Exception as exc:
            logger.exception("section '%s' raised", name)
            s = SectionResult(name)
            s.fail(f"{name}_error", f"validation aborted with an unexpected error: {exc!r}")
            return s

    sections = [
        _safe("structure", lambda: structure.run(stage1, header, rows, sensor)),
        _safe("timestamps", lambda: timestamps.run(stage1, header, rows, sensor, figdir)),
        _safe("calibration", lambda: calibration.run(stage1, rows, sensor, frames, figdir)),
        _safe("ego_pose", lambda: ego_pose.run(stage1, header, rows, sensor, figdir)),
        _safe("labels", lambda: labels.run(stage1, rows, sensor, frames, figdir)),
    ]
    return sections


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the Stage-1 intermediate representation.")
    ap.add_argument("--stage1", required=True, help="Path to the Stage-1 output directory.")
    ap.add_argument("--config", default=None, help="Pipeline config.yaml (recommended).")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="Frame indices to deep-inspect (default: 3 evenly spaced).")
    ap.add_argument("--out", default=None, help="Report/figure output dir (default: stage1).")
    ap.add_argument("--show-passes", action="store_true", help="List passing checks too.")
    args = ap.parse_args(argv)

    try:
        sections = validate(args.stage1, args.config, args.frames, args.out)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    out_dir = args.out or args.stage1
    ctx = {"stage1_dir": args.stage1, "config": args.config or "(none)"}
    report.print_report("Intermediate Representation Validation", sections, ctx,
                        show_passes=args.show_passes)
    report.write_yaml_report("Intermediate Representation Validation", sections,
                             out_dir, "validation_intermediate.yaml", ctx)

    return 1 if report.overall_status(sections) == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())