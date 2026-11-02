"""
KITTI dataset validation pipeline.
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
from . import structure, calibration, consistency, labels

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("validate_kitti")


def _default_frames(stage1: Optional[str], kitti_dir: str, k: int = 3) -> List[int]:
    """Evenly spaced IR frame indices (if IR available) else low frame indices."""
    n = 0
    if stage1 and os.path.isfile(os.path.join(stage1, "timestamps.csv")):
        _, rows = loaders.load_timestamps_csv(os.path.join(stage1, "timestamps.csv"))
        n = len(rows)
    if n <= 0:
        return list(range(k))
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def validate(kitti_dir: str, stage1: Optional[str], config: Optional[str],
             cameras: Optional[List[str]], frames: Optional[List[int]],
             out_dir: Optional[str]) -> List[SectionResult]:
    if not os.path.isdir(kitti_dir):
        raise FileNotFoundError(f"KITTI directory not found: {kitti_dir}")

    sensor: Optional[SensorModel] = None
    if config:
        try:
            sensor = load_sensor_model(config)
        except Exception as exc:
            logger.warning("Could not load sensor model from %s: %s", config, exc)

    if frames is None:
        frames = _default_frames(stage1, kitti_dir)

    figdir = os.path.join(out_dir or kitti_dir, "validation_figures_kitti")
    os.makedirs(figdir, exist_ok=True)

    def _safe(name, fn):
        try:
            return fn()
        except Exception as exc:
            logger.exception("section '%s' raised", name)
            s = SectionResult(name)
            s.fail(f"{name}_error", f"validation aborted with an unexpected error: {exc!r}")
            return s

    return [
        _safe("structure", lambda: structure.run(kitti_dir)),
        _safe("calibration", lambda: calibration.run(kitti_dir, frames, stage1, sensor)),
        _safe("consistency", lambda: consistency.run(kitti_dir, stage1, sensor, cameras, frames)),
        _safe("labels", lambda: labels.run(kitti_dir, frames, stage1, sensor, cameras, figdir)),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a generated KITTI dataset.")
    ap.add_argument("--kitti", required=True, help="Path to the KITTI output directory.")
    ap.add_argument("--stage1", default=None, help="IR directory (enables consistency checks).")
    ap.add_argument("--config", default=None, help="Pipeline config.yaml (recommended).")
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="Override the KITTI camera selection (keys, in order).")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="IR frame indices to deep-inspect (default: 3 evenly spaced).")
    ap.add_argument("--out", default=None, help="Report/figure output dir (default: kitti dir).")
    ap.add_argument("--show-passes", action="store_true", help="List passing checks too.")
    args = ap.parse_args(argv)

    try:
        sections = validate(args.kitti, args.stage1, args.config, args.cameras,
                            args.frames, args.out)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    out_dir = args.out or args.kitti
    ctx = {"kitti_dir": args.kitti, "stage1_dir": args.stage1 or "(none)",
           "config": args.config or "(none)"}
    report.print_report("KITTI Dataset Validation", sections, ctx,
                        show_passes=args.show_passes)
    report.write_yaml_report("KITTI Dataset Validation", sections, out_dir,
                             "validation_kitti.yaml", ctx)

    return 1 if report.overall_status(sections) == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())