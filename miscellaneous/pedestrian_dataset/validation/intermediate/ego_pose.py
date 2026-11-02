"""
IR ego-pose checks.

Figure
    ir_ego_trajectory.png: top-down (x, y) trajectory.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List

import numpy as np

from ..common import loaders, viz
from ..common.results import SectionResult
from ..common.sensors import SensorModel

_REF = "lidar_front"


def run(stage1: str, header: List[str], rows: List[Dict[str, str]],
        sensor: SensorModel, figdir: str) -> SectionResult:
    r = SectionResult("ego_pose")
    path = os.path.join(stage1, "ego_poses.csv")
    if not os.path.isfile(path):
        r.skip("ego_present", "ego_poses.csv absent — ego poses not available.")
        return r

    poses = loaders.load_ego_poses_csv(path)

    # Row-count alignment with groups.
    if len(poses) != len(rows):
        r.fail("ego_row_count",
               f"ego_poses.csv has {len(poses)} rows but timestamps.csv has {len(rows)}.",
               value=float(len(poses)))
    else:
        r.ok("ego_row_count", f"ego_poses.csv aligned with {len(rows)} groups.")

    # Stamp alignment with the reference LiDAR column.
    ref = _REF if _REF in header else (sensor.lidar_keys[0] if sensor and sensor.lidar_keys else None)
    if ref and ref in header and len(poses) == len(rows):
        mism = sum(1 for p, row in zip(poses, rows)
                   if p["stamp"] and row.get(ref, "").strip()
                   and p["stamp"] != row.get(ref, "").strip())
        if mism:
            r.fail("ego_stamp_align", f"{mism} ego stamps do not match the {ref} column.",
                   value=float(mism))
        else:
            r.ok("ego_stamp_align", f"ego stamps match the {ref} column.")

    # Availability + finiteness.
    n_missing = sum(1 for p in poses if p["x"] is None)
    frac = n_missing / max(len(poses), 1)
    if frac == 0:
        r.ok("ego_available", f"all {len(poses)} groups have a pose.")
    elif frac < 0.1:
        r.warn("ego_available", f"{n_missing}/{len(poses)} ({frac*100:.1f}%) groups have no pose.",
               value=frac)
    else:
        r.fail("ego_available", f"{n_missing}/{len(poses)} ({frac*100:.1f}%) groups have no pose.",
               value=frac)

    present = [p for p in poses if p["x"] is not None]
    nonfinite = sum(1 for p in present
                    if not all(math.isfinite(p[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")))
    if nonfinite:
        r.fail("ego_finite", f"{nonfinite} pose row(s) contain non-finite values.")
    elif present:
        r.ok("ego_finite", f"all {len(present)} poses are finite.")

    # Angle ranges.
    bad_ang = sum(1 for p in present if abs(p["yaw"]) > math.pi + 1e-3)
    if bad_ang:
        r.warn("ego_angles", f"{bad_ang} pose(s) have |yaw| > π (unexpected angle convention?).")

    # Continuity: flag large jumps relative to the median step.
    if len(present) >= 3:
        xy = np.array([[p["x"], p["y"]] for p in present], dtype=float)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        med = float(np.median(steps)) if len(steps) else 0.0
        thresh = max(med * 8.0, 5.0)
        n_jumps = int((steps > thresh).sum())
        if n_jumps:
            r.warn("ego_continuity",
                   f"{n_jumps} translation jump(s) > {thresh:.1f} m between consecutive groups "
                   f"(median step {med:.2f} m).", value=float(steps.max()))
        else:
            r.ok("ego_continuity", f"trajectory continuous (median step {med:.2f} m).")

        r.add_figure(viz.xy_trajectory(
            xy[:, 0].tolist(), xy[:, 1].tolist(),
            os.path.join(figdir, "ir_ego_trajectory.png"),
            "IR — ego trajectory (top-down)"))

    return r