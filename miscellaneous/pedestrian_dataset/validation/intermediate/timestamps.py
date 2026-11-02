"""
IR temporal-coherence checks.

Figures
    dim_dt_histograms.png: |delta t| distribution per secondary sensor.
    dim_gap_timeline.png:  inter-group gap scatter.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..common import loaders, viz
from ..common.results import SectionResult
from ..common.sensors import SensorModel

_MS_PER_NS = 1e-6
_REF = "lidar_front"


def _ns(stem: str) -> Optional[int]:
    return loaders.filename_to_ns(stem)


def run(stage1: str, header: List[str], rows: List[Dict[str, str]],
        sensor: SensorModel, figdir: str) -> SectionResult:
    r = SectionResult("timestamps")

    ref = _REF if _REF in header else (sensor.lidar_keys[0] if sensor and sensor.lidar_keys else None)
    if ref is None or ref not in header:
        r.fail("reference_column", f"reference LiDAR column not found in CSV header: {header}")
        return r

    tol = sensor.tolerance_ms if (sensor and sensor.tolerance_ms) else None
    interval = sensor.sampling_interval_ms if (sensor and sensor.sampling_interval_ms) else None

    secondary = [c for c in header if c != ref]

    # --- per-sensor |delta t| ---
    deltas: Dict[str, List[float]] = {c: [] for c in secondary}
    for row in rows:
        ref_ns = _ns(row.get(ref, ""))
        if ref_ns is None:
            continue
        for c in secondary:
            s_ns = _ns(row.get(c, ""))
            if s_ns is not None:
                deltas[c].append(abs(s_ns - ref_ns) * _MS_PER_NS)

    if tol is None:
        r.skip("dt_tolerance", "no tolerance from config; skipping |deltat| threshold checks.")
    else:
        for c, vals in deltas.items():
            if not vals:
                continue
            arr = np.array(vals)
            n_over = int((arr > tol).sum())
            pct = n_over / len(arr) * 100
            if n_over == 0:
                r.ok(f"dt_{c}", f"{c}: all {len(arr)} |delta t| ≤ {tol:g} ms (max={arr.max():.2f}).",
                     value=float(arr.max()))
            elif pct < 5.0:
                r.warn(f"dt_{c}", f"{c}: {n_over}/{len(arr)} ({pct:.1f}%) |delta t| > {tol:g} ms "
                       f"(max={arr.max():.2f}).", value=float(arr.max()))
            else:
                r.fail(f"dt_{c}", f"{c}: {n_over}/{len(arr)} ({pct:.1f}%) |delta t| > {tol:g} ms "
                       f"(max={arr.max():.2f}).", value=float(arr.max()))

    # --- missing camera slots ---
    for c in [x for x in header if x.startswith("camera_")]:
        n_missing = sum(1 for row in rows if not row.get(c, "").strip())
        frac = n_missing / max(len(rows), 1)
        if n_missing == len(rows):
            r.skip(f"present_{c}", f"{c}: column entirely empty — camera not present on this vehicle.")
        elif frac == 0:
            r.ok(f"present_{c}", f"{c}: all {len(rows)} slots filled.")
        elif frac < 0.3:
            r.warn(f"present_{c}", f"{c}: {n_missing}/{len(rows)} ({frac*100:.1f}%) slots empty "
                   "(occasional dropout).", value=frac)
        else:
            r.fail(f"present_{c}", f"{c}: {n_missing}/{len(rows)} ({frac*100:.1f}%) slots empty "
                   "— sustained topic dropout.", value=frac)

    # --- inter-group gaps ---
    ref_ns_list = [n for n in (_ns(row.get(ref, "")) for row in rows) if n is not None]
    gaps = [(ref_ns_list[i + 1] - ref_ns_list[i]) * _MS_PER_NS
            for i in range(len(ref_ns_list) - 1)] if len(ref_ns_list) >= 2 else []
    if gaps and interval:
        arr = np.array(gaps)
        n_warn = int((arr > interval * 1.5).sum())
        n_fail = int((arr > interval * 2.0).sum())
        if n_fail:
            r.fail("gaps", f"{n_fail} inter-group gap(s) > 2x interval (max={arr.max():.0f} ms).",
                   value=float(arr.max()))
        elif n_warn:
            r.warn("gaps", f"{n_warn} inter-group gap(s) > 1.5x interval (max={arr.max():.0f} ms).",
                   value=float(arr.max()))
        else:
            r.ok("gaps", f"all {len(gaps)} gaps within 1.5x interval (max={arr.max():.0f} ms).",
                 value=float(arr.max()))
    elif not gaps:
        r.skip("gaps", "too few groups to compute inter-group gaps.")

    # --- no reused timestamps per sensor ---
    for c in header:
        stamps = [row[c].strip() for row in rows if row.get(c, "").strip()]
        dupes = len(stamps) - len(set(stamps))
        if dupes:
            r.fail(f"unique_{c}", f"{c}: {dupes} timestamp(s) reused across groups.",
                   value=float(dupes))

    # --- figures ---
    if any(deltas.values()):
        r.add_figure(viz.histogram_grid(
            {k: v for k, v in deltas.items() if v}, tol,
            viz_path(figdir, "ir_dt_histograms.png"),
            "IR — per-sensor |Δt| from lidar_front", "|Δt| (ms)"))
    if gaps:
        hl = []
        if interval:
            hl = [(interval, f"interval={interval:g}ms"),
                  (interval * 2, "2x interval")]
        r.add_figure(viz.scatter_timeline(
            gaps, viz_path(figdir, "ir_gap_timeline.png"),
            "IR — inter-group gap timeline", "gap (ms)", hl))

    return r


def viz_path(figdir: str, name: str) -> str:
    import os
    return os.path.join(figdir, name)