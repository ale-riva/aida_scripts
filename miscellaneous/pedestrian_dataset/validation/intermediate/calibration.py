"""
IR calibration & transform checks, plus the LiDAR→image projection diagnostic.

Figures
    ir_projection_frame_<id>_<camera>.png: merged LiDAR cloud projected onto
        each camera's RAW image for the sampled frames.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..common import geometry as geo
from ..common import loaders, viz
from ..common.results import SectionResult
from ..common.sensors import SensorModel

_VALID_MODELS = {"PINHOLE", "FISHEYE"}


# ---------------------------------------------------------------------------
# Intrinsics / extrinsics validity
# ---------------------------------------------------------------------------

def _check_intrinsics(calib: dict, r: SectionResult) -> None:
    intr = calib.get("intrinsics", {})
    if not intr:
        r.fail("intrinsics_present", "calibration.yaml has no intrinsics.")
        return
    for name, d in intr.items():
        issues = []
        for k in ("fx", "fy", "cx", "cy", "width", "height"):
            if k not in d:
                issues.append(f"missing {k}")
        if not issues:
            if d["fx"] <= 0 or d["fy"] <= 0:
                issues.append("fx/fy not positive")
            if not (0 <= d["cx"] <= d["width"]) or not (0 <= d["cy"] <= d["height"]):
                issues.append("principal point outside frame")
            model = str(d.get("model_type", "PINHOLE")).upper()
            if model not in _VALID_MODELS:
                issues.append(f"unknown model_type '{model}'")
            elif model == "FISHEYE" and "fisheye_dist" not in d:
                issues.append("FISHEYE without fisheye_dist")
        if issues:
            r.fail(f"intrinsics_{name}", f"{name}: " + "; ".join(issues))
        else:
            r.ok(f"intrinsics_{name}",
                 f"{name}: {d['width']}×{d['height']}, model={d.get('model_type','PINHOLE')}.")


def _is_rigid(T: np.ndarray) -> Optional[str]:
    """Return None if T is a valid 4×4 rigid transform, else a reason string."""
    if T.shape != (4, 4):
        return f"shape {T.shape} != (4,4)"
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
        return "rotation not orthonormal"
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
        return f"det(R)={np.linalg.det(R):.4f} (not a proper rotation)"
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-6):
        return "bottom row != [0,0,0,1]"
    return None


def _check_extrinsics(calib: dict, sensor: SensorModel, r: SectionResult) -> None:
    extr = calib.get("extrinsics", {})
    if not extr:
        r.fail("extrinsics_present", "calibration.yaml has no extrinsics.")
        return

    for frame, T in extr.items():
        reason = _is_rigid(np.asarray(T, dtype=np.float64))
        if reason:
            r.fail(f"extrinsic_{frame}", f"{frame}: {reason}.")
        else:
            r.ok(f"extrinsic_{frame}", f"{frame}: valid rigid transform.")

    # reference_frame
    ref = calib.get("reference_frame")
    if sensor and sensor.reference_frame and ref and ref != sensor.reference_frame:
        r.warn("reference_frame", f"calibration reference_frame='{ref}' differs from "
               f"config '{sensor.reference_frame}'.")
    elif ref:
        r.ok("reference_frame", f"reference_frame = '{ref}'.")

    # Required frames for downstream conversion/labels.
    if sensor:
        needed = []
        for lk in sensor.lidar_keys:
            needed.append(sensor.lidar_frame_id(lk) or lk)
        for ck in sensor.camera_keys:
            fid = sensor.camera_frame_id(ck)
            if fid:
                needed.append(fid)
        if sensor.ego_child_frame:
            needed.append(sensor.ego_child_frame)
        for fr in needed:
            if fr not in extr:
                sev = r.fail if (fr == sensor.ego_child_frame or "lidar" in fr) else r.warn
                sev(f"frame_required_{fr}",
                    f"frame '{fr}' is required downstream but missing from extrinsics.")
        if sensor.ego_child_frame and sensor.ego_child_frame in extr:
            r.ok("anchor_frame",
                 f"annotation anchor frame '{sensor.ego_child_frame}' present (labels can be placed).")


# ---------------------------------------------------------------------------
# Projection diagnostic
# ---------------------------------------------------------------------------

def _merged_cloud_ref(stage1: str, row: Dict[str, str], sensor: SensorModel,
                      extr: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Read every LiDAR .pcd for this row, transform to ring_center, merge -> (N,3)."""
    if sensor is None:
        return None
    parts = []
    for lk in sensor.lidar_keys:
        ts = row.get(lk, "").strip()
        if not ts:
            continue
        pcd = os.path.join(stage1, "samples", lk, f"{ts}.pcd")
        xyzi = loaders.read_pcd_xyzi(pcd)
        if xyzi is None or len(xyzi) == 0:
            continue
        frame = sensor.lidar_frame_id(lk) or lk
        T = extr.get(frame)
        xyz = xyzi[:, :3].astype(np.float64)
        if T is not None:
            h = np.hstack([xyz, np.ones((len(xyz), 1))])
            xyz = (T @ h.T).T[:, :3]
        parts.append(xyz)
    if not parts:
        return None
    return np.vstack(parts)


def _projection_figures(stage1: str, rows: List[Dict[str, str]], sensor: SensorModel,
                        calib: dict, frames: List[int], figdir: str, r: SectionResult) -> None:
    if sensor is None:
        r.skip("projection_figures", "no config/sensor model — skipping projection overlays.")
        return
    extr = calib.get("extrinsics", {})
    intr = calib.get("intrinsics", {})
    made = 0
    for fidx in frames:
        if fidx >= len(rows):
            continue
        row = rows[fidx]
        cloud = _merged_cloud_ref(stage1, row, sensor, extr)
        if cloud is None or len(cloud) == 0:
            continue
        for ck in sensor.camera_keys:
            ts = row.get(ck, "").strip()
            if not ts:
                continue
            cname = sensor.camera_name(ck)
            cam_intr = intr.get(cname)
            cframe = sensor.camera_frame_id(ck)
            if cam_intr is None or cframe is None or cframe not in extr:
                continue
            img = loaders.load_image(os.path.join(stage1, "samples", ck, f"{ts}.png"))
            if img is None:
                continue
            uv, depth = geo.project_lidar_raw(cloud, cam_intr, extr[cframe])
            p = viz.lidar_projection_overlay(
                img, uv, depth,
                os.path.join(figdir, f"ir_projection_frame_{fidx:06d}_{ck}.png"),
                f"IR LiDAR→{ck} ({cname}) frame {fidx:06d}")
            if p:
                r.add_figure(p); made += 1
    if made:
        r.ok("projection_figures", f"generated {made} LiDAR→image projection overlay(s).",
             value=float(made))
    else:
        r.skip("projection_figures", "no projection overlays could be generated "
               "(missing images, clouds, or extrinsics).")


def run(stage1: str, rows: List[Dict[str, str]], sensor: SensorModel,
        frames: List[int], figdir: str) -> SectionResult:
    r = SectionResult("calibration")
    calib_path = os.path.join(stage1, "calibration.yaml")
    if not os.path.isfile(calib_path):
        r.fail("calibration_present", "calibration.yaml missing.")
        return r
    calib = loaders.load_calibration(calib_path)
    _check_intrinsics(calib, r)
    _check_extrinsics(calib, sensor, r)
    _projection_figures(stage1, rows, sensor, calib, frames, figdir, r)
    return r