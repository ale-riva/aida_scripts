"""
IR label checks (only when a labels/ directory is present).

Figures
    ir_labels_frame_<id>_<camera>.png: for each sampled frame and camera:
        green  = camera 2-D boxes
        orange = LiDAR 3-D boxes projected onto the RAW image
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common import geometry as geo
from ..common import loaders, viz
from ..common.results import SectionResult
from ..common.sensors import SensorModel

_LEGACY = ("lidar_front", "lidar_left", "lidar_right")


def _check_layout(labels_dir: str, r: SectionResult) -> bool:
    """Return True if the fused labels/lidar/ layout is present."""
    fused = os.path.join(labels_dir, "lidar")
    legacy = [d for d in _LEGACY if os.path.isdir(os.path.join(labels_dir, d))]
    if os.path.isdir(fused):
        n = len([f for f in os.listdir(fused) if f.endswith(".json")])
        r.ok("labels_layout", f"labels/lidar/ present (fused format): {n} JSON file(s).",
             value=float(n))
        if legacy:
            r.warn("labels_legacy", f"legacy per-sensor folders {legacy} present but ignored.")
        return True
    if legacy:
        r.fail("labels_layout", f"only legacy per-sensor LiDAR labels {legacy} found — the "
               "converter reads labels/lidar/ (fused), so label_2 would be empty.")
    else:
        r.warn("labels_layout", "labels/ present but no labels/lidar/ folder found.")
    return False


def _check_schema(stage1: str, rows: List[Dict[str, str]], sensor: SensorModel,
                  frames: List[int], r: SectionResult) -> None:
    labels_dir = os.path.join(stage1, "labels")
    ref = sensor.lidar_keys[0] if (sensor and sensor.lidar_keys) else "lidar_front"
    bad_lidar = 0; n_lidar = 0
    for fidx in frames:
        if fidx >= len(rows):
            continue
        ts = rows[fidx].get(ref, "").strip()
        jf = loaders.find_label_file(os.path.join(labels_dir, "lidar"), ts)
        for ann in loaders.load_labels_json(jf) if jf else []:
            n_lidar += 1
            psr = ann.get("psr", {})
            if not psr.get("position") or not psr.get("scale"):
                bad_lidar += 1
    if n_lidar:
        (r.ok if bad_lidar == 0 else r.fail)(
            "lidar_label_schema",
            f"{n_lidar - bad_lidar}/{n_lidar} sampled LiDAR annotations have psr.position+scale.")
    else:
        r.skip("lidar_label_schema", "no LiDAR annotations in sampled frames.")

    if sensor:
        bad_cam = 0; n_cam = 0
        for fidx in frames:
            if fidx >= len(rows):
                continue
            for ck in sensor.camera_keys:
                ts = rows[fidx].get(ck, "").strip()
                jf = loaders.find_label_file(os.path.join(labels_dir, ck), ts) if ts else None
                for ann in loaders.load_labels_json(jf) if jf else []:
                    n_cam += 1
                    bb = ann.get("bbox", {})
                    if "position" not in bb or "width" not in bb or "height" not in bb:
                        bad_cam += 1
        if n_cam:
            (r.ok if bad_cam == 0 else r.fail)(
                "camera_label_schema",
                f"{n_cam - bad_cam}/{n_cam} sampled camera annotations have a complete bbox.")


def _bbox_to_aabb(bb: dict) -> Optional[Tuple[float, float, float, float]]:
    pos = bb.get("position", {})
    x, y = pos.get("x"), pos.get("y")
    w, h = bb.get("width"), bb.get("height")
    if None in (x, y, w, h):
        return None
    ang = float(bb.get("rotation", 0.0))
    if abs(ang) < 1e-6:
        return (x, y, x + w, y + h)
    cx, cy = x + w / 2.0, y + h / 2.0
    a = np.deg2rad(ang)
    c, s = np.cos(a), np.sin(a)
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot = corners @ np.array([[c, s], [-s, c]]).T + [cx, cy]
    return (rot[:, 0].min(), rot[:, 1].min(), rot[:, 0].max(), rot[:, 1].max())


def _overlays(stage1: str, rows: List[Dict[str, str]], sensor: SensorModel,
              frames: List[int], figdir: str, r: SectionResult) -> None:
    if sensor is None:
        r.skip("label_overlays", "no config/sensor model — skipping annotation overlays.")
        return
    calib_path = os.path.join(stage1, "calibration.yaml")
    if not os.path.isfile(calib_path):
        return
    calib = loaders.load_calibration(calib_path)
    extr = calib["extrinsics"]; intr = calib["intrinsics"]
    labels_dir = os.path.join(stage1, "labels")
    ref = sensor.lidar_keys[0] if sensor.lidar_keys else "lidar_front"

    anchor = sensor.ego_child_frame
    T_cog_to_ref = extr.get(anchor) if anchor else None
    if T_cog_to_ref is None:
        r.warn("label_overlays", f"annotation anchor frame '{anchor}' missing from extrinsics — "
               "cannot place 3-D labels.")
        return

    made = 0
    for fidx in frames:
        if fidx >= len(rows):
            continue
        row = rows[fidx]
        lidar_ts = row.get(ref, "").strip()
        jf = loaders.find_label_file(os.path.join(labels_dir, "lidar"), lidar_ts)
        lidar_anns = loaders.load_labels_json(jf) if jf else []

        # Pre-compute 3-D box corners in ring_center for every annotation.
        boxes_ref = []
        for ann in lidar_anns:
            psr = ann.get("psr", {})
            pos = psr.get("position", {}); scale = psr.get("scale", {})
            rot = psr.get("rotation", {})
            if not pos or not scale:
                continue
            try:
                p = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
                corners = geo.lidar_box_corners_ref(
                    p, float(scale["x"]), float(scale["y"]), float(scale["z"]),
                    float(rot.get("z", 0.0)), T_cog_to_ref)
                boxes_ref.append(corners)
            except Exception:
                continue

        for ck in sensor.camera_keys:
            ts = row.get(ck, "").strip()
            if not ts:
                continue
            cname = sensor.camera_name(ck)
            cam_intr = intr.get(cname); cframe = sensor.camera_frame_id(ck)
            if cam_intr is None or cframe is None or cframe not in extr:
                continue
            img = loaders.load_image(os.path.join(stage1, "samples", ck, f"{ts}.png"))
            if img is None:
                continue

            # Project 3-D corners onto the raw image.
            boxes3d_uv = []
            for corners_ref in boxes_ref:
                uv, depth = geo.project_lidar_raw(corners_ref, cam_intr, extr[cframe])
                if np.isfinite(uv).all() and (depth > 0).all():
                    boxes3d_uv.append(uv)

            # This camera's own 2-D boxes.
            cjf = loaders.find_label_file(os.path.join(labels_dir, ck), ts)
            boxes2d = []
            for ann in loaders.load_labels_json(cjf) if cjf else []:
                aabb = _bbox_to_aabb(ann.get("bbox", {}))
                if aabb:
                    boxes2d.append(aabb)

            if not boxes2d and not boxes3d_uv:
                continue
            p = viz.annotation_overlay(
                img, boxes2d, boxes3d_uv, geo.BOX_EDGES,
                os.path.join(figdir, f"ir_labels_frame_{fidx:06d}_{ck}.png"),
                f"IR labels — {ck} ({cname}) frame {fidx:06d}  "
                f"[green=2D, orange=3D->proj]")
            if p:
                r.add_figure(p); made += 1
    if made:
        r.ok("label_overlays", f"generated {made} annotation overlay(s).", value=float(made))
    else:
        r.skip("label_overlays", "no annotation overlays generated (no visible labels in samples).")


def run(stage1: str, rows: List[Dict[str, str]], sensor: SensorModel,
        frames: List[int], figdir: str) -> SectionResult:
    r = SectionResult("labels")
    labels_dir = os.path.join(stage1, "labels")
    if not os.path.isdir(labels_dir):
        r.skip("labels_present", "no labels/ directory — annotations are optional.")
        return r
    if not _check_layout(labels_dir, r):
        return r
    _check_schema(stage1, rows, sensor, frames, r)
    _overlays(stage1, rows, sensor, frames, figdir, r)
    return r