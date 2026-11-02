"""
KITTI label_2 checks + annotation overlay.

Figure
    kitti_labels_frame_<id>.png: the 2-D box (green) and the 3-D box projected (orange).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

from ..common import geometry as geo
from ..common import loaders, viz
from ..common.results import SectionResult
from ..common.sensors import SensorModel

# Map IR frame index -> the ids whose labels we visualize. Built by consistency
# reconstruction when available; otherwise we just use the sampled ids directly.


def _project_P2(corners_cam: np.ndarray, P2: np.ndarray) -> Optional[np.ndarray]:
    """Project (8,3) optical-frame corners with a 3×4 P matrix. None if behind cam."""
    if (corners_cam[:, 2] <= 1e-6).any():
        return None
    h = np.hstack([corners_cam, np.ones((corners_cam.shape[0], 1))])
    uv = (P2 @ h.T).T
    return uv[:, :2] / uv[:, 2:3]


def _check_one(labels: List[dict], W: int, H: int, fid: str, r: SectionResult) -> None:
    for i, lb in enumerate(labels):
        if lb["type"] == "DontCare":
            continue
        issues = []
        if not (lb["h"] > 0 and lb["w"] > 0 and lb["l"] > 0):
            issues.append("non-positive h/w/l")
        if not (lb["x1"] < lb["x2"] and lb["y1"] < lb["y2"]):
            issues.append("2-D box not ordered")
        if W and H and not (-1 <= lb["x1"] and lb["x2"] <= W + 1 and
                            -1 <= lb["y1"] and lb["y2"] <= H + 1):
            issues.append("2-D box outside image")
        if abs(lb["ry"]) > math.pi + 1e-2 or abs(lb["alpha"]) > math.pi + 1e-2:
            issues.append("alpha/ry outside [-π,π]")
        if issues:
            r.warn(f"label_{fid}_{i}", f"{fid} obj {i} ({lb['type']}): " + "; ".join(issues))


def run(kitti_dir: str, frames: List[int], stage1: Optional[str],
        sensor: Optional[SensorModel], cameras: Optional[List[str]],
        figdir: str) -> SectionResult:
    r = SectionResult("labels")
    label_dir = os.path.join(kitti_dir, "training", "label_2")
    if not os.path.isdir(label_dir):
        r.fail("label_dir", "training/label_2/ missing.")
        return r

    # Are any labels populated at all?
    label_files = loaders.list_dir(label_dir, ".txt")
    nonempty = [f for f in label_files
                if os.path.getsize(os.path.join(label_dir, f)) > 0]
    if not nonempty:
        r.skip("labels_present", "all label_2 files are empty (no annotations in the IR).")
        return r
    r.ok("labels_present", f"{len(nonempty)}/{len(label_files)} label_2 files are populated.",
         value=float(len(nonempty)))

    # Map sampled IR frame indices -> KITTI ids (via the same reconstruction the
    # consistency module uses) so overlays land on the right images.
    sampled_ids: List[str] = []
    if stage1 and sensor is not None:
        try:
            from .consistency import _reconstruct
            _, rows = loaders.load_timestamps_csv(os.path.join(stage1, "timestamps.csv"))
            selected = sensor.resolve_kitti_cameras(cameras)
            for s in _reconstruct(rows, selected):
                if s.frame in set(frames):
                    sampled_ids.append(s.id)
        except Exception:
            sampled_ids = []
    if not sampled_ids:
        # Fall back to the lowest ids so schema/plausibility + overlays still run.
        all_ids = sorted(os.path.splitext(f)[0] for f in label_files)
        sampled_ids = all_ids[: max(len(frames), 3)]

    calib_dir = os.path.join(kitti_dir, "training", "calib")
    img_dir = os.path.join(kitti_dir, "training", "image_2")
    made = 0
    for sid in sampled_ids:
        lpath = os.path.join(label_dir, f"{sid}.txt")
        labels = loaders.load_kitti_labels(lpath)
        img = loaders.load_image(os.path.join(img_dir, f"{sid}.png"))
        H, W = (img.shape[0], img.shape[1]) if img is not None else (0, 0)
        _check_one(labels, W, H, sid, r)

        if img is None or not labels:
            continue
        P2 = loaders.kitti_P(loaders.load_kitti_calib(os.path.join(calib_dir, f"{sid}.txt")), "P2")
        if P2 is None:
            continue
        boxes2d, boxes3d = [], []
        for lb in labels:
            if lb["type"] == "DontCare":
                continue
            boxes2d.append((lb["x1"], lb["y1"], lb["x2"], lb["y2"]))
            corners = geo.kitti_box_corners_cam(lb["h"], lb["w"], lb["l"],
                                                lb["x"], lb["y"], lb["z"], lb["ry"])
            uv = _project_P2(corners, P2)
            if uv is not None:
                boxes3d.append(uv)
        if not boxes2d:
            continue
        p = viz.annotation_overlay(
            img, boxes2d, boxes3d, geo.BOX_EDGES,
            os.path.join(figdir, f"kitti_labels_frame_{sid}.png"),
            f"KITTI labels — id {sid}  [green=2D, orange=3D→P2]")
        if p:
            r.add_figure(p); made += 1
    if made:
        r.ok("label_overlays", f"generated {made} KITTI label overlay(s).", value=float(made))
    return r