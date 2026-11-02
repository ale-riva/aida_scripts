"""
KITTI <-> IR consistency (requires --stage1 and --config).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common import geometry as geo
from ..common import loaders
from ..common.results import SectionResult
from ..common.sensors import SensorModel


class _Sample:
    __slots__ = ("id", "frame", "cam", "ts", "is_first")

    def __init__(self, sid: str, frame: int, cam: str, ts: str, is_first: bool):
        self.id = sid; self.frame = frame; self.cam = cam; self.ts = ts
        self.is_first = is_first


def _reconstruct(rows: List[Dict[str, str]], cameras: List[str]) -> List[_Sample]:
    """Mirror the converter's interleaving to predict every KITTI sample."""
    out: List[_Sample] = []
    counter = 0
    for frame_idx, row in enumerate(rows):
        first_in_frame = True
        for cam in cameras:
            ts = row.get(cam, "").strip()
            if not ts:
                continue
            out.append(_Sample(loaders.frame_id(counter), frame_idx, cam, ts, first_in_frame))
            first_in_frame = False
            counter += 1
    return out


def _camera_K(stage1: str, sensor: SensorModel) -> Dict[str, Tuple[np.ndarray, str, dict]]:
    """{camera_key: (expected_P2_K_new_3x3, model_type, intrinsics)} from IR calibration."""
    out: Dict[str, Tuple[np.ndarray, str, dict]] = {}
    calib = loaders.load_calibration(os.path.join(stage1, "calibration.yaml"))
    intr = calib["intrinsics"]
    for ck in sensor.camera_keys:
        d = intr.get(sensor.camera_name(ck))
        if d is None:
            continue
        model = str(d.get("model_type", "PINHOLE")).upper()
        try:
            # Fully rectified dataset: P2 holds K_new for every model.
            K = geo.optimal_undistorted_K(d)
        except Exception:
            continue
        out[ck] = (K, model, d)
    return out


def _rectify_bgr(raw_bgr: np.ndarray, intr: dict) -> Optional[np.ndarray]:
    """
    Independently reproduce the converter's rectified image from a raw BGR image,
    so image_2 can be content-checked without importing pipeline code. Uses the
    same K_new as the P2 cross-check. Returns None on failure.
    """
    try:
        import cv2
        K = geo.camera_matrix(intr)
        K_new = geo.optimal_undistorted_K(intr)
        w = int(intr.get("width", raw_bgr.shape[1]))
        h = int(intr.get("height", raw_bgr.shape[0]))
        if str(intr.get("model_type", "PINHOLE")).upper() == "FISHEYE":
            D = np.array(intr.get("fisheye_dist", [0, 0, 0, 0]), dtype=np.float64).reshape(4, 1)
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
            return cv2.remap(raw_bgr, m1, m2, cv2.INTER_LINEAR)
        rd = intr.get("radial_dist", [0.0, 0.0, 0.0])
        td = intr.get("tangential_dist", [0.0, 0.0])
        D = np.array([rd[0], rd[1], td[0], td[1], rd[2]], dtype=np.float64)
        return cv2.undistort(raw_bgr, K, D, newCameraMatrix=K_new)
    except Exception:
        return None


def run(kitti_dir: str, stage1: Optional[str], sensor: Optional[SensorModel],
        cameras: Optional[List[str]], frames: List[int]) -> SectionResult:
    r = SectionResult("consistency")
    if not stage1 or sensor is None:
        r.skip("consistency", "no --stage1/--config — KITTI↔IR consistency checks skipped.")
        return r

    csv_path = os.path.join(stage1, "timestamps.csv")
    if not os.path.isfile(csv_path):
        r.fail("ir_csv", f"timestamps.csv not found in {stage1}.")
        return r
    _, rows = loaders.load_timestamps_csv(csv_path)

    try:
        selected = sensor.resolve_kitti_cameras(cameras)
    except ValueError as exc:
        r.fail("camera_selection", str(exc))
        return r
    r.ok("camera_selection", f"selected cameras: {selected}")

    expected = _reconstruct(rows, selected)
    exp_by_id = {s.id: s for s in expected}

    # --- count ---
    disk_ids = {os.path.splitext(f)[0] for f in loaders.list_dir(
        os.path.join(kitti_dir, "training", "velodyne"), ".bin")}
    if len(disk_ids) != len(expected):
        r.fail("count", f"reconstruction predicts {len(expected)} samples but velodyne/ has "
               f"{len(disk_ids)}.", value=float(len(expected)))
    else:
        r.ok("count", f"sample count matches reconstruction ({len(expected)}).",
             value=float(len(expected)))
    if disk_ids != set(exp_by_id):
        only_disk = sorted(disk_ids - set(exp_by_id))[:3]
        only_exp = sorted(set(exp_by_id) - disk_ids)[:3]
        r.fail("id_set", f"id set differs from reconstruction (disk-only {only_disk}, "
               f"expected-only {only_exp}).")

    cam_K = _camera_K(stage1, sensor)

    # --- per-id camera mapping (P2) + velodyne grouping ---
    calib_dir = os.path.join(kitti_dir, "training", "calib")
    vdir = os.path.join(kitti_dir, "training", "velodyne")
    n_map_bad = n_velo_bad = 0
    checked = 0
    for sid in sorted(disk_ids & set(exp_by_id)):
        s = exp_by_id[sid]
        checked += 1
        # camera mapping via P2
        cpath = os.path.join(calib_dir, f"{sid}.txt")
        if cam_K and os.path.isfile(cpath):
            P2 = loaders.kitti_P(loaders.load_kitti_calib(cpath), "P2")
            if P2 is not None and s.cam in cam_K:
                err = float(np.abs(P2[:, :3] - cam_K[s.cam][0]).max())
                if err > 1e-3:
                    n_map_bad += 1
                    if n_map_bad <= 2:
                        r.fail(f"camera_map_{sid}", f"id {sid}: P2 does not match predicted "
                               f"camera '{s.cam}' (max|ΔK|={err:.3f}).")
        # velodyne grouping
        bpath = os.path.join(vdir, f"{sid}.bin")
        is_link = os.path.islink(bpath)
        if s.is_first:
            if is_link:
                n_velo_bad += 1
                if n_velo_bad <= 2:
                    r.fail(f"velo_real_{sid}", f"id {sid}: first sample of frame {s.frame} should "
                           "be a real .bin but is a symlink.")
        else:
            if not is_link:
                n_velo_bad += 1
                if n_velo_bad <= 2:
                    r.fail(f"velo_link_{sid}", f"id {sid}: non-first sample of frame {s.frame} "
                           "should be a symlink but is a real file.")
            else:
                target = os.readlink(bpath)
                first_id = next(x.id for x in expected if x.frame == s.frame and x.is_first)
                if os.sep in target or (os.altsep and os.altsep in target):
                    n_velo_bad += 1
                    if n_velo_bad <= 2:
                        r.fail(f"velo_rel_{sid}", f"id {sid}: symlink target '{target}' is not a "
                               "bare relative filename.")
                elif target != f"{first_id}.bin":
                    n_velo_bad += 1
                    if n_velo_bad <= 2:
                        r.fail(f"velo_target_{sid}", f"id {sid}: symlink points to '{target}', "
                               f"expected '{first_id}.bin'.")
                elif not os.path.exists(bpath):
                    n_velo_bad += 1
                    r.fail(f"velo_dangling_{sid}", f"id {sid}: symlink target does not exist.")

    if cam_K and n_map_bad == 0:
        r.ok("camera_map", f"all {checked} ids map to the predicted camera (P2 matches).")
    elif cam_K:
        r.fail("camera_map", f"{n_map_bad}/{checked} ids map to the wrong camera.",
               value=float(n_map_bad))
    if n_velo_bad == 0:
        r.ok("velodyne_links", f"velodyne real/symlink grouping correct for all {checked} ids.")
    else:
        r.fail("velodyne_links", f"{n_velo_bad} velodyne grouping/symlink error(s).",
               value=float(n_velo_bad))

    # --- image equivalence on sampled frames ---
    # image_2 is now the RECTIFIED image for every camera, so it is no longer a
    # byte copy of the IR sample. Independently rectify the IR image and compare.
    import cv2
    img_bad = img_ok = 0
    sampled_ids = [s for s in expected if s.frame in set(frames)]
    for s in sampled_ids:
        kpath = os.path.join(kitti_dir, "training", "image_2", f"{s.id}.png")
        ipath = os.path.join(stage1, "samples", s.cam, f"{s.ts}.png")
        if not os.path.isfile(kpath) or not os.path.isfile(ipath):
            img_bad += 1
            r.fail(f"image_present_{s.id}", f"id {s.id}: missing KITTI or IR image.")
            continue
        intr = cam_K.get(s.cam, (None, None, None))[2]
        k_bgr = cv2.imread(kpath, cv2.IMREAD_COLOR)
        i_bgr = cv2.imread(ipath, cv2.IMREAD_COLOR)
        if k_bgr is None or i_bgr is None:
            img_bad += 1
            r.fail(f"image_read_{s.id}", f"id {s.id}: could not read KITTI or IR image.")
            continue
        expected_img = _rectify_bgr(i_bgr, intr) if intr else None
        if expected_img is None:
            # No intrinsics to rectify with: fall back to a dimension check.
            if k_bgr.shape == i_bgr.shape:
                img_ok += 1
            else:
                img_bad += 1
                r.fail(f"image_dims_{s.id}", f"id {s.id}: image_2 dims differ from IR.")
            continue
        if k_bgr.shape != expected_img.shape:
            img_bad += 1
            r.fail(f"image_dims_{s.id}", f"id {s.id}: image_2 dims differ from rectified IR.")
        elif float(np.abs(k_bgr.astype(np.int16) - expected_img.astype(np.int16)).mean()) <= 1.5:
            img_ok += 1
        else:
            img_bad += 1
            r.fail(f"image_match_{s.id}", f"id {s.id}: image_2 does not match the rectified IR "
                   f"image ({s.cam}) — was it left distorted?")
    if sampled_ids:
        (r.ok if img_bad == 0 else r.fail)(
            "image_match", f"{img_ok}/{img_ok + img_bad} sampled images match the rectified IR source.")
    return r