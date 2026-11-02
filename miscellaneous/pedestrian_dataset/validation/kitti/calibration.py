"""
KITTI calibration-file checks.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common import geometry as geo
from ..common import loaders
from ..common.results import SectionResult
from ..common.sensors import SensorModel


def _expected_cameras(stage1: Optional[str], sensor: Optional[SensorModel]
                      ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Build {camera_key: (expected_K_3x3, expected_Tr_3x4)} from the IR calibration.
    Empty if IR/sensor unavailable.
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if not stage1 or sensor is None:
        return out
    calib_path = os.path.join(stage1, "calibration.yaml")
    if not os.path.isfile(calib_path):
        return out
    calib = loaders.load_calibration(calib_path)
    intr, extr = calib["intrinsics"], calib["extrinsics"]
    for ck in sensor.camera_keys:
        name = sensor.camera_name(ck)
        cframe = sensor.camera_frame_id(ck)
        d = intr.get(name)
        if d is None or cframe is None or cframe not in extr:
            continue
        try:
            # The dataset is fully rectified: P2 always holds K_new for every
            # camera model (both PINHOLE and FISHEYE).
            K = geo.optimal_undistorted_K(d)
            Tr = geo.tr_velo_to_cam(extr[cframe])
        except Exception:
            continue
        out[ck] = (K, Tr)
    return out


def _structural(calib: Dict[str, np.ndarray], fid: str, r: SectionResult) -> Optional[np.ndarray]:
    """Return the P2 3×4 matrix if structurally valid, else None."""
    sizes = {"P0": 12, "P1": 12, "P2": 12, "P3": 12, "R0_rect": 9, "Tr_velo_to_cam": 12}
    issues = []
    for key, n in sizes.items():
        v = calib.get(key)
        if v is None:
            issues.append(f"missing {key}")
        elif v.size != n:
            issues.append(f"{key} has {v.size} values (expected {n})")
    if issues:
        r.fail(f"calib_keys_{fid}", f"{fid}: " + "; ".join(issues))
        return None

    for key in ("P0", "P1", "P3"):
        if not np.allclose(calib[key], 0.0):
            r.fail(f"p_zeros_{fid}", f"{fid}: {key} is not zero-filled.")
            break
    else:
        r.ok(f"p_zeros_{fid}", f"{fid}: P0/P1/P3 zero-filled, P2 holds the active camera.")

    if not np.allclose(calib["R0_rect"].reshape(3, 3), np.eye(3), atol=1e-6):
        r.fail(f"r0_rect_{fid}", f"{fid}: R0_rect is not the identity.")
    else:
        r.ok(f"r0_rect_{fid}", f"{fid}: R0_rect = identity.")

    P2 = calib["P2"].reshape(3, 4)
    if not np.allclose(P2[:, 3], 0.0, atol=1e-6):
        r.warn(f"p2_trans_{fid}", f"{fid}: P2 translation column is non-zero "
               f"({P2[:, 3].round(3).tolist()}).")
    return P2


def _cross_check(P2: np.ndarray, Tr: Optional[np.ndarray], fid: str,
                 expected: Dict[str, Tuple[np.ndarray, np.ndarray]], r: SectionResult) -> None:
    if not expected:
        return
    K = P2[:, :3]
    # Identify the camera whose K best matches P2's 3×3 block.
    best_key, best_err = None, None
    for ck, (Kc, _) in expected.items():
        err = float(np.abs(K - Kc).max())
        if best_err is None or err < best_err:
            best_key, best_err = ck, err
    if best_key is None:
        return
    if best_err <= 1e-3:
        r.ok(f"p2_match_{fid}", f"{fid}: P2 matches IR camera '{best_key}' "
             f"(max|ΔK|={best_err:.2e}).")
    else:
        r.fail(f"p2_match_{fid}", f"{fid}: P2 does not match any IR camera intrinsics "
               f"(closest '{best_key}', max|ΔK|={best_err:.3f}).")
        return

    if Tr is not None:
        Tr_exp = expected[best_key][1]
        err = float(np.abs(Tr - Tr_exp).max())
        if err <= 1e-4:
            r.ok(f"tr_match_{fid}", f"{fid}: Tr_velo_to_cam matches recomputed transform for "
                 f"'{best_key}' (max|Δ|={err:.2e}).")
        else:
            r.fail(f"tr_match_{fid}", f"{fid}: Tr_velo_to_cam differs from recomputed "
                   f"AXIS_T·inv(T_cam_to_ref) for '{best_key}' (max|Δ|={err:.3f}).")


def run(kitti_dir: str, frames: List[int], stage1: Optional[str],
        sensor: Optional[SensorModel]) -> SectionResult:
    r = SectionResult("calibration")
    calib_dir = os.path.join(kitti_dir, "training", "calib")
    if not os.path.isdir(calib_dir):
        r.fail("calib_dir", "training/calib/ missing.")
        return r

    expected = _expected_cameras(stage1, sensor)
    if not expected:
        r.skip("cross_check", "no IR calibration/config — P2↔camera and Tr cross-checks skipped.")

    # Sample calib ids spread across the dataset rather than frame-index ids:
    # with N interleaved cameras, ids frame*N would all land on the same camera,
    # so P2/Tr would only ever be checked for one camera. An even spread over all
    # ids hits every camera in the interleave.
    all_ids = sorted(os.path.splitext(f)[0] for f in loaders.list_dir(calib_dir, ".txt"))
    if not all_ids:
        r.fail("calib_files", "no calib .txt files found.")
        return r
    n_cams = (sensor.num_cameras if sensor and sensor.num_cameras else 7)
    k = min(len(all_ids), max(len(frames), n_cams))
    if k >= len(all_ids):
        sample_ids = all_ids
    else:
        sample_ids = [all_ids[round(i * (len(all_ids) - 1) / (k - 1))] for i in range(k)]

    for fid in sample_ids:
        path = os.path.join(calib_dir, f"{fid}.txt")
        if not os.path.isfile(path):
            r.fail(f"calib_file_{fid}", f"calib file missing: {fid}.txt")
            continue
        calib = loaders.load_kitti_calib(path)
        P2 = _structural(calib, fid, r)
        if P2 is not None and expected:
            Tr = loaders.kitti_Tr(calib)
            _cross_check(P2, Tr, fid, expected, r)
    return r