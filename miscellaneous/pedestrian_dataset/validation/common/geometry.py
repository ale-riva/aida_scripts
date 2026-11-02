"""
Projection and coordinate-transform helpers used by the diagnostic figures and
the calibration cross-checks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Body (x=fwd, y=left, z=up) -> optical (x=right, y=down, z=fwd)
AXIS_T = np.array([
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def camera_matrix(intr: dict) -> np.ndarray:
    """3x3 pinhole matrix from an intrinsics dict (fx, fy, cx, cy)."""
    return np.array([
        [intr["fx"], 0.0,        intr["cx"]],
        [0.0,        intr["fy"], intr["cy"]],
        [0.0,        0.0,        1.0],
    ], dtype=np.float64)


def optimal_undistorted_K(intr: dict) -> np.ndarray:
    """
    New camera matrix K_new for the rectified (KITTI) image, computed the same
    way as utils.image_utils.optimal_undistorted_K so the validator's expected
    P2 matches the converter for BOTH models:
        FISHEYE  -> cv2.fisheye.estimateNewCameraMatrixForUndistortRectify (balance=0)
        PINHOLE  -> cv2.getOptimalNewCameraMatrix (alpha=0)
    """
    import cv2
    K = camera_matrix(intr)
    w = int(intr.get("width", 0))
    h = int(intr.get("height", 0))
    model = str(intr.get("model_type", "PINHOLE")).upper()
    if model == "FISHEYE":
        D = np.array(intr.get("fisheye_dist", [0, 0, 0, 0]), dtype=np.float64).reshape(4, 1)
        return np.asarray(cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0), dtype=np.float64)
    rd = intr.get("radial_dist", [0.0, 0.0, 0.0])
    td = intr.get("tangential_dist", [0.0, 0.0])
    dist = np.array([rd[0], rd[1], td[0], td[1], rd[2]], dtype=np.float64)
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    return np.asarray(K_new, dtype=np.float64)


# ---------------------------------------------------------------------------
# Rigid transforms
# ---------------------------------------------------------------------------

def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def tr_velo_to_cam(T_cam_to_ref: np.ndarray) -> np.ndarray:
    """3x4 ring_center -> camera-optical transform (KITTI Tr_velo_to_cam)."""
    return (AXIS_T @ invert_T(T_cam_to_ref))[:3, :]


def ref_to_optical(points_ref: np.ndarray, T_cam_to_ref: np.ndarray) -> np.ndarray:
    """Transform (N,3) points from ring_center to a camera's optical frame."""
    Tr = AXIS_T @ invert_T(T_cam_to_ref)            # 4x4
    n = points_ref.shape[0]
    homog = np.hstack([points_ref, np.ones((n, 1))])
    return (Tr @ homog.T).T[:, :3]


def apply_Tr(points_ref: np.ndarray, Tr_3x4: np.ndarray) -> np.ndarray:
    """Apply a KITTI 3x4 Tr matrix to (N,3) points -> (N,3) optical points."""
    n = points_ref.shape[0]
    homog = np.hstack([points_ref, np.ones((n, 1))])
    return (Tr_3x4 @ homog.T).T


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _front_mask(pts_opt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return pts_opt[:, 2] > eps


def project_pinhole(pts_opt: np.ndarray, K: np.ndarray,
                    dist: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project (N,3) optical-frame points with a pinhole model.
    dist (OpenCV order [k1,k2,p1,p2,k3]) is applied if provided. Returns (N,2).
    Points behind the camera are returned as NaN (caller should mask).
    """
    import cv2
    out = np.full((pts_opt.shape[0], 2), np.nan, dtype=np.float64)
    front = _front_mask(pts_opt)
    if not front.any():
        return out
    obj = pts_opt[front].reshape(-1, 1, 3).astype(np.float64)
    rvec = np.zeros(3); tvec = np.zeros(3)
    D = None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    uv, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    out[front] = uv.reshape(-1, 2)
    return out


def project_fisheye(pts_opt: np.ndarray, K: np.ndarray, D4: np.ndarray) -> np.ndarray:
    """Project (N,3) optical points with the OpenCV fisheye model. Returns (N,2)."""
    import cv2
    out = np.full((pts_opt.shape[0], 2), np.nan, dtype=np.float64)
    front = _front_mask(pts_opt)
    if not front.any():
        return out
    obj = pts_opt[front].reshape(-1, 1, 3).astype(np.float64)
    rvec = np.zeros((3, 1)); tvec = np.zeros((3, 1))
    D = np.asarray(D4, dtype=np.float64).reshape(4, 1)
    uv, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
    out[front] = uv.reshape(-1, 2)
    return out


def project_lidar_raw(points_ref: np.ndarray, intr: dict,
                      T_cam_to_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project ring_center points onto a RAW (distorted) IR image using the
    camera's native intrinsics + distortion model.
    Returns (uv (N,2), depth (N,)) with NaN uv for points behind the camera.
    """
    pts_opt = ref_to_optical(points_ref, T_cam_to_ref)
    depth = pts_opt[:, 2]
    K = camera_matrix(intr)
    model = str(intr.get("model_type", "PINHOLE")).upper()
    if model == "FISHEYE":
        uv = project_fisheye(pts_opt, K, intr.get("fisheye_dist", [0, 0, 0, 0]))
    else:
        radial = intr.get("radial_dist", [0, 0, 0])
        tang = intr.get("tangential_dist", [0, 0])
        dist = [radial[0], radial[1], tang[0], tang[1],
                radial[2] if len(radial) > 2 else 0.0]
        uv = project_pinhole(pts_opt, K, dist)
    return uv, depth


def project_ref_with_calib(points_ref: np.ndarray, P: np.ndarray,
                           Tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project ring_center points using an actual KITTI calib file: optical = Tr @ p,
    then uv = P @ [optical;1]. Validates P + Tr together against the (undistorted)
    KITTI image. Returns (uv (N,2), depth (N,)).
    """
    opt = apply_Tr(points_ref, Tr)                  # (N,3)
    depth = opt[:, 2]
    out = np.full((points_ref.shape[0], 2), np.nan, dtype=np.float64)
    front = depth > 1e-6
    if front.any():
        oh = np.hstack([opt[front], np.ones((front.sum(), 1))])
        img = (P @ oh.T).T                          # (M,3)
        out[front] = img[:, :2] / img[:, 2:3]
    return out, depth


# ---------------------------------------------------------------------------
# 3-D box corners
# ---------------------------------------------------------------------------

def box_corners_local(l: float, w: float, h: float) -> np.ndarray:
    """
    8 corners of a 3-D box centred at the origin, axis-aligned in a body frame
    where x=length, y=width, z=height. Returns (8,3).
    """
    x = l / 2.0; y = w / 2.0; z = h / 2.0
    return np.array([
        [ x,  y,  z], [ x, -y,  z], [-x, -y,  z], [-x,  y,  z],
        [ x,  y, -z], [ x, -y, -z], [-x, -y, -z], [-x,  y, -z],
    ], dtype=np.float64)


def yaw_matrix_z(rz: float) -> np.ndarray:
    c, s = np.cos(rz), np.sin(rz)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def lidar_box_corners_ref(position_cog: np.ndarray, l: float, w: float, h: float,
                          rz: float, T_cog_to_ref: np.ndarray) -> np.ndarray:
    """
    Build the 8 corners of a fused-LiDAR-annotation box (anchored in the cog
    frame, heading rz about z) and express them in ring_center. Returns (8,3).
    """
    corners = box_corners_local(l, w, h)
    corners = (yaw_matrix_z(rz) @ corners.T).T + position_cog
    homog = np.hstack([corners, np.ones((8, 1))])
    return (T_cog_to_ref @ homog.T).T[:, :3]


def kitti_box_corners_cam(h: float, w: float, l: float,
                          x: float, y: float, z: float, ry: float) -> np.ndarray:
    """
    8 corners of a KITTI 3-D box in the camera optical frame. KITTI location
    (x,y,z) is the bottom-face centre; ry is yaw about the optical y-axis.
    Returns (8,3).
    """
    x_c = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_c = [   0,    0,    0,    0,   -h,   -h,   -h,   -h]
    z_c = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    corners = np.array([x_c, y_c, z_c], dtype=np.float64)            # (3,8)
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    corners = R @ corners
    corners[0, :] += x; corners[1, :] += y; corners[2, :] += z
    return corners.T                                                # (8,3)


# 12 edges of the 8-corner box (index pairs), shared by both corner orderings.
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),     # top face
    (4, 5), (5, 6), (6, 7), (7, 4),     # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),     # verticals
]