"""
Shared camera undistortion utilities.

Used by both converters/kitti_converter.py and converters/nuscenes_converter.py
so that the undistortion logic and the K_new computation always stay in sync.

Two camera models are supported, matching the intrinsics dict written by
extraction/calibration_extractor.py:

  PINHOLE : radial (k1, k2, k3) + tangential (p1, p2) distortion.
  FISHEYE : equidistant (k1, k2, k3, k4) distortion (OpenCV fisheye model).

"""

from __future__ import annotations

import numpy as np
import cv2


# Camera body -> camera optical frame rotation (identical to _AXIS_T in kitti_converter.py).
#
# URDF/body axes:  x = forward,  y = left,    z = up
# Optical axes:    x = right,    y = down,    z = into scene
#
#   x_opt = -y_body   (right   = -left)
#   y_opt = -z_body   (down    = -up)
#   z_opt =  x_body   (forward = into scene)
_CAM_BODY_TO_OPT = np.array([
    [0, -1,  0],
    [0,  0, -1],
    [1,  0,  0],
], dtype=np.float64)


def rotation_to_homogeneous(R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    return T

_AXIS_T = rotation_to_homogeneous(_CAM_BODY_TO_OPT)




def camera_matrix(intr: dict) -> np.ndarray:
    """
    Return the 3x3 camera matrix K built from an intrinsics dict.
    """
    return np.array([
        [intr.get("fx", 0.0), 0.0,                 intr.get("cx", 0.0)],
        [0.0,                 intr.get("fy", 0.0), intr.get("cy", 0.0)],
        [0.0,                 0.0,                 1.0],
    ], dtype=np.float64)


def optimal_undistorted_K(intr: dict) -> np.ndarray:
    """
    Compute the optimal new 3x3 camera matrix K_new for an undistorted image.

    Uses alpha=0 (PINHOLE) / balance=0 (FISHEYE) so that the undistorted image
    contains no black border pixels. 
    """
    if not intr:
        return np.eye(3, dtype=np.float64)

    K     = camera_matrix(intr)
    w     = int(intr.get("width",  0))
    h     = int(intr.get("height", 0))
    model = intr.get("model_type", "PINHOLE").upper()

    if model == "FISHEYE":
        fd   = intr.get("fisheye_dist", [0.0, 0.0, 0.0, 0.0])
        dist = np.array(fd, dtype=np.float64).reshape(4, 1)   # cv2.fisheye expects (4, 1)
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, (w, h), np.eye(3, dtype=np.float64), balance=0,
        )
    else:   # PINHOLE (default)
        rd   = intr.get("radial_dist",     [0.0, 0.0, 0.0])
        td   = intr.get("tangential_dist", [0.0, 0.0])
        # OpenCV distortion order: [k1, k2, p1, p2, k3]
        dist = np.array([rd[0], rd[1], td[0], td[1], rd[2]], dtype=np.float64)
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)

    return np.asarray(K_new, dtype=np.float64)


def undistort_image(img_np: np.ndarray, intr: dict) -> np.ndarray:
    """
    Undistort a uint8 BGR (or RGB) numpy image according to the lens model in "intr".
    """
    if not intr:
        return img_np

    K     = camera_matrix(intr)
    w     = int(intr.get("width",  img_np.shape[1]))
    h     = int(intr.get("height", img_np.shape[0]))
    model = intr.get("model_type", "PINHOLE").upper()

    if model == "FISHEYE":
        fd   = intr.get("fisheye_dist", [0.0, 0.0, 0.0, 0.0])
        dist = np.array(fd, dtype=np.float64).reshape(4, 1)
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, (w, h), np.eye(3, dtype=np.float64), balance=0,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), K_new, (w, h), cv2.CV_16SC2,
        )
        return cv2.remap(img_np, map1, map2, cv2.INTER_LINEAR)
    else:   # PINHOLE
        rd   = intr.get("radial_dist",     [0.0, 0.0, 0.0])
        td   = intr.get("tangential_dist", [0.0, 0.0])
        dist = np.array([rd[0], rd[1], td[0], td[1], rd[2]], dtype=np.float64)
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        return cv2.undistort(img_np, K, dist, newCameraMatrix=K_new)

def undistort_points(points_px: np.ndarray, intr: dict) -> np.ndarray:
    """
    Map raw (distorted) image pixels to the rectified image produced by
    undistort_image / optimal_undistorted_K.
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    if not intr or pts.size == 0:
        return np.asarray(points_px, dtype=np.float64).reshape(-1, 2)

    K     = camera_matrix(intr)
    K_new = optimal_undistorted_K(intr)
    model = intr.get("model_type", "PINHOLE").upper()

    if model == "FISHEYE":
        fd   = intr.get("fisheye_dist", [0.0, 0.0, 0.0, 0.0])
        dist = np.array(fd, dtype=np.float64).reshape(4, 1)
        out  = cv2.fisheye.undistortPoints(pts, K, dist, P=K_new)
    else:   # PINHOLE
        rd   = intr.get("radial_dist",     [0.0, 0.0, 0.0])
        td   = intr.get("tangential_dist", [0.0, 0.0])
        dist = np.array([rd[0], rd[1], td[0], td[1], rd[2]], dtype=np.float64)
        out  = cv2.undistortPoints(pts, K, dist, P=K_new)

    return np.asarray(out, dtype=np.float64).reshape(-1, 2)