"""
KITTI label converter: Stage-1 JSON annotations -> KITTI label_2 .txt format.

Coordinate-frame pipeline
LiDAR annotations come from the fused (merged) point cloud, stored under
labels/lidar/, and are expressed relative to the center of gravity of the
fused LiDAR system, using the convention:
    fused axes:  x = forward,  y = left,  z = up
"rotation.z" is the heading yaw (z is the vertical axis in this convention)
and "scale.x/y/z" = length/width/height. The fused frame shares its origin
*and* axis convention with the calibration reference_frame (ring_center /
center_of_gravity) - i.e. it is the same body frame used everywhere else in
this pipeline (x=fwd, y=left, z=up); no axis flip is needed or applied.

The reference camera for each 3-D KITTI label is the camera of the sample
being written (KITTI is one image = one camera): the 3-D box is expressed in
that camera's optical frame and its 2-D box is sourced from that camera's
annotations.

All missing / unrepresented KITTI fields:
    truncated -> 0.0  (not annotated)
    occluded  -> 0    (not annotated)
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple
# from .kitti_converter import _build_projection_matrix as _bpm

import numpy as np

from utils.image_utils import undistort_points as _undistort_points

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Camera body -> camera optical frame rotation (same as _AXIS_T in kitti_converter.py).
# Body:    x = forward, y = left,  z = up
# Optical: x = right,   y = down,  z = into scene
_R_BODY_TO_OPT = np.array([
    [0, -1,  0],
    [0,  0, -1],
    [1,  0,  0],
], dtype=np.float64)

# Fallback fused-LiDAR-annotation-frame -> ring_center transform, used only
# when calibration.yaml has no extrinsic for the annotation anchor frame
# (see write_kitti_labels_for_sample). The fused frame and ring_center share
# the same axis convention (x=fwd, y=left, z=up) but NOT necessarily the
# same origin: center_of_gravity and ring_center are distinct, physically
# separated points on the vehicle (confirmed via the URDF for at least the
# EVA rig - a fixed rigid-body translation of several tens of cm, no
# rotation). This identity matrix is only the degraded fallback for when
# that real, vehicle-specific transform can't be resolved; it is NOT the
# primary source of truth.
_T_LIDAR_TO_REF = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> list:
    """Load a JSON annotation file. Returns an empty list if the file is absent."""
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        logger.warning("Annotation file does not contain a list: %s", path)
        return []
    return data


def _find_label_file(labels_dir: str, sensor_key: str, stem: str) -> str:
    """Return the annotation JSON path for a given sensor key and file stem."""
    return os.path.join(labels_dir, sensor_key, f"{stem}.json")


# ---------------------------------------------------------------------------
# 2-D camera label conversion
# ---------------------------------------------------------------------------

def _axis_aligned_box(
    x_tl: float, y_tl: float,
    w: float, h: float,
    angle_deg: float,
    intr: Optional[dict] = None,
) -> Tuple[float, float, float, float]:
    """
    Convert a (possibly rotated) 2-D bounding box to an axis-aligned KITTI box,
    expressed in the rectified (undistorted) image space when intrinsics are
    given.

    Parameters
    x_tl, y_tl : top-left corner of the box before rotation (pixels, raw image)
    w, h        : box width and height (pixels)
    angle_deg   : clockwise rotation of the box around its centre (degrees)
    intr        : camera intrinsics dict. When provided, the box outline is
                  mapped from the raw image into the rectified image (the space
                  image_2/ and P2 live in) before the axis-aligned rectangle is
                  taken. When None, the box is returned in raw pixels unchanged.

    Returns (x1, y1, x2, y2): axis-aligned bounding rectangle.
    """
    cx = x_tl + w / 2.0
    cy = y_tl + h / 2.0
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Four corners relative to centre
    half_w = w / 2.0
    half_h = h / 2.0
    corners_local = [
        (-half_w, -half_h),
        ( half_w, -half_h),
        ( half_w,  half_h),
        (-half_w,  half_h),
    ]

    corners = []
    for dx, dy in corners_local:
        rx = cos_a * dx - sin_a * dy + cx
        ry = sin_a * dx + cos_a * dy + cy
        corners.append((rx, ry))

    if not intr:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    # Rectified space: undistortion is non-linear, so the straight box edges in
    # the raw image become curved. Sample several points along each edge before
    # mapping, then take the axis-aligned rectangle of the undistorted points.
    n_per_edge = 8
    boundary = []
    for i in range(4):
        x0, y0 = corners[i]
        x1c, y1c = corners[(i + 1) % 4]
        for t in np.linspace(0.0, 1.0, n_per_edge, endpoint=False):
            boundary.append((x0 + t * (x1c - x0), y0 + t * (y1c - y0)))

    und = _undistort_points(np.array(boundary, dtype=np.float64), intr)
    x1, y1 = float(und[:, 0].min()), float(und[:, 1].min())
    x2, y2 = float(und[:, 0].max()), float(und[:, 1].max())

    # Clamp to the rectified image bounds (same convention as the projection
    # fallback in _project_box3d_to_image).
    img_w = float(intr.get("width", 0))
    img_h = float(intr.get("height", 0))
    if img_w > 0 and img_h > 0:
        x1 = min(max(0.0, x1), img_w)
        x2 = min(max(0.0, x2), img_w)
        y1 = min(max(0.0, y1), img_h)
        y2 = min(max(0.0, y2), img_h)

    return x1, y1, x2, y2


def _build_2d_box_from_camera_ann(
    ann: dict, intr: Optional[dict] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract (x1, y1, x2, y2) from a camera annotation dict, mapped into the
    rectified image space when intrinsics are provided.
    Returns None if required fields are missing.
    """
    bbox = ann.get("bbox", {})
    pos = bbox.get("position", {})
    x_tl = pos.get("x")
    y_tl = pos.get("y")
    w    = bbox.get("width")
    h    = bbox.get("height")
    rot  = bbox.get("rotation", 0.0)

    if any(v is None for v in (x_tl, y_tl, w, h)):
        logger.warning("Camera annotation missing position/size fields: %s", ann)
        return None

    # Skip empty-scene camera annotations initialised to zero by the annotator.
    if float(w) == 0.0 and float(h) == 0.0:
        return None

    return _axis_aligned_box(float(x_tl), float(y_tl), float(w), float(h), float(rot), intr)


# ---------------------------------------------------------------------------
# 3-D LiDAR label conversion
# ---------------------------------------------------------------------------

def _apply_extrinsic(
    pos_local: np.ndarray,
    T_local_to_ref: np.ndarray,
) -> np.ndarray:
    """
    Transform a point from a local frame into the reference (ring_center) frame.

    Parameters
    pos_local      : (3,) array in the local frame (e.g. the fused LiDAR frame)
    T_local_to_ref : (4, 4) transform mapping the local frame into ring_center

    Returns (3,) position in ring_center frame (x=fwd, y=left, z=up).
    """
    p_h = np.array([pos_local[0], pos_local[1], pos_local[2], 1.0], dtype=np.float64)
    p_ref = (T_local_to_ref @ p_h)[:3]
    return p_ref


def _ref_position_to_optical(
    pos_ref: np.ndarray,
    T_cam_to_ref: np.ndarray,
) -> np.ndarray:
    """
    Transform a position from ring_center (body) frame to the camera optical frame.

    Uses the same formula as _build_tr_velo_to_cam in kitti_converter.py:
        Tr = (_AXIS_T @ inv(T_cam_to_ref))[:3, :]
        p_opt = Tr @ [p_ref; 1]

    Parameters
    pos_ref      : (3,) position in ring_center frame
    T_cam_to_ref : (4, 4) extrinsic of the camera (T_cam_body -> ring_center)

    Returns (3,) position in camera optical frame (x=right, y=down, z=fwd).
    """
    _AXIS_T_4x4 = np.array([
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [1,  0,  0,  0],
        [0,  0,  0,  1],
    ], dtype=np.float64)
    Tr = (_AXIS_T_4x4 @ np.linalg.inv(T_cam_to_ref))[:3, :]   # (3, 4)
    p_h = np.append(pos_ref, 1.0)
    return Tr @ p_h                                              # (3,)


def _yaw_to_optical_ry(
    rz_lidar: float,
    T_lidar_to_ref: np.ndarray,
    T_cam_to_ref: np.ndarray,
) -> float:
    """
    Convert the object heading angle to the KITTI ry convention, for an
    annotation expressed in the fused LiDAR frame (x=fwd, y=left, z=up).

    The fused frame's vertical axis is z, so the heading is stored as
    rotation.z (a rotation about the fused frame's vertical axis). KITTI ry is
    the heading measured in the specific camera's optical frame (rotation
    about the optical y-axis, relative to the optical z viewing axis), so it
    must be computed per camera.

    Steps:
      1. Build the object's heading (length-axis) unit vector in the fused
         frame: L = (cos rz, sin rz, 0).
      2. Rotate it into ring_center:           L_ref = R_lidar_to_ref @ L.
      3. Rotate it into the camera optical frame using the camera's ACTUAL
         ref->optical rotation (the rotation block of the same transform the
         position pipeline uses, Tr = _AXIS_T @ inv(T_cam_to_ref)):
             R_ref_to_opt = _R_BODY_TO_OPT @ R_cam_to_ref^T
             L_opt = R_ref_to_opt @ L_ref.
      4. Extract ry so the KITTI box length axis (cos ry, 0, -sin ry) aligns
         with L_opt:  ry = atan2(-L_opt_z, L_opt_x).

    Parameters
    T_lidar_to_ref : (4, 4) fused-LiDAR-annotation-frame -> ring_center
                     transform (resolved from calibration.yaml; see
                     write_kitti_labels_for_sample). Only its rotation block
                     is used here, since rotating a heading doesn't depend
                     on the translation between the two frames' origins.
    T_cam_to_ref   : (4, 4) extrinsic of the sample's camera
                     (T_cam_body -> ring_center). Its rotation block makes the
                     heading per-camera correct, matching the position pipeline
                     in _ref_position_to_optical.

    Returns ry in [-pi, pi].
    """
    # Step 1: object heading (length-axis) direction in the fused frame.
    L_lidar = np.array([math.cos(rz_lidar), math.sin(rz_lidar), 0.0], dtype=np.float64)

    # Step 2: rotate into ring_center body frame.
    R_l2r = T_lidar_to_ref[:3, :3]
    L_ref = R_l2r @ L_lidar

    # Step 3: rotate into this camera's optical frame, using the same rotation
    # the position pipeline applies: (_AXIS_T @ inv(T_cam_to_ref))[:3, :3],
    # whose rotation block is _R_BODY_TO_OPT @ R_cam_to_ref^T.
    R_ref_to_opt = _R_BODY_TO_OPT @ T_cam_to_ref[:3, :3].T
    L_opt = R_ref_to_opt @ L_ref

    # Step 4: heading of the length axis about the optical y-axis.
    ry = math.atan2(-L_opt[2], L_opt[0])
    return float(ry)


def _compute_alpha(ry: float, x_opt: float, z_opt: float) -> float:
    """
    Compute the KITTI observation angle alpha.

    alpha = ry - arctan2(x_cam, z_cam)
    Clamped to [-pi, pi].
    """
    alpha = ry - math.atan2(x_opt, z_opt)
    # Normalise to [-pi, pi]
    while alpha > math.pi:
        alpha -= 2 * math.pi
    while alpha < -math.pi:
        alpha += 2 * math.pi
    return float(alpha)


# ---------------------------------------------------------------------------
# 3-D box projection (fallback 2-D box when camera annotation is absent)
# ---------------------------------------------------------------------------

def _build_box3d_corners_in_cam(
    pos_opt_bottom: np.ndarray,
    h: float, w: float, l: float,
    ry: float,
) -> np.ndarray:
    """
    Build the 8 corners of the 3-D bounding box in the camera optical frame.

    KITTI convention: pos_opt_bottom is the bottom-centre (ground-level, centre
    in x/z).  The box extends upward by h (negative optical-y direction).

    Returns (8, 3) float64 array.
    """
    x0, y0, z0 = pos_opt_bottom
    half_w = w / 2.0
    half_l = l / 2.0

    # Corners in object-local frame (bottom face y=0, top face y=-h)
    corners_local = np.array([
        [ half_w,   0,  half_l],
        [-half_w,   0,  half_l],
        [-half_w,   0, -half_l],
        [ half_w,   0, -half_l],
        [ half_w,  -h,  half_l],
        [-half_w,  -h,  half_l],
        [-half_w,  -h, -half_l],
        [ half_w,  -h, -half_l],
    ], dtype=np.float64)

    # Rotation about camera y-axis by ry
    cos_ry, sin_ry = math.cos(ry), math.sin(ry)
    R_ry = np.array([
        [ cos_ry, 0, sin_ry],
        [      0, 1,      0],
        [-sin_ry, 0, cos_ry],
    ], dtype=np.float64)

    corners_cam = (R_ry @ corners_local.T).T + np.array([x0, y0, z0])
    return corners_cam   # (8, 3)


def _project_box3d_to_image(
    pos_opt_bottom: np.ndarray,
    h: float, w: float, l: float,
    ry: float,
    P: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Project the 8 corners of the 3-D box onto the reference camera image and
    return the axis-aligned bounding rectangle of the in-image corners.

    Returns None if all corners project behind the camera (z <= 0) or if
    no corner falls inside the image bounds.

    Parameters
    P     : (3, 4) KITTI projection matrix for the reference camera
    img_w : image width in pixels (for clipping, 0 = no clipping)
    img_h : image height in pixels (for clipping, 0 = no clipping)
    """
    corners_cam = _build_box3d_corners_in_cam(pos_opt_bottom, h, w, l, ry)

    # Keep only corners in front of the camera
    in_front = corners_cam[:, 2] > 0.1
    if not np.any(in_front):
        return None

    pts = corners_cam[in_front]   # (K, 3)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])          # (K, 4)
    img_pts = (P @ pts_h.T).T               # (K, 3)
    u = img_pts[:, 0] / img_pts[:, 2]
    v = img_pts[:, 1] / img_pts[:, 2]

    x1, y1 = float(u.min()), float(v.min())
    x2, y2 = float(u.max()), float(v.max())

    # If image dimensions provided, clip to image bounds
    if img_w > 0 and img_h > 0:
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(img_w), x2)
        y2 = min(float(img_h), y2)
        if x2 <= x1 or y2 <= y1:
            return None   # box entirely outside image

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Per-frame label assembly
# ---------------------------------------------------------------------------

def _kitti_line_from_lidar_ann(
    ann: dict,
    T_lidar_to_ref: np.ndarray,
    T_cam0_to_ref: np.ndarray,
    box2d: Optional[Tuple[float, float, float, float]],
    P_cam0: Optional[np.ndarray] = None,
    img_w: int = 0,
    img_h: int = 0,
) -> Optional[str]:
    """
    Convert one fused-LiDAR annotation dict (labels/lidar/) to a KITTI label
    line string.

    The annotation is expressed in the fused LiDAR frame (x=fwd, y=left,
    z=up; origin = center of gravity of the fused LiDAR system):
      * "rotation.z" is the heading yaw (z is the vertical axis in this frame).
      * "scale.x/y/z" = length/width/height (x is forward, z is up here).

    Parameters
    ann             : one entry from the fused LiDAR JSON
    T_lidar_to_ref  : (4,4) fused-LiDAR-annotation-frame -> ring_center
                      transform, resolved from calibration.yaml's extrinsic
                      for the annotation anchor frame (see
                      write_kitti_labels_for_sample); falls back to
                      _T_LIDAR_TO_REF (identity) if unresolved.
    T_cam0_to_ref   : (4,4) extrinsic for the KITTI reference camera
    box2d           : (x1, y1, x2, y2) from the matched camera annotation, or None.
                      When None and P_cam0 is provided, the 2-D box is derived by
                      projecting the 3-D box corners onto the reference camera image.
    P_cam0          : (3, 4) KITTI projection matrix for the reference camera.
                      Used only when box2d is None.
    img_w, img_h    : reference camera image dimensions for projection clipping.

    Returns a KITTI label line, or None if essential data is missing.
    """
    obj_type = ann.get("obj_type", "DontCare")
    psr      = ann.get("psr", {})
    pos      = psr.get("position", {})
    rot      = psr.get("rotation", {})
    scale    = psr.get("scale", {})

    # Position in fused frame (x=fwd, y=left, z=up)
    try:
        px = float(pos["x"])
        py = float(pos["y"])
        pz = float(pos["z"])
    except (KeyError, TypeError, ValueError):
        logger.warning("LiDAR annotation missing position fields: %s", ann)
        return None

    # Scale: x=length, y=width, z=height (metres)
    try:
        l = float(scale["x"])
        w = float(scale["y"])
        h = float(scale["z"])
    except (KeyError, TypeError, ValueError):
        logger.warning("LiDAR annotation missing scale fields: %s", ann)
        return None

    # Skip empty-scene annotations: when the annotator initialises a frame
    # with no object present, all fields (including scale) are set to 0.
    # A box with zero dimensions is physically meaningless and must be dropped.
    if h == 0.0 and w == 0.0 and l == 0.0:
        return None

    # Heading: rotation about the fused frame's z-axis (= vehicle vertical = up).
    rz = float(rot.get("z", 0.0))

    # --- 3-D position pipeline ---
    pos_lidar = np.array([px, py, pz], dtype=np.float64)

    # Step 1: fused frame -> ring_center (body frame, x=fwd, y=left, z=up)
    pos_ref = _apply_extrinsic(pos_lidar, T_lidar_to_ref)

    # Step 2: ring_center -> camera-0 optical frame
    pos_opt = _ref_position_to_optical(pos_ref, T_cam0_to_ref)

    # Step 3: centroid -> KITTI bottom-centre (shift down by h/2 along optical y)
    pos_opt_bottom = pos_opt.copy()
    pos_opt_bottom[1] += h / 2.0

    x_cam, y_cam, z_cam = pos_opt_bottom

    # --- Orientation ---
    ry    = _yaw_to_optical_ry(rz, T_lidar_to_ref, T_cam0_to_ref)
    alpha = _compute_alpha(ry, pos_opt[0], pos_opt[2])   # use centroid for alpha

    # --- 2-D box ---
    # Priority:
    #   1. Annotated box from the reference camera (most accurate).
    #   2. Projection of the 3-D box onto the reference camera image (fallback
    #      when the object is not annotated in the reference camera but is still
    #      physically visible there — this is the KITTI-correct behaviour).
    #   3. Sentinel (-1, -1, -1, -1) only when the object is behind the camera
    #      or entirely outside the image.
    if box2d is not None:
        x1, y1, x2, y2 = box2d
    elif P_cam0 is not None:
        projected = _project_box3d_to_image(
            pos_opt_bottom, h, w, l, ry, P_cam0, img_w, img_h
        )
        if projected is not None:
            x1, y1, x2, y2 = projected
        else:
            x1 = y1 = x2 = y2 = -1.0
    else:
        x1 = y1 = x2 = y2 = -1.0

    # KITTI format:
    # type  trunc  occ  alpha  x1 y1 x2 y2  h w l  x y z  ry
    line = (
        f"{obj_type} "
        f"0.00 0 "
        f"{alpha:.4f} "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{h:.4f} {w:.4f} {l:.4f} "
        f"{x_cam:.4f} {y_cam:.4f} {z_cam:.4f} "
        f"{ry:.4f}"
    )
    return line


def _build_cam0_projection(
    cam0_intrinsics: Optional[dict],
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Build the (3, 4) KITTI projection matrix for the reference camera, used
    as the 3-D box projection fallback when no camera annotation is present.

    Uses the rectified intrinsics K_new for every camera model (PINHOLE and
    FISHEYE), matching the rectified image_2/ and P2, so the fallback 2-D box
    lands in the same space as the rectified image. Returns (P_cam0, img_w,
    img_h); P_cam0 is None when cam0_intrinsics is falsy or required fields are
    missing.
    """
    if not cam0_intrinsics:
        return None, 0, 0

    from utils.image_utils import optimal_undistorted_K as _ouk
    K_new = _ouk(cam0_intrinsics)
    P_cam0 = np.hstack([K_new, np.zeros((3, 1), dtype=np.float64)])
    img_w = int(cam0_intrinsics.get("width",  0))
    img_h = int(cam0_intrinsics.get("height", 0))
    return P_cam0, img_w, img_h


def _build_obj_id_to_box2d(
    camera_annotations_by_cam: Dict[str, List[dict]],
    ref_cam_key: Optional[str],
    ref_cam_intrinsics: Optional[dict] = None,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Build a 2-D box lookup from the reference camera annotations ONLY.
    The 2-D box in a KITTI label must be expressed in the reference camera
    image space.  Using annotations from any other camera would place the
    box in a different image coordinate system.

    ref_cam_intrinsics, when given, maps each box from the raw image into the
    rectified image space (matching image_2/ and P2).
    """
    obj_id_to_box2d: Dict[str, Tuple[float, float, float, float]] = {}
    ref_anns = camera_annotations_by_cam.get(ref_cam_key, []) if ref_cam_key else []
    for ann in ref_anns:
        obj_id = str(ann.get("obj_id", ""))
        if obj_id and obj_id not in obj_id_to_box2d:
            box = _build_2d_box_from_camera_ann(ann, ref_cam_intrinsics)
            if box is not None:
                obj_id_to_box2d[obj_id] = box
    return obj_id_to_box2d


def build_kitti_label_lines(
    lidar_annotations: List[dict],
    camera_annotations_by_cam: Dict[str, List[dict]],
    extrinsics: dict,
    cam0_frame_id: str,
    ref_cam_key: Optional[str] = None,
    cam0_intrinsics: Optional[dict] = None,
    T_lidar_to_ref: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Build all KITTI label lines for one sample, from the fused LiDAR
    annotation format (labels/lidar/, expressed in the fused-LiDAR frame
    x=fwd, y=left, z=up, anchored at the center of gravity of the fused
    LiDAR system).

    Parameters
    lidar_annotations        : list of annotation dicts from the fused LiDAR JSON
    camera_annotations_by_cam: {cam_key -> list of annotation dicts} for all cameras
    extrinsics                : calibration.yaml extrinsics dict {frame_id -> [[4x4]]}
    cam0_frame_id             : URDF frame_id of the reference camera (most forward-facing)
    ref_cam_key               : pipeline key (e.g. "camera_1") of the reference camera;
                                the 2-D box is sourced ONLY from this camera's annotations.
    cam0_intrinsics           : intrinsics dict for the reference camera (from calibration.yaml).
                                Used to build P_cam0 for 3-D box projection fallback.
    T_lidar_to_ref            : (4,4) fused-LiDAR-annotation-frame -> ring_center
                                transform. Pass the calibration.yaml extrinsic for
                                the annotation anchor frame (e.g. center_of_gravity)
                                here - see write_kitti_labels_for_sample, which
                                resolves it from PipelineConfig.ego_pose.child_frame.
                                Defaults to _T_LIDAR_TO_REF (identity) if not given,
                                which is only correct if that frame is physically
                                coincident with ring_center.

    Returns a list of KITTI label line strings (one per object).
    """
    if T_lidar_to_ref is None:
        T_lidar_to_ref = _T_LIDAR_TO_REF

    T_cam0_raw = extrinsics.get(cam0_frame_id)
    if T_cam0_raw is None:
        logger.warning("No extrinsic found for camera-0 frame '%s' - skipping labels.", cam0_frame_id)
        return []

    T_cam0_to_ref = np.array(T_cam0_raw, dtype=np.float64)

    obj_id_to_box2d = _build_obj_id_to_box2d(camera_annotations_by_cam, ref_cam_key, cam0_intrinsics)
    P_cam0, img_w, img_h = _build_cam0_projection(cam0_intrinsics)

    lines: List[str] = []
    for ann in lidar_annotations:
        obj_id = str(ann.get("obj_id", ""))
        box2d  = obj_id_to_box2d.get(obj_id)

        line = _kitti_line_from_lidar_ann(
            ann, T_lidar_to_ref, T_cam0_to_ref, box2d,
            P_cam0=P_cam0, img_w=img_w, img_h=img_h,
        )
        if line is not None:
            lines.append(line)

    return lines


def write_kitti_labels_for_sample(
    fid: str,
    row: dict,
    config,
    stage1_dir: str,
    output_dir: str,
    extrinsics: dict,
    camera,
    calibration: Optional[dict] = None,
) -> bool:
    """
    Convert all annotations for one KITTI sample and write the label file to
    training/label_2/<fid>.txt.

    A KITTI sample is one image = one camera = one CSV row. The 3-D box is
    expressed in `camera`'s optical frame, and the 2-D box is sourced from
    `camera`'s own annotations (labels/<camera.key>/), falling back to
    projecting the 3-D box onto `camera`'s image when an object is not
    annotated in this camera.

    LiDAR annotations are read exclusively from labels/lidar/ (the fused
    point cloud format). The timestamp used to look up the fused LiDAR JSON
    is always the primary LiDAR's (config.lidars[0], i.e. lidar_front), since
    the fused cloud is sampled at the same matched timestamp.

    Parameters
    fid         : 6-digit KITTI frame identifier string (e.g. "000042")
    row         : timestamps.csv row dict {col_name -> timestamp_str}
    config      : PipelineConfig
    stage1_dir  : Stage-1 output directory
    output_dir  : KITTI output directory
    extrinsics  : dict loaded from calibration.yaml (extrinsics section)
    camera      : CameraConfig of this sample's camera (the KITTI P2 camera)
    calibration : full calibration dict (extrinsics + intrinsics). This camera's
                  P matrix is used to project the 3-D box onto the image as a
                  fallback 2-D box when no camera annotation exists.

    Returns True if label_2/<fid>.txt was written with at least one object,
    False if it was written empty (no labels/ dir, no matched annotations, or
    every candidate object was filtered out).
    """
    labels_dir = os.path.join(stage1_dir, "labels")

    if not os.path.isdir(labels_dir):
        # Labels folder absent: write empty file (backwards-compatible behaviour)
        _write_empty(output_dir, fid)
        return False

    # Primary LiDAR: lidar_front (index 0) - used for timestamp lookup.
    primary_lidar = config.lidars[0]
    lidar_ts = row.get(primary_lidar.key, "").strip()
    if not lidar_ts:
        logger.warning("[KITTI labels] Frame %s: no timestamp for %s.", fid, primary_lidar.key)
        _write_empty(output_dir, fid)
        return False

    lidar_json = _find_label_file(labels_dir, "lidar", lidar_ts)
    lidar_anns = _load_json(lidar_json)

    if not lidar_anns:
        _write_empty(output_dir, fid)
        return False

    # The fused-LiDAR annotations are anchored at the vehicle's
    # center_of_gravity frame (config.ego_pose.child_frame), which is a
    # distinct, physically offset point from ring_center (the frame all
    # other extrinsics - and the merged point cloud - are expressed in).
    # Resolve that offset from calibration.yaml; fall back to identity with
    # a warning if it isn't present (e.g. an older Stage-1 run, or a
    # vehicle/URDF without a distinct center_of_gravity link), in which case
    # labels degrade to the previous (origin-coincident) assumption.
    cog_frame_id = getattr(getattr(config, "ego_pose", None), "child_frame", None) or "center_of_gravity"
    T_lidar_to_ref_raw = extrinsics.get(cog_frame_id)
    if T_lidar_to_ref_raw is not None:
        T_lidar_to_ref = np.array(T_lidar_to_ref_raw, dtype=np.float64)
    else:
        logger.warning(
            "[KITTI labels] No calibration extrinsic found for the annotation "
            "anchor frame '%s'; treating it as coincident with the reference "
            "frame (identity). If the fused-LiDAR annotations are actually "
            "anchored elsewhere, every 3-D label will be offset by the "
            "missing transform - make sure '%s' is included in the "
            "sensor_links passed to extract_calibration().",
            cog_frame_id, cog_frame_id,
        )
        T_lidar_to_ref = _T_LIDAR_TO_REF

    # 2-D boxes come from this sample's camera only (keyed by cam.key).
    cam_ts = row.get(camera.key, "").strip()
    camera_anns: Dict[str, List[dict]] = {}
    if cam_ts:
        cam_json = _find_label_file(labels_dir, camera.key, cam_ts)
        anns = _load_json(cam_json)
        if anns:
            camera_anns[camera.key] = anns

    cam_intrinsics = (calibration or {}).get("intrinsics", {}).get(camera.name)

    lines = build_kitti_label_lines(
        lidar_annotations        = lidar_anns,
        camera_annotations_by_cam= camera_anns,
        extrinsics               = extrinsics,
        cam0_frame_id            = camera.frame_id,
        ref_cam_key              = camera.key,
        cam0_intrinsics          = cam_intrinsics,
        T_lidar_to_ref           = T_lidar_to_ref,
    )

    out_path = os.path.join(output_dir, "training", "label_2", f"{fid}.txt")
    with open(out_path, "w") as fh:
        if lines:
            fh.write("\n".join(lines) + "\n")

    return bool(lines)


def _write_empty(output_dir: str, fid: str) -> None:
    path = os.path.join(output_dir, "training", "label_2", f"{fid}.txt")
    open(path, "w").close()