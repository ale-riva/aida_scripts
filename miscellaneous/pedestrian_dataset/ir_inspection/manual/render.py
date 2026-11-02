"""
Pure rendering and geometry functions for the frame inspection tool.

Every function here takes arrays/dicts already in memory and returns
arrays/images in memory - no file I/O, no knowledge of the Stage-1
directory layout. That split is what keeps this module easy to read and
easy to reuse independently of inspect_frames.py's I/O/interaction code.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHITE_BGR = (255, 255, 255)
BLACK_BGR = (0, 0, 0)
GRAY_BGR = (120, 120, 120)
TILE_BORDER_BGR = (60, 60, 60)

KEPT_COLOR_BGR = (0, 200, 0)
DELETED_COLOR_BGR = (0, 0, 230)
DELETED_BANNER_BGR = (0, 0, 200)

BOX_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)

MIN_DEPTH = 0.1
BOX_LINE_THICKNESS = 2
BEV_BOX_THICKNESS = 2

# [x_min, y_min, z_min, x_max, y_max, z_max] in reference_frame.
DEFAULT_POINT_CLOUD_RANGE = [-120.0, -120.0, -5.0, 120.0, 120.0, 5.0]

DEFAULT_CANVAS_WIDTH = 2000
DEFAULT_CANVAS_HEIGHT = 1000

DEFAULT_VIEW_YAW = math.radians(90.0)
DEFAULT_VIEW_PITCH = math.radians(85.0)

# Camera body -> camera optical frame axis conversion.
# body:    x = forward, y = left,  z = up
# optical: x = right,   y = down,  z = into scene (forward)
_AXIS_T = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def apply_rigid(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N,3) array of points."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points.reshape(0, 3)
    r = T[:3, :3]
    t = T[:3, 3]
    return points @ r.T + t


def rotate_points(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """
    Rotate an (N,3) array of points for the interactive 3D view.

    yaw:   rotation about the world Z (up) axis, radians.
    pitch: subsequent rotation about the (already-yawed) X axis, radians,
           tilting the view up/down so the Z dimension of boxes becomes
           visible.

    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points.reshape(0, 3)

    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    cp, sp = math.cos(pitch), math.sin(pitch)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float64,
    )

    R = Rx @ Rz
    return points @ R.T


def psr_to_corners_ref(
    position: dict,
    rotation_z: float,
    scale: dict,
    T_local_to_ref: np.ndarray,
) -> np.ndarray:
    """
    Build the 8 corners of a PSR box (position/rotation.z/scale, expressed in
    its local annotation frame) and transform them into reference_frame via
    T_local_to_ref (4x4). Corner order matches BOX_EDGES.
    """
    dx = float(scale.get("x", 0.0))  # length (local x)
    dy = float(scale.get("y", 0.0))  # width  (local y)
    dz = float(scale.get("z", 0.0))  # height (local z)

    local_corners = np.array(
        [
            [dx / 2.0, dy / 2.0, -dz / 2.0],
            [dx / 2.0, -dy / 2.0, -dz / 2.0],
            [-dx / 2.0, -dy / 2.0, -dz / 2.0],
            [-dx / 2.0, dy / 2.0, -dz / 2.0],
            [dx / 2.0, dy / 2.0, dz / 2.0],
            [dx / 2.0, -dy / 2.0, dz / 2.0],
            [-dx / 2.0, -dy / 2.0, dz / 2.0],
            [-dx / 2.0, dy / 2.0, dz / 2.0],
        ],
        dtype=np.float64,
    )

    yaw = float(rotation_z)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    center = np.array(
        [float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0))],
        dtype=np.float64,
    )

    corners_local = local_corners @ rot.T + center
    return apply_rigid(T_local_to_ref, corners_local)


def build_projection_matrix(K_new: np.ndarray) -> np.ndarray:
    """3x4 pinhole projection matrix from a 3x3 (undistorted) camera matrix."""
    fx, fy = K_new[0, 0], K_new[1, 1]
    cx, cy = K_new[0, 2], K_new[1, 2]
    return np.array(
        [
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float64,
    )


def body_to_optical_transform(T_cam_to_ref: np.ndarray) -> np.ndarray:
    """3x4 transform mapping a reference_frame point into camera optical frame."""
    return (_AXIS_T @ np.linalg.inv(T_cam_to_ref))[:3, :]


def project_to_camera(
    corners_ref: np.ndarray,
    T_cam_to_ref: np.ndarray,
    K_new: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project (N,3) reference_frame points into a camera's undistorted pixel
    coordinates.

    Returns (uv (N,2), depth (N,)). depth is the optical-frame z coordinate;
    any corner/edge with depth <= MIN_DEPTH is behind the camera and must not
    be drawn by the caller (see draw_box_wireframe, which already does this).
    """
    corners_ref = np.asarray(corners_ref, dtype=np.float64)
    Tr = body_to_optical_transform(T_cam_to_ref)  # (3,4)
    pts_h = np.hstack([corners_ref, np.ones((len(corners_ref), 1))])  # (N,4)
    opt = (Tr @ pts_h.T).T  # (N,3)

    P = build_projection_matrix(K_new)  # (3,4)
    opt_h = np.hstack([opt, np.ones((len(opt), 1))])  # (N,4)
    proj = (P @ opt_h.T).T  # (N,3)

    depth = opt[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = proj[:, :2] / proj[:, 2:3]

    return uv, depth


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def put_text(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.6,
    color: Tuple[int, int, int] = WHITE_BGR,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK_BGR, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def letterbox_resize(image: np.ndarray, target_width: int, target_height: int,
                      fill_color: Tuple[int, int, int] = BLACK_BGR) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.full((target_height, target_width, 3), fill_color, dtype=np.uint8)

    scale = min(target_width / src_w, target_height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_height, target_width, 3), fill_color, dtype=np.uint8)
    x0 = (target_width - new_w) // 2
    y0 = (target_height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# 3D view (left) panel
# ---------------------------------------------------------------------------

def orthographic_project(
    points_rot: np.ndarray,
    width: int,
    height: int,
    point_cloud_range: Sequence[float] = DEFAULT_POINT_CLOUD_RANGE,
    zoom: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthographically project already-rotated (N,3) points onto a
    width x height canvas.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    extent = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6) / 2.0
    scale = (min(width, height) / 2.0) / extent * 0.9 * zoom  # 0.9 leaves a small margin

    u = width / 2.0 + points_rot[:, 0] * scale
    v = height / 2.0 - points_rot[:, 2] * scale
    return u, v


def _draw_ortho_box(
    canvas: np.ndarray,
    corners_rot: np.ndarray,
    color_bgr,
    width: int,
    height: int,
    point_cloud_range: Sequence[float],
    zoom: float = 1.0,
) -> None:
    u, v = orthographic_project(corners_rot, width, height, point_cloud_range, zoom)
    pts = np.stack([u, v], axis=1)

    for i0, i1 in BOX_EDGES:
        p0, p1 = pts[i0], pts[i1]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue
        cv2.line(canvas, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                  color_bgr, BEV_BOX_THICKNESS, cv2.LINE_AA)


def render_bev_panel(
    points_xyz: np.ndarray,
    boxes: List[dict],
    width: int,
    height: int,
    deleted: bool,
    frame_idx: int,
    total: int,
    point_cloud_range: Sequence[float] = DEFAULT_POINT_CLOUD_RANGE,
    view_yaw: float = DEFAULT_VIEW_YAW,
    view_pitch: float = DEFAULT_VIEW_PITCH,
    view_zoom: float = 1.0,
) -> np.ndarray:
    """
    Interactive, rotatable orthographic 3D view (replaces the old fixed BEV).

    boxes: list of {"obj_id": str, "obj_type": str, "corners_ref": (8,3)}
    view_yaw / view_pitch: current camera rotation in radians, driven by the
    caller's keyboard handling (A/D yaw, W/S pitch). view_zoom: current zoom
    factor (+/- keys, >1 zooms in). Re-rendering with new values is the only
    thing needed to rotate/zoom the view.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if len(points_xyz) > 0:
        valid = (
            (points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_max) &
            (points_xyz[:, 1] >= y_min) & (points_xyz[:, 1] <= y_max) &
            (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)
        )
        xyz = points_xyz[valid]
        if len(xyz) > 0:
            xyz_rot = rotate_points(xyz, view_yaw, view_pitch)
            u, v = orthographic_project(xyz_rot, width, height, point_cloud_range, view_zoom)
            u = np.round(u).astype(np.int32)
            v = np.round(v).astype(np.int32)
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            canvas[v[inside], u[inside]] = GRAY_BGR

    box_color = DELETED_COLOR_BGR if deleted else KEPT_COLOR_BGR
    for box in boxes:
        corners_rot = rotate_points(box["corners_ref"], view_yaw, view_pitch)
        _draw_ortho_box(canvas, corners_rot, box_color, width, height, point_cloud_range, view_zoom)

    put_text(canvas, f"Frame {frame_idx + 1}/{total}",
              (20, 35), scale=0.65, thickness=2)

    dim_text = f"Range (m): X[{x_min:.1f}, {x_max:.1f}] Y[{y_min:.1f}, {y_max:.1f}] Z[{z_min:.1f}, {z_max:.1f}]"
    put_text(canvas, dim_text, (20, 60), scale=0.5, color=(180, 180, 180), thickness=1)

    view_text = f"View yaw={math.degrees(view_yaw):.0f} deg  pitch={math.degrees(view_pitch):.0f} deg  zoom={view_zoom:.2f}x"
    put_text(canvas, view_text, (20, 82), scale=0.5, color=(180, 180, 180), thickness=1)

    return canvas


# ---------------------------------------------------------------------------
# Camera (right) panel
# ---------------------------------------------------------------------------

def camera_grid_shape(n: int, panel_width: int, panel_height: int, image_aspect: float = 16 / 9) -> Tuple[int, int]:
    """
    Return (rows, cols) for n tiles inside a panel_width x panel_height area
    that maximizes the actual displayed size of each (image_aspect = w/h)
    camera image.

    A grid whose cell aspect ratio doesn't match the cameras' aspect ratio
    wastes most of each cell as letterbox padding. For landscape cameras
    (aspect > 1) inside a roughly square panel, this consistently picks
    fewer, wider columns (a "column" layout) over a wide, shallow grid,
    because that is what actually lets each image render bigger.
    """
    if n <= 0:
        return 1, 1

    best_rows, best_cols, best_scale = 1, n, -1.0
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w = panel_width / cols
        cell_h = panel_height / rows
        scale = min(cell_w / image_aspect, cell_h)  # limiting dimension, in "image_aspect x 1" units
        if scale > best_scale:
            best_scale = scale
            best_rows, best_cols = rows, cols

    return best_rows, best_cols


def draw_box_wireframe(image_bgr: np.ndarray, uv: np.ndarray, depth: np.ndarray,
                        color_bgr, label: str) -> bool:
    """uv: (8,2) pixel coords, depth: (8,) optical-frame z. Draws in place."""
    h, w = image_bgr.shape[:2]
    drawn = False

    for i0, i1 in BOX_EDGES:
        if depth[i0] <= MIN_DEPTH or depth[i1] <= MIN_DEPTH:
            continue
        p0, p1 = uv[i0], uv[i1]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue
        cv2.line(image_bgr, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                  color_bgr, BOX_LINE_THICKNESS, cv2.LINE_AA)
        drawn = True

    # if drawn:
    #     usable = (depth > MIN_DEPTH) & np.isfinite(uv).all(axis=1)
    #     if np.any(usable):
    #         text_xy = np.mean(uv[usable], axis=0)
    #         if 0 <= text_xy[0] < w and 0 <= text_xy[1] < h:
    #             put_text(image_bgr, label, tuple(np.round(text_xy).astype(int)),
    #                       scale=0.45, color=color_bgr, thickness=1)

    return drawn


def render_camera_panel(tiles: List[dict], width: int, height: int) -> np.ndarray:
    """
    tiles: list of {"key": str, "image": Optional[np.ndarray] (BGR),
                     "box_count": int}
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    if not tiles:
        put_text(panel, "No cameras for this frame", (30, height // 2), scale=0.7)
        return panel

    image_aspect = 16 / 9
    for tile in tiles:
        img = tile.get("image")
        if img is not None and img.shape[0] > 0:
            image_aspect = img.shape[1] / img.shape[0]
            break

    rows, cols = camera_grid_shape(len(tiles), width, height, image_aspect)
    cell_w = width // cols
    cell_h = height // rows

    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        x0, y0 = col * cell_w, row * cell_h
        x1 = width if col == cols - 1 else (col + 1) * cell_w
        y1 = height if row == rows - 1 else (row + 1) * cell_h
        tile_w, tile_h = x1 - x0, y1 - y0

        img = tile.get("image")
        if img is None:
            cell = np.full((tile_h, tile_w, 3), BLACK_BGR, dtype=np.uint8)
            put_text(cell, f"{tile['key']}: no image", (12, tile_h // 2), scale=0.55)
        else:
            cell = letterbox_resize(img, tile_w, tile_h)
            put_text(cell, f"{tile['key']} | boxes={tile['box_count']}", (12, 28), scale=0.55)

        panel[y0:y1, x0:x1] = cell
        cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), TILE_BORDER_BGR, 1)

    return panel


# ---------------------------------------------------------------------------
# Composite canvas
# ---------------------------------------------------------------------------

def build_composite(bev_panel: np.ndarray, camera_panel: np.ndarray, deleted: bool, unsaved: bool) -> np.ndarray:
    """bev_panel and camera_panel must already share the same height."""
    height = bev_panel.shape[0]
    left_w = bev_panel.shape[1]
    right_w = camera_panel.shape[1]
    width = left_w + right_w

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :left_w] = bev_panel
    canvas[:, left_w:] = camera_panel
    cv2.line(canvas, (left_w, 0), (left_w, height - 1), (180, 180, 180), 2)

    if deleted:
        border_thickness = 6
        cv2.rectangle(canvas, (0, 0), (width, height), DELETED_BANNER_BGR, border_thickness)
        
        text = "FRAME DELETED"
        text_scale = 1.0
        text_thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness + 2)
        
        pad = 10
        x_center = int(width // 2 * 0.8)
        cv2.rectangle(canvas, (x_center - (text_w//2) - pad, border_thickness), 
                              (x_center + (text_w//2) + pad, border_thickness + text_h + pad * 2), 
                              DELETED_BANNER_BGR, -1)
                              
        put_text(canvas, text, (x_center - (text_w//2), border_thickness + text_h + pad), 
                 scale=text_scale, color=WHITE_BGR, thickness=text_thickness)

    footer = "<-/-> navigate   A/D yaw   W/S pitch   +/- zoom   X delete   R restore   Q/Esc quit"
    if unsaved:
        footer += "   [unsaved changes]"
    put_text(canvas, footer, (12, height - 12), scale=0.5)

    return canvas


def build_save_prompt_overlay(canvas: np.ndarray) -> np.ndarray:
    """Dim the canvas and draw the save/discard/cancel prompt on top of it."""
    overlay = canvas.copy()
    dark = (canvas.astype(np.float32) * 0.35).astype(np.uint8)
    h, w = canvas.shape[:2]

    box_w, box_h = 560, 120
    x0, y0 = (w - box_w) // 2, (h - box_h) // 2
    cv2.rectangle(dark, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), -1)
    cv2.rectangle(dark, (x0, y0), (x0 + box_w, y0 + box_h), (200, 200, 200), 2)

    put_text(dark, "Save changes before exiting?", (x0 + 30, y0 + 45), scale=0.7, thickness=2)
    put_text(dark, "[Y] Save   [N] Discard   [C] Cancel", (x0 + 30, y0 + 85), scale=0.6)

    return dark