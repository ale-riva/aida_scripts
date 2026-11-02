#!/usr/bin/env python3
"""
Frame Inspection and Manual Filtering Tool.

Lets a reviewer step through every frame (row) of a Stage-1 timestamps.csv,
see the merged LiDAR point cloud + its 3D labels next to all camera images
with those labels projected onto them, mark frames for exclusion, and export
a filtered copy of the timeline. The original timestamps.csv is never
modified.

Usage
    python inspect_frames.py <stage1_output_directory>

Keys
    Right / N   next frame
    Left  / P   previous frame
    A / D       rotate 3D view left/right (yaw)
    W / S       tilt 3D view up/down (pitch)
    + / -       zoom 3D view in/out
    X           mark current frame deleted
    R           restore current frame
    Q / Esc     quit (prompts to save if there are unsaved changes)
"""

from __future__ import annotations

import sys
import os

# Get the directory of the current script, then go up one level to the root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

import argparse
import csv
import json
import logging
import math
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import yaml

from utils.pcd_reader import read_pcd, to_kitti_bin
from utils.image_utils import undistort_image, optimal_undistorted_K

import render

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

_WARNED: Set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


# ---------------------------------------------------------------------------
# Stage-1 directory reading
# ---------------------------------------------------------------------------

def read_timestamps_csv(stage1_dir: str) -> Tuple[List[str], List[Dict[str, str]]]:
    path = os.path.join(stage1_dir, "timestamps.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"timestamps.csv not found in: {stage1_dir}")
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def discover_sensors(stage1_dir: str) -> Dict[str, str]:
    """
    Return {sensor_key: "lidar" | "camera"}, discovered from samples/<key>/
    by sniffing one file's extension per sub-directory.
    """
    samples_dir = os.path.join(stage1_dir, "samples")
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"samples/ not found in: {stage1_dir}")

    sensor_types: Dict[str, str] = {}
    for key in sorted(os.listdir(samples_dir)):
        sensor_dir = os.path.join(samples_dir, key)
        if not os.path.isdir(sensor_dir):
            continue

        kind = None
        for fname in os.listdir(sensor_dir):
            if fname.endswith(".pcd"):
                kind = "lidar"
                break
            if fname.endswith(".png"):
                kind = "camera"
                break

        if kind is None:
            logger.warning("Could not determine sensor type for '%s' (no .pcd/.png found) - skipping.", key)
            continue

        sensor_types[key] = kind

    if not sensor_types:
        raise ValueError(f"No lidar/camera sensors discovered under: {samples_dir}")

    return sensor_types


def load_calibration(stage1_dir: str) -> dict:
    """
    Load calibration.yaml and merge in the small inspection_config.yaml
    sidecar (annotation_anchor_frame + sensors: {key -> frame_id/name}).
    """
    calib_path = os.path.join(stage1_dir, "calibration.yaml")
    inspection_path = os.path.join(stage1_dir, "inspection_config.yaml")

    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"calibration.yaml not found in: {stage1_dir}")
    if not os.path.isfile(inspection_path):
        raise FileNotFoundError(
            f"inspection_config.yaml not found in: {stage1_dir}\n"
            "This file is required by inspect_frames.py. It is written by Stage 1 "
            "(extraction/inspection_metadata_writer.py::write_inspection_config). "
            "Re-run Stage 1 to generate it."
        )

    with open(calib_path) as fh:
        calibration = yaml.safe_load(fh) or {}
    with open(inspection_path) as fh:
        inspection_config = yaml.safe_load(fh) or {}

    for key in ("annotation_anchor_frame", "sensors"):
        if key not in inspection_config:
            raise ValueError(f"inspection_config.yaml is missing required key '{key}'")

    for key in ("reference_frame", "extrinsics", "intrinsics"):
        if key not in calibration:
            raise ValueError(f"calibration.yaml is missing required key '{key}'")

    calibration["annotation_anchor_frame"] = inspection_config["annotation_anchor_frame"]
    calibration["sensors"] = inspection_config["sensors"]
    return calibration


# ---------------------------------------------------------------------------
# Per-frame data loading
# ---------------------------------------------------------------------------

def _points_xyz(points_struct: np.ndarray) -> np.ndarray:
    return to_kitti_bin(points_struct)[:, :3].astype(np.float64)


def load_frame_points_and_boxes(
    stage1_dir: str,
    calibration: dict,
    sensor_types: Dict[str, str],
    row: Dict[str, str],
) -> Tuple[np.ndarray, List[dict]]:
    """
    Merge all available lidar point clouds into reference_frame, and load the
    fused-LiDAR 3D labels (also transformed into reference_frame).

    Returns (points_xyz (N,3), boxes: [{"obj_id", "obj_type", "corners_ref"}]).
    """
    extrinsics = calibration.get("extrinsics", {})
    sensors = calibration.get("sensors", {})
    lidar_keys = sorted(k for k, t in sensor_types.items() if t == "lidar")

    # --- merged point cloud ---
    parts = []
    for key in lidar_keys:
        ts = row.get(key, "").strip()
        if not ts:
            continue

        pcd_path = os.path.join(stage1_dir, "samples", key, f"{ts}.pcd")
        if not os.path.isfile(pcd_path):
            logger.warning("Missing PCD file: %s", pcd_path)
            continue

        sensor_info = sensors.get(key)
        if sensor_info is None:
            _warn_once(f"no_sensor_info_{key}", f"No inspection_config.yaml entry for '{key}' - skipping this lidar.")
            continue

        T_raw = extrinsics.get(sensor_info["frame_id"])
        if T_raw is None:
            _warn_once(f"no_extrinsic_{key}", f"No extrinsic for '{key}' (frame_id={sensor_info['frame_id']}) - skipping this lidar.")
            continue

        try:
            struct = read_pcd(pcd_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pcd_path, exc)
            continue

        xyz = _points_xyz(struct)
        xyz_ref = render.apply_rigid(np.array(T_raw, dtype=np.float64), xyz)
        parts.append(xyz_ref)

    points_xyz = np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.float64)

    # --- annotation anchor -> reference_frame transform ---
    anchor_frame = calibration.get("annotation_anchor_frame")
    T_lidar_to_ref_raw = extrinsics.get(anchor_frame) if anchor_frame else None
    if T_lidar_to_ref_raw is None:
        _warn_once(
            "no_anchor_extrinsic",
            f"No extrinsic found for annotation_anchor_frame '{anchor_frame}' - "
            "treating labels as already expressed in reference_frame (identity fallback). "
            "3D boxes may be offset from their true position.",
        )
        T_lidar_to_ref = np.eye(4, dtype=np.float64)
    else:
        T_lidar_to_ref = np.array(T_lidar_to_ref_raw, dtype=np.float64)

    # --- boxes: first lidar column (sorted) with a matching label file wins ---
    boxes: List[dict] = []
    for key in lidar_keys:
        ts = row.get(key, "").strip()
        if not ts:
            continue

        label_path = os.path.join(stage1_dir, "labels", "lidar", f"{ts}.json")
        if not os.path.isfile(label_path):
            continue

        try:
            with open(label_path) as fh:
                anns = json.load(fh)
        except Exception as exc:
            logger.warning("Failed to read label file %s: %s", label_path, exc)
            anns = []

        if not isinstance(anns, list):
            anns = []

        for ann in anns:
            psr = ann.get("psr", {})
            corners_ref = render.psr_to_corners_ref(
                psr.get("position", {}),
                float(psr.get("rotation", {}).get("z", 0.0)),
                psr.get("scale", {}),
                T_lidar_to_ref,
            )
            boxes.append({
                "obj_id": str(ann.get("obj_id", "")),
                "obj_type": str(ann.get("obj_type", "")),
                "corners_ref": corners_ref,
            })

        break  # only the first matching lidar column's label file is used

    return points_xyz, boxes


def load_camera_tiles(
    stage1_dir: str,
    calibration: dict,
    sensor_types: Dict[str, str],
    row: Dict[str, str],
    boxes: List[dict],
    deleted: bool,
) -> List[dict]:
    """
    For every camera present in this row: undistort the image and project
    every box's wireframe onto it.

    Returns a list of {"key", "image" (BGR or None), "box_count"} tiles, one
    per camera that has an image for this frame (cameras absent from the row
    contribute no tile at all).
    """
    extrinsics = calibration.get("extrinsics", {})
    intrinsics = calibration.get("intrinsics", {})
    sensors = calibration.get("sensors", {})
    camera_keys = sorted(k for k, t in sensor_types.items() if t == "camera")

    box_color = render.DELETED_COLOR_BGR if deleted else render.KEPT_COLOR_BGR
    tiles: List[dict] = []

    for key in camera_keys:
        ts = row.get(key, "").strip()
        if not ts:
            continue  # no image for this camera in this frame -> no tile

        png_path = os.path.join(stage1_dir, "samples", key, f"{ts}.png")
        image = None
        box_count = 0

        if not os.path.isfile(png_path):
            logger.warning("Missing image file: %s", png_path)
        else:
            raw = cv2.imread(png_path, cv2.IMREAD_COLOR)
            if raw is None:
                logger.warning("Failed to decode image: %s", png_path)
            else:
                sensor_info = sensors.get(key)
                intr = intrinsics.get(sensor_info["name"]) if sensor_info else None
                T_cam_to_ref_raw = extrinsics.get(sensor_info["frame_id"]) if sensor_info else None

                if sensor_info is None:
                    _warn_once(f"no_sensor_info_{key}", f"No inspection_config.yaml entry for '{key}' - showing raw image, no boxes.")
                    image = raw
                elif intr is None or T_cam_to_ref_raw is None:
                    _warn_once(f"no_calib_{key}", f"Missing intrinsics/extrinsics for '{key}' - showing raw image, no boxes.")
                    image = raw
                else:
                    image = undistort_image(raw, intr)
                    K_new = optimal_undistorted_K(intr)
                    T_cam_to_ref = np.array(T_cam_to_ref_raw, dtype=np.float64)
                    for box in boxes:
                        uv, depth = render.project_to_camera(box["corners_ref"], T_cam_to_ref, K_new)
                        label = f"{box['obj_type']} {box['obj_id']}".strip()
                        if render.draw_box_wireframe(image, uv, depth, box_color, label):
                            box_count += 1

        tiles.append({"key": key, "image": image, "box_count": box_count})

    return tiles


# ---------------------------------------------------------------------------
# Deletion state / export
# ---------------------------------------------------------------------------

def _lidar_signature(row: Dict[str, str], lidar_keys: List[str]) -> Tuple[str, ...]:
    return tuple(row.get(k, "").strip() for k in lidar_keys)


def load_baseline_deleted(
    stage1_dir: str,
    rows: List[Dict[str, str]],
    lidar_keys: List[str],
) -> Set[int]:
    """
    If timestamps_curated.csv already exists (a previous review session),
    pre-mark as deleted every row whose lidar-column values don't appear in
    it. Returns an empty set if no curated file exists yet.
    """
    curated_path = os.path.join(stage1_dir, "timestamps_curated.csv")
    if not os.path.isfile(curated_path):
        return set()

    with open(curated_path, newline="") as fh:
        curated_signatures = {_lidar_signature(r, lidar_keys) for r in csv.DictReader(fh)}

    deleted = {
        idx for idx, row in enumerate(rows)
        if _lidar_signature(row, lidar_keys) not in curated_signatures
    }

    if deleted:
        logger.info("Resumed previous review from %s: %d frame(s) pre-marked deleted.", curated_path, len(deleted))
    return deleted


def export_curated_csv(
    stage1_dir: str,
    header: List[str],
    rows: List[Dict[str, str]],
    deleted_indices: Set[int],
) -> str:
    """
    Write timestamps_curated.csv: same header/column order/row order as
    timestamps.csv, minus the deleted rows. Never touches timestamps.csv.
    Atomic write (temp file + os.replace).
    """
    out_path = os.path.join(stage1_dir, "timestamps_curated.csv")
    tmp_path = out_path + ".tmp"

    with open(tmp_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for idx, row in enumerate(rows):
            if idx in deleted_indices:
                continue
            writer.writerow(row)

    os.replace(tmp_path, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

# Arrow-key codes returned by cv2.waitKeyEx vary across OpenCV builds and
# platforms (GTK/Qt/Win32 backends). The sets below cover the common cases;
# N/P are provided as a guaranteed-portable fallback regardless of backend.
_LEFT_CODES = {81, 65361, 2424832, ord("p"), ord("P")}
_RIGHT_CODES = {83, 65363, 2555904, ord("n"), ord("N")}
_QUIT_CODES = {27, ord("q"), ord("Q")}
_DELETE_CODES = {ord("x"), ord("X")}
_RESTORE_CODES = {ord("r"), ord("R")}

# 3D view rotation (D is intentionally not bound to delete anymore, since it
# is used here for yaw; delete moved to X - see _DELETE_CODES above).
_YAW_LEFT_CODES = {ord("a"), ord("A")}
_YAW_RIGHT_CODES = {ord("d"), ord("D")}
_PITCH_UP_CODES = {ord("w"), ord("W")}
_PITCH_DOWN_CODES = {ord("s"), ord("S")}
_ZOOM_IN_CODES = {ord("+"), ord("=")}
_ZOOM_OUT_CODES = {ord("-"), ord("_")}

VIEW_ROTATE_STEP = math.radians(5.0)
VIEW_PITCH_MIN = math.radians(-85.0)
VIEW_PITCH_MAX = math.radians(85.0)
VIEW_ZOOM_STEP = 1.1
VIEW_ZOOM_MIN = 0.2
VIEW_ZOOM_MAX = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Stage-1 frames and mark frames for exclusion.")
    parser.add_argument("stage1_dir", help="Path to a Stage-1 intermediate output directory")
    args = parser.parse_args()

    stage1_dir = os.path.abspath(args.stage1_dir)

    header, rows = read_timestamps_csv(stage1_dir)
    if not rows:
        print("timestamps.csv has no rows - nothing to inspect.")
        return

    sensor_types = discover_sensors(stage1_dir)
    calibration = load_calibration(stage1_dir)

    lidar_keys = sorted(k for k, t in sensor_types.items() if t == "lidar")
    if not lidar_keys:
        raise ValueError("No lidar sensors discovered under samples/ - nothing to visualize.")

    total = len(rows)
    deleted = load_baseline_deleted(stage1_dir, rows, lidar_keys)
    baseline = set(deleted)
    cursor = 0

    # Interactive 3D view state (yaw/pitch), independent of frame navigation.
    view_yaw = render.DEFAULT_VIEW_YAW
    view_pitch = render.DEFAULT_VIEW_PITCH
    view_zoom = 1.0

    window = "Frame Inspector"
    # WINDOW_GUI_NORMAL drops the Qt toolbar/status bar (pan, zoom, save icons)
    # that add nothing here since navigation is entirely keyboard-driven.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    cv2.resizeWindow(window, 1600, 800)

    def render_current() -> np.ndarray:
        row = rows[cursor]
        is_deleted = cursor in deleted

        points_xyz, boxes = load_frame_points_and_boxes(stage1_dir, calibration, sensor_types, row)
        tiles = load_camera_tiles(stage1_dir, calibration, sensor_types, row, boxes, is_deleted)

        left_w = render.DEFAULT_CANVAS_WIDTH // 2
        right_w = render.DEFAULT_CANVAS_WIDTH - left_w
        height = render.DEFAULT_CANVAS_HEIGHT

        bev_panel = render.render_bev_panel(
            points_xyz, boxes, left_w, height, is_deleted, cursor, total,
            view_yaw=view_yaw, view_pitch=view_pitch, view_zoom=view_zoom,
        )
        camera_panel = render.render_camera_panel(tiles, right_w, height)
        unsaved = deleted != baseline

        return render.build_composite(bev_panel, camera_panel, is_deleted, unsaved)

    canvas = render_current()
    cv2.imshow(window, canvas)

    while True:
        key = cv2.waitKeyEx(0)

        if key in _QUIT_CODES:
            if deleted != baseline:
                cv2.imshow(window, render.build_save_prompt_overlay(canvas))
                choice = cv2.waitKeyEx(0) & 0xFF
                choice_char = chr(choice).lower() if 0 <= choice < 256 else ""

                if choice_char == "y":
                    path = export_curated_csv(stage1_dir, header, rows, deleted)
                    print(f"Saved: {path}  ({total - len(deleted)}/{total} frames kept)")
                    break
                elif choice_char == "n":
                    break
                else:  # cancel -> back to browsing
                    cv2.imshow(window, canvas)
                    continue
            else:
                break

        elif key in _RIGHT_CODES:
            if cursor < total - 1:
                cursor += 1
                canvas = render_current()
                cv2.imshow(window, canvas)

        elif key in _LEFT_CODES:
            if cursor > 0:
                cursor -= 1
                canvas = render_current()
                cv2.imshow(window, canvas)

        elif key in _DELETE_CODES:
            deleted.add(cursor)
            canvas = render_current()
            cv2.imshow(window, canvas)

        elif key in _RESTORE_CODES:
            deleted.discard(cursor)
            canvas = render_current()
            cv2.imshow(window, canvas)

        elif key in _YAW_LEFT_CODES:
            view_yaw = (view_yaw - VIEW_ROTATE_STEP) % (2 * math.pi)
            canvas = render_current()
            cv2.imshow(window, canvas)

        elif key in _YAW_RIGHT_CODES:
            view_yaw = (view_yaw + VIEW_ROTATE_STEP) % (2 * math.pi)
            canvas = render_current()
            cv2.imshow(window, canvas)

        elif key in _PITCH_UP_CODES:
            view_pitch = min(VIEW_PITCH_MAX, view_pitch + VIEW_ROTATE_STEP)
            canvas = render_current()
            cv2.imshow(window, canvas)

        elif key in _PITCH_DOWN_CODES:
            view_pitch = max(VIEW_PITCH_MIN, view_pitch - VIEW_ROTATE_STEP)
            canvas = render_current()
            cv2.imshow(window, canvas)

        elif key in _ZOOM_IN_CODES:
            view_zoom = min(VIEW_ZOOM_MAX, view_zoom * VIEW_ZOOM_STEP)
            canvas = render_current()
            cv2.imshow(window, canvas)

        elif key in _ZOOM_OUT_CODES:
            view_zoom = max(VIEW_ZOOM_MIN, view_zoom / VIEW_ZOOM_STEP)
            canvas = render_current()
            cv2.imshow(window, canvas)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()