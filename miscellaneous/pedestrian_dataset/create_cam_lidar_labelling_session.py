#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import json
import shutil
import yaml
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.pcd_reader import (
    to_xyz_bin,
    merge_and_transform as _merge_and_transform,
    write_xyz_pcd_binary as _write_pcd
)
from utils.shared_loader import _load_calibration, _load_timestamps_csv, logger
from utils.image_utils import (
    undistort_image as _undistort_image,
    _CAM_BODY_TO_OPT,
    optimal_undistorted_K as _optimal_undistorted_K,
)

_LIDAR_SENSORS = [
    "lidar_front",
    "lidar_left",
    "lidar_right",
]

_SCRIPT_DIR = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a labelling_session folder by merging lidar samples and copying labels."
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset folder containing samples/, calibration.yaml, timestamps.csv, etc.",
    )

    parser.add_argument(
        "car_name",
        type=str,
        help=(
            "car name (must have a corresponding setting file in config folder)"
        ),
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Optional input labels directory. "
            "If not provided, the script assumes dataset_root/labels."
        ),
    )

    parser.add_argument(
        "--raw-labels",
        action="store_true",
        help=(
            "If set, input labels already contain the annotator field. "
            "If not set, annotator is added to each label object and the JSON is copied."
        ),
    )

    parser.add_argument(
        "--with-cam",
        action="store_true",
        help=(
            "If set, undistort the images and add them to the labelling session"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in labelling_session.",
    )

    return parser.parse_args()


def validate_dataset_root(dataset_root: Path) -> None:
    required_paths = [
        dataset_root / "samples",
        dataset_root / "samples" / "lidar_front",
        dataset_root / "samples" / "lidar_left",
        dataset_root / "samples" / "lidar_right",
        dataset_root / "timestamps.csv",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


def create_ls_lidar_dirs(dataset_root: Path) -> tuple[Path, Path, Path]:
    labelling_session_dir = dataset_root / "labelling_session"
    output_lidar_dir = labelling_session_dir / "lidar"
    output_labels_dir = labelling_session_dir / "label"

    output_lidar_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    return labelling_session_dir, output_lidar_dir, output_labels_dir

def create_ls_camera_dirs(dataset_root: Path, camera_names: List[str]) -> tuple[Path, List[Path], Path]:
    labelling_session_dir = dataset_root / "labelling_session"
    camera_main_dir = labelling_session_dir / "camera"
    output_calibrations_dir = labelling_session_dir / "calib" / "camera"

    output_camera_dirs: List[Path] = []
    for camera in camera_names:
        output_camera_dir = camera_main_dir / camera
        output_camera_dir.mkdir(parents=True, exist_ok=True)
        output_camera_dirs.append(output_camera_dir)
    output_calibrations_dir.mkdir(parents=True, exist_ok=True)

    return labelling_session_dir, output_camera_dirs, output_calibrations_dir


def load_camera_name_map(car_name: str) -> dict[str, str]:
    """
    Load a sensor settings YAML file and return a dictionary:

        {
            "camera_0": "camera_sl",
            "camera_1": "camera_fl",
            ...
        }
    """
    
    sensor_yaml_path = _SCRIPT_DIR / "config" / f"settings_{car_name}.yaml"
    with sensor_yaml_path.open("r") as f:
        data = yaml.safe_load(f)
    cameras = data.get("cameras", {})
    camera_name_map: dict[str, str] = {}
    for camera_key, camera_cfg in cameras.items():
        if not isinstance(camera_cfg, dict):
            continue
        camera_name = camera_cfg.get("name")
        if camera_name is None:
            continue
        camera_name_map[camera_key] = camera_name

    return camera_name_map

def add_annotator_to_label(
    input_label_path: Path,
    output_label_path: Path,
) -> None:
    """
    Read a label JSON, add annotator information, and write it to output.

    This function contains only generic JSON handling.
    Adapt the exact JSON field name if needed.
    """

    with input_label_path.open("r") as f:
        label_data = json.load(f)

    if not isinstance(label_data, list):
        raise TypeError(
            f"Expected label file to contain a list of objects, "
            f"but got {type(label_data).__name__}: {input_label_path}"
        )

    for obj in label_data:
        if not isinstance(obj, dict):
            raise TypeError(
                f"Expected every label entry to be a dict, "
                f"but got {type(obj).__name__}: {input_label_path}"
            )

        obj["annotator"] = "a"

    with output_label_path.open("w") as f:
        json.dump(label_data, f, indent=2)

def write_camera(dataset_root, cam_key: str, row:dict, intr:dict, out_dir:Path)->None:
    ref_stem = row['lidar_front']
    cam_stem = row[cam_key]
    src = os.path.join(dataset_root, "samples", cam_key, f"{cam_stem}.png")
    if not os.path.isfile(src):
        logger.warning(
            "[LabSes] Camera image not found for %s at %s - skipping.", cam_key, src
        )
        return
    
    img = Image.open(src).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    img_np = _undistort_image(img_np, intr)
    img = Image.fromarray(img_np)

    out_path = out_dir / f"{ref_stem}.jpg"
    img.save(out_path, "JPEG", quality=80)

def write_calibrations(intr:dict, extr:dict, cam_names:list[str], out_dir: Path,):
    for cam_name in cam_names:
        sustech_calib = build_calib_sustech(cam_name, intr, extr)

        output_path = out_dir / f"{cam_name}.json"
        with output_path.open("w") as f:
                json.dump(sustech_calib, f, indent=4)

def build_calib_sustech(camera_name:str, intr: dict, extr: dict)->dict:
    """
    Build one camera calibration dictionary in the format:

    {
        "extrinsic": [16 values, row-major 4x4],
        "intrinsic": [9 values, row-major 3x3]
    }

    camera_name: camera_fl, camera_fr, camera_cl, ...
    """

    if camera_name not in extr:
        raise KeyError(f"Missing extrinsic calibration for camera: {camera_name}")

    if camera_name not in intr:
        raise KeyError(f"Missing intrinsic calibration for camera: {camera_name}")

    T_raw = extr.get(camera_name)
    T_opt_to_ref = np.eye(4, dtype=np.float64)
    if T_raw is not None:
        T = np.array(extr.get(camera_name), dtype=np.float64)   # T_cam_body -> reference
        T_opt_to_ref[:3, :3] = T[:3, :3] @ _CAM_BODY_TO_OPT.T   
        T_opt_to_ref[:3, 3] = T[:3, 3]
        T_ref_to_opt = np.linalg.inv(T_opt_to_ref) #reference -> T_cam_body (direction needed by sustechpoint)
    else:
        logger.warning(
            "[LabelSes] No extrinsic for camera '%s'; using identity.",camera_name
        )

    if T_opt_to_ref.shape != (4, 4):
        raise ValueError(
            f"Extrinsic for {camera_name} must have shape (4, 4), got {T.shape}"
        )

    cam_intr = intr[camera_name]

    if cam_intr:
        K_new = _optimal_undistorted_K(cam_intr)
    else:
        logger.warning(
            "[nuScenes] No intrinsics for camera '%s'; camera_intrinsic will be empty.",
            camera_name,
        )

    return {
        "extrinsic": T_ref_to_opt.reshape(-1).tolist(),
        "intrinsic": K_new.reshape(-1).tolist(),
    }


def process_label(
    ref_stem: str,
    input_labels_dir: Path,
    output_labels_dir: Path,
    overwrite: bool,
    raw_labels: bool,
) -> None:
    """
    Process one label:
      - find corresponding label
      - if annotator_in_labels is True, copy the existing label JSON
      - otherwise, add annotator and write JSON to labelling_session/label
    """

    label_path = input_labels_dir / f"{ref_stem}.json"
    output_label_path = output_labels_dir / f"{ref_stem}.json"

    if not label_path.is_file():
        print(f"No label found for sample: {label_path}")
        return

    if output_label_path.exists() or output_label_path.is_symlink():
        if not overwrite:
            print(f"Skipping existing label: {output_label_path}")
            return

        output_label_path.unlink()

    if raw_labels:
        shutil.copy2(label_path, output_label_path)
    else:
        add_annotator_to_label(
            input_label_path=label_path,
            output_label_path=output_label_path,
        )

def main() -> None:
    args = parse_args()

    dataset_root: Path = args.dataset_root.resolve()
    validate_dataset_root(dataset_root)

    # Resolve input labels directory
    input_labels_dir = args.labels_dir
    if input_labels_dir is None:
        input_labels_dir = dataset_root / "labels" / "lidar"
    else:
        input_labels_dir = input_labels_dir.resolve()

    if not input_labels_dir.exists():
        raise FileNotFoundError(f"Input labels directory not found: {input_labels_dir}")
    
    # Load camera name mapping
    camera_map = load_camera_name_map(args.car_name)

    # Load stage1 info
    headers, timestamp_rows = _load_timestamps_csv(dataset_root)
    calibration = _load_calibration(dataset_root)
    extrinsics = calibration.get("extrinsics", {})
    intrinsics = calibration.get("intrinsics", {})

    # Compute cam
    _, output_lidar_dir, output_labels_dir = create_ls_lidar_dirs(dataset_root)

    if args.with_cam: 
       cam_names = camera_map.values()
       _, output_camera_dirs, output_calib_dir = create_ls_camera_dirs(dataset_root, cam_names)
       write_calibrations(intrinsics, extrinsics, cam_names, output_calib_dir)


    print(f"Found {len(timestamp_rows)} timestamp entries.")
    print(f"Writing merged lidar to: {output_lidar_dir}")
    print(f"Writing labels to: {output_labels_dir}")

    n_frames = len(timestamp_rows)

    for row in tqdm(timestamp_rows, total=n_frames):
        ref_stem = row["lidar_front"]
        output_pcd_path = output_lidar_dir / f"{ref_stem}.pcd"

        # merge pointclouds
        if output_pcd_path.is_file() and not args.overwrite:
            logger.warning("Skipping existing lidar %s:" ,output_pcd_path)
        else:
            merged = _merge_and_transform(row, _LIDAR_SENSORS, dataset_root, extrinsics, to_xyz_bin)
            if len(merged) == 0:
                logger.warning("[LabelSes] Empty merged point cloud written to: %s", output_pcd_path)
            _write_pcd(output_pcd_path, merged)

        # process label (add annotator and save it in the labelling session)
        process_label(ref_stem, input_labels_dir, output_labels_dir, args.overwrite, args.raw_labels)

        # undistort and save new cameras in labelling session
        if args.with_cam:
            cam_keys = camera_map.keys()
            for out_cam_dir, cam_key in zip(output_camera_dirs, cam_keys):
                out_cam_path = out_cam_dir / f"{ref_stem}.jpg"
                if out_cam_path.exists() and not args.overwrite:
                    logger.warning("Skipping existing cam %s:" ,output_pcd_path)
                else:
                    write_camera(dataset_root, cam_key, row, intrinsics[camera_map[cam_key]], out_cam_dir)

    print("Done.")


if __name__ == "__main__":
    main()