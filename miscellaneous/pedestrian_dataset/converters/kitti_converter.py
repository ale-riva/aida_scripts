"""
Stage-2 converter: intermediate format -> KITTI dataset layout.

Output structure
<output_dir>/
├── ImageSets/                # populated by the separate split script
├── training/
│   ├── image_2/              # one image per (frame, camera) sample
│   │   └── 000000.png  ...
│   ├── velodyne/             # merged point cloud (lidar_front + lidar_left + lidar_right)
│   │   └── 000000.bin  ...   #   float32 x y z intensity in the ring_center frame
│   ├── calib/
│   │   └── 000000.txt  ...   # per-sample calibration (active camera in P2)
│   └── label_2/
│       └── 000000.txt  ...   # KITTI-format labels for the sample's camera (empty when no annotations are present)
└── metadata/                 # sample_mapping.csv, kitti_metadata.yaml, labeled.txt, unlabeled.txt

Camera interleaving (e.g. 3 selected cameras camera_0, camera_1, camera_2):
    000000 -> camera_0, frame_0
    000001 -> camera_1, frame_0
    000002 -> camera_2, frame_0
    000003 -> camera_0, frame_1
    000004 -> camera_1, frame_1
    000005 -> camera_2, frame_1
    ...
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

from .base_converter import BaseConverter
from utils.image_utils import (
    optimal_undistorted_K as _optimal_undistorted_K,
    undistort_image       as _undistort_image,
    _AXIS_T,
)
from utils.pcd_reader import read_pcd, to_kitti_bin
from utils.shared_loader import _load_calibration, _load_timestamps_csv
from .kitti_label_converter import write_kitti_labels_for_sample
from .kitti_metadata_writer import SampleRecord, write_kitti_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample model
# ---------------------------------------------------------------------------

@dataclass
class _KittiSample:
    """One KITTI sample = one camera image for one LiDAR frame (CSV row)."""
    frame_index: int     # index of the source timestamps.csv row (one LiDAR frame)
    camera: object       # CameraConfig of this sample's camera
    row: dict            # the source timestamps.csv row


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_id(index: int) -> str:
    """Return a zero-padded 6-digit frame identifier, e.g. '000042'."""
    return f"{index:06d}"


def _build_projection_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Return the 3x4 KITTI projection matrix for a pinhole camera."""
    return np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0],
    ], dtype=np.float64)


def _build_tr_velo_to_cam(T_cam_to_ref: np.ndarray) -> np.ndarray:
    """
    Compute the 3x4 KITTI Tr_velo_to_cam matrix.

    Returns:(3, 4)  transform that maps ring_center points into the camera OPTICAL frame,
                    ready for pinhole projection (z_cam > 0 = in front of camera).
    """
    return (_AXIS_T @ np.linalg.inv(T_cam_to_ref))[:3, :]


def _mat_to_kitti_str(mat: np.ndarray) -> str:
    """Flatten a matrix row-major and return space-separated float strings."""
    return " ".join(f"{v:.8e}" for v in mat.flatten())


def _projection_for_camera(camera, intrinsics: dict) -> np.ndarray:
    """
    Build the 3x4 KITTI projection matrix for one camera.
    """
    cam_intr = intrinsics.get(camera.name)
    if not cam_intr:
        return np.zeros((3, 4), dtype=np.float64)

    K_new = _optimal_undistorted_K(cam_intr)
    return _build_projection_matrix(
        K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2],
    )


def _write_calib_file_single_cam(path: str, camera, calibration: dict) -> None:
    """
    Write one KITTI-format calibration .txt file for a single active camera.
    """
    intrinsics = calibration.get("intrinsics", {})
    extrinsics = calibration.get("extrinsics", {})

    zeros   = np.zeros((3, 4), dtype=np.float64)
    R0_rect = np.eye(3, dtype=np.float64)

    P2 = _projection_for_camera(camera, intrinsics)

    T_cam_to_ref = extrinsics.get(camera.frame_id)
    if T_cam_to_ref is not None:
        Tr = _build_tr_velo_to_cam(np.array(T_cam_to_ref, dtype=np.float64))
    else:
        logger.warning(
            "[KITTI] No extrinsic for camera frame '%s' - Tr_velo_to_cam zero-filled.",
            camera.frame_id,
        )
        Tr = zeros

    lines = [
        f"P0: {_mat_to_kitti_str(zeros)}",
        f"P1: {_mat_to_kitti_str(zeros)}",
        f"P2: {_mat_to_kitti_str(P2)}",
        f"P3: {_mat_to_kitti_str(zeros)}",
        f"R0_rect: {_mat_to_kitti_str(R0_rect)}",
        f"Tr_velo_to_cam: {_mat_to_kitti_str(Tr)}",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class KITTIConverter(BaseConverter):
    """
    Reads the Stage-1 intermediate folder and writes a KITTI-format dataset.

    Samples
    The selected cameras (config.kitti_selected_cameras, or all cameras when
    unset) are interleaved frame-by-frame. Each sample is one camera image plus
    its own calib, label and velodyne entry.

    Point cloud
    All LiDARs are merged into a single cloud expressed in the ring_center
    frame (aligned with center_of_gravity); every sensor is transformed via its
    extrinsic from calibration.yaml before concatenation. Samples that share a
    LiDAR frame point at a single real .bin via relative symlinks.

    Cameras
    - PINHOLE cameras: images are copied directly; P2 uses the original intrinsics.
    - FISHEYE cameras: images are undistorted to a pinhole-equivalent image
      (balance=0); P2 uses the corresponding K_new. Extrinsics are unchanged.

    Labels
    If a labels/ subdirectory is present, JSON annotations are converted to KITTI
    label_2 format against the sample's own camera. If labels/ is absent the
    label files are written empty.

    Metadata
    A metadata/ folder is written alongside training/, containing
    sample_mapping.csv (per-sample traceability back to Stage-1), labeled.txt
    / unlabeled.txt (split by label_2 content) and kitti_metadata.yaml
    (dataset-level summary). See kitti_metadata_writer.py.
    """

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def convert(self) -> None:
        logger.info("[KITTI] Starting conversion from: %s", self.stage1_dir)
        logger.info("[KITTI] Output directory:         %s", self.output_dir)

        calibration  = _load_calibration(self.stage1_dir)
        header, rows = _load_timestamps_csv(self.stage1_dir)

        if not rows:
            logger.warning("[KITTI] No sample groups in timestamps.csv - nothing to convert.")
            return

        # Selected cameras (CLI/config), preserving selection order. Falls back
        # to all configured cameras when no explicit selection was resolved.
        cameras = (
            getattr(self.config, "kitti_selected_cameras", None)
            or sorted(self.config.cameras, key=lambda c: c.key)
        )
        logger.info(
            "[KITTI] Selected cameras (%d): %s",
            len(cameras), [c.key for c in cameras],
        )

        # Configured LiDAR keys, sorted - the same set merged into velodyne/*.bin
        # (see _merge_lidars) and used as the per-sample timestamp columns in
        # metadata/sample_mapping.csv.
        lidar_keys = sorted(l.key for l in self.config.lidars)

        intrinsics = calibration.get("intrinsics", {})
        extrinsics = calibration.get("extrinsics", {})

        fisheye_cameras = [
            cam for cam in cameras
            if intrinsics.get(cam.name, {}).get("model_type", "PINHOLE").upper() == "FISHEYE"
        ]
        if fisheye_cameras:
            logger.info(
                "[KITTI] %d fisheye camera(s) will be undistorted: %s",
                len(fisheye_cameras), [c.name for c in fisheye_cameras],
            )

        has_labels = os.path.isdir(os.path.join(self.stage1_dir, "labels"))
        if has_labels:
            logger.info("[KITTI] Labels directory found - converting annotations.")
        else:
            logger.info("[KITTI] No labels directory found - label_2 files will be empty.")

        self._make_dirs()

        # Build the interleaved sample list (skips absent (camera, frame) pairs).
        samples = self._build_samples(rows, cameras)
        if not samples:
            logger.warning("[KITTI] No valid (camera, frame) samples - nothing to convert.")
            return

        # frame_index -> KITTI id whose velodyne/.bin holds the real merged cloud
        # for that frame. Later samples of the same frame symlink to it.
        frame_to_real_fid: dict[int, str] = {}
        n_symlinks = 0

        # One SampleRecord per generated sample, for metadata/sample_mapping.csv.
        metadata_records: list[SampleRecord] = []

        for idx, sample in enumerate(samples):
            fid = _frame_id(idx)

            # velodyne (.bin): first surviving sample of a frame writes the real
            # file; subsequent samples of that frame are relative symlinks.
            real_fid = frame_to_real_fid.get(sample.frame_index)
            if real_fid is None:
                self._write_velodyne(sample.row, fid, calibration)
                frame_to_real_fid[sample.frame_index] = fid
            else:
                self._symlink_velodyne(fid, real_fid)
                n_symlinks += 1

            # image_2 (.png): this sample's camera only (undistorted if fisheye).
            self._write_image(sample.row, sample.camera, fid, intrinsics)

            # calib (.txt): active camera in P2; P0/P1/P3 zero-filled.
            calib_path = os.path.join(self.output_dir, "training", "calib", f"{fid}.txt")
            _write_calib_file_single_cam(calib_path, sample.camera, calibration)

            # label_2 (.txt): computed against this sample's camera.
            is_labeled = write_kitti_labels_for_sample(
                fid         = fid,
                row         = sample.row,
                config      = self.config,
                stage1_dir  = self.stage1_dir,
                output_dir  = self.output_dir,
                extrinsics  = extrinsics,
                camera      = sample.camera,
                calibration = calibration,
            )

            metadata_records.append(SampleRecord(
                kitti_id            = fid,
                source_frame_index  = sample.frame_index,
                is_labeled          = is_labeled,
                camera_key          = sample.camera.key,
                camera_name         = sample.camera.name,
                camera_frame_id     = sample.camera.frame_id,
                camera_timestamp    = sample.row.get(sample.camera.key, "").strip(),
                lidar_timestamps    = {lk: sample.row.get(lk, "").strip() for lk in lidar_keys},
                velodyne_source_id  = frame_to_real_fid[sample.frame_index],
            ))

        logger.info(
            "[KITTI] Conversion complete: %d samples from %d frames "
            "(%d real velodyne clouds, %d symlinks).",
            len(samples), len(frame_to_real_fid),
            len(frame_to_real_fid), n_symlinks,
        )
        logger.info(
            "[KITTI] ImageSets/ is empty - run generate_splits.py to create "
            "train.txt / val.txt / trainval.txt."
        )

        write_kitti_metadata(
            config                  = self.config,
            stage1_dir              = self.stage1_dir,
            output_dir              = self.output_dir,
            cameras                 = cameras,
            lidar_keys              = lidar_keys,
            records                 = metadata_records,
            num_unique_lidar_frames = len(frame_to_real_fid),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_samples(self, rows: list[dict], cameras: list) -> list[_KittiSample]:
        """
        Interleave the selected cameras frame-by-frame into a flat sample list.

        For every frame (CSV row) and every selected camera (in selection
        order), a sample is appended when that camera has a non-empty timestamp.
        Samples with an absent timestamp are skipped (no placeholder).
        """
        samples: list[_KittiSample] = []
        skipped = 0
        for frame_index, row in enumerate(rows):
            for cam in cameras:
                ts = row.get(cam.key, "").strip()
                if not ts:
                    skipped += 1
                    continue
                samples.append(
                    _KittiSample(frame_index=frame_index, camera=cam, row=row)
                )
        if skipped:
            logger.info(
                "[KITTI] Skipped %d (camera, frame) samples with no timestamp.",
                skipped,
            )
        return samples

    def _make_dirs(self) -> None:
        """Create the standard KITTI output subdirectories."""
        subdirs = [
            os.path.join("training", "image_2"),
            os.path.join("training", "velodyne"),
            os.path.join("training", "calib"),
            os.path.join("training", "label_2"),
            "ImageSets",
        ]
        for sub in subdirs:
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)
    # TODO: adapt to common _merge_and_transform in pcd_reader.py
    def _merge_lidars(self, row: dict, calibration: dict) -> np.ndarray:
        """
        Read all LiDAR PCD files for this sample, transform every sensor into the ring_center frame using the extrinsic
        matrices from calibration.yaml, and return a single merged Nx4 float32 array (x, y, z, intensity).

        Intensity is normalised per-sensor by to_kitti_bin before merging.
        """
        extrinsics   = calibration.get("extrinsics", {})
        lidar_keys   = sorted(l.key for l in self.config.lidars)
        merged_parts: list[np.ndarray] = []

        for lidar_key in lidar_keys:
            ts_str = row.get(lidar_key, "").strip()
            if not ts_str:
                logger.warning("[KITTI] Missing timestamp for %s - skipping.", lidar_key)
                continue

            pcd_path = os.path.join(self.stage1_dir, "samples", lidar_key, f"{ts_str}.pcd")
            if not os.path.isfile(pcd_path):
                logger.warning("[KITTI] PCD not found: %s - skipping.", pcd_path)
                continue

            try:
                points_struct = read_pcd(pcd_path)
                points_kitti  = to_kitti_bin(points_struct)   # (N, 4) float32
            except Exception as exc:
                logger.error("[KITTI] Failed to read %s: %s", pcd_path, exc)
                continue

            # All sensors are transformed to ring_center.
            T_raw = extrinsics.get(lidar_key)
            if T_raw is None:
                logger.warning(
                    "[KITTI] No extrinsic found for %s - using points as-is.", lidar_key
                )
                merged_parts.append(points_kitti)
                continue

            T = np.array(T_raw, dtype=np.float64)             # (4, 4)
            xyz = points_kitti[:, :3].astype(np.float64)      # (N, 3)
            ones = np.ones((len(xyz), 1), dtype=np.float64)
            xyz_h = np.hstack([xyz, ones])                    # (N, 4)
            xyz_transformed = (T @ xyz_h.T).T[:, :3]          # (N, 3)

            transformed = np.hstack([
                xyz_transformed.astype(np.float32),
                points_kitti[:, 3:4],                         # intensity unchanged
            ])
            merged_parts.append(transformed)

        if not merged_parts:
            return np.zeros((0, 4), dtype=np.float32)

        return np.vstack(merged_parts).astype(np.float32)

    def _write_velodyne(self, row: dict, fid: str, calibration: dict) -> None:
        """
        Merge all LiDAR point clouds into a single cloud in the ring_center
        frame and write it as a KITTI binary .bin file.
        """
        out_path = os.path.join(self.output_dir, "training", "velodyne", f"{fid}.bin")
        merged   = self._merge_lidars(row, calibration)

        if len(merged) == 0:
            logger.warning("[KITTI] Frame %s: empty merged point cloud.", fid)

        if os.path.lexists(out_path):
            os.remove(out_path)
        merged.tofile(out_path)

    def _symlink_velodyne(self, fid: str, real_fid: str) -> None:
        """
        Create a relative symlink velodyne/<fid>.bin -> <real_fid>.bin so that
        camera samples sharing a LiDAR frame don't duplicate the cloud on disk.
        """
        velo_dir  = os.path.join(self.output_dir, "training", "velodyne")
        link_path = os.path.join(velo_dir, f"{fid}.bin")
        target    = f"{real_fid}.bin"   # relative target within the same dir

        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(target, link_path)

    def _write_image(self, row: dict, camera, fid: str, intrinsics: dict) -> None:
        """
        Write the image for one camera into training/image_2/<fid>.png.

        Images are undistorted (rectified) for every camera model (both PINHOLE
        and FISHEYE) so that image_2/, P2 and the 2-D boxes all share the same
        rectified image space. Undistortion preserves the original (width,
        height).
        """
        ts_str = row.get(camera.key, "").strip()
        if not ts_str:
            # Defensive: the sample list already excludes absent timestamps.
            return

        src = os.path.join(self.stage1_dir, "samples", camera.key, f"{ts_str}.png")
        dst = os.path.join(self.output_dir, "training", "image_2", f"{fid}.png")

        if not os.path.isfile(src):
            logger.warning(
                "[KITTI] Frame %s: image not found for %s at %s", fid, camera.key, src
            )
            return

        cam_intr = intrinsics.get(camera.name, {})

        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(
                "[KITTI] Frame %s: could not read image for %s at %s", fid, camera.key, src
            )
            return
        undistorted = _undistort_image(img, cam_intr)
        cv2.imwrite(dst, undistorted)