"""
Binary PCD v0.7 reader.

Reads the files written by extraction/lidar_extractor.py and returns the
point data as a NumPy structured array, preserving all fields (x, y, z, intensity, ring, timestamp, ...).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from typing import List

import logging
logger = logging.getLogger(__name__)
import os

# Map PCD TYPE + SIZE characters/bytes to NumPy dtype codes
_PCD_TYPE_TO_NP = {
    ("F", 4): "f4",
    ("F", 8): "f8",
    ("I", 1): "i1",
    ("I", 2): "i2",
    ("I", 4): "i4",
    ("I", 8): "i8",
    ("U", 1): "u1",
    ("U", 2): "u2",
    ("U", 4): "u4",
    ("U", 8): "u8",
}


def read_pcd(path: str) -> np.ndarray:
    """
    Read a binary PCD v0.7 file produced by lidar_extractor.
    """
    with open(path, "rb") as fh:
        fields:  list[str] = []
        sizes:   list[int] = []
        types:   list[str] = []
        counts:  list[int] = []
        num_points = 0

        # Parse ASCII header line by line until "DATA binary"
        while True:
            raw_line = fh.readline()
            if not raw_line:
                raise ValueError(f"Reached EOF before DATA line in: {path}")
            line = raw_line.decode("ascii", errors="replace").strip()

            if line.startswith("#") or not line:
                continue
            key = line.split()[0].upper()

            if key == "FIELDS":
                fields = line.split()[1:]
            elif key == "SIZE":
                sizes = [int(x) for x in line.split()[1:]]
            elif key == "TYPE":
                types = line.split()[1:]
            elif key == "COUNT":
                counts = [int(x) for x in line.split()[1:]]
            elif key == "POINTS":
                num_points = int(line.split()[1])
            elif key == "DATA":
                data_type = line.split()[1].lower()
                break   # binary data starts immediately after this line

        if data_type != "binary":
            raise ValueError(
                f"Only 'DATA binary' PCD files are supported; got 'DATA {data_type}' in {path}"
            )

        if not fields:
            raise ValueError(f"No FIELDS line found in PCD header: {path}")

        # Build a NumPy structured dtype for the packed (non-padded) layout
        dtype_entries = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            np_type = _PCD_TYPE_TO_NP.get((typ.upper(), size))
            if np_type is None:
                raise ValueError(
                    f"Unsupported PCD field type='{typ}' size={size} in {path}"
                )
            if count > 1:
                dtype_entries.append((name, np_type, count))
            else:
                dtype_entries.append((name, np_type))

        dtype = np.dtype(dtype_entries)
        raw_bytes = fh.read(num_points * dtype.itemsize)
        return np.frombuffer(raw_bytes, dtype=dtype, count=num_points)

def write_xyz_pcd_binary(out_path: str, points: np.ndarray) -> None:
    """
    Write a binary PCD file containing only x, y, z fields.

    Any extra input fields are discarded.

    Output format:
        FIELDS x y z
        SIZE   4 4 4
        TYPE   F F F
        COUNT  1 1 1
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2:
        raise ValueError(f"points must be a 2D array, got shape {points.shape}")

    if points.shape[1] < 3:
        raise ValueError(
            f"points must have at least 3 columns for x, y, z, got shape {points.shape}"
        )

    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float32)

    n_points = xyz.shape[0]

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n_points}\n"
        "DATA binary\n"
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(xyz.tobytes(order="C"))


def to_kitti_bin(points: np.ndarray) -> np.ndarray:
    """
    Convert a structured point array (from read_pcd) to the KITTI binary format:
    Nx4 float32 array with columns [x, y, z, intensity].

    Intensity is normalised to [0, 1] if its range exceeds 1.0 (e.g. when the LiDAR reports raw counts in 0-255 or 0-65535).
    """
    n   = len(points)
    out = np.zeros((n, 4), dtype=np.float32)

    for src, dst in (("x", 0), ("y", 1), ("z", 2)):
        if src in points.dtype.names:
            out[:, dst] = points[src].astype(np.float32)

    if "intensity" in points.dtype.names:
        intensity = points["intensity"].astype(np.float32)
        max_val   = float(intensity.max())
        if max_val > 1.0:
            intensity = intensity / max_val
        out[:, 3] = intensity

    return out


def to_nuscenes_bin(points: np.ndarray) -> np.ndarray:
    """
    Convert a structured point array (from read_pcd) to the nuScenes binary format:
    Nx5 float32 array with columns [x, y, z, intensity, ring].

    Intensity is normalised to [0, 1] using the same logic as to_kitti_bin.

    ring is read from the 'ring' field of the PCD structured array. The column
    is filled with zeros if the field is not present (unstructured cloud).
    """
    n   = len(points)
    out = np.zeros((n, 5), dtype=np.float32)

    for src, dst in (("x", 0), ("y", 1), ("z", 2)):
        if src in points.dtype.names:
            out[:, dst] = points[src].astype(np.float32)

    if "intensity" in points.dtype.names:
        intensity = points["intensity"].astype(np.float32)
        max_val   = float(intensity.max())
        if max_val > 1.0:
            intensity = intensity / max_val
        out[:, 3] = intensity

    if "ring" in points.dtype.names:
        out[:, 4] = points["ring"].astype(np.float32)

    return out

def to_xyz_bin(points: np.ndarray) -> np.ndarray:
    """
    Convert a structured point array (from read_pcd) to an XYZ-only format:
    Nx3 float32 array with columns [x, y, z].

    Missing fields are filled with zeros.
    """
    n = len(points)
    out = np.zeros((n, 3), dtype=np.float32)

    for src, dst in (("x", 0), ("y", 1), ("z", 2)):
        if src in points.dtype.names:
            out[:, dst] = points[src].astype(np.float32)

    return out

def get_num_points_in_bbox(pcl_path: str, center: List[float], extent: List[float], rpy:List[float], n_fields_per_point:int):

    # Check file extension and load accordingly
    if isinstance(pcl_path, str):
        if pcl_path.endswith('.bin'):
            xyzi = np.fromfile(pcl_path, dtype=np.float32).reshape(-1,n_fields_per_point)
            pcd_cloud = o3d.geometry.PointCloud()
            pcd_cloud.points = o3d.utility.Vector3dVector(xyzi[:, :3])
        elif pcl_path.endswith('.pcd'):
            pcd_cloud = o3d.io.read_point_cloud(pcl_path)
        else:
            raise ValueError(f"Unsupported file format: {pcl_path}. Must be .bin or .pcd")
    else:
        # Assume it's already a PointCloud object
        pcd_cloud = pcl_path

    R = o3d.geometry.get_rotation_matrix_from_xyz(np.array(rpy))
    obb = o3d.geometry.OrientedBoundingBox(np.array(center), R, np.array(extent))
    
    # Get indices of points inside the bounding box
    indices = obb.get_point_indices_within_bounding_box(pcd_cloud.points)
    
    # Count points inside
    num_points = len(indices)

    return num_points

def merge_and_transform(row: dict, lidar_keys: List[str], stage1_dir: str, extrinsics: dict,
                        point_reader:function) -> np.ndarray:
    """
    Read every LiDAR PCD, transform each cloud into the common reference frame using the
    extrinsic matrices from calibration.yaml, and concatenate them.

    The first 3 columns are assumed to be xyz.
    Any additional columns are preserved unchanged.
    """

    merged_parts: List[np.ndarray] = []
    n_features = 0

    for lidar_key in lidar_keys:
        ts_str = row.get(lidar_key, "").strip()
        if not ts_str:
            logger.warning("[PcdReader] Missing timestamp for %s - skipping sensor.", lidar_key)
            continue

        pcd_path = os.path.join(stage1_dir, "samples", lidar_key, f"{ts_str}.pcd")
        if not os.path.isfile(pcd_path):
            logger.warning("[PcdReader] PCD file not found: %s - skipping.", pcd_path)
            continue

        try:
            points_struct = read_pcd(pcd_path)
            point = point_reader(points_struct)  # (N, F), first 3 columns must be x y z
            point = np.asarray(point, dtype=np.float32)

            if point.ndim != 2 or point.shape[1] < 3:
                raise ValueError(
                    f"Expected point cloud with shape (N, F) and F >= 3, got {point.shape}"
                )

            n_features = point.shape[1]

        except Exception as exc:
            logger.error("[PcdReader] Failed to read %s: %s", pcd_path, exc)
            continue

        # Apply extrinsic transform to bring xyz into the reference frame.
        # Extra features after xyz are preserved unchanged.
        T_raw = extrinsics.get(lidar_key)

        if T_raw is not None:
            T = np.array(T_raw, dtype=np.float64)  # (4, 4)

            xyz = point[:, :3].astype(np.float64)  # (N, 3)
            xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)])  # (N, 4)
            xyz_t = (T @ xyz_h.T).T[:, :3].astype(np.float32)  # (N, 3)
            extra_features = point[:, 3:]  # (N, F-3), can be empty
            point = np.hstack([xyz_t, extra_features])

        else:
            logger.warning(
                "[PcdReader] No extrinsic found for '%s' - using sensor-frame coordinates.",
                lidar_key,
            )

        merged_parts.append(point.astype(np.float32, copy=False))

    if not merged_parts:
        return np.zeros((0, n_features), dtype=np.float32)

    return np.vstack(merged_parts)