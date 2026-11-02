"""
Loaders for every on-disk file needed.

Covered artefacts
    Intermediate (Stage 1):
        timestamps.csv
        info_extraction.yaml
        ego_poses.csv
        calibration.yaml
        samples/<sensor>/<ts>.{pcd,png}
        labels/lidar/<ts>.json
        labels/<camera_key>/<ts>.json
    KITTI:
        training/calib/<id>.txt
        training/label_2/<id>.txt
        training/velodyne/<id>.bin
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Timestamp filename <-> nanoseconds  (mirrors utils.timestamp_utils; kept local
# so the validator does not depend on the pipeline package layout)
# ---------------------------------------------------------------------------

_NS_PER_SEC = 1_000_000_000
_TS_RE = re.compile(r"^(\d{8})_(\d{6})_(\d{9})$")


def filename_to_ns(stem: str) -> Optional[int]:
    """Parse a canonical timestamp stem (YYYYMMDD_HHMMSS_<9-digit ns>) to ns.
    Returns None if the stem is empty or malformed."""
    stem = (stem or "").strip()
    if not stem:
        return None
    m = _TS_RE.match(stem)
    if not m:
        return None
    date_str, time_str, nsub = m.group(1), m.group(2), m.group(3)
    dt = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    sec = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    return sec * _NS_PER_SEC + int(nsub)


def frame_id(idx: int) -> str:
    """Zero-padded 6-digit KITTI id, e.g. 42 -> '000042'."""
    return f"{idx:06d}"


# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------

def load_timestamps_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (header, rows). Each row maps column -> timestamp stem ('' if absent)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return header, rows


def load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_metadata(path: str) -> dict:
    return load_yaml(path)


def load_ego_poses_csv(path: str) -> List[Dict[str, Optional[float]]]:
    """
    Return a list of row dicts with keys: stamp, x, y, z, roll, pitch, yaw.
    Numeric cells are floats, or None when the cell is empty (pose unavailable).
    """
    out: List[Dict[str, Optional[float]]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            row: Dict[str, Optional[float]] = {"stamp": (r.get("stamp") or "").strip()}
            for k in ("x", "y", "z", "roll", "pitch", "yaw"):
                v = (r.get(k) or "").strip()
                row[k] = float(v) if v else None
            out.append(row)
    return out


def load_calibration(path: str) -> dict:
    """
    Load calibration.yaml and return:
        {
          "reference_frame": str,
          "extrinsics": {frame_id: np.ndarray(4,4)},   # T_sensor_to_reference
          "intrinsics": {camera_name: dict},           # fx,fy,cx,cy,width,height,model_type,...
        }
    """
    raw = load_yaml(path)
    extr_raw = raw.get("extrinsics", {}) or {}
    extrinsics = {k: np.array(v, dtype=np.float64) for k, v in extr_raw.items()}
    return {
        "reference_frame": raw.get("reference_frame"),
        "extrinsics": extrinsics,
        "intrinsics": raw.get("intrinsics", {}) or {},
    }


def load_labels_json(path: str) -> List[dict]:
    """Load a label JSON file (lidar or camera). Returns [] if absent/invalid."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def find_label_file(labels_subdir: str, ts_stem: str) -> Optional[str]:
    """
    Return labels_subdir/<ts_stem>.json if present, else None.
    (The pipeline names label files by the sensor timestamp stem.)
    """
    if not ts_stem:
        return None
    cand = os.path.join(labels_subdir, f"{ts_stem}.json")
    return cand if os.path.isfile(cand) else None


# ---------------------------------------------------------------------------
# KITTI artefacts
# ---------------------------------------------------------------------------

# KITTI label_2 column layout (15 fields)
KITTI_LABEL_FIELDS = (
    "type", "truncated", "occluded", "alpha",
    "x1", "y1", "x2", "y2",
    "h", "w", "l", "x", "y", "z", "ry",
)


def load_kitti_calib(path: str) -> Dict[str, np.ndarray]:
    """
    Parse a KITTI calib .txt into {key: flat float np.ndarray}.
    Keys typically: P0, P1, P2, P3, R0_rect, Tr_velo_to_cam.
    """
    out: Dict[str, np.ndarray] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            nums = [float(v) for v in vals.split()]
            out[key.strip()] = np.array(nums, dtype=np.float64)
    return out


def kitti_P(calib: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    """Return a 3x4 projection matrix for the given key, or None."""
    v = calib.get(key)
    if v is None or v.size != 12:
        return None
    return v.reshape(3, 4)


def kitti_Tr(calib: Dict[str, np.ndarray], key: str = "Tr_velo_to_cam") -> Optional[np.ndarray]:
    v = calib.get(key)
    if v is None or v.size != 12:
        return None
    return v.reshape(3, 4)


def load_kitti_labels(path: str) -> List[dict]:
    """Parse a KITTI label_2 .txt into a list of dicts keyed by KITTI_LABEL_FIELDS."""
    out: List[dict] = []
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 15:
                continue
            d: dict = {"type": parts[0]}
            for i, name in enumerate(KITTI_LABEL_FIELDS[1:], start=1):
                d[name] = float(parts[i])
            out.append(d)
    return out


def read_velodyne_bin(path: str) -> np.ndarray:
    """Read a KITTI velodyne .bin as an (N,4) float32 array (x,y,z,intensity)."""
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 4 != 0:
        # Return what we can; structural check reports the size violation.
        data = data[: data.size - (data.size % 4)]
    return data.reshape(-1, 4)


def list_dir(directory: str, ext: str) -> List[str]:
    """Sorted basenames in `directory` ending with `ext` (incl. symlinks)."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(ext))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image as an (H,W,3) uint8 RGB array, or None on failure."""
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Minimal binary PCD reader (xyz[+intensity]) — independent of utils.pcd_reader
# ---------------------------------------------------------------------------

_PCD_TYPE = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
             ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}


def read_pcd_xyzi(path: str) -> Optional[np.ndarray]:
    """
    Read a binary PCD (as written by the pipeline) and return an (N,4) float32
    array of x, y, z, intensity. intensity is 0 if the field is absent.
    Returns None if the file can't be parsed (ascii PCD or unexpected layout).
    """
    try:
        with open(path, "rb") as fh:
            fields: List[str] = []
            sizes: List[int] = []
            types: List[str] = []
            counts: List[int] = []
            num_points = 0
            data_fmt = None
            while True:
                line = fh.readline().decode("ascii", "replace").strip()
                if not line:
                    return None
                tok = line.split()
                key = tok[0].upper()
                if key == "FIELDS":
                    fields = tok[1:]
                elif key == "SIZE":
                    sizes = [int(x) for x in tok[1:]]
                elif key == "TYPE":
                    types = [x for x in tok[1:]]
                elif key == "COUNT":
                    counts = [int(x) for x in tok[1:]]
                elif key == "POINTS":
                    num_points = int(tok[1])
                elif key == "DATA":
                    data_fmt = tok[1].lower()
                    break
            if data_fmt != "binary" or not fields:
                return None
            if not counts:
                counts = [1] * len(fields)

            dtype_entries = []
            for name, sz, ty, cnt in zip(fields, sizes, types, counts):
                code = _PCD_TYPE.get((ty.upper(), sz))
                if code is None:
                    return None
                dtype_entries.append((name, code, cnt) if cnt > 1 else (name, code))
            dtype = np.dtype(dtype_entries)
            raw = fh.read(num_points * dtype.itemsize)
            arr = np.frombuffer(raw, dtype=dtype, count=num_points)

        names = arr.dtype.names
        if not all(a in names for a in ("x", "y", "z")):
            return None
        out = np.zeros((len(arr), 4), dtype=np.float32)
        out[:, 0] = arr["x"]; out[:, 1] = arr["y"]; out[:, 2] = arr["z"]
        if "intensity" in names:
            out[:, 3] = arr["intensity"]
        # Drop non-finite points (NaNs are common in raw LiDAR scans).
        finite = np.isfinite(out).all(axis=1)
        return out[finite]
    except Exception:
        return None