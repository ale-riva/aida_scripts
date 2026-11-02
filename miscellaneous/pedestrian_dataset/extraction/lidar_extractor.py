"""
Reads PointCloud2 messages from the BAG for matched timestamps and saves
them as .pcd files in the Stage-1 intermediate folder structure.
Output path: <output_base_dir>/<bag_stem>/samples/lidarN/<YYYYMMDD_HHMMSS_ns>.pcd

All fields present in the PointCloud2 message (x, y, z, intensity, ring, ...)
are preserved in the output PCD using a binary PCD writer that reads directly
from the raw message buffer.
"""

from __future__ import annotations

import logging
import os
import struct

import numpy as np
from sensor_msgs.msg import PointField

from utils.timestamp_utils import ns_to_filename

logger = logging.getLogger(__name__)

_MS_TO_NS = 1_000_000

# PointField datatype -> (numpy dtype string, byte size, PCD type char)
_PF_TO_NP: dict = {
    PointField.INT8:    ("i1", 1, "I"),
    PointField.UINT8:   ("u1", 1, "U"),
    PointField.INT16:   ("i2", 2, "I"),
    PointField.UINT16:  ("u2", 2, "U"),
    PointField.INT32:   ("i4", 4, "I"),
    PointField.UINT32:  ("u4", 4, "U"),
    PointField.FLOAT32: ("f4", 4, "F"),
    PointField.FLOAT64: ("f8", 8, "F"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_lidar_frame(
    parser,
    topic: str,
    timestamp_ns: int,
    out_path: str,
    tolerance_ms: float,
    on_existing: str = "skip",
) -> bool:
    """
    Extract one PointCloud2 message from "parser" at "timestamp_ns" and write
    it to "out_path" as a binary .pcd file preserving ALL fields.

    Parameters
    parser        : MCAPBagFileParser | BagFileParser
    topic         : ROS topic string for this LiDAR
    timestamp_ns  : target timestamp in nanoseconds
    out_path      : full path of the output .pcd file
    tolerance_ms  : half-width of the fetch window in milliseconds
    on_existing   : "skip" | "overwrite" | "abort"

    Returns: bool  (True if written, False if skipped).
    """
    if os.path.exists(out_path):
        if on_existing == "skip":
            logger.debug("Skipping existing file: %s", out_path)
            return False
        elif on_existing == "abort":
            raise FileExistsError(f"Output file already exists: {out_path}")
        # overwrite: fall through

    half_window_ns = int(tolerance_ms * _MS_TO_NS)

    messages = parser.get_messages(
        topics=topic,
        start_time=timestamp_ns - half_window_ns,
        end_time=timestamp_ns + half_window_ns,
    )

    if not messages:
        logger.warning("No message found for topic %s at ts=%d", topic, timestamp_ns)
        return False

    _, cloud_msg = messages[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_pcd(cloud_msg, out_path)
    return True


def build_lidar_output_path(
    output_base_dir: str, bag_stem: str, lidar_key: str, timestamp_ns: int
) -> str:
    """
    Return the canonical .pcd output path for a given sample.
    Example:  .../2025-10-19/samples/lidar_front/20251019_112258_225748974.pcd
    """
    stem = ns_to_filename(timestamp_ns)
    return os.path.join(output_base_dir, bag_stem, "samples", lidar_key, f"{stem}.pcd")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_pcd(cloud_msg, out_path: str) -> None:
    """
    Write a PointCloud2 message to a binary PCD file, preserving every field
    (x, y, z, intensity, ring, timestamp, ...) with its original data type.

    The result is a valid PCD v0.7 binary file.
    """
    fields = sorted(cloud_msg.fields, key=lambda f: f.offset)
    num_points = cloud_msg.width * cloud_msg.height

    if num_points == 0:
        logger.warning("Empty point cloud - writing zero-point PCD: %s", out_path)

    # ------------------------------------------------------------------ #
    # Step 1: build a numpy dtype that mirrors the full point_step layout
    # (real fields + anonymous padding bytes for any gaps)
    # ------------------------------------------------------------------ #
    padded_dtype_list = []
    cursor = 0
    for i, field in enumerate(fields):
        if field.offset > cursor:
            gap = field.offset - cursor
            padded_dtype_list.append((f"_pad_{i}", "u1") if gap == 1 else (f"_pad_{i}", "u1", gap))
            cursor = field.offset
        np_type, byte_size, _ = _PF_TO_NP[field.datatype]
        if field.count > 1:
            padded_dtype_list.append((field.name, np_type, field.count))
        else:
            padded_dtype_list.append((field.name, np_type))
        cursor += byte_size * field.count
    if cursor < cloud_msg.point_step:
        tail = cloud_msg.point_step - cursor
        padded_dtype_list.append(("_pad_end", "u1") if tail == 1 else ("_pad_end", "u1", tail))

    padded_dtype = np.dtype(padded_dtype_list)

    # ------------------------------------------------------------------ #
    # Step 2: interpret the raw buffer (zero-copy view)
    # ------------------------------------------------------------------ #
    raw = bytes(cloud_msg.data)
    arr = np.frombuffer(raw, dtype=padded_dtype, count=num_points)

    # ------------------------------------------------------------------ #
    # Step 3: pack only the real fields (no padding) into a tight array
    # ------------------------------------------------------------------ #
    real_dtype_list = []
    for field in fields:
        np_type, _, _ = _PF_TO_NP[field.datatype]
        if field.count > 1:
            real_dtype_list.append((field.name, np_type, field.count))
        else:
            real_dtype_list.append((field.name, np_type))

    packed = np.zeros(num_points, dtype=np.dtype(real_dtype_list))
    for field in fields:
        packed[field.name] = arr[field.name]

    binary_data = packed.tobytes()

    # ------------------------------------------------------------------ #
    # Step 4: assemble and write the PCD header + binary payload
    # ------------------------------------------------------------------ #
    field_names  = []
    field_sizes  = []
    field_types  = []
    field_counts = []
    for field in fields:
        _, byte_size, pcd_type = _PF_TO_NP[field.datatype]
        field_names.append(field.name)
        field_sizes.append(str(byte_size))
        field_types.append(pcd_type)
        field_counts.append(str(field.count))

    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS "    + " ".join(field_names),
        "SIZE "      + " ".join(field_sizes),
        "TYPE "      + " ".join(field_types),
        "COUNT "     + " ".join(field_counts),
        f"WIDTH {num_points}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {num_points}",
        "DATA binary",
    ]
    header_bytes = ("\n".join(header_lines) + "\n").encode("ascii")

    with open(out_path, "wb") as fh:
        fh.write(header_bytes)
        fh.write(binary_data)

    logger.debug(
        "PCD written: %s  (%d pts, fields: %s)",
        out_path, num_points, " ".join(field_names),
    )