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
from operator import attrgetter

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
    header_timestamp_ns: int | None = None,
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
    header_timestamp_ns : require this exact header stamp at the recording timestamp

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
    exact = header_timestamp_ns is not None

    messages = parser.get_messages(
        topics=topic,
        start_time=timestamp_ns if exact else timestamp_ns - half_window_ns,
        end_time=timestamp_ns + 1 if exact else timestamp_ns + half_window_ns,
    )
    if exact:
        messages = [
            (ts, msg) for ts, msg in messages
            if ts == timestamp_ns and msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec == header_timestamp_ns
        ]

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

def read_cloud_points(cloud_msg) -> np.ndarray:
    """View all PointCloud2 fields, respecting offsets, row padding and byte order."""
    fields = sorted(cloud_msg.fields, key=attrgetter("offset"))
    byte_order = ">" if cloud_msg.is_bigendian else "<"
    formats = []
    for field in fields:
        numpy_type, _, _ = _PF_TO_NP[field.datatype]
        dtype = np.dtype(byte_order + numpy_type)
        if field.count > 1:
            formats.append((dtype, (field.count,)))
        else:
            formats.append(dtype)

    # Preserve the message layout so organized clouds with padded rows read correctly.
    dtype = np.dtype({
        "names": [field.name for field in fields],
        "formats": formats,
        "offsets": [field.offset for field in fields],
        "itemsize": cloud_msg.point_step,
    })
    return np.ndarray(
        shape=(cloud_msg.height, cloud_msg.width),
        dtype=dtype,
        buffer=bytes(cloud_msg.data),
        strides=(cloud_msg.row_step, cloud_msg.point_step),
    ).reshape(-1)


def _write_pcd(cloud_msg, out_path: str, point_indices=None) -> None:
    """
    Write a PointCloud2 message to a binary PCD file, preserving every field
    (x, y, z, intensity, ring, timestamp, ...) with its original data type.

    If point_indices is provided, write only that subset (indices or boolean mask).
    The result is a valid PCD v0.7 binary file, in little-endian byte order.
    """
    fields = sorted(cloud_msg.fields, key=attrgetter("offset"))
    points = read_cloud_points(cloud_msg)
    if point_indices is not None:
        points = points[point_indices]
    num_points = len(points)

    if num_points == 0:
        logger.warning("Empty point cloud - writing zero-point PCD: %s", out_path)

    # Build the packed layout and PCD header fields in the same order.
    packed_fields = []
    field_names = []
    field_sizes = []
    field_types = []
    field_counts = []
    for field in fields:
        numpy_type, byte_size, pcd_type = _PF_TO_NP[field.datatype]
        little_endian_type = "<" + numpy_type
        if field.count > 1:
            packed_fields.append((field.name, little_endian_type, field.count))
        else:
            packed_fields.append((field.name, little_endian_type))
        field_names.append(field.name)
        field_sizes.append(str(byte_size))
        field_types.append(pcd_type)
        field_counts.append(str(field.count))

    # Copy field values into contiguous little-endian storage without padding.
    packed = np.zeros(num_points, dtype=np.dtype(packed_fields))
    for name in field_names:
        packed[name] = points[name]

    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS " + " ".join(field_names),
        "SIZE " + " ".join(field_sizes),
        "TYPE " + " ".join(field_types),
        "COUNT " + " ".join(field_counts),
        f"WIDTH {num_points}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {num_points}",
        "DATA binary",
    ]
    header_bytes = ("\n".join(header_lines) + "\n").encode("ascii")

    with open(out_path, "wb") as fh:
        fh.write(header_bytes)
        fh.write(packed.tobytes())

    logger.debug(
        "PCD written: %s  (%d pts, fields: %s)",
        out_path, num_points, " ".join(field_names),
    )
