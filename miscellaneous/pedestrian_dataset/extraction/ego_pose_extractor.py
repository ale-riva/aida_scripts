"""
Loads ego pose data from a /tf topic for a specific parent -> child frame pair,
and resolves the closest pose for each matched sample group.

Source message type: tf2_msgs/msg/TFMessage
Each message carries a list of geometry_msgs/msg/TransformStamped.
Only transforms whose (frame_id, child_frame_id) match the configured pair are extracted.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_MS_TO_NS = 1_000_000


# ---------------------------------------------------------------------------
# Phase 1 - data loading  (called once per bag file)
# ---------------------------------------------------------------------------

def load_ego_poses_from_parser(
    parser,
    topic: str,
    parent_frame: str,
    child_frame: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read all TF messages from "parser" on "topic", extract every transform
    matching "parent_frame -> child_frame", and return them sorted by the
    transform header timestamp.

    Returns
    ts_arr   : "np.ndarray" of shape (N,), dtype int64  (nanoseconds since epoch)
    pose_arr : "np.ndarray" of shape (N, 7), dtype float64
               columns: [x, y, z, qx, qy, qz, qw]

    Both arrays are sorted by "ts_arr".  Returns empty arrays when no
    matching transforms are found.
    """
    # Use the bag-level publish timestamps as a bounding window for the query.
    all_ts = parser.get_message_timestamps(topic)
    if not all_ts:
        logger.warning("Topic '%s' has no messages in this parser - skipping.", topic)
        return np.array([], dtype=np.int64), np.zeros((0, 7), dtype=np.float64)

    messages = parser.get_messages(
        topics=topic,
        start_time=min(all_ts),
        end_time=max(all_ts),
    )

    ts_list: List[int] = []
    pose_list: List[List[float]] = []

    for _, tf_msg in messages:
        for transform in tf_msg.transforms:
            if (transform.header.frame_id == parent_frame and transform.child_frame_id == child_frame):
                stamp = transform.header.stamp
                ts_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                tr = transform.transform.translation
                rot = transform.transform.rotation
                ts_list.append(ts_ns)
                pose_list.append([tr.x, tr.y, tr.z, rot.x, rot.y, rot.z, rot.w])

    if not ts_list:
        logger.warning(
            "No transforms found for '%s' -> '%s' on topic '%s' in this parser.",
            parent_frame, child_frame, topic,
        )
        return np.array([], dtype=np.int64), np.zeros((0, 7), dtype=np.float64)

    ts_arr = np.array(ts_list, dtype=np.int64)
    pose_arr = np.array(pose_list, dtype=np.float64)
    order = np.argsort(ts_arr)

    logger.info(
        "Loaded %d ego poses ('%s' -> '%s') from topic '%s'.",
        len(ts_arr), parent_frame, child_frame, topic,
    )
    return ts_arr[order], pose_arr[order]


# ---------------------------------------------------------------------------
# Phase 2 - per-group lookup  (called once per matched group)
# ---------------------------------------------------------------------------

def find_ego_pose_at(
    ref_ts: int,
    ts_arr: np.ndarray,
    pose_arr: np.ndarray,
    tolerance_ms: float,
) -> Optional[Dict[str, float]]:
    """
    Return the ego pose closest to "ref_ts", preferably within "tolerance_ms".

    If the nearest available pose exceeds the tolerance window, a warning is
    emitted and the nearest pose is returned anyway (fallback).

    Parameters
    ref_ts       : reference LiDAR timestamp in nanoseconds
    ts_arr       : sorted array of ego pose timestamps (ns), shape (N,)
    pose_arr     : corresponding [x, y, z, qx, qy, qz, qw], shape (N, 7)
    tolerance_ms : desired maximum distance in milliseconds

    Returns: dict with keys "x, y, z, roll, pitch, yaw" or None
    """
    if len(ts_arr) == 0:
        logger.error(
            "Ego pose dataset is empty - cannot produce a pose for ref_ts=%d.",
            ref_ts,
        )
        return None

    idx   = int(np.argmin(np.abs(ts_arr - np.int64(ref_ts))))
    dt_ns = int(abs(int(ts_arr[idx]) - ref_ts))
    dt_ms = dt_ns / _MS_TO_NS

    if dt_ns > int(tolerance_ms * _MS_TO_NS):
        logger.warning(
            "Ego pose fallback: ref_ts=%d - nearest pose is %.2f ms away "
            "(tolerance=%.1f ms). Using nearest available pose.",
            ref_ts, dt_ms, tolerance_ms,
        )

    return _pose_dict_from_row(pose_arr[idx])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pose_dict_from_row(row: np.ndarray) -> Dict[str, float]:
    """
    Convert a [x, y, z, qx, qy, qz, qw] array to a pose dict that
    includes RPY angles derived from the quaternion.
    """
    x, y, z, qx, qy, qz, qw = row
    roll, pitch, yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz")
    return {
        "x":     float(x),
        "y":     float(y),
        "z":     float(z),
        "roll":  float(roll),
        "pitch": float(pitch),
        "yaw":   float(yaw),
    }