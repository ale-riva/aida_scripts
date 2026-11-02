"""
Implements the camera-LiDAR timestamp matching algorithm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MS_TO_NS = 1_000_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SampleGroup:
    """
    One matched sample group: one timestamp per sensor.
    A None value means the sensor was absent (optional camera slot).
    """
    lidar_timestamps: Dict[str, int]             # configured LiDAR keys, in settings order
    camera_timestamps: Dict[str, Optional[int]]  # {"camera_0": ts_ns, ..., "camera_6": None}


# ---------------------------------------------------------------------------
# Low-level helpers  (reused from 00_get_preannotated_samples.py)
# ---------------------------------------------------------------------------

def _find_closest_idx(target: int, array: np.ndarray) -> int:
    """Return the index of the element in *array* closest to *target*."""
    return int(np.argmin(np.abs(array - target)))


def _find_closest_within(target: int, array: np.ndarray, tolerance_ms: float) -> Optional[int]:
    """
    Return the index of the closest element within (+-)tolerance_ms, or None.
    Adapted from polimove_tools `find_closest_idx_saturated`.
    """
    idx = _find_closest_idx(target, array)
    if abs(int(target) - int(array[idx])) <= int(tolerance_ms * _MS_TO_NS):
        return idx
    return None


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------

def match_timestamps(
    lidar_timestamps: Dict[str, List[int]],
    camera_timestamps: Dict[str, List[int]],
    tolerance_ms: float,
    sampling_interval_ms: float,
) -> List[SampleGroup]:
    """
    Produce a list of temporally-aligned sample groups sampled at a fixed interval.

    Algorithm:
    1. Use the first configured LiDAR's timestamps as the reference.
    2. Starting from its first timestamp, search for a valid group:
       a. For every present camera, find the closest timestamp within (+-)tolerance_ms.
       b. For every other LiDAR, find the closest timestamp within (+-)tolerance_ms.
       c. If ALL present sensors have a match -> record the group.
       d. Otherwise advance to the next reference timestamp and retry.
    3. Once a valid group is found, jump forward by "sampling_interval_ms" and repeat
       from the nearest reference timestamp at or after that point.

    Returns: List[SampleGroup]
    """
    lidar_keys = list(lidar_timestamps)  # preserve settings order; index 0 is the reference
    camera_keys = sorted(camera_timestamps.keys())  # ["camera_0", ..., "camera_6"]

    if not lidar_keys:
        logger.warning("No LiDARs configured. Returning empty list.")
        return []
    empty_lidars = [key for key in lidar_keys if not len(lidar_timestamps[key])]
    if empty_lidars:
        logger.warning("Configured LiDAR(s) have no timestamps: %s. Returning empty list.", empty_lidars)
        return []

    ref_key = lidar_keys[0]
    ref_ts = np.array(lidar_timestamps[ref_key], dtype=np.int64)

    # Pre-convert all other sensors to numpy arrays for fast lookup
    other_lidar_arrays = {k: np.array(lidar_timestamps[k],  dtype=np.int64) for k in lidar_keys[1:]}
    camera_arrays = {k: np.array(camera_timestamps[k], dtype=np.int64) for k in camera_keys if len(camera_timestamps[k]) > 0}
    absent_cameras = {k for k in camera_keys if len(camera_timestamps.get(k, [])) == 0}
    
    groups: List[SampleGroup] = [] # the output list of matched groups to return
    used_ref_idx: set = set()      # guard against timestamp reuse

    ref_idx = 0                       # current position in ref_ts
    next_target: Optional[int] = None # minimum timestamp (ns) for the next group

    skipped = 0 # diagnostic counter

    while ref_idx < len(ref_ts):

        # Skip timestamps that are before the next sampling target
        if next_target is not None and ref_ts[ref_idx] < next_target:
            ref_idx += 1
            continue

        # Advance past already-used reference timestamps
        if ref_idx in used_ref_idx:
            ref_idx += 1
            continue

        ref_t = int(ref_ts[ref_idx]) # current reference timestamp, in ns

        ## Try to match every present camera
        cam_match: Dict[str, Optional[int]] = {}
        valid = True

        for cam_key in camera_keys:
            if cam_key in absent_cameras:
                cam_match[cam_key] = None      # mark absent cameras with None
                continue
            arr  = camera_arrays[cam_key]
            cidx = _find_closest_within(ref_t, arr, tolerance_ms)
            if cidx is None:
                valid = False
                break
            cam_match[cam_key] = int(arr[cidx])

        if not valid:
            skipped += 1
            ref_idx += 1
            continue

        ## Match the other LiDARs (after cameras because they are less likely to be missing)
        lidar_match: Dict[str, int] = {ref_key: ref_t}

        for ldr_key in lidar_keys[1:]:
            arr  = other_lidar_arrays[ldr_key]
            lidx = _find_closest_within(ref_t, arr, tolerance_ms)
            if lidx is None:
                valid = False
                break
            lidar_match[ldr_key] = int(arr[lidx])

        if not valid:
            skipped += 1
            ref_idx += 1
            continue

        ## Valid group found
        groups.append(SampleGroup(lidar_timestamps=lidar_match, camera_timestamps=cam_match))
        used_ref_idx.add(ref_idx)

        if skipped > 0:
            logger.debug("Skipped %d LiDAR timestamp(s) before finding group #%d.", skipped, len(groups))
            skipped = 0

        # Jump forward by sampling_interval_ms
        next_target = ref_t + int(sampling_interval_ms * _MS_TO_NS)
        ref_idx += 1

    logger.info("Matching complete: %d valid sample groups found.", len(groups))
    return groups