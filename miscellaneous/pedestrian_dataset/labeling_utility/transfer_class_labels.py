#!/usr/bin/env python3
r"""Transfer classes between matching SUSTechPOINTS 3-D label files.

Input folders contain JSON files directly; subfolders are not scanned. Files are
paired by filename and written directly into the output folder.
Boxes must use the same coordinate frame;
psr.rotation is in radians, using the annotator's Three.js XYZ Euler convention.
Only destination obj_type fields change; geometry, annotator and other fields stay.

Usage
    python transfer_class_labels.py --src-labels /path/to/source \
        --dst-labels /path/to/destination --out-dir /path/to/output \
        --iou-thr 0.5

Omit --out-dir to update destination files in place. Requires NumPy and SciPy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import combinations, product
import json
import logging
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_PLANE_TRIPLES = np.array(list(combinations(range(12), 3)))
_CORNER_SIGNS = np.array(list(product((-1, 1), repeat=3)))


# ---------------------------------------------------------------------------
# Oriented 3-D box geometry
# ---------------------------------------------------------------------------

def _box_geometry(label: dict):
    """Return box geometry, or None for a zero/negative-size placeholder."""
    psr = label["psr"]
    center = np.array([psr["position"][axis] for axis in "xyz"], dtype=np.float64)
    size = np.array([psr["scale"][axis] for axis in "xyz"], dtype=np.float64)
    angles = np.array([psr.get("rotation", {}).get(axis, 0.0) for axis in "xyz"], dtype=np.float64)
    if not all(np.isfinite(values).all() for values in (center, size, angles)):
        raise ValueError("Box coordinates, dimensions and rotations must be finite")
    if np.any(size <= 0):
        return None
    rotation = Rotation.from_euler("XYZ", angles).as_matrix()
    half_size = size / 2
    corners = (_CORNER_SIGNS * half_size) @ rotation.T + center
    return center, half_size, rotation, corners.min(axis=0), corners.max(axis=0), float(np.prod(size))


def _box_iou(src, dst) -> float:
    """Intersect two fully oriented boxes using their twelve bounding planes."""
    if src is None or dst is None:
        return 0.0
    src_center, src_half, src_rot, src_min, src_max, src_volume = src
    dst_center, dst_half, dst_rot, dst_min, dst_max, dst_volume = dst
    if np.any(np.minimum(src_max, dst_max) <= np.maximum(src_min, dst_min)):
        return 0.0

    # Work relative to the source center to avoid large world-coordinate offsets.
    src_normals = np.vstack((src_rot.T, -src_rot.T))
    dst_normals = np.vstack((dst_rot.T, -dst_rot.T))
    normals = np.vstack((src_normals, dst_normals))
    offsets = np.concatenate((
        np.tile(src_half, 2),
        np.tile(dst_half, 2) + dst_normals @ (dst_center - src_center),
    ))

    # Every vertex of the intersection lies on at least three bounding planes.
    matrices = normals[_PLANE_TRIPLES]
    independent = np.abs(np.linalg.det(matrices)) > 1e-10
    vertices = np.linalg.solve(
        matrices[independent], offsets[_PLANE_TRIPLES[independent]][..., None],
    )[..., 0]
    epsilon = 1e-9 * max(float(src_half.max()), float(dst_half.max()))
    vertices = vertices[np.all(vertices @ normals.T <= offsets + epsilon, axis=1)]
    if len(vertices) < 4:
        return 0.0
    try:
        intersection = float(ConvexHull(vertices).volume)
    except QhullError:
        return 0.0  # Touching faces or edges have no intersection volume.
    max_intersection = min(src_volume, dst_volume)
    if np.isclose(intersection, max_intersection, rtol=1e-10, atol=0):
        intersection = max_intersection
    intersection = min(intersection, max_intersection)
    return intersection / (src_volume + dst_volume - intersection)


def match_boxes(src_labels: list[dict], dst_labels: list[dict], iou_thr: float) -> list[tuple[int, int, float]]:
    """Greedily assign the highest remaining IoU; each box is used at most once."""
    if not 0 <= iou_thr <= 1:
        raise ValueError("iou_thr must be between 0 and 1")
    src_boxes = [_box_geometry(label) for label in src_labels]
    dst_boxes = [_box_geometry(label) for label in dst_labels]
    candidates = []
    for src_idx, src in enumerate(src_boxes):
        for dst_idx, dst in enumerate(dst_boxes):
            iou = _box_iou(src, dst)
            if iou > 0 and iou >= iou_thr:
                candidates.append((src_idx, dst_idx, iou))
    candidates.sort(key=lambda match: (-match[2], match[0], match[1]))

    matches = []
    used_src = set()
    used_dst = set()
    for src_idx, dst_idx, iou in candidates:
        if src_idx in used_src or dst_idx in used_dst:
            continue
        matches.append((src_idx, dst_idx, iou))
        used_src.add(src_idx)
        used_dst.add(dst_idx)
    return matches


# ---------------------------------------------------------------------------
# Label transfer and CLI
# ---------------------------------------------------------------------------

def _load_labels(path: Path) -> list[dict]:
    with path.open() as fh:
        labels = json.load(fh)
    if not isinstance(labels, list) or any(not isinstance(label, dict) for label in labels):
        raise ValueError(f"{path}: expected a list of label objects")
    return labels


def transfer_classes(src_labels: Path, dst_labels: Path, out_dir: Path | None = None,
                     iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Write updated destination labels and return (files, matched, unmatched)."""
    src_labels = Path(src_labels).expanduser().resolve()
    dst_labels = Path(dst_labels).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else dst_labels
    for directory in (src_labels, dst_labels):
        if not directory.is_dir():
            raise NotADirectoryError(f"Label directory not found: {directory}")
    if not 0 <= iou_thr <= 1:
        raise ValueError("iou_thr must be between 0 and 1")
    if out_dir == src_labels:
        raise ValueError("Output must not overwrite the source labels")
    paths = sorted(path for path in dst_labels.glob("*.json") if path.is_file())
    if not paths:
        raise ValueError(f"No JSON labels found in {dst_labels}")
    matched = unmatched = 0
    src_unmatched_by_class = Counter()
    for dst_path in paths:
        filename = dst_path.name
        src_path = src_labels / filename
        src = _load_labels(src_path) if src_path.is_file() else []
        if not src_path.is_file():
            logger.warning("No source labels for %s; destination classes will be Misc", filename)
        for index, label in enumerate(src):
            if not isinstance(label.get("obj_type"), str) or not label["obj_type"]:
                raise ValueError(f"{src_path}: label {index} has no class in obj_type")
        dst = _load_labels(dst_path)
        try:
            matches = match_boxes(src, dst, iou_thr)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid box in {src_path} or {dst_path}: {exc}") from exc

        updated = deepcopy(dst)
        for label in updated:
            label["obj_type"] = "Misc"
        for src_idx, dst_idx, _ in matches:
            updated[dst_idx]["obj_type"] = src[src_idx]["obj_type"]

        out_path = out_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Replace each file only after its complete JSON has been written.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=out_path.parent, suffix=".tmp", delete=False) as fh:
                temporary = Path(fh.name)
                json.dump(updated, fh, indent=2, allow_nan=False)
                fh.write("\n")
            os.replace(temporary, out_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        matched += len(matches)
        unmatched += len(dst) - len(matches)
        matched_src = {src_idx for src_idx, _, _ in matches}
        src_unmatched_by_class.update(
            label["obj_type"] for src_idx, label in enumerate(src) if src_idx not in matched_src
        )
        logger.info("%s: %d matched, %d Misc", filename, len(matches), len(dst) - len(matches))
    destination_names = {path.name for path in paths}
    for src_path in src_labels.glob("*.json"):
        if src_path.is_file() and src_path.name not in destination_names:
            src_unmatched_by_class.update(label["obj_type"] for label in _load_labels(src_path))
    class_counts = ", ".join(
        f"{class_name}: {count}" for class_name, count in sorted(src_unmatched_by_class.items())
    )
    logger.info(
        "Matching summary:\n"
        "  Source labels not matched: %d (%s)\n"
        "  Destination labels not matched: %d\n"
        "  Destination labels matched: %d",
        sum(src_unmatched_by_class.values()), class_counts, unmatched, matched,
    )
    return len(paths), matched, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-labels", "--src_labels", type=Path, required=True, help="Folder directly containing source JSON files")
    parser.add_argument("--dst-labels", "--dst_labels", type=Path, required=True, help="Folder directly containing destination JSON files")
    parser.add_argument("--out-dir", "--out_dir", type=Path, help="Write JSON files directly here; omitted means update destination in place")
    parser.add_argument("--iou-thr", "--iou_thr", type=float, default=0.5, help="Minimum 3-D IoU, inclusive (default: 0.5)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        files, matched, unmatched = transfer_classes(args.src_labels, args.dst_labels, args.out_dir, args.iou_thr)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    logger.info("Updated %d files: %d classes transferred, %d unmatched boxes set to Misc", files, matched, unmatched)


if __name__ == "__main__":
    main()
