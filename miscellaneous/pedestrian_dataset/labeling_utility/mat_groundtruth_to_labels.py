#!/usr/bin/env python3
"""
Converts a MATLAB Video Labeler `groundTruth` export (.mat file) into the
camera-annotation JSON format expected by the bag_to_fusion_dataset pipeline's
`labels/camera_N/<timestamp>.json` files.

Output
------
For every image referenced in the .mat file, a JSON file named
`<timestamp_stem>.json` is written to `<labels_dir>/camera_0/`, where
`<timestamp_stem>` is extracted from the image filename and must match the
`YYYYMMDD_HHMMSS_ns` stem used elsewhere in the pipeline.

Each JSON file is a list of annotation objects following the project's
camera annotation schema.

Conventions applied in this conversion
------------------------------------------------------------------
1. Coordinates are converted from MATLAB's 1-based pixel indexing to
   0-based (x -> x - 1, y -> y - 1) to match the Python/OpenCV convention
   used by the rest of the pipeline.
2. Frames with multiple detected objects produce multiple entries in the
   output list, with obj_id "0", "1", "2", ... in detection order.
3. Frames with no detection still produce a JSON file, containing a single
   annotation entry with position/width/height all zeroed (matching the
   "empty-scene" convention already recognised by
   kitti_label_converter._build_2d_box_from_camera_ann, which treats a
   width=0/height=0 box as "no object" and skips it downstream).

Usage
-----
    python3 mat_groundtruth_to_labels.py \\
        --mat-file /path/to/<mat_file_name>/test_with_few_samples.mat \\
        --labels-dir /path/to/<bag_folder_name>/labels \\
        --camera-key camera_0 \\
        --image-width 1928 \\
        --image-height 500
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
from typing import List, Optional

import numpy as np
import scipy.io as sio
import scipy.io.matlab._mio5 as mio5

logger = logging.getLogger("mat_groundtruth_to_labels")

_TS_RE = re.compile(r"(\d{8}_\d{6}_\d{9})")


# ---------------------------------------------------------------------------
# Low-level MCOS subsystem parsing
# ---------------------------------------------------------------------------

def _load_mcos_subsystem(mat_file_path: str):
    """
    Extract and parse the MCOS subsystem blob embedded in a .mat file's
    `__function_workspace__` variable.

    The blob has its own internal mini mat5-stream layout:
        [0:4]  version + endian marker ("\\x00\\x01IM")
        [4:8]  4 reserved/padding bytes
        [8:]   a single miMATRIX element (the "FileWrapper__" struct)

    To reuse scipy's mat5 reader, we wrap this in a synthetic 128-byte mat5
    file header (116-byte text + 8-byte subsystem-offset placeholder) and
    feed it back into scipy's low-level VarReader5, forcing `process=True`
    so the anonymous (nameless) top-level variable is fully decoded instead
    of being left as raw bytes.

    Returns
    -------
    (top_level_dict, subsystem_array)
        top_level_dict    : the normal scipy.io.loadmat() result for the
                             original file (contains the opaque MCOS handle).
        subsystem_array   : the decoded FileWrapper__ struct array.
    """
    top = sio.loadmat(mat_file_path, simplify_cells=False)
    if "__function_workspace__" not in top:
        raise ValueError(
            f"'{mat_file_path}' does not contain a __function_workspace__ "
            "subsystem - it does not look like a MATLAB object (MCOS) export."
        )

    fw_bytes = top["__function_workspace__"].tobytes()

    fake_header_text = b"MATLAB 5.0 MAT-file, synthetic header for subsystem parse".ljust(116, b" ")
    fake_subsys_offset = b"\x00" * 8
    version_endian = b"\x00\x01IM"
    reconstructed = fake_header_text + fake_subsys_offset + version_endian + fw_bytes[8:]

    buf = io.BytesIO(reconstructed)
    reader = mio5.MatFile5Reader(buf, struct_as_record=True)
    reader.mat_stream.seek(0)
    reader.initialize_read()
    reader.read_file_header()
    hdr, _ = reader.read_var_header()
    subsystem_array = reader.read_var_array(hdr, process=True)

    return top, subsystem_array


def _find_ground_truth_struct(subsystem_array: np.ndarray) -> np.ndarray:
    """
    Locate the `groundTruth` struct (fields: DataSource, LabelDefinitions,
    LabelData, PolygonOrder, Version) inside the FileWrapper__ cell array.

    MATLAB stores this array of object property-cells with the actual class
    instance struct(s) appearing after a run of small scalar/placeholder
    cells. We scan for the first cell whose dtype names match the expected
    groundTruth field set, rather than hard-coding its index, so this keeps
    working if MATLAB changes the surrounding layout in a future release.
    """
    fw_struct = subsystem_array[0, 0]
    mcos_tuple = fw_struct["MCOS"][0]

    field_names = mcos_tuple.dtype.names or ()
    cell_array = None
    for name in field_names:
        candidate = mcos_tuple[name]
        if (
            isinstance(candidate, np.ndarray)
            and candidate.dtype == object
            and candidate.ndim == 2
            and candidate.shape[1] == 1
            and candidate.shape[0] > 0
        ):
            cell_array = candidate
            break

    if cell_array is None:
        raise ValueError(
            "Could not locate the MCOS object-property cell array inside the "
            f"parsed record (fields found: {field_names}). This likely means "
            "scipy's internal MAT5 parsing (scipy.io.matlab._mio5) changed "
            "again in a way this script doesn't yet handle."
        )

    expected_fields = {"DataSource", "LabelDefinitions", "LabelData"}

    for i in range(cell_array.shape[0]):
        cell = cell_array[i, 0]
        names = getattr(getattr(cell, "dtype", None), "names", None)
        if names and expected_fields.issubset(set(names)):
            return cell

    raise ValueError(
        "Could not locate a groundTruth struct (DataSource/LabelDefinitions/"
        "LabelData fields) inside the .mat file's MCOS subsystem. "
        "The file may not be a Video Labeler groundTruth export, or "
        "MATLAB's internal serialization layout has changed."
    )


# ---------------------------------------------------------------------------
# groundTruth field extraction
# ---------------------------------------------------------------------------

def _extract_image_paths(gt_struct: np.ndarray) -> List[str]:
    """Return the list of image file paths in DataSource.Source, in row order."""
    datasource = gt_struct["DataSource"][0, 0]
    source = datasource["Source"][0, 0]  # (N, 1) object array of 1-element string arrays
    paths = []
    for i in range(source.shape[0]):
        cell = source[i, 0]
        paths.append(str(np.asarray(cell).reshape(-1)[0]))
    return paths


def _extract_label_names(gt_struct: np.ndarray) -> List[str]:
    """Return the list of class names from LabelDefinitions.Name."""
    label_defs = gt_struct["LabelDefinitions"][0, 0]
    name_col = label_defs["Name"]
    names = []
    # name_col shape is (1, n_classes); each cell holds a 1-element string array.
    for i in range(name_col.shape[1] if name_col.ndim > 1 else name_col.shape[0]):
        cell = name_col[0, i] if name_col.ndim > 1 else name_col[i, 0]
        names.append(str(np.asarray(cell).reshape(-1)[0]))
    return names


def _extract_boxes_for_class(gt_struct: np.ndarray, class_name: str) -> List[np.ndarray]:
    """
    Return a list (one entry per frame/row) of Nx4 arrays of [x, y, w, h]
    boxes (MATLAB 1-based pixel convention) for the given class.
    An empty frame yields a (0, 4)-shaped array.
    """
    label_data = gt_struct["LabelData"][0, 0]
    if class_name not in label_data.dtype.names:
        raise ValueError(
            f"Class '{class_name}' not found in LabelData columns: "
            f"{label_data.dtype.names}"
        )

    column = label_data[class_name]
    n_rows = column.shape[0]
    boxes_per_frame: List[np.ndarray] = []
    for i in range(n_rows):
        cell = column[i, 0]
        arr = np.asarray(cell, dtype=float)
        if arr.size == 0:
            boxes_per_frame.append(np.zeros((0, 4), dtype=float))
        else:
            boxes_per_frame.append(arr.reshape(-1, 4))
    return boxes_per_frame


# ---------------------------------------------------------------------------
# Timestamp / filename handling
# ---------------------------------------------------------------------------

def _extract_timestamp_stem(image_path: str) -> str:
    """
    Extract the canonical `YYYYMMDD_HHMMSS_ns` stem from an image file path
    (handles both Windows and POSIX separators).
    """
    basename = image_path.replace("\\", "/").rsplit("/", 1)[-1]
    stem, _ = os.path.splitext(basename)
    match = _TS_RE.search(stem)
    if match:
        return match.group(1)
    return stem  # fall back to the bare filename stem if pattern doesn't match


# ---------------------------------------------------------------------------
# Annotation JSON construction
# ---------------------------------------------------------------------------

def _build_annotation_list(
    boxes: np.ndarray,
    class_name: str,
    image_width: int,
    image_height: int,
    one_based_to_zero_based: bool = True,
) -> List[dict]:
    """
    Build the list of annotation dicts for a single frame.

    If `boxes` has zero rows, a single zeroed "empty-scene" annotation is
    returned (position/width/height all 0) so that every frame still
    produces a JSON file with a consistent structure.
    """
    if boxes.shape[0] == 0:
        return [
            {
                "obj_id": "0",
                "obj_type": class_name,
                "bbox": {
                    "position": {"x": 0, "y": 0},
                    "width": 0,
                    "height": 0,
                    "rotation": 0,
                },
                "image_width": image_width,
                "image_height": image_height,
            }
        ]

    annotations = []
    for obj_idx, (x, y, w, h) in enumerate(boxes):
        if one_based_to_zero_based:
            x = x - 1
            y = y - 1
        annotations.append(
            {
                "obj_id": str(obj_idx),
                "obj_type": class_name,
                "bbox": {
                    "position": {"x": int(x), "y": int(y)},
                    "width": int(w),
                    "height": int(h),
                    "rotation": 0,
                },
                "image_width": image_width,
                "image_height": image_height,
            }
        )
    return annotations


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def convert(
    mat_file_path: str,
    labels_dir: str,
    camera_key: str = "camera_0",
    image_width: int = 1928,
    image_height: int = 500,
    class_name: Optional[str] = None,
) -> None:
    """
    Convert a groundTruth .mat file into per-frame camera annotation JSON
    files under `<labels_dir>/<camera_key>/`.

    Parameters
    ----------
    mat_file_path : path to the .mat file exported from MATLAB Video Labeler.
    labels_dir    : path to the `labels/` directory inside the Stage-1
                    intermediate output folder (created if missing).
    camera_key    : sensor key the annotations belong to (e.g. "camera_0").
    image_width   : pixel width of the annotated images (used in output JSON).
    image_height  : pixel height of the annotated images (used in output JSON).
    class_name    : which LabelDefinitions class to convert. If None and the
                    .mat file contains exactly one class, that class is used
                    automatically; otherwise an error is raised.
    """
    logger.info("Reading .mat file: %s", mat_file_path)
    _, subsystem_array = _load_mcos_subsystem(mat_file_path)
    gt_struct = _find_ground_truth_struct(subsystem_array)

    image_paths = _extract_image_paths(gt_struct)
    label_names = _extract_label_names(gt_struct)
    logger.info("Found %d frames, %d label class(es): %s", len(image_paths), len(label_names), label_names)

    if class_name is None:
        if len(label_names) != 1:
            raise ValueError(
                f"Multiple label classes found {label_names}; "
                "please specify --class-name explicitly."
            )
        class_name = label_names[0]

    boxes_per_frame = _extract_boxes_for_class(gt_struct, class_name)

    if len(boxes_per_frame) != len(image_paths):
        raise ValueError(
            f"Row count mismatch: {len(image_paths)} image paths vs "
            f"{len(boxes_per_frame)} LabelData rows for class '{class_name}'."
        )

    out_dir = os.path.join(labels_dir, camera_key)
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    n_multi = 0
    n_empty = 0
    for image_path, boxes in zip(image_paths, boxes_per_frame):
        stem = _extract_timestamp_stem(image_path)
        annotations = _build_annotation_list(
            boxes, class_name, image_width, image_height, one_based_to_zero_based=True
        )

        if boxes.shape[0] == 0:
            n_empty += 1
        elif boxes.shape[0] > 1:
            n_multi += 1
            logger.warning(
                "Frame '%s' has %d objects for class '%s' (multiple objects per frame).",
                stem, boxes.shape[0], class_name,
            )

        out_path = os.path.join(out_dir, f"{stem}.json")
        with open(out_path, "w") as fh:
            json.dump(annotations, fh, indent=2)
        n_written += 1

    logger.info(
        "Wrote %d annotation files to '%s' (%d empty-scene, %d multi-object frames).",
        n_written, out_dir, n_empty, n_multi,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a MATLAB Video Labeler groundTruth .mat export "
                     "into the bag_to_fusion_dataset camera label JSON format."
    )
    parser.add_argument("--mat-file", required=True, help="Path to the .mat file.")
    parser.add_argument(
        "--labels-dir", required=True,
        help="Path to the labels/ directory inside the Stage-1 intermediate output folder."
    )
    parser.add_argument(
        "--camera-key", default="camera_0",
        help="Sensor key the annotations belong to (default: camera_0)."
    )
    parser.add_argument("--image-width", type=int, default=1928, help="Image width in pixels.")
    parser.add_argument("--image-height", type=int, default=500, help="Image height in pixels.")
    parser.add_argument(
        "--class-name", default=None,
        help="LabelDefinitions class to convert (default: auto-detect if only one class)."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    convert(
        mat_file_path=args.mat_file,
        labels_dir=args.labels_dir,
        camera_key=args.camera_key,
        image_width=args.image_width,
        image_height=args.image_height,
        class_name=args.class_name,
    )


if __name__ == "__main__":
    main()