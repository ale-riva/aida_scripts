#!/usr/bin/env python3
"""Convert paired PCD point clouds and JSON 3D boxes to lidar-only nuScenes format.

Input assumptions
-----------------
- PCD and JSON files have the same filename stem.
- Supported stems are YYYYMMDD_HHMMSS_NNNNNNNNN, 19-digit Unix
  nanoseconds, or arbitrary frame identifiers such as 000001.
- Every annotation is exported as category vehicle.car (Racecar), with
  attribute vehicle.moving and maximum visibility.
- Each JSON file contains a list of objects with the structure:
    obj_id, obj_type, psr.position, psr.rotation, psr.scale
- All point clouds and boxes are expressed in the LIDAR_TOP frame.
- psr.scale is [length=x, width=y, height=z].
- psr.rotation.z is yaw in radians.

Output
------
<output>/
  samples/LIDAR_TOP/*.pcd.bin
  maps/map.png
  v1.0-trainval/*.json

The generated lidar files contain float32 values in nuScenes order:
    x, y, z, intensity, ring
where ring is set to zero because it is unavailable in the input PCD.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import uuid
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


TOKEN_NAMESPACE = uuid.UUID("39b92dac-1d66-4b67-872e-8262cdda362b")
IDENTITY_TRANSLATION = [0.0, 0.0, 0.0]
IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0]  # Quaternion: w, x, y, z.

CATEGORY_NAME = "vehicle.car"
CATEGORY_DESCRIPTION = "Racecar"
ATTRIBUTE_NAME = "vehicle.moving"
ATTRIBUTE_DESCRIPTION = "Vehicle is moving."
MAX_VISIBILITY_TOKEN = "4"

# Standard nuScenes visibility intervals.
VISIBILITY_DEFINITIONS = [
    ("1", "v0-40", "Visibility of object is between 0% and 40%."),
    ("2", "v40-60", "Visibility of object is between 40% and 60%."),
    ("3", "v60-80", "Visibility of object is between 60% and 80%."),
    ("4", "v80-100", "Visibility of object is between 80% and 100%."),
]


@dataclass(frozen=True)
class Frame:
    stem: str
    pcd_path: Path
    json_path: Path
    timestamp_us: int
    labels: list[dict[str, Any]]


def make_token(record_type: str, key: str) -> str:
    """Generate deterministic 32-character nuScenes-style tokens."""
    return uuid.uuid5(TOKEN_NAMESPACE, f"{record_type}:{key}").hex


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
        file.write("\n")


def unix_nanoseconds_to_name(unix_nanoseconds: int) -> str:
    """Convert Unix nanoseconds to YYYYMMDD_HHMMSS_NNNNNNNNN in UTC."""
    seconds, nanoseconds = divmod(unix_nanoseconds, 1_000_000_000)
    date_time = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{date_time:%Y%m%d_%H%M%S}_{nanoseconds:09d}"


def normalize_frame_name(stem: str) -> str:
    """Normalize only the output filename.

    - ``YYYYMMDD_HHMMSS_NNNNNNNNN``: keep unchanged.
    - A 19-digit Unix timestamp in nanoseconds: convert to the formatted name.
    - Any other name, such as ``000001`` or ``frame_a``: keep unchanged.
    """
    if re.fullmatch(r"\d{19}", stem):
        return unix_nanoseconds_to_name(int(stem))

    return stem


def load_labels(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if not isinstance(labels, list):
        raise ValueError(f"Expected a JSON list in {json_path}")

    for index, label in enumerate(labels):
        try:
            psr = label["psr"]
            _ = psr["position"]["x"]
            _ = psr["position"]["y"]
            _ = psr["position"]["z"]
            _ = psr["rotation"]["z"]
            _ = psr["scale"]["x"]
            _ = psr["scale"]["y"]
            _ = psr["scale"]["z"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Malformed object {index} in {json_path}: missing PSR fields"
            ) from error

    return labels


def discover_frames(
    pcd_in: Path,
    json_in: Path,
) -> list[Frame]:
    pcd_by_stem = {
        path.stem: path
        for path in pcd_in.glob("*.pcd")
        if path.is_file()
    }
    json_by_stem = {
        path.stem: path
        for path in json_in.glob("*.json")
        if path.is_file()
    }

    common_stems = set(pcd_by_stem) & set(json_by_stem)
    missing_pcd = sorted(set(json_by_stem) - set(pcd_by_stem))
    missing_json = sorted(set(pcd_by_stem) - set(json_by_stem))

    for stem in missing_pcd:
        print(f"Warning: missing PCD for {json_by_stem[stem].name}")
    for stem in missing_json:
        print(f"Warning: missing JSON for {pcd_by_stem[stem].name}")

    if not common_stems:
        raise RuntimeError("No matching .pcd/.json pairs were found")

    def frame_sort_key(stem: str) -> tuple[int, int | str]:
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    sorted_stems = sorted(common_stems, key=frame_sort_key)
    normalized_stems = [normalize_frame_name(stem) for stem in sorted_stems]

    if len(normalized_stems) != len(set(normalized_stems)):
        raise ValueError(
            "Two or more input filenames produce the same normalized output name"
        )

    frames: list[Frame] = []

    for frame_index, (source_stem, output_stem) in enumerate(
        zip(sorted_stems, normalized_stems)
    ):
        frames.append(
            Frame(
                stem=output_stem,
                pcd_path=pcd_by_stem[source_stem],
                json_path=json_by_stem[source_stem],
                timestamp_us=frame_index,
                labels=load_labels(json_by_stem[source_stem]),
            )
        )

    return frames


def _pcd_numpy_dtype(type_code: str, size: int) -> np.dtype:
    mapping = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "<i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "<u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }
    try:
        return np.dtype(mapping[(type_code.upper(), size)])
    except KeyError as error:
        raise ValueError(f"Unsupported PCD field type: TYPE={type_code}, SIZE={size}") from error


def read_pcd_xyzi(path: Path) -> np.ndarray:
    """Read x, y, z and intensity from ASCII or uncompressed binary PCD."""
    header: dict[str, list[str]] = {}

    with path.open("rb") as file:
        while True:
            raw_line = file.readline()
            if not raw_line:
                raise ValueError(f"PCD header in {path} has no DATA line")

            line = raw_line.decode("ascii", errors="strict").strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                break

        fields = header.get("FIELDS") or header.get("FIELD")
        if not fields:
            raise ValueError(f"Missing FIELDS entry in {path}")

        sizes = [int(value) for value in header.get("SIZE", [])]
        types = [value.upper() for value in header.get("TYPE", [])]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]

        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"Inconsistent PCD field metadata in {path}")

        point_count = int(
            header.get(
                "POINTS",
                [str(int(header.get("WIDTH", ["0"])[0]) * int(header.get("HEIGHT", ["1"])[0]))],
            )[0]
        )
        data_mode = header["DATA"][0].lower()

        field_column_start: dict[str, int] = {}
        column = 0
        for name, count in zip(fields, counts):
            field_column_start[name] = column
            column += count

        if data_mode == "ascii":
            points = np.loadtxt(file, dtype=np.float64)
            if points.size == 0:
                points = np.empty((0, column), dtype=np.float64)
            elif points.ndim == 1:
                points = points.reshape(1, -1)

            if points.shape[1] != column:
                raise ValueError(
                    f"Expected {column} columns in {path}, found {points.shape[1]}"
                )

            def extract_ascii(name: str) -> np.ndarray:
                return points[:, field_column_start[name]]

            extractor = extract_ascii

        elif data_mode == "binary":
            dtype_fields: list[Any] = []
            for name, size, type_code, count in zip(fields, sizes, types, counts):
                base_dtype = _pcd_numpy_dtype(type_code, size)
                if count == 1:
                    dtype_fields.append((name, base_dtype))
                else:
                    dtype_fields.append((name, base_dtype, (count,)))

            structured = np.fromfile(file, dtype=np.dtype(dtype_fields), count=point_count)
            if structured.size != point_count:
                raise ValueError(
                    f"Expected {point_count} binary points in {path}, "
                    f"found {structured.size}"
                )

            def extract_binary(name: str) -> np.ndarray:
                values = np.asarray(structured[name])
                if values.ndim != 1:
                    values = values[:, 0]
                return values

            extractor = extract_binary

        elif data_mode == "binary_compressed":
            raise NotImplementedError(
                f"{path} uses DATA binary_compressed. Convert it to DATA binary "
                "or DATA ascii before running this script."
            )
        else:
            raise ValueError(f"Unsupported DATA mode {data_mode!r} in {path}")

        missing_xyz = [name for name in ("x", "y", "z") if name not in fields]
        if missing_xyz:
            raise ValueError(f"Missing fields {missing_xyz} in {path}")

        intensity_name = next(
            (name for name in ("intensity", "reflectivity", "i") if name in fields),
            None,
        )

        x = np.asarray(extractor("x"), dtype=np.float32)
        y = np.asarray(extractor("y"), dtype=np.float32)
        z = np.asarray(extractor("z"), dtype=np.float32)
        intensity = (
            np.asarray(extractor(intensity_name), dtype=np.float32)
            if intensity_name is not None
            else np.zeros_like(x, dtype=np.float32)
        )

        return np.column_stack((x, y, z, intensity)).astype(np.float32, copy=False)


def write_nuscenes_lidar_bin(points_xyzi: np.ndarray, path: Path) -> None:
    ring = np.zeros((points_xyzi.shape[0], 1), dtype=np.float32)
    points_nuscenes = np.hstack((points_xyzi.astype(np.float32, copy=False), ring))
    path.parent.mkdir(parents=True, exist_ok=True)
    points_nuscenes.tofile(path)


def yaw_to_quaternion(yaw: float) -> list[float]:
    half_yaw = 0.5 * yaw
    return [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]


def count_points_in_box(
    points_xyz: np.ndarray,
    center: np.ndarray,
    length: float,
    width: float,
    height: float,
    yaw: float,
) -> int:
    relative = points_xyz - center.reshape(1, 3)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)

    # Rotate points by -yaw into the annotation box frame.
    local_x = cosine * relative[:, 0] + sine * relative[:, 1]
    local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
    local_z = relative[:, 2]

    inside = (
        (np.abs(local_x) <= length * 0.5)
        & (np.abs(local_y) <= width * 0.5)
        & (np.abs(local_z) <= height * 0.5)
    )
    return int(np.count_nonzero(inside))


def write_grayscale_png(path: Path, width: int = 10, height: int = 10) -> None:
    """Write a dependency-free white 8-bit grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    scanlines = b"".join(b"\x00" + bytes([255]) * width for _ in range(height))
    png = signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def build_dataset(
    frames: list[Frame],
    output_root: Path,
    version: str,
    scene_name: str,
    scene_description: str,
    samples_per_scene: int,
    overwrite: bool,
) -> None:
    output_root_openpcdet = output_root / version
    metadata_root = output_root_openpcdet / version
    samples_root = output_root_openpcdet / "samples" / "LIDAR_TOP"
    map_relative_path = Path("maps") / "map.png"
    map_path = output_root_openpcdet / map_relative_path

    if metadata_root.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata directory already exists: {metadata_root}. Use --overwrite to replace it."
        )

    if overwrite:
        shutil.rmtree(metadata_root, ignore_errors=True)
        shutil.rmtree(samples_root, ignore_errors=True)

    metadata_root.mkdir(parents=True, exist_ok=True)
    samples_root.mkdir(parents=True, exist_ok=True)
    write_grayscale_png(map_path)

    category_token = make_token("category", CATEGORY_NAME)
    moving_attribute_token = make_token("attribute", ATTRIBUTE_NAME)

    category_records = [
        {
            "token": category_token,
            "name": CATEGORY_NAME,
            "description": CATEGORY_DESCRIPTION,
            "index": 0,
        }
    ]

    attribute_records = [
        {
            "token": moving_attribute_token,
            "name": ATTRIBUTE_NAME,
            "description": ATTRIBUTE_DESCRIPTION,
        }
    ]

    visibility_records = [
        {
            "token": token,
            "level": level,
            "description": description,
        }
        for token, level, description in VISIBILITY_DEFINITIONS
    ]

    sensor_token = make_token("sensor", "LIDAR_TOP")
    calibrated_sensor_token = make_token("calibrated_sensor", "LIDAR_FRONT_identity")
    log_token = make_token("log", scene_name)
    map_token = make_token("map", "custom-map")

    scene_count = math.ceil(len(frames) / samples_per_scene)
    scene_names = [
        f"{scene_name}-{scene_index + 1:04d}"
        for scene_index in range(scene_count)
    ]
    scene_tokens = [
        make_token("scene", current_scene_name)
        for current_scene_name in scene_names
    ]

    sensor_records = [
        {
            "token": sensor_token,
            "channel": "LIDAR_TOP",
            "modality": "lidar",
        }
    ]

    calibrated_sensor_records = [
        {
            "token": calibrated_sensor_token,
            "sensor_token": sensor_token,
            "translation": IDENTITY_TRANSLATION,
            "rotation": IDENTITY_ROTATION,
            "camera_intrinsic": [],
        }
    ]

    first_datetime = datetime.fromtimestamp(frames[0].timestamp_us / 1_000_000, tz=timezone.utc)
    log_records = [
        {
            "token": log_token,
            "logfile": scene_name,
            "vehicle": "custom_vehicle",
            "date_captured": first_datetime.strftime("%Y-%m-%d"),
            "location": "custom",
        }
    ]

    map_records = [
        {
            "token": map_token,
            "log_tokens": [log_token],
            "category": "semantic_prior",
            "filename": map_relative_path.as_posix(),
        }
    ]

    sample_tokens = [make_token("sample", frame.stem) for frame in frames]
    sample_data_tokens = [make_token("sample_data", frame.stem) for frame in frames]
    ego_pose_tokens = [make_token("ego_pose", frame.stem) for frame in frames]

    sample_records: list[dict[str, Any]] = []
    sample_data_records: list[dict[str, Any]] = []
    ego_pose_records: list[dict[str, Any]] = []
    annotation_records: list[dict[str, Any]] = []

    # key: (scene index, source obj_id), value: annotation tokens in time order.
    instance_annotation_tokens: dict[tuple[int, str], list[str]] = defaultdict(list)

    for frame_index, frame in enumerate(
        tqdm(frames, desc="Converting samples", unit="sample")
    ):
        sample_token = sample_tokens[frame_index]
        sample_data_token = sample_data_tokens[frame_index]
        ego_pose_token = ego_pose_tokens[frame_index]

        scene_index = frame_index // samples_per_scene
        scene_start_index = scene_index * samples_per_scene
        scene_end_index = min(scene_start_index + samples_per_scene, len(frames))

        sample_records.append(
            {
                "token": sample_token,
                "timestamp": frame.timestamp_us,
                "scene_token": scene_tokens[scene_index],
                "prev": (
                    sample_tokens[frame_index - 1]
                    if frame_index > scene_start_index
                    else ""
                ),
                "next": (
                    sample_tokens[frame_index + 1]
                    if frame_index + 1 < scene_end_index
                    else ""
                ),
            }
        )

        ego_pose_records.append(
            {
                "token": ego_pose_token,
                "translation": IDENTITY_TRANSLATION,
                "rotation": IDENTITY_ROTATION,
                "timestamp": frame.timestamp_us,
            }
        )

        output_lidar_name = f"{frame.stem}.pcd.bin"
        output_lidar_path = samples_root / output_lidar_name
        points_xyzi = read_pcd_xyzi(frame.pcd_path)
        write_nuscenes_lidar_bin(points_xyzi, output_lidar_path)

        sample_data_records.append(
            {
                "token": sample_data_token,
                "sample_token": sample_token,
                "ego_pose_token": ego_pose_token,
                "calibrated_sensor_token": calibrated_sensor_token,
                "filename": (Path("samples") / "LIDAR_TOP" / output_lidar_name).as_posix(),
                "fileformat": "bin",
                "width": 0,
                "height": 0,
                "timestamp": frame.timestamp_us,
                "is_key_frame": True,
                "prev": (
                    sample_data_tokens[frame_index - 1]
                    if frame_index > scene_start_index
                    else ""
                ),
                "next": (
                    sample_data_tokens[frame_index + 1]
                    if frame_index + 1 < scene_end_index
                    else ""
                ),
            }
        )

        for object_index, label in enumerate(frame.labels):
            psr = label["psr"]
            position = psr["position"]
            rotation = psr["rotation"]
            scale = psr["scale"]

            center = np.array(
                [float(position["x"]), float(position["y"]), float(position["z"])],
                dtype=np.float64,
            )
            yaw = float(rotation["z"])
            length = float(scale["x"])
            width = float(scale["y"])
            height = float(scale["z"])

            if length <= 0.0 or width <= 0.0 or height <= 0.0:
                raise ValueError(
                    f"Non-positive box size in {frame.json_path}, object {object_index}"
                )

            source_obj_id = str(label.get("obj_id", object_index))
            instance_token = make_token(
                "instance",
                f"{scene_names[scene_index]}:{CATEGORY_NAME}:{source_obj_id}",
            )
            annotation_token = make_token(
                "sample_annotation",
                f"{frame.stem}:{CATEGORY_NAME}:{source_obj_id}:{object_index}",
            )

            instance_annotation_tokens[(scene_index, source_obj_id)].append(
                annotation_token
            )

            annotation_records.append(
                {
                    "token": annotation_token,
                    "sample_token": sample_token,
                    "instance_token": instance_token,
                    "attribute_tokens": [moving_attribute_token],
                    "visibility_token": MAX_VISIBILITY_TOKEN,
                    "translation": center.tolist(),
                    # Input scale: x=length, y=width, z=height.
                    # nuScenes size: width, length, height.
                    "size": [width, length, height],
                    "rotation": yaw_to_quaternion(yaw),
                    "num_lidar_pts": count_points_in_box(
                        points_xyz=points_xyzi[:, :3],
                        center=center,
                        length=length,
                        width=width,
                        height=height,
                        yaw=yaw,
                    ),
                    "num_radar_pts": 0,
                    "prev": "",
                    "next": "",
                }
            )


    annotation_by_token = {record["token"]: record for record in annotation_records}
    instance_records: list[dict[str, Any]] = []

    for (
        scene_index,
        source_obj_id,
    ), ann_tokens in sorted(instance_annotation_tokens.items()):
        for index, ann_token in enumerate(ann_tokens):
            annotation_by_token[ann_token]["prev"] = ann_tokens[index - 1] if index > 0 else ""
            annotation_by_token[ann_token]["next"] = (
                ann_tokens[index + 1] if index + 1 < len(ann_tokens) else ""
            )

        instance_records.append(
            {
                "token": make_token(
                    "instance",
                    f"{scene_names[scene_index]}:{CATEGORY_NAME}:{source_obj_id}",
                ),
                "category_token": category_token,
                "nbr_annotations": len(ann_tokens),
                "first_annotation_token": ann_tokens[0],
                "last_annotation_token": ann_tokens[-1],
            }
        )

    scene_records: list[dict[str, Any]] = []

    for scene_index, current_scene_name in enumerate(scene_names):
        first_frame_index = scene_index * samples_per_scene
        last_frame_index = min(
            first_frame_index + samples_per_scene,
            len(frames),
        ) - 1

        scene_records.append(
            {
                "token": scene_tokens[scene_index],
                "name": current_scene_name,
                "description": scene_description,
                "log_token": log_token,
                "nbr_samples": last_frame_index - first_frame_index + 1,
                "first_sample_token": sample_tokens[first_frame_index],
                "last_sample_token": sample_tokens[last_frame_index],
            }
        )

    tables = {
        "attribute": attribute_records,
        "calibrated_sensor": calibrated_sensor_records,
        "category": category_records,
        "ego_pose": ego_pose_records,
        "instance": instance_records,
        "log": log_records,
        "map": map_records,
        "sample": sample_records,
        "sample_annotation": annotation_records,
        "sample_data": sample_data_records,
        "scene": scene_records,
        "sensor": sensor_records,
        "visibility": visibility_records,
    }

    for table_name, records in tables.items():
        write_json(metadata_root / f"{table_name}.json", records)

    print("\nnuScenes dataset created successfully")
    print(f"Output root:       {output_root_openpcdet}")
    print(f"Version:           {version}")
    print(f"Samples:           {len(sample_records)}")
    print(f"Annotations:       {len(annotation_records)}")
    print(f"Instances:         {len(instance_records)}")
    print(f"Categories:        {len(category_records)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert paired PCD and JSON 3D labels to lidar-only nuScenes format."
    )
    parser.add_argument(
        "--pcd-in",
        type=Path,
        required=True,
        help="Directory containing input .pcd files.",
    )
    parser.add_argument(
        "--json-in",
        type=Path,
        required=True,
        help="Directory containing corresponding .json label files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output nuScenes dataroot.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="Metadata version directory name. Default: v1.0-custom.",
    )
    parser.add_argument(
        "--scene-name",
        default="scene-0001",
        help="Name of the single generated scene.",
    )
    parser.add_argument(
        "--scene-description",
        default="Custom lidar-only scene converted from PCD and JSON labels.",
        help="Description stored in scene.json.",
    )
    parser.add_argument(
        "--samples-per-scene",
        type=int,
        default=40,
        help="Maximum number of samples in each scene. Default: 40.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing version metadata and LIDAR_TOP samples.",
    )

    args = parser.parse_args()

    pcd_in = args.pcd_in.expanduser().resolve()
    json_in = args.json_in.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not pcd_in.is_dir():
        raise NotADirectoryError(f"PCD input directory not found: {pcd_in}")
    if not json_in.is_dir():
        raise NotADirectoryError(f"JSON input directory not found: {json_in}")
    if args.samples_per_scene <= 0:
        raise ValueError("--samples-per-scene must be positive")

    frames = discover_frames(
        pcd_in=pcd_in,
        json_in=json_in,
    )
    build_dataset(
        frames=frames,
        output_root=output_root,
        version=args.version,
        scene_name=args.scene_name,
        scene_description=args.scene_description,
        samples_per_scene=args.samples_per_scene,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
