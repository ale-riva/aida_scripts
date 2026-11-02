#!/usr/bin/env python3
"""
Export fast nuScenes sample visualizations with the official devkit style.

For every sample, this script writes one rectangular PNG containing:
  - a bird's-eye-view LiDAR point cloud rendered by
    ``NuScenes.render_sample_data``;
  - only the camera images on which at least one 3D annotation box is visible.

Camera boxes are still transformed, visibility-filtered and drawn with the
nuScenes-devkit geometry and ``Box.render`` implementation. The expensive part
that is avoided is calling ``render_sample_data`` a second time after checking
whether a camera contains any visible boxes.

Expected dataset layout:

    <dataset_root>/
        v1.0-*/
        samples/
        sweeps/
        maps/

Output:

    <dataset_root>/vis/<scene-name>_<frame-index>_<sample-token>.png

The input may be either <dataset_root> or the version directory itself.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib

# Must be selected before importing pyplot or the nuScenes devkit.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

REQUIRED_VERSION_TABLES = (
    "sample.json",
    "sample_data.json",
    "scene.json",
    "sensor.json",
    "calibrated_sensor.json",
)

# Fixed rectangular output. There is intentionally no command-line resize
# option. 16 x 9 inches at 120 dpi produces a 1920 x 1080 image.
OUTPUT_FIGSIZE = (16.0, 9.0)
OUTPUT_DPI = 120
PNG_COMPRESS_LEVEL = 1

LIDAR_VIEW_ASPECT_RATIO = 16.0 / 9.0
LIDAR_MIN_HALF_HEIGHT_M = 25.0
LIDAR_BOX_MARGIN_M = 5.0

CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

# Filled in inside main() so importing the script does not require the devkit.
NuScenes = object
BoxVisibility = object
Quaternion = object
box_in_image = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one fast nuScenes-devkit-style visualization per sample. "
            "Only cameras containing at least one visible projected box are shown."
        )
    )
    parser.add_argument(
        "dataroot",
        type=Path,
        help=(
            "Path to the nuScenes dataset root, or directly to a version folder "
            "such as /data/nuscenes/v1.0-trainval."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Version folder name, for example v1.0-trainval. If omitted, the "
            "first valid version folder in lexical order is used."
        ),
    )
    parser.add_argument(
        "--lidar-channel",
        default=None,
        help=(
            "LiDAR channel to render. LIDAR_TOP is preferred by default; "
            "otherwise the first available LiDAR channel is used."
        ),
    )
    parser.add_argument(
        "--nsweeps",
        type=int,
        default=1,
        help="Number of LiDAR sweeps to aggregate. Default: 1.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after exporting this many samples in total.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help=(
            "Only export the given scene name or scene token. May be repeated. "
            "By default all scenes are exported."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite PNG files that already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when a sample cannot be rendered.",
    )
    parser.add_argument(
        "--verbose-devkit",
        action="store_true",
        help="Print nuScenes devkit loading details.",
    )
    return parser.parse_args()


def is_version_directory(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_VERSION_TABLES)


def resolve_dataset_layout(
    input_path: Path,
    requested_version: Optional[str],
) -> Tuple[Path, str, Path]:
    """Return ``(devkit_dataroot, version_name, output_root)``."""
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if is_version_directory(input_path):
        version_dir = input_path
        if requested_version is not None and requested_version != version_dir.name:
            raise ValueError(
                f"Input is already version directory '{version_dir.name}', but "
                f"--version requested '{requested_version}'."
            )
    elif requested_version is not None:
        version_dir = input_path / requested_version
        if not is_version_directory(version_dir):
            raise FileNotFoundError(
                f"Requested version is not a valid nuScenes version directory: {version_dir}"
            )
    else:
        candidates = sorted(
            (child for child in input_path.iterdir() if is_version_directory(child)),
            key=lambda path: path.name,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No nuScenes version folder found directly inside: {input_path}\n"
                f"A version folder must contain: {', '.join(REQUIRED_VERSION_TABLES)}"
            )
        version_dir = candidates[0]

    dataroot = version_dir.parent
    output_root = dataroot / "vis"
    return dataroot, version_dir.name, output_root


def natural_key(value: str) -> Tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def ordered_camera_items(
    items: Sequence[Tuple[str, str]],
) -> list[Tuple[str, str]]:
    order = {channel: index for index, channel in enumerate(CAMERA_ORDER)}
    return sorted(
        items,
        key=lambda item: (order.get(item[0], len(order)), natural_key(item[0])),
    )


def collect_sample_sensors(
    nusc: NuScenes,
    sample: dict,
) -> Tuple[list[Tuple[str, str]], list[Tuple[str, str]]]:
    """Collect LiDAR and camera keyframes for a sample.

    ``sample_data`` records are already reverse-indexed by the devkit, so each
    lookup is constant-time. No image or point-cloud file is opened here.
    """
    lidar_items: list[Tuple[str, str]] = []
    camera_items: list[Tuple[str, str]] = []

    for channel, sample_data_token in sample.get("data", {}).items():
        sample_data = nusc.get("sample_data", sample_data_token)
        modality = sample_data.get("sensor_modality")

        if modality == "lidar":
            lidar_items.append((channel, sample_data_token))
        elif modality == "camera":
            camera_items.append((channel, sample_data_token))

    return lidar_items, ordered_camera_items(camera_items)


def choose_lidar(
    lidar_items: Sequence[Tuple[str, str]],
    requested_channel: Optional[str],
) -> Optional[Tuple[str, str]]:
    if not lidar_items:
        return None

    by_channel = {channel: token for channel, token in lidar_items}
    if requested_channel is not None:
        if requested_channel not in by_channel:
            available = ", ".join(sorted(by_channel))
            raise KeyError(
                f"Requested LiDAR channel '{requested_channel}' is unavailable. "
                f"Available channels: {available}"
            )
        return requested_channel, by_channel[requested_channel]

    if "LIDAR_TOP" in by_channel:
        return "LIDAR_TOP", by_channel["LIDAR_TOP"]

    return sorted(lidar_items, key=lambda item: natural_key(item[0]))[0]


def iter_scene_samples(nusc: NuScenes, scene: dict) -> Iterable[Tuple[int, dict]]:
    token = scene["first_sample_token"]
    frame_index = 0
    visited: set[str] = set()

    while token:
        if token in visited:
            raise RuntimeError(
                f"Detected a loop in the sample chain of scene "
                f"'{scene.get('name', scene['token'])}'."
            )
        visited.add(token)

        sample = nusc.get("sample", token)
        yield frame_index, sample

        token = sample.get("next", "")
        frame_index += 1


def safe_scene_name(scene: dict) -> str:
    name = scene.get("name") or scene["token"][:12]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or scene["token"][:12]


def sample_boxes(nusc: NuScenes, sample: dict) -> list:
    """Instantiate each global annotation box exactly once for this sample."""
    return [nusc.get_box(annotation_token) for annotation_token in sample.get("anns", [])]


def compute_lidar_view_limits(
    nusc: NuScenes,
    lidar_sample_data_token: str,
    global_boxes: Sequence,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return a centered 16:9 flat-ego crop containing every annotation.

    This reproduces the devkit's ``use_flat_vehicle_coordinates=True`` box
    transform without calling ``get_sample_data`` before rendering. Therefore
    LiDAR annotations are not transformed twice merely to determine the crop.
    """
    required_half_width = 0.0
    required_half_height = 0.0

    if global_boxes:
        sample_data = nusc.get("sample_data", lidar_sample_data_token)
        pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

        yaw = Quaternion(pose["rotation"]).yaw_pitch_roll[0]
        flat_rotation_inverse = Quaternion(
            scalar=math.cos(yaw / 2.0),
            vector=[0.0, 0.0, math.sin(yaw / 2.0)],
        ).inverse
        pose_translation = np.asarray(pose["translation"], dtype=np.float64)

        corner_sets = []
        for source_box in global_boxes:
            box = source_box.copy()
            box.translate(-pose_translation)
            box.rotate(flat_rotation_inverse)
            corner_sets.append(box.corners())

        corners = np.concatenate(corner_sets, axis=1)
        required_half_width = float(np.max(np.abs(corners[0, :]))) + LIDAR_BOX_MARGIN_M
        required_half_height = float(np.max(np.abs(corners[1, :]))) + LIDAR_BOX_MARGIN_M

    half_height = max(
        LIDAR_MIN_HALF_HEIGHT_M,
        required_half_height,
        required_half_width / LIDAR_VIEW_ASPECT_RATIO,
    )
    half_width = LIDAR_VIEW_ASPECT_RATIO * half_height

    return (-half_width, half_width), (-half_height, half_height)


def visible_camera_panels(
    nusc: NuScenes,
    camera_items: Sequence[Tuple[str, str]],
    global_boxes: Sequence,
) -> list[Tuple[str, str, list, np.ndarray, int, int]]:
    """Return only cameras containing at least one visible projected box.

    Each source box is copied from the sample-level cache, transformed into the
    camera coordinate frame using the same equations as ``get_sample_data``,
    and filtered with the devkit's ``box_in_image`` function. The transformed
    boxes are retained and later passed directly to ``Box.render``. This avoids
    repeating all transforms inside ``render_sample_data``.
    """
    if not global_boxes:
        return []

    panels: list[Tuple[str, str, list, np.ndarray, int, int]] = []

    for channel, sample_data_token in camera_items:
        sample_data = nusc.get("sample_data", sample_data_token)
        calibrated_sensor = nusc.get(
            "calibrated_sensor",
            sample_data["calibrated_sensor_token"],
        )
        pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

        intrinsic = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float64)
        width = int(sample_data["width"])
        height = int(sample_data["height"])
        image_size = (width, height)

        pose_translation = np.asarray(pose["translation"], dtype=np.float64)
        sensor_translation = np.asarray(
            calibrated_sensor["translation"],
            dtype=np.float64,
        )
        pose_rotation_inverse = Quaternion(pose["rotation"]).inverse
        sensor_rotation_inverse = Quaternion(calibrated_sensor["rotation"]).inverse

        visible_boxes = []
        for source_box in global_boxes:
            box = source_box.copy()

            # Global -> ego at camera timestamp.
            box.translate(-pose_translation)
            box.rotate(pose_rotation_inverse)

            # Ego -> camera.
            box.translate(-sensor_translation)
            box.rotate(sensor_rotation_inverse)

            if box_in_image(
                box,
                intrinsic,
                image_size,
                vis_level=BoxVisibility.ANY,
            ):
                visible_boxes.append(box)

        if visible_boxes:
            panels.append(
                (
                    channel,
                    sample_data_token,
                    visible_boxes,
                    intrinsic,
                    width,
                    height,
                )
            )

    return panels


def render_camera_panel(
    nusc: NuScenes,
    ax: plt.Axes,
    sample_data_token: str,
    boxes: Sequence,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Render a camera panel using the same devkit primitives and styling."""
    image_path = nusc.get_sample_data_path(sample_data_token)

    # Keep the file handle open only while Pillow decodes the image. Matplotlib
    # receives the decoded array, so the file can close immediately afterward.
    with Image.open(image_path) as image:
        ax.imshow(np.asarray(image))

    for box in boxes:
        color = np.asarray(nusc.explorer.get_color(box.name), dtype=np.float64) / 255.0
        box.render(
            ax,
            view=intrinsic,
            normalize=True,
            colors=(color, color, color),
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


def render_sample_figure(
    nusc: NuScenes,
    sample: dict,
    lidar_channel: Optional[str],
    nsweeps: int,
) -> plt.Figure:
    lidar_items, camera_items = collect_sample_sensors(nusc, sample)
    selected_lidar = choose_lidar(lidar_items, lidar_channel)

    # One annotation construction pass per sample, reused for both LiDAR crop
    # calculation and all camera visibility/projection transforms.
    global_boxes = sample_boxes(nusc, sample)
    camera_panels = visible_camera_panels(nusc, camera_items, global_boxes)

    panel_count = len(camera_panels) + (1 if selected_lidar is not None else 0)
    if panel_count == 0:
        raise RuntimeError(
            "Sample has no LiDAR and no camera containing a visible annotation box."
        )

    # At most three panels per row keeps camera panels readable while preserving
    # a fixed landscape canvas. No tight_layout/constrained_layout pass is used.
    columns = min(3, panel_count)
    rows = math.ceil(panel_count / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=OUTPUT_FIGSIZE,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    axis_index = 0

    if selected_lidar is not None:
        lidar_name, lidar_token = selected_lidar
        ax = flat_axes[axis_index]
        axis_index += 1

        x_limits, y_limits = compute_lidar_view_limits(
            nusc,
            lidar_token,
            global_boxes,
        )
        devkit_axes_limit = max(abs(x_limits[1]), abs(y_limits[1]))

        # Keep the official devkit point-cloud plot.
        nusc.render_sample_data(
            lidar_token,
            with_anns=True,
            box_vis_level=BoxVisibility.ANY,
            axes_limit=devkit_axes_limit,
            ax=ax,
            nsweeps=nsweeps,
            underlay_map=False,
            use_flat_vehicle_coordinates=True,
            verbose=False,
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(lidar_name, fontsize=10, pad=2)

    for channel, token, boxes, intrinsic, width, height in camera_panels:
        ax = flat_axes[axis_index]
        axis_index += 1

        render_camera_panel(
            nusc=nusc,
            ax=ax,
            sample_data_token=token,
            boxes=boxes,
            intrinsic=intrinsic,
            width=width,
            height=height,
        )
        ax.set_title(channel, fontsize=10, pad=2)

    for ax in flat_axes[axis_index:]:
        ax.axis("off")

    fig.suptitle(
        f"sample {sample['token']}   timestamp {sample.get('timestamp', 'unknown')}",
        fontsize=11,
        y=0.995,
    )

    # Fixed margins are considerably cheaper than fig.tight_layout(), which
    # measures every title, image and annotation for every exported sample.
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.02,
        top=0.955,
        wspace=0.025,
        hspace=0.12,
    )
    return fig


def selected_scenes(
    nusc: NuScenes,
    filters: Optional[Sequence[str]],
) -> list[dict]:
    scenes = sorted(
        nusc.scene,
        key=lambda scene: natural_key(scene.get("name", scene["token"])),
    )
    if not filters:
        return scenes

    requested = set(filters)
    selected = [
        scene
        for scene in scenes
        if scene.get("name") in requested or scene["token"] in requested
    ]
    found = {scene.get("name") for scene in selected} | {
        scene["token"] for scene in selected
    }
    missing = sorted(requested - found)
    if missing:
        raise KeyError(f"Unknown scene name/token: {', '.join(missing)}")
    return selected


def main() -> int:
    args = parse_args()

    global NuScenes, BoxVisibility, Quaternion, box_in_image
    try:
        from nuscenes.nuscenes import NuScenes as NuScenesClass
        from nuscenes.utils.geometry_utils import (
            BoxVisibility as BoxVisibilityClass,
            box_in_image as devkit_box_in_image,
        )
        from pyquaternion import Quaternion as QuaternionClass
    except ImportError as exc:
        raise RuntimeError(
            "nuScenes devkit is not installed. Install it with: "
            "pip install nuscenes-devkit"
        ) from exc

    NuScenes = NuScenesClass
    BoxVisibility = BoxVisibilityClass
    Quaternion = QuaternionClass
    box_in_image = devkit_box_in_image

    if args.nsweeps <= 0:
        raise ValueError("--nsweeps must be greater than zero.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero.")

    dataroot, version, output_root = resolve_dataset_layout(
        args.dataroot,
        args.version,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(
        version=version,
        dataroot=str(dataroot),
        verbose=args.verbose_devkit,
    )
    scenes = selected_scenes(nusc, args.scene)

    scene_sample_counts = [scene.get("nbr_samples") for scene in scenes]
    if scene_sample_counts and all(
        isinstance(count, int) and count >= 0 for count in scene_sample_counts
    ):
        progress_total: Optional[int] = sum(scene_sample_counts)
    else:
        progress_total = None

    if args.max_samples is not None:
        progress_total = (
            min(progress_total, args.max_samples)
            if progress_total is not None
            else args.max_samples
        )

    attempted = 0
    exported = 0
    skipped = 0
    failed = 0
    stop = False

    with tqdm(
        total=progress_total,
        desc="Exporting nuScenes",
        unit="sample",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(exported=0, skipped=0, failed=0)

        for scene_index, scene in enumerate(scenes, start=1):
            scene_name = safe_scene_name(scene)

            for frame_index, sample in iter_scene_samples(nusc, scene):
                if args.max_samples is not None and attempted >= args.max_samples:
                    stop = True
                    break

                attempted += 1
                progress.set_description(
                    f"Scene {scene_index}/{len(scenes)}: {scene_name}"
                )
                output_path = output_root / (
                    f"{scene_name}_{frame_index:06d}_{sample['token'][:12]}.png"
                )

                if output_path.exists() and not args.overwrite:
                    skipped += 1
                    progress.update(1)
                    progress.set_postfix(
                        exported=exported,
                        skipped=skipped,
                        failed=failed,
                    )
                    continue

                fig: Optional[plt.Figure] = None
                try:
                    fig = render_sample_figure(
                        nusc=nusc,
                        sample=sample,
                        lidar_channel=args.lidar_channel,
                        nsweeps=args.nsweeps,
                    )
                    fig.savefig(
                        output_path,
                        dpi=OUTPUT_DPI,
                        pil_kwargs={"compress_level": PNG_COMPRESS_LEVEL},
                    )
                    exported += 1
                except Exception as exc:
                    failed += 1
                    tqdm.write(
                        f"ERROR sample {sample['token']}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    if args.fail_fast:
                        raise
                finally:
                    if fig is not None:
                        plt.close(fig)
                    else:
                        plt.close("all")

                    progress.update(1)
                    progress.set_postfix(
                        exported=exported,
                        skipped=skipped,
                        failed=failed,
                    )

            if stop:
                break

    print(
        f"Finished: attempted={attempted}, exported={exported}, "
        f"skipped={skipped}, failed={failed}\nOutput: {output_root}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
