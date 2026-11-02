"""
Stage 1:  BAG -> intermediate format  (timestamp matching + extraction)
Stage 2:  intermediate -> KITTI / nuScenes / Waymo  (format conversion)

Usage
    python main.py --config config/config_eva.yaml
    python main.py --config config/config_minerva.yaml --stage 1
    python main.py --config config/clusters.yaml --stage 1 --from_det
    python main.py --config config/clusters.yaml --stage 1 --from_det --cluster
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time

import numpy as np

from utils.config_loader import load_config, load_cluster_config, resolve_kitti_cameras
from bag_reader import make_parser
from timestamp_matching.matcher import match_timestamps
from extraction.lidar_extractor import extract_lidar_frame, build_lidar_output_path
from extraction.camera_extractor import extract_camera_frame, build_camera_output_path
from extraction.calibration_extractor import extract_calibration
from extraction.ego_pose_extractor import load_ego_poses_from_parser, find_ego_pose_at
from extraction.inspection_metadata_writer import write_inspection_config
from file_writer.csv_writer import write_csv
from file_writer.metadata_writer import write_metadata
from file_writer.ego_pose_csv_writer import write_ego_pose_csv

os.environ['TZ'] = 'Europe/Rome'
time.tzset()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort_bag_files(bag_dir: str, extension: str):
    """
    Return .mcap / .db3 files in "bag_dir" sorted by trailing numeric index.
    See "extract_and_sort_files" from pointcloud_dataset_generation.
    """
    pattern = re.compile(rf".*_(\d+)\.{extension}$")
    pairs = []
    for fname in os.listdir(bag_dir):
        m = pattern.search(fname)
        if m:
            pairs.append((fname, int(m.group(1))))
    pairs.sort(key=lambda x: x[1])
    return [p[0] for p in pairs]


def _detect_extension(bag_dir: str) -> str:
    for fname in os.listdir(bag_dir):
        if fname.endswith(".mcap"):
            return "mcap"
        if fname.endswith(".db3"):
            return "db3"
    raise ValueError(f"No .mcap or .db3 files found in: {bag_dir}")


def _bag_stem(bag_dir: str) -> str:
    """Use the bag folder name as the Stage-1 sub-directory name."""
    return os.path.basename(os.path.normpath(bag_dir))


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def run_stage1(config) -> str:
    """
    Execute Stage 1: timestamp matching + sensor data extraction + calibration.
    Returns the path to the Stage-1 output directory for this bag.
    """
    bag_dir = config.bag_dir
    ext = _detect_extension(bag_dir)
    files = _sort_bag_files(bag_dir, ext)

    if not files:
        logger.error("No bag files found in %s", bag_dir)
        sys.exit(1)

    stem = _bag_stem(bag_dir)
    stage1_out = os.path.join(config.output_base_dir, stem)
    tolerance_ms = config.matching.tolerance_ms
    sampling_interval_ms = config.matching.sampling_interval_ms
    ref_key = config.lidars[0].key  # lidar_front is always the reference

    logger.info(">>> STAGE 1")
    logger.info("BAG folder  : %s  (%d files, .%s)", bag_dir, len(files), ext)
    logger.info("Output dir  : %s", stage1_out)
    logger.info("Tolerance   : (+-)%.1f ms", tolerance_ms)
    logger.info("Interval    : %.1f ms", sampling_interval_ms)

    ## Collect timestamps from every file for all sensors
    all_lidar_ts = {l.key: [] for l in config.lidars}
    all_camera_ts = {c.key: [] for c in config.cameras}
    file_ref_ranges = {}  # {fname: (min_ref_ts, max_ref_ts)}

    # Ego pose: accumulate chunks from each file
    ego_ts_chunks: list = []
    ego_pose_chunks: list = []

    for fname in files:  #TODO: optimize the loop by only opening each file once
        fpath = os.path.join(bag_dir, fname)
        parser = make_parser(fpath)
        logger.info("Reading timestamps from %s ...", fname)

        for lidar in config.lidars:
            ts = parser.get_message_timestamps(lidar.topic)
            all_lidar_ts[lidar.key].extend(ts)
            # Record the time span of the reference LiDAR in this file
            if lidar.key == ref_key and ts:
                file_ref_ranges[fname] = (min(ts), max(ts))

        for cam in config.cameras:
            ts = parser.get_message_timestamps(cam.topic)
            all_camera_ts[cam.key].extend(ts)

        # Load ego poses from /tf topic in this file
        file_ego_ts, file_ego_poses = load_ego_poses_from_parser(
            parser,
            config.ego_pose.topic,
            config.ego_pose.parent_frame,
            config.ego_pose.child_frame,
        )
        if len(file_ego_ts) > 0:
            ego_ts_chunks.append(file_ego_ts)
            ego_pose_chunks.append(file_ego_poses)

    # Merge ego pose chunks from all files into one sorted array
    if ego_ts_chunks:
        all_ego_ts = np.concatenate(ego_ts_chunks)
        all_ego_data = np.vstack(ego_pose_chunks)
        order = np.argsort(all_ego_ts)
        all_ego_ts = all_ego_ts[order]
        all_ego_data = all_ego_data[order]
        logger.info("Total ego poses collected across all files: %d", len(all_ego_ts))
    else:
        all_ego_ts   = np.array([], dtype=np.int64)
        all_ego_data = np.zeros((0, 7), dtype=np.float64)
        logger.warning(
            "No ego poses found in any bag file for topic '%s' "
            "('%s' -> '%s'). ego_poses.csv will have empty numeric cells.",
            config.ego_pose.topic,
            config.ego_pose.parent_frame,
            config.ego_pose.child_frame,
        )

    ## Timestamp matching
    logger.info("Running timestamp matching ...")
    groups = match_timestamps(
        lidar_timestamps = all_lidar_ts,
        camera_timestamps = all_camera_ts,
        tolerance_ms = tolerance_ms,
        sampling_interval_ms = sampling_interval_ms,
    )
    logger.info("Found %d valid sample groups.", len(groups))

    if not groups:
        logger.warning("No valid sample groups found. Check tolerance / sampling parameters.")
        return stage1_out

    ## Resolve ego pose for every matched group
    logger.info("Resolving ego poses for %d groups ...", len(groups))
    ego_pose_results = [
        find_ego_pose_at(
            ref_ts       = group.lidar_timestamps[ref_key],
            ts_arr       = all_ego_ts,
            pose_arr     = all_ego_data,
            tolerance_ms = config.matching.ego_pose_tolerance_ms,
        )
        for group in groups
    ]

    ## Extract sensor data for each matched group
    # Open each bag file once and extract only the groups whose reference LiDAR timestamp falls within that file's time range.
    logger.info("Extracting sensor data ...")
    tolerance_bound_ns = int(tolerance_ms * 1_000_000)

    for fname in files:
        ref_range = file_ref_ranges.get(fname)
        if ref_range is None:
            logger.warning("No reference timestamps found in %s, skipping.", fname)
            continue

        file_min_ts, file_max_ts = ref_range
        lower_bound = file_min_ts - tolerance_bound_ns
        upper_bound = file_max_ts + tolerance_bound_ns

        file_groups = []
        for g in groups:
            timestamp = g.lidar_timestamps.get(ref_key, -1)
            if lower_bound <= timestamp <= upper_bound:
                file_groups.append(g)

        if not file_groups:
            logger.info("No groups to extract from %s - skipping.", fname)
            continue

        logger.info("Extracting %d groups from %s ...", len(file_groups), fname)

        fpath  = os.path.join(bag_dir, fname)
        parser = make_parser(fpath)

        for group in file_groups:

            # LiDAR frames
            for lidar in config.lidars:
                ts = group.lidar_timestamps.get(lidar.key)
                if ts is None:
                    continue
                out_path = build_lidar_output_path(config.output_base_dir, stem, lidar.key, ts)
                extract_lidar_frame(parser, lidar.topic, ts, out_path, tolerance_ms, config.on_existing)

            # Camera frames
            for cam in config.cameras:
                ts = group.camera_timestamps.get(cam.key)
                if ts is None:
                    continue
                out_path = build_camera_output_path(config.output_base_dir, stem, cam.key, ts)
                extract_camera_frame(parser, cam.topic, ts, out_path, tolerance_ms, cam.compressed, config.on_existing)

    ## Write CSV + metadata + ego poses
    csv_path       = os.path.join(stage1_out, "timestamps.csv")
    meta_path      = os.path.join(stage1_out, "info_extraction.yaml")
    ego_pose_path  = os.path.join(stage1_out, "ego_poses.csv")

    write_csv(groups, csv_path)
    write_metadata(config, files, len(groups), meta_path, first_timestamp_ns=groups[0].lidar_timestamps[ref_key])
    write_ego_pose_csv(groups, ego_pose_results, ref_key, ego_pose_path)

    ## Extract and write calibration.
    sensor_links = (
        [l.frame_id for l in config.lidars] +
        [c.frame_id for c in config.cameras] +
        # The fused-LiDAR annotations (labels/lidar/) are anchored at this
        # frame, which is physically distinct from reference_frame
        # (ring_center) - the KITTI label converter needs its extrinsic too.
        [config.ego_pose.child_frame]
    )
    camera_frame_ids = [c.frame_id for c in config.cameras]

    extract_calibration(
        urdf_path         = config.urdf_path,
        intrinsics_yaml_path = config.calibration_yaml,
        stage1_out        = stage1_out,
        reference_frame   = config.reference_frame,
        sensor_links      = sensor_links,
        camera_frame_ids  = camera_frame_ids,
    )

    write_inspection_config(config, stage1_out)

    logger.info("Stage 1 complete. Output: %s", stage1_out)
    return stage1_out


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def run_stage2(config, stage1_dir: str) -> None:
    """
    Execute Stage 2: convert the Stage-1 intermediate data to each requested output format.
    """
    from converters import get_converter

    logger.info(">>> STAGE 2")

    for fmt in config.output_formats:
        out_dir = os.path.join(config.output_base_dir, os.path.basename(stage1_dir) + f"_{fmt}")
        converter = get_converter(fmt, config, stage1_dir, out_dir)
        logger.info("Converting to %s -> %s", fmt.upper(), out_dir)
        converter.convert()

    logger.info("Stage 2 complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="bag_to_fusion_dataset: BAG -> KITTI / nuScenes / Waymo")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument(
        "--from_det", action="store_true",
        help="Run Stage 1 from 3-D detections using the detection extraction config",
    )
    parser.add_argument(
        "--cluster", action="store_true",
        help="With --from_det: save selected detection clusters instead of full LiDAR clouds",
    )
    parser.add_argument("--stage", type=int, choices=[1, 2, 12], default=12, help="Which stage(s) to run: 1, 2, or 12 (both). Default: 12")
    parser.add_argument(
        "--cameras", nargs="+", default=None, metavar="CAMERA_KEY",
        help="KITTI only: cameras to include, by key, in interleaving order "
             "(e.g. --cameras camera_0 camera_2). Overrides kitti.cameras in the "
             "config. Default: all configured cameras.",
    )
    args = parser.parse_args()
    if args.cluster and not args.from_det:
        parser.error("--cluster requires --from_det")
    if args.from_det and args.stage == 2:
        parser.error("--from_det is a Stage-1 option; use --stage 1 or omit --stage")
    return args


def main():
    args   = _parse_args()
    if args.from_det:
        from extraction.cluster_extractor import run_detection_extraction

        config = load_cluster_config(args.config)
        if os.path.isfile(config.bag_dir):
            files = [config.bag_dir]
        else:
            ext = _detect_extension(config.bag_dir)
            names = _sort_bag_files(config.bag_dir, ext)
            # Also accept filenames without a numeric split suffix.
            names += sorted(
                name for name in os.listdir(config.bag_dir)
                if name.endswith(f".{ext}") and name not in names
            )
            files = [os.path.join(config.bag_dir, name) for name in names]
        run_detection_extraction(config, files, cluster=args.cluster)
        return

    config = load_config(args.config)

    config.kitti_selected_cameras = resolve_kitti_cameras(
        config.cameras, args.cameras or config.kitti_camera_selection
    )

    stage1_dir = os.path.join(config.output_base_dir, _bag_stem(config.bag_dir))

    if args.stage in (1, 12):
        stage1_dir = run_stage1(config)

    if args.stage in (2, 12):
        run_stage2(config, stage1_dir)


if __name__ == "__main__":
    main()
