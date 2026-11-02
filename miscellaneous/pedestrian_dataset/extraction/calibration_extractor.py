"""
Calibration extraction and conversion.

Reads:
  - camera_calibrations_<robot>.param.yaml  ->  intrinsics (K, distortion, frame dims)
  - <robot>.urdf or /tf_static               ->  extrinsics (sensor poses via chained transforms)

Produces:
  - <stage1_out>/calibration.yaml

Coordinate convention
All extrinsic transforms are expressed as 4x4 homogeneous matrices in the
reference frame defined by "reference_frame" (passed by the caller from the
vehicle settings file).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import yaml

from utils.UrdfParser import URDFParser
from utils.tf_static_parser import TFStaticParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URDF sensor-link inference
# ---------------------------------------------------------------------------

def _infer_sensor_links(urdf_parser: URDFParser) -> List[str]:
    """
    Return all leaf links in the URDF kinematic tree. Sensor frames are almost always leaves.
    """
    all_links = set(urdf_parser.links.keys())
    links_with_children = set(urdf_parser.children_map.keys())
    leaf_links = sorted(all_links - links_with_children)
    logger.info(
        "sensor_links not provided - inferred %d leaf nodes from URDF: %s",
        len(leaf_links), leaf_links,
    )
    return leaf_links


# ---------------------------------------------------------------------------
# Camera intrinsics parsing
# ---------------------------------------------------------------------------

def _parse_camera_intrinsics(yaml_path: str) -> Dict[str, dict]:
    """
    Parse camera_calibrations_<robot>.param.yaml.

    Returns a dictionary keyed by camera name (e.g. "camera_cl") with entries:
        frame_id, topic, model_type, fx, fy, cx, cy, width, height,
        and model-specific distortion fields:
          - PINHOLE: radial_dist [k1, k2, k3], tangential_dist [p1, p2]
          - FISHEYE: fisheye_dist [k1, k2, k3, k4]

    The "model_type" key is always present (defaults to "PINHOLE" when the
    field is absent, for backward compatibility with older calibration files).
    """
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)

    # Navigate /**:ros__parameters
    try:
        params = raw["/**"]["ros__parameters"]
    except KeyError:
        raise ValueError(
            f"Unexpected structure in {yaml_path}: "
            "expected '/**:ros__parameters' at the top level."
        )

    cameras: Dict[str, dict] = {}
    for cam_key, cam_data in params.items():
        if not cam_key.startswith("CAM_"):
            continue

        name       = cam_data["NAME"]
        k          = cam_data["K"]
        model_type = cam_data.get("MODEL_TYPE", "PINHOLE").upper()

        entry: dict = {
            "frame_id":   cam_data["FRAME_ID"],
            "topic":      cam_data["TOPIC"],
            "model_type": model_type,
            "fx":         float(k["FX"]),
            "fy":         float(k["FY"]),
            "cx":         float(k["CX"]),
            "cy":         float(k["CY"]),
            "width":      int(cam_data["FRAME_DIM"][0]),
            "height":     int(cam_data["FRAME_DIM"][1]),
        }

        if model_type == "FISHEYE":
            if "FISHEYE_DIST" not in cam_data:
                raise ValueError(
                    f"Camera '{name}' declares MODEL_TYPE=FISHEYE but is missing "
                    "the FISHEYE_DIST field."
                )
            entry["fisheye_dist"] = [float(v) for v in cam_data["FISHEYE_DIST"]]
        else:
            # PINHOLE (default) - requires RADIAL_DIST + TANGENTIAL_DIST
            if "RADIAL_DIST" not in cam_data or "TANGENTIAL_DIST" not in cam_data:
                raise ValueError(
                    f"Camera '{name}' declares MODEL_TYPE=PINHOLE (or has no MODEL_TYPE) "
                    "but is missing RADIAL_DIST or TANGENTIAL_DIST."
                )
            entry["radial_dist"]     = [float(v) for v in cam_data["RADIAL_DIST"]]
            entry["tangential_dist"] = [float(v) for v in cam_data["TANGENTIAL_DIST"]]

        cameras[name] = entry

    return cameras


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(
    extrinsics: Dict[str, np.ndarray],
    intrinsics: Dict[str, dict],
    camera_frame_ids: Optional[List[str]] = None,
) -> None:
    """
    Cross-check cameras present in URDF extrinsics against the intrinsics file.
    """
    extrinsic_keys = set(extrinsics.keys())
    intrinsic_cams = set(intrinsics.keys())

    if camera_frame_ids is not None:
        config_cams = set(camera_frame_ids)

        # Cameras in config that are absent from the URDF extrinsics
        missing_extrinsics = config_cams - extrinsic_keys
        if missing_extrinsics:
            logger.warning(
                "Camera(s) in vehicle settings but missing from URDF extrinsics: %s",
                sorted(missing_extrinsics),
            )

        # Cameras in config that have no intrinsic calibration
        missing_intrinsics = config_cams - intrinsic_cams
        if missing_intrinsics:
            logger.warning(
                "Camera(s) in vehicle settings but missing from intrinsics file: %s",
                sorted(missing_intrinsics),
            )

        # Cameras in the intrinsics file that the config doesn't know about
        extra_intrinsics = intrinsic_cams - config_cams
        if extra_intrinsics:
            logger.warning(
                "Camera(s) in intrinsics file but not listed in vehicle settings: %s",
                sorted(extra_intrinsics),
            )
    else:
        # Fallback: compare intrinsics <-> extrinsics directly, without any
        # name-prefix assumption to identify camera links.
        missing_extrinsics = intrinsic_cams - extrinsic_keys
        if missing_extrinsics:
            logger.warning(
                "Camera(s) in intrinsics file but missing from URDF extrinsics: %s",
                sorted(missing_extrinsics),
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_calibration(
    urdf_path: str,
    intrinsics_yaml_path: Optional[str],
    stage1_out: str,
    reference_frame: str = "center_of_gravity",
    sensor_links: Optional[List[str]] = None,
    camera_frame_ids: Optional[List[str]] = None,
    tf_static: Optional[TFStaticParser] = None,
) -> dict:
    """
    Parse URDF or recorded static transforms, produce a unified calibration dict, and write
    it to <stage1_out>/calibration.yaml.

    Parameters
    urdf_path             : path to the car URDF file, or "from_tf_static".
    intrinsics_yaml_path  : camera intrinsics YAML; optional when no cameras are exported.
    stage1_out            : directory where calibration.yaml is written.
    reference_frame       : used as the common reference for all extrinsic matrices.
    sensor_links          : explicit list of URDF link names to compute extrinsics for.
    camera_frame_ids      : subset of "sensor_links" that are cameras; used only to produce more precise validation warnings.
    tf_static             : accumulated /tf_static transforms when urdf_path is "from_tf_static".

    Returns
    dictionary with keys:
        "reference_frame" : str   (as supplied by the caller)
        "extrinsics"      : {link_name: [[4x4 float matrix as nested list]]}
        "intrinsics"      : {camera_name: {fx, fy, cx, cy, ...}}
    """
    logger.info("Extrinsics source        : %s", urdf_path)
    logger.info("Reference frame          : %s", reference_frame)

    if urdf_path == "from_tf_static":
        if tf_static is None:
            raise ValueError("from_tf_static requires static transforms loaded from the input bag")
        extrinsic_parser = tf_static
    else:
        extrinsic_parser = URDFParser(urdf_path)

    if sensor_links is None:
        sensor_links = sorted(tf_static.frames) if urdf_path == "from_tf_static" else _infer_sensor_links(extrinsic_parser)
    else:
        logger.info("Using %d sensor link(s) from vehicle settings.", len(sensor_links))

    # Compute T_sensor_to_reference_frame for every sensor.
    extrinsics: Dict[str, list] = {}
    for link in sensor_links:
        T = extrinsic_parser.get_transform(link, reference_frame)
        if T is not None:
            extrinsics[link] = T.tolist()
        else:
            if urdf_path == "from_tf_static":
                raise ValueError(f"No /tf_static transform path from '{link}' to reference frame '{reference_frame}'")
            logger.warning("Could not compute transform for '%s'; skipping.", link)

    intrinsics = {}
    if camera_frame_ids != [] and intrinsics_yaml_path:
        intrinsics = _parse_camera_intrinsics(intrinsics_yaml_path)
        _validate(extrinsics, intrinsics, camera_frame_ids=camera_frame_ids)
    elif camera_frame_ids:
        raise ValueError("input.calibration_yaml is required when cameras are exported")

    calibration = {
        # reference_frame: the coordinate frame in which all extrinsics are expressed.
        # The merged LiDAR point cloud is also expressed in this frame.
        "reference_frame": reference_frame,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }

    os.makedirs(stage1_out, exist_ok=True)
    out_path = os.path.join(stage1_out, "calibration.yaml")
    with open(out_path, "w") as fh:
        yaml.dump(calibration, fh, default_flow_style=False, sort_keys=True)

    logger.info(
        "Calibration written to %s  (%d sensors, %d cameras with intrinsics)",
        out_path, len(extrinsics), len(intrinsics),
    )
    return calibration


def load_tf_static_from_parser(parser, tf_static: TFStaticParser) -> None:
    """Accumulate static transforms from one bag split without modifying bag readers."""
    if not hasattr(parser, "reader") and "/tf_static" not in parser.topic_id:
        return
    messages = parser.get_messages(topics="/tf_static", start_time=0, end_time=int(np.iinfo(np.int64).max))
    for _, message in messages:
        tf_static.add_message(message)
