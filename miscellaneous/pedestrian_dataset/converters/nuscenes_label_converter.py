"""
nuScenes label converter: Stage-1 JSON annotations -> nuScenes sample_annotations.json format.

LiDAR annotations are expressed in the fused / CoG / ego frame:
    x = forward
    y = left
    z = up

nuScenes sample_annotation requires:
    translation in global frame
    rotation in global frame
    size in [width, length, height]

Ego pose convention:
    ego_translation: ego position in global frame [x, y, z]
    ego_rotation:    ego orientation in global frame as nuScenes quaternion [w, x, y, z]
"""

from __future__ import annotations
import os
import json
import logging
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from utils.pcd_reader import get_num_points_in_bbox

logger = logging.getLogger(__name__)


def _get_pos_rot_size(det: dict) -> Tuple[List[float], List[float], List[float]]:
    """
    load sample annotation (intermediate representation)
    Input convention:
        position: CoG frame [x, y, z]
        rotation: roll, pitch, yaw CoG frame [rad]
        scale:    length, width, height
    """

    pos_x = float(det["psr"]["position"]["x"])
    pos_y = float(det["psr"]["position"]["y"])
    pos_z = float(det["psr"]["position"]["z"])
    position = [pos_x, pos_y, pos_z]

    rot_x = float(det["psr"]["rotation"]["x"])
    rot_y = float(det["psr"]["rotation"]["y"])
    rot_z = float(det["psr"]["rotation"]["z"])
    rotation = [rot_x, rot_y, rot_z]
    
    scale_x = float(det["psr"]["scale"]["x"])  # length
    scale_y = float(det["psr"]["scale"]["y"])  # width
    scale_z = float(det["psr"]["scale"]["z"])  # height
    scale = [scale_x, scale_y, scale_z]


    return position, rotation, scale

def _extract_labels_from_json(label_path: str) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Read Stage-1 intermediate labels.

    Returns:
        pos_list:   List([x, y, z]) CoG frame
        rot_list:   List([roll, pitch, yaw]) CoG frame
        scale_list: List([length, width, height])
    """

    if not os.path.isfile(label_path):
        logger.debug("[nuScenes Converter] Label file not found: %s", label_path)
        return [], [], []

    with open(label_path, "r") as f:
        labels = json.load(f)

    pos_list: List[List[float]] = []
    rot_list: List[List[float]] = []
    scale_list: List[List[float]] = []

    for det in labels:
        position, rotation, scale = _get_pos_rot_size(det)

        # check sizes of translation, rotation, size
        if len(scale) != 3:
            raise ValueError(
                f"[nuScenes Converter] incorrect dimension of size input: "
                f"expected 3, got {len(scale)}"
            )

        if len(position) != 3:
            raise ValueError(
                f"[nuScenes Converter] incorrect dimension of translation input: "
                f"expected 3, got {len(position)}"
            )

        if len(rotation) != 3:
            raise ValueError(
                f"[nuScenes Converter] incorrect dimension of rotation input: "
                f"expected 3, got {len(rotation)}"
            )

        pos_list.append(position)
        rot_list.append(rotation)
        scale_list.append(scale)


    return pos_list, rot_list, scale_list


def _normalize_quaternion_wxyz(q_wxyz: List[float]) -> List[float]:
    """
    Normalize a quaternion in nuScenes order [w, x, y, z].
    """

    if len(q_wxyz) != 4:
        raise ValueError(
            f"[nuScenes Converter] ego_rotation must have 4 elements [w, x, y, z], got {len(q_wxyz)}"
        )

    q = np.asarray(q_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm <= 0.0:
        raise ValueError(
            f"[nuScenes Converter] Invalid zero-norm ego quaternion: {q_wxyz}"
        )

    return (q / norm).tolist()


def _transform_box_to_global(
    bbox_pos: List[float],
    bbox_rot_rpy: List[float],
    ego_pos: List[float],
    ego_rot_wxyz: List[float],
) -> Tuple[List[float], List[float]]:
    """
    Transform a 3D box pose from ego / CoG frame to global frame.

    Inputs:
        bbox_pos:
            box center in ego / CoG frame, [x, y, z]

        bbox_rot_rpy:
            box roll, pitch, yaw in ego / CoG frame, radians

        ego_pos:
            ego position in global frame, [x, y, z]

        ego_rot_wxyz:
            ego orientation in global frame, nuScenes quaternion [w, x, y, z]

    Returns:
        position_global:
            box center in global frame

        rotation_global_quat:
            box orientation in global frame, nuScenes quaternion [w, x, y, z]
    """

    if len(bbox_pos) != 3:
        raise ValueError(
            f"[nuScenes Converter] position_ego must have 3 elements, got {len(bbox_pos)}"
        )

    if len(bbox_rot_rpy) != 3:
        raise ValueError(
            f"[nuScenes Converter] rotation_ego_rpy must have 3 elements, got {len(bbox_rot_rpy)}"
        )

    if len(ego_pos) != 3:
        raise ValueError(
            f"[nuScenes Converter] ego_translation must have 3 elements, got {len(ego_pos)}"
        )

    bbox_pos_np = np.asarray(bbox_pos, dtype=np.float64)
    ego_pos_np = np.asarray(ego_pos, dtype=np.float64)

    ego_rot_wxyz = _normalize_quaternion_wxyz(ego_rot_wxyz)

    # scipy accepts [w, x, y, z] when scalar_first=True.
    ego_rot = Rotation.from_quat(ego_rot_wxyz, scalar_first=True)

    # Label rotation is roll/pitch/yaw in ego frame.
    bbox_rot = Rotation.from_euler("xyz", bbox_rot_rpy)

    # Position:
    #   p_global = R_global_ego * p_ego + t_global_ego
    bbox_pos_global = ego_rot.apply(bbox_pos_np) + ego_pos_np

    # Orientation:
    #   R_global_box = R_global_ego * R_ego_box
    bbox_rot_global = ego_rot * bbox_rot

    bbox_rot_global_wxyz = bbox_rot_global.as_quat(scalar_first=True)

    return bbox_pos_global.tolist(), bbox_rot_global_wxyz.tolist()


def _size_lwh_to_wlh(size_lwh: List[float]) -> List[float]:
    """
    Convert internal size convention [length, width, height]
    to nuScenes size convention [width, length, height].
    """

    if len(size_lwh) != 3:
        raise ValueError(
            f"[nuScenes Converter] size_lwh must have 3 elements, got {len(size_lwh)}"
        )

    length = float(size_lwh[0])
    width = float(size_lwh[1])
    height = float(size_lwh[2])

    return [width, length, height]


def convert_label_to_nus(
    tok: str,
    sample_token: str,
    instance_token: str,
    attribute_tokens: List[str],
    visibility_token: str,
    label_uri: str,
    lidar_sample_uri: Optional[str],
    ego_translation: List[float],
    ego_rotation: List[float],
    num_radar_pts: Optional[int] = None,
    next_tok: Optional[str] = None,
    prev_tok: Optional[str] = None,
) -> List[dict]:
    """
    Convert Stage-1 3D labels into nuScenes sample_annotation entries.

    Important:
        - num_lidar_pts is computed in ego frame, because the stored LiDAR points
          are in the fused / CoG / ego frame.
        - sample_annotation.translation and sample_annotation.rotation are written
          in global frame, as required by nuScenes.
        - ego_rotation must be a nuScenes quaternion [w, x, y, z].
    """
    translation_list, rotation_list, size_list =  _extract_labels_from_json(label_uri)

    n_detections = len(translation_list)
    sample_annotations: List[dict] = []
    if n_detections ==0:
        return sample_annotations

    for translation, rotation, size in zip(translation_list, rotation_list, size_list):

        # compute number of lidar points in bbox
        if lidar_sample_uri:
            # for nuscenes, each point has [x, y, z, intensity, ring]
            num_lidar_pts = get_num_points_in_bbox(lidar_sample_uri, translation, size, rotation, n_fields_per_point=5)
        else:
            num_lidar_pts = 0

        translation_global, rotation_global_qt = _transform_box_to_global(translation,rotation,
            ego_translation,ego_rotation)

        size_wlh = _size_lwh_to_wlh(size)

        ann_tok = tok

        sample_annotation = {
            "token": ann_tok,
            "sample_token": sample_token,
            "instance_token": instance_token,
            "attribute_tokens": attribute_tokens,
            "visibility_token": visibility_token,
            "translation": translation_global,
            "size": size_wlh,
            "rotation": rotation_global_qt,
            "num_lidar_pts": int(num_lidar_pts),
            "num_radar_pts": int(num_radar_pts) if num_radar_pts is not None else 0,
            "next": str(next_tok) if next_tok is not None else "",
            "prev": str(prev_tok) if prev_tok is not None else "",
        }

        sample_annotations.append(sample_annotation)

    return sample_annotations