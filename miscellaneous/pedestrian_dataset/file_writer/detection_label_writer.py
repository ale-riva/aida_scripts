"""Write detected 3-D boxes in the Stage-1 labels/lidar/ JSON format."""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from utils.timestamp_utils import ns_to_filename

_CLASS_NAMES = {
    0: "Pedestrian", 1: "Car", 2: "Truck", 3: "Bus", 4: "Motorcycle", 5: "Bicycle",
}


def write_detection_labels(detections, timestamp_ns: int, output_dir: str, on_existing: str = "skip", transform=None) -> bool:
    """Write grouped (message, bbox) pairs, optionally transformed into the merge reference."""
    out_path = os.path.join(output_dir, "labels", "lidar", f"{ns_to_filename(timestamp_ns)}.json")
    if os.path.exists(out_path):
        if on_existing == "skip":
            return False
        elif on_existing == "abort":
            raise FileExistsError(f"Output file already exists: {out_path}")

    labels = []
    matrix = np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64)
    for detection, bbox in detections:
        x, y, z, length, width, height, yaw = [float(value) for value in bbox]
        x, y, z = (matrix[:3, :3] @ [x, y, z] + matrix[:3, 3]).tolist()
        roll, pitch, yaw = (Rotation.from_matrix(matrix[:3, :3]) * Rotation.from_euler("z", yaw)).as_euler("xyz").tolist()
        class_id = int(detection.meas_class)
        labels.append({
            "obj_id": str(detection.box_id),
            "obj_type": _CLASS_NAMES.get(class_id, str(class_id)),
            "psr": {
                "position": {"x": x, "y": y, "z": z},
                "rotation": {"x": roll, "y": pitch, "z": yaw},
                "scale": {"x": length, "y": width, "z": height},
            },
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(labels, fh, indent=2, allow_nan=False)
    return True
