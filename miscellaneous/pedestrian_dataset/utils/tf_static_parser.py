"""Resolve sensor-to-reference transforms from recorded /tf_static messages."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


class TFStaticParser:
    """Provide the same get_transform(source, target) convention as URDFParser."""

    def __init__(self):
        self.frames = {}

    def add_message(self, tf_msg) -> None:
        for transform in tf_msg.transforms:
            parent = transform.header.frame_id.lstrip("/")
            child = transform.child_frame_id.lstrip("/")
            if not parent or not child or parent == child:
                raise ValueError("Static transforms must have distinct, nonempty parent and child frames")
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()
            matrix[:3, 3] = [translation.x, translation.y, translation.z]
            if not np.isfinite(matrix).all():
                raise ValueError(f"Nonfinite static transform: {parent} -> {child}")
            # TF supplies child -> parent. Store both directions for chained lookup.
            previous = self.frames.get(child, {}).get(parent)
            if previous is not None and not np.allclose(previous, matrix, rtol=0, atol=1e-8):
                raise ValueError(f"Conflicting static transforms: {parent} -> {child}")
            self.frames.setdefault(child, {})[parent] = matrix
            self.frames.setdefault(parent, {})[child] = np.linalg.inv(matrix)

    def get_transform(self, source: str, target: str) -> Optional[np.ndarray]:
        source, target = source.lstrip("/"), target.lstrip("/")
        if source == target:
            return np.eye(4, dtype=np.float64)
        pending = deque([(source, np.eye(4, dtype=np.float64))])
        visited = {source}
        while pending:
            current, current_from_source = pending.popleft()
            for neighbor, neighbor_from_current in self.frames.get(current, {}).items():
                if neighbor in visited:
                    continue
                neighbor_from_source = neighbor_from_current @ current_from_source
                if neighbor == target:
                    return neighbor_from_source
                visited.add(neighbor)
                pending.append((neighbor, neighbor_from_source))
        return None
