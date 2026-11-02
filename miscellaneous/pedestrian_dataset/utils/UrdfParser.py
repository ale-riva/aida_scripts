#!/usr/bin/env python3
"""
Parse URDF file and compute transformation matrices between frames.
"""

import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Optional
from scipy.spatial.transform import Rotation as R


class URDFParser:
    """Parse URDF file and build kinematic chain."""

    def __init__(self, urdf_file: str):
        """
        Initialize URDF parser.

        Args:
            urdf_file: Path to URDF file
        """
        self.urdf_file = urdf_file
        self.links: Dict[str, Dict] = {}
        self.joints: Dict[str, Dict] = {}
        self.parent_map: Dict[str, str] = {}          # child  -> parent
        self.children_map: Dict[str, List[str]] = {}  # parent -> [children]
        self.root_link: Optional[str] = None

        self._parse_urdf()

    def _parse_urdf(self) -> None:
        """Parse URDF XML file."""
        tree = ET.parse(self.urdf_file)
        root = tree.getroot()

        # Parse all links
        for link in root.findall('link'):
            link_name = link.get('name')
            self.links[link_name] = {'name': link_name}

        # Parse all joints
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')

            parent_elem = joint.find('parent')
            child_elem  = joint.find('child')
            origin_elem = joint.find('origin')

            if parent_elem is not None and child_elem is not None:
                parent_link = parent_elem.get('link')
                child_link  = child_elem.get('link')

                # Parse origin (position and orientation)
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]

                if origin_elem is not None:
                    xyz_str = origin_elem.get('xyz', '0.0 0.0 0.0')
                    rpy_str = origin_elem.get('rpy', '0.0 0.0 0.0')
                    xyz = list(map(float, xyz_str.split()))
                    rpy = list(map(float, rpy_str.split()))

                self.joints[joint_name] = {
                    'name':   joint_name,
                    'type':   joint_type,
                    'parent': parent_link,
                    'child':  child_link,
                    'xyz':    xyz,
                    'rpy':    rpy,
                }

                # Build parent-child mapping
                self.parent_map[child_link] = parent_link
                if parent_link not in self.children_map:
                    self.children_map[parent_link] = []
                self.children_map[parent_link].append(child_link)

        # Find root link (link with no parent)
        for link_name in self.links:
            if link_name not in self.parent_map:
                self.root_link = link_name
                break

    def _xyz_rpy_to_matrix(self, xyz: List[float], rpy: List[float]) -> np.ndarray:
        """
        Convert XYZ position and RPY rotation to 4x4 transformation matrix.

        Args:
            xyz: [x, y, z] position of child origin expressed in parent frame
            rpy: [roll, pitch, yaw] extrinsic XYZ rotations in radians
                 (ROS/URDF convention: lowercase 'xyz' = extrinsic = Rz @ Ry @ Rx)

        Returns:
            4x4 float64 homogeneous transformation matrix
        """
        # Create rotation matrix from RPY
        rot = R.from_euler('xyz', rpy)   # extrinsic XYZ -> Rz @ Ry @ Rx
        rotation_matrix = rot.as_matrix()

        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3]  = xyz

        return T

    def _find_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find path between two frames using BFS.

        Returns:
            Ordered list of link names [source, ..., target], or None.
        """
        if source == target:
            return []
        
        # BFS to find path
        from collections import deque

        queue   = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            # Check children
            if current in self.children_map:
                for child in self.children_map[current]:
                    if child == target:
                        return path + [child]
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, path + [child]))

            # Check parent
            if current in self.parent_map:
                parent = self.parent_map[current]
                if parent == target:
                    return path + [parent]
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, path + [parent]))

        return None

    def get_transform(self, source: str, target: str) -> Optional[np.ndarray]:
        """
        Return the 4x4 matrix that maps points from "source" frame to "target" frame

        Args:
            source: name of the source link
            target: name of the target link

        Returns:
            4x4 float64 ndarray, or None if a frame is unknown or no path exists.
        """
        # Validate frames exist
        if source not in self.links or target not in self.links:
            print(f"Error: frame '{source}' or '{target}' not found in URDF")
            return None
        
        # Find path between frames
        path = self._find_path(source, target)
        if path is None:
            print(f"Error: no path found between '{source}' and '{target}'")
            return None
        
        # Compute cumulative transformation
        T_total = np.eye(4)

        # Forward direction (parent -> child)
        for i in range(len(path) - 1):
            current    = path[i]
            next_frame = path[i + 1]

            # Find joint connecting current to next
            joint_found = False

            # Going UP the tree: next_frame is the parent of current.
            # T_joint = T_child→parent; apply directly so that
            # T_total maps FROM source TO next_frame (parent).
            if (current in self.parent_map and self.parent_map[current] == next_frame):
                for jd in self.joints.values():
                    if jd['parent'] == next_frame and jd['child'] == current:
                        T = self._xyz_rpy_to_matrix(jd['xyz'], jd['rpy'])
                        T_total = T @ T_total
                        joint_found = True
                        break

            # Going DOWN the tree: next_frame is a child of current.
            # T_joint = T_child→parent; invert it so that
            # T_total maps FROM source TO next_frame (child).
            elif (current in self.children_map and next_frame in self.children_map[current]):
                for jd in self.joints.values():
                    if jd['parent'] == current and jd['child'] == next_frame:
                        T = self._xyz_rpy_to_matrix(jd['xyz'], jd['rpy'])
                        T_total = np.linalg.inv(T) @ T_total
                        joint_found = True
                        break

            if not joint_found:
                print(f"Error: could not find joint between '{current}' and '{next_frame}'")
                print(f"  Current parent_map[{current}] = {self.parent_map.get(current, 'None')}")
                print(f"  Children of {current}: {self.children_map.get(current, [])}")
                return None

        return T_total

    def list_frames(self) -> List[str]:
        """Return a sorted list of all link names."""
        return sorted(self.links.keys())


def main():
    """Example usage."""
    import sys

    # Usage: python urdf_transform.py <urdf_file> <source_frame> <target_frame>
    if len(sys.argv) < 4:
        print("Usage: python UrdfParser.py <urdf_file> <source_frame> <target_frame>")
        print("Example: python UrdfParser.py eva.urdf lidar_front ring_center")
        sys.exit(1)

    urdf_file    = sys.argv[1]
    source_frame = sys.argv[2]
    target_frame = sys.argv[3]

    # Parse URDF
    parser = URDFParser(urdf_file)

    # List available frames
    print("Available frames:")
    for frame in parser.list_frames():
        print(f"  - {frame}")
    print()

    # Get transformation
    print(f"Computing transform: '{source_frame}' → '{target_frame}' ...")
    T = parser.get_transform(source_frame, target_frame)

    if T is not None:
        print("\nTransformation matrix (4x4):")
        print(T)
        print("\nFormatted:")
        print(f"  Position [m]: {T[:3, 3]}")
        print(f"  Rotation matrix:\n{T[:3, :3]}")
        rot = R.from_matrix(T[:3, :3])
        print(f"  RPY [rad]: {rot.as_euler('xyz')}")
        print(f"  RPY [deg]: {np.degrees(rot.as_euler('xyz'))}")


if __name__ == '__main__':
    main()