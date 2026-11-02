#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def bin_to_pcd(bin_folder_path: Path, pcd_folder_path: Path) -> None:

    bin_paths = sorted(bin_folder_path.glob("*.bin"))
    for bin_path in bin_paths:
        bin_name = bin_path.stem
        pcd_path = pcd_folder_path / Path(bin_name+".pcd")


        raw_data = np.fromfile(bin_path, dtype=np.float32)

        if raw_data.size % 4 != 0:
            raise ValueError(
                f"Invalid BIN size: found {raw_data.size} float32 values. "
                "Expected x, y, z, intensity for each point."
            )

        points = raw_data.reshape(-1, 4)
        num_points = points.shape[0]

        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z intensity\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {num_points}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {num_points}\n"
            "DATA binary\n"
        )

        pcd_path.parent.mkdir(parents=True, exist_ok=True)

        with pcd_path.open("wb") as pcd_file:
            pcd_file.write(header.encode("ascii"))
            points.tofile(pcd_file)

        print(f"Converted {num_points} points")
        # print(f"Input:  {bin_path}")
        # print(f"Output: {pcd_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one XYZI float32 BIN point cloud to PCD."
    )
    # parser.add_argument("input_bin_folder", type=Path)
    # parser.add_argument("output_pcd_folder", type=Path)

    # args = parser.parse_args()

    bin_to_pcd(
        # args.input_bin_folder.expanduser(),
        # args.output_pcd_folder.expanduser(),

        Path("/mnt/storage/PoliMOVE/DATASETS/LIDAR/IRIS/20240827_IMS_ontrack_01/TrainingDataset/bin"),
        Path("/mnt/storage/PoliMOVE/DATASETS/LIDAR/IRIS/conversion/20240827_IMS_ontrack_01/pcd")
    )


if __name__ == "__main__":
    main()