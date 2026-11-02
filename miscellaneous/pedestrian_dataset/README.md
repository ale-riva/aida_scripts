# bag_to_fusion_dataset

Converts ROS 2 BAG files (`.mcap` / `.db3`) containing camera and LiDAR data into camera-LiDAR fusion datasets in standard formats (KITTI, nuScenes, Waymo).

Supports **EVA** (7 cameras, 3 LiDARs), **Minerva** (6 cameras, 3 LiDARs) not tested yet.

## How It Works

The pipeline runs in two stages.

### Stage 1 (timestamp matching + data extraction)

Reads the BAG, matches camera and LiDAR timestamps, and saves the aligned data into an intermediate folder structure alongside a CSV timestamp table and metadata files. Parses also calibration files and writes a single `calibration.yaml` into the intermediate output folder.

#### Intermediate Output Folder Layout

```
<output_base_dir>/<bag_folder_name>/
├── samples/
│   ├── camera_0/       <YYYYMMDD_HHMMSS_ns>.png  ...
│   │   ...
│   ├── camera_6/       <YYYYMMDD_HHMMSS_ns>.png  ...
│   ├── lidar_front/    <YYYYMMDD_HHMMSS_ns>.pcd  ...
│   ├── lidar_left/     <YYYYMMDD_HHMMSS_ns>.pcd  ...
│   └── lidar_right/    <YYYYMMDD_HHMMSS_ns>.pcd  ...
├── calibration.yaml
├── ego_poses.csv
├── info_extraction.yaml
└── timestamps.csv
```

#### Coordinate Conventions in the Intermediate Representation

The reference frame for EVA is `center_of_gravity`. It has the same axis orientation as the vehicle body: **x = forward, y = left, z = up**.

| Data | Frame | Axes |
|---|---|---|
| `samples/camera_N/*.png` | (sensor capture, not rectified) | pixels |
| `samples/lidar_front/*.pcd` | `lidar_front` sensor frame | x = up, y = right, z = forward |
| `samples/lidar_left/*.pcd` | `lidar_left` sensor frame | x = up, y = right, z = forward |
| `samples/lidar_right/*.pcd` | `lidar_right` sensor frame | x = up, y = right, z = forward |
| `calibration.yaml` (`extrinsics`) | `reference_frame` | `reference_frame`, 4x4 homogeneous |
| `calibration.yaml` (`intrinsics`) | camera optical frame | fx, fy, cx, cy, distortion coefficients |
| `ego_poses.csv` | `map` → `center_of_gravity` | position (x,y,z, metres) + roll, pitch, yaw (radians)  |

PCD files are written directly from the raw `PointCloud2` message with no coordinate transformation applied. Stage 2 converters apply the extrinsic matrix from `calibration.yaml` to bring all clouds into the common reference frame before merging.

The `timestamps.csv` has one row per sample group:

```
lidar_front, lidar_left, lidar_right, camera_0, camera_1, ..., camera_6
```

Absent sensors (e.g. `camera_6` on Minerva) leave the cell empty; the column is always present.

PCD files preserve all original PointCloud2 fields (`x`, `y`, `z`, `intensity`, `ring`, `timestamp`, ...) with their original data types.

### Stage 2 (format conversion)

Reads the intermediate folder and converts it into the target dataset format(s). Multiple output formats can be requested in a single run; each produces a separate output directory named `<bag_folder_name>_<format>`.

## Annotations

The pipeline runs without annotations by default. To include ground-truth labels in the KITTI output, place a `labels/` directory inside the Stage-1 intermediate folder (alongside `samples/`) before running Stage 2.

If `labels/` is absent, `label_2/` files are written empty. If it is present, each label file is populated from the corresponding JSON annotations. Label conversion currently supports a single annotated object per frame.

> **Note:** For manual annotation, the following tools are recommended: for camera annotations, [Label Studio](https://labelstud.io/) or the MATLAB Video Labeler app; for LiDAR annotations, [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS) via the first four steps of the `pointcloud_dataset_generation` pipeline (located in this `polimove_tools` folder).

#### Annotation Folder Structure

LiDAR annotations come from the fused (merged) point cloud, under `labels/lidar/`. Annotations may be **partial** - e.g. only one camera and the fused LiDAR folder may be annotated; any sensor missing a `labels/<sensor>/` folder (or a JSON file for a given timestamp) is simply skipped.

```
<output_base_dir>/<bag_folder_name>/
└── labels/
    ├── camera_0/   <YYYYMMDD_HHMMSS_ns>.json  ...
    │   ...
    ├── camera_6/   <YYYYMMDD_HHMMSS_ns>.json  ...
    └── lidar/      <YYYYMMDD_HHMMSS_ns>.json  ...
```

Filenames match the corresponding sensor sample (same timestamp stem as the `.pcd` / `.png` files; the LiDAR timestamp used is always the primary LiDAR's, `lidar_front`). Each JSON file contains a list of annotation objects.

**Camera annotation format** (`labels/camera_N/<timestamp>.json`):
```json
[
  {
    "obj_id": "0",
    "obj_type": "Car",
    "bbox": {
      "position": { "x": 803, "y": 311 },
      "width": 245,
      "height": 135,
      "rotation": 0
    },
    "image_width": 1928,
    "image_height": 500
  }
]
```

`position` is the top-left corner of the bounding box in pixels; `rotation` is a clockwise rotation in degrees.

**LiDAR annotation format** (`labels/lidar/<timestamp>.json`):
```json
[
  {
    "obj_id": "0",
    "obj_type": "Car",
    "psr": {
      "position": { "x": 14.347, "y": -0.170, "z": -0.241 },
      "rotation": { "x": 0,      "y": 0,      "z": -0.053 },
      "scale":    { "x": 5.11,   "y": 2.0,    "z": 1.35   }
    }
  }
]
```

Position, rotation, and scale are expressed in the fused LiDAR frame (x = forward, y = left, z = up), relative to the center of gravity of the fused LiDAR system. `scale.x` = length, `scale.y` = width, `scale.z` = height (metres). `rotation.z` is the heading yaw (rotation about the fused frame's z-axis = vehicle vertical).

`obj_id` is matched across LiDAR and camera annotations to associate 3-D boxes with 2-D detections. If an object is not annotated in the reference camera, the 2-D bounding box is derived automatically by projecting the 3-D box onto the reference camera image.

## Dataset Inspection

Stage-1 intermediate representation can be inspected, three ways:

- **Manual inspection** (`ir_inspection/manual/`): step through frames one at a time, viewing the fused BEV point cloud and each camera image with the 3D annotations overlaid. Frames can be discarded, producing `timestamps_curated.csv`. Requires `inspection_config.yaml`.
- **Automatic inspection & balancing** (`ir_inspection/automatic/`): analyzes the whole dataset at once, displays plots of the 3D annotation distribution, and optionally generates a balanced `timestamps_balanced.csv` by removing frames according to configurable rules.
- **Multi-IR comparison** (`ir_inspection/automatic/`): displays the same statistical plots as automatic inspection, but computed over several Stage-1 datasets combined, as if they were a single unified dataset.

```bash
# Manual
python inspect_frame.py <stage1_dir>

# Automatic
python inspect_dataset.py <stage1_dir> --input timestamps.csv --config balancing_config.yaml

# Multi-IR comparison
python compare_datasets.py \
  --dataset <stage1_dir_a> timestamps.csv \
  --dataset <stage1_dir_b> timestamps.csv \
  --config balancing_config.yaml
```

## Installation

It is suggested to create a virtual environment on top of the Docker image and install the missing dependencies as follows:

```bash
pip install -r requirements.txt
```

## Configuration

All parameters are set in a YAML config file `config/config.yaml`. To add a new vehicle, create a new `settings_<vehicle>.yaml` file and reference it from a new config.

## Usage

```bash
# Run both stages (default)
python main.py --config config/config.yaml

# Run Stage 1 only
python main.py --config config/config.yaml --stage 1

# Run Stage 2 only (assumes Stage 1 output already exists)
python main.py --config config/config.yaml --stage 2
```

## Calibration Files

The pipeline expects the following two files to be provided externally alongside the BAG:

- `camera_calibrations_<vehicle>.param.yaml`: camera intrinsics (pinhole and fisheye supported)
- `<vehicle>.urdf`: sensor extrinsics

## Supported Output Formats

| Format | Status | Annotations | Multi-camera | Tested on |
|---|---|---|---|---|
| KITTI | Implemented | Yes | Yes | EVA |
| nuScenes | Implemented | Yes | Yes | EVA |
| Waymo | Not implemented | No | No | - |

### KITTI

All LiDAR sources are merged into a single point cloud expressed in the reference frame (`center_of_gravity`). Both pinhole and fisheye cameras are supported; images are undistorted and saved as PNG. Labels are written when a `labels/` directory is present in the Stage-1 output (see [Annotations](#annotations)); otherwise `label_2/` files are empty.

The output follows the standard KITTI mono layout: each sample is one `(frame, camera)` pair, so a single `image_2/<id>.png` and its matching `calib/`, `label_2/` and `velodyne/` entries all refer to exactly one camera.

Output layout:

```
<bag_folder_name>_kitti/
├── ImageSets/           # populated by generate_splits.py
├── training/
│   ├── image_2/         # one image per (frame, camera) sample
│   │   └── 000000.png  ...
│   ├── velodyne/        # merged point cloud: float32 x y z intensity.
│   │   └── 000000.bin  ... 
│   ├── calib/           # per-sample calib; active camera in P2.
│   │   └── 000000.txt  ...
│   └── label_2/         # empty or populated if labels/ is present
│       └── 000000.txt  ...
└── metadata/            # dataset summary + per-sample traceability
    ├── sample_mapping.csv
    ├── kitti_metadata.yaml
    ├── labeled.txt
    └── unlabeled.txt
```

With `N` selected cameras, sample indices interleave per frame, e.g. for `camera_0, camera_1, camera_2`:

```
000000 -> camera_0, frame_0
000001 -> camera_1, frame_0
000002 -> camera_2, frame_0
000003 -> camera_0, frame_1
000004 -> camera_1, frame_1
000005 -> camera_2, frame_1
...
```

If a selected camera has no valid timestamp for a frame, that sample is skipped.

#### Camera Selection

To choose which cameras to export set `kitti.cameras` in `config.yaml` or override it at runtime on the command line:
```bash
python main.py --config config.yaml --stage 2 --cameras camera_0 camera_2
```

#### Calibration files

Each `calib/<id>.txt` describes the single camera that produced the matching image: that camera occupies the standard `P2` slot, while `P0`, `P1` and `P3` are zero-filled (`0 0 ...`) for compatibility with DetMatch. `R0_rect` is the identity (the undistortion is baked into `image_2`/`K_new`, so no extra rectification rotation is needed) and `Tr_velo_to_cam` maps the merged velodyne cloud into that camera's optical frame.

> **Note:** The 2-D box in `label_2` is sourced from the sample's own camera and is expressed in the rectified image space (the same space as `image_2` and `P2`). If the object is not annotated in that camera, the 2-D box is derived by projecting the 3-D box onto that camera's rectified image.

#### Dataset splits

`ImageSets/` is left empty by the converter. Generate the splits with the standalone script:
```bash
python generate_splits_kitti.py --dataset <bag_folder_name>_kitti --val-ratio 0.2 --seed 42
```
This writes `train.txt`, `val.txt` and `trainval.txt`. `test.txt` and a `testing/` directory are intentionally not created.

#### Metadata

`metadata/` holds dataset summary and per-sample traceability:

- `sample_mapping.csv` - one row per `kitti_id`: `source_frame_index` (row in Stage-1's `timestamps.csv`), `is_labeled`, `camera_key`/`camera_name`/`camera_frame_id`/`camera_timestamp`, one `<lidar_key>_timestamp` per LiDAR, and `velodyne_source_id` (the `kitti_id` owning the real `velodyne/<id>.bin`).
- `kitti_metadata.yaml` - vehicle, location, date captured, camera selection order, merged LiDAR keys, and sample/label counts.
- `labeled.txt` / `unlabeled.txt` - KITTI ids split by whether `label_2/<id>.txt` has any objects.

### nuScenes

All LiDAR sources are merged into a single `LIDAR_TOP` cloud expressed in the reference frame. Both pinhole and fisheye cameras are supported; images are undistorted and saved as JPEG. The sensor-to-channel mapping is defined in `config/nuscenes_channel_map_<vehicle>.yaml`.

Output layout:

```
<bag_folder_name>_nuscenes/
├── maps/
│   └── map.png               # copied from track
├── v1.0-trainval/
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   └── visibility.json
├── samples/
│   ├── LIDAR_TOP/
│   │   └── <scene>__LIDAR_TOP__<timestamp_us>.pcd.bin
│   ├── CAM_FRONT/
│   │   └── <scene>__CAM_FRONT__<timestamp_us>.jpg
│   └── ...
└── sweeps/
    └── <channel>/            # empty (every frame is a key frame)
```

#### Timestamps File Selection

By default, Stage 2 reads `timestamps.csv` (every extracted frame). To convert from `timestamps_curated.csv` or `timestamps_balanced.csv` instead (see [Dataset Inspection](#dataset-inspection)), set `nuscenes.timestamps_file` in `config.yaml`.

#### Dataset splits

To generate the splits use the standalone script:
```bash
python generate_splits_nuscenes.py --dataset <bag_folder_name>_nuscenes --val-ratio 0.2 --seed 42
```
This writes the `splits.json` file.

#### Merging Multiple Intermediate Representations

A single nuScenes dataset can be produced from several Stage-1 intermediate folders (e.g. multiple bags recorded on different days). List the additional folders under `nuscenes.merge_datasets` in `config.yaml`:

```yaml
nuscenes:
  merge_datasets:
    - stage1_dir: "/data/2025-10-04_11-53_lidar_radar_camera"
      timestamps_file: "timestamps.csv"
    - stage1_dir: "/data/2025-10-08_10-21_lidar_radar_camera"
      timestamps_file: "timestamps_curated.csv"
      location: "Monza"       # optional - defaults to scene.location
      description: ""         # optional - defaults to scene.description
```

The primary bag (`input.bag_dir`) plus every entry in `merge_datasets` each become one nuScenes `log`; all logs are merged into the single dataset at `output.base_dir`.

> **Assumption:** `category`, `attribute`, `sensor`, and `visibility` definitions are identical across all merged recordings. `calibrated_sensor` is rebuilt per log from that log's own `calibration.yaml` (extrinsics/intrinsics can differ between recording sessions). `log` and `map` entries remain one per merged folder.

#### Merging Already-Built nuScenes Datasets

The `merge_nuscenes_datasets.py` script merges two or more nuScenes datasets:

```bash
python merge_nuscenes_datasets.py \
  --inputs <bag_folder_name_a>_nuscenes <bag_folder_name_b>_nuscenes \
  --output <merged_output_dir>
```

It produces the same result as running Stage 2 once with all the underlying Stage-1 folders listed together in `nuscenes.merge_datasets`, but works straight from the finished nuScenes folders.