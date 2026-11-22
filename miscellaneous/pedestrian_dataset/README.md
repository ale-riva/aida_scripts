# Pedestrian dataset extraction

Export LiDAR frames matched to 3-D detections, with optional camera images.

```bash
python main.py config/clusters.yaml
```

Set the bag path, sensor settings, and output options in [clusters.yaml](config/clusters.yaml). The input can be an MCAP or SQLite (`.db3`) bag file or a directory of bag splits.

## Expected bag topics

| Data | When needed |
| --- | --- |
| 3-D detections (`move_msgs/msg/Detector3DMeasList`) | Every run. The example uses `/perception/multimodal_3d_detections`. |
| LiDAR (`sensor_msgs/msg/PointCloud2`) | When reading samples from the bag. Clouds and detections must share a coordinate frame. |
| Ego pose (`tf2_msgs/msg/TFMessage`) | Record the configured parent-to-child transform on the ego-pose topic to populate the pose CSV. |
| Camera images (`sensor_msgs/msg/Image` or `CompressedImage`) | For selected cameras when reading samples from the bag. |
| Calibration (`/tf_static` and camera `CameraInfo` topics) | When the config selects recorded transforms or camera intrinsics. A URDF and camera calibration YAML can supply these instead. |

## Output

`output.base_dir` contains full-cloud PCDs in `samples/<lidar_key>/` or cropped PCDs in `clusters/<class_id>/`, optional PNGs in `samples/<camera_key>/`, and 3-D box labels in `labels/lidar/`. It also holds `calibration.yaml`, `timestamps.csv`, `ego_poses.csv`, `detection_confidences.csv`, and `detection_point_counts.csv`.

## Supported workflows

- Sample detection-matched frames at a configured interval by setting both `input.reuse_dataset` paths to `""`. Set only `timestamps_csv` to select exact sensor timestamps from the bag.
- Filter detections by class, export full clouds or per-detection crops, and select which cameras to include.
- Reuse a prior export by setting both `input.reuse_dataset.timestamps_csv` and `input.reuse_dataset.samples`. The `samples` path points to the prior export root, which supplies samples, calibration, and ego poses. The bag supplies fresh detections. Samples are linked by default and copied when `output.copy_samples` is enabled.
- Prepare labelling sessions with `create_cam_lidar_labelling_session.py`, export edited classes with `export_label_session_csv.py`, or inspect and balance datasets with the scripts in `ir_inspection/`.
