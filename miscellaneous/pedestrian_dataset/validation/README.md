# Dataset Validation Framework

Two **independent** validation pipelines with a shared `common/` layer:

| Pipeline | Validates | Entry point |
|---|---|---|
| `intermediate/` | the Stage-1 output (intermediate representation) | `validate_intermediate` |
| `kitti/` | the generated KITTI dataset | `validate_kitti` |
| `nuscenes/` | TODO | TODO |



Each pipeline prints a console summary, writes a YAML report, and saves diagnostic figures.

## 1. Validate the intermediate representation

```bash
python -m validation.intermediate.validate_intermediate \
    --stage1 /path/to/output/<bag_name> \
    --config config/config.yaml
```

Checks: directory/file structure and metadata consistency, sensor
synchronization,
calibration validity (intrinsics + extrinsics) and coordinate transforms,
ego poses, labels (when a `labels/` directory is present).

Figures (written to `<stage1>/validation_figures_intermediate/`):
- `ir_dt_histograms.png`, `ir_gap_timeline.png`, `ir_ego_trajectory.png`.
- `ir_projection_frame_*_<camera>.png`: merged LiDAR cloud projected onto each camera image.
- `ir_labels_frame_*_<camera>.png`: camera 2-D boxes (green) + LiDAR 3-D boxes
  projected onto the image (orange).

## 2. Validate the KITTI dataset

```bash
python -m validation.kitti.validate_kitti \
    --kitti  /path/to/output/<bag_name>_kitti \
    --stage1 /path/to/output/<bag_name> \
    --config config/config.yaml
```

Checks: official KITTI layout, `.bin` sizes, file-count parity and id
contiguity across `image_2/ calib/ label_2/ velodyne/`, `ImageSets/` splits when
present, calibration (`P0/P1/P3` zero-filled, `R0_rect` identity,
`P2` matching a camera's intrinsics, `Tr_velo_to_cam` matching the recomputed
transform), consistency with the intermediate representation, `label_2` schema +
2-D/3-D overlays on `image_2`.

Figures: `<kitti>/validation_figures_kitti/kitti_labels_frame_*.png`.

## Options (both pipelines)

- `--frames i j k`: frame indices to inspect specific images (default: 3).
- `--cameras camera_0 camera_2` (KITTI only): the camera selection used.
- `--out /path/to/report_dir`: where to write the report + figures (default: the validated dir).
- `--show-passes`: list passing checks too (default shows only warn/fail/skip).

### Running without `--config`

`--config` is optional. Without it the sensor model
(key <-> name <-> frame_id, tolerances, camera selection) is unknown, so projection /
label overlays, parameter cross-checks, and the KITTI <-> IR consistency checks are
skipped and only structural checks run.