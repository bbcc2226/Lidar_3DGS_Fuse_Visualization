# KITTI Scripts

These scripts adapt KITTI raw drives and LiDAR SLAM results for this repository,
evaluate optimized trajectories against KITTI OXTS, and generate LiDAR-based 3DGS
initialization data. Run commands from the repository root.

## Script status

| Script | Status | Purpose |
|---|---|---|
| `prepare_kitti_pipeline.py` | Reusable | Prepare timestamps, optimized LIO poses, rectified intrinsics, camera-LiDAR extrinsic, associations, and a C++ pipeline config. |
| `evaluate_kitti_poses.py` | Reusable | Compare TUM camera trajectories with KITTI OXTS using rigid alignment without scale. |
| `merge_camera_corrected_lidar_rgb.py` | Reusable | Transform local LiDAR scans with optimized camera poses and color them from rectified images. |
| `compare_lidar_poses.py` | Reusable diagnostic | Convert optimized camera poses back to LiDAR poses and compare them with raw and optimized LIO references. |
| `report_kitti_run.py` | C++-pipeline diagnostic | Produce per-frame landmark coverage and BA reports from the richer root C++ output format. |

The first four scripts are generally useful. `report_kitti_run.py` remains
useful for the root C++ pipeline but expects artifacts that the Python workflow
does not export.

## Prepare a KITTI drive

Configure `kitti_prepare.json` with the KITTI sequence, LiDAR SLAM directory,
camera, image range, and `pose_field`. Use `optimized_pose` for the optimized LIO
trajectory. Keep `camera_time_offset_seconds` at `0.0` when both streams already
share the same time basis.

```bash
python3 scripts/kitti/prepare_kitti_pipeline.py \
  --config kitti_prepare.json \
  --prepare-only
```

The prepared directory is range-specific, for example:

```text
output/kitti_prepared/image_03_0000000070_0000000500/
```

It contains `timestamps.txt`, `key_frames.jsonl`, `intrinsics.txt`,
`camera_lidar_extrinsic.json`, `image_lidar_association.csv`,
`preparation_summary.json`, and `pipeline.yaml`.

## Run the two-round Python workflow

Adapt the prepared directory to the legacy Python input layout:

```bash
P=output/kitti_prepared/image_03_0000000070_0000000500
D=data_kitti_70_500
SEQ=/path/to/2011_10_03_drive_0027_sync

rm -rf "$D"
mkdir -p "$D"
grep -v '^#' "$P/timestamps.txt" > "$D/timestamps.txt"
cp "$P/key_frames.jsonl" "$P/intrinsics.txt" "$D/"
cp "$P/camera_lidar_extrinsic.json" "$D/tuned_camera_lidar_extrinsic.json"
ln -s "$SEQ/image_03/data" "$D/undistorted"
```

Run both rounds:

```bash
./scripts/python_version/run_multiround_pipeline.sh \
  --data data_kitti_70_500 \
  --first-output output/kitti_0027_70_500_multiround_first \
  --output output/kitti_0027_70_500_multiround_final \
  --max-images -1 \
  --translation-prior-weight 100 \
  --rotation-prior-weight 10 \
  --force
```

This performs pose initialization, triangulation, first-round BA, validation,
track extension, second-round BA, and final validation. No time offset option is
used; camera poses are initialized by direct interpolation at image timestamps.

## Evaluate against OXTS

```bash
python3 scripts/kitti/evaluate_kitti_poses.py \
  --sequence /path/to/2011_10_03_drive_0027_sync \
  --camera image_03 \
  --start-id 70 \
  --poses \
    prior=output/kitti_0027_70_500_multiround_first/poses_lio_prior_tum.txt \
    round1=output/kitti_0027_70_500_multiround_first/poses_optimized_tum.txt \
    round2=output/kitti_0027_70_500_multiround_final/poses_optimized_tum.txt \
  --json output/kitti_0027_70_500_multiround_oxts.json
```

The report includes rigidly aligned ATE, orientation error, one-frame relative
pose error, and position second-difference jitter. OXTS is an independent KITTI
reference, but it is not survey-grade camera ground truth.

## Merge LiDAR with final poses

`merge_camera_corrected_lidar_rgb.py` expects a dataset package containing
`images/` and `provenance/`. The provenance directory must contain:

```text
preparation_summary.json
intrinsics.txt
camera_lidar_extrinsic.json
key_frames.jsonl
timestamps.txt
poses_optimized_tum.txt
```

Then run:

```bash
python3 scripts/kitti/merge_camera_corrected_lidar_rgb.py \
  --dataset output/kitti_0027_70_500_multiround_final_dataset \
  --output output/kitti_0027_70_500_lidar_rgb_multiround_final \
  --voxel-size 0.05 \
  --maximum-image-dt 0.30 \
  --maximum-images-per-scan 5
```

The main products are `merged_lidar_full.ply`, `merged_lidar_rgb.ply`,
`scan_manifest.csv`, projection previews, and `summary.json`.

## Create points3D.txt initialization

The generic LiDAR initializer remains under `tools/` because it is not specific
to KITTI:

```bash
python3 tools/lidar_init_3dgs_dataset.py \
  --reconstruction output/kitti_0027_70_500_multiround_final \
  --lidar-ply output/kitti_0027_70_500_lidar_rgb_multiround_final/merged_lidar_rgb.ply \
  --images data_kitti_70_500/undistorted \
  --output output/kitti_0027_70_500_multiround_final_3dgs_lidar_init \
  --max-camera-distance 60.0 \
  --max-knn-distance 0.5
```

This keeps the final optimized `cameras.txt` and pose-only `images.txt`, replaces
visual SfM points with filtered RGB LiDAR points in `points3D.txt`, and links the
training images.

## Generated files versus scripts

Directories under `output/`, feature caches, logs, plots, reports, and prepared
range directories are generated artifacts. They can be regenerated when the
source images, LIO results, calibration, and configuration are available.

The scripts in this folder are reusable source utilities and should be kept.
