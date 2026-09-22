# LIO-prior visual camera poses for 3DGS Python version

This directory contains a LIO-based visual SLAM workflow for optimizing camera
poses and generating training data for 3D Gaussian Splatting (3DGS). It uses a
calibrated image sequence, LIO poses, camera intrinsics, and camera/LiDAR
calibration to produce an optimized camera trajectory and a sparse 3D
reconstruction at metric scale.

The final output includes the camera poses, 3D points, and COLMAP-compatible
files needed to initialize 3DGS training. The recommended entry point is the
multiround pipeline, which first builds an initial reconstruction and then
refines the camera poses and geometry.

## What the pipeline does

The pipeline uses the LIO trajectory as the camera-pose prior, applies the
camera/LiDAR calibration, and optimizes the poses and sparse scene geometry
together. This preserves metric scale while producing a consistent camera model
and 3DGS-ready training data.

```text
calibrated images + LIO poses + camera/LiDAR calibration
            |
            v
  LIO-based camera pose initialization
    [lio_camera_pose.cpp]
            |
            v
  Feature detection, matching, and track building
    [triangulate_sparse.py]
       |
       v
  Geometric filtering and landmark triangulation
    [triangulate_sparse.py]
       |
       v
  First-round pose and landmark optimization
    [bundle_adjust.cpp]
       |
       v
  Track extension using optimized poses
    [extend_tracks.py]
       |
       v
  Second-round pose and landmark optimization
    [bundle_adjust.cpp]
       |
       v
  Validation and 3DGS model export
    [test_outputs.py]
            |
            v
        optimized poses + sparse 3D model
            |
            v
          3DGS training data
```

## Required inputs

The input directory uses the repository's legacy `data/` layout:

- `timestamps.txt`: one image filename and timestamp per row;
- `undistorted/`: calibrated, undistorted or already-rectified images;
- `intrinsics.txt`: the $3 \times 3$ camera intrinsic matrix;
- `key_frames.jsonl`: timestamped LIO poses and LiDAR sweep references; and
- `tuned_camera_lidar_extrinsic.json`: the rigid camera/LiDAR calibration.

The camera and LIO timestamps must share the same clock. Camera initialization
uses direct interpolation, $t_{\text{lidar}} = t_{\text{image}}$, and the pose
stored in the prepared `lio_pose` field. KITTI preparation can promote
`optimized_pose` into that field before this workflow runs.

## Main outputs

- `poses_lio_prior_tum.txt`: initial LIO-based camera-pose trajectory.
- `poses_optimized_tum.txt`: optimized camera trajectory in TUM format.
- `landmarks_optimized.ply`: optimized sparse scene geometry.
- `intrinsics_refined.txt`: refined camera intrinsics.
- `evaluation.json`: reprojection, pose-prior, and held-out validation metrics.
- `sparse/0/`: COLMAP-compatible cameras, images, and 3D points for 3DGS.
- `cache/`: reusable feature and matching data for repeated runs.

Together, the optimized poses, input images, and `sparse/0/` form the 3DGS
training data. Optional utilities can additionally generate diagnostics,
filtered 3DGS packages, LiDAR depth supervision, or dense point clouds.

This implementation remains useful for reproducing the Python/Ceres experiments
and for the multiround KITTI workflow. The repository-root C++ pipeline is the
maintained production implementation; see [ARCHITECTURE.md](../../ARCHITECTURE.md)
for its component design and migration status.

## Running the pipeline

Run this command from the repository root after building the helper binaries.
The launcher performs initialization, feature processing, triangulation, two
rounds of bundle adjustment, track extension, and validation.

```bash
scripts/python_version/run_multiround_pipeline.sh \
  --data data \
  --first-output output/lio_camera_pose_first \
  --output output/lio_camera_pose_final \
  --max-images -1
```

Use `--max-images 40` for a small test. The required binaries are
`build/lio_camera_pose` and `build/lio_bundle_adjust`.

### Optional outputs

- `diagnose_landmarks.py` and `export_filtered_3dgs.py`: inspect and filter the
  sparse model.
- `prepare_depth_supervision.py`, `generate_global_lidar_depth.py`, and
  `add_floor_plane_depth.py`: create depth supervision.
- `fuse_dense_cloud.py` and `dense_mvs_triangulate.py`: create dense point
  clouds.

These tools operate on the final output and are not required for the standard
3DGS training data.

## Running on KITTI

First prepare the KITTI sequence with
`scripts/kitti/prepare_kitti_pipeline.py`. The Python workflow needs the
following data layout:

| Python-version expects | `scripts/kitti/prepare_kitti_pipeline.py` writes |
|---|---|
| `timestamps.txt` with `name ts` rows and no header | a `# image_filename ...` header line |
| `tuned_camera_lidar_extrinsic.json` | `camera_lidar_extrinsic.json` (same `T_camera_lidar` key) |
| `undistorted/<name>.png` | images stay in the KITTI `image_03/data` folder (already rectified) |
| Camera/LiDAR timestamps | Direct timestamp alignment; no time offset is applied |

The preparation script writes a KITTI package; the commands below convert it to
the Python workflow layout and run the same multiround pipeline used above.

```bash
# 1. Prepare KITTI inputs.
python3 scripts/kitti/prepare_kitti_pipeline.py --prepare-only
P=output/kitti_prepared/image_03_0000000070_0000000500
SEQ=$(python3 -c "import json;print(json.load(open('kitti_prepare.json'))['sequence_directory'])")

# 2. Convert to the Python-version layout.
D=data_kitti_70_500
O=output/kitti_python_version_70_500
mkdir -p "$D"
grep -v '^#' "$P/timestamps.txt" > "$D/timestamps.txt"
cp "$P/key_frames.jsonl" "$P/intrinsics.txt" "$D/"
cp "$P/camera_lidar_extrinsic.json" "$D/tuned_camera_lidar_extrinsic.json"
ln -sfn "$SEQ/image_03/data" "$D/undistorted"

# 3. Build the required helper binaries.
mkdir -p build
g++ -O2 -std=c++17 scripts/python_version/lio_camera_pose.cpp \
  -o build/lio_camera_pose -I/usr/include/eigen3
g++ -O2 -std=c++17 scripts/python_version/bundle_adjust.cpp \
  -o build/lio_bundle_adjust -I/usr/include/eigen3 -lceres -lglog -lgflags -pthread

# 4. Run initialization, two-round optimization, and validation.
scripts/python_version/run_multiround_pipeline.sh \
  --data "$D" \
  --first-output "${O}_first" \
  --output "${O}_final" \
  --max-images -1
```



