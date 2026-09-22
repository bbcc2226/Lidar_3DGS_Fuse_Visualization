# LIO Visual Bundle Adjustment

`lio_visual_ba` is a C++17, LIO-based visual SLAM pipeline for camera-pose
optimization and sparse 3D reconstruction. It combines calibrated images,
LIO poses, camera intrinsics, and camera/LiDAR calibration to preserve metric
scale and produce a consistent camera trajectory.

The output is a sparse, COLMAP-compatible reconstruction with the camera poses,
3D points, tracks, and metrics needed to prepare training data for 3D Gaussian
Splatting (3DGS).


For setup and execution, see [Build and run the C++ version](#build-and-run-the-c-version).

## Contents

- [Features](#features)
- [Build and run the C++ version](#build-and-run-the-c-version)
- [Outputs](#5-inspect-the-outputs)
- [Prepare and run KITTI](#prepare-and-run-kitti-with-lidar-slam-poses)
- [Compare with the Python pipeline](#compare-with-the-preserved-python-pipeline)
- [Architecture](ARCHITECTURE.md)

## Features

- LIO and camera timestamp synchronization;
- calibrated camera-pose initialization;
- SIFT feature extraction and caching;
- temporal and non-local feature matching;
- geometrically verified multi-view tracks;
- sparse landmark triangulation and filtering;
- weak-frame recovery and track extension;
- incremental, local, and global Ceres bundle adjustment;
- sparse PLY and COLMAP text export; and
- validation and JSON metrics.

## Build and run the C++ version

Run the following commands from the repository root, which contains
`CMakeLists.txt`, `src/`, `include/`, and `data/`.

### 1. Install the build dependencies

You need a C++17 compiler, CMake 3.16 or newer, Eigen3, OpenCV (including SIFT),
Ceres Solver, nlohmann_json, yaml-cpp, and GoogleTest when building tests.

On Ubuntu/Debian, install the development packages:

```bash
sudo apt update
sudo apt install build-essential cmake libeigen3-dev libopencv-dev \
  libceres-dev nlohmann-json3-dev libyaml-cpp-dev libgtest-dev
```

Python is optional for the output-comparison test and comparison tools. The C++
reconstruction executable does not require Python at runtime.

### 2. Configure, build, and test

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The executable is `build/lio_visual_ba_pipeline`. After changing
C++ code, rerun the build command. To build without GoogleTest or unit tests,
configure with `-DBUILD_TESTING=OFF` and omit the CTest command.

### 3. Configure your dataset

Create your own configuration beside the example:

```bash
cp -n pipeline.example.yaml pipeline.local.yaml
```

Edit `pipeline.local.yaml`. Set these values for your dataset:

```yaml
inputs:
  timestamp_file: data/timestamps.txt
  image_directory: data/undistorted
  intrinsics_file: data/intrinsics.txt
  extrinsic_file: data/tuned_camera_lidar_extrinsic.json
  trajectory_file: data/key_frames.jsonl
output_directory: output/lio_visual_ba_cpp_local
image_width: 1596
image_height: 1197
camera_time_offset_seconds: -0.4
maximum_cpu_threads: 4
maximum_images: 10
skip_missing_images: true
require_strictly_increasing_timestamps: true
```

These dimensions and time offset are examples; use your image dimensions and
calibrated camera/LiDAR time offset. Keep the feature, matching, mapping, and
optimization sections copied from the example, adjusting them as needed.

All relative input and output paths are resolved against the **YAML file's
location**, not the shell's working directory. Start with `maximum_images: 10`
for a short run, then set `maximum_images: -1` to process all accepted images.
See the [Dataset I/O contract](ARCHITECTURE.md#dataset-io-contract) for input
formats and loading policies.

### 4. Run the C++ pipeline

```bash
./build/lio_visual_ba_pipeline \
  --config pipeline.local.yaml
```

Alternatively, the supplied ten-image smoke configuration uses the repository's
example dataset paths:

```bash
./build/lio_visual_ba_pipeline \
  --config pipeline.smoke.yaml
```

The configured output directory must be absent or empty. For another run, choose
a new `output_directory` in YAML. Publication is staged and validated before the
completed directory becomes visible.

### 5. Inspect the outputs

On success, the executable prints camera, landmark, and verified-match counts,
plus the output directory. Exit code `0` means success, `1` indicates a
configuration/processing failure, and `2` indicates incorrect command-line usage.

Inside your configured output directory:

| Output | Contents |
|---|---|
| `poses_lio_prior_tum.txt` | Initial LIO-derived camera poses |
| `poses_optimized_tum.txt` | Optimized camera poses |
| `sparse_points.ply`, `landmarks_optimized.ply` | Sparse 3D landmarks |
| `tracks.csv`, `track_summary.csv`, `observations.csv` | Track and observation records |
| `heldout.csv` | Matches excluded from reconstruction fitting |
| `intrinsics_refined.txt` | Exported camera intrinsics; current BA keeps intrinsics fixed |
| `evaluation.json` | Reconstruction counts, recovery statistics, and reprojection metrics |
| `sparse/0/` | Validated COLMAP `cameras.txt`, `images.txt`, and `points3D.txt` |

If configuration fails, check the reported missing dependency or YAML/input path.
If compilation runs out of memory, reduce parallelism to `-j1`. If publication
reports a nonempty output directory, select a fresh output path in the YAML file.

## Implementation details

The README intentionally stays focused on setup and execution. Detailed input
formats, transform conventions, component responsibilities, data contracts, and
pipeline stage interactions are documented in
[`ARCHITECTURE.md`](ARCHITECTURE.md):

- [Shared pipeline architecture](ARCHITECTURE.md#shared-pipeline-architecture)
- [Repository architecture](ARCHITECTURE.md#2-current-repository-architecture)
- [Core data contracts](ARCHITECTURE.md#5-core-data-contracts)
- [Component responsibilities](ARCHITECTURE.md#6-component-responsibilities)
- [`DatasetIO` API](include/DatasetIo.hpp)

## Prepare and run KITTI with LiDAR SLAM poses

Edit kitti_prepare.json to select the sequence, camera, inclusive image-ID range,
LiDAR SLAM folder, timestamp policy, and output directories. The example points at
the supplied drive_0027_sync image_03 sequence and kitti_point_cloud folder.

Run preparation and C++ reconstruction from the repository root:

```bash
python3 scripts/kitti/prepare_kitti_pipeline.py --config kitti_prepare.json
```

Image IDs are numeric PNG filenames and both configured range endpoints
are included. For a one-off range, pass --start-image-id and --end-image-id.
Use --prepare-only to generate inputs and inspect them before running C++.

The script matches the image timestamp file to numeric PNG names, parses
key_frames.jsonl, and uses each optimized_pose as the C++ trajectory's lio_pose.
It preserves and validates each saved LiDAR cloud path. For image_03, it converts
KITTI's P_rect_03, R_rect_00, and Velodyne-to-camera calibration into a pinhole
intrinsics file and the T_camera_lidar JSON expected by the C++ loader. It writes
image_lidar_association.csv with the nearest SLAM frame/cloud and signed time
difference. The C++ pipeline itself interpolates the selected trajectory at each
camera timestamp plus camera_time_offset_seconds.

By default, image IDs whose query times are outside SLAM pose coverage are clipped
and counted in preparation_summary.json. Set outside_pose_range to error to reject
a partially covered range. The LiDAR log has nine duplicate-time records in this drive. They are repeated
revisions of the same key frame and cloud; the default duplicate_pose_policy,
last_revision, keeps the last optimized pose for those duplicates and reports
the count. Duplicate timestamps referring to different key frames are rejected.
Change the policy to error to reject any duplicates.

The script also writes a generated pipeline.yaml, then
runs the root build/lio_visual_ba_pipeline unless --prepare-only was supplied.
Prepared files are stored under output/kitti_prepared; reconstruction products
are written to pipeline_output_directory.

This adapter validates and retains LiDAR point-cloud references, but the current
C++ visual reconstruction does not consume or fuse point-cloud geometry; it uses
the optimized LiDAR SLAM poses and camera images.

## Compare with the preserved Python pipeline

Use preserved Python reference outputs in `output/lio_visual_ba_python` (or adjust
`--python-output` below). From the repository root:

```bash
./build/lio_visual_ba_pipeline \
  --config pipeline.example.yaml

python3 tools/compare_sparse_outputs.py \
  --python-output output/lio_visual_ba_python \
  --cpp-output output/lio_visual_ba_cpp \
  --report output/sparse_parity_report.json
```

The comparison aligns TUM poses by timestamp and reports translation and
rotation RMSE/max differences, landmark and observation count differences,
reprojection metric differences, every applied tolerance, and an overall
pass/fail decision. Exit code 0 means all configured tolerances passed; exit
code 2 means valid outputs were compared but at least one tolerance failed.
Thresholds can be changed with `--translation-rmse-m`,
`--rotation-rmse-deg`, `--landmark-relative`, `--observation-relative`, and
`--reprojection-p90-delta-px`.

The default pose tolerances are 0.10 m translation RMSE and 1.5 degrees
rotation RMSE. They measure agreement between two independently ordered robust
optimization paths; they are not claims of absolute physical accuracy.

The comparison is intentionally numeric rather than byte-for-byte: OpenCV
feature ordering, robust estimation, and Ceres solve paths can produce slightly
different but equivalent sparse reconstructions.

The test suite covers geometry, parsing, configuration-driven loading, caches,
matching, tracks, mapping, optimization, stage orchestration, and output validation.
Run CTest for the current test inventory and results; unit tests do not establish
full-dataset parity by themselves.

## Review workflow

For every component, the review report includes:

- behavior added;
- files created or changed;
- important design choices;
- build and test results;
- compatibility risks; and
- the next proposed component.

`README.md` is updated whenever the implemented structure or pipeline flow
changes. The detailed target design and migration order are recorded in
`ARCHITECTURE.md`.
