# C++ Visual Reconstruction Pipeline

## 1. Goal and migration boundary

The maintained C++17 pipeline lives in the repository-root `src/` and
`include/`, built by the root `CMakeLists.txt`. It replaces the sparse
reconstruction runtime previously developed in the temporary migration project.

The project owns the feature-matching and bundle-adjustment pipeline:

1. synchronize camera frames with the LIO trajectory;
2. initialize calibrated camera poses;
3. extract and cache image features;
4. match temporal and non-local image pairs;
5. construct conflict-free multi-view feature tracks;
6. triangulate and geometrically filter sparse landmarks;
7. extend tracks into weakly supported frames;
8. run local, incremental, and global bundle adjustment;
9. export a sparse COLMAP-text-compatible reconstruction; and
10. validate outputs and write machine-readable metrics.

Dense reconstruction, LiDAR depth-map generation, depth supervision, and
Gaussian-splat training are explicitly outside this project's scope.

### Root integration and remaining legacy retirement

The temporary `standalone_cpp/` project has been integrated into the repository
root. Its source, headers, tests, tools, YAML configurations, README, and this
architecture document now live at the root. The CMake project is named
`lio_visual_ba`, and builds use `cmake -S . -B build`. The CLI entry point is
`src/main.cpp`. The root build contains only the maintained pipeline and its tests.

The legacy `standalone/` directory remains a behavioral reference until final
acceptance. Its executables are no longer targets in the maintained CMake build.
Retire superseded programs later, relocating any still-needed utilities before
removing that directory. Preserve reference outputs and migration tests needed
to verify compatibility.

## Shared pipeline architecture

The C++ and Python implementations follow the same LIO-based visual SLAM
architecture. Both use LIO poses and camera/LiDAR calibration to initialize a
metric camera trajectory, optimize that trajectory with sparse visual geometry,
and export data suitable for 3D Gaussian Splatting (3DGS) training.

```text
images + timestamps + LIO poses + calibration
                     |
                     v
          camera-pose initialization from LIO
                     |
                     v
         feature detection, matching, and tracks
                     |
                     v
          geometric verification and triangulation
                     |
                     v
          camera-pose and landmark optimization
                     |
                     v
             track extension and cleanup
                     |
                     v
       sparse model, poses, and validation metrics
                     |
                     v
              3DGS training data
```

The architecture is shared; the implementation differs:

- **C++**: maintained production pipeline in `src/` and `include/`, built with
  CMake and executed through `lio_visual_ba_pipeline`.
- **Python**: reference and experimentation workflow in
  `scripts/python_version/`, using `run_multiround_pipeline.sh` for the same
  initialization, geometry, optimization, and validation stages.

The shared output contract includes optimized camera poses, sparse landmarks,
tracks, validation metrics, and a COLMAP-compatible `sparse/0/` model. Dense
depth generation and other optional products remain separate post-processing
tools rather than part of the core sparse-SLAM architecture.

## 2. Current repository architecture

The repository currently has two overlapping implementations.

### Maintained C++ pipeline

- `include/` and `src/` contain the flat component APIs and implementations.
- `src/main.cpp` loads a YAML configuration and runs `Pipeline::RunPipeline`.
- `tests/*_test.cpp` contains the maintained regression tests, explicitly listed
  in CMake. Older CamelCase test files are retained as historical source and are
  not part of the current target.
- Configuration files resolve data and output paths relative to their own location.

### Standalone reconstruction implementation

- `standalone/lio_camera_pose.cpp` initializes camera poses from interpolated
  LIO poses and camera-to-LiDAR calibration.
- `standalone/bundle_adjust.cpp` performs local/global Ceres BA, filtering,
  evaluation, and reconstruction export.
- `standalone/triangulate_sparse.py` performs feature caching, pair matching,
  loop candidate selection, managed-track union, triangulation, weak-frame
  recovery, and sparse artifact export.
- `standalone/extend_tracks.py` adds observations to established landmarks by
  pose-guided descriptor search and re-triangulation.
- `standalone/run_incremental_pipeline.py` and
  `standalone/run_rebuild_from_optimized.py` orchestrate incremental mapping.
- `standalone/diagnose_landmarks.py`, `standalone/test_outputs.py`, and
  `standalone/export_filtered_3dgs.py` provide sparse-reconstruction
  diagnostics, validation, and filtered export.

The dense and depth Python utilities in `standalone/` are outside this migration.
Their behavior remains unchanged; still-needed utilities will be relocated before
the legacy directory is retired, as described in section 1.

The new project will use the standalone pipeline's observable file formats and
behavior as the compatibility reference, while extracting reusable algorithms
from both existing C++ implementations where appropriate.

## 3. Current component organization

The library exposes seven feature components and two Core headers.
See the [source index](README.md#source-structure-and-feature-guides) for links to
each feature's API, implementation, invariants, and tests in the overall README.

```text
include/
  status.hpp, types.hpp
  dataset_io.hpp, geometry.hpp, feature_processor.hpp
  mapper.hpp, optimizer.hpp, reconstruction_exporter.hpp, pipeline.hpp
src/
  main.cpp
  dataset_io.cpp, geometry.cpp, feature_processor.cpp
  mapper.cpp, optimizer.cpp, reconstruction_exporter.cpp
  pipeline.cpp, pipeline_frontend.cpp, pipeline_backend.cpp, pipeline_rematcher.cpp
```

Both directories are flat. Headers are included by filename; the C++ namespace
`lio_visual_ba` remains unchanged. File-level comments describe ownership and
critical invariants; algorithm comments explain stage boundaries and constraints.

Feature documentation is consolidated in the root README. Component classes
provide static operations with explicit options and inputs; they do not hide
mutable global dataset state. Pure geometry algorithms use the `geometry`
namespace. Shared option/result types remain in `lio_visual_ba`.

`DatasetIO` owns YAML dataset settings, file loading, and input validation.
`PipelineConfig` extends `DatasetIOConfig`; Pipeline delegates its dataset
configuration parsing and loading to DatasetIO. Sparse writers are grouped with
export/validation. Pipeline's large stages retain separate implementation files
behind one public class. Tests remain divided by behavior.

All symbols live in `lio_visual_ba` or its `geometry` namespace. Applications
parse CLI options, invoke component APIs, and translate failures into exit codes.

## 4. Runtime data flow

```text
calibration + timestamps + LIO keyframes + images + LiDAR sweeps
                              |
                              v
                    synchronized camera poses
                              |
                              v
                 cached features and pair matches
                              |
                              v
        temporal/loop matching -> conflict-free feature tracks
                              |
                              v
              multi-view triangulation and filtering
                              |
                              v
       incremental local BA -> periodic/final global BA
                              |
                              v
                    sparse reconstruction export
                              |
                              v
                 validation and JSON metrics
```

The canonical in-memory model uses stable integer IDs for cameras, features,
tracks, and landmarks. File writers may compact IDs for COLMAP-compatible
exports, but the mapping must be explicit and deterministic.

## 5. Core data contracts

- `Pose3d`: unit quaternion plus translation, with named `world_from_local`
  semantics at API boundaries.
- `CameraIntrinsics`: `fx`, `fy`, `cx`, `cy`, image width, and image height.
- `CameraFrame`: stable ID, filename, timestamp, initial pose, optimized pose,
  and optimization state.
- `Feature`: pixel coordinate, scale, response, octave, and descriptor row.
- `FeatureObservation`: camera ID and feature ID.
- `FeatureTrack`: stable track ID and at most one observation per camera.
- `Landmark`: stable ID, 3D position, color, observations, and quality metrics.
- `Reconstruction`: calibration, cameras, tracks, landmarks, and provenance.
- `PipelineMetrics`: per-stage counts, rejection reasons, timing, and quality
  quantiles, serialized as JSON.

No API may use an unnamed transform convention. Transform variables and fields
must state their direction, such as `world_from_camera` or
`camera_from_lidar`.

## Dataset I/O contract

`DatasetIO` owns YAML configuration, input paths, camera records, calibration,
and the LIO trajectory. Its public API is declared in
[`include/dataset_io.hpp`](include/dataset_io.hpp) and implemented in
[`src/dataset_io.cpp`](src/dataset_io.cpp).

### Required inputs

- `timestamp_file`: one `image_filename timestamp_seconds` record per line.
  Blank lines and comments beginning with `#` are ignored. Filenames are
  resolved relative to `image_directory`.
- `image_directory`: directory containing the configured camera images.
- `intrinsics_file`: comma- or whitespace-delimited pinhole $3 \times 3$
  calibration matrix. Image width and height are supplied separately in YAML.
- `extrinsic_file`: JSON camera/LiDAR calibration containing `T_camera_lidar`.
  `DatasetIO` validates the rigid transform and returns its inverse as
  `lidar_from_camera`.
- `trajectory_file`: consecutive JSON objects containing `timestamp` and
  `lio_pose`. Objects may be one-line or pretty-printed and are not wrapped in
  a JSON array.

The trajectory loader reads `lio_pose.translation` in meters and
`lio_pose.quaternion_xyzw` as `[qx, qy, qz, qw]`, interprets the pose as
`world_from_lidar`, normalizes valid quaternions, sorts by timestamp, and
rejects duplicate timestamps. At least two valid trajectory poses are required.
Additional metadata such as `key_frame_id`, `saved_frame_path`, and
`optimized_pose` is ignored by this loader.

### Configuration rules

`DatasetIO::LoadConfig` resolves relative paths against the YAML file's
directory. Image dimensions must be positive. Accepted image records receive
compact camera IDs in input order after missing-image filtering; disabling
strict timestamp ordering preserves that order and does not sort the records.
`maximum_images` can cap the accepted records, while `-1` means unlimited.
Camera time offset is applied later by Mapper during pose initialization, not by
the input parser.

The core transform convention is:

```text
point_world = rotation_world_from_local * point_local
            + translation_world_from_local
world_from_camera = world_from_lidar * lidar_from_camera
```

Dataset loading returns data only. Mapper owns LIO interpolation and camera-pose
initialization; FeatureProcessor owns image decoding and visual features.
DatasetIO tests cover calibration, image-list, trajectory, and YAML-to-dataset
loading behavior.

## 6. Component responsibilities

### Core and I/O

Core owns status/error propagation, configuration, IDs, and common data types.
DatasetIO owns dataset YAML settings, camera calibration, image lists, and JSON
LIO trajectories. Exporter owns sparse output formats; FeatureProcessor owns
binary caches. Parsing validates counts, finite values,
quaternion norms, matrix dimensions, ID references, and file consistency.

### Geometry

Geometry owns pose composition/interpolation, projection/unprojection,
fundamental matrices, epipolar/Sampson error, cheirality, parallax, DLT
triangulation, reprojection statistics, and voxel indexing. Algorithms do not
perform file I/O.

### Features and tracks

Features owns OpenCV SIFT/Shi-Tomasi extraction, grid balancing, versioned
binary caches, mutual ratio matching, MAGSAC verification, pair-cache reuse,
disjoint-set track construction, one-observation-per-camera enforcement, weak
frame recovery, and landmark-guided track extension.

### Mapping and optimization

Mapping selects temporal and LIO-spatial loop candidates and controls the
bootstrap/batch/local/global schedule. Optimization owns Ceres parameter
blocks, manifolds, robust losses, pose/landmark/intrinsic priors, observation
cleanup, landmark selection, and held-out evaluation.

### Export and validation

Export writes trajectories, sparse PLY, tracks, observations, metrics, and
strict COLMAP text files for downstream reconstruction consumers. Validation independently
reopens generated artifacts and checks counts, references, finite geometry,
quaternion normalization, reprojection thresholds, and required files.

## 7. Executables

- `lio_visual_ba_pipeline`: complete or incremental pipeline; replaces both Python
  controllers and the shell controller.
- `lio_visual_ba_pose_init`: pose initialization parity/debug tool.
- `lio_visual_ba_triangulate`: feature, matching, track, and sparse triangulation tool.
- `lio_visual_ba_bundle_adjust`: local/global BA and cleanup tool.
- `lio_visual_ba_extend_tracks`: established-landmark completion tool.
- `lio_visual_ba_validate`: output validator suitable for CTest and CI.

The focused tools make each migration stage testable. `lio_visual_ba_pipeline` calls
the same library functions directly; it does not spawn the focused tools.

## 8. Dependency policy

Required dependencies:

- C++17 standard library;
- Eigen3 for linear algebra;
- OpenCV for image I/O, SIFT, optical flow, and robust geometry;
- Ceres Solver for bundle adjustment;
- nlohmann_json for JSON/JSONL and metrics;
- yaml-cpp for configuration; and
- GoogleTest for tests.

Sophus and TBB are optional initially. Pose operations can use Eigen directly,
and concurrency will use a small bounded C++ worker abstraction unless measured
performance justifies TBB. No Python runtime is permitted.

## 9. Compatibility outputs

The C++ pipeline will preserve these important existing artifacts where their
stage applies:

- `poses_lio_prior_tum.txt`, `poses_optimized_tum.txt`;
- `sparse_points.ply`, `coverage_points.ply`, `landmarks_optimized.ply`;
- `tracks.csv`, `observations.csv`, `heldout.csv`;
- `triangulation_metrics.json`, `track_extension_metrics.json`;
- `evaluation.json`, `incremental_history.json`;
- `intrinsics_refined.txt`; and
- `sparse/0/cameras.txt`, `sparse/0/images.txt`, `sparse/0/points3D.txt`.

New binary caches will have a magic value, schema version, endianness marker,
record counts, source/config fingerprint, and checksum. Partial cache writes
use a temporary file followed by an atomic rename.

## 10. Error handling, determinism, and observability

- Library code returns typed results or throws only at executable boundaries;
  one policy will be chosen in the core API and used consistently.
- Invalid input fails with a file path and actionable reason.
- Randomized algorithms accept a configured seed.
- Iteration over ID-indexed data is deterministic before serialization.
- Each stage records wall time, input/output counts, rejection counts, and
  relevant thresholds.
- Long stages report progress without changing machine-readable output.
- Resume verifies configuration and input fingerprints before reusing caches.

## 11. Verification strategy

Every migrated Python stage receives:

1. unit tests for its pure geometry and parsing primitives;
2. a small deterministic integration fixture;
3. schema/count/finite-value validation of its outputs; and
4. a parity comparison against a preserved Python reference output.

Parity uses tolerances instead of byte identity for floating-point products.
The Python implementation remains available as a reference until its C++
replacement passes. Tests cover transform direction, timestamp interpolation,
track conflicts, cheirality, low parallax, reprojection pruning, cache
corruption, interrupted resume, and COLMAP cross-reference validity.

## 12. Historical migration sequence (pre-consolidation)

The following sequence records the original migration plan. Its old filenames
and review gates are historical; the current file organization is in section 3.
Future feature changes should update the relevant component guide and tests.

1. `CMakeLists.txt`: define the library, focused applications, dependencies,
   warnings, sanitizable test targets, and install layout.
2. `include/status.hpp`: define the project-wide error/result model.
3. `include/types.hpp`: define IDs and the canonical data model.
4. `include/lio_visual_ba/geometry/pose.hpp` plus implementation/test: establish and
   verify transform composition, inversion, and interpolation conventions.
5. Core input readers and tests: timestamps, intrinsics, extrinsics, JSONL LIO
   poses, TUM poses, and PLY.
6. `pose_initializer`: replace `lio_camera_pose.cpp` using the shared library.
7. Camera geometry primitives and tests.
8. Feature extraction, grid balancing, and versioned feature cache.
9. Pair matching, geometric verification, and versioned pair cache.
10. Conflict-safe track builder and deterministic DSU.
11. Multi-view triangulation, filtering, weak-frame recovery, and sparse
    outputs; replace `triangulate_sparse.py`.
12. Ceres bundle adjuster, staged cleanup, selection, evaluation, and strict
    sparse export; replace `bundle_adjust.cpp` without its monolithic layout.
13. Incremental mapper and resume state; replace
    `run_incremental_pipeline.py`.
14. Track extender; replace `extend_tracks.py` and `recover_weak_frames.py`.
15. Sparse-landmark diagnostics, filtered export, and validator; replace the
    Python tools that operate on feature tracks and BA results.
16. Unified `lio_visual_ba_pipeline`, configuration, end-to-end parity tests, and
    documentation.
17. After explicit approval, retire the Python feature-matching and BA runtime
    path. Reference outputs
    and migration tests remain archived until final acceptance.

## 13. Historical migration review gate

For each review unit, implementation stops after reporting:

- the behavior added;
- every file created or changed;
- important API and algorithm decisions;
- exact build/test commands and results;
- compatibility differences or remaining risks; and
- the next proposed review unit.

No next unit begins until it is approved or revised from feedback.
