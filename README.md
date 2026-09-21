# LIO Visual Bundle Adjustment

`lio_visual_ba` is a C++17 pipeline for LIO-guided visual feature
matching, sparse landmark reconstruction, and bundle adjustment.

The code is organized by feature component. The existing code under `standalone/`
remains the behavioral reference for numeric parity comparisons.

For setup and execution, see [Build and run the C++ version](#build-and-run-the-c-version).

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

## Pipeline flow

```text
timestamps.txt       key_frames.jsonl       camera-LiDAR calibration
(camera timestamps)  (LiDAR LIO poses)
      |                       |                         |
      +-----------------------+-------------------------+
                              |
                              v
              synchronize timestamps and interpolate LIO poses
                              |
                              v
                    initialize world_from_camera poses
                              |
                              v
                   load images and extract SIFT features
                              |
                              v
               select and match candidate image pairs
                              |
                   +----------+----------+
                   |                     |
                   v                     v
         temporal pair matching   LIO-spatial loop matching
                   |                     |
                   +----------+----------+
                              |
                              v
             epipolar verification and conflict-safe track union
                              |
                              v
             multi-view triangulation and geometric filtering
                              |
                              v
             incremental local BA and periodic global BA
                              |
                              v
               weak-frame recovery and track extension
                              |
                              v
                 final global BA and outlier cleanup
                              |
                              v
       poses + sparse landmarks + tracks + COLMAP text + metrics
```

### Camera image timestamps: `timestamps.txt`

The file selected by `inputs.timestamp_file` has one image per line, with two
whitespace-separated fields: `image_filename timestamp_seconds`.

```text
# image_filename timestamp_seconds
IMG_2129.png 1785715868.878000
IMG_2130.png 1785715870.894000
IMG_2131.png 1785715875.047000
```

Image filenames are resolved relative to `inputs.image_directory`. Timestamps are
in **seconds** and must increase strictly by default. Blank lines and comments
starting with `#` are accepted. The filename itself is configurable: a file named
`timestampx.txt` works when `inputs.timestamp_file` points to it.

### LiDAR LIO poses: `key_frames.jsonl`

The file selected by `inputs.trajectory_file` contains consecutive JSON objects.
Here is a minimal example with two LIO poses (illustrative values):

```jsonl
{"timestamp":1785715868.0,"lio_pose":{"translation":[0.0,0.0,0.0],"quaternion_xyzw":[0.0,0.0,0.0,1.0]}}
{"timestamp":1785715875.0,"lio_pose":{"translation":[0.7,0.0,0.0],"quaternion_xyzw":[0.0,0.0,0.0,1.0]}}
```

| Field | Meaning |
|---|---|
| `timestamp` | LIO pose timestamp in seconds, on the same time basis as the camera timestamps |
| `lio_pose.translation` | LiDAR origin in world coordinates: `[x, y, z]`, in meters |
| `lio_pose.quaternion_xyzw` | LiDAR-to-world rotation: `[qx, qy, qz, qw]`; identity is `[0, 0, 0, 1]` |

The loader reads `lio_pose` as `world_from_lidar`. It also accepts pretty-printed
objects spanning multiple lines, as in the repository dataset; objects are not
wrapped in a JSON array or separated by commas. Additional fields such as
`key_frame_id`, `saved_frame_path`, and `optimized_pose` are ignored by this loader.
At least two valid poses with distinct timestamps are required. The loader sorts
poses by timestamp and normalizes valid nonzero quaternions.

### How the two timestamp streams connect

For each image, Mapper queries the trajectory at:

```text
query_time = image_timestamp + camera_time_offset_seconds
world_from_camera = interpolated_world_from_lidar * lidar_from_camera
```

Translation is interpolated linearly and rotation uses quaternion SLERP. Queries
outside the trajectory range use the nearest endpoint pose. For example, an image
at `1785715868.878` with offset `-0.4` queries LIO time `1785715868.478`.
`lidar_from_camera` comes from inverting the calibration file's `T_camera_lidar`.

## Source structure and feature guides

This README documents both `include/` (public APIs) and `src/` (implementations).
Both directories are flat: headers are included by filename (for example,
`#include "pipeline.hpp"`), and C++ namespaces still organize the symbols.

```text
project/
  include/
    dataset_io.hpp
    feature_processor.hpp
    geometry.hpp
    mapper.hpp
    optimizer.hpp
    pipeline.hpp
    reconstruction_exporter.hpp
    status.hpp
    types.hpp
  src/
    main.cpp
    dataset_io.cpp
    feature_processor.cpp
    geometry.cpp
    mapper.cpp
    optimizer.cpp
    pipeline.cpp
    pipeline_frontend.cpp
    pipeline_backend.cpp
    pipeline_rematcher.cpp
    reconstruction_exporter.cpp
  tests/
    *_test.cpp
  tools/
  CMakeLists.txt
  pipeline.example.yaml
  README.md
```

Each feature has one public header and a corresponding implementation. Public
operations are static methods on a component class; Geometry uses a namespace for
pure calculations. Options and result structs live beside the owning API. Core
keeps the shared data/error contracts. Focused test files live directly in `tests/`, with no `unit/` subfolder.

| Component | Public header | How it works, critical parts, interactions |
|---|---|---|
| DatasetIO | `include/dataset_io.hpp` | [Dataset I/O](#dataset-io) |
| Geometry | `include/geometry.hpp` | [Geometry](#geometry) |
| FeatureProcessor | `include/feature_processor.hpp` | [Features and tracks](#feature-processing) |
| Mapper | `include/mapper.hpp` | [Mapping](#mapping) |
| Optimizer | `include/optimizer.hpp` | [Optimization](#optimization) |
| ReconstructionExporter | `include/reconstruction_exporter.hpp` | [Export and validation](#reconstruction-export-and-validation) |
| Pipeline | `include/pipeline.hpp` | [Pipeline stages](#pipeline-orchestration) |
| Shared contracts | `include/{types,status}.hpp` | [Core](#shared-data-and-error-contracts) |

Implementation files use the same component names directly under `src/`.
Pipeline retains separate frontend/backend/rematcher implementation files behind
its single public header because these stages are substantial.

### Configuration-driven dataset loading

Change input paths, image dimensions, image count, missing-image behavior, and
image timestamp policy in YAML; no dataset-specific C++ edits are required.
`DatasetIO::LoadConfig` accepts the dataset portion of the same configuration used
by `Pipeline::LoadPipelineConfig`. `DatasetIO::LoadDataset` consumes that configuration.
See the [I/O guide](#dataset-io) for a dataset-only example and key semantics.

### Updating callers after consolidation

Old feature-specific include paths are replaced by the component header. Function
names and option/result types are retained, with class/namespace qualification:

```cpp
#include "feature_processor.hpp"
#include "geometry.hpp"

auto features = lio_visual_ba::FeatureProcessor::ExtractFeatures(camera_id, image, options);
auto inverse = lio_visual_ba::geometry::InversePose(world_from_camera);
```

Reconstruction writers formerly under `io/` now belong to
`ReconstructionExporter`. Applications and tests use the new API. External C++
callers must update includes and qualifications; YAML keys and artifact formats
remain compatible, with two additional optional dataset policy keys. The input-path
struct is now named `DatasetInputPaths` and belongs to DatasetIO.

## Implemented flow

The executable C++ sparse pipeline now covers the full configured flow:

```text
timestamps.txt + image directory
  -> validated, strictly ordered timestamp records
  -> missing-image policy and maximum-image limit
  -> compact stable CameraId assignment
  -> CameraFrame records
  -> validated 3x3 pinhole intrinsics
  -> validated T_camera_lidar JSON, inverted to lidar_from_camera
  -> Pose3d
  -> validation/normalization
  -> inversion and composition
  -> point transformation
  -> timestamp interpolation using translation lerp + quaternion SLERP
  -> SIFT and grid-balanced feature extraction
  -> temporal and LIO-spatial candidate matching
  -> deterministic held-out split and conflict-safe tracks
  -> robust multi-view triangulation and observation pruning
  -> incremental local and periodic/final global Ceres BA
  -> repeated final solve/prune cleanup
  -> atomic sparse outputs and independent COLMAP validation
```

## Transform convention

All poses use explicitly named `outer_from_inner` semantics. `Pose3d` stores a
rotation and translation that transform a local point into the world frame:

```text
point_world = rotation_world_from_local * point_local
            + translation_world_from_local
```

Composition is ordered as:

```text
world_from_camera = world_from_lidar * lidar_from_camera
```

This convention must be preserved at every file and API boundary.

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
See the [DatasetIO guide](#dataset-io) for input formats and loading policies.

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

## Dataset I/O

Public API: [`dataset_io.hpp`](include/dataset_io.hpp).
Implementation: [`dataset_io.cpp`](src/dataset_io.cpp).

### Key features and how they work

- **Configuration loading — `DatasetIO::LoadConfig`:** reads the `inputs` map,
  image dimensions, image limit, missing-image policy, and timestamp policy from
  YAML. Relative paths are resolved against the configuration file's directory.
  Dataset-only callers do not need an output directory or optimization settings.
- **Dataset loading — `DatasetIO::LoadDataset`:** loads camera records, pinhole
  intrinsics, camera/LiDAR extrinsics, and the LIO trajectory using that configuration.
  The returned `DatasetInputs` contains data; pose initialization belongs to Mapper.
- **Image records — `LoadCameraFrames`:** reads `filename timestamp` rows, applies
  the missing-file policy and image cap, and assigns compact camera IDs in accepted
  record order. It checks file existence; FeatureProcessor decodes image pixels later.
- **Calibration:** reads a comma/space-delimited 3×3 matrix and validates positive
  focal lengths and the supported pinhole form. The JSON `T_camera_lidar` must be a
  proper rigid transform; the loader returns its inverse, `lidar_from_camera`.
- **Trajectory:** reads consecutive JSON objects, including pretty-printed objects,
  validates translation/quaternion values, normalizes quaternions, sorts timestamps,
  and rejects duplicate timestamps. At least two poses are required.

### Configuration, not embedded dataset values

The same keys work in a dataset-only file or a full pipeline file:

```yaml
inputs:
  timestamp_file: data/timestamps.txt
  image_directory: data/images
  intrinsics_file: data/intrinsics.txt
  extrinsic_file: data/extrinsic.json
  trajectory_file: data/key_frames.jsonl
image_width: 800
image_height: 600
maximum_images: -1              # unlimited; use a positive value to cap images
skip_missing_images: true
require_strictly_increasing_timestamps: true
```

The numbers above are examples. Width and height must be supplied as positive
values. Focal lengths, principal point, extrinsics, and trajectory values come from
input files. Omitted policy keys use the defaults in `ImageListOptions`. Format
invariants such as 3×3/4×4 matrix sizes and numerical validation tolerances are
implementation rules, not dataset tuning parameters.

```cpp
auto config = lio_visual_ba::DatasetIO::LoadConfig("dataset.yaml");
if (!config.ok()) return config.status();
auto dataset = lio_visual_ba::DatasetIO::LoadDataset(config.value());
if (!dataset.ok()) return dataset.status();
// dataset.value() contains cameras, intrinsics, extrinsic, and trajectory.
```

`PipelineConfig` extends `DatasetIOConfig`; `Pipeline::LoadPipelineConfig` delegates
these settings to `DatasetIO::LoadConfig`. The frontend passes the configuration to
`LoadDataset`. For programmatic configurations, `inputs.image_directory` is the
authoritative directory used by `LoadDataset`.

### Critical parts

Do not invert the calibration twice. Keep timestamps in seconds, and apply camera
time offset during Mapper pose initialization, not during parsing. Camera IDs
refer to accepted rows, so filtering images changes downstream IDs. Disabling strict
image timestamp ordering preserves file order; it does not sort camera frames.
Check `Result::ok()` before accessing loaded data. Errors identify their parsing
or loading stage and, where available, the offending path or record.

### Interactions and verification

Pipeline supplies YAML configuration → DatasetIO loads data → Mapper initializes
poses → FeatureProcessor opens the configured image paths. Geometry provides
transform validation and inversion. Reconstruction writers live in Exporter.

Tests: `calibration_io_test.cpp`, `image_list_io_test.cpp`, `trajectory_io_test.cpp`,
and the YAML-to-dataset checks in `pipeline_config_test.cpp` under `tests/`.

## Geometry

Public API: [`geometry.hpp`](include/geometry.hpp).
Implementation: [`geometry.cpp`](src/geometry.cpp).

### Key features and how they work

- **Pose operations:** validate/normalize quaternions, invert and compose poses,
  transform points, and interpolate translation linearly with quaternion SLERP.
- **Camera model:** project camera/world points, unproject pixels at a supplied
  depth, calculate camera centers, and construct `K [R | t]` projection matrices.
- **Epipolar geometry:** derive relative pose, essential and fundamental matrices,
  then calculate Sampson error for candidate pixel correspondences.
- **Triangulation:** solve multi-view DLT using SVD, then evaluate depth, parallax,
  and reprojection errors against `TriangulationOptions`.

These pure functions are grouped under `lio_visual_ba::geometry`; their option
and result structs remain in `lio_visual_ba`. No object state or file I/O is needed.

### Critical parts

`world_from_camera` maps a camera point into world coordinates. Projection must
use the inverse transform. Composition is `world_from_lidar * lidar_from_camera`.
Sampson error is the square root of squared Sampson distance, expressed in pixels.
Do not compare it against a squared-pixel threshold. Projection rejects points at
or behind the minimum depth. Low parallax can make a numerically solvable point
geometrically unreliable; triangulation checks both geometry and reprojection.

The interpolation primitive requires a query inside its sample interval; Mapper
implements endpoint clamping for trajectory queries. Keep these policies distinct.

### Interactions and verification

DatasetIO uses pose validation; Mapper uses interpolation and triangulation;
FeatureProcessor uses fundamental matrices and Sampson filtering; Optimizer and
Exporter use projection and pose conversion. Geometry never invokes those stages.

Tests: `pose_test.cpp`, `camera_model_test.cpp`, `epipolar_test.cpp`, and
`triangulation_test.cpp` under `tests/`.

## Feature processing

Public API: [`feature_processor.hpp`](include/feature_processor.hpp).
Implementation: [`feature_processor.cpp`](src/feature_processor.cpp).

### Key features and how they work

- **Extraction — `FeatureProcessor::ExtractFeatures`:** decodes an image or accepts
  a `cv::Mat`, extracts SIFT features, and supplements weak grid cells with corner
  candidates and SIFT descriptors. Core features precede supplemental features.
- **Descriptor matching:** applies nearest-neighbor ratio filtering, optionally
  requires mutual matches, and can filter with the pose-derived fundamental matrix.
- **Visual verification:** estimates a fundamental matrix robustly using OpenCV,
  then checks inlier count, ratio, and spatial grid coverage in both images.
- **Feature/pair caches:** serialize features or matches with identifying metadata
  and integrity checks, and reject incompatible or damaged cached data.
- **Tracks — `BuildFeatureTracks`:** unions matched observations with a disjoint-set
  structure. A merge is rejected if it would put two observations from the same
  camera into a track. Retained tracks receive deterministic IDs.

### Critical parts

Feature IDs must match descriptor-row indices. Preserve `core_feature_count` and
core/supplemental ordering because downstream triangulation uses different support
rules. Ratio thresholds and geometric thresholds solve different problems;
passing descriptor matching does not establish a valid geometric correspondence.

Track conflict resolution is order-dependent: preserve pair/matcher order so early
matches keep their established priority. The robust estimator protects its seeded
OpenCV RNG operation with a mutex. Cache fingerprints must change with relevant
inputs/options; pair keys also carry camera IDs, feature fingerprints, and counts.

### Configuration and interactions

`FeatureExtractionOptions`, `FeatureMatchingOptions`, `VisualGeometryOptions`, and
`TrackBuilderOptions` are declared beside the class. Pipeline populates supported
settings from YAML and passes these options explicitly. Mapper supplies candidate
pairs and poses; Geometry filters matches; tracks feed Mapper triangulation.

Cache APIs are available to callers, but the current Pipeline frontend extracts
features and matches directly; it does not call the disk-cache APIs automatically.

Tests: `feature_extractor_test.cpp`, `feature_matcher_test.cpp`,
`feature_cache_test.cpp`, `pair_cache_test.cpp`, and `track_builder_test.cpp`.

## Mapping

Public API: [`mapper.hpp`](include/mapper.hpp).
Implementation: [`mapper.cpp`](src/mapper.cpp).

### Key features and how they work

- **Pose initialization — `Mapper::InitializeCameraPoses`:** queries the LIO
  trajectory at image timestamp plus configured camera offset, interpolates or
  clamps to an endpoint, and composes with `lidar_from_camera`.
- **Candidate selection — `SelectCandidatePairs`:** builds temporal-neighbor pairs
  and non-local loop candidates constrained by separation, camera-center distance,
  viewing angle, and per-frame limits.
- **Landmark construction — `BuildSparseLandmarks`:** collects track observations
  with their camera poses, triangulates through Geometry, optionally prunes
  inconsistent observations, and rejects insufficient or geometrically weak tracks.

### Critical parts

Use seconds for time offsets and named transform directions. Pair-selection order
feeds the order-dependent track builder. Preserve original LIO poses when using
optimized poses for geometric decisions; Optimizer still needs immutable priors.

Supplemental tracks have their own minimum support. Exactly-three-view core tracks
have an additional parallax threshold. Observation pruning and triangulation gates
can remove support, so re-check validity after pruning. Respect the configured
landmark limit and coverage policy; the backend also selects a coverage-balanced
set before optimization.

### Configuration and interactions

Pipeline passes `PairSelectionOptions`, `LandmarkBuilderOptions`, and
`camera_time_offset_seconds`. DatasetIO supplies camera records, the trajectory,
and extrinsic. Geometry provides interpolation/projection/triangulation.
FeatureProcessor turns candidate pairs into tracks; Mapper turns those tracks into
landmarks; Optimizer refines their positions and camera poses.

Tests: `pose_initializer_test.cpp`, `pair_selector_test.cpp`, and
`landmark_builder_test.cpp` under `tests/`.

## Optimization

Public API: [`optimizer.hpp`](include/optimizer.hpp).
Implementation: [`optimizer.cpp`](src/optimizer.cpp).

### Key features and how they work

- **Landmark-only BA — `Optimizer::OptimizeLandmarks`:** optimizes 3D positions with
  fixed cameras and intrinsics using robust Ceres reprojection residuals.
- **Joint BA — `OptimizeCamerasAndLandmarks`:** optimizes camera poses and landmarks
  with robust reprojection terms and initial LIO pose priors. Configured anchor or
  constant cameras remain fixed.
- **Cleanup — `RefineWithOutlierCleanup`:** alternates joint solves with removal of
  observations above the reprojection threshold and under-supported landmarks.
- **Window query — `SelectLocalBundleAdjustmentWindow`:** independently grows a
  deterministic co-visibility window, breaking ties by camera ID.
- **Local BA — `OptimizeLocalBundleAdjustment`:** solves a temporal window ending
  at the seed, retaining observing cameras outside that window as fixed boundary
  constraints. It merges the successful result into the reconstruction.
- **Incremental schedule — `RunIncrementalOptimization`:** builds camera prefixes
  in vector order, runs local solves, runs global solves at the configured interval,
  and optionally finishes with a global solve.

### Critical parts

The co-visibility query and the temporal local solver are separate APIs: the local
solver currently does not use the co-visibility selector. A seed camera is variable
unless explicitly anchored. Pose priors always refer to `initial_world_from_camera`,
not the output of the previous solve; otherwise repeated windows can accumulate
prior drift. Intrinsics remain fixed in these solvers.

Prefix solves remove future observations, preventing future-frame leakage. Local
and incremental workflows work on candidate copies and merge after success.
Under-supported local windows can be skipped by the incremental scheduler; other
failures propagate. Respect minimum observation counts after cleanup. Pose
quaternion storage/manifold conventions must agree with Ceres and Eigen.

### Configuration and interactions

The option/report structs and all public methods live in `optimizer.hpp`.
Pipeline translates the YAML `optimization` section into local/global/cleanup
options, including iteration limits, priors, robust-loss thresholds, and scheduling.
Mapper supplies the initial reconstruction; Geometry evaluates projection quality;
Pipeline can rerun refinement after recovery or track extension; Exporter consumes
the final optimized state.

Tests: `bundle_adjuster_test.cpp`, `local_window_selector_test.cpp`,
`local_bundle_adjuster_test.cpp`, and `incremental_optimizer_test.cpp`.

## Reconstruction export and validation

Public API: [`reconstruction_exporter.hpp`](include/reconstruction_exporter.hpp).
Implementation: [`reconstruction_exporter.cpp`](src/reconstruction_exporter.cpp).

### Key features and how they work

- **Sparse artifact writers:** `ReconstructionExporter` writes TUM camera poses,
  landmark PLY, track/observation CSV, and JSON metrics using the existing formats.
- **COLMAP writer — `WriteColmapTextReconstruction`:** builds camera/image/point
  tables and observation cross-references, converts poses into COLMAP conventions,
  and writes `cameras.txt`, `images.txt`, and `points3D.txt`.
- **Independent validator — `ValidateColmapTextReconstruction`:** reopens those
  three files, parses their records independently, and checks IDs, finite values,
  quaternion normalization, observation references, and reciprocal point tracks.

### Critical parts

COLMAP entity IDs are one-based; image observation indices (`POINT2D_IDX`) are
zero-based. Exported world-to-camera poses differ from the internal
`world_from_camera` representation. Do not change ID compaction without updating
both image observations and point tracks. The image-support threshold can exclude
cameras from COLMAP output while the in-memory reconstruction retains them.

Individual writers use temporary-file publication. Complete-directory staging,
cleanup on failure, and final publication belong to `Pipeline::PublishPipelineOutputs`.
The validator should remain independent of writer internals so it can catch
incorrect cross-references produced by a writer change.

### Configuration and interactions

Writer paths and data are explicit arguments. `ColmapTextOptions` controls pose
selection and minimum camera support; the current pipeline sets minimum support to
one. Pipeline supplies the YAML output directory, assembles evaluation metrics,
writes compatibility artifacts, validates the staged COLMAP model, and publishes.
Geometry provides transform conversion/validation; Core supplies reconstruction data.

Tests: `reconstruction_io_test.cpp`, `colmap_text_writer_test.cpp`,
`reconstruction_validator_test.cpp`, and publication cases in `pipeline_test.cpp`.

## Pipeline orchestration

Public API: [`pipeline.hpp`](include/pipeline.hpp).
Implementation: [`pipeline.cpp`](src/pipeline.cpp).

### Files

| File | Responsibility |
|---|---|
| `pipeline.cpp` | YAML pipeline settings, full run, staged output publication and metrics |
| `pipeline_frontend.cpp` | Dataset loading, initial poses, features, matching and weak-frame pair recovery |
| `pipeline_backend.cpp` | Tracks, landmarks, coverage selection, optimization, PnP recovery and track extension |
| `pipeline_rematcher.cpp` | Rebuild matches using optimized seed poses and existing features |

These files implement one `Pipeline` class. Keeping the large stages separate
avoids a single roughly 2,000-line implementation; callers include only `pipeline.hpp`.

### Key features and how they work

1. **Configuration:** `LoadPipelineConfig` delegates dataset settings to
   `DatasetIO::LoadConfig`, reads the remaining YAML settings, resolves the output
   directory relative to YAML, and validates pipeline options.
2. **Frontend:** loads configured inputs, initializes poses, extracts features,
   selects temporal/loop pairs, and performs descriptor/pose filtering with optional
   independent visual geometry verification. A deterministic subset of eligible
   temporal core matches is held out. Weak track support or spatial coverage can
   trigger additional recovery pairs before triangulation.
3. **Backend:** constructs conflict-safe tracks, triangulates landmarks, selects
   coverage, runs incremental BA and cleanup, optionally performs PnP recovery,
   then optionally extends tracks by projected landmark/descriptor search. Added
   observations trigger further refinement. Export attributes are refreshed.
4. **Optional rematching:** `RunPipeline` can run a preliminary backend with
   extension/PnP disabled, rematch existing features using its optimized poses, and
   run the backend again. Initial LIO poses remain unchanged as BA priors.
5. **Publication:** writes all outputs to a sibling staging directory, independently
   validates COLMAP data, writes metrics, and renames the directory into place.

### Critical parts

Preserve held-out exclusion when rebuilding matches; these matches must not feed
track construction. Geometric pair rejection is counted and skipped, while input
or processing failures propagate. Track extension must enforce unique assignments
and revalidate geometry; a nearby descriptor alone is not enough.

Recovery occurs at multiple points: frontend recovery adds pairs, backend PnP
recovery refines weak cameras, and track extension adds landmark observations.
Each has distinct counters and YAML switches. Do not conflate their thresholds.
Output directories must be absent or empty. Never publish a partially written or
unvalidated reconstruction. Thread/seed settings affect OpenCV/Ceres behavior.

### Configuration and interactions

See [`pipeline.example.yaml`](pipeline.example.yaml) for the supported
settings and [`DatasetIO`](#dataset-io) for dataset-only configuration.
The executable is deliberately thin:

```cpp
auto config = Pipeline::LoadPipelineConfig(configuration_file);
if (!config.ok()) return config.status();
auto result = Pipeline::RunPipeline(config.value());
```

The sequence is DatasetIO → Mapper pose/pair selection → FeatureProcessor → Mapper
landmarks → Optimizer → ReconstructionExporter. Geometry supports the stages, and
Core defines the shared data/error contracts. Backend recovery helpers stay private
to the pipeline because they coordinate several components.

Tests: `pipeline_config_test.cpp`, `pipeline_frontend_test.cpp`,
`pipeline_backend_test.cpp`, and `pipeline_test.cpp`. Numeric comparison with the
preserved Python outputs uses `tools/compare_sparse_outputs.py`; it is separate
from the unit tests and requires generated datasets/outputs.

## Shared data and error contracts

[`types.hpp`](include/types.hpp) defines typed IDs, poses, camera/feature records, tracks,
landmarks, reconstruction state, and provenance. [`status.hpp`](include/status.hpp)
defines `ErrorCode`, `Status`, and `Result<T>`.

### Critical parts

Camera/feature/track/landmark IDs are distinct types. Preserve stable references
when filtering or compacting records. Each track must contain at most one
observation per camera. Poses use explicitly named `outer_from_inner` semantics;
initial LIO poses and optimized poses are stored separately. Feature IDs correspond
to descriptor rows. Check `Result::ok()` before `value()` and propagate errors with
context instead of silently treating failures as empty results.

### Interactions

Every component consumes these contracts. Keep algorithm settings and methods in
their feature component; Core must not depend on Pipeline or another high-level
stage. Shared data structs remain separate from the stateless component classes.

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
