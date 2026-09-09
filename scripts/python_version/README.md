# LIO-prior visual camera poses for 3DGS

This pipeline uses no COLMAP command or library. It uses OpenCV SIFT and Ceres
local/global bundle adjustment. Synchronization is t_lidar = t_image - 0.4 s.

## Running the pipeline

Run these commands from the repository root. The C++ stages require CMake,
Eigen, OpenCV, Ceres, nlohmann-json, and yaml-cpp. The Python stages require
Python 3 with NumPy and OpenCV; landmark diagnostic plots also require
Matplotlib.

The standard pipeline builds the two C++ executables and then runs pose
initialization, sparse triangulation, bundle adjustment, and output validation:

```bash
bash scripts/python_version/run_pipeline.sh \
  data output/lio_camera_pose -1
```

The positional arguments are `DATA_DIR`, `OUTPUT_DIR`, and `MAX_IMAGES`.
Use `-1` for every available image or a small value such as `40` for a smoke
test. The wrapper can also be launched from another directory because it
resolves the repository root from its own location.

The equivalent commands, useful when debugging one stage, are:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lio_camera_pose lio_bundle_adjust -j2
./build/lio_camera_pose --data data --output output/lio_camera_pose \
  --max-images -1 --time-offset -0.4
python3 scripts/python_version/triangulate_sparse.py \
  --data data --output output/lio_camera_pose --max-images -1
./build/lio_bundle_adjust --data data --output output/lio_camera_pose
python3 scripts/python_version/test_outputs.py \
  --data data --output output/lio_camera_pose
```

### Incremental alternative

The incremental controller is an alternative to `run_pipeline.sh`, not an
additional required stage. Build first, then let the controller grow the
active image set and alternate triangulation with local or global BA:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lio_camera_pose lio_bundle_adjust -j2
python3 scripts/python_version/run_incremental_pipeline.py \
  --data data --output output/lio_camera_pose_incremental --max-images -1
```

Use `--resume` to continue an interrupted incremental output. A clean run
should use a new output directory; the feature and pair caches are stored
under `<output>/cache`.

### Optional post-processing

After either reconstruction path, diagnose landmark support and optionally
export a filtered 3DGS dataset:

```bash
python3 scripts/python_version/diagnose_landmarks.py \
  --data data --input output/lio_camera_pose \
  --output output/lio_camera_pose/landmark_diagnostics
python3 scripts/python_version/export_filtered_3dgs.py \
  --input output/lio_camera_pose --images data/undistorted \
  --diagnostics output/lio_camera_pose/landmark_diagnostics \
  --output output/lio_camera_pose_3dgs --exclude-zero
```

An incremental run with a feature cache can extend existing tracks, then run
BA again on the extended output:

```bash
python3 scripts/python_version/extend_tracks.py \
  --data data --input output/lio_camera_pose_incremental \
  --output output/lio_camera_pose_extended \
  --cache-source output/lio_camera_pose_incremental/cache
./build/lio_bundle_adjust --data data \
  --output output/lio_camera_pose_extended
```

For targeted PnP recovery, create a text file containing whitespace-separated
zero-based frame indices, then run recovery and BA:

```bash
python3 scripts/python_version/recover_weak_frames.py \
  --data data --input output/lio_camera_pose_incremental \
  --output output/lio_camera_pose_recovered \
  --intrinsics output/lio_camera_pose_incremental/intrinsics_refined.txt \
  --cache-source output/lio_camera_pose_incremental/cache \
  --target-frames-file weak_frames.txt
./build/lio_bundle_adjust --data data \
  --output output/lio_camera_pose_recovered
```

### Optional depth and dense products

These stages use an optimized reconstruction but are independent of landmark
diagnosis and filtering. Package the existing local LiDAR depth maps directly:

```bash
python3 scripts/python_version/prepare_depth_supervision.py \
  --data data --reconstruction output/lio_camera_pose \
  --output output/depth_supervision
```

Or first complete them with nearby global LiDAR sweeps, then optionally add a
conservative low-confidence floor plane:

```bash
python3 scripts/python_version/generate_global_lidar_depth.py \
  --data data --reconstruction output/lio_camera_pose \
  --local-depth data/depth_maps --output output/depth_global
python3 scripts/python_version/add_floor_plane_depth.py \
  --data data --reconstruction output/lio_camera_pose \
  --input-depth output/depth_global --output output/depth_with_floor
```

Two independent dense point-cloud options are available. Fuse supplied depth
maps, or triangulate optical flow between nearby images:

```bash
python3 scripts/python_version/fuse_dense_cloud.py \
  --data data \
  --poses output/lio_camera_pose/poses_optimized_tum.txt \
  --output output/lio_camera_pose/dense_fused.ply
python3 scripts/python_version/dense_mvs_triangulate.py \
  --data data \
  --poses output/lio_camera_pose/poses_optimized_tum.txt \
  --output output/lio_camera_pose/dense_mvs.ply
```

Every Python command supports `--help`. Output directories should be distinct
between experiments because several stages intentionally overwrite their own
metrics and intermediate files.

Run every image:

    bash scripts/python_version/run_pipeline.sh data output/lio_camera_pose -1

Run a 40-image smoke test:

    bash scripts/python_version/run_pipeline.sh data output/lio_camera_pose_smoke 40

Stages: continuous LIO interpolation, calibrated camera initialization,
temporal SIFT matching, prior-pose epipolar verification, multi-view tracks,
parallax/cheirality/reprojection-filtered triangulation, overlapping local BA,
full global BA with robust LIO priors, and held-out evaluation.

Before BA, the triangulator now measures provisional three-view landmark and
8x6 grid support in every image. Images below either 50 landmarks or 16
occupied cells are matched over a wider 20-frame window and against up to
eight spatially nearby seed poses (including optimized poses supplied with
`--pose-file`). Recovery matches must be mutual. They normally pass the
pose-prior epipolar check; when that check fails, a weak frame may instead use
a MAGSAC fundamental matrix estimated only from mutual descriptor matches. The
fallback requires at least 20 inliers, a 25% inlier ratio, and support in at
least six image-grid cells. Sparse grid cells receive supplemental corner locations
with SIFT descriptors; only supplemental tracks validated in at least four
views are admitted to BA. `triangulation_metrics.json` records support before
and after recovery and the final strong support for every image.

The recovery targets and search effort are configurable with
`--target-landmarks-per-image`, `--target-grid-cells`,
`--weak-recovery-gap`, `--weak-spatial-distance`, and
`--weak-spatial-pairs`. Targets are goals, not a reason to accept geometrically
weak landmarks: an image can remain below target when the data has insufficient
repeatable texture or viewpoint overlap.

After an initial reconstruction, `extend_tracks.py` projects established
landmarks into every frame and merges descriptor-consistent observations into
their existing tracks. Frames below `--target-landmarks-per-image` use the
wider `--weak-search-radius`. If their pose-prior epipolar test fails, the same
distributed visual-geometry test can accept the observation. Every addition
remains one-to-one per frame and must survive multi-view re-triangulation and
reprojection cleanup. The metrics separate pose-prior and visual-only
additions. For example:

    python3 scripts/python_version/extend_tracks.py --data data \
      --input output/lio_camera_pose_incremental_v2 \
      --output output/lio_camera_pose_extended \
      --cache-source output/lio_camera_pose_incremental_v2/cache \
      --target-landmarks-per-image 50 --weak-search-radius 60

Outputs include poses_optimized_tum.txt, landmarks_optimized.ply,
evaluation.json, heldout.csv, tracks.csv, and 3DGS-compatible text files in
sparse/0. Held-out consistency and LIO agreement do not replace independent
survey, mocap, or RTK ground truth for proving absolute physical accuracy.

The `sparse/0` export contains only images with at least one surviving landmark
observation. Image and point IDs are compactly remapped, `images.txt` includes
the retained 2D-to-3D associations, and `points3D.txt` includes their matching
tracks. `poses_optimized_tum.txt` still retains every optimized/LIO-supported
camera for diagnostics.

For a strict reprojection target, run bundle adjustment with
`--max-reprojection-error 0.3`. After robust global BA, the optimizer removes
observations outside the target, refines the retained inlier set, and applies a
final target check. `evaluation.json` records the target, whether its maximum
was below it, and median/p90/p95/maximum reprojection errors. Tightening this threshold
trades observation density for a cleaner inlier set.

To reduce BA to a strong, image-covering subset, set `--max-landmarks`. The
selector first reserves strong tracks for weakly supported images up to
`--min-landmarks-per-image`, then fills the remaining budget using track
length, temporal span, parallax, and initial reprojection quality:

    ./build/lio_bundle_adjust --data data --output output/pruned \
      --max-landmarks 10000 --min-landmarks-per-image 20

Selection happens before BA and compactly remaps landmark IDs. The
`landmark_selection` section of `evaluation.json` records input/retained counts
and achieved image coverage. Omitting `--max-landmarks` preserves the previous
behavior.

For example, to refine an existing full-sequence result without rerunning
feature extraction and triangulation:

    ./build/lio_bundle_adjust --data data \
      --output output/lio_camera_pose_sub03 \
      --intrinsics output/lio_camera_pose_sub03/intrinsics_refined.txt \
      --local-start 537 --max-reprojection-error 0.3 \
      --cleanup-iterations 100

## Incremental mapper with loop landmark merging

The incremental controller follows this registration flow:

1. bootstrap 10 LIO-initialized cameras;
2. add 5 cameras per cycle;
3. match the new prefix against the six-frame recent window;
4. discover nonlocal candidates by LIO proximity and viewing direction;
5. mutually verify nonlocal SIFT matches with the LIO epipolar model;
6. union matching feature components, which extends existing tracks and merges
   landmarks observed on separate visits;
7. re-triangulate merged/new tracks and remove cheirality, parallax, and
   reprojection failures;
8. run local BA over the newest 15-camera window;
9. run global BA every 50 images and once at completion.

Run all images:

    python3 scripts/python_version/run_incremental_pipeline.py --data data \
      --output output/lio_camera_pose_incremental --max-images -1

Useful controls include `--bootstrap`, `--batch-size`, `--recent-window`,
`--global-ba-period`, `--loop-min-separation`, `--loop-max-distance`, and
`--loop-max-per-frame`. `incremental_history.json` records landmark, loop, and
reprojection statistics after every camera batch.

## Managed-track and accelerated BA version

The incremental path caches grid-balanced SIFT features and descriptor-pair
matches under `<output>/cache`. Cache files are versioned and reused by later
camera batches and safe reruns. Defaults use 15-camera batches and run global
BA every 100 cameras to avoid rebuilding 107 five-camera stages.

Track union rejects merges that introduce two observations from one camera.
Landmarks use all-view DLT, inconsistent observations are pruned, and tracks
are re-triangulated. Grid-balanced Shi-Tomasi locations with SIFT descriptors
supplement weakly textured cells without bypassing geometric checks.
Supplemental grid features are excluded from `sparse_points.ply`, `tracks.csv`,
local/global BA, and held-out pose evaluation. They are exported separately as
`coverage_points.ply` after multi-view validation, so coverage can improve
without weakening camera-pose optimization.

Local BA includes every observation of a locally touched landmark. Cameras
outside the window are fixed boundary cameras. A weak landmark residual tied
to the initial multi-view triangulation uses adaptive uncertainty from track
length and maximum parallax; reprojection remains the primary constraint.

Use a new output directory for a clean full run:

    python3 scripts/python_version/run_incremental_pipeline.py --data data \
      --output output/lio_camera_pose_incremental_v2 --max-images -1
