# V2 to extended-tracker changes report

## Scope

V2 is the managed incremental mapper driven by
`run_incremental_pipeline.py` and `triangulate_sparse.py`. The extended tracker
is the pose-guided completion pass in `extend_tracks.py`, followed by bundle
adjustment. It augments existing landmarks; it does not replace V2 pose
initialization, loop matching, or initial triangulation.

## Processing changes

| Area | V2 managed mapper | Extended tracker |
|---|---|---|
| Processing direction | Builds the reconstruction incrementally in image batches | Revisits every selected frame after a V2 reconstruction exists |
| Candidate source | Temporal pairs plus LIO-proximity nonlocal loop candidates | Projects every existing 3D landmark into candidate frames |
| Feature data | Cached, grid-balanced core SIFT features and pair matches | Reuses cached core SIFT locations/descriptors |
| Landmark descriptor | Pairwise feature descriptors used during track construction | Median descriptor formed from the landmark's existing observations |
| Search | Pair matching inside recent/loop candidate image pairs | Local pixel-grid search around each projected landmark, default radius 18 px |
| Descriptor gates | Mutual/pairwise matching and ratio tests in triangulation | Maximum descriptor distance 240 and ratio threshold 0.82 |
| Geometry gates | Prior-pose epipolar verification, cheirality, parallax, and multiview reprojection | Positive depth, maximum depth 30 m, image bounds, and 1.25 px epipolar error against the nearest existing observation |
| Conflict handling | Rejects track unions that create two features from one camera | Greedy one-to-one feature assignment per frame; competing landmark proposals are rejected |
| 3D update | Multiview DLT triangulates new or merged tracks | Retrangulates each augmented landmark from all old and proposed observations |
| New-observation cleanup | Track-level triangulation filters | Removes only newly proposed observations above 2 px reprojection error; original observations remain for BA cleanup |
| Pose handling | Local BA on the newest window, fixed boundary cameras, periodic global BA | Track extension keeps poses fixed, then writes V2 poses as BA initialization for a subsequent global refinement |
| Held-out data | Used only for evaluation | Explicitly not used to accept extended observations |

## New extended-tracker controls

`extend_tracks.py` adds controls for `--search-radius`, `--descriptor-ratio`,
`--descriptor-max`, `--epipolar-px`, `--final-reprojection-px`, `--max-depth`,
an optional target-frame list, a separate cache source, and refined
intrinsics. It also records proposed/retained/rejected observations and
one-to-one assignment conflicts in `track_extension_metrics.json`.

## Recorded result

The saved extension metrics report:

- 101,484 initial observations;
- 79,696 geometrically and descriptively proposed observations;
- 64,649 retained new observations;
- 15,047 rejected new observations; and
- 11,066 one-to-one assignment conflicts.

The subsequent full-sequence extended result passes its evaluation with 537
poses, 164,692 post-BA observations, 0.315804 px median reprojection error, and
0.854868 px p90 reprojection error. Its held-out epipolar median/p90 are
0.203634/1.510298 px and median LIO translation/rotation deltas are
0.055929 m/1.090580 degrees.

The extension metrics and final evaluation were saved at different pipeline
checkpoints, so their raw landmark counts should not be subtracted directly.
The defensible conclusion is that the extension substantially increases
cross-frame support for existing landmarks, while robust BA removes additions
that do not remain jointly consistent.

## Current strict export added after the extended tracker

The bundle-adjustment exporter now omits cameras with zero surviving landmark
observations from `sparse/0`, compactly remaps image and point IDs, writes real
2D-to-3D entries in `images.txt`, and writes the matching tracks in
`points3D.txt`. The complete pose trajectory remains in
`poses_optimized_tum.txt` for diagnostics.
