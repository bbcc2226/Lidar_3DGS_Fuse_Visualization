# Qt OpenGL application

Build from the repository root:

```bash
cmake -S . -B build
cmake --build build --target 3dgs_cpp_qt_viewer -j$(nproc)
./build/3dgs_cpp_qt_viewer
```

For the presentation-focused version, launch:

```bash
./build/3dgs_cpp_qt_viewer --demo
```

Demo mode shows loading progress while reading the 3DGS file, then prepares
the renderer and exploration path. It keeps only Explore and Navigate. It preloads
`data/point_cloud_curr_best.ply`, `data/9_12_manual_path.json`, and
`data/9_10walkable_cells.json`; Explore retains camera/key interaction,
playback speed, and play/pause/resume, while Navigate retains the destination
chat box. In demo **Explore**, enable **Show semantic objects (2D boxes)** to
see only verified (Confirmed) objects with confidence labeled `high` and their
approximate bounds update along
the path. Click a box,
including during playback, to highlight it in cyan and show only its description in
the right panel. Playback continues without changing the camera or route.
The overlay is hidden in Navigate and restored when returning to Explore.
Launching without `--demo` keeps the complete editor as the default.

The left pane is an interactive `QOpenGLWidget`. Left-drag to rotate in yaw and
pitch, Shift+left-drag for pitch-only rotation, right-drag to pan, and use the
mouse wheel to dolly into or away from the point cloud. Select **Navigate** to
automatically align the scene's shortest principal axis with +Z, reset the
orientation, and lock each unmodified left-drag to either
horizontal yaw or vertical pitch once that direction is clearly dominant.
Ambiguous diagonal movement waits instead of rotating both axes. Click the X, Y, or Z
endpoint on the lower-left orientation gizmo to snap to that axis. The right
pane contains dataset and view controls.

The **3DGS view / Colored LiDAR view** selector switches between the reconstructed
Gaussian scene and the merged colored LiDAR cloud in
`data/lidar_related/dense_lidar_rgb_latest_50MiB.ply`. The LiDAR cloud is loaded
prepared in a background task and cached, so switching back and forth does not
reload or reparse it. Switching
sources preserves the current camera and robot viewpoint, allowing a direct
visual comparison even when the two maps are only approximately registered.
The LiDAR display currently keeps the complete merged cloud for maximum
structure detail. The original merged file is never modified; spatial sampling
code remains available if the full-density upload is too heavy for a particular
GPU.
Colored LiDAR also avoids repeated depth sorting after camera interactions,
which keeps navigation responsive; its first upload still pays the one-time
buffer cost.
The camera pose and LiDAR extrinsic remain available for future registration
adjustments. The same switch is available in demo **Explore** after the initial
3DGS scene has loaded; demo **Navigate** remains unchanged.

For semantic map screenshots in the normal app, open **Semantic** and enable
**Show all objects in view**. This displays object names and IDs whose centers
are in the current view, ignoring the name filter and hiding bounding boxes.
Enabling it keeps the camera in place; object selection does not reposition the
camera while it is enabled. Uncheck it to restore selected-object inspection. Enable **Only verified objects**
to restrict the overlay to objects marked Confirmed. Enable **Use approximate 2D
boxes** to draw screen-aligned rectangles enclosing the projected 3D bounds,
including in the all-objects view. These approximate boxes are not segmentation
outlines. With 2D boxes disabled, the all-objects view remains labels only.

Click inside a 2D box to select and scroll to its object in the list without
moving the camera. If boxes overlap, the smallest box wins. Dragging still
rotates the view. A name filter hiding the clicked list entry is cleared.
Use **Confirm**, **Unsure**, or **Incorrect**, then **Update current object file**
to write `review_status` into the loaded database and save its companion review
file. Other database fields are preserved. **Save reviews** and **Save reviews
as...** remain available for saving review results separately.

A bottom-right top-down minimap is extracted automatically from the loaded
point cloud as a lightweight XY occupancy landscape. In **Navigate** mode, the
orange arrow shows the current robot/camera position and heading.

## Smooth optimized trajectory

`TrajectoryProcessing` reads the COLMAP-style world-to-camera poses in
`data/output_optimized_camera.txt`, inverts each pose to recover its world-space
camera center, and creates a Gaussian-smoothed XY path at a fixed height:

```cpp
TrajectoryProcessing trajectory;
if (!trajectory.loadOptimizedCameraPoses(
        "data/output_optimized_camera.txt")) {
    throw std::runtime_error(trajectory.lastError());
}

TrajectorySmoothingOptions options;
options.window_radius = 4;
options.gaussian_sigma = 2.0;
// Leave fixed_height as NaN to use the median camera height.
if (!trajectory.smoothTrajectory(options)) {
    throw std::runtime_error(trajectory.lastError());
}
trajectory.saveSmoothTrajectory("data/output_smoothed_trajectory.txt");
```

The output columns are `IMAGE_ID X Y Z IMAGE_NAME`. Both raw and smoothed
positions remain in the input pose's world coordinate system; apply the same
scene alignment and normalization matrices as the point cloud before drawing
them in the viewer.

The viewer's **Open trajectory...** button performs that aligned path directly:
load the point cloud, switch to **Navigate** to establish Z-up, and select
`data/output_optimized_camera.txt`. The minimap shows raw poses in translucent
gray, the aligned and fixed-height smoothed path in cyan, its start in green,
and its end in red. Changing working mode recomputes the path in the current
scene-aligned frame.

The minimap also shows an experimental 3DGS-only clearance map: red cells are
Gaussian-derived obstacles, orange cells lack clearance for the default 0.30 m
robot radius plus 0.10 m margin, faint green cells are candidate walkable
space, and magenta marks an unresolved pose/geometry conflict. Yellow is the
original smoothed proposal; bright green is the path after unsafe samples are projected
to nearby candidate cells. This map is conservative diagnostic output, not yet
a certified navigation map.

The main 3D viewport renders the same trajectory in the aligned scene: the raw
camera path is a faint gray line, the corrected fixed-height path is bright green, and
green/red points mark its start/end. The path follows the active orbit or
Navigate camera because it uses the same view and projection matrices as the
Gaussian scene.

**Edit Path** switches to a Z-up orthographic view for manual walk-path
selection. Left-click empty space to append a point, left-drag an existing
point to move it, and right-click or press Delete/Backspace to remove the
selected point. The wheel zooms the orthographic view. The connected bright
green manual path remains visible after returning to Explore or Navigate.

**Save path...** stores the ordered manual control points as JSON in the
original `3dgs_world` coordinate frame. The viewer reverses its temporary
normalization and Z-up display alignment before writing, so a later robot-path
player can interpolate the saved points in scene coordinates. Saving uses an
atomic file replacement to avoid leaving a partial path file.

For simulated playback, use **Load robot path...**, choose a saved manual-path
JSON file, and press **Play / Pause**. The default speed is about `0.17 m/s`
(one third of the original speed). Playback uses metric arc length in
`3dgs_world`, a 0.4 m look-ahead, a 1.2 s yaw filter, and a 20 degree/s yaw-rate
limit. **Playback camera height** selects **Constant height** (initially
1.10 m above the estimated floor) or **Original trajectory height** (the
height of each path point). Paths with `"height_mode": "recorded"`, including
the generated KITTI camera path, select original height when loaded. You can
switch modes while paused or playing. While playback is
active, viewport navigation is locked to the simulated robot. After pausing,
left-drag looks around, Shift+left-drag adjusts pitch, right-drag pans, the
wheel dollies, and Up/Down changes height. Resuming restores the path position
and simulated heading. Up/Down height adjustments persist across resume;
original-height mode treats them as an offset from the recorded trajectory.
Reloading a path clears that offset.
**Stop** returns
to the first point; reaching
the end leaves the camera at the final point.
**Clear loaded path** stops playback and removes the manual path and its
destination marker without unloading the 3DGS scene or optimized trajectory.

The **Navigate & Path** tab includes a navigation request search/chat field and
a **Navigate** button. Enter a class or alias, optionally with description
keywords, for example `go to the white mug` or `go to the mug with a blue handle`.
The longest known class or alias identifies the object category. All remaining
keywords must appear as whole words in the stored object description, ignoring
case and punctuation (`a`, `an`, `the`, `with`, and `and` are ignored).
This is literal keyword filtering, without synonym inference or negation handling.
Description filters also combine with supported spatial relations, for example
`go to the white mug near the microwave`; the reference remains a class or alias.
Only confirmed matching objects enter destination ranking and route planning.
If no description matches, navigation reports this instead of selecting another
object of the same class.

**Working mode** selects between **Explore**, the complete-scene free-orbit
view, and **Navigate**, a level first-person Z-up view starting near the center
of the normalized scene. The Navigate start is intentionally provisional and
can later be replaced by the first camera or trajectory pose.

In Z-up mode, press the Up or Down arrow key to raise or lower the camera along
world Z without changing its viewing direction or horizontal position.
Wheel zoom follows the full mouse-wheel or trackpad delta, and right-drag uses
an increased screen-relative pan sensitivity. Both retain fixed camera height
in Z-up mode.

Use **Exposure (EV)** to compare display brightness with another viewer without
changing the stored SH coefficients. `0.0` preserves the trained color values;
each `+1.0` EV doubles displayed RGB intensity.

GPU-assisted record reordering is experimental and disabled by default. It can
be enabled for diagnostics with `LIDAR_3DGS_GPU_REORDER=1`, but the CPU radix
path remains the stable default, particularly on integrated GPUs.

Rasterization controls preserve reference defaults at `0.3` low-pass variance,
`256` px maximum splat half-size, and `0.0` minimum opacity. For a sharper demo,
try variance `0.15`; for floaters, try maximum size `128` and minimum opacity
`0.01`, adjusting conservatively to avoid holes or flicker.

Enable **Highlight large splats** to diagnose the broad translucent patches
that can make a scene look blurred. Splats become increasingly red above half
of the selected maximum projected half-size and are fully red near the limit.
This diagnostic changes only their displayed color. The decorative viewport
grid is off by default and can be restored with **Show viewport grid**.
**Suppress oversized splats** is an optional diagnostic. It fades projections
between one and two times the selected maximum size, but can remove valid
trained scene detail, so it is disabled for normal rendering.

The normal fragment path uses a compact normalized Gaussian footprint matching
SuperSplat's `exp(-4)` boundary convention. Conventional framebuffer MSAA is
disabled because the splat kernel already performs analytic antialiasing.
When a projected ellipse reaches the maximum-size guard, both its quad and
inverse covariance are reduced together so the Gaussian still reaches zero at
the boundary without exposing polygon edges.


`OpenGLWidget` owns only OpenGL resources, render data, camera/render state,
and the initialize/resize/paint lifecycle. `ViewerController` owns mouse input,
file selection, and point-cloud loading. `MainWindow` only builds the controls
and connects their signals.

The top-down orthographic editor projects Gaussian sizes independently of depth,
so splats retain consistent surface coverage at different scene heights.

Top-down editing removes Gaussians whose full vertical support extends above
the ceiling cutoff, reducing elongated ceiling and upper-wall artifacts. The
floor-relative cutoff also works without a loaded camera trajectory.

Explore starts with a cleaned orthographic overview. **Path overview** returns
there and pauses playback; the wheel zooms, dragging returns to orbit, and
playback restores the path camera. Editing and overview retain the original
Gaussian footprints, trim the sparse outer 0.25% with a 10 cm margin, and use
a slightly lower ceiling cutoff to reduce spikes while preserving floor
coverage. These display filters do not modify the PLY or path/cell data.
