# Qt OpenGL application

Build from the repository root:

```bash
cmake -S . -B build
cmake --build build --target 3dgs_cpp_qt_viewer -j$(nproc)
./build/3dgs_cpp_qt_viewer
```

The left pane is an interactive `QOpenGLWidget`. Left-drag to rotate in yaw and
pitch, Shift+left-drag for pitch-only rotation, right-drag to pan, and use the
mouse wheel to dolly into or away from the point cloud. Select **Navigate** to
automatically align the scene's shortest principal axis with +Z, reset the
orientation, and lock each unmodified left-drag to either
horizontal yaw or vertical pitch once that direction is clearly dominant.
Ambiguous diagonal movement waits instead of rotating both axes. Click the X, Y, or Z
endpoint on the lower-left orientation gizmo to snap to that axis. The right
pane contains dataset and view controls.

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
limit. The camera is placed 1.0 m above the estimated floor. While playback is
active, viewport navigation is locked to the simulated robot. After pausing,
left-drag looks around, Shift+left-drag adjusts pitch, right-drag pans, the
wheel dollies, and Up/Down changes height. Resuming restores the path position
and simulated heading, discarding those temporary inspection adjustments.
**Stop** returns
to the first point; reaching
the end leaves the camera at the final point.
**Clear loaded path** stops playback and removes the manual path and its
destination marker without unloading the 3DGS scene or optimized trajectory.

The **Navigate & Path** tab includes a navigation request search/chat field and
a **Navigate** button. They are layout placeholders only and intentionally have
no callback until destination lookup or conversational navigation is added.

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
