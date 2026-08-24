# Qt OpenGL application

Build from the repository root:

```bash
cmake -S . -B build
cmake --build build --target 3dgs_cpp_qt_viewer -j$(nproc)
./build/3dgs_cpp_qt_viewer
```

The left pane is an interactive `QOpenGLWidget`. Left-drag to rotate in yaw and
pitch, Shift+left-drag for pitch-only rotation, right-drag to pan, and use the
mouse wheel to dolly into or away from the point cloud. Enable **Z-up constrained
rotation** to reset the orientation and lock each unmodified left-drag to either
horizontal yaw or vertical pitch once that direction is clearly dominant.
Ambiguous diagonal movement waits instead of rotating both axes. Click the X, Y, or Z
endpoint on the lower-left orientation gizmo to snap to that axis. The right
pane contains dataset and view controls.

In Z-up mode, press the Up or Down arrow key to raise or lower the camera along
world Z without changing its viewing direction or horizontal position.

Use **Align floor: 3 points** when the PLY coordinate frame is tilted. Click
three widely separated Gaussian locations on the same floor surface. The
resulting plane normal is rotated onto the visualization +Z axis.

`OpenGLWidget` owns only OpenGL resources, render data, camera/render state,
and the initialize/resize/paint lifecycle. `ViewerController` owns mouse input,
file selection, and point-cloud loading. `MainWindow` only builds the controls
and connects their signals.
