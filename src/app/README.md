# Qt OpenGL application

Build from the repository root:

```bash
cmake -S . -B build
cmake --build build --target 3dgs_cpp_qt_viewer -j$(nproc)
./build/3dgs_cpp_qt_viewer
```

The left pane is an interactive `QOpenGLWidget`. Left-drag to rotate in yaw and
pitch, right-drag to pan, and use the mouse wheel to dolly into or away from the
point cloud. Click the X, Y, or Z
endpoint on the lower-left orientation gizmo to snap to that axis. The right
pane contains dataset and view controls.

`OpenGLWidget` owns only OpenGL resources, render data, camera/render state,
and the initialize/resize/paint lifecycle. `ViewerController` owns mouse input,
file selection, and point-cloud loading. `MainWindow` only builds the controls
and connects their signals.
