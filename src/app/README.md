# Qt OpenGL application

Build from the repository root:

```bash
cmake -S . -B build
cmake --build build --target 3dgs_cpp_qt_viewer -j$(nproc)
./build/3dgs_cpp_qt_viewer
```

The left pane is an interactive `QOpenGLWidget`. Drag horizontally to rotate
the sample object and use the mouse wheel to zoom. The right pane contains
starter controls and is the intended home for dataset, renderer, and bundle
adjustment controls.
