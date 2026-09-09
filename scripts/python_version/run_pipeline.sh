#!/usr/bin/env bash
set -euo pipefail
data_dir="${1:-data}"
output_dir="${2:-output/lio_camera_pose}"
max_images="${3:--1}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lio_camera_pose lio_bundle_adjust -j2
./build/lio_camera_pose --data "$data_dir" --output "$output_dir" --max-images "$max_images" --time-offset -0.4
python3 standalone/triangulate_sparse.py --data "$data_dir" --output "$output_dir" --max-images "$max_images"
./build/lio_bundle_adjust --data "$data_dir" --output "$output_dir"
python3 standalone/test_outputs.py --data "$data_dir" --output "$output_dir"
