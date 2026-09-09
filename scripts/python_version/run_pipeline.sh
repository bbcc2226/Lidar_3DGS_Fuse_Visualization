#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

data_dir="${1:-data}"
output_dir="${2:-output/lio_camera_pose}"
max_images="${3:--1}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lio_camera_pose lio_bundle_adjust -j2
./build/lio_camera_pose --data "$data_dir" --output "$output_dir" --max-images "$max_images" --time-offset -0.4
python3 "$script_dir/triangulate_sparse.py" --data "$data_dir" --output "$output_dir" --max-images "$max_images"
./build/lio_bundle_adjust --data "$data_dir" --output "$output_dir"
python3 "$script_dir/test_outputs.py" --data "$data_dir" --output "$output_dir"
