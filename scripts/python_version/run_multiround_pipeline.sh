#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

# Purpose: Print command usage and supported multiround options.
# Inputs: None; reads no positional parameters.
# Outputs: Writes help text to standard output.
usage() {
    cat <<'EOF'
Usage: scripts/python_version/run_multiround_pipeline.sh [options]

Run camera initialization, sparse triangulation, first-round BA, track
extension, second-round BA, pruning, and final validation.

Options:
  --data DIR                    Prepared data directory
  --first-output DIR            First-round reconstruction output
  --base DIR                    Alias for --first-output
  --cache-source DIR            Feature cache (default: FIRST_OUTPUT/cache)
  --output DIR                  New second-round output directory
  --max-images N                Maximum images; -1 uses all (default: -1)
  --translation-prior-weight N  Translation prior weight (default: 100)
  --rotation-prior-weight N     Rotation prior weight (default: 10)
  --search-radius N             Normal-frame search radius in pixels (default: 20)
  --weak-search-radius N        Weak-frame search radius in pixels (default: 60)
  --target-landmarks N          Target landmarks per image (default: 50)
  --skip-first-round            Reuse an existing first-round output
  --force                       Remove outputs that this invocation rebuilds
  -h, --help                    Show this help

Drive 0035 defaults are used when no options are supplied.
EOF
}

data_dir="data_kitti_0035_0_119"
first_output="output/kitti_0035_python_0_119_multiround_first"
cache_source=""
output_dir="output/kitti_0035_python_0_119_tprior100_extended"
max_images="-1"
translation_prior_weight="100"
rotation_prior_weight="10"
search_radius="20"
weak_search_radius="60"
target_landmarks="50"
skip_first_round=0
force=0

while (($#)); do
    case "$1" in
        --data) data_dir="$2"; shift 2 ;;
        --first-output|--base) first_output="$2"; shift 2 ;;
        --cache-source) cache_source="$2"; shift 2 ;;
        --output) output_dir="$2"; shift 2 ;;
        --max-images) max_images="$2"; shift 2 ;;
        --translation-prior-weight) translation_prior_weight="$2"; shift 2 ;;
        --rotation-prior-weight) rotation_prior_weight="$2"; shift 2 ;;
        --search-radius) search_radius="$2"; shift 2 ;;
        --weak-search-radius) weak_search_radius="$2"; shift 2 ;;
        --target-landmarks) target_landmarks="$2"; shift 2 ;;
        --skip-first-round) skip_first_round=1; shift ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

intrinsics="$data_dir/intrinsics.txt"
if [[ -z "$cache_source" ]]; then
    cache_source="$first_output/cache"
fi
required=(
    "$intrinsics"
    "$data_dir/timestamps.txt"
    "$data_dir/key_frames.jsonl"
    "$data_dir/tuned_camera_lidar_extrinsic.json"
    "$data_dir/undistorted"
    "build/lio_camera_pose"
    "build/lio_bundle_adjust"
)
for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input is missing: $path" >&2
        exit 2
    fi
done

# Purpose: Create an empty stage output directory according to --force policy.
# Inputs: $1 is the output path; reads the global force flag.
# Outputs: Creates or replaces the directory, or exits if it is non-empty.
prepare_output() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        mkdir -p "$path"
        return
    fi
    if ((force)); then
        rm -rf -- "$path"
        mkdir -p "$path"
    elif [[ -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Output is not empty: $path (use --force to replace it)" >&2
        exit 2
    fi
}

if ((!skip_first_round)); then
    prepare_output "$first_output"
    first_log="$first_output/first_round.log"
    : > "$first_log"

    echo "[1/7] Initializing camera poses from the LIO trajectory"
    ./build/lio_camera_pose \
        --data "$data_dir" \
        --output "$first_output" \
        --max-images "$max_images" \
        2>&1 | tee -a "$first_log"

    echo "[2/7] Extracting features, matching, and triangulating first-round landmarks"
    python3 "$script_dir/triangulate_sparse.py" \
        --data "$data_dir" \
        --output "$first_output" \
        --intrinsics "$intrinsics" \
        --max-images "$max_images" \
        --target-landmarks-per-image "$target_landmarks" \
        2>&1 | tee -a "$first_log"

    echo "[3/7] Running first-round BA, pruning, and cleanup BA"
    ./build/lio_bundle_adjust \
        --data "$data_dir" \
        --output "$first_output" \
        --intrinsics "$intrinsics" \
        --max-images "$max_images" \
        --local-start 0 \
        --translation-prior-weight "$translation_prior_weight" \
        --rotation-prior-weight "$rotation_prior_weight" \
        --no-robust-pose-prior \
        2>&1 | tee -a "$first_log"

    echo "[4/7] Validating first-round reconstruction"
    python3 "$script_dir/test_outputs.py" \
        --data "$data_dir" \
        --output "$first_output" \
        --expected-images "$max_images" \
        2>&1 | tee -a "$first_log"
else
    echo "[1-4/7] Reusing first-round reconstruction: $first_output"
fi

first_required=(
    "$first_output/poses_optimized_tum.txt"
    "$first_output/poses_lio_prior_tum.txt"
    "$first_output/landmarks_optimized.ply"
    "$first_output/tracks.csv"
    "$first_output/heldout.csv"
    "$cache_source/features_v5_weak_recovery"
)
for path in "${first_required[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "First-round artifact is missing: $path" >&2
        exit 2
    fi
done

prepare_output "$output_dir"
log="$output_dir/multiround.log"
: > "$log"

echo "[5/7] Extending landmark tracks from first-round optimized poses"
python3 "$script_dir/extend_tracks.py" \
    --data "$data_dir" \
    --input "$first_output" \
    --output "$output_dir" \
    --intrinsics "$intrinsics" \
    --cache-source "$cache_source" \
    --search-radius "$search_radius" \
    --weak-search-radius "$weak_search_radius" \
    --descriptor-ratio 0.82 \
    --descriptor-max 240 \
    --epipolar-px 1.25 \
    --final-reprojection-px 2.0 \
    --max-depth 30 \
    --target-landmarks-per-image "$target_landmarks" \
    2>&1 | tee -a "$log"

echo "[6/7] Running second-round BA, pruning, and cleanup BA"
./build/lio_bundle_adjust \
    --data "$data_dir" \
    --output "$output_dir" \
    --intrinsics "$intrinsics" \
    --max-images "$max_images" \
    --local-start 0 \
    --translation-prior-weight "$translation_prior_weight" \
    --rotation-prior-weight "$rotation_prior_weight" \
    --no-robust-pose-prior \
    2>&1 | tee -a "$log"

echo "[7/7] Validating final reconstruction"
python3 "$script_dir/test_outputs.py" \
    --data "$data_dir" \
    --output "$output_dir" \
    --expected-images "$max_images" \
    2>&1 | tee -a "$log"

python3 - "$first_output" "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

first_output = Path(sys.argv[1])
output = Path(sys.argv[2])
first = json.loads((first_output / "evaluation.json").read_text())
extension = json.loads((output / "track_extension_metrics.json").read_text())
evaluation = json.loads((output / "evaluation.json").read_text())
summary = {
    "first_round_output": str(first_output.resolve()),
    "final_output": str(output.resolve()),
    "first_round": {
        "landmarks": first["active_landmarks"],
        "observations": first["observations"],
        "reprojection_median_px": first["reprojection_median_px"],
        "reprojection_p90_px": first["reprojection_p90_px"],
        "pass": first["pass"],
    },
    "track_extension": {
        "proposed_observations": extension["proposed_observations"],
        "retained_new_observations": extension["retained_new_observations"],
        "rejected_new_observations": extension["rejected_new_observations"],
    },
    "second_round": {
        "landmarks": evaluation["active_landmarks"],
        "observations": evaluation["observations"],
        "reprojection_median_px": evaluation["reprojection_median_px"],
        "reprojection_p90_px": evaluation["reprojection_p90_px"],
        "pass": evaluation["pass"],
    },
    "translation_prior_weight": evaluation["translation_prior_weight"],
    "rotation_prior_weight": evaluation["rotation_prior_weight"],
}
(output / "multiround_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
