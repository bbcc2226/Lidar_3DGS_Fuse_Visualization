#!/usr/bin/env python3
"""Compare legacy Python and C++ sparse reconstruction outputs."""

import argparse
import json
import math
from pathlib import Path


def load_poses(path):
    poses = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 8:
            raise ValueError(f"invalid TUM pose line in {path}: {line}")
        timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, values)
        poses[round(timestamp, 9)] = ((tx, ty, tz), (qx, qy, qz, qw))
    return poses


def quaternion_angle_degrees(first, second):
    dot = abs(sum(a * b for a, b in zip(first, second)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def relative_delta(first, second):
    return abs(second - first) / max(abs(first), 1.0)


def compare(python_output, cpp_output, thresholds):
    python_metrics = json.loads((python_output / "evaluation.json").read_text())
    cpp_metrics = json.loads((cpp_output / "evaluation.json").read_text())
    python_poses = load_poses(python_output / "poses_optimized_tum.txt")
    cpp_poses = load_poses(cpp_output / "poses_optimized_tum.txt")
    shared_timestamps = sorted(set(python_poses) & set(cpp_poses))
    translation_errors = []
    rotation_errors = []
    for timestamp in shared_timestamps:
        python_translation, python_rotation = python_poses[timestamp]
        cpp_translation, cpp_rotation = cpp_poses[timestamp]
        translation_errors.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(
            python_translation, cpp_translation))))
        rotation_errors.append(quaternion_angle_degrees(python_rotation, cpp_rotation))

    metric_deltas = {}
    for key in sorted(set(python_metrics) & set(cpp_metrics)):
        if isinstance(python_metrics[key], (int, float)) and not isinstance(python_metrics[key], bool) \
                and isinstance(cpp_metrics[key], (int, float)) and not isinstance(cpp_metrics[key], bool):
            metric_deltas[key] = cpp_metrics[key] - python_metrics[key]

    checks = {
        "all_cpp_poses_aligned": len(shared_timestamps) == len(cpp_poses) == len(python_poses),
        "translation_rmse": rms(translation_errors) <= thresholds["translation_rmse_m"],
        "rotation_rmse": rms(rotation_errors) <= thresholds["rotation_rmse_deg"],
        "landmark_count": relative_delta(python_metrics.get("landmarks", 0),
                                         cpp_metrics.get("landmarks", 0))
                          <= thresholds["landmark_relative"],
        "observation_count": relative_delta(python_metrics.get("observations", 0),
                                            cpp_metrics.get("observations", 0))
                             <= thresholds["observation_relative"],
    }
    if "reprojection_p90_px" in python_metrics and "reprojection_p90_px" in cpp_metrics:
        checks["reprojection_p90"] = (
            cpp_metrics["reprojection_p90_px"] - python_metrics["reprojection_p90_px"]
            <= thresholds["reprojection_p90_delta_px"])
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "pose_comparison": {
            "python_poses": len(python_poses),
            "cpp_poses": len(cpp_poses),
            "aligned_poses": len(shared_timestamps),
            "translation_rmse_m": rms(translation_errors),
            "translation_max_m": max(translation_errors, default=0.0),
            "rotation_rmse_deg": rms(rotation_errors),
            "rotation_max_deg": max(rotation_errors, default=0.0),
        },
        "python_metrics": python_metrics,
        "cpp_metrics": cpp_metrics,
        "cpp_minus_python_metrics": metric_deltas,
        "thresholds": thresholds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-output", type=Path, required=True)
    parser.add_argument("--cpp-output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--translation-rmse-m", type=float, default=0.10)
    parser.add_argument("--rotation-rmse-deg", type=float, default=1.5)
    parser.add_argument("--landmark-relative", type=float, default=0.10)
    parser.add_argument("--observation-relative", type=float, default=0.10)
    parser.add_argument("--reprojection-p90-delta-px", type=float, default=0.25)
    args = parser.parse_args()
    thresholds = {
        "translation_rmse_m": args.translation_rmse_m,
        "rotation_rmse_deg": args.rotation_rmse_deg,
        "landmark_relative": args.landmark_relative,
        "observation_relative": args.observation_relative,
        "reprojection_p90_delta_px": args.reprojection_p90_delta_px,
    }
    try:
        report = compare(args.python_output, args.cpp_output, thresholds)
    except (OSError, ValueError, json.JSONDecodeError) as exception:
        parser.error(str(exception))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
