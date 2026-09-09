#!/usr/bin/env python3
"""Validate a completed reconstruction and print its evaluation metrics."""

import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, default=Path("data"))
p.add_argument("--output", type=Path, default=Path("output/lio_camera_pose"))
p.add_argument("--expected-images", type=int, default=-1)
a = p.parse_args()
m = json.load(open(a.output / "evaluation.json"))
poses = [x.split() for x in open(a.output / "poses_optimized_tum.txt") if x.strip()]
available = sum(
    1
    for line in open(a.data / "timestamps.txt")
    if (a.data / "undistorted" / line.split()[0]).exists()
)
expected = available if a.expected_images < 0 else min(available, a.expected_images)
assert m["pass"] is True, m
assert len(poses) == m["images"] == expected, (len(poses), m["images"], expected)
assert m["landmarks"] >= 100 and m["heldout_matches"] >= 30
assert m["reprojection_median_px"] <= 2 and m["reprojection_p90_px"] <= 4
assert m["heldout_epipolar_median_px"] <= 2 and m["heldout_epipolar_p90_px"] <= 5
assert (
    m["lio_translation_delta_median_m"] <= 0.3
    and m["lio_rotation_delta_median_deg"] <= 3
)
timestamps = []
for z in poses:
    vals = list(map(float, z))
    assert len(vals) == 8 and all(math.isfinite(x) for x in vals)
    assert abs(math.sqrt(sum(x * x for x in vals[4:8])) - 1) < 1e-5
    timestamps.append(vals[0])
assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
with open(a.output / "landmarks_optimized.ply") as f:
    header = []
    for line in f:
        header.append(line.strip())
        if line.strip() == "end_header":
            break
n = int(next(x.split()[-1] for x in header if x.startswith("element vertex")))
assert n == m["landmarks"], (n, m["landmarks"])
for required in [
    "sparse/0/cameras.txt",
    "sparse/0/images.txt",
    "sparse/0/points3D.txt",
]:
    assert (a.output / required).stat().st_size > 0, required
print(
    json.dumps(
        {"pass": True, "poses": len(poses), "landmarks": n, "metrics": m}, indent=2
    )
)
