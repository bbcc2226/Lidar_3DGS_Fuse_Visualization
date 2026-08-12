#!/usr/bin/env python3
"""Print a histogram of observation counts per landmark in the landmark cache,
and check how many images are still covered if low-observation landmarks are
dropped (e.g. keeping only landmarks with >= 3 or >= 4 observations).

Reads the binary cache format written by SIFT::save_landmarks() (see
src/SIFT.cpp) without needing any image-processing dependencies.

Usage:
  python3 scripts/landmark_cache_stats.py
  python3 scripts/landmark_cache_stats.py --cache ../data/landmarks_cache.bin
  python3 scripts/landmark_cache_stats.py --coverage-thresholds 2,3,4 \
      --timestamps ../data/timestamps.txt
"""

from __future__ import annotations

import argparse
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List


def _read_exact(f, nbytes: int) -> bytes:
    data = f.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"Unexpected EOF: wanted {nbytes} bytes, got {len(data)}")
    return data


@dataclass
class LandmarkSummary:
    observation_count: int
    camera_ids: List[int] = field(default_factory=list)


def load_landmark_summaries(cache_path: str) -> List[LandmarkSummary]:
    summaries: List[LandmarkSummary] = []

    with open(cache_path, "rb") as f:
        landmark_count = struct.unpack("<q", _read_exact(f, 8))[0]

        for _ in range(landmark_count):
            struct.unpack("<i", _read_exact(f, 4))          # landmark_id
            _read_exact(f, 1)                                # optimized
            _read_exact(f, 24)                               # initial_position
            _read_exact(f, 24)                               # optimized_position

            observation_count = struct.unpack("<q", _read_exact(f, 8))[0]
            camera_ids: List[int] = []

            for _ in range(observation_count):
                camera_id = struct.unpack("<i", _read_exact(f, 4))[0]
                _read_exact(f, 4)                             # keypoint_idx
                _read_exact(f, 16)                            # pixel_x, pixel_y
                _read_exact(f, 8)                             # depth
                _read_exact(f, 8)                             # optimized_depth
                _read_exact(f, 1)                             # optimized
                camera_ids.append(camera_id)

            summaries.append(LandmarkSummary(observation_count, camera_ids))

    return summaries


def count_total_images(timestamps_path: str) -> int:
    with open(timestamps_path, "r") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="../data/landmarks_cache.bin")
    parser.add_argument(
        "--coverage-thresholds",
        default="2,3,4",
        help="Comma-separated min-observation thresholds to check image coverage for.",
    )
    parser.add_argument(
        "--timestamps",
        default="../data/timestamps.txt",
        help="Path used to count the total number of images (one per line).",
    )
    args = parser.parse_args()

    summaries = load_landmark_summaries(args.cache)

    counts: Counter = Counter(s.observation_count for s in summaries)
    total_landmarks = sum(counts.values())
    print(f"Total landmarks: {total_landmarks}")
    print(f"{'observations':>12}  {'landmarks':>10}  {'%':>6}")
    # Start at 2 (the minimum a landmark can have) even if the cache has none,
    # e.g. because prior_min_landmark_observations filtered them out.
    lowest = min(2, min(counts, default=2))
    highest = max(counts, default=2)
    for observation_count in range(lowest, highest + 1):
        landmarks = counts.get(observation_count, 0)
        pct = 100.0 * landmarks / total_landmarks if total_landmarks else 0.0
        print(f"{observation_count:>12}  {landmarks:>10}  {pct:6.2f}")

    try:
        total_images = count_total_images(args.timestamps)
    except OSError as exc:
        print(f"\n[coverage] Could not read --timestamps '{args.timestamps}': {exc}")
        return

    thresholds = [int(t) for t in args.coverage_thresholds.split(",") if t.strip()]

    print(f"\nTotal images (from {args.timestamps}): {total_images}")
    print(f"{'min_obs':>8}  {'images_covered':>14}  {'%':>6}  {'missing_images':>14}")
    for threshold in sorted(thresholds):
        covered_camera_ids = set()
        for summary in summaries:
            if summary.observation_count >= threshold:
                covered_camera_ids.update(summary.camera_ids)

        covered = len(covered_camera_ids)
        pct = 100.0 * covered / total_images if total_images else 0.0
        missing = max(0, total_images - covered)
        print(f"{threshold:>8}  {covered:>14}  {pct:6.2f}  {missing:>14}")


if __name__ == "__main__":
    main()

