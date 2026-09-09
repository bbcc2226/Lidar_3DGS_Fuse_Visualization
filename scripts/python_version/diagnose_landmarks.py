#!/usr/bin/env python3
"""Report per-image and per-track quality for an optimized reconstruction."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def qrot(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def poses(path):
    out = []
    for line in path.open():
        z = line.split()
        if len(z) != 8:
            continue
        x, y, zz, w = map(float, z[4:8])
        out.append((np.array(z[1:4], float), qrot([w, x, y, zz])))
    return out


def ply(path):
    with path.open() as f:
        n = 0
        for line in f:
            if line.startswith("element vertex "):
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        return np.array(
            [[float(x) for x in f.readline().split()[:3]] for _ in range(n)]
        )


def project(X, pose, K):
    t, R = pose
    q = (X - t) @ R
    if q[2] <= 0:
        return None
    return np.array([K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--min-track-length", type=int, default=4)
    ap.add_argument("--max-median-reprojection-px", type=float, default=0.75)
    ap.add_argument("--max-p90-reprojection-px", type=float, default=1.5)
    ap.add_argument("--min-parallax-deg", type=float, default=1.0)
    ap.add_argument("--grid-x", type=int, default=8)
    ap.add_argument("--grid-y", type=int, default=6)
    a = ap.parse_args()
    out = a.output or a.input / "landmark_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    K = np.loadtxt(a.input / "intrinsics_refined.txt", delimiter=",")
    P = poses(a.input / "poses_optimized_tum.txt")
    X = ply(a.input / "landmarks_optimized.ply")
    names = [line.split()[0] for line in (a.data / "timestamps.txt").open()][: len(P)]
    sample = cv2.imread(str(a.data / "undistorted" / names[0]))
    h, w = sample.shape[:2]
    tracks = defaultdict(list)
    for r in csv.DictReader((a.input / "tracks.csv").open()):
        tracks[int(r["track"])].append((int(r["frame"]), float(r["u"]), float(r["v"])))
    quality = {}
    frame_obs = defaultdict(list)
    for tid, obs in tracks.items():
        if tid >= len(X):
            continue
        errs = []
        positive = True
        rays = []
        for fi, u, v in obs:
            uv = project(X[tid], P[fi], K)
            if uv is None:
                positive = False
                err = 1e9
            else:
                err = float(np.linalg.norm(uv - [u, v]))
            errs.append(err)
            ray = X[tid] - P[fi][0]
            rays.append(ray / (np.linalg.norm(ray) + 1e-12))
        max_angle = 0.0
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                max_angle = max(
                    max_angle,
                    math.degrees(math.acos(np.clip(np.dot(rays[i], rays[j]), -1, 1))),
                )
        med = float(np.median(errs))
        p90 = float(np.percentile(errs, 90))
        span = max(x[0] for x in obs) - min(x[0] for x in obs)
        strong = (
            len(obs) >= a.min_track_length
            and med <= a.max_median_reprojection_px
            and p90 <= a.max_p90_reprojection_px
            and max_angle >= a.min_parallax_deg
            and positive
        )
        quality[tid] = {
            "length": len(obs),
            "median": med,
            "p90": p90,
            "parallax": max_angle,
            "positive": positive,
            "strong": strong,
            "loop": span >= 20,
        }
        for fi, u, v in obs:
            frame_obs[fi].append((tid, u, v))
    rows = []
    for fi in range(len(P)):
        obs = frame_obs[fi]
        strong = [o for o in obs if quality[o[0]]["strong"]]
        cells = set()
        counts = defaultdict(int)
        for _, u, v in strong:
            cell = (
                min(a.grid_x - 1, max(0, int(u * a.grid_x / w))),
                min(a.grid_y - 1, max(0, int(v * a.grid_y / h))),
            )
            cells.add(cell)
            counts[cell] += 1
        maxfrac = max(counts.values()) / len(strong) if strong else 1.0
        lengths = [quality[o[0]]["length"] for o in obs]
        errs = [quality[o[0]]["median"] for o in obs]
        rows.append(
            {
                "frame": fi,
                "image": names[fi],
                "observed_landmarks": len(obs),
                "strong_landmarks": len(strong),
                "strong_fraction": len(strong) / len(obs) if obs else 0,
                "loop_supported_strong": sum(quality[o[0]]["loop"] for o in strong),
                "median_track_length": float(np.median(lengths)) if lengths else 0,
                "median_landmark_reprojection_px": (
                    float(np.median(errs)) if errs else None
                ),
                "occupied_grid_cells": len(cells),
                "grid_coverage_fraction": len(cells) / (a.grid_x * a.grid_y),
                "largest_cell_fraction": maxfrac,
            }
        )
    with (out / "per_image_landmark_quality.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0])
        wr.writeheader()
        wr.writerows(rows)
    lengths = np.array([q["length"] for q in quality.values()])
    summary = {
        "images": len(rows),
        "tracked_landmarks": len(quality),
        "strong_landmarks": sum(q["strong"] for q in quality.values()),
        "strong_definition": {
            "min_track_length": a.min_track_length,
            "max_median_reprojection_px": a.max_median_reprojection_px,
            "max_p90_reprojection_px": a.max_p90_reprojection_px,
            "min_parallax_deg": a.min_parallax_deg,
            "positive_depth_all_observations": True,
        },
        "track_length": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
            "count_3": int(np.sum(lengths == 3)),
            "count_4": int(np.sum(lengths == 4)),
            "count_5_plus": int(np.sum(lengths >= 5)),
        },
        "per_image": {
            "strong_median": float(np.median([r["strong_landmarks"] for r in rows])),
            "strong_p10": float(
                np.percentile([r["strong_landmarks"] for r in rows], 10)
            ),
            "coverage_median": float(
                np.median([r["grid_coverage_fraction"] for r in rows])
            ),
        },
        "weakest_images": sorted(
            rows, key=lambda r: (r["strong_landmarks"], r["grid_coverage_fraction"])
        )[:20],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    ax[0].plot(x, [r["observed_landmarks"] for r in rows], label="all", lw=1)
    ax[0].plot(x, [r["strong_landmarks"] for r in rows], label="strong", lw=1)
    ax[0].set_ylabel("landmarks")
    ax[0].legend()
    ax[1].plot(x, [r["median_track_length"] for r in rows], lw=1)
    ax[1].set_ylabel("median track length")
    ax[2].plot(
        x,
        [100 * r["grid_coverage_fraction"] for r in rows],
        label="occupied cells",
        lw=1,
    )
    ax[2].plot(
        x,
        [100 * r["largest_cell_fraction"] for r in rows],
        label="largest-cell share",
        lw=1,
    )
    ax[2].set_ylabel("percent")
    ax[2].set_xlabel("image index")
    ax[2].legend()
    for z in ax:
        z.grid(alpha=0.25)
    fig.suptitle("Per-image landmark strength and spatial distribution")
    fig.tight_layout()
    fig.savefig(out / "per_image_landmark_quality.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
