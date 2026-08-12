#!/usr/bin/env python3
"""View the landmark CSV (world_x, world_y, world_z per landmark) as a sparse
point cloud.

Reads the CSV written by BaHelper::save_landmark_observations_csv():
    landmark_id,world_x,world_y,world_z,camera_id,image_name,pixel_x,pixel_y

Each landmark_id repeats once per observation with the same world position,
so we dedupe to one point per landmark before plotting/exporting.

Usage:
    python3 scripts/view_landmark_pointcloud.py
    python3 scripts/view_landmark_pointcloud.py --color-by observations
    python3 scripts/view_landmark_pointcloud.py --export-ply ../data/landmarks.ply
"""
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_points(csv_path):
    """Return (landmark_ids, xyz Nx3 array, num_observations per landmark)."""
    positions = {}
    obs_counts = defaultdict(int)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = int(row["landmark_id"])
            positions[lid] = (float(row["world_x"]), float(row["world_y"]), float(row["world_z"]))
            obs_counts[lid] += 1

    landmark_ids = sorted(positions.keys())
    xyz = np.array([positions[lid] for lid in landmark_ids])
    counts = np.array([obs_counts[lid] for lid in landmark_ids])
    return landmark_ids, xyz, counts


def export_ply(path, xyz, colors=None):
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            else:
                f.write(f"{x} {y} {z}\n")
    print(f"Wrote {n} points to {path}")


def counts_to_colors(counts):
    """Map observation counts to a red->green colormap for quick visual QA."""
    norm = (counts - counts.min()) / max(counts.max() - counts.min(), 1)
    cmap = plt.get_cmap("viridis")
    rgba = cmap(norm)
    return (rgba[:, :3] * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="../data/landmark_observations.csv")
    parser.add_argument("--color-by", choices=["none", "observations"], default="observations",
                         help="Color points by number of observations, or plain.")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--export-ply", default=None,
                         help="If set, also write the point cloud to this .ply path.")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip the interactive matplotlib window (useful with --export-ply only).")
    args = parser.parse_args()

    landmark_ids, xyz, counts = load_points(args.csv)
    if xyz.size == 0:
        print(f"No landmarks found in {args.csv}")
        return

    print(f"Loaded {len(landmark_ids)} landmarks from {args.csv}")

    colors = counts_to_colors(counts) if args.color_by == "observations" else None

    if args.export_ply:
        export_ply(args.export_ply, xyz, colors)

    if args.no_show:
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    c = (colors / 255.0) if colors is not None else "steelblue"
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=args.point_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Sparse landmark point cloud ({len(landmark_ids)} points)")

    # Equal aspect ratio so structure isn't visually distorted.
    max_range = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2.0
    mid = (xyz.max(axis=0) + xyz.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()


if __name__ == "__main__":
    main()
