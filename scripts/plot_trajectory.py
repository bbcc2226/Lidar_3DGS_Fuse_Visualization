#!/usr/bin/env python3
"""Plot and compare camera trajectories saved by the bundle adjustment pipeline.

Each trajectory file is TUM format, one pose per line:
    timestamp x y z qx qy qz qw

Usage:
    python3 scripts/plot_trajectory.py
    python3 scripts/plot_trajectory.py --files ../data/trajectory/initial.txt ../data/trajectory/final.txt
    python3 scripts/plot_trajectory.py --dir ../data/trajectory --all-iters
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np


def load_trajectory(path):
    data = np.loadtxt(path, ndmin=2)
    if data.size == 0:
        raise ValueError(f"No poses found in {path}")
    # columns: timestamp x y z qx qy qz qw
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quats = data[:, 4:8]
    order = np.argsort(timestamps)
    return timestamps[order], positions[order], quats[order]


def quat_to_heading(quats):
    """Camera forward axis (+Z in camera frame) rotated into world frame."""
    qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    heading = np.stack(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        axis=1,
    )
    norms = np.linalg.norm(heading, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return heading / norms


def align_start(positions):
    """Shift a trajectory so it starts at the origin, for easier shape comparison."""
    return positions - positions[0]


def compute_ate(ref_positions, positions):
    """Root-mean-square distance between two trajectories of equal length."""
    if len(ref_positions) != len(positions):
        return None
    diff = positions - ref_positions
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def default_files(traj_dir, all_iters):
    files = []
    initial_path = os.path.join(traj_dir, "initial.txt")
    if os.path.exists(initial_path):
        files.append(initial_path)

    if all_iters:
        iter_files = sorted(
            glob.glob(os.path.join(traj_dir, "iter_*.txt")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
        )
        files.extend(iter_files)

    final_path = os.path.join(traj_dir, "final.txt")
    if os.path.exists(final_path):
        files.append(final_path)

    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="../data/trajectory", help="Directory containing trajectory files.")
    parser.add_argument("--files", nargs="+", default=None, help="Explicit list of trajectory files to plot.")
    parser.add_argument("--all-iters", action="store_true", help="Also plot every per-iteration trajectory in --dir.")
    parser.add_argument("--align-start", action="store_true", help="Shift each trajectory to start at the origin.")
    parser.add_argument("--top-view", action="store_true", help="Plot X-Y top-down view instead of 3D.")
    parser.add_argument("--heading", action="store_true", help="Draw camera heading arrows along each trajectory.")
    parser.add_argument("--heading-stride", type=int, default=10, help="Draw a heading arrow every N poses (default 10).")
    parser.add_argument("--heading-scale", type=float, default=1.0, help="Length scale for heading arrows.")
    parser.add_argument("--fancy", action="store_true", help="Use a cleaner time-colored comparison style for top-view plots.")
    parser.add_argument("--out", default=None, help="If set, save the figure to this path instead of showing it.")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")

    files = args.files if args.files else default_files(args.dir, args.all_iters)
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit(f"No trajectory files found (looked in {args.dir}).")

    trajectories = []
    for f in files:
        timestamps, positions, quats = load_trajectory(f)
        if args.align_start:
            positions = align_start(positions)
        trajectories.append((os.path.splitext(os.path.basename(f))[0], positions, quats))

    fig = plt.figure(figsize=(10, 7), facecolor="white")
    ax = fig.add_subplot(111) if args.top_view else fig.add_subplot(111, projection="3d")
    colors = ["#1769aa", "#e07a35", "#2a9d8f", "#8c5aa8", "#c44536"]

    for index, (name, positions, quats) in enumerate(trajectories):
        color = colors[index % len(colors)]
        if args.fancy and args.top_view and len(trajectories) == 2 and index == 0:
            line, = ax.plot(
                positions[:, 0], positions[:, 1], color="#9aa5b1", linewidth=1.6,
                linestyle="--", alpha=0.7, label=f"{name} (reference)")
        elif args.fancy and args.top_view and len(trajectories) == 2 and index == 1:
            points = positions[:, :2].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            line_collection = LineCollection(
                segments, cmap="viridis", norm=Normalize(0, max(1, len(segments) - 1)),
                linewidth=2.8, alpha=0.95, label=name)
            line_collection.set_array(np.arange(len(segments)))
            ax.add_collection(line_collection)
            line = line_collection
            ax.autoscale_view()
            ax.scatter(
                positions[:, 0], positions[:, 1], c=np.arange(len(positions)),
                cmap="viridis", norm=Normalize(0, max(1, len(positions) - 1)),
                s=5, alpha=0.55, linewidths=0, zorder=3)
        elif args.top_view:
            line, = ax.plot(
                positions[:, 0], positions[:, 1], color=color, linewidth=2.4, label=name)
        else:
            line, = ax.plot(
                positions[:, 0], positions[:, 1], positions[:, 2],
                color=color, linewidth=2.4, label=name)

        start = positions[0]
        finish = positions[-1]
        if args.top_view:
            ax.scatter(*start[:2], color=color, edgecolor="white", s=70, zorder=5)
            ax.scatter(*finish[:2], color=color, edgecolor="white", marker="s", s=70, zorder=5)
        else:
            ax.scatter(*start, color=color, edgecolor="white", s=70)
            ax.scatter(*finish, color=color, edgecolor="white", marker="s", s=70)

        if args.heading:
            headings = quat_to_heading(quats)
            idx = np.arange(0, len(positions), max(1, args.heading_stride))
            if args.top_view:
                ax.quiver(
                    positions[idx, 0], positions[idx, 1],
                    headings[idx, 0], headings[idx, 1],
                    color=line.get_color(), scale=1.0 / args.heading_scale, scale_units="xy", width=0.003,
                )
            else:
                ax.quiver(
                    positions[idx, 0], positions[idx, 1], positions[idx, 2],
                    headings[idx, 0], headings[idx, 1], headings[idx, 2],
                    color=line.get_color(), length=args.heading_scale, normalize=False,
                )

    if len(trajectories) == 2:
        (name_a, pos_a, _), (name_b, pos_b, _) = trajectories
        ate = compute_ate(pos_a, pos_b)
        if ate is not None:
            print(f"RMSE between '{name_a}' and '{name_b}': {ate:.4f}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if not args.top_view:
        ax.set_zlabel("Z")
    ax.set_title("Camera trajectory comparison", fontsize=15, fontweight="bold", pad=14)
    ax.legend(frameon=True, facecolor="white", edgecolor="#d0d7de")
    ax.grid(True, color="#d9e1e8", linewidth=0.8, alpha=0.8)
    if args.top_view:
        ax.set_aspect("equal", adjustable="box")
        if args.fancy and len(trajectories) == 2:
            colorbar = fig.colorbar(line_collection, ax=ax, pad=0.02, fraction=0.04)
            colorbar.set_label("Frame progression")
    else:
        ax.set_box_aspect((1.25, 1.0, 0.65))
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
