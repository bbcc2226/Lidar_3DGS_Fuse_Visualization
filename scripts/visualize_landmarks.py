#!/usr/bin/env python3
"""Visualize landmark observations to sanity-check that a landmark's
observations across different images really are the same physical point.

Reads the CSV written by BaHelper::save_landmark_observations_csv():
    landmark_id,world_x,world_y,world_z,camera_id,image_name,pixel_x,pixel_y

For each selected landmark, crops a small patch around the observed pixel in
every observing image and shows them side by side.

Usage:
    python3 scripts/visualize_landmarks.py
    python3 scripts/visualize_landmarks.py --landmark-ids 12 57 103
    python3 scripts/visualize_landmarks.py --num-samples 8 --min-observations 4
    python3 scripts/visualize_landmarks.py --out-dir ../output/landmark_checks
"""
import argparse
import csv
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def load_observations(csv_path):
    """Group CSV rows by landmark_id -> list of (camera_id, image_name, px, py)."""
    landmarks = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            landmarks[int(row["landmark_id"])].append(
                (
                    int(row["camera_id"]),
                    row["image_name"],
                    float(row["pixel_x"]),
                    float(row["pixel_y"]),
                )
            )
    return landmarks


def pick_landmarks(landmarks, landmark_ids, num_samples, min_observations, seed):
    if landmark_ids:
        return [lid for lid in landmark_ids if lid in landmarks]

    candidates = [lid for lid, obs in landmarks.items() if len(obs) >= min_observations]
    if num_samples <= 0:
        return candidates

    random.Random(seed).shuffle(candidates)
    return candidates[:num_samples]


def crop_patch(img_dir, image_name, px, py, half_size):
    path = os.path.join(img_dir, image_name)
    img = Image.open(path).convert("RGB")

    left = int(px - half_size)
    top = int(py - half_size)
    right = int(px + half_size)
    bottom = int(py + half_size)

    patch = img.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(patch)
    # Mark the exact observed pixel (center of the crop, offset by the clamp above).
    cx = px - left
    cy = py - top
    r = 4
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="red", width=2)
    return patch


def visualize_landmark(landmark_id, observations, img_dir, half_size, out_dir):
    observations = sorted(observations, key=lambda o: o[0])  # sort by camera_id
    n = len(observations)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]

    for ax, (camera_id, image_name, px, py) in zip(axes, observations):
        patch = crop_patch(img_dir, image_name, px, py, half_size)
        ax.imshow(patch)
        ax.set_title(f"cam {camera_id}\n{image_name}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Landmark {landmark_id} ({n} observations)")
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"landmark_{landmark_id}.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="../data/landmark_observations.csv")
    parser.add_argument("--img-dir", default="../data/undistorted")
    parser.add_argument("--landmark-ids", type=int, nargs="+", default=None,
                         help="Specific landmark IDs to visualize (overrides sampling).")
    parser.add_argument("--num-samples", type=int, default=6,
                         help="Number of landmarks to randomly sample if --landmark-ids is not given. "
                              "Use <= 0 to save all landmarks matching --min-observations.")
    parser.add_argument("--min-observations", type=int, default=2,
                         help="Only sample landmarks with at least this many observations.")
    parser.add_argument("--patch-half-size", type=float, default=40.0,
                         help="Half-width/height (pixels) of the crop shown around each observation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None,
                         help="If set, save one PNG per landmark here instead of showing interactively.")
    args = parser.parse_args()

    landmarks = load_observations(args.csv)
    if not landmarks:
        print(f"No observations found in {args.csv}")
        return

    selected = pick_landmarks(
        landmarks, args.landmark_ids, args.num_samples, args.min_observations, args.seed)

    if not selected:
        print("No landmarks matched the given selection criteria.")
        return

    print(f"Visualizing {len(selected)} landmark(s): {selected}")
    for landmark_id in selected:
        visualize_landmark(
            landmark_id, landmarks[landmark_id], args.img_dir, args.patch_half_size, args.out_dir)


if __name__ == "__main__":
    main()
