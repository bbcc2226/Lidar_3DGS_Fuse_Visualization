#!/usr/bin/env python3
"""Merge local LiDAR scans using corrected camera poses and color from selected rectified images."""

import argparse, csv, json, sys
from pathlib import Path
from functools import lru_cache
import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import minimum_filter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_ply(path):
    with path.open("rb") as f:
        count = None
        fmt = None
        props = []
        vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            if line.startswith("element "):
                vertex = line.split()[1] == "vertex"
                if vertex:
                    count = int(line.split()[2])
            if vertex and line.startswith("property "):
                props.append(line.split()[-1])
            if line == "end_header":
                break
        if fmt != "ascii" or count is None:
            raise ValueError(f"Expected ASCII vertex PLY: {path}")
        points = np.loadtxt(
            f,
            dtype=np.float64,
            max_rows=count,
            usecols=tuple(props.index(x) for x in ["x", "y", "z"]),
            ndmin=2,
        )
    assert len(points) == count
    return points


def write_ply(path, xyz, rgb, support, counts, has_rgb):
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("has_rgb", "u1"),
            ("scan_support", "<u2"),
            ("point_count", "<u4"),
        ]
    )
    out = np.empty(len(xyz), dtype=dtype)
    for i, key in enumerate(["x", "y", "z"]):
        out[key] = xyz[:, i]
    for i, key in enumerate(["red", "green", "blue"]):
        out[key] = rgb[:, i]
    out["has_rgb"] = has_rgb
    out["scan_support"] = support
    out["point_count"] = counts
    header = f"ply\nformat binary_little_endian 1.0\ncomment world_from_lidar derived from corrected camera poses\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar has_rgb\nproperty ushort scan_support\nproperty uint point_count\nend_header\n"
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        out.tofile(f)
    return dtype.itemsize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--maximum-image-dt", type=float, default=0.30)
    ap.add_argument("--maximum-images-per-scan", type=int, default=5)
    args = ap.parse_args()
    assert (
        args.voxel_size > 0
        and args.maximum_image_dt > 0
        and args.maximum_images_per_scan > 0
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "previews").mkdir()
    prov = args.dataset / "provenance"
    meta = json.loads((prov / "preparation_summary.json").read_text())
    k = np.loadtxt(prov / "intrinsics.txt", delimiter=",")
    width, height = meta["image_dimensions"]
    tc_l = np.array(
        json.loads((prov / "camera_lidar_extrinsic.json").read_text())["T_camera_lidar"]
    )
    names = []
    times = []
    for line in (prov / "timestamps.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, t = line.split()
        names.append(name)
        times.append(float(t) + meta["camera_time_offset_seconds"])
    times = np.array(times)
    poses = np.loadtxt(prov / "poses_optimized_tum.txt")
    assert len(poses) == len(times)
    rotations = Rotation.from_quat(poses[:, 4:8])
    r = rotations.as_matrix()
    camera_positions = poses[:, 1:4]
    selected = np.array([(args.dataset / "images" / name).is_file() for name in names])
    image_indices = np.flatnonzero(selected)
    selected_times = times[selected]
    selected_ids = [int(Path(names[i]).stem) for i in image_indices]
    records = [
        json.loads(line)
        for line in (prov / "key_frames.jsonl").read_text().splitlines()
    ]
    records = [
        record
        for record in records
        if selected_times[0] <= record["timestamp"] <= selected_times[-1]
    ]
    assert records
    scan_times = np.array([record["timestamp"] for record in records])
    t0 = times[0]
    rc = Slerp(times - t0, rotations)(scan_times - t0).as_matrix()
    ct = np.column_stack(
        [np.interp(scan_times, times, camera_positions[:, j]) for j in range(3)]
    )
    rl = rc @ tc_l[:3, :3]
    lt = ct + np.einsum("nij,j->ni", rc, tc_l[:3, 3])
    np.savetxt(
        args.output / "corrected_lidar_keyframe_poses_tum.txt",
        np.column_stack((scan_times, lt, Rotation.from_matrix(rl).as_quat())),
        fmt="%.12f",
        header="keyframe_timestamp tx ty tz qx qy qz qw; world_from_lidar",
    )

    @lru_cache(maxsize=16)
    def read_image(i):
        image = cv2.imread(str(args.dataset / "images" / names[i]))
        assert image is not None and image.shape[:2] == (height, width)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def project(world, i):
        camera = (world - camera_positions[i]) @ r[i]
        z = camera[:, 2]
        valid = np.isfinite(camera).all(axis=1) & (z > 0.2)
        idx = np.flatnonzero(valid)
        pixels = camera[idx] @ k.T
        uv = pixels[:, :2] / pixels[:, 2:3]
        keep = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < width - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height - 1)
        )
        idx = idx[keep]
        uv = uv[keep]
        depth = z[idx]
        pixel = np.rint(uv).astype(int)
        flat = pixel[:, 1] * width + pixel[:, 0]
        zbuffer = np.full(width * height, np.inf)
        np.minimum.at(zbuffer, flat, depth)
        nearest = minimum_filter(
            zbuffer.reshape(height, width), size=3, mode="constant", cval=np.inf
        ).ravel()[flat]
        visible = depth <= nearest + (0.15 + 0.01 * depth)
        return idx[visible], uv[visible], depth[visible]

    world_parts = []
    rgb_parts = []
    scan_parts = []
    color_counts = []
    rows = []
    used_images = set()
    preview_ids = set(np.linspace(0, len(records) - 1, 6, dtype=int).tolist())
    for si, record in enumerate(records):
        points = read_ply(Path(record["saved_frame_path"]))
        original_count = len(points)
        valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 0.1)
        points = points[valid]
        world = points @ rl[si].T + lt[si]
        colors = np.zeros((len(world), 3), dtype=np.uint8)
        colored = np.zeros(len(world), dtype=bool)
        order = image_indices[np.argsort(np.abs(times[image_indices] - scan_times[si]))]
        order = order[np.abs(times[order] - scan_times[si]) <= args.maximum_image_dt][
            : args.maximum_images_per_scan
        ]
        per_image = {}
        for i in order:
            i = int(i)
            idx, uv, depth = project(world, i)
            take = ~colored[idx]
            idx = idx[take]
            uv = uv[take]
            if not len(idx):
                continue
            image = read_image(i)
            xy = np.floor(uv).astype(int)
            x = xy[:, 0]
            y = xy[:, 1]
            w = uv - xy
            wx = w[:, 0, None]
            wy = w[:, 1, None]
            color = (
                (1 - wx) * (1 - wy) * image[y, x]
                + wx * (1 - wy) * image[y, x + 1]
                + (1 - wx) * wy * image[y + 1, x]
                + wx * wy * image[y + 1, x + 1]
            )
            colors[idx] = np.clip(np.rint(color), 0, 255).astype(np.uint8)
            colored[idx] = True
            used_images.add(i)
            per_image[names[i]] = len(idx)
        if si in preview_ids and len(order):
            i = int(order[0])
            image = read_image(i)
            idx, uv, depth = project(world, i)
            canvas = image.copy()
            depth_color = cv2.applyColorMap(
                np.clip(255 * (1 - np.minimum(depth, 80) / 80), 0, 255).astype(
                    np.uint8
                ),
                cv2.COLORMAP_TURBO,
            ).reshape(-1, 3)[:, ::-1]
            for pixel, col in zip(np.rint(uv).astype(int), depth_color):
                cv2.circle(canvas, tuple(pixel), 1, tuple(int(x) for x in col), -1)
            pair = np.concatenate([image, canvas], axis=0)
            pair = cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
            cv2.putText(
                pair,
                f'Image {int(Path(names[i]).stem)} / LiDAR keyframe {record["key_frame_id"]}: original + depth projection',
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(
                str(
                    args.output
                    / "previews"
                    / f"image_{Path(names[i]).stem}_projection.jpg"
                ),
                pair,
            )
        row = {
            "key_frame_id": record["key_frame_id"],
            "timestamp": record["timestamp"],
            "cloud_path": record["saved_frame_path"],
            "raw_points": original_count,
            "valid_points": len(world),
            "colored_points": int(colored.sum()),
            "candidate_images": len(order),
            "nearest_image_dt_seconds": (
                float(np.min(np.abs(times[order] - scan_times[si])))
                if len(order)
                else None
            ),
            "image_color_contributions": json.dumps(per_image),
        }
        rows.append(row)
        world_parts.append(world.astype(np.float32))
        rgb_parts.append(colors)
        scan_parts.append(np.full(len(world), si, dtype=np.int32))
        color_counts.append(colored)
        if (si + 1) % 10 == 0 or si + 1 == len(records):
            print(
                f"Scans {si+1}/{len(records)}; latest scan {colored.sum():,}/{len(world):,} points have RGB",
                flush=True,
            )
    xyz = np.concatenate(world_parts)
    rgb = np.concatenate(rgb_parts)
    scan = np.concatenate(scan_parts)
    has = np.concatenate(color_counts)
    del world_parts, rgb_parts, scan_parts, color_counts
    vox = np.floor(xyz / args.voxel_size).astype(np.int64)
    unique, inv = np.unique(vox, axis=0, return_inverse=True)
    n = len(unique)
    del vox, unique
    count = np.bincount(inv, minlength=n)
    color_count = np.bincount(inv, weights=has.astype(int), minlength=n)
    sum_xyz = np.zeros((n, 3))
    sum_color_xyz = np.zeros((n, 3))
    sum_rgb = np.zeros((n, 3))
    for j in range(3):
        sum_xyz[:, j] = np.bincount(inv, weights=xyz[:, j], minlength=n)
        sum_color_xyz[:, j] = np.bincount(inv[has], weights=xyz[has, j], minlength=n)
        sum_rgb[:, j] = np.bincount(inv[has], weights=rgb[has, j], minlength=n)
    support = np.bincount(
        np.unique(inv.astype(np.int64) * len(records) + scan) // len(records),
        minlength=n,
    )
    color_support = np.bincount(
        np.unique(inv[has].astype(np.int64) * len(records) + scan[has]) // len(records),
        minlength=n,
    )
    centers = sum_xyz / count[:, None]
    mask = color_count > 0
    colors = np.full((n, 3), 128, dtype=np.uint8)
    colors[mask] = np.clip(
        np.rint(sum_rgb[mask] / color_count[mask, None]), 0, 255
    ).astype(np.uint8)
    colored_centers = sum_color_xyz[mask] / color_count[mask, None]
    assert np.isfinite(centers).all() and mask.any() and np.all(support >= 1)
    record_size = write_ply(
        args.output / "merged_lidar_full.ply",
        centers,
        colors,
        support,
        count,
        mask.astype(np.uint8),
    )
    write_ply(
        args.output / "merged_lidar_rgb.ply",
        colored_centers,
        colors[mask],
        color_support[mask],
        color_count[mask],
        np.ones(mask.sum(), dtype=np.uint8),
    )
    with (args.output / "scan_manifest.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    (args.output / "used_image_names.txt").write_text(
        "".join(names[i] + "\n" for i in sorted(used_images))
    )
    summary = {
        "camera_pose_source": str((prov / "poses_optimized_tum.txt").resolve()),
        "selected_image_range": [min(selected_ids), max(selected_ids)],
        "available_images": len(image_indices),
        "images_contributing_color": len(used_images),
        "lidar_scans": len(records),
        "keyframe_id_range": [records[0]["key_frame_id"], records[-1]["key_frame_id"]],
        "input_points": sum(row["raw_points"] for row in rows),
        "valid_points": len(xyz),
        "raw_colored_points": int(has.sum()),
        "voxel_size_m": args.voxel_size,
        "merged_full_points": n,
        "merged_rgb_points": int(mask.sum()),
        "full_cloud_uncolored_points": int((~mask).sum()),
        "world_bbox_min": centers.min(axis=0).tolist(),
        "world_bbox_max": centers.max(axis=0).tolist(),
        "maximum_image_dt_seconds": args.maximum_image_dt,
        "maximum_images_per_scan": args.maximum_images_per_scan,
        "visibility_test": "Per-scan 3x3-pixel nearest-depth filter; accept within 0.15 m + 1% of depth. Closest-time valid image supplies each point RGB; voxel RGB averages colored samples only.",
        "geometry_transform": "Interpolate final world_from_camera at LiDAR keyframe time, then compose T_world_lidar = T_world_camera @ T_camera_lidar. Apply once to local scan XYZ.",
        "source_cloud_frame_evidence": "offline_lidar_slam/src/FrontEnd.cpp SaveLIOFrame saves local raw_cloud_; global pose transformation is commented out.",
        "output_vertex_bytes": record_size,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sample = np.linspace(
        0, len(colored_centers) - 1, min(180000, len(colored_centers)), dtype=int
    )
    points = colored_centers[sample]
    col = colors[mask][sample] / 255
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(points[:, 0], points[:, 1], c=col, s=0.25, rasterized=True)
    ax.plot(
        camera_positions[selected, 0],
        camera_positions[selected, 1],
        color="red",
        lw=0.7,
        label="Corrected camera path",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.legend()
    ax.set_title("Merged RGB LiDAR — corrected camera trajectory")
    fig.tight_layout()
    fig.savefig(args.output / "overview_top.png", dpi=180)
    plt.close(fig)
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[::2, 0],
        points[::2, 1],
        points[::2, 2],
        c=col[::2],
        s=0.3,
        depthshade=False,
    )
    limits = np.percentile(points, [1, 99], axis=0)
    ax.set_box_aspect(np.maximum(limits[1] - limits[0], 1))
    ax.view_init(elev=45, azim=-65)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_zlabel("World Z (m)")
    ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(args.output / "overview_3d.png", dpi=180)
    plt.close(fig)
    text = f"""# Merged LiDAR with image RGB and corrected camera poses

Merged **{len(records)} local LiDAR scans** within the timestamps of images **{min(selected_ids)}–{max(selected_ids)}**. Geometry uses the final camera poses from the 60–500 warm-up reconstruction, converted into LiDAR poses at each scan timestamp. RGB comes only from the selected 70–500 images.

- [RGB point cloud](merged_lidar_rgb.ply): **{mask.sum():,} points** with observed image colors.
- [Full point cloud](merged_lidar_full.ply): **{n:,} points**, including **{(~mask).sum():,} uncolored points**. Uncolored points are displayed gray and marked `has_rgb=0`; gray is a display fallback, not measured RGB.
- [Corrected LiDAR keyframe poses](corrected_lidar_keyframe_poses_tum.txt).
- [Scan and image association manifest](scan_manifest.csv) and [summary](summary.json).

Coordinates are meters in the existing SLAM world frame. PLY format is binary little-endian XYZ/RGB, with `has_rgb`, `scan_support` and `point_count`. Merge voxel size is **{args.voxel_size*100:.0f} cm**. The RGB-only cloud averages only colored samples for both XYZ and RGB; the full cloud averages all valid samples for XYZ. `scan_support` counts distinct contributing scans.

Each local scan is transformed exactly once: `T_world_lidar = T_world_camera @ T_camera_lidar`. Camera translation is linearly interpolated and orientation uses SLERP. Scans outside the selected timestamp interval are excluded, so no extrapolation is used. No raw LIO pose is used for geometry. The LiDAR point cloud has not been registered again with ICP or fused into the existing 3DGS model.

For color, use up to {args.maximum_images_per_scan} images within {args.maximum_image_dt:.2f} s, selecting the nearest-time visible observation per point. A per-scan depth filter reduces occlusion artifacts but is approximate on sparse LiDAR. RGB is bilinearly sampled, then averaged within each voxel. This is rigid per-scan alignment; no extra within-scan deskew or moving-object removal is applied. Any existing camera-pose discrepancies can affect map alignment.

![Top view](overview_top.png)

![3D overview](overview_3d.png)

## Projection checks

"""
    for p in sorted((args.output / "previews").glob("*.jpg")):
        text += f"![{p.stem}](previews/{p.name})\n\n"
    (args.output / "README.md").write_text(text)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
