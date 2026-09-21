#!/usr/bin/env python3
"""Build a 3DGS dataset whose sparse/0/points3D.txt is initialized from a merged
RGB LiDAR cloud instead of the SfM landmarks.

The cloud must already be in the same world frame as the reconstruction's
images.txt (true for scripts/kitti/merge_camera_corrected_lidar_rgb.py output).
Points far from every camera and isolated outliers are removed because 3DGS
derives each initial Gaussian scale from its nearest-neighbour distance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
             "uint": "u4", "float": "f4", "double": "f8", "float32": "f4", "float64": "f8",
             "uint8": "u1", "int32": "i4"}


def read_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        fmt = None
        count = None
        fields: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                _, name, n = line.split()
                in_vertex = name == "vertex"
                if in_vertex:
                    count = int(n)
            elif line.startswith("property") and in_vertex:
                _, ptype, pname = line.split()
                fields.append((pname, PLY_TYPES[ptype]))
            elif line == "end_header":
                break
        if count is None:
            raise ValueError(f"{path}: no vertex element")
        if fmt == "binary_little_endian":
            data = np.fromfile(f, dtype=np.dtype([(n, "<" + t) for n, t in fields]), count=count)
        elif fmt == "binary_big_endian":
            data = np.fromfile(f, dtype=np.dtype([(n, ">" + t) for n, t in fields]), count=count)
        elif fmt == "ascii":
            data = np.loadtxt(f, dtype=np.dtype([(n, t) for n, t in fields]), max_rows=count, ndmin=1)
        else:
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    names = data.dtype.names
    if "red" in names:
        rgb = np.column_stack([data["red"], data["green"], data["blue"]])
    elif "r" in names:
        rgb = np.column_stack([data["r"], data["g"], data["b"]])
    else:
        raise ValueError(f"{path}: cloud has no RGB properties")
    if rgb.dtype.kind == "f":
        rgb = np.clip(rgb * 255.0, 0, 255)
    return xyz, rgb.astype(np.uint8)


def read_image_pose_lines(images_txt: Path) -> list[str]:
    """Return the pose header line of every image (COLMAP text alternates pose/observation lines)."""
    lines = [l for l in images_txt.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    headers = lines[0::2]
    if not headers or any(len(h.split()) != 10 for h in headers):
        raise ValueError(f"{images_txt}: unexpected COLMAP images.txt layout")
    return headers


def camera_centers(headers: list[str]) -> np.ndarray:
    centers = []
    for line in headers:
        parts = line.split()
        qw, qx, qy, qz = map(float, parts[1:5])
        t = np.array(parts[5:8], float)
        r_cw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        centers.append(-r_cw.T @ t)
    return np.array(centers)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reconstruction", type=Path, required=True,
                   help="directory containing sparse/0/{cameras,images}.txt")
    p.add_argument("--lidar-ply", type=Path, required=True,
                   help="merged RGB LiDAR cloud in the reconstruction world frame")
    p.add_argument("--images", type=Path, required=True, help="directory with the training PNGs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-camera-distance", type=float, default=60.0,
                   help="drop points farther than this from every camera (m)")
    p.add_argument("--max-knn-distance", type=float, default=0.5,
                   help="drop isolated points whose mean 3-NN distance exceeds this (m)")
    p.add_argument("--max-points", type=int, default=0,
                   help="random subsample to at most this many points (0 = keep all)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    src_sparse = a.reconstruction / "sparse" / "0"
    xyz, rgb = read_ply_xyz_rgb(a.lidar_ply)
    total = len(xyz)
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[finite], rgb[finite]

    headers = read_image_pose_lines(src_sparse / "images.txt")
    centers = camera_centers(headers)
    cam_dist, _ = cKDTree(centers).query(xyz, k=1)
    near = cam_dist <= a.max_camera_distance
    xyz, rgb, cam_dist = xyz[near], rgb[near], cam_dist[near]
    removed_far = int((~near).sum())

    knn, _ = cKDTree(xyz).query(xyz, k=4)
    knn_mean = knn[:, 1:].mean(axis=1)
    dense = knn_mean <= a.max_knn_distance
    xyz, rgb = xyz[dense], rgb[dense]
    removed_isolated = int((~dense).sum())

    if a.max_points and len(xyz) > a.max_points:
        keep = np.random.default_rng(a.seed).choice(len(xyz), a.max_points, replace=False)
        keep.sort()
        xyz, rgb = xyz[keep], rgb[keep]

    out_sparse = a.output / "sparse" / "0"
    out_images = a.output / "images"
    out_sparse.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)
    (out_sparse / "cameras.txt").write_text((src_sparse / "cameras.txt").read_text())
    # Keep only the pose line per image; the SfM 2D-3D links no longer refer to valid IDs.
    kept = []
    names = []
    for line in headers:
        kept.extend((line, ""))
        names.append(line.split()[9])
    (out_sparse / "images.txt").write_text("\n".join(kept) + "\n")

    with (out_sparse / "points3D.txt").open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(xyz)}, source: {a.lidar_ply}\n")
        for i, (pt, col) in enumerate(zip(xyz, rgb), 1):
            f.write(f"{i} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {col[0]} {col[1]} {col[2]} 0\n")

    for name in names:
        src = (a.images / name).resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = out_images / name
        if not dst.exists():
            os.symlink(src, dst)

    final_knn, _ = cKDTree(xyz).query(xyz, k=4)
    final_scale = np.sqrt((final_knn[:, 1:] ** 2).mean(axis=1))
    report = {
        "lidar_ply": str(a.lidar_ply.resolve()),
        "reconstruction": str(a.reconstruction.resolve()),
        "cameras": len(centers),
        "images_linked": len(names),
        "input_points": total,
        "removed_non_finite": int((~finite).sum()),
        "removed_far_from_cameras": removed_far,
        "removed_isolated": removed_isolated,
        "output_points": len(xyz),
        "max_camera_distance_m": a.max_camera_distance,
        "max_knn_distance_m": a.max_knn_distance,
        "initial_gaussian_scale_m": {
            "median": float(np.median(final_scale)),
            "p90": float(np.percentile(final_scale, 90)),
            "max": float(final_scale.max()),
        },
        "sparse_model": str(out_sparse.resolve()),
        "images_directory": str(out_images.resolve()),
    }
    (a.output / "init_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
