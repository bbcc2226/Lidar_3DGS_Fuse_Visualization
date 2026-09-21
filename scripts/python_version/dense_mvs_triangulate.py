#!/usr/bin/env python3
"""Triangulate a dense cloud from optical flow between nearby camera frames."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def load_poses(p):
    """Load camera poses from a TUM trajectory file.

    Purpose:
        Convert timestamped translations and XYZW quaternions into matrices.
    Inputs:
        p: Path to a whitespace-delimited TUM pose file.
    Outputs:
        List of ``(timestamp, rotation, translation)`` tuples, where rotation
        maps camera coordinates to world coordinates.
    """
    out = []
    for line in open(p):
        z = line.split()
        if len(z) != 8:
            continue
        t = np.array(z[1:4], float)
        x, y, zz, w = map(float, z[4:8])
        R = np.array(
            [
                [1 - 2 * (y * y + zz * zz), 2 * (x * y - zz * w), 2 * (x * zz + y * w)],
                [2 * (x * y + zz * w), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - x * w)],
                [2 * (x * zz - y * w), 2 * (y * zz + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        out.append((float(z[0]), R, t))
    return out


def projection(R, C, K):
    """Build a world-to-image projection matrix.

    Purpose:
        Combine a camera-to-world rotation, camera center, and intrinsics.
    Inputs:
        R: 3x3 camera-to-world rotation matrix.
        C: Three-element camera center in world coordinates.
        K: 3x3 camera intrinsic matrix.
    Outputs:
        A 3x4 projection matrix mapping homogeneous world points to pixels.
    """
    Rc = R.T
    return K @ np.column_stack((Rc, -Rc @ C))


def bilinear(im, x, y):
    """Sample an image or flow channel at floating-point coordinates.

    Purpose:
        Apply OpenCV bilinear remapping with zero-valued out-of-bounds samples.
    Inputs:
        im: Source NumPy image array.
        x: Array of horizontal sample coordinates.
        y: Array of vertical sample coordinates with the same shape as ``x``.
    Outputs:
        Array of interpolated values with the coordinate arrays' shape.
    """
    return cv2.remap(
        im,
        x.astype(np.float32),
        y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def main():
    """Triangulate and voxel-filter a dense image-derived point cloud.

    Purpose:
        Track a pixel grid with bidirectional optical flow, enforce photometric
        and geometric consistency, triangulate matches, and aggregate voxels.
    Inputs:
        Command-line paths for data, poses, intrinsics, and output plus optical
        flow, geometry, support, depth, and voxel filtering parameters.
    Outputs:
        Writes an ASCII PLY cloud and a JSON metrics file; prints metrics and
        exits with status 0 on sufficient density or status 2 otherwise.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--intrinsics", type=Path, default=None)
    p.add_argument("--poses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--grid-step", type=int, default=6)
    p.add_argument("--pair-gaps", type=int, nargs="+", default=[1, 2])
    p.add_argument("--min-parallax-deg", type=float, default=0.35)
    p.add_argument("--max-fb-error", type=float, default=1.0)
    p.add_argument("--max-gray-error", type=float, default=35)
    p.add_argument("--max-reprojection-px", type=float, default=1.5)
    p.add_argument("--max-epipolar-px", type=float, default=1.25)
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--min-pair-support", type=int, default=2)
    p.add_argument(
        "--support-neighborhood",
        type=int,
        default=0,
        help="Count independent image-pair support in neighboring voxels.",
    )
    p.add_argument("--max-depth", type=float, default=30)
    a = p.parse_args()
    K = np.loadtxt(a.intrinsics or a.data / "intrinsics.txt", delimiter=",")
    poses = load_poses(a.poses)
    names = [line.split()[0] for line in open(a.data / "timestamps.txt")][: len(poses)]
    acc = {}
    raw = fb_ok = epi_ok = tri_ok = 0
    pairs = 0
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for i in range(len(poses)):
        im1 = cv2.imread(str(a.data / "undistorted" / names[i]))
        if im1 is None:
            continue
        g1 = cv2.resize(
            cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), None, fx=a.scale, fy=a.scale
        )
        for gap in a.pair_gaps:
            j = i + gap
            if j >= len(poses):
                continue
            im2 = cv2.imread(str(a.data / "undistorted" / names[j]))
            if im2 is None:
                continue
            R1, C1 = poses[i][1:]
            R2, C2 = poses[j][1:]
            baseline = np.linalg.norm(C2 - C1)
            if baseline < 0.02:
                continue
            g2 = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), g1.shape[::-1])
            f12 = dis.calc(g1, g2, None)
            f21 = dis.calc(g2, g1, None)
            yy, xx = np.mgrid[
                a.grid_step // 2 : g1.shape[0] : a.grid_step,
                a.grid_step // 2 : g1.shape[1] : a.grid_step,
            ]
            x2 = xx + f12[yy, xx, 0]
            y2 = yy + f12[yy, xx, 1]
            inside = (
                (x2 >= 1) & (x2 < g2.shape[1] - 2) & (y2 >= 1) & (y2 < g2.shape[0] - 2)
            )
            bx = bilinear(f21[..., 0], x2, y2)
            by = bilinear(f21[..., 1], x2, y2)
            fb = np.hypot(f12[yy, xx, 0] + bx, f12[yy, xx, 1] + by)
            val2 = bilinear(g2, x2, y2)
            photo = np.abs(g1[yy, xx].astype(float) - val2.astype(float))
            valid = inside & (fb < a.max_fb_error) & (photo < a.max_gray_error)
            raw += valid.size
            fb_ok += int(valid.sum())
            u1 = np.column_stack((xx[valid] / a.scale, yy[valid] / a.scale))
            u2 = np.column_stack((x2[valid] / a.scale, y2[valid] / a.scale))
            if len(u1) < 8:
                continue
            P1 = projection(R1, C1, K)
            P2 = projection(R2, C2, K)
            R21 = R2.T @ R1
            t21 = R2.T @ (C1 - C2)
            tx = np.array(
                [[0, -t21[2], t21[1]], [t21[2], 0, -t21[0]], [-t21[1], t21[0], 0]]
            )
            Ki = np.linalg.inv(K)
            F = Ki.T @ tx @ R21 @ Ki
            h1 = np.column_stack((u1, np.ones(len(u1))))
            h2 = np.column_stack((u2, np.ones(len(u2))))
            l2 = (F @ h1.T).T
            l1 = (F.T @ h2.T).T
            numer = np.abs(np.sum(h2 * l2, axis=1))
            epi = np.maximum(
                numer / np.maximum(np.hypot(l2[:, 0], l2[:, 1]), 1e-12),
                numer / np.maximum(np.hypot(l1[:, 0], l1[:, 1]), 1e-12),
            )
            keep = np.isfinite(epi) & (epi < a.max_epipolar_px)
            u1 = u1[keep]
            u2 = u2[keep]
            epi_ok += int(keep.sum())
            if len(u1) < 8:
                continue
            X = cv2.triangulatePoints(P1, P2, u1.T, u2.T).T
            good = np.abs(X[:, 3]) > 1e-9
            X[good, :3] /= X[good, 3, None]
            X = X[:, :3]
            c1 = (X - C1) @ R1
            c2 = (X - C2) @ R2
            good &= (
                (c1[:, 2] > 0.1)
                & (c2[:, 2] > 0.1)
                & (c1[:, 2] < a.max_depth)
                & (c2[:, 2] < a.max_depth)
            )
            ray1 = X - C1
            ray2 = X - C2
            cs = np.sum(ray1 * ray2, axis=1) / (
                np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1) + 1e-12
            )
            angle = np.degrees(np.arccos(np.clip(cs, -1, 1)))
            good &= angle >= a.min_parallax_deg
            q1 = (P1 @ np.column_stack((X, np.ones(len(X)))).T).T
            q2 = (P2 @ np.column_stack((X, np.ones(len(X)))).T).T
            q1 = q1[:, :2] / q1[:, 2, None]
            q2 = q2[:, :2] / q2[:, 2, None]
            err = np.maximum(
                np.linalg.norm(q1 - u1, axis=1), np.linalg.norm(q2 - u2, axis=1)
            )
            good &= np.isfinite(err) & (err < a.max_reprojection_px)
            X = X[good]
            uv = u1[good]
            tri_ok += len(X)
            if not len(X):
                continue
            rgb = im1[
                np.clip(uv[:, 1].astype(int), 0, im1.shape[0] - 1),
                np.clip(uv[:, 0].astype(int), 0, im1.shape[1] - 1),
                ::-1,
            ].astype(float)
            vox = np.floor(X / a.voxel).astype(np.int64)
            uniq, inv = np.unique(vox, axis=0, return_inverse=True)
            sx = np.zeros((len(uniq), 3))
            sc = np.zeros((len(uniq), 3))
            cnt = np.zeros(len(uniq), int)
            np.add.at(sx, inv, X)
            np.add.at(sc, inv, rgb)
            np.add.at(cnt, inv, 1)
            pair_id = (i, j)
            for key, x, c, n in zip(
                map(tuple, uniq), sx / cnt[:, None], sc / cnt[:, None], cnt
            ):
                if key in acc:
                    z = acc[key]
                    z[0] += x
                    z[1] += c
                    z[2].add(pair_id)
                    z[3] += int(n)
                    z[4] += 1
                else:
                    acc[key] = [x.copy(), c.copy(), {pair_id}, int(n), 1]
            pairs += 1
        if (i + 1) % 25 == 0:
            print(
                "images",
                i + 1,
                "pairs",
                pairs,
                "triangulated",
                tri_ok,
                "voxels",
                len(acc),
                flush=True,
            )
    selected = []
    radius = max(0, a.support_neighborhood)
    for key, z in acc.items():
        supporting_pairs = set(z[2])
        if len(supporting_pairs) < a.min_pair_support and radius:
            k = np.asarray(key)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        neighbor = acc.get(tuple(k + (dx, dy, dz)))
                        if neighbor is not None:
                            supporting_pairs.update(neighbor[2])
        if len(supporting_pairs) >= a.min_pair_support:
            selected.append((z, len(supporting_pairs)))
    xyz = np.array([z[0] / z[4] for z, _ in selected])
    rgb = np.clip(np.array([z[1] / z[4] for z, _ in selected]), 0, 255).astype(np.uint8)
    support = np.array([s for _, s in selected])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(xyz))
        f.write(
            "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty ushort pair_support\nend_header\n"
        )
        for x, c, s in zip(xyz, rgb, support):
            f.write("%g %g %g %d %d %d %d\n" % (*x, *c, s))
    m = {
        "pass": bool(len(xyz) >= 100000),
        "images": len(poses),
        "image_pairs": pairs,
        "sampled_correspondences": raw,
        "forward_backward_photo_consistent": fb_ok,
        "explicit_epipolar_consistent": epi_ok,
        "max_epipolar_error_px": a.max_epipolar_px,
        "triangulated_after_geometry_filters": tri_ok,
        "dense_points": len(xyz),
        "voxel_size_m": a.voxel,
        "min_pair_support": a.min_pair_support,
        "support_neighborhood_voxels": radius,
        "median_pair_support": float(np.median(support)) if len(support) else 0,
        "geometry_source": "images only; optimized poses; no LiDAR depth",
    }
    with open(a.output.with_suffix(".json"), "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))
    raise SystemExit(0 if m["pass"] else 2)


if __name__ == "__main__":
    main()
