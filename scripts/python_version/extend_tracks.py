#!/usr/bin/env python3
"""Pose-guided completion of existing SIFT landmark tracks."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def poses(path):
    """Load camera centers and rotations from a TUM trajectory.

    Purpose:
        Convert valid pose rows into the representation used by track projection.
    Inputs:
        path: Path to a whitespace-delimited TUM pose file.
    Outputs:
        List of ``(camera_center, camera_to_world_rotation)`` tuples.
    """
    result = []
    for line in open(path):
        z = line.split()
        if len(z) != 8:
            continue
        C = np.asarray(z[1:4], float)
        x, y, qz, w = map(float, z[4:8])
        R = np.array(
            [
                [1 - 2 * (y * y + qz * qz), 2 * (x * y - qz * w), 2 * (x * qz + y * w)],
                [2 * (x * y + qz * w), 1 - 2 * (x * x + qz * qz), 2 * (y * qz - x * w)],
                [2 * (x * qz - y * w), 2 * (y * qz + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        result.append((C, R))
    return result


def ply(path):
    """Load landmark positions and colors from an ASCII PLY file.

    Purpose:
        Recover the sparse point data that will be retriangulated and exported.
    Inputs:
        path: Path to an ASCII PLY containing XYZ and RGB vertex properties.
    Outputs:
        Tuple of an Nx3 point array and an Nx3 color array.
    """
    points = []
    colors = []
    header = True
    for line in open(path):
        if header:
            if line.strip() == "end_header":
                header = False
            continue
        z = line.split()
        if len(z) >= 6:
            points.append(list(map(float, z[:3])))
            colors.append(list(map(int, z[3:6])))
    return np.asarray(points), np.asarray(colors)


def project(X, pose, K):
    """Project world-space landmarks into one camera.

    Purpose:
        Predict pixel locations and camera-space depths for track extension.
    Inputs:
        X: Nx3 array of world-space landmarks.
        pose: ``(camera_center, camera_to_world_rotation)`` tuple.
        K: 3x3 camera intrinsic matrix.
    Outputs:
        Tuple ``(pixels, depths)`` containing an Nx2 array and an N-element array.
    """
    C, R = pose
    q = (X - C) @ R
    uv = np.column_stack(
        (K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2], K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2])
    )
    return uv, q[:, 2]


def epipolar(a, b, ua, ub, K):
    """Measure pose-derived symmetric epipolar error for one correspondence.

    Purpose:
        Check whether two pixels agree with the known relative camera geometry.
    Inputs:
        a: First ``(camera_center, camera_to_world_rotation)`` pose.
        b: Second pose in the same representation.
        ua: Two-element pixel coordinate in the first image.
        ub: Two-element pixel coordinate in the second image.
        K: Shared 3x3 camera intrinsic matrix.
    Outputs:
        Symmetric point-to-epipolar-line distance in pixels.
    """
    Ca, Ra = a
    Cb, Rb = b
    Rba = Rb.T @ Ra
    t = Rb.T @ (Ca - Cb)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Ki = np.linalg.inv(K)
    F = Ki.T @ tx @ Rba @ Ki
    x = np.r_[ua, 1.0]
    y = np.r_[ub, 1.0]
    l = F @ x
    lt = F.T @ y
    n = abs(y @ l)
    return max(
        n / max(np.hypot(l[0], l[1]), 1e-12), n / max(np.hypot(lt[0], lt[1]), 1e-12)
    )


def epipolar_f(F, ua, ub):
    """Measure symmetric epipolar error using an estimated fundamental matrix.

    Purpose:
        Validate a pixel correspondence independently of the pose prior.
    Inputs:
        F: 3x3 fundamental matrix mapping the first image to the second.
        ua: Two-element pixel coordinate in the first image.
        ub: Two-element pixel coordinate in the second image.
    Outputs:
        Symmetric point-to-epipolar-line distance in pixels.
    """
    x = np.r_[ua, 1.0]
    y = np.r_[ub, 1.0]
    l = F @ x
    lt = F.T @ y
    n = abs(y @ l)
    return max(
        n / max(np.hypot(l[0], l[1]), 1e-12), n / max(np.hypot(lt[0], lt[1]), 1e-12)
    )


def visual_fundamental(
    xya, desa, xyb, desb, threshold, min_inliers, min_inlier_ratio, grid_shape=(4, 3)
):
    """Estimate image-pair geometry without using pose priors.

    Purpose:
        Match descriptors, robustly estimate a fundamental matrix, and reject
        weak or spatially degenerate consensus sets.
    Inputs:
        xya: Nx2 keypoint coordinates from the first image.
        desa: NxD descriptors corresponding to ``xya``.
        xyb: Mx2 keypoint coordinates from the second image.
        desb: MxD descriptors corresponding to ``xyb``.
        threshold: Fundamental-matrix inlier threshold in pixels.
        min_inliers: Minimum accepted inlier count.
        min_inlier_ratio: Minimum accepted fraction of mutual matches.
        grid_shape: Spatial grid dimensions used for degeneracy checking.
    Outputs:
        Tuple ``(F, inlier_count)``; ``F`` is ``None`` when estimation or the
        quality guards fail.
    """
    if len(desa) < 8 or len(desb) < 8:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = {}
    for pair in matcher.knnMatch(desa.astype(np.float32), desb.astype(np.float32), k=2):
        if len(pair) > 1 and pair[0].distance < 0.75 * pair[1].distance:
            forward[pair[0].queryIdx] = pair[0].trainIdx
    reverse = {}
    for pair in matcher.knnMatch(desb.astype(np.float32), desa.astype(np.float32), k=2):
        if len(pair) > 1 and pair[0].distance < 0.75 * pair[1].distance:
            reverse[pair[0].queryIdx] = pair[0].trainIdx
    matches = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    if len(matches) < max(8, min_inliers):
        return None, 0
    pa = np.asarray([xya[i] for i, _ in matches], np.float32)
    pb = np.asarray([xyb[j] for _, j in matches], np.float32)
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    F, mask = cv2.findFundamentalMat(pa, pb, method, threshold, 0.999, 10000)
    if F is None or np.asarray(F).shape != (3, 3) or mask is None:
        return None, 0
    mask = mask.ravel().astype(bool)
    count = int(mask.sum())
    if count < min_inliers or count / len(matches) < min_inlier_ratio:
        return None, count

    # A line/patch-sized consensus is not enough to constrain general geometry.
    def occupied(points):
        """Count occupied normalized grid cells for a point set.

        Purpose:
            Detect fundamental-matrix consensus concentrated in a small region.
        Inputs:
            points: Nx2 array of inlier pixel coordinates.
        Outputs:
            Number of distinct cells occupied in the enclosing ``grid_shape``.
        """
        span = np.ptp(points, axis=0)
        norm = (points - points.min(axis=0)) / np.maximum(span, 1.0)
        cells = np.floor(norm * np.asarray(grid_shape)).astype(int)
        cells = np.minimum(cells, np.asarray(grid_shape) - 1)
        return len({tuple(x) for x in cells})

    if occupied(pa[mask]) < 6 or occupied(pb[mask]) < 6:
        return None, count
    return np.asarray(F, float), count


def pixel_grid(xy, cell):
    """Index keypoints by fixed-size pixel cells.

    Purpose:
        Limit descriptor searches to spatially nearby image features.
    Inputs:
        xy: Nx2 array of pixel coordinates.
        cell: Cell width and height in pixels.
    Outputs:
        Mapping from integer ``(cell_x, cell_y)`` keys to keypoint indices.
    """
    grid = defaultdict(list)
    for i, (u, v) in enumerate(xy):
        grid[(int(u // cell), int(v // cell))].append(i)
    return grid


def nearby(grid, uv, cell):
    """Find feature indices in the 3x3 cell neighborhood around a pixel.

    Purpose:
        Retrieve coarse spatial candidates before exact radius filtering.
    Inputs:
        grid: Cell-to-feature mapping produced by :func:`pixel_grid`.
        uv: Two-element query pixel coordinate.
        cell: Cell width and height in pixels.
    Outputs:
        List of candidate feature indices from adjacent cells.
    """
    x, y = int(uv[0] // cell), int(uv[1] // cell)
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out.extend(grid.get((x + dx, y + dy), ()))
    return out


def triangulate_all(observations, ps, K):
    """Triangulate a landmark from all of its track observations.

    Purpose:
        Solve the homogeneous multi-view DLT system with fixed camera poses.
    Inputs:
        observations: Iterable of ``(frame, u, v, is_new)`` observations.
        ps: Sequence of camera-center and rotation tuples indexed by frame.
        K: 3x3 camera intrinsic matrix.
    Outputs:
        Three-element world-space landmark position.
    """
    A = []
    for f, u, v, _ in observations:
        C, R = ps[f]
        Rc = R.T
        P = K @ np.column_stack((Rc, -Rc @ C))
        A.extend((u * P[2] - P[0], v * P[2] - P[1]))
    _, _, vt = np.linalg.svd(np.asarray(A))
    h = vt[-1]
    return h[:3] / h[3]


def main():
    """Extend existing landmark tracks into additional camera frames.

    Purpose:
        Use pose projection, descriptors, epipolar checks, and visual fallback
        geometry to add observations before fixed-pose retriangulation.
    Inputs:
        Command-line data, input/output, intrinsics, cache, target-frame paths,
        and descriptor, geometry, search, depth, and support thresholds.
    Outputs:
        Writes extended tracks, sparse points, copied pose metadata, and track
        extension metrics; prints metrics and returns ``None``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--intrinsics", type=Path)
    ap.add_argument("--cache-source", type=Path)
    ap.add_argument(
        "--target-frames-file", type=Path, help="optional frame indices, one per line"
    )
    ap.add_argument("--search-radius", type=float, default=18.0)
    ap.add_argument("--descriptor-ratio", type=float, default=0.82)
    ap.add_argument("--descriptor-max", type=float, default=240.0)
    ap.add_argument("--epipolar-px", type=float, default=1.25)
    ap.add_argument("--final-reprojection-px", type=float, default=2.0)
    ap.add_argument("--max-depth", type=float, default=30.0)
    ap.add_argument("--target-landmarks-per-image", type=int, default=50)
    ap.add_argument(
        "--weak-search-radius",
        type=float,
        default=60.0,
        help="projection search radius for weak frames",
    )
    ap.add_argument("--visual-geometry-px", type=float, default=1.5)
    ap.add_argument("--visual-min-inliers", type=int, default=20)
    ap.add_argument("--visual-min-inlier-ratio", type=float, default=0.25)
    ap.add_argument(
        "--visual-max-anchor-frames",
        type=int,
        default=8,
        help="best-supported anchor frames tested per weak image",
    )
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    K = np.loadtxt(a.intrinsics or a.data / "intrinsics.txt", delimiter=",")
    ps = poses(a.input / "poses_optimized_tum.txt")
    X, COL = ply(a.input / "landmarks_optimized.ply")
    names = [x.split()[0] for x in open(a.data / "timestamps.txt")][: len(ps)]
    obs = defaultdict(list)
    for r in csv.DictReader(open(a.input / "tracks.csv")):
        obs[int(r["track"])].append(
            [int(r["frame"]), float(r["u"]), float(r["v"]), False]
        )
    cache_root = a.cache_source or a.input / "cache"
    cache_candidates = [
        cache_root / "features_v5_weak_recovery",
        cache_root / "features_v4_core_coverage",
    ]
    cache = next((p for p in cache_candidates if p.is_dir()), None)
    if cache is None:
        raise FileNotFoundError("no supported feature cache under %s" % cache_root)
    reps = [[] for _ in range(len(X))]
    by_frame = defaultdict(list)
    for tid, oo in obs.items():
        for o in oo:
            by_frame[o[0]].append((tid, o[1], o[2]))
    support_before = [
        len({tid for tid, _, _ in by_frame.get(f, ())}) for f in range(len(ps))
    ]
    weak_frames = {
        f for f, n in enumerate(support_before) if n < a.target_landmarks_per_image
    }
    # Recover cached SIFT descriptors belonging to existing observations.
    for f, items in by_frame.items():
        z = np.load(cache / (Path(names[f]).stem + ".npz"))
        xy = z["xy"][: int(z["core_count"])]
        des = z["des"][: len(xy)]
        for tid, u, v in items:
            dd = np.sum((xy - [u, v]) ** 2, axis=1)
            k = int(np.argmin(dd))
            d = math.sqrt(float(dd[k]))
            if d <= 2.0:
                reps[tid].append(des[k].astype(float))
    rep = np.zeros((len(X), 128), np.float32)
    valid_rep = np.zeros(len(X), bool)
    for i, r in enumerate(reps):
        if r:
            v = np.median(np.asarray(r), axis=0)
            rep[i] = v
            valid_rep[i] = True
    existing = {(tid, o[0]) for tid, oo in obs.items() for o in oo}
    added = 0
    conflicts = 0
    target_frames = (
        None
        if a.target_frames_file is None
        else {int(x) for x in a.target_frames_file.read_text().split()}
    )
    # Frame-centric exhaustive landmark completion. Geometry limits candidate
    # pixels before descriptor comparisons; greedy assignment is one-to-one.
    feature_data = {}

    def features(f):
        """Load and memoize core cached features for one frame.

        Purpose:
            Avoid repeatedly reading a frame's keypoints and descriptors.
        Inputs:
            f: Integer frame index into ``names``.
        Outputs:
            Tuple ``(xy, descriptors)`` containing core cached SIFT features.
        """
        if f not in feature_data:
            z = np.load(cache / (Path(names[f]).stem + ".npz"))
            n = int(z["core_count"])
            feature_data[f] = (z["xy"][:n], z["des"][:n].astype(np.float32))
        return feature_data[f]

    visual_models = {}
    visual_attempts = 0
    visual_successes = 0
    visual_accepted = 0
    prior_accepted = 0

    def visual_model(anchor, f):
        """Get or estimate pose-independent geometry for an image pair.

        Purpose:
            Cache fundamental matrices used when the pose prior rejects a match.
        Inputs:
            anchor: Integer index of the established observation frame.
            f: Integer index of the target frame.
        Outputs:
            A 3x3 fundamental matrix, or ``None`` when estimation fails.
    """
        nonlocal visual_attempts, visual_successes
        key = (anchor, f)
        if key not in visual_models:
            visual_attempts += 1
            xa, da = features(anchor)
            xb, db = features(f)
            visual_models[key] = visual_fundamental(
                xa,
                da,
                xb,
                db,
                a.visual_geometry_px,
                a.visual_min_inliers,
                a.visual_min_inlier_ratio,
            )
            visual_successes += int(visual_models[key][0] is not None)
        return visual_models[key][0]

    for f in range(len(ps)):
        if target_frames is not None and f not in target_frames:
            continue
        xy, des = features(f)
        des = des.astype(float, copy=False)
        uv, depth = project(X, ps[f], K)
        inside = (
            valid_rep
            & (depth > 0.1)
            & (depth < a.max_depth)
            & (uv[:, 0] >= 0)
            & (uv[:, 1] >= 0)
        )
        im = cv2.imread(str(a.data / "undistorted" / names[f]), cv2.IMREAD_GRAYSCALE)
        inside &= (uv[:, 0] < im.shape[1]) & (uv[:, 1] < im.shape[0])
        radius = a.weak_search_radius if f in weak_frames else a.search_radius
        tids = np.flatnonzero(inside)
        grid = pixel_grid(xy, radius)
        proposals = []
        visual_pending = []
        for tid in tids:
            cands = nearby(grid, uv[tid], radius)
            if cands:
                cands = [k for k in cands if np.linalg.norm(xy[k] - uv[tid]) <= radius]
            if (int(tid), f) in existing or not cands:
                continue
            ds = np.linalg.norm(des[cands] - rep[tid], axis=1)
            order = np.argsort(ds)
            best = order[0]
            ratio = ds[best] / max(ds[order[1]], 1e-9) if len(order) > 1 else 0.0
            if ds[best] > a.descriptor_max or ratio > a.descriptor_ratio:
                continue
            k = cands[best]
            anchor = min(obs[int(tid)], key=lambda o: abs(o[0] - f))
            prior_ok = (
                epipolar(ps[anchor[0]], ps[f], np.array(anchor[1:3]), xy[k], K)
                <= a.epipolar_px
            )
            item = (
                float(ds[best]),
                float(np.linalg.norm(xy[k] - uv[tid])),
                int(tid),
                int(k),
            )
            if prior_ok:
                proposals.append((*item, False))
            elif f in weak_frames:
                visual_pending.append((*item, anchor))
        # Estimate expensive image-only geometry only for anchor images that
        # support the most landmark proposals in this target frame.
        anchor_counts = defaultdict(int)
        for *_, anchor in visual_pending:
            anchor_counts[anchor[0]] += 1
        selected = {
            fr
            for fr, _ in sorted(
                anchor_counts.items(), key=lambda x: (-x[1], abs(x[0] - f))
            )[: a.visual_max_anchor_frames]
        }
        for score, offset, tid, k, anchor in visual_pending:
            if anchor[0] not in selected:
                continue
            Fv = visual_model(anchor[0], f)
            if (
                Fv is not None
                and epipolar_f(Fv, np.array(anchor[1:3]), xy[k]) <= a.visual_geometry_px
            ):
                proposals.append((score, offset, tid, k, True))
        used_features = set()
        for _, _, tid, k, used_visual in sorted(proposals):
            if k in used_features:
                conflicts += 1
                continue
            used_features.add(k)
            obs[tid].append([f, float(xy[k, 0]), float(xy[k, 1]), True])
            existing.add((tid, f))
            added += 1
            visual_accepted += int(used_visual)
            prior_accepted += int(not used_visual)
        if (f + 1) % 50 == 0:
            print(
                "TRACK_EXTEND images",
                f + 1,
                "added",
                added,
                "assignment_conflicts",
                conflicts,
                flush=True,
            )
    # Robust point-only refinement and new-observation cleanup with poses fixed.
    retained_new = 0
    rejected_new = 0
    for tid, oo in obs.items():
        candidate = triangulate_all(oo, ps, K)
        clean = []
        for o in oo:
            f, u, v, is_new = o
            q = (candidate - ps[f][0]) @ ps[f][1]
            e = (
                np.hypot(
                    K[0, 0] * q[0] / q[2] + K[0, 2] - u,
                    K[1, 1] * q[1] / q[2] + K[1, 2] - v,
                )
                if q[2] > 0.1
                else 1e9
            )
            if not is_new or e <= a.final_reprojection_px:
                clean.append(o)
                retained_new += int(is_new)
            else:
                rejected_new += 1
        if len(clean) >= 3:
            X[tid] = candidate
            obs[tid] = clean
    support_after = [0] * len(ps)
    strong_support_after = [0] * len(ps)
    for oo in obs.values():
        for fr, _, _, _ in oo:
            support_after[fr] += 1
            if len(oo) >= 4:
                strong_support_after[fr] += 1
    with open(a.output / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "frame", "u", "v"])
        for tid in sorted(obs):
            for fr, u, v, _ in sorted(obs[tid]):
                w.writerow([tid, fr, u, v])
    with open(a.output / "sparse_points.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(X))
        f.write(
            "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for x, c in zip(X, COL):
            f.write("%g %g %g %d %d %d\n" % (*x, *c))
    for name in ["poses_lio_prior_tum.txt", "heldout.csv", "metrics.json"]:
        (a.output / name).write_bytes((a.input / name).read_bytes())
    (a.output / "poses_initial_tum.txt").write_bytes(
        (a.input / "poses_optimized_tum.txt").read_bytes()
    )
    if a.intrinsics:
        (a.output / "intrinsics_input.txt").write_bytes(a.intrinsics.read_bytes())
    metrics = {
        "landmarks": len(obs),
        "initial_observations": sum(len(v) for v in by_frame.values()),
        "proposed_observations": added,
        "retained_new_observations": retained_new,
        "rejected_new_observations": rejected_new,
        "assignment_conflicts": conflicts,
        "search_radius_px": a.search_radius,
        "weak_search_radius_px": a.weak_search_radius,
        "feature_cache": str(cache),
        "weak_frames": len(weak_frames),
        "support_before": support_before,
        "support_after": support_after,
        "strong_support_after": strong_support_after,
        "weak_frames_meeting_target_after": sum(
            support_after[f] >= a.target_landmarks_per_image for f in weak_frames
        ),
        "zero_landmark_frames_before": sum(n == 0 for n in support_before),
        "zero_landmark_frames_after": sum(n == 0 for n in support_after),
        "visual_geometry_attempts": visual_attempts,
        "visual_geometry_successes": visual_successes,
        "visual_max_anchor_frames": a.visual_max_anchor_frames,
        "visual_geometry_accepted_observations": visual_accepted,
        "prior_geometry_accepted_observations": prior_accepted,
        "target_frames": len(target_frames) if target_frames is not None else len(ps),
        "heldout_used": False,
    }
    json.dump(metrics, open(a.output / "track_extension_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
