#!/usr/bin/env python3
"""Build and triangulate multi-view feature tracks from initialized poses."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_poses(path):
    """Read timestamp-indexed translations and quaternions from TUM poses.

    Purpose:
        Normalize pose quaternions and support timestamp-based frame association.
    Inputs:
        path: Path to a whitespace-delimited TUM pose file.
    Outputs:
        Dictionary mapping timestamps rounded to six decimals to
        ``(translation, normalized_wxyz_quaternion)`` tuples.
    """
    poses = {}
    with open(path) as f:
        for line in f:
            z = line.split()
            if len(z) != 8:
                continue
            ts = float(z[0])
            t = np.array(z[1:4], float)
            x, y, zz, w = map(float, z[4:8])
            q = np.array([w, x, y, zz])
            q /= np.linalg.norm(q)
            poses[round(ts, 6)] = (t, q)
    return poses


def qrot(q):
    """Convert a WXYZ quaternion into a rotation matrix.

    Purpose:
        Construct the camera-to-world rotation used by sparse geometry.
    Inputs:
        q: Four normalized quaternion components ordered as ``(w, x, y, z)``.
    Outputs:
        A 3x3 NumPy rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_frames(data, pose_file, limit):
    """Associate available images with timestamped initialized poses.

    Purpose:
        Load the image and pose state required for feature extraction and matching.
    Inputs:
        data: Dataset directory containing timestamps and undistorted images.
        pose_file: Path to the TUM trajectory used for initialization.
        limit: Maximum frames to load; non-positive values mean no limit.
    Outputs:
        List of frame dictionaries containing name, timestamp, translation,
        quaternion, rotation, and BGR image data.
    """
    poses = read_poses(pose_file)
    frames = []
    for line in open(data / "timestamps.txt"):
        name, ts = line.split()
        ts = float(ts)
        key = round(ts, 6)
        p = data / "undistorted" / name
        if key not in poses or not p.exists():
            continue
        t, q = poses[key]
        im = cv2.imread(str(p))
        frames.append(dict(name=name, ts=ts, t=t, q=q, R=qrot(q), im=im))
        if limit > 0 and len(frames) >= limit:
            break
    return frames


def skew(t):
    """Construct the skew-symmetric cross-product matrix of a vector.

    Purpose:
        Represent translation in essential/fundamental matrix calculations.
    Inputs:
        t: Three-element vector.
    Outputs:
        A 3x3 matrix such that ``skew(t) @ x`` equals ``t cross x``.
    """
    x, y, z = t
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


def fundamental(a, b, K):
    """Compute a fundamental matrix from two initialized camera poses.

    Purpose:
        Supply pose-prior epipolar geometry for descriptor match filtering.
    Inputs:
        a: First frame dictionary with rotation and camera center.
        b: Second frame dictionary in the same format.
        K: Shared 3x3 camera intrinsic matrix.
    Outputs:
        A 3x3 fundamental matrix mapping pixels in ``a`` to lines in ``b``.
    """
    R1, t1 = a["R"], a["t"]
    R2, t2 = b["R"], b["t"]
    R = R2.T @ R1
    t = R2.T @ (t1 - t2)
    Ki = np.linalg.inv(K)
    return Ki.T @ skew(t) @ R @ Ki


def sampson(p, q, F):
    """Compute the Sampson epipolar distance for one pixel match.

    Purpose:
        Approximate geometric reprojection error for match acceptance and QA.
    Inputs:
        p: Two-element pixel coordinate in the first image.
        q: Two-element pixel coordinate in the second image.
        F: 3x3 fundamental matrix.
    Outputs:
        Non-negative Sampson distance in pixel units.
    """
    x = np.array([p[0], p[1], 1.0])
    y = np.array([q[0], q[1], 1.0])
    Fx = F @ x
    Fty = F.T @ y
    n = float(y @ Fx)
    return abs(n) / math.sqrt(
        max(1e-12, Fx[0] ** 2 + Fx[1] ** 2 + Fty[0] ** 2 + Fty[1] ** 2)
    )


def visual_fundamental(points_a, points_b, threshold, min_inliers=20, min_ratio=0.25):
    """Estimate robust image-only geometry for weak-frame recovery.

    Purpose:
        Fit a fundamental matrix and reject insufficient or spatially
        concentrated inlier sets without relying on LIO poses.
    Inputs:
        points_a: Nx2 matched pixel coordinates in the first image.
        points_b: Nx2 corresponding pixels in the second image.
        threshold: Robust estimator's inlier threshold in pixels.
        min_inliers: Minimum accepted inlier count.
        min_ratio: Minimum accepted inlier fraction.
    Outputs:
        Tuple ``(F, mask)`` containing the 3x3 matrix and boolean inlier mask,
        or ``(None, None)`` when estimation or quality checks fail.
    """
    if len(points_a) < max(8, min_inliers):
        return None, None
    pa = np.asarray(points_a, np.float32)
    pb = np.asarray(points_b, np.float32)
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    F, mask = cv2.findFundamentalMat(pa, pb, method, threshold, 0.999, 10000)
    if F is None or np.asarray(F).shape != (3, 3) or mask is None:
        return None, None
    mask = mask.ravel().astype(bool)
    count = int(mask.sum())
    if count < min_inliers or count / len(pa) < min_ratio:
        return None, None

    def cells(p):
        """Count occupied cells in a normalized 4x3 image grid.

        Purpose:
            Reject image-only geometry supported by a line or small patch.
        Inputs:
            p: Nx2 array of inlier pixel coordinates.
        Outputs:
            Number of distinct normalized grid cells occupied.
        """
        span = np.ptp(p, axis=0)
        z = (p - p.min(axis=0)) / np.maximum(span, 1.0)
        ij = np.minimum(np.floor(z * [4, 3]).astype(int), [3, 2])
        return len({tuple(x) for x in ij})

    if cells(pa[mask]) < 6 or cells(pb[mask]) < 6:
        return None, None
    return np.asarray(F, float), mask


class DSU:
    """Disjoint-set structure that prevents duplicate-frame track observations.

    Purpose:
        Merge pairwise feature matches into multi-view tracks while maintaining
        at most one feature from each camera frame per connected component.
    Inputs:
        Constructed with the total number of detected feature nodes.
    Outputs:
        Instances expose representative lookup and conflict-aware union methods.
    """

    def __init__(self, n):
        """Initialize singleton feature components.

        Purpose:
            Allocate parent pointers and empty per-component frame membership.
        Inputs:
            n: Number of feature nodes.
        Outputs:
            Returns ``None`` after initializing the instance in place.
        """
        self.p = list(range(n))
        self.frames = {}

    def find(self, x):
        """Find a feature component's representative with path compression.

        Purpose:
            Resolve track membership efficiently across repeated match merges.
        Inputs:
            x: Integer feature-node index.
        Outputs:
            Integer representative index for the node's component.
        """
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b, frame_a, frame_b):
        """Merge two feature components unless their frame sets conflict.

        Purpose:
            Build tracks while preventing two observations from the same frame.
        Inputs:
            a: First integer feature-node index.
            b: Second integer feature-node index.
            frame_a: Frame index associated with ``a``.
            frame_b: Frame index associated with ``b``.
        Outputs:
            ``True`` if already joined or successfully merged; ``False`` when
            the components contain overlapping camera frames.
        """
        a, b = self.find(a), self.find(b)
        if a == b:
            return True
        fa = self.frames.get(a, {frame_a})
        fb = self.frames.get(b, {frame_b})
        if fa.intersection(fb):
            return False
        if len(fa) < len(fb):
            a, b, fa, fb = b, a, fb, fa
        self.p[b] = a
        self.frames[a] = fa | fb
        self.frames.pop(b, None)
        return True


def projection(f, K):
    """Build a world-to-image projection matrix for one frame.

    Purpose:
        Combine the initialized camera pose with shared intrinsics for DLT.
    Inputs:
        f: Frame dictionary containing camera-to-world rotation and center.
        K: 3x3 camera intrinsic matrix.
    Outputs:
        A 3x4 homogeneous projection matrix.
    """
    R = f["R"].T
    return K @ np.column_stack((R, -R @ f["t"]))


def project(f, K, x):
    """Project one world-space point into a frame.

    Purpose:
        Evaluate positive depth and pixel reprojection during track filtering.
    Inputs:
        f: Frame dictionary containing camera pose.
        K: 3x3 camera intrinsic matrix.
        x: Three-element world-space point.
    Outputs:
        Two-element pixel coordinate, or ``None`` for non-positive depth.
    """
    c = f["R"].T @ (x - f["t"])
    if c[2] <= 1e-6:
        return None
    p = K @ c
    return p[:2] / p[2]


def find_loop_pairs(
    frames, max_gap, min_separation, max_distance, max_per_frame, max_view_angle
):
    """Select spatially and directionally compatible nonlocal image pairs.

    Purpose:
        Add bounded loop-closure candidates beyond the normal temporal window.
    Inputs:
        frames: Sequence of frame dictionaries with poses.
        max_gap: Local-pair gap, retained for the caller's pairing contract.
        min_separation: Minimum frame-index separation for loop candidates.
        max_distance: Maximum camera-center distance in meters.
        max_per_frame: Maximum loop candidates retained for each source frame.
        max_view_angle: Maximum optical-axis difference in degrees.
    Outputs:
        Sorted list of unique ``(first_frame, second_frame)`` index pairs.
    """
    pairs = set()
    for i, a in enumerate(frames):
        candidates = []
        forward_a = a["R"][:, 2]
        for j in range(i + min_separation, len(frames)):
            b = frames[j]
            distance = float(np.linalg.norm(a["t"] - b["t"]))
            if distance > max_distance:
                continue
            angle = math.degrees(
                math.acos(np.clip(np.dot(forward_a, b["R"][:, 2]), -1, 1))
            )
            if angle <= max_view_angle:
                candidates.append((distance, j))
        for _, j in sorted(candidates)[:max_per_frame]:
            pairs.add((i, j))
    return sorted(pairs)


def frame_support(dsu, offsets, frames, grid_x=8, grid_y=6, min_track_length=4):
    """Measure conservative pre-triangulation track support by frame.

    Purpose:
        Identify images needing rescue matching based on long-track count and
        spatial grid coverage.
    Inputs:
        dsu: Feature-track disjoint-set structure.
        offsets: Per-frame starting node offsets with a final total sentinel.
        frames: Frame dictionaries containing images and keypoints.
        grid_x: Number of horizontal support cells.
        grid_y: Number of vertical support cells.
        min_track_length: Minimum component size counted as support.
    Outputs:
        List of ``(supported_track_count, occupied_cell_count)`` per frame.
    """
    component_sizes = {}
    for n in range(offsets[-1]):
        root = dsu.find(n)
        component_sizes[root] = component_sizes.get(root, 0) + 1
    support = []
    for fi, f in enumerate(frames):
        cells = set()
        count = 0
        h, w = f["im"].shape[:2]
        for k, kp in enumerate(f["kp"]):
            if component_sizes.get(dsu.find(offsets[fi] + k), 0) < min_track_length:
                continue
            count += 1
            u, v = kp.pt
            cells.add(
                (
                    min(grid_x - 1, int(u * grid_x / w)),
                    min(grid_y - 1, int(v * grid_y / h)),
                )
            )
        support.append((count, len(cells)))
    return support


def weak_recovery_pairs(
    frames,
    weak,
    normal_gap,
    recovery_gap,
    spatial_distance,
    spatial_per_frame,
    max_view_angle,
):
    """Choose expanded temporal and spatial pairs for weak frames.

    Purpose:
        Give under-supported images additional matching opportunities beyond the
        normal local window while respecting distance and view direction.
    Inputs:
        frames: Sequence of frame dictionaries with poses.
        weak: Iterable of weak frame indices.
        normal_gap: Existing local temporal matching radius.
        recovery_gap: Expanded temporal recovery radius.
        spatial_distance: Maximum camera-center distance for spatial pairs.
        spatial_per_frame: Maximum spatial candidates per weak frame.
        max_view_angle: Maximum optical-axis difference in degrees.
    Outputs:
        Sorted list of unique recovery frame-index pairs.
    """
    pairs = set()
    for i in weak:
        for j in range(
            max(0, i - recovery_gap), min(len(frames), i + recovery_gap + 1)
        ):
            if i != j and abs(i - j) > normal_gap:
                pairs.add(tuple(sorted((i, j))))
        candidates = []
        forward = frames[i]["R"][:, 2]
        for j, b in enumerate(frames):
            if i == j or abs(i - j) <= normal_gap:
                continue
            distance = float(np.linalg.norm(frames[i]["t"] - b["t"]))
            if distance > spatial_distance:
                continue
            angle = math.degrees(
                math.acos(np.clip(np.dot(forward, b["R"][:, 2]), -1, 1))
            )
            if angle <= max_view_angle:
                candidates.append((distance, j))
        for _, j in sorted(candidates)[:spatial_per_frame]:
            pairs.add(tuple(sorted((i, j))))
    return sorted(pairs)


def triangulate(
    frames,
    K,
    max_gap,
    epi_px,
    min_parallax,
    reproj_px,
    loop_min_separation,
    loop_max_distance,
    loop_max_per_frame,
    loop_max_view_angle,
    cache_dir,
    three_view_min_parallax,
    target_landmarks=50,
    target_grid_cells=16,
    recovery_gap=20,
    recovery_spatial_distance=1.0,
    recovery_spatial_per_frame=8,
    grid_x=8,
    grid_y=6,
    grid_min_sift=20,
    grid_extra_features=40,
):
    """Extract, match, group, triangulate, and validate sparse feature tracks.

    Purpose:
        Build core and grid-balanced features, match local/loop/recovery pairs,
        form conflict-free tracks, and triangulate BA-quality landmarks.
    Inputs:
        frames: Mutable frame dictionaries; cached keypoints/descriptors are added.
        K: 3x3 camera intrinsic matrix.
        max_gap: Maximum local temporal pair gap.
        epi_px: Maximum accepted epipolar error in pixels.
        min_parallax: Minimum general-track triangulation angle in degrees.
        reproj_px: Median reprojection threshold in pixels.
        loop_min_separation: Minimum loop frame-index separation.
        loop_max_distance: Maximum loop camera-center distance in meters.
        loop_max_per_frame: Maximum loop candidates per frame.
        loop_max_view_angle: Maximum loop optical-axis angle in degrees.
        cache_dir: Directory for feature and pair caches.
        three_view_min_parallax: Minimum parallax for three-view core tracks.
        target_landmarks: Desired supported landmarks per image.
        target_grid_cells: Desired occupied support cells per image.
        recovery_gap: Expanded weak-frame temporal radius.
        recovery_spatial_distance: Weak-frame spatial radius in meters.
        recovery_spatial_per_frame: Maximum spatial recovery pairs per frame.
        grid_x: Horizontal feature-coverage cell count.
        grid_y: Vertical feature-coverage cell count.
        grid_min_sift: Core SIFT count that suppresses extra cell features.
        grid_extra_features: Maximum supplemental corners detected per weak cell.
    Outputs:
        Tuple ``(points, tracks, coverage_points, heldout_matches, stats)`` where
        point entries contain XYZ, RGB, and reprojection error and ``stats`` is
        a metrics dictionary.
    """

    sift = cv2.SIFT_create(nfeatures=6000)
    feature_cache = cache_dir / "features_v5_weak_recovery"
    feature_cache.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    for f in frames:
        cache = feature_cache / (Path(f["name"]).stem + ".npz")
        if cache.exists():
            z = np.load(cache)
            xy = z["xy"]
            des = z["des"]
            f["core_count"] = int(z["core_count"])
            f["kp"] = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in xy]
            f["des"] = des if len(des) else None
        else:
            g = cv2.cvtColor(f["im"], cv2.COLOR_BGR2GRAY)
            f["kp"], f["des"] = sift.detectAndCompute(g, None)
            f["core_count"] = len(f["kp"])
            f["core_count"] = len(f["kp"])
            mask = np.full(g.shape, 255, np.uint8)
            for kp in f["kp"]:
                cv2.circle(mask, tuple(np.rint(kp.pt).astype(int)), 5, 0, -1)
            extra = []
            gy, gx = grid_y, grid_x
            for yy in range(gy):
                for xx in range(gx):
                    y0, y1 = yy * g.shape[0] // gy, (yy + 1) * g.shape[0] // gy
                    x0, x1 = xx * g.shape[1] // gx, (xx + 1) * g.shape[1] // gx
                    core_here = sum(
                        x0 <= kp.pt[0] < x1 and y0 <= kp.pt[1] < y1 for kp in f["kp"]
                    )
                    if core_here >= grid_min_sift:
                        continue
                    corners = cv2.goodFeaturesToTrack(
                        g[y0:y1, x0:x1],
                        grid_extra_features,
                        0.003,
                        6,
                        mask=mask[y0:y1, x0:x1],
                        blockSize=5,
                    )
                    if corners is None:
                        continue
                    for c in corners[:, 0]:
                        extra.append(
                            cv2.KeyPoint(float(c[0] + x0), float(c[1] + y0), 5.0)
                        )
            if extra:
                kp2, des2 = sift.compute(g, extra)
                if des2 is not None:
                    f["kp"] = list(f["kp"]) + list(kp2)
                    f["des"] = des2 if f["des"] is None else np.vstack((f["des"], des2))
            xy = np.array([k.pt for k in f["kp"]], np.float32)
            des = f["des"] if f["des"] is not None else np.empty((0, 128), np.float32)
            np.savez(cache, xy=xy, des=des, core_count=np.int32(f["core_count"]))
        offsets.append(offsets[-1] + len(f["kp"]))
    dsu = DSU(offsets[-1])
    held = []
    accepted = 0
    loop_matches = 0
    conflict_rejected = 0
    pruned_observations = 0
    quality_rejected = 0
    visual_pairs_attempted = 0
    visual_pairs_accepted = 0
    visual_matches_accepted = 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pair_cache = cache_dir / "pairs_v5_weak_recovery"
    pair_cache.mkdir(parents=True, exist_ok=True)
    local_pairs = [
        (i, j)
        for i in range(len(frames))
        for j in range(i + 1, min(len(frames), i + max_gap + 1))
    ]
    loop_pairs = find_loop_pairs(
        frames,
        max_gap,
        loop_min_separation,
        loop_max_distance,
        loop_max_per_frame,
        loop_max_view_angle,
    )
    loop_set = set(loop_pairs)
    processed_pairs = set()

    def match_pair(i, j, recovery=False):
        """Match one image pair and merge accepted features into tracks.

        Purpose:
            Reuse pair caches, apply pose or visual epipolar geometry, reserve
            held-out matches, and reject duplicate-frame track merges.
        Inputs:
            i: First frame index.
            j: Second frame index.
            recovery: Whether to require mutual matching and estimate visual geometry.
        Outputs:
            Returns ``None``; mutates the disjoint set, held-out set, processed
            pairs, caches, and match counters captured from ``triangulate``.
        """
        nonlocal accepted, loop_matches, conflict_rejected, visual_pairs_attempted, visual_pairs_accepted, visual_matches_accepted
        if (i, j) in processed_pairs:
            return
        processed_pairs.add((i, j))
        if frames[i]["des"] is None or frames[j]["des"] is None:
            return
        cache = pair_cache / ("%06d_%06d.npz" % (i, j))
        if cache.exists():
            z = np.load(cache)
            qidx = z["qidx"]
            tidx = z["tidx"]
            mutual = z["mutual"]
        else:
            reverse = {}
            if recovery or (i, j) in loop_set:
                for rp in bf.knnMatch(frames[j]["des"], frames[i]["des"], k=2):
                    if len(rp) > 1 and rp[0].distance < 0.75 * rp[1].distance:
                        reverse[rp[0].queryIdx] = rp[0].trainIdx
            qidx = []
            tidx = []
            mutual = []
            for pair in bf.knnMatch(frames[i]["des"], frames[j]["des"], k=2):
                if len(pair) < 2 or pair[0].distance >= 0.75 * pair[1].distance:
                    continue
                m = pair[0]
                qidx.append(m.queryIdx)
                tidx.append(m.trainIdx)
                mutual.append(
                    reverse.get(m.trainIdx) == m.queryIdx
                    if recovery or (i, j) in loop_set
                    else True
                )
            np.savez(
                cache,
                qidx=np.asarray(qidx, np.int32),
                tidx=np.asarray(tidx, np.int32),
                mutual=np.asarray(mutual, bool),
            )
            qidx = np.asarray(qidx, np.int32)
            tidx = np.asarray(tidx, np.int32)
            mutual = np.asarray(mutual, bool)
        F = fundamental(frames[i], frames[j], K)
        # Recovery pairs are descriptor-first.  If their mutual matches form a
        # strong, distributed image-only consensus, that geometry can rescue
        # true matches rejected by an inaccurate LIO relative pose.
        Fv = None
        if recovery:
            eligible = np.flatnonzero(mutual)
            pa = [frames[i]["kp"][int(qidx[k])].pt for k in eligible]
            pb = [frames[j]["kp"][int(tidx[k])].pt for k in eligible]
            visual_pairs_attempted += 1
            Fv, _ = visual_fundamental(pa, pb, epi_px)
            visual_pairs_accepted += int(Fv is not None)
        for qi, ti, mu in zip(qidx, tidx, mutual):
            if (recovery or (i, j) in loop_set) and not mu:
                continue
            p = frames[i]["kp"][qi].pt
            q = frames[j]["kp"][ti].pt
            e = sampson(p, q, F)
            prior_ok = e <= epi_px
            visual_ok = Fv is not None and sampson(p, q, Fv) <= epi_px
            if not prior_ok and not visual_ok:
                continue
            if not recovery and (i * 73856093 + j * 19349663 + int(qi)) % 10 == 0:
                if qi < frames[i]["core_count"] and ti < frames[j]["core_count"]:
                    held.append((i, j, p, q))
                continue
            if not dsu.union(offsets[i] + int(qi), offsets[j] + int(ti), i, j):
                conflict_rejected += 1
                continue
            accepted += 1
            visual_matches_accepted += int(visual_ok and not prior_ok)
            if (i, j) in loop_set:
                loop_matches += 1

    for i, j in local_pairs + loop_pairs:
        match_pair(i, j)
    initial_support = frame_support(dsu, offsets, frames, grid_x, grid_y)
    weak = [
        i
        for i, (count, cells) in enumerate(initial_support)
        if count < target_landmarks or cells < target_grid_cells
    ]
    recovery_pairs = weak_recovery_pairs(
        frames,
        weak,
        max_gap,
        recovery_gap,
        recovery_spatial_distance,
        recovery_spatial_per_frame,
        loop_max_view_angle,
    )
    before_recovery = accepted
    for i, j in recovery_pairs:
        match_pair(i, j, True)
    recovered_support = frame_support(dsu, offsets, frames, grid_x, grid_y)
    # Include the DSU root feature itself. The former parent-pointer test
    # accidentally dropped exactly one observation from every track, turning
    # many true four-view tracks into apparent three-view tracks.
    root_sizes = {}
    for n in range(offsets[-1]):
        r = dsu.find(n)
        root_sizes[r] = root_sizes.get(r, 0) + 1
    groups = {}
    for i, f in enumerate(frames):
        for k, kp in enumerate(f["kp"]):
            n = offsets[i] + k
            r = dsu.find(n)
            if root_sizes[r] > 1:
                groups.setdefault(r, []).append((i, k, np.array(kp.pt)))
    points = []
    tracks = []
    coverage_points = []
    coverage_tracks = []
    for obs in groups.values():
        by_frame = {}
        for x in obs:
            by_frame.setdefault(x[0], x)
        obs = sorted(by_frame.values())
        if len(obs) < 3:
            continue
        a, b = obs[0], obs[-1]
        ca, cb = frames[a[0]]["t"], frames[b[0]]["t"]
        ra = frames[a[0]]["R"] @ np.linalg.inv(K) @ np.r_[a[2], 1.0]
        rb = frames[b[0]]["R"] @ np.linalg.inv(K) @ np.r_[b[2], 1.0]
        angle = math.degrees(
            math.acos(
                np.clip(
                    abs(np.dot(ra / np.linalg.norm(ra), rb / np.linalg.norm(rb))), -1, 1
                )
            )
        )
        if angle < min_parallax:
            continue
        A = []
        for fi, ki, uv in obs:
            P = projection(frames[fi], K)
            u, v = uv
            A.extend((u * P[2] - P[0], v * P[2] - P[1]))
        _, _, vh = np.linalg.svd(np.asarray(A), full_matrices=False)
        Xh = vh[-1]
        if abs(Xh[3]) < 1e-9:
            continue
        X = Xh[:3] / Xh[3]
        errors = []
        for fi, ki, uv in obs:
            pp = project(frames[fi], K, X)
            errors.append(np.inf if pp is None else float(np.linalg.norm(pp - uv)))
        keep = [
            k for k, e in enumerate(errors) if np.isfinite(e) and e <= 2 * reproj_px
        ]
        pruned_observations += len(obs) - len(keep)
        obs = [obs[k] for k in keep]
        if len(obs) < 3:
            continue
        if len(keep) < len(errors):
            A = []
            for fi, ki, uv in obs:
                P = projection(frames[fi], K)
                u, v = uv
                A.extend((u * P[2] - P[0], v * P[2] - P[1]))
            _, _, vh = np.linalg.svd(np.asarray(A), full_matrices=False)
            Xh = vh[-1]
            if abs(Xh[3]) < 1e-9:
                continue
            X = Xh[:3] / Xh[3]
        errors = []
        for fi, ki, uv in obs:
            pp = project(frames[fi], K, X)
            errors.append(np.inf if pp is None else float(np.linalg.norm(pp - uv)))
        if (
            not np.all(np.isfinite(errors))
            or np.median(errors) > reproj_px
            or np.percentile(errors, 90) > 2 * reproj_px
        ):
            continue
        is_core = all(ki < frames[fi]["core_count"] for fi, ki, uv in obs)
        if is_core and len(obs) == 3:
            rays = [X - frames[fi]["t"] for fi, ki, uv in obs]
            rays = [r / (np.linalg.norm(r) + 1e-12) for r in rays]
            max_angle = max(
                math.degrees(math.acos(np.clip(np.dot(rays[a], rays[b]), -1, 1)))
                for a in range(3)
                for b in range(a + 1, 3)
            )
            if max_angle < three_view_min_parallax:
                quality_rejected += 1
                continue
        a = obs[0]
        u, v = np.rint(a[2]).astype(int)
        im = frames[a[0]]["im"]
        u = np.clip(u, 0, im.shape[1] - 1)
        v = np.clip(v, 0, im.shape[0] - 1)
        rgb = im[v, u, ::-1]
        item = (X, rgb, np.median(errors))
        # Grid-balanced features are admitted to BA only after stronger
        # multi-view validation. They are no longer merely diagnostic points.
        grid_strong = (
            not is_core
            and len(obs) >= 4
            and np.median(errors) <= reproj_px
            and np.percentile(errors, 90) <= 1.5 * reproj_px
        )
        if is_core or grid_strong:
            points.append(item)
            tracks.append(obs)
        else:
            coverage_points.append(item)
            coverage_tracks.append(obs)
    final_support = [[0, set()] for _ in frames]
    for (X, c, e), tr in zip(points, tracks):
        if len(tr) < 4 or e > reproj_px:
            continue
        for fi, ki, uv in tr:
            h, w = frames[fi]["im"].shape[:2]
            u, v = uv
            final_support[fi][0] += 1
            final_support[fi][1].add(
                (
                    min(grid_x - 1, int(u * grid_x / w)),
                    min(grid_y - 1, int(v * grid_y / h)),
                )
            )
    final_support = [(count, len(cells)) for count, cells in final_support]
    held_err = [
        sampson(p, q, fundamental(frames[i], frames[j], K)) for i, j, p, q in held
    ]
    loop_landmarks = sum(
        any(tr[k + 1][0] - tr[k][0] > max_gap for k in range(len(tr) - 1))
        for tr in tracks
    )
    return (
        points,
        tracks,
        coverage_points,
        held,
        dict(
            features=offsets[-1],
            accepted_matches=accepted,
            heldout_matches=len(held),
            conflicting_track_merges_rejected=conflict_rejected,
            pruned_track_observations=pruned_observations,
            moderate_quality_tracks_rejected=quality_rejected,
            coverage_landmarks=len(coverage_points),
            coverage_observations=sum(map(len, coverage_tracks)),
            weak_images_detected=len(weak),
            recovery_pairs=len(recovery_pairs),
            recovery_accepted_matches=accepted - before_recovery,
            visual_recovery_pairs_attempted=visual_pairs_attempted,
            visual_recovery_pairs_accepted=visual_pairs_accepted,
            visual_only_recovery_matches=visual_matches_accepted,
            images_meeting_target_before=sum(
                c >= target_landmarks and g >= target_grid_cells
                for c, g in initial_support
            ),
            images_meeting_target_after_matching=sum(
                c >= target_landmarks and g >= target_grid_cells
                for c, g in recovered_support
            ),
            images_meeting_strong_target_final=sum(
                c >= target_landmarks and g >= target_grid_cells
                for c, g in final_support
            ),
            target_landmarks_per_image=target_landmarks,
            target_occupied_grid_cells=target_grid_cells,
            per_image_support_before=[
                {"landmarks": c, "occupied_grid_cells": g} for c, g in initial_support
            ],
            per_image_support_after_matching=[
                {"landmarks": c, "occupied_grid_cells": g} for c, g in recovered_support
            ],
            per_image_strong_support_final=[
                {"landmarks": c, "occupied_grid_cells": g} for c, g in final_support
            ],
            loop_candidates=len(loop_pairs),
            accepted_loop_matches=loop_matches,
            loop_landmarks=loop_landmarks,
            heldout_epipolar_median_px=float(np.median(held_err)) if held_err else None,
            heldout_epipolar_p90_px=(
                float(np.percentile(held_err, 90)) if held_err else None
            ),
        ),
    )


def save(out, points, tracks, coverage_points, held, stats, min_points=100):
    """Serialize triangulation products and update metrics.

    Purpose:
        Export BA landmarks, diagnostic coverage points, observations, held-out
        matches, a compressed track archive, and triangulation statistics.
    Inputs:
        out: Output directory path.
        points: Accepted ``(XYZ, RGB, error)`` landmark entries.
        tracks: Observation lists corresponding to ``points``.
        coverage_points: Diagnostic point entries excluded from BA.
        held: Held-out pair matches used for validation.
        stats: Metrics dictionary updated in place with output statistics.
        min_points: Minimum accepted landmark count for a successful exit.
    Outputs:
        Writes PLY, CSV, NPZ, and JSON artifacts and returns ``None``; raises
        ``SystemExit(2)`` when fewer than ``min_points`` landmarks are available.
    """
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sparse_points.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(points))
        f.write(
            "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float error\nend_header\n"
        )
        for x, c, e in points:
            f.write("%g %g %g %d %d %d %g\n" % (*x, *c, e))
    with open(out / "coverage_points.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(coverage_points))
        f.write(
            "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float error\nend_header\n"
        )
        for x, c, e in coverage_points:
            f.write("%g %g %g %d %d %d %g\n" % (*x, *c, e))
    with open(out / "tracks.csv", "w") as f:
        f.write("track,frame,u,v\n")
        for i, tr in enumerate(tracks):
            for o in tr:
                f.write("%d,%d,%.9g,%.9g\n" % (i, o[0], o[2][0], o[2][1]))
    with open(out / "heldout.csv", "w") as f:
        f.write("frame1,frame2,u1,v1,u2,v2\n")
        for i, j, p, q in held:
            f.write("%d,%d,%.9g,%.9g,%.9g,%.9g\n" % (i, j, p[0], p[1], q[0], q[1]))
    np.savez_compressed(
        out / "tracks.npz",
        xyz=np.array([p[0] for p in points]),
        rgb=np.array([p[1] for p in points]),
        frame=np.array([o[0] for tr in tracks for o in tr], np.int32),
        uv=np.array([o[2] for tr in tracks for o in tr]),
        track=np.array([i for i, tr in enumerate(tracks) for o in tr], np.int32),
    )
    errors = [p[2] for p in points]
    stats.update(
        triangulated_points=len(points),
        observations=sum(map(len, tracks)),
        triangulation_reprojection_median_px=(
            float(np.median(errors)) if errors else None
        ),
        triangulation_reprojection_p90_px=(
            float(np.percentile(errors, 90)) if errors else None
        ),
    )
    with open(out / "triangulation_metrics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    if len(points) < min_points:
        raise SystemExit(2)


def main():
    """Run sparse multi-view triangulation from initialized camera poses.

    Purpose:
        Load cameras and intrinsics, configure feature matching and weak-frame
        recovery, triangulate tracks, and save reconstruction artifacts.
    Inputs:
        Command-line dataset/output and optional pose/intrinsics paths plus image,
        matching, geometry, loop, recovery, coverage, and quality parameters.
    Outputs:
        Writes feature caches and sparse reconstruction artifacts through
        :func:`save`; returns ``None`` unless the minimum-point check exits.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("output/lio_camera_pose"))
    p.add_argument("--pose-file", type=Path)
    p.add_argument("--intrinsics", type=Path)
    p.add_argument("--max-images", type=int, default=40)
    p.add_argument("--max-pair-gap", type=int, default=6)
    p.add_argument("--epipolar-px", type=float, default=4)
    p.add_argument("--min-parallax-deg", type=float, default=0.5)
    p.add_argument("--three-view-min-parallax-deg", type=float, default=0.75)
    p.add_argument("--reprojection-px", type=float, default=4)
    p.add_argument("--loop-min-separation", type=int, default=20)
    p.add_argument("--loop-max-distance", type=float, default=0.5)
    p.add_argument("--loop-max-per-frame", type=int, default=3)
    p.add_argument("--loop-max-view-angle", type=float, default=60)
    p.add_argument("--min-points", type=int, default=100)
    p.add_argument("--target-landmarks-per-image", type=int, default=50)
    p.add_argument(
        "--target-grid-cells",
        type=int,
        default=16,
        help="target occupied cells in the 8x6 image grid",
    )
    p.add_argument("--weak-recovery-gap", type=int, default=20)
    p.add_argument("--weak-spatial-distance", type=float, default=1.0)
    p.add_argument("--weak-spatial-pairs", type=int, default=8)
    p.add_argument("--grid-min-sift", type=int, default=20)
    p.add_argument("--grid-extra-features", type=int, default=40)
    a = p.parse_args()
    intrinsics = a.intrinsics or a.data / "intrinsics.txt"
    # A rerun can use the last BA solution for spatial-neighbor discovery and
    # epipolar gating; a clean run naturally falls back to the LIO prior.
    previous_optimized = a.output / "poses_optimized_tum.txt"
    pose_file = a.pose_file or (
        previous_optimized
        if previous_optimized.exists()
        else a.output / "poses_lio_prior_tum.txt"
    )
    K = np.loadtxt(intrinsics, delimiter=",")
    frames = load_frames(a.data, pose_file, a.max_images)
    pts, tr, cpts, held, st = triangulate(
        frames,
        K,
        a.max_pair_gap,
        a.epipolar_px,
        a.min_parallax_deg,
        a.reprojection_px,
        a.loop_min_separation,
        a.loop_max_distance,
        a.loop_max_per_frame,
        a.loop_max_view_angle,
        a.output / "cache",
        a.three_view_min_parallax_deg,
        a.target_landmarks_per_image,
        a.target_grid_cells,
        a.weak_recovery_gap,
        a.weak_spatial_distance,
        a.weak_spatial_pairs,
        8,
        6,
        a.grid_min_sift,
        a.grid_extra_features,
    )
    st.update(
        pose_file=str(pose_file),
        intrinsics_file=str(intrinsics),
        three_view_min_parallax_deg=a.three_view_min_parallax_deg,
    )
    initialization_metrics = a.output / "metrics.json"
    time_offset_seconds = 0.0
    if initialization_metrics.is_file():
        time_offset_seconds = float(json.loads(initialization_metrics.read_text()).get(
            "time_offset_seconds", time_offset_seconds))
    st.update(images=len(frames), time_offset_seconds=time_offset_seconds)
    save(a.output, pts, tr, cpts, held, st, a.min_points)


if __name__ == "__main__":
    main()
