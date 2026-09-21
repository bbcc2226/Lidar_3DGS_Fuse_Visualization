#!/usr/bin/env python3
"""Recover landmark-free cameras using neighbor track matches and PnP-RANSAC."""

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from extend_tracks import ply, poses


def main():
    """Recover weak camera poses and observations with PnP-RANSAC.

    Purpose:
        Link cached pair matches to established landmarks, estimate target-frame
        poses, and add geometrically verified inlier observations.
    Inputs:
        Command-line data, reconstruction, output, intrinsics, feature-cache,
        and target-frame paths plus neighbor and reprojection thresholds.
    Outputs:
        Writes updated tracks, sparse points, initial poses, copied priors and
        intrinsics, and ``pnp_recovery.json``; prints progress and returns ``None``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--cache-source", type=Path, required=True)
    p.add_argument("--target-frames-file", type=Path, required=True)
    p.add_argument("--neighbor-gap", type=int, default=10)
    p.add_argument("--feature-link-px", type=float, default=2.0)
    p.add_argument("--pnp-reprojection-px", type=float, default=3.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    K = np.loadtxt(a.intrinsics, delimiter=",")
    ps = poses(a.input / "poses_optimized_tum.txt")
    X, C = ply(a.input / "landmarks_optimized.ply")
    names = [x.split()[0] for x in open(a.data / "timestamps.txt")][: len(ps)]
    cache = a.cache_source / "features_v4_core_coverage"
    pairs = a.cache_source / "pairs_v4_core_coverage"
    targets = {int(x) for x in a.target_frames_file.read_text().split()}
    obs = defaultdict(list)
    for r in csv.DictReader((a.input / "tracks.csv").open()):
        obs[int(r["track"])].append([int(r["frame"]), float(r["u"]), float(r["v"])])
    existing = {(tid, o[0]) for tid, v in obs.items() for o in v}
    byframe = defaultdict(list)
    for tid, v in obs.items():
        for fr, u, w in v:
            byframe[fr].append((tid, u, w))
    feature_track = {}
    # Reconnect exported track observations to cached feature indices by their
    # pixel locations; the cache does not store the final track IDs.
    for fr, items in byframe.items():
        z = np.load(cache / (Path(names[fr]).stem + ".npz"))
        xy = z["xy"]
        for tid, u, v in items:
            d = np.sum((xy - [u, v]) ** 2, axis=1)
            k = int(np.argmin(d))
            if d[k] <= a.feature_link_px * a.feature_link_px:
                feature_track[(fr, k)] = tid
    pose_lines = [
        line
        for line in (a.input / "poses_optimized_tum.txt").read_text().splitlines()
        if len(line.split()) == 8
    ]
    stats = []
    for f in sorted(targets):
        proposals = defaultdict(list)
        for n in range(
            max(0, f - a.neighbor_gap), min(len(ps), f + a.neighbor_gap + 1)
        ):
            if n == f:
                continue
            lo, hi = min(f, n), max(f, n)
            path = pairs / ("%06d_%06d.npz" % (lo, hi))
            if not path.exists():
                continue
            z = np.load(path)
            for qi, ti in zip(z["qidx"], z["tidx"]):
                kf, kn = (int(qi), int(ti)) if f == lo else (int(ti), int(qi))
                tid = feature_track.get((n, kn))
                if tid is not None and (tid, f) not in existing:
                    proposals[(tid, kf)].append(n)
        best = {}
        for (tid, kf), support in proposals.items():
            old = best.get(tid)
            if old is None or len(support) > len(old[1]):
                best[tid] = (kf, support)
        zf = np.load(cache / (Path(names[f]).stem + ".npz"))
        xy = zf["xy"]
        tids = []
        uv = []
        for tid, (kf, support) in best.items():
            if tid < len(X):
                tids.append(tid)
                uv.append(xy[kf])
        accepted = []
        ok = False
        if len(tids) >= 6:
            obj = X[tids].astype(np.float64)
            img = np.asarray(uv, np.float64)
            R0 = ps[f][1].T
            t0 = -R0 @ ps[f][0]
            rv0 = cv2.Rodrigues(R0)[0]
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                obj,
                img,
                K,
                None,
                rv0,
                t0.reshape(3, 1),
                True,
                2000,
                a.pnp_reprojection_px,
                0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok and inliers is not None and len(inliers) >= 6:
                idx = inliers.ravel()
                accepted = [(tids[i], *uv[i]) for i in idx]
                Rcw = cv2.Rodrigues(rv)[0]
                Cw = -Rcw.T @ tv.ravel()
                qw = np.sqrt(max(0, 1 + np.trace(Rcw.T))) / 2
                R = Rcw.T
                qx = (R[2, 1] - R[1, 2]) / (4 * qw)
                qy = (R[0, 2] - R[2, 0]) / (4 * qw)
                qz = (R[1, 0] - R[0, 1]) / (4 * qw)
                zz = pose_lines[f].split()
                pose_lines[f] = " ".join(
                    [
                        zz[0],
                        *(f"{v:.16g}" for v in Cw),
                        f"{qx:.16g}",
                        f"{qy:.16g}",
                        f"{qz:.16g}",
                        f"{qw:.16g}",
                    ]
                )
                for tid, u, v in accepted:
                    obs[tid].append([f, float(u), float(v)])
                    existing.add((tid, f))
        stats.append(
            {
                "frame": f,
                "image": names[f],
                "candidates": len(tids),
                "pnp_inliers": len(accepted),
                "success": bool(ok and len(accepted) >= 6),
            }
        )
        print(stats[-1], flush=True)
    with (a.output / "tracks.csv").open("w", newline="") as q:
        w = csv.writer(q)
        w.writerow(["track", "frame", "u", "v"])
        for tid in sorted(obs):
            for fr, u, v in sorted(obs[tid]):
                w.writerow([tid, fr, u, v])
    with (a.output / "sparse_points.ply").open("w") as q:
        q.write(
            "ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            % len(X)
        )
        for x, c in zip(X, C):
            q.write("%g %g %g %d %d %d\n" % (*x, *c))
    (a.output / "poses_initial_tum.txt").write_text("\n".join(pose_lines) + "\n")
    for name in ("poses_lio_prior_tum.txt", "heldout.csv"):
        shutil.copy2(a.input / name, a.output / name)
    shutil.copy2(a.intrinsics, a.output / "intrinsics_input.txt")
    report = {
        "targets": len(targets),
        "recovered": sum(x["success"] for x in stats),
        "added_observations": sum(x["pnp_inliers"] for x in stats),
        "frames": stats,
    }
    (a.output / "pnp_recovery.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
