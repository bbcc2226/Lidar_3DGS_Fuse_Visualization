#!/usr/bin/env python3
"""Complete local LiDAR depth using spatially nearby sweeps from the full map."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


def quat_rot(q):
    x, y, z, w = map(float, q)
    n = max(np.linalg.norm([x, y, z, w]), 1e-12)
    x, y, z, w = np.asarray([x, y, z, w]) / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        np.float64,
    )


def json_objects(path):
    buf = ""
    balance = 0
    for line in open(path):
        buf += line
        balance += line.count("{") - line.count("}")
        if balance == 0 and buf.strip():
            yield json.loads(buf)
            buf = ""


def load_poses(path):
    out = []
    for line in open(path):
        z = line.split()
        if len(z) == 8:
            out.append((float(z[0]), np.asarray(z[1:4], float), quat_rot(z[4:8])))
    return out


class CloudCache:
    def __init__(self, limit=256):
        self.limit = limit
        self.items = OrderedDict()

    def get(self, path):
        path = str(path)
        if path in self.items:
            x = self.items.pop(path)
            self.items[path] = x
            return x
        skip = 0
        count = None
        with open(path) as f:
            for line in f:
                skip += 1
                if line.startswith("element vertex "):
                    count = int(line.split()[-1])
                if line.strip() == "end_header":
                    break
        x = np.loadtxt(
            path,
            dtype=np.float32,
            skiprows=skip,
            max_rows=count,
            usecols=(0, 1, 2),
            ndmin=2,
        )
        self.items[path] = x
        if len(self.items) > self.limit:
            self.items.popitem(last=False)
        return x


def colorize(depth, valid, max_depth):
    inv = np.zeros_like(depth, np.float32)
    inv[valid] = 1 / np.maximum(depth[valid], 1e-3)
    lo = 1 / max_depth
    hi = float(np.percentile(inv[valid], 99)) if valid.any() else 1
    x = np.clip((inv - lo) / max(hi - lo, 1e-8), 0, 1)
    c = cv2.applyColorMap(np.rint(255 * x).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~valid] = 0
    return c


def label(im, text):
    x = im.copy()
    cv2.rectangle(x, (0, 0), (320, 38), (0, 0, 0), -1)
    cv2.putText(
        x,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--local-depth", type=Path, default=Path("data/depth_maps"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-sweeps", type=int, default=80)
    p.add_argument("--radius-m", type=float, default=5.0)
    p.add_argument("--max-depth", type=float, default=30.0)
    p.add_argument("--min-sweep-support", type=int, default=2)
    p.add_argument("--surface-tolerance-m", type=float, default=0.08)
    p.add_argument("--overlap-tolerance-m", type=float, default=0.12)
    p.add_argument("--splat-radius", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="*")
    p.add_argument(
        "--preview-frames", type=int, nargs="*", default=[50, 108, 150, 250, 400, 520]
    )
    a = p.parse_args()
    for d in ("depth_npy", "depth_mm", "masks", "confidence", "previews"):
        (a.output / d).mkdir(parents=True, exist_ok=True)
    intr = a.reconstruction / "intrinsics_refined.txt"
    if not intr.exists():
        intr = a.reconstruction / "intrinsics_input.txt"
    if not intr.exists():
        intr = a.data / "intrinsics.txt"
    K = np.loadtxt(intr, delimiter=",")
    poses = load_poses(a.reconstruction / "poses_optimized_tum.txt")
    stamps = [x.split() for x in open(a.data / "timestamps.txt")][: len(poses)]
    sweeps = []
    for x in json_objects(a.data / "key_frames.jsonl"):
        pose = x.get("optimized_pose", x["lio_pose"])
        C = np.asarray(pose["translation"], float)
        R = quat_rot(pose["quaternion_xyzw"])
        path = a.data / str(x["saved_frame_path"]).removeprefix("./")
        sweeps.append((float(x["timestamp"]), C, R, path))
    centers = np.asarray([x[1] for x in sweeps])
    cache = CloudCache()
    selected = set(a.frames) if a.frames else set(range(len(poses)))
    reports = []
    before_total = after_total = added_total = 0
    offsets = (
        [(0, 0)]
        if a.splat_radius <= 0
        else [
            (x, y)
            for x in range(-a.splat_radius, a.splat_radius + 1)
            for y in range(-a.splat_radius, a.splat_radius + 1)
        ]
    )
    for frame in sorted(selected):
        name = stamps[frame][0]
        image = cv2.imread(str(a.data / "undistorted" / name))
        local = cv2.imread(
            str(a.local_depth / (name + "_depth.tiff")), cv2.IMREAD_UNCHANGED
        )
        if image is None or local is None:
            continue
        local = local.astype(np.float32)
        h, w = local.shape
        local_ok = np.isfinite(local) & (local > 0.1) & (local < a.max_depth)
        _, Cc, Rc = poses[frame]
        dist = np.linalg.norm(centers - Cc, axis=1)
        idx = np.flatnonzero(dist <= a.radius_m)
        idx = idx[np.argsort(dist[idx])[: a.max_sweeps]]
        zbuf = np.full(h * w, np.inf, np.float32)
        support = np.zeros(h * w, np.uint16)
        for si in idx:
            _, Cl, Rl, path = sweeps[int(si)]
            pl = cache.get(path)
            pw = pl @ Rl.T + Cl
            pc = (pw - Cc) @ Rc
            good = (pc[:, 2] > 0.1) & (pc[:, 2] < a.max_depth)
            pc = pc[good]
            if not len(pc):
                continue
            u = np.rint(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(np.int32)
            v = np.rint(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(np.int32)
            z = pc[:, 2].astype(np.float32)
            pix = []
            zs = []
            for dx, dy in offsets:
                ok = (u + dx >= 0) & (u + dx < w) & (v + dy >= 0) & (v + dy < h)
                pix.append((v[ok] + dy) * w + (u[ok] + dx))
                zs.append(z[ok])
            pix = np.concatenate(pix)
            zs = np.concatenate(zs)
            if not len(pix):
                continue
            order = np.lexsort((zs, pix))
            pix = pix[order]
            zs = zs[order]
            first = np.r_[True, pix[1:] != pix[:-1]]
            pix = pix[first]
            zs = zs[first]
            old = zbuf[pix]
            closer = zs < old - a.surface_tolerance_m
            agree = np.abs(zs - old) <= a.surface_tolerance_m
            zbuf[pix[closer]] = zs[closer]
            support[pix[closer]] = 1
            support[pix[agree & ~closer]] += 1
            zbuf[pix[agree & ~closer]] = np.minimum(
                old[agree & ~closer], zs[agree & ~closer]
            )
        global_depth = zbuf.reshape(h, w)
        global_support = support.reshape(h, w)
        global_ok = np.isfinite(global_depth) & (global_support >= a.min_sweep_support)
        overlap = local_ok & global_ok
        consistent = ~overlap | (
            np.abs(global_depth - local)
            <= np.maximum(a.overlap_tolerance_m, 0.02 * local)
        )
        fill = global_ok & ~local_ok & consistent
        combined = np.where(local_ok, local, np.where(fill, global_depth, 0)).astype(
            np.float32
        )
        valid = combined > 0
        conf = np.zeros((h, w), np.float32)
        conf[local_ok] = 1.0
        conf[fill] = np.clip(global_support[fill] / 8.0, 0.15, 0.65)
        stem = Path(name).stem
        np.save(a.output / "depth_npy" / (stem + ".npy"), combined)
        mm = np.zeros((h, w), np.uint16)
        mm[valid] = np.clip(np.rint(combined[valid] * 1000), 1, 65535).astype(np.uint16)
        cv2.imwrite(str(a.output / "depth_mm" / (stem + ".png")), mm)
        cv2.imwrite(
            str(a.output / "masks" / (stem + ".png")), valid.astype(np.uint8) * 255
        )
        cv2.imwrite(
            str(a.output / "confidence" / (stem + ".png")),
            np.rint(conf * 65535).astype(np.uint16),
        )
        before = int(local_ok.sum())
        after = int(valid.sum())
        added = int(fill.sum())
        before_total += before
        after_total += after
        added_total += added
        reports.append(
            {
                "frame": frame,
                "image": name,
                "sweeps": len(idx),
                "local_pixels": before,
                "global_added_pixels": added,
                "combined_pixels": after,
                "coverage": after / valid.size,
            }
        )
        if frame in a.preview_frames:
            lc = colorize(np.where(local_ok, local, 0), local_ok, a.max_depth)
            cc = colorize(combined, valid, a.max_depth)
            new = np.zeros_like(image)
            new[fill] = (0, 255, 255)
            panel = np.hstack(
                (
                    label(image, "Original"),
                    label(lc, "Local-window depth"),
                    label(cc, "Global-completed depth"),
                    label(new, "New global pixels"),
                )
            )
            cv2.imwrite(
                str(
                    a.output
                    / "previews"
                    / (f"{frame:06d}_{stem}_global_comparison.jpg")
                ),
                panel,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            )
        if (len(reports) % 25) == 0:
            print("frames", len(reports), "added_pixels", added_total, flush=True)
    report = {
        "frames": len(reports),
        "max_sweeps": a.max_sweeps,
        "radius_m": a.radius_m,
        "min_sweep_support": a.min_sweep_support,
        "local_valid_pixels": before_total,
        "combined_valid_pixels": after_total,
        "global_added_pixels": added_total,
        "coverage_gain": after_total / max(before_total, 1) - 1,
        "intrinsics": str(intr),
        "per_frame": reports,
    }
    (a.output / "global_depth_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
