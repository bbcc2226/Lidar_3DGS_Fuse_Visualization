#!/usr/bin/env python3
"""Evaluate camera trajectories (TUM format) against KITTI OXTS GPS/INS.

Each trajectory is Umeyama-aligned (rotation + translation, no scale) to the
OXTS-derived rectified-camera poses before computing ATE. Relative pose error
uses one-frame steps and is alignment independent. A jitter statistic (second
difference of position) exposes frame-to-frame noise that ATE hides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def read_calibration(path: Path) -> dict[str, np.ndarray]:
    values = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key.strip()] = np.array(raw.split(), float)
        except ValueError:
            continue
    return values


def rigid(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation.reshape(3, 3)
    m[:3, 3] = translation
    return m


def camera_from_imu(sequence: Path, camera: str) -> np.ndarray:
    imu_velo = read_calibration(sequence / "calib_imu_to_velo.txt")
    velo_cam = read_calibration(sequence / "calib_velo_to_cam.txt")
    cam = read_calibration(sequence / "calib_cam_to_cam.txt")
    rect = np.eye(4)
    rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
    p = cam[f"P_rect_{camera[-2:]}"].reshape(3, 4)
    baseline = np.eye(4)
    baseline[:3, 3] = np.linalg.inv(p[:, :3]) @ p[:, 3]
    return (
        baseline
        @ rect
        @ rigid(velo_cam["R"], velo_cam["T"])
        @ rigid(imu_velo["R"], imu_velo["T"])
    )


def oxts_camera_poses(sequence: Path, camera: str, ids: range) -> tuple[np.ndarray, R]:
    t_cam_imu = camera_from_imu(sequence, camera)
    rows = [np.loadtxt(sequence / "oxts" / "data" / f"{i:010d}.txt") for i in ids]
    earth_radius = 6378137.0
    scale = np.cos(np.radians(rows[0][0]))
    centers, rotations = [], []
    for row in rows:
        lat, lon, alt, roll, pitch, yaw = row[:6]
        x = scale * np.radians(lon) * earth_radius
        y = scale * earth_radius * np.log(np.tan(np.radians(90.0 + lat) / 2.0))
        world_from_imu = rigid(
            R.from_euler("xyz", [roll, pitch, yaw]).as_matrix(), [x, y, alt]
        )
        world_from_cam = world_from_imu @ np.linalg.inv(t_cam_imu)
        centers.append(world_from_cam[:3, 3])
        rotations.append(world_from_cam[:3, :3])
    return np.array(centers), R.from_matrix(np.array(rotations))


def read_tum(path: Path) -> tuple[np.ndarray, R]:
    data = np.loadtxt(path)
    return data[:, 1:4], R.from_quat(data[:, 4:8])


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ms, md = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - ms).T @ (dst - md))
    d = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        d[2, 2] = -1
    rotation = vt.T @ d @ u.T
    return rotation, md - rotation @ ms


def relative_steps(centers: np.ndarray, rotations: R) -> tuple[np.ndarray, R]:
    prev = rotations[:-1]
    steps = np.einsum("nij,nj->ni", prev.inv().as_matrix(), centers[1:] - centers[:-1])
    return steps, prev.inv() * rotations[1:]


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def evaluate(
    centers: np.ndarray, rotations: R, gt_centers: np.ndarray, gt_rotations: R
) -> dict:
    rotation, translation = umeyama(centers, gt_centers)
    aligned = (rotation @ centers.T).T + translation
    aligned_rot = R.from_matrix(rotation) * rotations
    ate = np.linalg.norm(aligned - gt_centers, axis=1)
    rot_err = (aligned_rot * gt_rotations.inv()).magnitude()
    gt_steps, gt_rel = relative_steps(gt_centers, gt_rotations)
    steps, rel = relative_steps(aligned, aligned_rot)
    rpe_t = np.linalg.norm(steps - gt_steps, axis=1)
    rpe_r = (rel * gt_rel.inv()).magnitude()
    jitter = np.linalg.norm(np.diff(centers, 2, axis=0), axis=1)
    return {
        "ate_m": stats(ate),
        "rotation_vs_gt_deg": stats(np.degrees(rot_err)),
        "rpe_1frame_translation_m": stats(rpe_t),
        "rpe_1frame_rotation_deg": stats(np.degrees(rpe_r)),
        "position_jitter_2nd_diff_m": stats(jitter),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", type=Path, required=True, help="KITTI *_sync sequence directory"
    )
    parser.add_argument("--camera", default="image_03")
    parser.add_argument(
        "--start-id", type=int, required=True, help="image ID of the first pose"
    )
    parser.add_argument(
        "--poses",
        type=Path,
        nargs="+",
        required=True,
        help="TUM pose files (name=path or path)",
    )
    parser.add_argument("--json", type=Path, help="optional output JSON path")
    args = parser.parse_args()

    trajectories = {}
    for item in args.poses:
        text = str(item)
        name, path = (
            text.split("=", 1) if "=" in text else (Path(text).parent.name, text)
        )
        trajectories[name] = read_tum(Path(path))
    count = {len(c) for c, _ in trajectories.values()}
    if len(count) != 1:
        raise ValueError(f"trajectories have different lengths: {count}")
    n = count.pop()
    gt_centers, gt_rotations = oxts_camera_poses(
        args.sequence, args.camera, range(args.start_id, args.start_id + n)
    )

    report = {
        "frames": n,
        "image_id_range": [args.start_id, args.start_id + n - 1],
        "oxts_position_jitter_2nd_diff_m": stats(
            np.linalg.norm(np.diff(gt_centers, 2, axis=0), axis=1)
        ),
        "trajectories": {},
    }
    for name, (centers, rotations) in trajectories.items():
        report["trajectories"][name] = evaluate(
            centers, rotations, gt_centers, gt_rotations
        )

    header = f"{'trajectory':24s} {'ATE med/p90/max m':>22s} {'rot med deg':>12s} {'RPE-t med/p90/max m':>24s} {'RPE-r med deg':>14s} {'jitter med/p90 m':>18s}"
    print(header)
    for name, m in report["trajectories"].items():
        a, r, t, rr, j = (
            m["ate_m"],
            m["rotation_vs_gt_deg"],
            m["rpe_1frame_translation_m"],
            m["rpe_1frame_rotation_deg"],
            m["position_jitter_2nd_diff_m"],
        )
        print(
            f"{name:24s} {a['median']:6.3f}/{a['p90']:6.3f}/{a['max']:6.3f}   {r['median']:10.2f} "
            f"{t['median']:7.3f}/{t['p90']:6.3f}/{t['max']:6.3f}   {rr['median']:12.3f} "
            f"{j['median']:8.3f}/{j['p90']:6.3f}"
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
