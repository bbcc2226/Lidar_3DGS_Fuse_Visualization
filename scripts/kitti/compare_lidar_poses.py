#!/usr/bin/env python3
"""Convert refined camera poses to world-from-LiDAR and compare timestamp-aligned references."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare_kitti_pipeline import read_lio_records


def stats(a):
    a = np.asarray(a)
    return dict(
        rmse=float(np.sqrt(np.mean(a * a))),
        median=float(np.median(a)),
        p90=float(np.percentile(a, 90)),
        p95=float(np.percentile(a, 95)),
        max=float(np.max(a)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconstruction", type=Path, required=True)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start-image-id", type=int)
    ap.add_argument("--end-image-id", type=int)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.prepared / "preparation_summary.json").read_text())
    source = Path(meta["lidar_slam_directory"])
    names = []
    times = []
    for line in (args.prepared / "timestamps.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, t = line.split()
        names.append(name)
        times.append(float(t))
    ids = np.array([int(Path(name).stem) for name in names])
    times = np.array(times)
    poses = np.loadtxt(args.reconstruction / "poses_optimized_tum.txt")
    prior = np.loadtxt(args.reconstruction / "poses_lio_prior_tum.txt")
    assert len(poses) == len(times)
    selected = np.ones(len(ids), dtype=bool)
    if args.start_image_id is not None:
        selected &= ids >= args.start_image_id
    if args.end_image_id is not None:
        selected &= ids <= args.end_image_id
    ids = ids[selected]
    times = times[selected] + meta["camera_time_offset_seconds"]
    poses = poses[selected]
    prior = prior[selected]
    tc_l = np.array(
        json.loads((args.prepared / "camera_lidar_extrinsic.json").read_text())[
            "T_camera_lidar"
        ]
    )

    def convert(camera):
        rc = Rotation.from_quat(camera[:, 4:8]).as_matrix()
        return rc @ tc_l[:3, :3], camera[:, 1:4] + np.einsum(
            "nij,j->ni", rc, tc_l[:3, 3]
        )

    rotation, position = convert(poses)
    prior_rotation, prior_position = convert(prior)

    def write_tum(name, r, t):
        np.savetxt(
            args.output / name,
            np.column_stack((times, t, Rotation.from_matrix(r).as_quat())),
            fmt="%.12f",
            header="query_timestamp tx ty tz qx qy qz qw; world_from_lidar",
        )

    write_tum("poses_corrected_lidar_tum.txt", rotation, position)
    summary = {
        "image_range": [int(ids[0]), int(ids[-1])],
        "images": len(ids),
        "transform": "T_world_lidar_corrected = T_world_camera_corrected @ T_camera_lidar",
        "reference_time": "image_timestamp + camera_time_offset_seconds; source translation linearly interpolated and orientation SLERP",
        "alignment": "None: compare in existing SLAM world coordinates; these are differences, not ground-truth errors.",
    }
    rows = [
        {"image_id": int(i), "query_timestamp": float(t)} for i, t in zip(ids, times)
    ]
    series = {}
    for field, label in [
        ("optimized_pose", "optimized_lidar_prior"),
        ("lio_pose", "raw_lio"),
    ]:
        records, _ = read_lio_records(source, field, "last_revision")
        rt = np.array([r["timestamp"] for r in records])
        t0 = rt[0]
        assert times[0] >= rt[0] and times[-1] <= rt[-1]
        ref_r = Slerp(
            rt - t0,
            Rotation.from_quat([r["lio_pose"]["quaternion_xyzw"] for r in records]),
        )(times - t0).as_matrix()
        source_position = np.array([r["lio_pose"]["translation"] for r in records])
        ref_t = np.column_stack(
            [np.interp(times, rt, source_position[:, j]) for j in range(3)]
        )
        delta_world = position - ref_t
        delta_local = np.einsum("nji,nj->ni", ref_r, delta_world)
        distance = np.linalg.norm(delta_world, axis=1)
        angle = (
            Rotation.from_matrix(np.einsum("nji,njk->nik", ref_r, rotation)).magnitude()
            * 180
            / np.pi
        )
        summary[label] = {
            "translation_m": stats(distance),
            "rotation_deg": stats(angle),
            "max_translation_image_id": int(ids[np.argmax(distance)]),
        }
        series[label] = (distance, angle)
        write_tum(f"poses_reference_{label}_tum.txt", ref_r, ref_t)
        for j, row in enumerate(rows):
            row[label + "_translation_m"] = float(distance[j])
            row[label + "_rotation_deg"] = float(angle[j])
            for k, axis in enumerate("xyz"):
                row[label + "_delta_world_" + axis + "_m"] = float(delta_world[j, k])
                row[label + "_delta_reference_lidar_" + axis + "_m"] = float(
                    delta_local[j, k]
                )
        if field == "optimized_pose":
            squared_sum = float(np.sum(distance**2))
            summary["early_frame_contribution"] = {
                "first_image_id": int(ids[0]),
                "first_frame_squared_translation_share_percent": (
                    100 * float(distance[0] ** 2) / squared_sum if squared_sum else 0.0
                ),
                "first_five_image_ids": ids[:5].tolist(),
                "first_five_squared_translation_share_percent": (
                    100 * float(np.sum(distance[:5] ** 2)) / squared_sum
                    if squared_sum
                    else 0.0
                ),
                "translation_rmse_excluding_first_five_m": (
                    float(np.sqrt(np.mean(distance[5:] ** 2)))
                    if len(distance) > 5
                    else None
                ),
            }
            dt = np.linalg.norm(prior_position - ref_t, axis=1)
            dr = Rotation.from_matrix(
                np.einsum("nji,njk->nik", ref_r, prior_rotation)
            ).magnitude()
            assert dt.max() < 1e-6 and dr.max() < 1e-6
            summary["prior_roundtrip_validation"] = {
                "maximum_translation_m": float(dt.max()),
                "maximum_rotation_rad": float(dr.max()),
            }
    with (args.output / "per_frame_lidar_pose_difference.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for label, (distance, angle) in series.items():
        axes[0].plot(ids, distance, label=label.replace("_", " "))
        axes[1].plot(ids, angle, label=label.replace("_", " "))
    axes[0].set_ylabel("LiDAR position difference (m)")
    axes[1].set_ylabel("LiDAR rotation difference (deg)")
    axes[1].set_xlabel("Original image ID")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "lidar_pose_difference.png", dpi=150)
    plt.close(fig)
    text = f"""# Corrected LiDAR pose comparison: images {ids[0]}–{ids[-1]}

Camera BA estimates are converted into world-from-LiDAR poses using `T_world_lidar = T_world_camera @ T_camera_lidar`. This includes the camera–LiDAR translation offset (lever arm), not only the rotation. The SLAM world origin is preserved.

Reference poses are interpolated from the source LiDAR log at each image timestamp plus the configured time offset. Both raw `lio_pose` and `optimized_pose` use the last revision for repeated keyframe timestamps. No trajectory alignment is fitted. These values measure **disagreement with each reference**, not error against independent ground truth. The raw-LIO comparison includes both the SLAM corrections already present in `optimized_pose` and the later visual-BA changes; it does not isolate the effect of BA.

| Reference | Translation RMSE m | Median m | P95 m | Max m | Rotation RMSE deg | Median deg | P95 deg | Max deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for label in series:
        t = summary[label]["translation_m"]
        r = summary[label]["rotation_deg"]
        text += f"| {label} | {t['rmse']:.4f} | {t['median']:.4f} | {t['p95']:.4f} | {t['max']:.4f} | {r['rmse']:.4f} | {r['median']:.4f} | {r['p95']:.4f} | {r['max']:.4f} |\n"
    early = summary["early_frame_contribution"]
    text += f"\nThe first image ({ids[0]}) accounts for {early['first_frame_squared_translation_share_percent']:.2f}% of the total squared position difference from the optimized LiDAR prior. The first five images account for {early['first_five_squared_translation_share_percent']:.2f}%. Position RMSE excluding those five images is {early['translation_rmse_excluding_first_five_m']:.4f} m. These percentages describe contribution to squared differences, not a fraction of ground-truth error.\n"
    text += "\nThe prior-camera → LiDAR round-trip was independently checked against the optimized source trajectory. See `summary.json` for numerical discrepancies. Translation changes can differ from camera-center changes because the sensor centers have different lever arms.\n\n![LiDAR pose differences](lidar_pose_difference.png)\n\n[Per-image differences](per_frame_lidar_pose_difference.csv) · [Corrected world-from-LiDAR TUM trajectory](poses_corrected_lidar_tum.txt)\n"
    (args.output / "REPORT.md").write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
