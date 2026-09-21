#!/usr/bin/env python3
"""Rebuild tracks from staged poses, extend them once, and run staged BA."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd, log, allow_two=False):
    """Run a subprocess while appending its command and output to a log.

    Purpose:
        Provide auditable execution and consistent failure handling for rebuild stages.
    Inputs:
        cmd: Sequence of executable and argument values.
        log: Path object for the append-only stage log.
        allow_two: Whether exit status 2 is accepted as a nonfatal quality result.
    Outputs:
        Returns ``None`` on an accepted status; otherwise raises ``RuntimeError``.
    """
    line = "+ " + " ".join(map(str, cmd))
    print(line, flush=True)
    with log.open("a") as f:
        f.write(line + "\n")
        f.flush()
        r = subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode and not (allow_two and r.returncode == 2):
        raise RuntimeError(f"failed ({r.returncode}): {line}")


def read(path):
    """Load a JSON document from disk.

    Purpose:
        Centralize JSON report and history loading for the rebuild pipeline.
    Inputs:
        path: Path to a JSON file.
    Outputs:
        Decoded Python object from the JSON document.
    """
    with path.open() as f:
        return json.load(f)


def main():
    """Rebuild, extend, and optimize tracks from staged camera poses.

    Purpose:
        Re-triangulate growing image batches from an optimized seed, run staged
        BA, extend tracks once, and compare the final result with the baseline.
    Inputs:
        Command-line data, seed, cache, intermediate/final output paths and
        image-count, batch, BA-period, parallax, and resume controls.
    Outputs:
        Writes reconstruction stages, logs, histories, and
        ``rebuild_final_report.json``; prints the report and returns ``None``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument(
        "--seed", type=Path, default=Path("output/lio_camera_pose_staged_intrinsics")
    )
    p.add_argument(
        "--cache-source", type=Path, default=Path("output/lio_camera_pose/cache")
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/lio_camera_pose_rebuilt_from_staged"),
    )
    p.add_argument(
        "--final-output",
        type=Path,
        default=Path("output/lio_camera_pose_rebuilt_extended_staged"),
    )
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--bootstrap", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--global-period", type=int, default=100)
    p.add_argument("--three-view-min-parallax-deg", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    pose = a.seed / "poses_optimized_tum.txt"
    K = a.seed / "intrinsics_refined.txt"
    prior = a.seed / "poses_lio_prior_tum.txt"
    for q in (pose, K, prior, a.cache_source):
        if not q.exists():
            raise FileNotFoundError(q)
    available = sum(
        (a.data / "undistorted" / line.split()[0]).exists()
        for line in (a.data / "timestamps.txt").open()
    )
    total = available if a.max_images < 0 else min(available, a.max_images)
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise RuntimeError(f"{a.output} is non-empty; choose a new output or --resume")
    a.output.mkdir(parents=True, exist_ok=True)
    log = a.output / "rebuild.log"
    history_file = a.output / "rebuild_history.json"
    if not a.resume:
        log.write_text("")
        for src, name in (
            (pose, "poses_seed_tum.txt"),
            (pose, "poses_initial_tum.txt"),
            (prior, "poses_lio_prior_tum.txt"),
            (K, "intrinsics_input.txt"),
        ):
            shutil.copy2(src, a.output / name)
        os.symlink(
            a.cache_source.resolve(), a.output / "cache", target_is_directory=True
        )
    history = read(history_file) if a.resume and history_file.exists() else []
    active = (
        min(total, history[-1]["active_images"] + a.batch_size)
        if history
        else min(total, a.bootstrap)
    )
    last_global = max(
        (x["active_images"] for x in history if x["global_ba"]), default=0
    )
    start = time.time()
    while True:
        run(
            [
                "python3",
                SCRIPT_DIR / "triangulate_sparse.py",
                "--data",
                a.data,
                "--output",
                a.output,
                "--pose-file",
                pose,
                "--intrinsics",
                K,
                "--max-images",
                active,
                "--max-pair-gap",
                "6",
                "--epipolar-px",
                "1.5",
                "--min-parallax-deg",
                "0.5",
                "--three-view-min-parallax-deg",
                a.three_view_min_parallax_deg,
                "--reprojection-px",
                "2.0",
                "--loop-min-separation",
                "20",
                "--loop-max-distance",
                "0.5",
                "--loop-max-per-frame",
                "3",
                "--min-points",
                "1",
            ],
            log,
        )
        tri = read(a.output / "triangulation_metrics.json")
        glob = active == total or active - last_global >= a.global_period
        cmd = [
            "./build/lio_bundle_adjust",
            "--data",
            a.data,
            "--output",
            a.output,
            "--intrinsics",
            K,
            "--max-images",
            active,
            "--local-start",
            max(0, active - 15),
        ]
        if not glob:
            cmd.append("--no-global")
        run(cmd, log, True)
        ev = read(a.output / "evaluation.json")
        row = {
            "active_images": active,
            "global_ba": glob,
            "landmarks": tri["triangulated_points"],
            "observations": tri["observations"],
            "three_view_rejected": tri.get("moderate_quality_tracks_rejected", 0),
            "loop_candidates": tri.get("loop_candidates", 0),
            "accepted_loop_matches": tri.get("accepted_loop_matches", 0),
            "loop_landmarks": tri.get("loop_landmarks", 0),
            "reprojection_median_px": ev["reprojection_median_px"],
            "reprojection_p90_px": ev["reprojection_p90_px"],
            "heldout_p90_px": ev["heldout_epipolar_p90_px"],
            "elapsed_seconds": time.time() - start,
        }
        history.append(row)
        history_file.write_text(json.dumps(history, indent=2))
        if active % 100 < a.batch_size or active == total:
            print(
                f"REBUILD {active}/{total}: landmarks={row['landmarks']} obs={row['observations']} nonlocal_loops={row['loop_landmarks']} p90={row['reprojection_p90_px']:.4f}",
                flush=True,
            )
        if glob:
            last_global = active
        if active == total:
            break
        active = min(total, active + a.batch_size)
    if a.final_output.exists() and any(a.final_output.iterdir()):
        raise RuntimeError(f"{a.final_output} must be empty")
    a.final_output.mkdir(parents=True, exist_ok=True)
    flog = a.final_output / "final_staged.log"
    flog.write_text("")
    run(
        [
            "python3",
            SCRIPT_DIR / "extend_tracks.py",
            "--data",
            a.data,
            "--input",
            a.output,
            "--output",
            a.final_output,
            "--intrinsics",
            K,
            "--cache-source",
            a.cache_source,
            "--search-radius",
            "20",
            "--descriptor-ratio",
            "0.82",
            "--descriptor-max",
            "240",
            "--epipolar-px",
            "1.25",
            "--final-reprojection-px",
            "2.0",
            "--max-depth",
            "30",
        ],
        flog,
    )
    run(
        [
            "./build/lio_bundle_adjust",
            "--data",
            a.data,
            "--output",
            a.final_output,
            "--intrinsics",
            K,
            "--max-images",
            total,
            "--local-start",
            "0",
            "--staged",
        ],
        flog,
        True,
    )
    final = read(a.final_output / "evaluation.json")
    base = read(a.seed / "evaluation.json")
    ext = read(a.final_output / "track_extension_metrics.json")
    report = {
        "pass": bool(final.get("pass")),
        "images": total,
        "timestamp_offset_seconds": 0.0,
        "initialization": "staged optimized poses; landmarks rebuilt from scratch",
        "three_view_policy": {"minimum_parallax_deg": a.three_view_min_parallax_deg},
        "extension": ext,
        "final": final,
        "baseline_staged": base,
        "delta": {
            "landmarks": final["landmarks"] - base["landmarks"],
            "observations": final["observations"] - base["observations"],
            "reprojection_median_px": final["reprojection_median_px"]
            - base["reprojection_median_px"],
            "reprojection_p90_px": final["reprojection_p90_px"]
            - base["reprojection_p90_px"],
            "heldout_epipolar_p90_px": final["heldout_epipolar_p90_px"]
            - base["heldout_epipolar_p90_px"],
        },
    }
    report["accepted_as_new_best"] = (
        report["pass"]
        and report["delta"]["reprojection_p90_px"] <= 0
        and report["delta"]["heldout_epipolar_p90_px"] <= 0
    )
    (a.final_output / "rebuild_final_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
