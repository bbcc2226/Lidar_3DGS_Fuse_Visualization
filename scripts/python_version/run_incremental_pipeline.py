#!/usr/bin/env python3
"""Run sparse reconstruction and bundle adjustment in growing image batches."""

import argparse
import json
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd, allow_two=False):
    """Run one pipeline subprocess and enforce its exit status.

    Purpose:
        Echo commands and stop incremental processing when a stage fails.
    Inputs:
        cmd: Sequence of executable and argument values.
        allow_two: Whether exit status 2 is accepted as a nonfatal quality result.
    Outputs:
        Returns ``None`` on an accepted status; otherwise raises ``SystemExit``
        with the subprocess status.
    """
    print("+", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(x) for x in cmd])
    if r.returncode and not (allow_two and r.returncode == 2):
        raise SystemExit(r.returncode)


def main():
    """Run triangulation and bundle adjustment over growing image batches.

    Purpose:
        Bootstrap camera initialization, periodically perform global BA, and
        record reconstruction quality as the active image set grows.
    Inputs:
        Command-line data/output paths and image-count, batch, local-window,
        global-BA, loop-closure, and resume controls.
    Outputs:
        Runs pipeline subprocesses, writes ``incremental_history.json``, prints
        the final history entry, and returns ``None``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument(
        "--output", type=Path, default=Path("output/lio_camera_pose_incremental")
    )
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--bootstrap", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--recent-window", type=int, default=6)
    p.add_argument("--global-ba-period", type=int, default=100)
    p.add_argument("--loop-min-separation", type=int, default=20)
    p.add_argument("--loop-max-distance", type=float, default=0.5)
    p.add_argument("--loop-max-per-frame", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    available = sum(
        1
        for line in open(a.data / "timestamps.txt")
        if (a.data / "undistorted" / line.split()[0]).exists()
    )
    total = available if a.max_images < 0 else min(available, a.max_images)
    a.output.mkdir(parents=True, exist_ok=True)
    history_path = a.output / "incremental_history.json"
    if a.resume and history_path.exists():
        history = json.load(open(history_path))
        last_global = max(
            (x["active_images"] for x in history if x["global_ba"]), default=0
        )
        active = min(total, history[-1]["active_images"] + a.batch_size)
        print(
            "RESUME",
            history[-1]["active_images"],
            "->",
            active,
            "last_global",
            last_global,
            flush=True,
        )
    else:
        run(
            [
                "./build/lio_camera_pose",
                "--data",
                a.data,
                "--output",
                a.output,
                "--max-images",
                total,
            ]
        )
        history = []
        active = min(a.bootstrap, total)
        last_global = 0
    while True:
        run(
            [
                "python3",
                SCRIPT_DIR / "triangulate_sparse.py",
                "--data",
                a.data,
                "--output",
                a.output,
                "--max-images",
                active,
                "--max-pair-gap",
                a.recent_window,
                "--loop-min-separation",
                a.loop_min_separation,
                "--loop-max-distance",
                a.loop_max_distance,
                "--loop-max-per-frame",
                a.loop_max_per_frame,
                "--min-points",
                "1",
            ]
        )
        tri = json.load(open(a.output / "triangulation_metrics.json"))
        # Periodic global BA limits drift; intermediate batches only optimize
        # the newest local window and its fixed boundary cameras.
        do_global = active == total or active - last_global >= a.global_ba_period
        cmd = [
            "./build/lio_bundle_adjust",
            "--data",
            a.data,
            "--output",
            a.output,
            "--max-images",
            active,
            "--local-start",
            max(0, active - 15),
        ]
        if not do_global:
            cmd += ["--no-global"]
        run(cmd, allow_two=True)
        ev = json.load(open(a.output / "evaluation.json"))
        history.append(
            dict(
                active_images=active,
                global_ba=do_global,
                landmarks=tri["triangulated_points"],
                observations=tri["observations"],
                loop_candidates=tri.get("loop_candidates", 0),
                accepted_loop_matches=tri.get("accepted_loop_matches", 0),
                conflicting_track_merges_rejected=tri.get(
                    "conflicting_track_merges_rejected", 0
                ),
                pruned_track_observations=tri.get("pruned_track_observations", 0),
                loop_landmarks=tri.get("loop_landmarks", 0),
                coverage_landmarks=tri.get("coverage_landmarks", 0),
                coverage_observations=tri.get("coverage_observations", 0),
                reprojection_median_px=ev["reprojection_median_px"],
            )
        )
        with open(a.output / "incremental_history.json", "w") as f:
            json.dump(history, f, indent=2)
        if do_global:
            last_global = active
        if active == total:
            break
        active = min(total, active + a.batch_size)
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
