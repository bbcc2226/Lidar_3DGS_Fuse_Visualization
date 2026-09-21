#!/usr/bin/env python3
"""Export a 3DGS dataset while excluding named cameras from a text sparse model."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def main():
    """Export a filtered COLMAP text model and linked image set for 3DGS.

    Purpose:
        Exclude cameras selected by landmark diagnostics while retaining the
        source camera model and landmarks.
    Inputs:
        Command-line paths for the source model, images, diagnostics, and output,
        plus an option to exclude cameras with zero observed landmarks.
    Outputs:
        Writes ``sparse/0`` model files, image symlinks, and
        ``filter_report.json``; prints the report and returns ``None``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exclude-zero", action="store_true")
    a = p.parse_args()
    rows = list(
        csv.DictReader((a.diagnostics / "per_image_landmark_quality.csv").open())
    )
    excluded = {
        r["image"] for r in rows if a.exclude_zero and int(r["observed_landmarks"]) == 0
    }
    sparse = a.output / "sparse" / "0"
    images_out = a.output / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.input / "sparse" / "0" / "cameras.txt", sparse / "cameras.txt")
    shutil.copy2(a.input / "sparse" / "0" / "points3D.txt", sparse / "points3D.txt")
    lines = (a.input / "sparse" / "0" / "images.txt").read_text().splitlines()
    kept = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        header = lines[i]
        points = lines[i + 1] if i + 1 < len(lines) else ""
        name = header.split()[-1]
        if name not in excluded:
            kept.extend((header, points))
        i += 2
    (sparse / "images.txt").write_text("\n".join(kept) + "\n")
    for r in rows:
        name = r["image"]
        if name in excluded:
            continue
        src = (a.images / name).resolve()
        dst = images_out / name
        if not src.exists():
            raise FileNotFoundError(src)
        if not dst.exists():
            os.symlink(src, dst)
    report = {
        "source": str(a.input),
        "images_total": len(rows),
        "images_kept": len(rows) - len(excluded),
        "images_excluded": len(excluded),
        "excluded_images": sorted(excluded),
        "sparse_model": str(sparse),
        "images_directory": str(images_out),
    }
    (a.output / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
