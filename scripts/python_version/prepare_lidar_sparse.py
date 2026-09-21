#!/usr/bin/env python3
"""Create a COLMAP text model whose initialization points come from a LiDAR TXT cloud."""

import argparse
import shutil
from pathlib import Path


def clear_image_point_ids(source: Path, destination: Path) -> None:
    """Copy COLMAP image records while clearing landmark associations.

    Purpose:
        Preserve camera poses and 2D observations without references to the
        source reconstruction's 3D point identifiers.
    Inputs:
        source: Path to the source COLMAP ``images.txt`` file.
        destination: Path where the modified ``images.txt`` is written.
    Outputs:
        Writes ``destination`` with every observation point ID set to ``-1``;
        returns ``None``.
    """
    lines = source.read_text().splitlines()
    output = []
    expect_header = True
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            output.append(line)
            continue
        if expect_header:
            if not stripped:
                output.append(line)
                continue
            output.append(line)
            expect_header = False
            continue

        fields = stripped.split()
        if fields:
            if len(fields) % 3:
                raise ValueError(f"Malformed images.txt observation line: {line[:120]}")
            for index in range(2, len(fields), 3):
                fields[index] = "-1"
            output.append(" ".join(fields))
        else:
            output.append("")
        expect_header = True
    destination.write_text("\n".join(output) + "\n")


def write_points3d(source: Path, destination: Path) -> int:
    """Convert an XYZRGB LiDAR text cloud to COLMAP points3D text format.

    Purpose:
        Assign sequential point IDs and create trackless initialization points.
    Inputs:
        source: Path to text rows containing at least ``x y z r g b``.
        destination: Path where COLMAP ``points3D.txt`` is written.
    Outputs:
        Number of LiDAR points written to ``destination``.
    """
    count = 0
    with source.open() as input_file, destination.open("w") as output_file:
        output_file.write("# 3D point list with one line of data per point:\n")
        output_file.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        output_file.write(
            "# LiDAR initialization points intentionally have empty tracks.\n"
        )
        for line in input_file:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"Malformed LiDAR point line: {line[:120]}")
            x, y, z, red, green, blue = fields[:6]
            count += 1
            output_file.write(f"{count} {x} {y} {z} {red} {green} {blue} 0\n")
    return count


def main() -> None:
    """Build a COLMAP sparse model initialized from LiDAR points.

    Purpose:
        Copy camera calibration, retain pose-only image records, and replace
        visual landmarks with an RGB LiDAR cloud.
    Inputs:
        Command-line source sparse-model, LiDAR text-cloud, and output paths.
    Outputs:
        Writes ``cameras.txt``, cleared ``images.txt``, and LiDAR
        ``points3D.txt``; prints the point count and returns ``None``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sparse", type=Path, required=True)
    parser.add_argument("--lidar-txt", type=Path, required=True)
    parser.add_argument("--output-sparse", type=Path, required=True)
    args = parser.parse_args()

    args.output_sparse.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_sparse / "cameras.txt", args.output_sparse / "cameras.txt")
    clear_image_point_ids(
        args.source_sparse / "images.txt", args.output_sparse / "images.txt"
    )
    count = write_points3d(args.lidar_txt, args.output_sparse / "points3D.txt")
    print(f"Wrote {count} LiDAR initialization points to {args.output_sparse}")


if __name__ == "__main__":
    main()
