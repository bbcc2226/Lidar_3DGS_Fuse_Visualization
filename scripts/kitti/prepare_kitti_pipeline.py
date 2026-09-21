#!/usr/bin/env python3
"""Prepare KITTI camera/LiDAR inputs and optionally run the root C++ pipeline.

KITTI calibration produces a rectified-camera-from-Velodyne transform. The C++
pipeline accepts world-from-LiDAR poses, so optimized_pose is written as its
trajectory field and the matching camera/LiDAR extrinsic is written separately.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def _matvec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(row[k] * x[k] for k in range(len(x))) for row in a]


def _inverse3(m: list[list[float]]) -> list[list[float]]:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isfinite(det) or abs(det) < 1e-15:
        raise ValueError("camera projection matrix has a singular 3x3 intrinsic block")
    return [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]


def _read_kitti_calibration(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key.strip()] = [float(value) for value in raw.split()]
        except ValueError:
            # KITTI calibration files include textual metadata such as calib_time.
            # Required numeric matrices/vectors are checked explicitly below.
            continue
    return values


def _required(values: dict[str, list[float]], key: str, count: int,
              path: Path) -> list[float]:
    value = values.get(key)
    if value is None or len(value) != count or not all(map(math.isfinite, value)):
        raise ValueError(f"{path}: expected {key} with {count} finite values")
    return value


def make_rectified_camera_calibration(sequence: Path, camera: str) -> tuple[list[list[float]], list[list[float]]]:
    """Return K and T_camera_lidar for KITTI's rectified image and Velodyne."""
    cam_path = sequence / "calib_cam_to_cam.txt"
    velo_path = sequence / "calib_velo_to_cam.txt"
    cam = _read_kitti_calibration(cam_path)
    velo = _read_kitti_calibration(velo_path)
    p = _required(cam, f"P_rect_{camera[-2:]}", 12, cam_path)
    r_rect = _required(cam, "R_rect_00", 9, cam_path)
    r_velo = _required(velo, "R", 9, velo_path)
    t_velo = _required(velo, "T", 3, velo_path)

    projection = [p[0:4], p[4:8], p[8:12]]
    intrinsics = [row[:3] for row in projection]
    # P_rect_03 includes the right-camera baseline in its fourth column.
    # Convert that term to a translation before composing KITTI's documented
    # velo -> cam0 -> rectified image projection chain.
    baseline_translation = _matvec(_inverse3(intrinsics), [row[3] for row in projection])
    rect = [r_rect[0:3], r_rect[3:6], r_rect[6:9]]
    velo_rotation = [r_velo[0:3], r_velo[3:6], r_velo[6:9]]
    rotation_camera_lidar = _matmul(rect, velo_rotation)
    rotated_translation = _matvec(rect, t_velo)
    translation_camera_lidar = [rotated_translation[k] + baseline_translation[k]
                                 for k in range(3)]
    extrinsic = [rotation_camera_lidar[row] + [translation_camera_lidar[row]]
                 for row in range(3)]
    extrinsic.append([0.0, 0.0, 0.0, 1.0])
    if intrinsics[0][0] <= 0.0 or intrinsics[1][1] <= 0.0:
        raise ValueError(f"{cam_path}: invalid focal length for {camera}")
    return intrinsics, extrinsic


def _timestamp_seconds(value: str, source: Path, line_number: int) -> float:
    """Parse KITTI's UTC timestamp without discarding sub-microsecond digits."""
    match = re.fullmatch(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.(\d+))?", value.strip())
    if not match:
        raise ValueError(f"{source}:{line_number}: invalid KITTI timestamp {value!r}")
    whole = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.timezone.utc).timestamp()
    fraction = float("0." + match.group(2)) if match.group(2) else 0.0
    return whole + fraction


def read_image_records(sequence: Path, camera: str) -> list[tuple[int, str, float]]:
    camera_dir = sequence / camera
    timestamp_file = camera_dir / "timestamps.txt"
    image_dir = camera_dir / "data"
    if not timestamp_file.is_file() or not image_dir.is_dir():
        raise ValueError(f"expected {timestamp_file} and {image_dir}")
    images: dict[int, Path] = {}
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".png" or not path.stem.isdigit():
            continue
        image_id = int(path.stem)
        if image_id in images:
            raise ValueError(f"duplicate numeric image ID {image_id} in {image_dir}")
        images[image_id] = path
    timestamps = [_timestamp_seconds(line, timestamp_file, n)
                  for n, line in enumerate(timestamp_file.read_text().splitlines(), 1)
                  if line.strip()]
    ids = sorted(images)
    if len(ids) != len(timestamps):
        raise ValueError(f"{camera}: {len(ids)} PNGs but {len(timestamps)} timestamp records")
    return [(image_id, images[image_id].name, timestamps[index])
            for index, image_id in enumerate(ids)]


def _read_json_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    decoder = json.JSONDecoder()
    result: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        record, cursor = decoder.raw_decode(text, cursor)
        if not isinstance(record, dict):
            raise ValueError(f"{path}: each trajectory record must be a JSON object")
        result.append(record)
    return result


def _validate_pose(pose: Any, path: Path, index: int) -> None:
    try:
        translation = pose["translation"]
        quaternion = pose["quaternion_xyzw"]
        values = [float(x) for x in translation] + [float(x) for x in quaternion]
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"{path}: record {index} has an invalid optimized pose") from exc
    if len(translation) != 3 or len(quaternion) != 4 or not all(map(math.isfinite, values)):
        raise ValueError(f"{path}: record {index} pose must have finite translation[3] and quaternion[4]")
    norm = math.sqrt(sum(float(x) ** 2 for x in quaternion))
    if norm < 1e-12:
        raise ValueError(f"{path}: record {index} has a zero quaternion")


def read_lio_records(lidar_dir: Path, pose_field: str,
                     duplicate_policy: str = "last_revision") -> tuple[list[dict[str, Any]], int]:
    path = lidar_dir / "key_frames.jsonl"
    records = _read_json_objects(path)
    if len(records) < 2:
        raise ValueError(f"{path}: expected at least two pose records")
    for index, record in enumerate(records):
        try:
            record["timestamp"] = float(record["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: record {index} has no numeric timestamp") from exc
        if not math.isfinite(record["timestamp"]):
            raise ValueError(f"{path}: record {index} timestamp is not finite")
        if pose_field not in record:
            raise ValueError(f"{path}: record {index} has no {pose_field!r} field")
        _validate_pose(record[pose_field], path, index)
        cloud_ref = record.get("saved_frame_path")
        if cloud_ref:
            cloud = Path(cloud_ref)
            if not cloud.is_absolute():
                cloud = lidar_dir / cloud
            if not cloud.is_file():
                raise ValueError(f"{path}: record {index} point cloud is missing: {cloud}")
            record["saved_frame_path"] = str(cloud.resolve())
        elif record.get("key_frame_id") is not None:
            raise ValueError(f"{path}: record {index} has no saved_frame_path for its LiDAR cloud")
        # The C++ trajectory reader consumes lio_pose. For this KITTI preparation
        # run, intentionally promote the SLAM-optimized pose to that field.
        record["lio_pose"] = record[pose_field]
    records.sort(key=lambda item: item["timestamp"])
    unique_records: list[dict[str, Any]] = []
    duplicates_removed = 0
    index = 0
    while index < len(records):
        end = index + 1
        while end < len(records) and records[end]["timestamp"] == records[index]["timestamp"]:
            end += 1
        group = records[index:end]
        if len(group) > 1:
            same_frame = all(
                item.get("key_frame_id") == group[0].get("key_frame_id")
                and item.get("saved_frame_path") == group[0].get("saved_frame_path")
                for item in group
            )
            if duplicate_policy != "last_revision" or not same_frame:
                raise ValueError(
                    f"{path}: duplicate timestamp {group[0]['timestamp']} does not identify "
                    "a safely replaceable revision of the same key frame")
            # The file is append-only and later records are later optimized revisions.
            unique_records.append(group[-1])
            duplicates_removed += len(group) - 1
        else:
            unique_records.append(group[0])
        index = end
    return unique_records, duplicates_removed


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a readable PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def prepare(config_path: Path, start_id: int | None = None,
            end_id: int | None = None) -> tuple[Path, Path, dict[str, Any]]:
    config = json.loads(config_path.read_text())
    sequence = Path(config["sequence_directory"]).expanduser().resolve()
    lidar_dir = Path(config["lidar_slam_directory"]).expanduser().resolve()
    camera = str(config.get("camera", "image_03"))
    begin = int(config["image_id_start"] if start_id is None else start_id)
    end = int(config["image_id_end"] if end_id is None else end_id)
    offset = float(config.get("camera_time_offset_seconds", 0.0))
    outside_policy = config.get("outside_pose_range", "clip")
    pose_field = config.get("pose_field", "optimized_pose")
    duplicate_pose_policy = config.get("duplicate_pose_policy", "last_revision")
    if begin < 0 or end < begin:
        raise ValueError("image ID range must satisfy 0 <= start <= end")
    if not math.isfinite(offset):
        raise ValueError("camera_time_offset_seconds must be finite")
    if outside_policy not in {"clip", "error"}:
        raise ValueError("outside_pose_range must be 'clip' or 'error'")

    all_images = read_image_records(sequence, camera)
    available_ids = {item[0] for item in all_images}
    missing_ids = [image_id for image_id in range(begin, end + 1) if image_id not in available_ids]
    if missing_ids:
        raise ValueError(f"requested image IDs are missing (first missing ID: {missing_ids[0]})")
    requested = [item for item in all_images if begin <= item[0] <= end]
    lio_records, duplicate_poses_removed = read_lio_records(
        lidar_dir, pose_field, duplicate_pose_policy)
    pose_times = [item["timestamp"] for item in lio_records]
    valid = [item for item in requested
             if pose_times[0] <= item[2] + offset <= pose_times[-1]]
    clipped = len(requested) - len(valid)
    if clipped and outside_policy == "error":
        raise ValueError(f"{clipped} requested image(s) fall outside LIO pose time coverage")
    if not valid:
        raise ValueError("no requested image timestamps overlap the LIO pose trajectory")

    first_query, last_query = valid[0][2] + offset, valid[-1][2] + offset
    first_pose = max(0, bisect.bisect_right(pose_times, first_query) - 1)
    last_pose = min(len(lio_records) - 1, bisect.bisect_left(pose_times, last_query))
    # Keep one trajectory sample on either side for C++ interpolation.
    first_pose = max(0, first_pose - 1)
    last_pose = min(len(lio_records) - 1, last_pose + 1)
    selected_poses = lio_records[first_pose:last_pose + 1]
    if len(selected_poses) < 2:
        raise ValueError("selected image range does not have two bracketing LIO poses")

    first_image_path = sequence / camera / "data" / valid[0][1]
    width, height = _png_size(first_image_path)
    for _, filename, _ in valid[1:]:
        other_size = _png_size(sequence / camera / "data" / filename)
        if other_size != (width, height):
            raise ValueError(f"image dimensions vary inside range: {filename} is {other_size}, expected {(width, height)}")
    intrinsics, t_camera_lidar = make_rectified_camera_calibration(sequence, camera)

    prepared_root = Path(config.get("prepared_data_directory", "output/kitti_prepared"))
    if not prepared_root.is_absolute():
        prepared_root = ROOT / prepared_root
    prepared_dir = prepared_root / f"{camera}_{valid[0][0]:010d}_{valid[-1][0]:010d}"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    timestamps_path = prepared_dir / "timestamps.txt"
    with timestamps_path.open("w") as stream:
        stream.write("# image_filename timestamp_seconds (UTC Unix seconds)\n")
        for _, filename, timestamp in valid:
            stream.write(f"{filename} {timestamp:.9f}\n")
    trajectory_path = prepared_dir / "key_frames.jsonl"
    with trajectory_path.open("w") as stream:
        for record in selected_poses:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    intrinsics_path = prepared_dir / "intrinsics.txt"
    with intrinsics_path.open("w") as stream:
        stream.write("\n".join(",".join(f"{x:.12g}" for x in row) for row in intrinsics) + "\n")
    extrinsic_path = prepared_dir / "camera_lidar_extrinsic.json"
    with extrinsic_path.open("w") as stream:
        json.dump({"T_camera_lidar": t_camera_lidar}, stream, indent=2)
        stream.write("\n")
    association_path = prepared_dir / "image_lidar_association.csv"
    with association_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "image_filename", "image_timestamp", "pose_key_frame_id",
                         "pose_timestamp", "pose_minus_image_plus_offset_seconds", "lidar_cloud"])
        for image_id, filename, image_time in valid:
            query = image_time + offset
            nearest_index = min(range(len(pose_times)), key=lambda j: abs(pose_times[j] - query))
            pose = lio_records[nearest_index]
            writer.writerow([image_id, filename, f"{image_time:.9f}", pose.get("key_frame_id", ""),
                             f"{pose_times[nearest_index]:.9f}",
                             f"{pose_times[nearest_index] - query:+.9f}",
                             pose.get("saved_frame_path", "")])

    pipeline_output = Path(config.get("pipeline_output_directory", "output/kitti_cpp"))
    if not pipeline_output.is_absolute():
        pipeline_output = ROOT / pipeline_output
    template_path = ROOT / config.get("pipeline_template", "pipeline.example.yaml")
    template = template_path.read_text()
    replacements = {
        "timestamp_file": str(timestamps_path.resolve()),
        "image_directory": str((sequence / camera / "data").resolve()),
        "intrinsics_file": str(intrinsics_path.resolve()),
        "extrinsic_file": str(extrinsic_path.resolve()),
        "trajectory_file": str(trajectory_path.resolve()),
        "output_directory": str(pipeline_output.resolve()),
        "image_width": str(width),
        "image_height": str(height),
        "camera_time_offset_seconds": repr(offset),
        "maximum_images": "-1",
    }
    for key, value in replacements.items():
        pattern = rf"(?m)^(\s*){re.escape(key)}:.*$"
        rendered = _yaml_string(value) if key in {
            "timestamp_file", "image_directory", "intrinsics_file", "extrinsic_file",
            "trajectory_file", "output_directory"
        } else value
        template, count = re.subn(
            pattern, lambda match: f"{match.group(1)}{key}: {rendered}", template, count=1)
        if count != 1:
            raise ValueError(f"pipeline template must contain a top-level {key} setting")
    pipeline_config_path = prepared_dir / "pipeline.yaml"
    pipeline_config_path.write_text(template)

    summary = {
        "camera": camera,
        "requested_image_id_range_inclusive": [begin, end],
        "effective_image_id_range_inclusive": [valid[0][0], valid[-1][0]],
        "images_selected": len(valid),
        "images_clipped_outside_pose_coverage": clipped,
        "pose_field_used": pose_field,
        "trajectory_records_written": len(selected_poses),
        "duplicate_pose_revisions_removed": duplicate_poses_removed,
        "trajectory_time_range": [selected_poses[0]["timestamp"], selected_poses[-1]["timestamp"]],
        "camera_time_offset_seconds": offset,
        "image_dimensions": [width, height],
        "image_directory": str((sequence / camera / "data").resolve()),
        "lidar_slam_directory": str(lidar_dir),
        "pipeline_config": str(pipeline_config_path.resolve()),
        "pipeline_output_directory": str(pipeline_output.resolve()),
        "lidar_cloud_note": "Cloud references are retained in the association CSV and JSONL. The current visual C++ BA uses the SLAM poses but does not fuse LiDAR cloud points.",
    }
    (prepared_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return prepared_dir, pipeline_config_path, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "kitti_prepare.json",
                        help="JSON configuration containing KITTI paths and inclusive image ID range")
    parser.add_argument("--start-image-id", type=int, help="override configured inclusive start ID")
    parser.add_argument("--end-image-id", type=int, help="override configured inclusive end ID")
    parser.add_argument("--prepare-only", action="store_true", help="write inputs/config but do not run C++")
    parser.add_argument("--cpp-executable", type=Path, help="override the root build executable path")
    args = parser.parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        prepared_dir, pipeline_config, summary = prepare(
            config_path, args.start_image_id, args.end_image_id)
        print(json.dumps(summary, indent=2), flush=True)
        if args.prepare_only:
            print("Prepared only. Run: build/lio_visual_ba_pipeline --config " +
                  str(pipeline_config.resolve()))
            return 0
        config = json.loads(config_path.read_text())
        executable = args.cpp_executable or Path(config.get("cpp_executable", "build/lio_visual_ba_pipeline"))
        if not executable.is_absolute():
            executable = ROOT / executable
        if not executable.is_file():
            raise ValueError(f"C++ pipeline executable not found: {executable}; build with cmake --build build -j2")
        return subprocess.run([str(executable), "--config", str(pipeline_config.resolve())],
                              cwd=ROOT, check=False).returncode
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"KITTI preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
