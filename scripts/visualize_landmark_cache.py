#!/usr/bin/env python3
"""Render landmark cache tracks as per-landmark crop strips.

This script reads the binary cache format written by SIFT::save_landmarks()
and exports images similar to output/prior_pose_landmark_track_images.

Binary format (little-endian on this platform):
    int64 landmark_count
    repeated landmark_count times:
      int32 landmark_id
      uint8 optimized
      float64[3] initial_position
      float64[3] optimized_position
      int64 observation_count
      repeated observation_count times:
        int32 camera_id
        int32 keypoint_idx
        float64 pixel_x
        float64 pixel_y
        float64 depth
        float64 optimized_depth
        uint8 optimized

Usage examples:
  python3 scripts/visualize_landmark_cache.py
  python3 scripts/visualize_landmark_cache.py --min-observations 4
  python3 scripts/visualize_landmark_cache.py --max-landmarks 200 --crop-radius 32
  python3 scripts/visualize_landmark_cache.py --cache ../data/landmarks_cache.bin \
      --img-dir ../data/undistorted --timestamps ../data/timestamps.txt
"""

from __future__ import annotations

import argparse
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


@dataclass
class Observation:
    camera_id: int
    keypoint_idx: int
    pixel_x: float
    pixel_y: float
    depth: float
    optimized_depth: float
    optimized: bool


@dataclass
class Landmark:
    landmark_id: int
    optimized: bool
    initial_position: Tuple[float, float, float]
    optimized_position: Tuple[float, float, float]
    observations: List[Observation]


def _read_exact(f, nbytes: int) -> bytes:
    data = f.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"Unexpected EOF: wanted {nbytes} bytes, got {len(data)}")
    return data


def load_landmark_cache(cache_path: str) -> List[Landmark]:
    landmarks: List[Landmark] = []

    with open(cache_path, "rb") as f:
        landmark_count = struct.unpack("<q", _read_exact(f, 8))[0]

        for _ in range(landmark_count):
            landmark_id = struct.unpack("<i", _read_exact(f, 4))[0]
            optimized = struct.unpack("<B", _read_exact(f, 1))[0] != 0

            initial_position = struct.unpack("<ddd", _read_exact(f, 24))
            optimized_position = struct.unpack("<ddd", _read_exact(f, 24))

            observation_count = struct.unpack("<q", _read_exact(f, 8))[0]
            observations: List[Observation] = []

            for _ in range(observation_count):
                camera_id = struct.unpack("<i", _read_exact(f, 4))[0]
                keypoint_idx = struct.unpack("<i", _read_exact(f, 4))[0]
                pixel_x, pixel_y = struct.unpack("<dd", _read_exact(f, 16))
                depth = struct.unpack("<d", _read_exact(f, 8))[0]
                optimized_depth = struct.unpack("<d", _read_exact(f, 8))[0]
                obs_optimized = struct.unpack("<B", _read_exact(f, 1))[0] != 0

                observations.append(
                    Observation(
                        camera_id=camera_id,
                        keypoint_idx=keypoint_idx,
                        pixel_x=pixel_x,
                        pixel_y=pixel_y,
                        depth=depth,
                        optimized_depth=optimized_depth,
                        optimized=obs_optimized,
                    )
                )

            landmarks.append(
                Landmark(
                    landmark_id=landmark_id,
                    optimized=optimized,
                    initial_position=initial_position,
                    optimized_position=optimized_position,
                    observations=observations,
                )
            )

    return landmarks


def load_camera_id_to_image_map(
    timestamps_path: str,
    image_dir: str,
    max_images: int = -1,
) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    camera_id = 0

    with open(timestamps_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            image_name = parts[0]
            image_path = os.path.join(image_dir, image_name)

            # Mirror DataLoader::load_camera behavior:
            # skip missing image without advancing camera_id.
            if not os.path.exists(image_path):
                continue

            mapping[camera_id] = image_name
            camera_id += 1

            if max_images >= 0 and camera_id >= max_images:
                break

    return mapping


def crop_with_padding(image: Image.Image, cx: int, cy: int, radius: int) -> Image.Image:
    crop_size = 2 * radius + 1
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius, cy + radius

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image.width - 1, x1)
    src_y1 = min(image.height - 1, y1)

    crop = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))

    if src_x0 <= src_x1 and src_y0 <= src_y1:
        src = image.crop((src_x0, src_y0, src_x1 + 1, src_y1 + 1))
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        crop.paste(src, (dst_x0, dst_y0))

    return crop


def render_landmark_strip(
    landmark: Landmark,
    camera_to_image: Dict[int, str],
    image_dir: str,
    image_cache: Dict[int, Image.Image],
    crop_radius: int,
) -> Image.Image | None:
    crops: List[Image.Image] = []

    for obs in landmark.observations:
        image_name = camera_to_image.get(obs.camera_id)
        if image_name is None:
            continue

        if obs.camera_id not in image_cache:
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None
            image_cache[obs.camera_id] = image

        image = image_cache.get(obs.camera_id)
        if image is None:
            continue

        cx = int(round(obs.pixel_x))
        cy = int(round(obs.pixel_y))
        crop = crop_with_padding(image, cx, cy, crop_radius)

        draw = ImageDraw.Draw(crop)
        local_center = (crop_radius, crop_radius)
        r = 4
        draw.ellipse(
            [
                (local_center[0] - r, local_center[1] - r),
                (local_center[0] + r, local_center[1] + r),
            ],
            outline=(255, 255, 0),
            width=1,
        )

        label = f"cam {obs.camera_id} kp {obs.keypoint_idx}"
        draw.text((4, crop.height - 14), label, fill=(0, 255, 0))

        crops.append(crop)

    if not crops:
        return None

    strip = Image.new("RGB", (sum(c.width for c in crops), max(c.height for c in crops)), (0, 0, 0))
    x_offset = 0
    for crop in crops:
        strip.paste(crop, (x_offset, 0))
        x_offset += crop.width

    title = f"landmark {landmark.landmark_id} ({len(crops)} obs)"
    draw_strip = ImageDraw.Draw(strip)
    draw_strip.text((6, 4), title, fill=(255, 255, 0))

    return strip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="../data/landmarks_cache.bin")
    parser.add_argument("--img-dir", default="../data/undistorted")
    parser.add_argument("--timestamps", default="../data/timestamps.txt")
    parser.add_argument("--out-dir", default="../output/landmark_cache_track_images")
    parser.add_argument("--crop-radius", type=int, default=32)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--max-landmarks", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    landmarks = load_landmark_cache(args.cache)
    camera_to_image = load_camera_id_to_image_map(
        args.timestamps,
        args.img_dir,
        max_images=args.max_images,
    )

    selected = [
        lm for lm in landmarks if len(lm.observations) >= args.min_observations
    ]
    if args.max_landmarks > 0:
        selected = selected[: args.max_landmarks]

    if not selected:
        print("No landmarks matched selection criteria.")
        return

    image_cache: Dict[int, Image.Image] = {}
    written = 0

    for landmark in selected:
        strip = render_landmark_strip(
            landmark,
            camera_to_image,
            args.img_dir,
            image_cache,
            args.crop_radius,
        )
        if strip is None:
            continue

        out_path = os.path.join(args.out_dir, f"tracker_{landmark.landmark_id}.png")
        strip.save(out_path)
        written += 1

    print(
        f"Loaded landmarks: {len(landmarks)} | selected: {len(selected)} | "
        f"wrote: {written} strips to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
