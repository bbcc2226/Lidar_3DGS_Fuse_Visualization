#!/usr/bin/env python3
"""Package LiDAR depth maps and render QA previews for 3DGS supervision."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def colorize(depth, valid, max_vis_depth):
    """Render valid metric depth as a colorized inverse-depth image.

    Purpose:
        Produce a QA visualization that emphasizes nearby geometry.
    Inputs:
        depth: HxW metric depth array.
        valid: HxW boolean validity mask.
        max_vis_depth: Farthest visualization depth in meters.
    Outputs:
        HxWx3 uint8 BGR image with invalid pixels set to black.
    """
    color = np.zeros((*depth.shape, 3), np.uint8)
    if not valid.any():
        return color
    # Inverse depth makes nearby geometry and alignment errors easier to see.
    inv = np.zeros_like(depth, np.float32)
    inv[valid] = 1.0 / np.maximum(depth[valid], 1e-3)
    hi = float(np.percentile(inv[valid], 99))
    lo = max(1.0 / max_vis_depth, float(np.percentile(inv[valid], 1)))
    scaled = np.clip((inv - lo) / max(hi - lo, 1e-8), 0, 1)
    color = cv2.applyColorMap(
        np.rint(255 * scaled).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    color[~valid] = 0
    return color


def confidence_map(valid):
    """Estimate confidence from local valid-depth density.

    Purpose:
        Down-weight isolated LiDAR splats without claiming learned uncertainty.
    Inputs:
        valid: HxW boolean mask of accepted LiDAR depth pixels.
    Outputs:
        HxW float32 confidence array in the range zero to one.
    """
    # Confidence reflects local LiDAR sampling support and avoids assigning high
    # weight to isolated splats. It is not a learned uncertainty estimate.
    density = cv2.boxFilter(valid.astype(np.float32), -1, (7, 7), normalize=True)
    conf = np.sqrt(np.clip(density, 0, 1)) * valid
    return conf.astype(np.float32)


def label(im, text):
    """Add a title strip to a preview image copy.

    Purpose:
        Identify panels in generated depth-supervision previews.
    Inputs:
        im: Source image array.
        text: Label text to draw at the top left.
    Outputs:
        Labeled copy of the input image.
    """
    out = im.copy()
    cv2.rectangle(out, (0, 0), (270, 38), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    """Package raw LiDAR depth into 3DGS supervision artifacts.

    Purpose:
        Validate and clean depth maps, derive masks and confidence, generate QA
        previews, and summarize per-frame coverage.
    Inputs:
        Command-line data, reconstruction, and output paths plus accepted depth
        range and preview frame indices.
    Outputs:
        Writes NPY and PNG depth, mask, confidence, preview, and JSON report
        files; prints report metadata and returns ``None``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=30.0)
    p.add_argument(
        "--sample-frames", type=int, nargs="*", default=[50, 108, 150, 250, 400, 520]
    )
    a = p.parse_args()
    for d in ("depth_npy", "depth_mm", "masks", "confidence", "previews"):
        (a.output / d).mkdir(parents=True, exist_ok=True)
    entries = []
    total_valid = 0
    total_pixels = 0
    previews = []
    stamps = [line.split() for line in open(a.data / "timestamps.txt")]
    for frame, (name, stamp) in enumerate(stamps):
        image = cv2.imread(str(a.data / "undistorted" / name), cv2.IMREAD_COLOR)
        source = a.data / "depth_maps" / (name + "_depth.tiff")
        depth = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:
            continue
        if depth.ndim != 2:
            raise ValueError(f"expected scalar depth: {source}")
        if depth.shape != image.shape[:2]:
            raise ValueError(
                f"depth/image size mismatch: {name} {depth.shape} {image.shape[:2]}"
            )
        depth = depth.astype(np.float32)
        valid = np.isfinite(depth) & (depth >= a.min_depth) & (depth <= a.max_depth)
        clean = np.where(valid, depth, 0).astype(np.float32)
        conf = confidence_map(valid)
        stem = Path(name).stem
        np.save(a.output / "depth_npy" / (stem + ".npy"), clean)
        mm = np.zeros_like(depth, np.uint16)
        mm[valid] = np.clip(np.rint(depth[valid] * 1000), 1, 65535).astype(np.uint16)
        cv2.imwrite(str(a.output / "depth_mm" / (stem + ".png")), mm)
        cv2.imwrite(
            str(a.output / "masks" / (stem + ".png")), valid.astype(np.uint8) * 255
        )
        cv2.imwrite(
            str(a.output / "confidence" / (stem + ".png")),
            np.rint(conf * 65535).astype(np.uint16),
        )
        count = int(valid.sum())
        total_valid += count
        total_pixels += valid.size
        entries.append(
            {
                "frame": frame,
                "image": name,
                "timestamp": float(stamp),
                "valid_pixels": count,
                "coverage": count / valid.size,
                "min_depth_m": float(clean[valid].min()) if count else None,
                "median_depth_m": float(np.median(clean[valid])) if count else None,
                "max_depth_m": float(clean[valid].max()) if count else None,
            }
        )
        if frame in a.sample_frames:
            color = colorize(clean, valid, a.max_depth)
            overlay = image.copy()
            overlay[valid] = np.rint(0.35 * image[valid] + 0.65 * color[valid]).astype(
                np.uint8
            )
            panel = np.hstack(
                (
                    label(image, "Original"),
                    label(color, "LiDAR depth"),
                    label(overlay, "Depth overlay"),
                )
            )
            out = a.output / "previews" / (f"{frame:06d}_{stem}_comparison.jpg")
            cv2.imwrite(str(out), panel, [cv2.IMWRITE_JPEG_QUALITY, 94])
            previews.append(str(out))
    intrinsics = a.reconstruction / "intrinsics_refined.txt"
    if not intrinsics.exists():
        intrinsics = a.reconstruction / "intrinsics_input.txt"
    if not intrinsics.exists():
        intrinsics = a.data / "intrinsics.txt"
    report = {
        "format": {
            "depth_npy": "float32 meters; invalid=0",
            "depth_mm": "uint16 millimeters; invalid=0",
            "mask": "uint8 0/255",
            "confidence": "uint16 normalized local sampling density",
        },
        "source": "LiDAR-derived camera-frame z-depth",
        "reconstruction": str(a.reconstruction),
        "poses": str(a.reconstruction / "poses_optimized_tum.txt"),
        "intrinsics": str(intrinsics),
        "frames_written": len(entries),
        "valid_pixels": total_valid,
        "mean_coverage": total_valid / max(total_pixels, 1),
        "sample_previews": previews,
        "frames": entries,
    }
    (a.output / "depth_supervision.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
