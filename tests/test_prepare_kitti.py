#!/usr/bin/env python3
"""Tests for KITTI data preparation and calibration conversion.

Purpose:
    Verify calibration composition, image-range selection, optimized LIO pose
    handling, duplicate revisions, and pose-coverage policies.
Inputs:
    Synthetic KITTI images, timestamps, calibration, LIO records, and pipeline
    configurations created in temporary directories by the test fixtures.
Outputs:
    Reports assertions through ``unittest`` and leaves no persistent test data.
"""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "kitti"))

from prepare_kitti_pipeline import make_rectified_camera_calibration, prepare


def write_png(path, width=100, height=80):
    """Write the minimum PNG header needed by the preparer's size reader.

    Purpose:
        Create a lightweight image fixture without requiring an image library.
    Inputs:
        path: Destination path for the synthetic PNG.
        width: Width encoded in the PNG IHDR chunk.
        height: Height encoded in the PNG IHDR chunk.
    Outputs:
        Writes a PNG header to ``path`` and returns ``None``.
    """
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + bytes([8, 2, 0, 0, 0])
    )


def write_calibration(sequence):
    """Write compact KITTI camera and Velodyne calibration fixtures.

    Purpose:
        Supply known intrinsics, projection baseline, rotation, and translation.
    Inputs:
        sequence: KITTI sequence directory that receives the calibration files.
    Outputs:
        Writes ``calib_cam_to_cam.txt`` and ``calib_velo_to_cam.txt`` and
        returns ``None``.
    """
    (sequence / "calib_cam_to_cam.txt").write_text(
        "calib_time: 15-Mar-2012 11:45:23\n"
        "R_rect_00: 1 0 0 0 1 0 0 0 1\n"
        "P_rect_03: 100 0 50 -10 0 100 40 0 0 0 1 0\n"
    )
    (sequence / "calib_velo_to_cam.txt").write_text(
        "R: 1 0 0 0 1 0 0 0 1\n"
        "T: 1 2 3\n"
    )


class KittiPrepareTest(unittest.TestCase):
    """Verify calibration conversion and prepared KITTI dataset behavior.

    Purpose:
        Exercise range selection, optimized-pose promotion, duplicate handling,
        pose-coverage policy, and generated configuration metadata.
    Inputs:
        Each test constructs an isolated temporary KITTI/LIO fixture.
    Outputs:
        Assertions report regressions through the ``unittest`` framework.
    """

    def test_rectified_camera_transform_includes_projection_baseline(self):
        """Verify the rectified transform includes the camera projection baseline.

        Purpose:
            Confirm camera-03 intrinsics and the projection offset are composed
            with the Velodyne-to-camera translation.
        Inputs:
            A temporary sequence containing deterministic calibration files.
        Outputs:
            Returns ``None`` after asserting the expected matrices.
        """
        with tempfile.TemporaryDirectory() as temp:
            sequence = Path(temp)
            write_calibration(sequence)
            intrinsics, transform = make_rectified_camera_calibration(
                sequence, "image_03"
            )

            self.assertEqual(
                intrinsics,
                [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
            )
            self.assertEqual(transform[0][3], 0.9)
            self.assertEqual(transform[1][3], 2.0)
            self.assertEqual(transform[2][3], 3.0)

    def _make_fixture(self, base, times, pose_times):
        """Create a synthetic KITTI sequence, LIO result, and pipeline config.

        Purpose:
            Share a deterministic filesystem fixture across preparation tests.
        Inputs:
            base: Empty temporary directory used as the fixture root.
            times: Integer second values used for image timestamps.
            pose_times: Epoch timestamps assigned to LIO pose records.
        Outputs:
            Tuple ``(config_path, config)`` containing the written JSON path and
            its mutable dictionary representation.
        """
        sequence = base / "sequence"
        image_data = sequence / "image_03" / "data"
        image_data.mkdir(parents=True)
        write_calibration(sequence)

        timestamp_text = "".join(
            f"2011-10-03 12:55:{second:02d}.000000000\n" for second in times
        )
        (sequence / "image_03" / "timestamps.txt").write_text(timestamp_text)
        for image_id in range(len(times)):
            write_png(image_data / f"{image_id:010d}.png")

        lidar = base / "lidar"
        clouds = lidar / "LIO_results"
        clouds.mkdir(parents=True)
        poses = []
        for key_frame_id, timestamp in enumerate(pose_times):
            cloud_name = f"c{key_frame_id}.ply"
            (clouds / cloud_name).write_text("ply\n")
            optimized_pose = {
                "translation": [key_frame_id, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            }
            poses.append(
                {
                    "timestamp": timestamp,
                    "optimized_pose": optimized_pose,
                    "lio_pose": {
                        "translation": [99, 0, 0],
                        "quaternion_xyzw": [0, 0, 0, 1],
                    },
                    "key_frame_id": key_frame_id,
                    "saved_frame_path": f"./LIO_results/{cloud_name}",
                }
            )
        (lidar / "key_frames.jsonl").write_text(
            "\n".join(json.dumps(pose) for pose in poses) + "\n"
        )

        template = base / "template.yaml"
        template.write_text(
            "inputs:\n"
            "  timestamp_file: x\n"
            "  image_directory: x\n"
            "  intrinsics_file: x\n"
            "  extrinsic_file: x\n"
            "  trajectory_file: x\n"
            "output_directory: x\n"
            "image_width: 1\n"
            "image_height: 1\n"
            "camera_time_offset_seconds: 0\n"
            "maximum_images: -1\n"
        )
        config_path = base / "config.json"
        config = {
            "sequence_directory": str(sequence),
            "lidar_slam_directory": str(lidar),
            "camera": "image_03",
            "image_id_start": 0,
            "image_id_end": len(times) - 1,
            "prepared_data_directory": str(base / "prepared"),
            "pipeline_output_directory": str(base / "result"),
            "pipeline_template": str(template),
        }
        config_path.write_text(json.dumps(config))
        return config_path, config

    def test_inclusive_range_uses_optimized_pose_and_generates_yaml(self):
        """Verify inclusive image selection, pose promotion, and YAML metadata.

        Purpose:
            Ensure selected images use optimized LIO poses and actual PNG size.
        Inputs:
            A three-image fixture restricted to image IDs one and two.
        Outputs:
            Returns ``None`` after asserting generated records and configuration.
        """
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            config_path, _ = self._make_fixture(
                base,
                [34, 35, 36],
                [1317646533.5, 1317646535, 1317646536, 1317646537],
            )
            config = json.loads(config_path.read_text())
            config.update(
                image_id_start=1,
                image_id_end=2,
                outside_pose_range="error",
            )
            config_path.write_text(json.dumps(config))

            output, pipeline_yaml, summary = prepare(config_path)

            self.assertEqual(summary["effective_image_id_range_inclusive"], [1, 2])
            self.assertEqual(summary["images_selected"], 2)
            rows = [
                json.loads(line)
                for line in (output / "key_frames.jsonl").read_text().splitlines()
            ]
            translation_by_time = {
                round(row["timestamp"]): row["lio_pose"]["translation"][0]
                for row in rows
            }
            self.assertEqual(translation_by_time[1317646535], 1)
            self.assertEqual(translation_by_time[1317646536], 2)
            self.assertIn("0000000001.png", (output / "timestamps.txt").read_text())

            yaml_text = pipeline_yaml.read_text()
            expected_image_directory = base / "sequence" / "image_03" / "data"
            self.assertIn(
                f'  image_directory: "{expected_image_directory}"',
                yaml_text,
            )
            self.assertIn("image_width: 100", yaml_text)
            self.assertIn("image_height: 80", yaml_text)

            association_text = (output / "image_lidar_association.csv").read_text()
            self.assertTrue(
                "cloud_1" in association_text or "c1.ply" in association_text,
                association_text,
            )

    def test_duplicate_same_frame_timestamp_uses_last_optimized_revision(self):
        """Verify duplicate revisions select the last unambiguous optimized pose.

        Purpose:
            Confirm same-frame revisions are deduplicated while conflicting key
            frame identities at one timestamp are rejected.
        Inputs:
            A fixture with an appended revision of one LIO record.
        Outputs:
            Returns ``None`` after asserting replacement and validation behavior.
        """
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            config_path, config = self._make_fixture(
                base,
                [34, 35, 36],
                [1317646533.5, 1317646535, 1317646536, 1317646537],
            )
            lidar = Path(config["lidar_slam_directory"])
            source = lidar / "key_frames.jsonl"
            records = [
                json.loads(line) for line in source.read_text().splitlines()
            ]
            revision = json.loads(json.dumps(records[2]))
            revision["optimized_pose"]["translation"][0] = 42
            records.append(revision)
            source.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n"
            )

            output, _, summary = prepare(config_path)

            self.assertEqual(summary["duplicate_pose_revisions_removed"], 1)
            selected = [
                json.loads(line)
                for line in (output / "key_frames.jsonl").read_text().splitlines()
            ]
            revised = [
                row for row in selected if round(row["timestamp"]) == 1317646536
            ]
            self.assertEqual(len(revised), 1)
            self.assertEqual(revised[0]["lio_pose"]["translation"][0], 42)

            revision["key_frame_id"] = 999
            records[-1] = revision
            source.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "same key frame"):
                prepare(config_path)

    def test_pose_range_clips_or_errors_explicitly(self):
        """Verify out-of-coverage images are clipped or rejected by policy.

        Purpose:
            Exercise both supported behaviors for image timestamps beyond the
            available LIO pose interval.
        Inputs:
            A three-image fixture covered by only two LIO timestamps.
        Outputs:
            Returns ``None`` after asserting clip metrics and the error policy.
        """
        with tempfile.TemporaryDirectory() as temp:
            config_path, config = self._make_fixture(
                Path(temp),
                [33, 34, 35],
                [1317646534, 1317646535],
            )

            _, _, summary = prepare(config_path)

            self.assertEqual(summary["images_selected"], 2)
            self.assertEqual(summary["images_clipped_outside_pose_coverage"], 1)

            config["outside_pose_range"] = "error"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "fall outside"):
                prepare(config_path)


if __name__ == "__main__":
    unittest.main()
