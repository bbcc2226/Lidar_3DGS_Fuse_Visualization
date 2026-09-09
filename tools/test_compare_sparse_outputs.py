import json
import tempfile
import unittest
from pathlib import Path

from compare_sparse_outputs import compare


class CompareSparseOutputsTest(unittest.TestCase):
    def test_reports_pose_and_metric_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_output = root / "python"
            cpp_output = root / "cpp"
            python_output.mkdir()
            cpp_output.mkdir()
            (python_output / "poses_optimized_tum.txt").write_text(
                "0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n")
            (cpp_output / "poses_optimized_tum.txt").write_text(
                "0 0 0 0 0 0 0 1\n1 1.01 0 0 0 0 0 1\n")
            (python_output / "evaluation.json").write_text(json.dumps(
                {"landmarks": 100, "observations": 300, "reprojection_p90_px": 1.0}))
            (cpp_output / "evaluation.json").write_text(json.dumps(
                {"landmarks": 102, "observations": 303, "reprojection_p90_px": 1.1}))
            report = compare(python_output, cpp_output, {
                "translation_rmse_m": 0.05,
                "rotation_rmse_deg": 1.0,
                "landmark_relative": 0.1,
                "observation_relative": 0.1,
                "reprojection_p90_delta_px": 0.25,
            })
            self.assertTrue(report["pass"])
            self.assertAlmostEqual(report["pose_comparison"]["translation_max_m"], 0.01)
            self.assertEqual(report["cpp_minus_python_metrics"]["landmarks"], 2)


if __name__ == "__main__":
    unittest.main()
