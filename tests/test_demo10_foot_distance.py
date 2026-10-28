"""Demo 10 脚部双目测距核心逻辑测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/oak_live_full_pose_tracking_demo.py"
SPEC = importlib.util.spec_from_file_location("demo10_foot_distance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


class Demo10FootDistanceTests(unittest.TestCase):
    def test_camera_distance_at_principal_point_equals_depth(self) -> None:
        intrinsics = demo.CameraIntrinsics(fx=500, fy=500, cx=320, cy=180)
        self.assertAlmostEqual(demo.camera_distance_m((320, 180), 2000, intrinsics), 2.0)

    def test_camera_distance_uses_xyz_range(self) -> None:
        intrinsics = demo.CameraIntrinsics(fx=100, fy=100, cx=0, cy=0)
        # 点 (100, 0)、Z=1m 对应相机坐标 (1, 0, 1)，直线距离为 sqrt(2)。
        self.assertAlmostEqual(demo.camera_distance_m((100, 0), 1000, intrinsics), np.sqrt(2))

    def test_depth_sample_uses_valid_median_and_ratio(self) -> None:
        depth = np.zeros((5, 5), dtype=np.uint16)
        depth[1:4, 1:4] = np.asarray(
            [[1000, 1010, 0], [990, 1000, 9000], [1000, 1020, 0]], dtype=np.uint16
        )
        sample = demo.sample_depth(depth, (2, 2), 3, 200, 8000)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.median_mm, 1000)
        self.assertAlmostEqual(sample.valid_ratio, 6 / 9)

    def test_primary_target_locks_largest_person(self) -> None:
        selector = demo.PrimaryTargetSelector(lost_frames=2)
        small = demo.TrackedPerson(1, (0, 0, 50, 100), 0.9)
        large = demo.TrackedPerson(2, (0, 0, 100, 200), 0.8)
        self.assertEqual(selector.select([small, large]).track_id, 2)
        # 即使另一个目标后来更大，只要 ID 2 仍存在，就保持锁定。
        larger_other = demo.TrackedPerson(1, (0, 0, 200, 300), 0.9)
        self.assertEqual(selector.select([larger_other, large]).track_id, 2)
        self.assertIsNone(selector.select([larger_other]))
        self.assertIsNone(selector.select([larger_other]))
        self.assertEqual(selector.select([larger_other]).track_id, 1)

    def test_auto_selects_only_valid_foot_candidate(self) -> None:
        pose_points = [(0, 0)] * 33
        pose_points[demo.LEFT_TOE] = (20, 20)
        pose_points[demo.LEFT_HEEL] = (10, 10)
        visibility = [0.0] * 33
        visibility[demo.LEFT_TOE] = 0.95
        visibility[demo.LEFT_HEEL] = 0.90
        pose = demo.MediaPipePose(tuple(pose_points), tuple(visibility))
        depth = np.zeros((40, 40), dtype=np.uint16)
        depth[17:24, 17:24] = 1500  # 只有 toe 周围存在有效深度。
        estimator = demo.FootDistanceEstimator(5, 200, 8000, 0.30, 5, 0.60)
        result = estimator.estimate(
            pose, (40, 40, 3), depth,
            demo.CameraIntrinsics(100, 100, 20, 20), 0.45,
        )
        self.assertIsNotNone(result["left"])
        assert result["left"] is not None
        self.assertEqual(result["left"].part, "toe")
        self.assertIsNone(result["right"])

    def test_first_jump_landing_requires_takeoff_then_return(self) -> None:
        detector = demo.FirstJumpLandingDetector(
            takeoff_ratio=0.10, landing_ratio=0.04,
            takeoff_frames=2, landing_frames=2, baseline_frames=3,
        )
        intrinsics = demo.CameraIntrinsics(100, 100, 50, 50)

        def feet(y: int):
            return {
                "left": demo.FootMeasurement("left", "toe", (45, y), 1000, 1, 1, 1),
                "right": demo.FootMeasurement("right", "toe", (55, y), 1000, 1, 1, 1),
            }

        for index, y in enumerate((100, 101, 100, 78, 75, 72, 85, 98)):
            self.assertIsNone(detector.update(feet(y), 200, index / 30, intrinsics, 7))
        event = detector.update(feet(100), 200, 8 / 30, intrinsics, 7)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertAlmostEqual(event.timestamp_s, 7 / 30)
        self.assertEqual(event.point, (45, 98))
        self.assertEqual(detector.state, "landed")

    def test_single_visible_foot_cannot_trigger_jump(self) -> None:
        detector = demo.FirstJumpLandingDetector(baseline_frames=1, takeoff_frames=1)
        intrinsics = demo.CameraIntrinsics(100, 100, 50, 50)
        one_foot = {
            "left": demo.FootMeasurement("left", "toe", (45, 50), 1000, 1, 1, 1),
            "right": None,
        }
        for index in range(10):
            self.assertIsNone(detector.update(one_foot, 200, index / 30, intrinsics, 1))
        self.assertNotEqual(detector.state, "airborne")

    def test_pose_drives_jump_even_when_depth_is_temporarily_missing(self) -> None:
        detector = demo.FirstJumpLandingDetector(
            takeoff_ratio=0.05, landing_ratio=0.02,
            takeoff_frames=2, landing_frames=2, baseline_frames=2,
        )
        intrinsics = demo.CameraIntrinsics(100, 100, 50, 50)

        def pose_at(y: int):
            points = [(0, 0)] * 33
            visibility = [0.0] * 33
            for index, x in ((demo.LEFT_TOE, 45), (demo.RIGHT_TOE, 55)):
                points[index] = (x, y)
                visibility[index] = 0.9
            return demo.MediaPipePose(tuple(points), tuple(visibility))

        empty = {"left": None, "right": None}
        for index, y in enumerate((100, 100, 85, 82, 75, 90, 98)):
            detector.update(empty, 200, index / 30, intrinsics, 1, pose_at(y), 0.45)
        self.assertEqual(detector.state, "airborne")
        measured = {
            "left": demo.FootMeasurement("left", "toe", (45, 100), 1000, 1, 0.9, 1),
            "right": None,
        }
        event = detector.update(measured, 200, 7 / 30, intrinsics, 1, pose_at(100), 0.45)
        self.assertIsNotNone(event)


if __name__ == "__main__":
    unittest.main()
