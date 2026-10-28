"""Demo 11 红球测距、地面拟合与首次落地状态机测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/oak_red_ball_landing_demo.py"
SPEC = importlib.util.spec_from_file_location("demo11_red_ball_landing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def measurement(frame: int, time_s: float, height: float) -> object:
    return demo.BallMeasurement(
        frame, time_s, time_s, "2026-08-07T12:00:00+08:00", 320, 180, 1.0, .03,
        0.0, 0.0, 1.0, height, 0.0, 1.0, 0.0, .9, 1.0, 1.3, None, "waiting",
    )


class Demo11Tests(unittest.TestCase):
    def test_detects_red_circle_and_rejects_small_noise(self) -> None:
        frame = np.zeros((200, 300, 3), np.uint8)
        cv2.circle(frame, (140, 90), 24, (0, 0, 255), -1)
        cv2.circle(frame, (20, 20), 2, (0, 0, 255), -1)
        found, _ = demo.detect_red_candidates(
            frame, (0, 90, 60), (12, 255, 255), (168, 90, 60), (179, 255, 255), 100, .58
        )
        self.assertEqual(len(found), 1)
        self.assertLess(np.linalg.norm(np.asarray(found[0].center) - (140, 90)), 2)

    def test_pixel_to_camera_at_principal_point(self) -> None:
        intrinsics = demo.CameraIntrinsics(500, 500, 320, 180)
        np.testing.assert_allclose(demo.pixel_to_camera((320, 180), 2.0, intrinsics), (0, 0, 2))

    def test_dynamic_radius_and_center_depth(self) -> None:
        intrinsics = demo.CameraIntrinsics(500, 500, 320, 180)
        contour = np.empty((0, 1, 2), np.int32)
        ball = demo.BallDetection((320, 180), 25, 1900, .9, contour)
        center, radius = demo.estimate_ball_geometry(ball, 1000, intrinsics)
        self.assertAlmostEqual(radius, 25 / 475, places=6)
        self.assertAlmostEqual(center[2], 1 + radius, places=6)

    def test_skin_contact_ring_and_release_gate(self) -> None:
        frame = np.zeros((160, 200, 3), np.uint8)
        # BGR 肤色块紧贴球体右侧。
        cv2.rectangle(frame, (105, 55), (155, 110), (80, 130, 190), -1)
        cv2.circle(frame, (90, 80), 20, (0, 0, 255), -1)
        contour = np.asarray([[[70, 80]], [[90, 60]], [[110, 80]], [[90, 100]]], np.int32)
        ball = demo.BallDetection((90, 80), 20, 1200, .9, contour)
        self.assertGreater(demo.skin_contact_ratio(frame, ball, 1.75), .06)
        gate = demo.HandReleaseGate(True, release_frames=3, skin_threshold=.06)
        self.assertFalse(gate.update(.20))
        self.assertFalse(gate.update(0))
        self.assertFalse(gate.update(0))
        self.assertTrue(gate.update(0))

    def test_release_gate_never_arms_without_first_seeing_hand(self) -> None:
        gate = demo.HandReleaseGate(True, release_frames=2, skin_threshold=.06)
        for _ in range(10):
            self.assertFalse(gate.update(0))

    def test_ground_plane_fit_projection_and_coordinates(self) -> None:
        rng = np.random.default_rng(2)
        xs = rng.uniform(-1, 1, 700)
        zs = rng.uniform(.5, 4, 700)
        ys = np.full(700, .9) + rng.normal(0, .003, 700)
        points = np.column_stack((xs, ys, zs))
        points = np.vstack((points, rng.uniform((-1, -.5, .5), (1, .5, 4), (120, 3))))
        plane = demo.fit_ground_plane(points, .9, .02, .65, .1)
        self.assertIsNotNone(plane)
        assert plane is not None
        self.assertAlmostEqual(plane.camera_height_m, .9, delta=.01)
        point = np.asarray((.4, .2, 2.0))
        projected = demo.project_to_plane(point, plane)
        self.assertAlmostEqual(demo.signed_plane_distance(projected, plane), 0, places=6)
        gx, forward = demo.ground_coordinates(projected, plane)
        self.assertAlmostEqual(gx, .4, delta=.02)
        self.assertGreater(forward, 1.9)

    def test_landing_requires_airborne_descent_and_slowdown(self) -> None:
        detector = demo.LandingDetector(.08, .025, 2, .12, .10)
        heights = [.20, .14, .07, .02, .01]
        event = None
        for index, height in enumerate(heights):
            _, event = detector.update(measurement(index, index * .1, height))
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.frame, 3)  # 回溯到候选接触的第一帧。

    def test_stationary_ball_on_floor_never_triggers(self) -> None:
        detector = demo.LandingDetector(.08, .025, 2, .12, .10)
        for index in range(10):
            state, event = detector.update(measurement(index, index * .1, .01))
        self.assertIsNone(event)
        self.assertNotEqual(state, "landed")

    def test_event_is_not_replaced_by_bounce_or_roll(self) -> None:
        detector = demo.LandingDetector(.08, .025, 2, .12, .10)
        for index, height in enumerate([.2, .12, .02, .01]):
            _, event = detector.update(measurement(index, index * .1, height))
        assert event is not None
        first_frame = event.frame
        for index, height in enumerate([.10, .03, .01], start=4):
            _, event = detector.update(measurement(index, index * .1, height))
        self.assertEqual(event.frame, first_frame)


if __name__ == "__main__":
    unittest.main()
