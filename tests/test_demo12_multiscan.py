"""Pure behavior tests for the independent Demo 12 helpers."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock

import numpy as np
from collections import deque
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from browse_multiscan_landing_review import (
    ReviewWindow,
    TARGET_LOCK_FRAMES,
    ContactResult,
    FootTrackSample,
    LandingAnalyzer,
    ScanCandidate,
    classify_contact_zone,
    fuse_scan_candidates,
    roi_from_bbox,
    should_show_contact_marker,
)
from browse_target_center_video import SmoothedTarget


def target() -> SmoothedTarget:
    return SmoothedTarget((100.0, 100.0), 600.0, 20.0, ((100.0, 100.0), (40.0, 20.0), 0.0), ((100.0, 100.0), (200.0, 100.0), 0.0), 15000.0, 0.8, True, True, 0)


class Demo12MultiscanTests(unittest.TestCase):
    def test_review_draws_metric_rings_without_contact_result(self) -> None:
        window = ReviewWindow(Path("unused.mp4"), None, None, 1.0, "")
        window.target_detector = Mock()
        window.target_detector.update.return_value = target()
        window._draw_text = Mock()
        window._draw(np.zeros((400, 600, 3), dtype=np.uint8), 0, 30, 100, False)
        labels = [call.args[1] for call in window._draw_text.call_args_list]
        self.assertEqual([label for label in labels if label.endswith("m")],
                         [f"{i * 0.2:.1f}m" for i in range(10, 0, -1)])

    def test_review_discards_locked_geometry_on_seek(self) -> None:
        window = ReviewWindow(Path("unused.mp4"), None, None, 1.0, "")
        window.target_detector = Mock()
        window.target_detector.update.return_value = target()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        for index in range(TARGET_LOCK_FRAMES):
            window._update_target(frame, index)
        self.assertIsNotNone(window.locked_target)
        window.target_detector.update.return_value = None
        detected, _ = window._update_target(frame, 0)
        self.assertIsNone(detected)
        self.assertIsNone(window.locked_target)
        window.target_detector.reset.assert_called_once()

    def test_review_discards_locked_geometry_after_camera_move(self) -> None:
        window = ReviewWindow(Path("unused.mp4"), None, None, 1.0, "")
        window.locked_target = target()
        window.camera_motion = Mock()
        window.camera_motion.update.return_value = True
        window.target_detector = Mock()
        window.target_detector.update.return_value = None
        detected, _ = window._update_target(np.zeros((400, 600, 3), dtype=np.uint8), 0)
        self.assertIsNone(detected)
        self.assertIsNone(window.locked_target)
        window.target_detector.reset.assert_called_once()

    def test_fusion_requires_two_independent_scans(self) -> None:
        self.assertIsNone(fuse_scan_candidates([ScanCandidate("A", 100, (20, 20, 80, 180), (50, 170), 0.8)], 30, 1000))

    def test_fusion_selects_three_scan_consensus(self) -> None:
        candidates = [
            ScanCandidate("A", 100, (20, 20, 80, 180), (50, 170), 0.8),
            ScanCandidate("B", 103, (22, 20, 82, 180), (52, 172), 0.9),
            ScanCandidate("C", 101, (21, 20, 81, 180), (51, 171), 0.7),
            ScanCandidate("A", 260, (400, 20, 480, 180), (440, 170), 0.95),
        ]
        result = fuse_scan_candidates(candidates, 30, 1000)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.support, 3)
        self.assertLess(result.frame_index, 120)

    def test_roi_covers_lower_body_and_stays_in_frame(self) -> None:
        self.assertEqual(roi_from_bbox((10, 20, 110, 220), (240, 160, 3)), (0, 90, 135, 240))

    def test_zone_classification(self) -> None:
        self.assertEqual(classify_contact_zone((100, 100), target()), "BULLSEYE")
        self.assertEqual(classify_contact_zone((170, 100), target()), "TARGET_DISK")
        self.assertEqual(classify_contact_zone((250, 100), target()), "OUTSIDE_TARGET")
        self.assertEqual(classify_contact_zone((100, 100), None), "UNKNOWN_TARGET")

    def test_contact_candidate_comes_from_foot_trajectory(self) -> None:
        history = deque(
            [
                FootTrackSample(100, (100, 300), "toe"),
                FootTrackSample(101, (100, 310), "toe"),
                FootTrackSample(102, (100, 321), "toe"),
                FootTrackSample(103, (100, 323), "toe"),
                FootTrackSample(104, (100, 324), "toe"),
            ],
            maxlen=6,
        )
        candidate = LandingAnalyzer._confirmed_contact(history)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.frame_index, 102)
        self.assertEqual(candidate.point, (100, 321))

    def test_continued_descent_after_candidate_is_not_contact(self) -> None:
        history = deque(
            [
                FootTrackSample(100, (100, 435)),
                FootTrackSample(101, (100, 455)),
                FootTrackSample(102, (100, 475)),
                FootTrackSample(103, (100, 479)),
                FootTrackSample(104, (100, 509)),
            ],
            maxlen=6,
        )
        self.assertIsNone(LandingAnalyzer._confirmed_contact(history))

    def test_near_ground_landmark_projects_to_sole(self) -> None:
        sample = FootTrackSample(118, (644, 513), "toe", 522.0)
        self.assertEqual(LandingAnalyzer._project_contact_point(sample), (644, 522))

    def test_small_pose_jitter_after_touchdown_is_allowed(self) -> None:
        history = deque(
            [
                FootTrackSample(116, (635, 472), "toe"),
                FootTrackSample(117, (641, 491), "toe"),
                FootTrackSample(118, (636, 505), "toe"),
                FootTrackSample(119, (635, 519), "toe"),
                FootTrackSample(120, (617, 522), "toe"),
                FootTrackSample(121, (613, 527), "toe"),
            ],
            maxlen=6,
        )
        candidate = LandingAnalyzer._confirmed_contact(history)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.frame_index, 119)

    def test_untrusted_ankle_fallback_cannot_confirm_contact(self) -> None:
        history = deque(
            [
                FootTrackSample(100, (100, 300), "toe"),
                FootTrackSample(101, (100, 315), "toe"),
                FootTrackSample(102, (100, 330), "unknown"),
                FootTrackSample(103, (100, 332), "unknown"),
                FootTrackSample(104, (100, 333), "unknown"),
            ],
            maxlen=6,
        )
        self.assertIsNone(LandingAnalyzer._confirmed_contact(history))

    def test_contact_marker_waits_for_confirmed_frame(self) -> None:
        result = ContactResult(120, 4.0, "left", "toe", (612, 519), "BULLSEYE", "fine", (0, 0, 1, 1), None)
        self.assertFalse(should_show_contact_marker(119, result))
        self.assertTrue(should_show_contact_marker(120, result))

    def test_first_visible_target_contact_handles_occluded_touchdown(self) -> None:
        history = deque(
            [FootTrackSample(120, (100, 100)), FootTrackSample(121, (101, 103)), FootTrackSample(122, (100, 102))],
            maxlen=6,
        )
        candidate = LandingAnalyzer._first_visible_target_contact(
            history,
            deque([(117, 500), (122, 530)], maxlen=6),
            target(),
            720,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.frame_index, 120)


if __name__ == "__main__":
    unittest.main()
