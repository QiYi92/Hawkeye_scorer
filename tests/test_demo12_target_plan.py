"""Geometry and playback behavior of the Demo 12 target inset."""
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from browse_multiscan_landing_review import ContactResult, ReviewWindow
from browse_target_center_video import SmoothedTarget
from target_plan_view import draw_target_plan, plan_marker_position, project_target_point


def target():
    return SmoothedTarget((400., 300.), 600., 20., None,
                          ((410., 305.), (200., 100.), 0.), 15000., .8, True, True, 0)


class TargetPlanTests(unittest.TestCase):
    def test_bullseye_is_origin_even_when_disk_center_differs(self):
        self.assertEqual(project_target_point((400, 300), target()), (0., 0.))

    def test_axes_and_outside_points_are_not_clipped(self):
        self.assertEqual(project_target_point((500, 300), target()), (1., 0.))
        self.assertEqual(project_target_point((400, 350), target()), (0., 1.))
        self.assertEqual(project_target_point((650, 300), target()), (2.5, 0.))

    def test_rotation_retains_camera_orientation_and_axis_conventions(self):
        for angle in (0, 30, 90, 145):
            t = replace(target(), disk_ellipse=((400., 300.), (200., 100.), angle))
            theta = math.radians(angle)
            point = (400 + 100 * math.cos(theta), 300 + 100 * math.sin(theta))
            np.testing.assert_allclose(project_target_point(point, t), (math.cos(theta), math.sin(theta)), atol=1e-12)
            equivalent = replace(t, disk_ellipse=((400., 300.), (100., 200.), angle + 90))
            np.testing.assert_allclose(project_target_point(point, t), project_target_point(point, equivalent), atol=1e-12)

    def test_invalid_geometry(self):
        self.assertIsNone(project_target_point((0, 0), None))
        for axes, angle in (((0, 100), 0), ((-1, 100), 0), ((float('nan'), 100), 0), ((100, 100), float('inf'))):
            self.assertIsNone(project_target_point((0, 0), replace(target(), disk_ellipse=((0, 0), axes, angle))))
        self.assertIsNone(project_target_point((float('nan'), 0), target()))

    def test_outside_marker_and_overflow_direction(self):
        position, overflow = plan_marker_position((1.1, 0))
        self.assertFalse(overflow)
        self.assertGreater(position[0], 180 + 128)
        position, overflow = plan_marker_position((-4, 2))
        self.assertTrue(overflow)
        self.assertEqual(position, (22, 277))

    def window(self, geometry=None):
        result = ContactResult(10, 1., 'left', 'toe', (450, 325), 'TARGET_DISK', 'fine', (0, 0, 100, 100), geometry)
        return ReviewWindow(Path('unused.mp4'), result, None, .75, '')

    def test_synchronization_replay_and_fixed_geometry(self):
        window = self.window(target())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch('browse_multiscan_landing_review.draw_target_plan') as draw:
            for index in (9, 10, 10, 30, 9, 10):
                window.locked_target = replace(target(), center=(0, 0))
                window._draw_plan(frame, index)
                self.assertEqual(draw.call_args.args[2], None if index < 10 else (.5, .5))

    def test_missing_geometry_and_no_result_keep_panel(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch('browse_multiscan_landing_review.draw_target_plan') as draw:
            self.window()._draw_plan(frame, 10)
            self.assertIsNone(draw.call_args.args[2])
            self.assertEqual(draw.call_args.args[3], '已确认触地，暂无靶盘定位')
            ReviewWindow(Path('unused.mp4'), None, None, .75, '')._draw_plan(frame, 0)
            self.assertEqual(draw.call_args.args[3], '未确认首次触地')

    def test_inset_stays_in_lower_left_at_multiple_resolutions(self):
        for height, width in ((360, 640), (720, 1280), (1080, 1920)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            text = Mock()
            draw_target_plan(frame, text, (4, -3), '已确认首次触地')
            labels = [call.args[1] for call in text.call_args_list]
            self.assertIn('落点超出示意范围', labels)
            self.assertIn('位置为几何估算', labels)
            self.assertEqual(len([label for label in labels if label.endswith('m')]), 10)
            self.assertTrue(np.any(frame[height // 2:, :width // 3]))
            self.assertFalse(np.any(frame[:, width // 3:]))


if __name__ == '__main__':
    unittest.main()
