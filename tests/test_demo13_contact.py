"""First-contact estimates require fresh masks, descent and independent ground evidence."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo13.contact import FirstFootContact, GroundReference, foot_bottom_points
from demo13.parsing import PartMasks
from demo13.tracking import Instance
from run_demo13 import arguments


class Ground:
    camera_ok = True
    camera_moved = False
    def update(self, frame, frame_id, athlete_mask=None):
        return self.summary()
    def supports(self, point):
        return self.camera_ok and abs(point[1]-200) <= 3
    def summary(self):
        return {"available": True, "camera_stable": self.camera_ok, "camera_moved": self.camera_moved}


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.detector = FirstFootContact(30, ground_line=(0, 200, 300, 200))
        self.detector.ground = Ground()
        self.frame = np.zeros((240, 320, 3), np.uint8)
        self.person = Instance(5, .9, [80, 40, 140, 220], np.ones((240, 320), bool), 0)

    def update(self, n, bottom, source="inferred"):
        labels = np.zeros((240, 320), np.uint8)
        if bottom is not None:
            labels[bottom-8:bottom+1, 100:116] = 2
        return self.detector.update(self.frame, n, n/30, self.person, labels, source)

    def descend(self):
        for n in range(11):
            self.update(n, 100+n*10)

    def test_backdates_contact_and_latches_once(self):
        self.descend()
        self.assertIsNone(self.update(11, 200)["event"])
        self.assertIsNone(self.update(12, 200)["event"])
        result = self.update(13, 200)
        event = result["event"]
        self.assertEqual(event["frame_id"], 10)
        self.assertEqual(event["confirmation_frame"], 13)
        self.assertEqual(event["point"][1], 200)
        self.assertTrue(event["is_estimate"])
        self.assertFalse(event["first_contact_guaranteed"])
        self.update(14, 180)
        self.assertEqual(self.detector.event, event)

    def test_stationary_feet_and_continued_descent_do_not_trigger(self):
        for n in range(20):
            self.update(n, 200)
        self.assertIsNone(self.detector.event)
        self.setUp()
        for n in range(13):
            self.update(n, 100+n*10)
        self.assertIsNone(self.detector.event)

    def test_mask_gap_during_candidate_prevents_later_false_first_contact(self):
        self.descend()
        self.update(11, 200)
        self.update(12, None)
        for n in range(13, 22):
            result = self.update(n, 200)
        self.assertIsNone(result["event"])
        self.assertEqual(result["reason"], "earlier_contact_candidate_obscured")

    def test_propagated_masks_cannot_confirm(self):
        self.descend()
        for n in range(11, 20):
            self.update(n, 200, "propagated")
        self.assertIsNone(self.detector.event)

    def test_ground_evidence_and_camera_stability_required(self):
        self.detector.ground.supports = lambda point: False
        self.descend()
        for n in range(11, 20):
            self.update(n, 200)
        self.assertIsNone(self.detector.event)
        self.setUp()
        self.descend()
        self.detector.ground.camera_ok = False
        self.detector.ground.camera_moved = True
        self.assertEqual(self.update(11, 200)["reason"], "camera_moved")

    def test_components_do_not_invent_midpoint_between_shoes(self):
        labels = np.zeros((240, 320), np.uint8)
        labels[180:201, 50:70] = 2
        labels[170:201, 110:130] = 2
        labels[210, 190] = 2  # tiny speck must not become the sole
        points = foot_bottom_points(labels)
        self.assertEqual(len(points), 2)
        for p in points:
            x, y = p["point"]
            self.assertEqual(labels[y, x], 2)
            self.assertEqual(y, 200)
            self.assertFalse(70 <= x < 110)

    def test_camera_movement_is_latched_and_local_line_not_extended(self):
        g = GroundReference((20, 200, 200, 200))
        rng = np.random.default_rng(12)
        frame = rng.integers(0, 255, (240, 320, 3), np.uint8)
        g.update(frame, 0)
        g.update(frame, 1)
        self.assertTrue(g.supports((100, 200)))
        self.assertFalse(g.supports((250, 200)))
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 8], [0, 1, 0]]), (320, 240))
        g.update(moved, 2)
        g.update(moved, 3)
        self.assertTrue(g.camera_moved)
        self.assertFalse(g.supports((100, 200)))

    def test_force_inference_overrides_part_interval(self):
        class Parser:
            calls = 0
            def parse(self, crop):
                self.calls += 1
                return np.zeros(crop.shape[:2], np.uint8)
        parser = Parser()
        parts = PartMasks(parser, 3)
        parts.update(self.frame, self.person, 0)
        self.assertEqual(parts.update(self.frame, self.person, 1, force_inference=True)[1], "inferred")
        self.assertEqual(parser.calls, 2)

    def test_boolean_switch_and_dependency_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory)/"video.mp4"
            video.touch()
            self.assertFalse(arguments(["--video", str(video), "--no-first-contact"]).first_contact)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(["--video", str(video), "--first-contact"])


if __name__ == "__main__":
    unittest.main()
