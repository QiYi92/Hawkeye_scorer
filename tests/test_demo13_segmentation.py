"""Geometry, identity gaps, parser propagation and streaming output contracts."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo13.geometry import encode_rle, mask_geometry, TemporalFeatures
from demo13.parsing import PartMasks, propagate_labels, TorchScriptParser
from demo13.tracking import Instance, PrimaryTrack
from run_demo13 import arguments, run


def instance(track_id=1, x=10):
    mask = np.zeros((64, 96), bool)
    mask[15:55, x:x + 16] = True
    return Instance(track_id, .9, [x, 15, x + 16, 55], mask, 0)


class GeometryTests(unittest.TestCase):
    def test_rle_round_trip_with_holes_and_empty_masks(self):
        rng = np.random.default_rng(13)
        for mask in [np.zeros((8, 6), bool), np.ones((8, 6), bool), rng.random((13, 17)) > .5]:
            rle = encode_rle(mask)
            decoded = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"])
            np.testing.assert_array_equal(decoded.reshape(rle["size"], order="F"), mask)

    def test_pixel_geometry_and_hole_area(self):
        mask = instance().mask
        g = mask_geometry(mask, contours=True)
        self.assertEqual(g["area"], 640)
        self.assertEqual(g["bbox"], [10, 15, 26, 55])
        self.assertEqual(g["bottom_y"], 54)
        self.assertEqual(g["centroid"], [17.5, 34.5])
        self.assertAlmostEqual(g["orientation"], 90)
        mask[30:40, 15:20] = False
        g = mask_geometry(mask, contours=True)
        self.assertEqual(g["area"], 590)
        self.assertEqual(len(g["contours"]), 2)
        self.assertIsNone(mask_geometry(np.zeros((5, 5), bool)))
        self.assertIsNone(mask_geometry(np.ones((5, 5), bool))["orientation"])

    def test_velocity_gap_and_axial_smoothing(self):
        f = TemporalFeatures(.5)
        g = mask_geometry(instance().mask)
        g["orientation"] = 179
        _, v = f.update(1, 0, 0., g)
        self.assertEqual(v, [None, None])
        newer = mask_geometry(instance(x=12).mask)
        newer["orientation"] = 1
        smooth, v = f.update(1, 1, .1, newer)
        np.testing.assert_allclose(v, [20, 0])
        self.assertAlmostEqual(smooth["orientation"], 0)
        _, v = f.update(1, 3, .3, newer)
        self.assertEqual(v, [None, None])


class TrackingTests(unittest.TestCase):
    def test_confirm_hold_and_explicit_new_identity(self):
        t = PrimaryTrack(2, 2)
        a, b = instance(7), instance(9)
        self.assertEqual(t.update([a], 0), (None, "searching"))
        self.assertEqual(t.update([a], 1), (7, "tracked"))
        self.assertEqual(t.update([b], 2), (7, "lost"))
        self.assertEqual(t.update([a, b], 3), (7, "tracked"))
        self.assertEqual(t.update([b], 4), (7, "lost"))
        self.assertEqual(t.update([b], 5), (7, "lost"))
        self.assertEqual(t.update([b], 6), (9, "tracked"))

    def test_confirmation_requires_consecutive_frames(self):
        t = PrimaryTrack(2, 2)
        t.update([instance()], 0)
        t.update([], 1)
        self.assertEqual(t.update([instance()], 2), (None, "searching"))


class ParserTests(unittest.TestCase):
    def test_schedule_clipping_and_reset_on_gap(self):
        class Parser:
            calls = 0
            def parse(self, crop):
                self.calls += 1
                return np.ones(crop.shape[:2], np.uint8)
        parser = Parser()
        parts = PartMasks(parser, 3)
        frame = np.zeros((64, 96, 3), np.uint8)
        for n, expected in enumerate(["inferred", "propagated", "propagated", "inferred"]):
            labels, source, age = parts.update(frame, instance(), n)
            self.assertEqual(source, expected)
            self.assertFalse(labels[~instance().mask].any())
            self.assertEqual(age, n % 3)
        parts.update(frame, None, 4)
        self.assertEqual(parts.update(frame, instance(), 5)[1], "inferred")
        self.assertEqual(parser.calls, 3)

    def test_backward_flow_translation(self):
        rng = np.random.default_rng(2)
        prev = rng.integers(0, 256, (96, 96), dtype=np.uint8)
        curr = cv2.warpAffine(prev, np.float32([[1, 0, 3], [0, 1, 2]]), (96, 96))
        labels = np.zeros((96, 96), np.uint8)
        labels[30:50, 30:50] = 2
        result = propagate_labels(prev, curr, labels)
        self.assertGreater((result[32:52, 33:53] == 2).mean(), .95)

    def test_torchscript_model_contract(self):
        import torch
        class Tiny(torch.nn.Module):
            def forward(self, x):
                return torch.cat([x[:, :1] * 0, x[:, :1] * 0 + 1, x[:, :1] * 0], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parts.pt"
            torch.jit.trace(Tiny(), torch.zeros(1, 3, 32, 32)).save(str(path))
            result = TorchScriptParser(path, size=32).parse(np.zeros((40, 20, 3), np.uint8))
            self.assertEqual(result.shape, (40, 20))
            self.assertTrue((result == 1).all())


class PipelineTests(unittest.TestCase):
    def test_video_and_data_include_missing_frame(self):
        class Segmenter:
            target_name = "person"
            calls = 0
            def __init__(self, *args):
                pass
            def infer(self, frame):
                self.calls += 1
                background = instance(track_id=9, x=60)
                return [background] if self.calls == 3 else [instance(x=10 + self.calls), background]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((64, 96, 3), np.uint8))
            writer.release()
            args = arguments(["--video", str(video), "--output", str(root / "result"), "--confirm-frames", "1", "--selection", "largest"])
            with patch("run_demo13.YoloSegmenter", Segmenter):
                result = run(args)
                with self.assertRaises(ValueError):
                    run(args)  # never overwrite previous outputs
            records = [json.loads(line) for line in (result / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertTrue(all(len(row["tracks"]) <= 1 for row in records))
            self.assertEqual(records[0]["candidate_count"], 2)
            self.assertEqual({t["track_id"] for row in records for t in row["tracks"]}, {1})
            self.assertFalse(list((result / "masks").glob("*_id9.png")))
            self.assertEqual(records[2]["primary_status"], "lost")
            self.assertEqual(records[2]["tracks"], [])
            self.assertEqual(records[3]["tracks"][0]["velocity_px_s"], [None, None])
            mask = cv2.imread(str(result / records[0]["tracks"][0]["mask"]["path"]), 0)
            self.assertEqual(int((mask > 0).sum()), 640)
            with (result / "tracks.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[2]["status"], "lost")
            self.assertEqual(rows[0]["leg_area"], "")
            cap = cv2.VideoCapture(str(result / "video_overlay.mp4"))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            cap.release()
            metrics = json.loads((result / "metrics.json").read_text())
            self.assertEqual(metrics["primary_lost_frames"], 1)
            self.assertEqual(metrics["stop_reason"], "end_of_video")


if __name__ == "__main__":
    unittest.main()
