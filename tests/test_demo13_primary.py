"""Select a descending arrival, hide bystanders, and conservatively reconnect IDs."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo13.tracking import DescendingAthleteTrack, Instance


def person(track_id, x=200, y=20, width=30, height=60):
    mask = np.zeros((400, 500), bool)
    mask[y:y+height, x:x+width] = True
    return Instance(track_id, .9, [x, y, x + width, y + height], mask, 0)


class DescendingTests(unittest.TestCase):
    def test_wait_for_smaller_descending_athlete_and_keep_it_on_ground(self):
        t = DescendingAthleteTrack()
        bystander = person(1, 60, 170, 80, 200)
        for n in range(15):
            self.assertEqual(t.update([bystander], n), (None, "searching"))
        for n in range(15, 30):
            identity, status = t.update([bystander, person(5, y=20+6*(n-15))], n)
        self.assertEqual((identity, status), (5, "tracked"))
        for n in range(30, 40):
            self.assertEqual(t.update([bystander, person(5, y=104)], n), (5, "tracked"))

    def test_ground_jitter_and_slow_walk_never_qualify(self):
        t = DescendingAthleteTrack()
        for n in range(60):
            self.assertEqual(t.update([person(1, y=150+n//2), person(2, x=300, y=170+n%3)], n),
                             (None, "searching"))

    def test_reconnect_new_tracker_id_with_one_logical_identity(self):
        t = DescendingAthleteTrack(lost_buffer=10)
        bystander = person(1, 80, 150)
        for n in range(10):
            t.update([person(5, y=20+n*6), bystander], n)
        self.assertEqual(t.update([bystander], 10), (5, "lost"))
        self.assertEqual(t.update([person(8, y=86), bystander], 11), (5, "lost"))
        self.assertEqual(t.update([person(8, y=92), bystander], 12), (5, "tracked"))
        self.assertEqual(t.source_track_id, 8)
        self.assertEqual(t.reassociations, 1)

    def test_do_not_switch_to_bystander_or_new_arrival_after_timeout(self):
        t = DescendingAthleteTrack(lost_buffer=5)
        for n in range(10):
            t.update([person(5, y=20+n*6), person(1, x=60, y=150)], n)
        for n in range(10, 40):
            self.assertEqual(t.update([person(1, x=60, y=150), person(9, x=350, y=n*6)], n), (5, "lost"))
        self.assertEqual(t.reassociations, 0)

    def test_ambiguous_arrivals_are_not_arbitrarily_selected(self):
        t = DescendingAthleteTrack()
        for n in range(12):
            self.assertEqual(t.update([person(5, y=20+n*6), person(6, x=350, y=20+n*6)], n),
                             (None, "searching"))
        self.assertEqual(t.reason, "ambiguous_descending_candidates")

    def test_reassociation_rejects_different_appearance(self):
        t = DescendingAthleteTrack(lost_buffer=10)
        green = np.full((400, 500, 3), (0, 255, 0), np.uint8)
        red = np.full((400, 500, 3), (0, 0, 255), np.uint8)
        for n in range(10):
            t.update([person(5, y=20+n*6)], n, green)
        t.update([person(8, y=80)], 10, red)
        self.assertEqual(t.update([person(8, y=86)], 11, red), (5, "lost"))


if __name__ == "__main__":
    unittest.main()
