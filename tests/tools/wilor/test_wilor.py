"""WiLoR's detections as one left and one right hand per frame (roboskills.tools.wilor.wilor.assign and usable)."""

import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.tools.wilor.wilor import GAP, assign, usable


def detections(rows):
    """hands.npz rows from (frame, wrist x, label right) with 20-pixel boxes and one turn."""
    frame, x, right = map(np.array, zip(*rows))
    wrist = np.c_[x, np.full(len(x), 100.0)]
    return dict(frame=frame, right=right, box=np.c_[wrist - 10, wrist + 10], keypoints_2d=wrist[:, None].repeat(21, 1),
                global_orient=np.tile(np.eye(3), (len(x), 1, 1)))  # fmt: skip


def test_a_hand_keeps_to_the_nearest_wrist_whatever_its_label():
    h = detections([(0, 0, False), (0, 100, True), (1, 2, True), (1, 101, True), (2, 99, False)])
    assert assign(h, 3) == {"Left": {0: 0, 1: 2}, "Right": {0: 1, 1: 3, 2: 4}}


def test_a_track_lost_for_longer_than_gap_is_seeded_by_the_label_again():
    h = detections([(0, 0, True), (GAP + 1, 500, True)])  # far from its last wrist, but that is forgotten
    assert assign(h, GAP + 2) == {"Left": {}, "Right": {0: 0, GAP + 1: 1}}
    h = detections([(0, 0, True), (GAP, 500, True)])  # within GAP frames the jump is too far
    assert assign(h, GAP + 1)["Right"] == {0: 0}


def test_usable_drops_the_other_chirality_and_a_flipped_turn():
    h = detections([(t, 50, True) for t in range(5)] + [(5, 50, False), (5, 300, False)])
    h["global_orient"][2] = Rotation.from_euler("x", 170, degrees=True).as_matrix()  # palm in for palm out
    assert assign(h, 6)["Right"] == {t: t for t in range(6)}  # frame 5's nearest is labelled left
    assert usable(h, 6) == {"Left": {5: 6}, "Right": {0: 0, 1: 1, 3: 3, 4: 4}}
