"""Planted feet in retargeting targets (roboskills.tools.robokit.feet), shared by motion_gen's characters and robots."""

import numpy as np

from roboskills.tools.robokit import feet


def test_a_planted_foot_is_held_where_it_landed_on_the_floor():
    frames = 12
    sole = np.array([[-0.05, 0, -0.02], [0.05, 0, -0.02]])  # a flat sole 2 cm under its body's origin
    soles = feet.Soles((0, 0, 1, 1), (sole, sole, sole, sole), np.full(4, 0.02))
    turns = np.tile(np.eye(3), (frames, 2, 1, 1))
    positions = np.zeros((frames, 2, 3))
    positions[:, 0] = np.c_[np.linspace(0, 0.11, frames), np.zeros(frames), np.full(frames, 0.05)]  # sliding, 3 cm up
    positions[:, 1] = [0.0, -0.2, 0.0]  # never planted, 2 cm into the floor
    down = np.zeros((frames, 4), bool)
    down[:, :2] = True
    out = feet.plant(soles, turns, positions, down, 30.0)
    assert np.allclose(out[:, 0, :2], [0, 0]) and np.allclose(out[:, 0, 2], 0.02)  # held, standing on the floor
    assert np.allclose(out[:, 1, 2], 0.02)  # no sole below the floor
    assert np.allclose(feet.lowest(soles, turns, out)[:, :2], 0)


def test_down_reads_gvhmrs_still_feet_and_drops_blips():
    still = np.zeros((10, 4))
    still[2:8, 0], still[5, 1] = 0.9, 0.9  # a one-frame blip is not a stance
    planted = feet.down(np.zeros((10, 4, 3)), 30.0, np.ones((10, 4, 3)), still)
    assert planted[:, 0].tolist() == [False] * 2 + [True] * 6 + [False] * 2 and not planted[:, 1].any()
