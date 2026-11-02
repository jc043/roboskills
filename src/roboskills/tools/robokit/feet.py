"""Planted feet in the retargeting targets, read off the person, for any body RoboKit retargets onto (motion_gen's
characters, `humanoid`'s robots).

`level(skin, fps)`: [T] the floor under the person each frame, from its posed skin [T, V, 3] (z-up): the lowest point,
its 1 s rolling minimum, smoothed. GVHMR's world drifts up and down by tens of centimetres over a clip; take it off
every height first, or `down` misses the frames the floor has risen on.
`down(points, fps, seen, still=None)`: [T, 4] which of the person's sole joints (`points` [T, 4, 3], `SOLE_JOINTS`) are
planted for 3 frames or more: those GVHMR predicts from the picture stand still (``still``, its `feet_still`, over a
half), else those within 3 cm of standing height and slower than 0.2 m/s across the view (``seen``, in its camera).
`plant(soles, turns, positions, down, fps)`: the target positions of the bodies carrying the `Soles` with each planted
foot held where it landed and its lowest sole point on the floor, no sole point below it (the bodies turned as
``turns``), the corrections blended between stances; the body keeps its travel. `lift(up, low)`: how far to ask
each sole above that target again after a solve left it at `low`: a solver that cannot reach a target (a joint range,
a leg without an ankle roll) sinks the sole. `step(down, miss, fps)`: a foot the legs could not keep on its spot lifted there, to land again.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d


SOLE_JOINTS = ("left_ankle", "left_foot", "right_ankle", "right_foot")  # SMPL-X's heels and toes, `down`'s order
NEAR_M, STILL_MPS, EASE_S = 0.03, 0.2, 0.15  # planted: this near its standing height, this still; a toe eases in
STEP_S = 0.2  # a foot that cannot hold its spot steps: lifted this long
SINK_M = 0.005  # a solved sole may sink this far, and its lift settle within this


@dataclass(frozen=True)
class Soles:
    """The four soles (left heel and toe, right heel and toe): ``index`` the body (a bone, a link) carrying each,
    ``points`` its sole surface [P, 3] in that body's frame from its origin, ``rest`` [4] the body's height standing
    flat on the floor."""

    index: tuple[int, ...]
    points: tuple[np.ndarray, ...]
    rest: np.ndarray


def lowest(soles: Soles, turns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """[T, 4] each sole's lowest point above the floor, for the bones' turns [T, J, 3, 3] and positions [T, J, 3]."""
    out = np.empty((len(turns), len(soles.index)))
    for k, (j, points) in enumerate(zip(soles.index, soles.points)):
        out[:, k] = (np.einsum("tij,pj->tpi", turns[:, j], points)[..., 2] + positions[:, j, None, 2]).min(1)
    return out


def level(skin: np.ndarray, fps: float) -> np.ndarray:
    floor = minimum_filter1d(skin[..., 2].min(1), max(round(fps), 1), mode="nearest")
    return uniform_filter1d(floor, max(round(fps / 4), 1), mode="nearest")  # within the minimum's reach: never above


def _steady(out: np.ndarray) -> np.ndarray:
    for column in range(out.shape[1]):  # blips under 3 frames dropped, then lifts under 3 frames bridged
        for value in (True, False):
            flips = np.flatnonzero(np.diff(np.r_[0, (out[:, column] == value).astype(int), 0]))
            for a, b in zip(flips[::2], flips[1::2]):
                if b - a < 3 and (value or 0 < a and b < len(out)):
                    out[a:b, column] = not value
    return out


def down(points: np.ndarray, fps: float, seen: np.ndarray, still: np.ndarray | None = None) -> np.ndarray:
    if still is not None:
        return _steady(still > 0.5)
    height = points[..., 2] - np.percentile(points[..., 2], 5, axis=0)
    velocity = np.gradient(seen, axis=0) * fps
    ray = seen / np.linalg.norm(seen, axis=-1, keepdims=True)
    velocity -= (velocity * ray).sum(-1, keepdims=True) * ray  # across the view: less its part along the ray
    return _steady((height < NEAR_M) & (np.linalg.norm(velocity, axis=-1) < STILL_MPS))


def step(down: np.ndarray, miss: np.ndarray, fps: float) -> np.ndarray:
    """`down` lifted for `STEP_S` where each stance first misses (`miss` [T, 2]: the legs could not reach its spot)."""
    out, swing = down.copy(), round(STEP_S * fps)
    for foot in range(2):
        starts = np.flatnonzero((held := out[:, 2 * foot : 2 * foot + 2].any(1)) & ~np.r_[False, held[:-1]])
        for a, b in zip(starts, [*starts[1:], len(held)]):
            if (late := np.flatnonzero(miss[a:b, foot])).size:
                out[a + late[0] : a + late[0] + swing, 2 * foot : 2 * foot + 2] = False
    return out


def lift(up: np.ndarray, low: np.ndarray) -> np.ndarray:
    """From the last ask `up` and the solved soles' lowest points `low` (same shape): up by what a sole sank, down by
    what it floats, never below its target; as continuous over the frames as the solve it measures."""
    return np.maximum(up - low, 0.0)


def plant(soles: Soles, turns: np.ndarray, positions: np.ndarray, down: np.ndarray, fps: float) -> np.ndarray:
    out, frames, bones = positions.copy(), np.arange(len(positions)), list(soles.index)
    for side in (0, 2):  # a foot: its heel and toe bones, moved up or down together
        ks, columns = [bones[side], bones[side + 1]], [side, side + 1]
        low = lowest(soles, turns, out)[:, columns].min(1)
        planted, flat = down[:, columns].any(1), down[:, columns].all(1)
        rest = soles.rest[columns]
        offset = np.where(flat, rest[0] - out[:, ks[0], 2], -low)  # heel and toe both down: the foot stands flat
        lift = np.interp(frames, frames[planted], offset[planted]) if planted.any() else np.zeros(len(out))
        out[:, ks, 2] += np.maximum(lift, -low)[:, None]  # and never a sole point below the floor
        if flat.any():  # the toe down to its rest height on flat frames, eased in and out, not below the floor
            near = np.abs(frames[:, None] - frames[flat][None]).min(1) / (EASE_S * fps)
            drop = np.interp(frames, frames[flat], (rest[1] - out[:, ks[1], 2])[flat])
            out[:, ks[1], 2] += drop * np.clip(1.0 - near, 0.0, 1.0)
            toe = [c for c, k in zip(columns, ks) if k == ks[1]]  # the soles the toe's body carries
            out[:, ks[1], 2] += np.maximum(-lowest(soles, turns, out)[:, toe].min(1), 0.0)
        held = {ks[0]: down[:, columns[0]], ks[1]: down[:, columns[1]]} if ks[0] != ks[1] else {ks[0]: planted}
        for k, sole in held.items():  # a planted sole keeps where it landed (one body's heel and toe: either)
            if not sole.any():
                continue
            starts = np.flatnonzero(sole & ~np.r_[False, sole[:-1]])
            anchor = out[starts[np.searchsorted(starts, frames, side="right") - 1].clip(0), k, :2]
            shift = np.zeros((len(out), 2))
            shift[sole] = (anchor - out[:, k, :2])[sole]
            for axis in range(2):
                shift[:, axis] = np.interp(frames, frames[sole], shift[sole, axis])
            out[:, k, :2] += shift
    return out
