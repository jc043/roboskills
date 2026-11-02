"""Frames GVHMR read wrong, re-made from the frames around them. GVHMR can read a fast turn backwards for a few frames
or throw a joint out for one; anything downstream (a character's retarget, a robot's) inherits it.

    python -m roboskills.tools.gvhmr.repair motion/smplx.npz   # rewrites it; prints the frames re-made

`recover_motion` runs it on its output, and the motion_gen workflow's `retarget_motion` on whatever it is given (a
repaired file has nothing left to re-make).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


MAJOR = (0, 1, 3, 4, 5, 15, 16, 17, 18)  # body_pose's hips, knees, spine, shoulders and elbows


def misread(d: dict) -> np.ndarray:
    """GVHMR's frames read wrong [T]: one where the body or a major joint goes out and comes back (to it and back turns
    over 30 degrees more than going straight past it; clean clips stay under about 11); the run between two body turns
    of over 90 degrees between frames (a person's spin is 10-20 a frame) less than half a second apart, read flipped
    and back; and beside a lone such turn, the side whose keypoints GVHMR saw less surely."""
    T, conf = len(d["world_global_orient"]), d["kp2d"][:, :, 2].mean(1)
    turn = lambda a, b: np.degrees((Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)).magnitude())  # noqa: E731
    bad = np.zeros(T, bool)
    for r in [d["world_global_orient"], *d["world_body_pose"].reshape(T, 21, 3)[:, MAJOR].transpose(1, 0, 2)]:
        bad[1:-1] |= turn(r[:-2], r[1:-1]) + turn(r[1:-1], r[2:]) - turn(r[:-2], r[2:]) > 30
    step = turn(d["world_global_orient"][:-1], d["world_global_orient"][1:])
    fast = step > 2 * np.median(step)
    jumps, paired = np.where(step > 90)[0], set()
    for a, b in zip(jumps[:-1], jumps[1:]):
        if b - a <= 15 and a not in paired:  # out and back within half a second
            bad[a + 1 : b + 1] = True
            paired |= {a, b}
    for t in (t for t in jumps if t not in paired):
        s, e = t, t + 1
        while s > 0 and fast[s - 1]:
            s -= 1
        while e < T - 1 and fast[e]:
            e += 1
        before, after = np.arange(s + 1, t + 1), np.arange(t + 1, e + 1)
        bad[before if len(before) and conf[before].mean() < conf[after].mean() else after] = True
    return bad


def remake(d: dict, bad: np.ndarray) -> dict:
    """`d` with the `bad` frames' turns slerped, and places interpolated, between the good frames around them."""
    good, at = np.where(~bad)[0], np.where(bad)[0]
    d = dict(d)
    for k in ("world_global_orient", "camera_global_orient", "world_body_pose", "camera_body_pose"):
        r = d[k].reshape(len(bad), -1, 3).copy()
        for j in range(r.shape[1]):
            r[at, j] = Slerp(good, Rotation.from_rotvec(r[good, j]))(np.clip(at, good[0], good[-1])).as_rotvec()
        d[k] = r.reshape(d[k].shape)
    for k in ("world_transl", "camera_transl"):
        d[k] = np.stack([np.interp(np.arange(len(bad)), good, d[k][good, i]) for i in range(3)], 1)
    return d


def repair(smplx: Path) -> list[int]:
    """Rewrite GVHMR's `smplx` with its `misread` frames re-made; the frames, also kept in it as `remade`."""
    d = dict(np.load(smplx))
    d = remake(d, bad) if (bad := misread(d)).any() else d
    np.savez(smplx, **d, remade=np.where(bad)[0])
    return np.where(bad)[0].tolist()


if __name__ == "__main__":
    print(repair(Path(sys.argv[1])))
