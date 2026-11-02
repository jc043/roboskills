"""GVHMR frames read wrong are found and re-made; a clean spin is left alone."""

import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.tools.gvhmr.repair import misread


def spin(T: int = 90, per_frame: float = 12.0) -> dict:
    """A person spinning about the vertical at `per_frame` degrees, every keypoint seen at 0.9."""
    yaw = np.radians(per_frame) * np.arange(T)
    orient = Rotation.from_euler("y", yaw[:, None]).as_rotvec()
    kp2d = np.zeros((T, 17, 3))
    kp2d[..., 2] = 0.9
    return {"world_global_orient": orient, "camera_global_orient": orient.copy(), "world_body_pose": np.zeros((T, 63)),
            "camera_body_pose": np.zeros((T, 63)), "world_transl": np.zeros((T, 3)), "camera_transl": np.zeros((T, 3)),
            "kp2d": kp2d, "fps": 30.0}  # fmt: skip


def flip(d: dict, frames: slice) -> None:
    d["world_global_orient"][frames] = (
        Rotation.from_euler("y", np.pi) * Rotation.from_rotvec(d["world_global_orient"][frames])
    ).as_rotvec()


def test_a_clean_spin_is_left_alone():
    assert not misread(spin()).any()


def test_a_run_read_flipped_and_back():
    d = spin()
    flip(d, slice(30, 34))
    assert np.where(misread(d))[0].tolist() == [30, 31, 32, 33]


def test_a_joint_thrown_out_for_one_frame():
    d = spin()
    d["world_body_pose"][50, 16 * 3 : 16 * 3 + 3] = [0, 0, 1.2]  # the right shoulder, 69 degrees for one frame
    assert np.where(misread(d))[0].tolist() == [50]
