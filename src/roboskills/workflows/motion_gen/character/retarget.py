"""GVHMR's motion on a Mixamo character through RoboKit's humanoid retargeting, the recipe in SKILL.md.

`retarget_motion(motion, character, out)`: each bone turns as its SMPL-X joint (GVHMR's or Kimodo's, read as a robot's
is: tools/robokit/humanoid.py's `person`, which re-makes the frames GVHMR read wrong), down the character's own
proportions; RoboKit fits the character's joint ranges to those targets frame by frame, the feet the person plants held
where they land (tools/robokit/feet.py, on the character's `SOLES`). `check_motion` measures the result.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.tools.robokit import feet
from roboskills.tools.robokit.humanoid import Y_TO_Z, person
from roboskills.workflows.motion_gen.character import contact
from roboskills.workflows.motion_gen.character.character import Character, load_character, positions, skinned
from roboskills.workflows.motion_gen.character.robot import SMPLX, robot, turns


TURN = (0, 0.5)  # a bone followed by its turn alone: its place comes from the bones above it
# The elbow and wrist follow the person's places, not only turns: a person's elbow bends sideways, a character's hinges
WEIGHTS = {"Hips": (30, 2), "Spine": TURN, "Spine1": TURN, "Spine2": (0.7, 0.7), "Neck": (0, 1), "Head": (0, 1),
           "Shoulder": TURN, "Arm": (1.5, 1), "ForeArm": (3, 0.5), "Hand": (6, 1.2), "UpLeg": (1.5, 1), "Leg": TURN,
           "Foot": (30, 2), "ToeBase": (30, 1)}  # role, its side's if it has one -> position, orientation  # fmt: skip
SETTLE = 8  # solves of the first frame before the clip: one, from the T-pose, leaves the arms short of it
LIMB = 0.1  # metres a foot or hand may stray from its target
STEPS = 3  # solves after the first, each letting the feet that could not hold their spot step
SOLES = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")  # a character's sole bones, in `feet.down`'s order


def character_soles(character: Character) -> feet.Soles:
    """The character's `SOLES`: the skin each bone and the bones below it (a foot's toe has its own) drive most."""
    dominant, index = character.skin_weights.argmax(1), [character.bone(role) for role in SOLES]
    below = {j: {j} for j in index}
    for k, p in enumerate(character.parents):  # parents come first
        for j in index:
            if p in below[j]:
                below[j].add(k)
    under = [list(below[j] - {k for i in index if i != j and i in below[j] for k in below[i]}) for j in index]
    points = tuple(character.verts[np.isin(dominant, u)] - character.rest_joints[j] for j, u in zip(index, under))
    return feet.Soles(tuple(index), points, character.rest_joints[index, 2])


def targets(character: Character, world: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """Each bone's world turn [T, J, 3, 3]: its SMPL-X joint's in `world` [T, joints, 3, 3] (named `joint_names`), a
    bone without one its parent's."""
    role = {bone: r for r, bone in character.roles.items()}
    out = np.empty((len(world), len(character.joint_names), 3, 3))
    for j, p in enumerate(character.parents):
        name = SMPLX.get(role[character.joint_names[j]])
        out[:, j] = world[:, joint_names.index(name)] if name else out[:, p]
    return out


def weights(character: Character) -> dict[str, tuple[float, float]]:
    """`WEIGHTS` by the character's bones."""
    sides = lambda role: (role,) if role in character.roles else (f"Left{role}", f"Right{role}")  # noqa: E731
    return {character.roles[r]: w for role, w in WEIGHTS.items() for r in sides(role) if r in character.roles}


def retarget_motion(motion: Path, character: Path, out: Path):
    """out/motion.npz, motion_studio.py's MOTION: `fps`, `turns` [T, J, 3, 3], `positions` [T, J, 3], `joint_names`,
    `studio_camera_direction`; and for the bake and the check `rest_joints`, `targets` [T, J, 3], the position each
    bone was asked for, `planted` [T, 4] (`feet.down` and `feet.step`) and `qpos`. The planted soles stand at z = 0."""
    from robokit.helpers.humanoid_retargeting import (
        HumanoidRetargeterOnline,
        HumanoidRetargeterOnlineConfig,
        LinkMapping,
    )
    from robokit.helpers.humanoid_retargeting.loaders.smplx import load_smplx_file
    from robokit.smplx import SMPLX_JOINT_NAMES

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c, p = load_character(character), person(motion, out)
    fps, planted = p.fps, p.planted
    np.savez(out / "rest.npz", root_orient=np.zeros((1, 3)), pose_body=np.zeros((1, 63)), trans=np.zeros((1, 3)),
             betas=p.betas, mocap_frame_rate=fps)  # fmt: skip
    rest = load_smplx_file(str(out / "rest.npz"))  # the person standing: its pelvis' height over its soles
    standing = float(rest.joints[0, 0, 1] - rest.vertices[0, :, 1].min())
    world = Rotation.from_quat(p.poses[..., 3:].reshape(-1, 4), scalar_first=True).as_matrix().reshape(-1, 55, 3, 3)
    wanted = targets(c, world @ Y_TO_Z.T, list(SMPLX_JOINT_NAMES))  # SMPL-X's y-up T-pose onto a Mixamo rest
    root = p.poses[:, 0, :3] * c.rest_joints[c.bone("Hips"), 2] / standing

    scored = weights(c)
    config = HumanoidRetargeterOnlineConfig(
        human_root_name=c.roles["Hips"], human_height_assumption=1.0, velocity_limit_dt=1 / fps,
        link_mapping={b: LinkMapping(b, np.zeros(3), np.zeros(4)) for b in scored},
    )  # fmt: skip
    config.task("target").position_weight = {b: w[0] for b, w in scored.items()}
    config.task("target").orientation_weight = {b: w[1] for b, w in scored.items()}
    config.task("rest").rest_q_arg, config.task("rest").weight = None, 0.05
    r = robot(c)
    solver = HumanoidRetargeterOnline(config, robot=r)
    solver.warmup(1)
    bones = [c.joint_names.index(b) for b in solver.human_joint_names]
    wxyz = Rotation.from_matrix(wanted[:, bones].reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
    soles_ = character_soles(c)
    for _ in range(STEPS + 1):  # a planted foot the legs cannot keep on its spot steps, and the clip is solved again
        aim = feet.plant(soles_, wanted, positions(c, wanted, root), planted, fps)
        frames = np.concatenate((aim[:, bones], wxyz.reshape(len(wanted), len(bones), 4)), axis=2)
        solve = lambda t: solver.solve_numpy(frames[t : t + 1], human_heights=np.ones(1))  # noqa: E731
        for _ in range(SETTLE):
            solve(0)
        qpos = np.concatenate([solve(t) for t in range(len(frames))])
        solved = turns(c, r, qpos)
        joints = positions(c, solved, qpos[:, :3])
        off = np.linalg.norm(joints[:, soles_.index] - aim[:, soles_.index], axis=-1) > LIMB / 2
        if not (miss := (off & planted).reshape(-1, 2, 2).any(2)).any():
            break
        planted = feet.step(planted, miss, fps)
    low = np.where(planted, feet.lowest(soles_, solved, joints), np.inf).min(1)  # the floor: where planted soles stand
    floor = np.array([0.0, 0.0, np.median(low[planted.any(1)]) if planted.any() else skinned(c, solved, joints)[..., 2].min()])  # fmt: skip
    np.savez(out / "motion.npz", fps=fps, turns=solved, positions=joints - floor, joint_names=np.array(c.joint_names),
             studio_camera_direction=p.toward, rest_joints=c.rest_joints, targets=aim - floor,
             planted=planted, qpos=qpos)  # fmt: skip


def check_motion(motion: Path, character: Path) -> dict:
    """`ok`, a `reason` and `limb_mm` (the feet's and hands' worst distance from their targets), `past_deg` (the worst
    joint's degrees past a person's range) and `step_deg` (the largest turn of a joint between frames, a pop)."""
    d, c = np.load(motion), load_character(character)
    if not np.isfinite(d["positions"]).all():
        return dict(ok=False, reason="the solved motion is not finite")
    limbs = [c.bone(f"{side}{limb}") for side in ("Left", "Right") for limb in ("Foot", "Hand")]
    limb = np.linalg.norm(d["positions"][:, limbs] - d["targets"][:, limbs], axis=2).max()
    measured = contact.joints(c, d["turns"]).values()
    report = dict(limb_mm=round(float(limb) * 1000, 1), past_deg=round(float(max(j["past"].max() for j in measured)), 1),
                  step_deg=round(float(max(j["step"].max() for j in measured)), 1))  # fmt: skip
    return dict(ok=bool(limb <= LIMB), reason="" if limb <= LIMB else f"a limb strays {limb:.3f} m from its target") | report  # fmt: skip
