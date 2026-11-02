"""A character's hand as a RoboKit hand, and a video's WiLoR hands (or an SMPL-X file's) retargeted onto its fingers.

`hand_robot(character, side)`: the hand bone as a floating base and every finger bone below it as revolute joints about
its anatomical axes, limited to a person's range (`contact.RANGES`): a finger's middle and end joints bend, its knuckle
bends and spreads, the thumb's joints also twist. `hand_spec(character, side)`: RoboKit's `HandSpec` for the 21
MANO/OpenPose keypoints on those bones.
`fit_fingers(character, side, keypoints)`: per frame, a hand's 21 keypoints go into the character hand's rest frame
(matched at the wrist and knuckles) and RoboKit's `HandRetargeterOnline` fits the finger joints to the directions of
their segments, so the hands' sizes do not matter. `retarget_fingers(character, motion, hands, out)` fits WiLoR's
keypoints and `smplx_fingers(character, motion, amass, out)` an AMASS SMPL-X file's hands (Kimodo's); each puts the
finger bones' world turns into a copy of the body motion. Given GVHMR's smplx.npz, the hand bones also turn as WiLoR's
hands (GVHMR sees no hands: its wrists are guesses, often tens of degrees off).
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from roboskills.tools.robokit.humanoid import MIRROR, SETTLE, SMOOTH, Y_TO_Z, camera_turns, hand_config, hand_keypoints
from roboskills.tools.wilor.wilor import usable
from roboskills.workflows.motion_gen.character import contact
from roboskills.workflows.motion_gen.character.character import Character, load_character


FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def _snap(v: np.ndarray) -> str:
    i = int(np.argmax(np.abs(v)))
    return ("" if v[i] > 0 else "-") + "xyz"[i]


def _bones(character: Character, side: str) -> list[int]:
    """The hand bone, then its finger bones, parents first."""
    roles = [f"{side}Hand"] + [f"{side}Hand{f}{k}" for f in FINGERS for k in (1, 2, 3, 4)]
    found = [character.bone(r) for r in roles if r in character.roles]
    return sorted(found, key=lambda j: (j != character.bone(f"{side}Hand"), j))


def hand_robot(character: Character, side: str):
    """The ``side`` hand as a RoboKit `Robot` with a floating base at the hand bone."""
    from robokit.robo import Robot, RobotSpec
    from robokit.robo.robot_spec import JointType

    bones = _bones(character, side)
    fingers = bones[1:]
    slot = {j: k for k, j in enumerate(fingers)}
    names = [character.joint_names[j] for j in fingers]
    n = 3 * len(fingers)
    joints = [f"{b}__{a}" for b in names for a in ("side", "bend", "twist")]
    parent_joints = np.empty(n, np.int32)
    for k, j in enumerate(fingers):
        p = character.parents[j]
        parent_joints[3 * k : 3 * k + 3] = (3 * slot[p] + 2 if p in slot else -1, 3 * k, 3 * k + 1)
    transforms = np.zeros((n, 7), np.float32)
    transforms[:, 3] = 1.0
    transforms[::3, :3] = character.rest_offsets[fingers]
    axes = np.tile(np.eye(3, dtype=np.float32), (len(fingers), 1))
    limits = np.zeros((n, 2), np.float32)  # unranged bones (the tips) stay fixed
    ranged = contact._axes(character)
    live = []
    for k, j in enumerate(fingers):
        if j not in ranged:
            continue
        key, along, bend, aside, rest, sign = ranged[j]
        (b0, b1), (s0, s1), (t0, t1) = contact.RANGES[key]
        # Spread about the palm's normal first, then bend: a bent knuckle still bends toward the palm, where bending
        # first would make the spread a turn of the curled finger about the hand.
        axes[3 * k : 3 * k + 3] = (np.cross(along, aside), np.cross(along, bend), along)
        twist = (t0, t1) if sign > 0 else (-t1, -t0)
        limits[3 * k : 3 * k + 3] = np.radians([(s0, s1), (b0 - rest, b1 - rest), twist])
        # A finger's middle and end joints are hinges and its knuckle bends and spreads without twisting, so a
        # segment's direction fixes its joints; the thumb keeps all three.
        live += [3 * k + a for a in (range(3) if key.startswith("Thumb") else (0, 1) if key == "Finger1" else (1,))]
    actuated = np.full(n, -1, np.int32)
    actuated[live] = np.arange(len(live))
    base = character.joint_names[bones[0]]
    links = [*names, base, *(f"__{a}__{b}" for a in ("side", "bend") for b in names)]
    link_parents = np.concatenate((np.arange(2, n, 3), [-1], np.arange(0, n, 3), np.arange(1, n, 3))).astype(np.int32)
    velocity = np.full(n, 40.0, np.float32)
    empty = {link: trimesh.Scene() for link in links}
    description = json.dumps([character.name, side, names, np.round(axes, 6).tolist(), np.round(limits, 6).tolist()])
    return Robot(RobotSpec(
        name=f"hand_{side}_{character.name}", robot_description=description, format="urdf",
        joint_names=joints, nonfixed_joint_names=joints, actuated_joint_names=[joints[k] for k in live],
        joint_types=np.full(n, JointType.REVOLUTE, np.int32), joint_axes=axes,
        joint_twists=np.concatenate((np.zeros_like(axes), axes), axis=1),
        actuated_joint_indices=actuated, parent_joint_indices=parent_joints, parent_joint_transforms=transforms,
        mimic_actuated_joint_indices=np.full(n, -1, np.int32), mimic_multipliers=np.ones(n, np.float32),
        mimic_offsets=np.zeros(n, np.float32), joint_limits=limits, joint_velocity_limits=velocity,
        actuated_joint_limits=limits[live].copy(), actuated_joint_velocity_limits=velocity[live].copy(),
        link_names=links, link_parent_joint_indices=link_parents, base_link_name=base,
        ee_link_names=[b for b in names if b.endswith("4")],
        link_visual_geometries=empty, link_collision_geometries=empty,
        local_collision_sphere_centers=np.empty((0, 3), np.float32), collision_sphere_radii=np.empty(0, np.float32),
        collision_spheres_link_indices=np.empty(0, np.int32), self_collision_ignored_pairs=np.empty((0, 2), np.int32),
        link_masses=np.zeros(len(links), np.float32), link_local_com_positions=np.zeros((len(links), 3), np.float32),
    ))  # fmt: skip


def _links(character: Character, side: str) -> dict[str, str]:
    bone = lambda role: character.roles[f"{side}{role}"]  # noqa: E731
    out = {"wrist": bone("Hand")}
    for k, point in enumerate(("cmc", "mcp", "ip", "tip"), 1):
        out[f"thumb_{point}"] = bone(f"HandThumb{k}")
    for f in FINGERS[1:]:
        for k, point in enumerate(("mcp", "pip", "dip", "tip"), 1):
            out[f"{f.lower()}_{point}"] = bone(f"Hand{f}{k}")
    return out


def hand_spec(character: Character, side: str):
    """RoboKit's `HandSpec` for the 21 keypoints on the ``side`` hand's bones (rest frames are the world's axes)."""
    from robokit.helpers.hand_retargeting import HandSpec
    from robokit.helpers.hand_retargeting.presets._common import FIVE_FINGERTIPS, TARGET_CHAINS, TARGET_NAMES
    from robokit.utils.ee_coord_utils import HandCoordinateSpec

    at = lambda role: character.rest_joints[character.bone(f"{side}{role}")]  # noqa: E731
    palm = np.cross(at("HandIndex1") - at("Hand"), at("HandPinky1") - at("Hand")) * (-1.0 if side == "Left" else 1.0)
    up = at("HandMiddle1") - at("Hand")
    return HandSpec(
        floating_base=True, target_names=TARGET_NAMES, target_link_names=_links(character, side),
        target_chains=TARGET_CHAINS, root_target_name="wrist", contact_target_names=FIVE_FINGERTIPS,
        root_link_coord_spec=HandCoordinateSpec(palm_forward_axis=_snap(palm), four_fingers_up_axis=_snap(up)),
    )  # fmt: skip


def _frame(wrist: np.ndarray, index: np.ndarray, middle: np.ndarray, pinky: np.ndarray) -> np.ndarray:
    """A hand's frame from its wrist and knuckles: x toward the middle knuckle, z normal to the knuckles' plane."""
    x = middle - wrist
    x = x / np.linalg.norm(x)
    z = np.cross(index - wrist, pinky - wrist)
    z = z - (z @ x) * x
    z = z / np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1)


def _local_turns(robot, qpos: np.ndarray) -> np.ndarray:
    """[T, bones, 3, 3] each finger bone's turn in its parent from RoboKit's [T, 7 + dofs]."""
    actuated = robot.spec.actuated_joint_indices
    angles = np.zeros((len(qpos), len(actuated)))
    angles[:, actuated >= 0] = qpos[:, 7 + actuated[actuated >= 0]]
    about = Rotation.from_rotvec((angles[..., None] * robot.spec.joint_axes).reshape(-1, 3))
    local = about.as_matrix().reshape(len(qpos), -1, 3, 3)
    return local[:, 0::3] @ local[:, 1::3] @ local[:, 2::3]


def _turn_hand(c: Character, turns: np.ndarray, h, rows: dict[int, int], side: str, camera: np.ndarray) -> None:
    """In `turns`, the ``side`` hand turned as WiLoR's (`global_orient` is its turn in the camera, a left hand's read
    mirrored; `camera` [T, 3, 3] takes the camera's axes into the character's world): interpolated between usable
    detections, smoothed over `SMOOTH` frames, and the part about the forearm given to the forearm (a wrist barely
    twists), which leaves the hand where it was."""
    forearm, hand = c.bone(side + "ForeArm"), c.bone(side + "Hand")
    at = np.array(sorted(rows))
    seen = h["global_orient"][[rows[t] for t in at]]
    seen = MIRROR @ seen @ MIRROR if side == "Left" else seen
    local = Rotation.from_matrix(turns[at, forearm].transpose(0, 2, 1) @ camera[at] @ seen @ Y_TO_Z.T)  # in the forearm
    q = Slerp(at, local)(np.clip(np.arange(len(turns)), at[0], at[-1])).as_quat()
    q *= np.where(np.cumsum(np.r_[0, np.einsum("ti,ti->t", q[1:], q[:-1]) < 0]) % 2, -1.0, 1.0)[:, None]
    local = Rotation.from_quat(gaussian_filter1d(q, SMOOTH, axis=0, mode="nearest"))
    along = c.rest_offsets[hand] / np.linalg.norm(c.rest_offsets[hand])
    inverse = local.inv().as_quat()  # swing x twist about the forearm; the twist taken from the inverse sits first
    twist = Rotation.from_quat(np.c_[(inverse[:, :3] @ along)[:, None] * along, inverse[:, 3]]).inv()
    turns[:, forearm] = turns[:, forearm] @ twist.as_matrix()
    turns[:, hand] = turns[:, forearm] @ (twist.inv() * local).as_matrix()


def fit_fingers(character: Character, side: str, keypoints: np.ndarray, settle: int = 0) -> np.ndarray:
    """The ``side`` hand's finger bones' turns in their parents [T, bones, 3, 3] (`_bones` after the hand) fitted to 21
    MANO/OpenPose `keypoints` [T, 21, 3], each frame's put into the character hand's rest frame (matched at the wrist
    and knuckles), the first `settle` more times first; a frame of NaNs takes its joints interpolated between the
    nearest fits (held before the first and after the last)."""
    from robokit.helpers.hand_retargeting import HandRetargeterOnline

    robot, spec = hand_robot(character, side), dataclasses.replace(hand_spec(character, side), floating_base=False)
    solver = HandRetargeterOnline(hand_config(spec.target_link_names), robot, spec)
    solver.warmup(1)
    at = lambda role: character.rest_joints[character.bone(side + role)]  # noqa: E731
    rest = _frame(at("Hand"), at("HandIndex1"), at("HandMiddle1"), at("HandPinky1"))
    q = np.zeros((len(keypoints), 7 + robot.spec.num_actuated_joints))
    q[:, 3] = 1.0
    seen = np.flatnonzero(np.isfinite(keypoints).all(axis=(1, 2)))
    for t in np.r_[np.repeat(seen[:1], settle), seen]:
        p = keypoints[t].astype(np.float64)
        p = (p - p[0]) @ _frame(p[0], p[5], p[9], p[17]) @ rest.T + at("Hand")
        q[t] = solver.solve_numpy(p[None].astype(np.float32))[0]
    if len(seen):
        for k in range(7, q.shape[1]):
            q[:, k] = np.interp(np.arange(len(q)), seen, q[seen, k])
    return _local_turns(robot, q)


def _save(c: Character, d: dict, turns: np.ndarray, keypoints: dict, out: Path, settle: int = 0) -> None:
    """``out``: the motion ``d`` with ``turns`` and each side's fingers fitted to its ``keypoints`` (`fit_fingers`), the
    finger bones' positions following (a forearm's twist leaves its hand where it was)."""
    for side, seen in keypoints.items():
        local = fit_fingers(c, side, seen, settle)
        for k, j in enumerate(_bones(c, side)[1:]):  # parents first
            turns[:, j] = turns[:, c.parents[j]] @ local[:, k]
    pos = d["positions"].copy()
    fingers = {b for side in keypoints for b in _bones(c, side)[1:]}
    for j, p in enumerate(c.parents):
        if j in fingers:
            pos[:, j] = pos[:, p] + turns[:, p] @ c.rest_offsets[j]
    d.update(turns=turns, positions=pos)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)


def retarget_fingers(character: Path, motion: Path, hands: Path, out: Path, smplx: Path | None = None) -> dict:
    """``out``: ``motion`` with its finger bones posed from ``hands`` (WiLoR's hands.npz) and, given ``smplx`` (the
    GVHMR smplx.npz ``motion`` was made from, whose camera places WiLoR's hand turns), its hands turned as WiLoR's.
    Frames without a usable detection of a hand take its joints interpolated between the nearest fits (held before the
    first and after the last). Returns per-side counts."""
    c, d, h = load_character(character), dict(np.load(motion)), np.load(hands)
    turns, frames = d["turns"].copy(), len(d["turns"])
    tracks, keypoints = usable(h, frames), {}
    if smplx is not None:  # GVHMR's camera into its world, then into the character's (`humanoid.person`)
        camera = Y_TO_Z @ camera_turns(np.load(smplx)).as_matrix()
    for side, rows in tracks.items():
        if smplx is not None and len(rows) > 1:
            _turn_hand(c, turns, h, rows, side, camera)
        keypoints[side] = np.full((frames, 21, 3), np.nan)  # MANO/OpenPose's
        keypoints[side][list(rows)] = h["keypoints_3d"][list(rows.values())]
    _save(c, d, turns, keypoints, out)
    return {side: dict(fitted=len(rows), frames=frames) for side, rows in tracks.items()}


def smplx_hands(amass: Path) -> dict[str, np.ndarray]:
    """Each side's 21 MANO/OpenPose keypoints [T, 21, 3] of the hands in an AMASS SMPL-X file (its `pose_hand`): the
    SMPL-X hand's joints and fingertip vertices, through RoboKit's SMPL-X."""
    from robokit.helpers.humanoid_retargeting.loaders.smplx import load_smplx_file
    from robokit.smplx import SMPLX_JOINT_NAMES, BodyModelSpecTensors, body_lbs_warp

    clip = load_smplx_file(str(amass), compute_vertices=False)  # its body, the hands left flat
    clip.state.full_pose_aa[:, SMPLX_JOINT_NAMES.index("left_index1") :] = clip.raw["pose_hand"].reshape(-1, 30, 3)
    posed = body_lbs_warp(BodyModelSpecTensors(spec=clip.spec, device=clip.device), clip.state)
    points = hand_keypoints(posed["T_world_joint"].numpy()[..., :3], posed["landmarks"].numpy())
    return {side.title(): p for side, p in points.items()}


def smplx_fingers(character: Path, motion: Path, amass: Path, out: Path) -> None:
    """``out``: ``motion`` with its finger bones posed as the hands of ``amass``, an AMASS SMPL-X file (Kimodo's holds
    SMPL-X's relaxed mean hand on every frame)."""
    d = dict(np.load(motion))
    _save(load_character(character), d, d["turns"].copy(), smplx_hands(amass), out, SETTLE)


if __name__ == "__main__":
    import sys

    print(retarget_fingers(*map(Path, sys.argv[1:6])))
