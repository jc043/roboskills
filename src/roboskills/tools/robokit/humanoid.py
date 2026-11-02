"""A video's person, GVHMR's body and WiLoR's hands, on any humanoid RoboKit retargets onto, by RoboKit's own
retargeting: the body by the robot's humanoid preset (`humanoid`: RoboKit's online retargeting, each frame started
from the one before), solved again with the feet the person plants held level on their spots (`feet`, as motion_gen's
characters' are), its wrists the body's (WiLoR's hand turns lie past a robot wrist's range and flip it), and each hand
RoboKit has a preset for fitted per frame by its online hand retargeter to the directions of the person's finger
segments, as motion_gen's character fingers are. Ours is only the glue: reading GVHMR and WiLoR, finding the robot's
feet, hands and palm frames from RoboKit's presets and assets, the planting, and writing motion.npz and the GLB.
SKILL.md lists the robots it has run on.

    python -m roboskills.tools.robokit humanoid --smplx motion/smplx.npz [--hands hands/hands.npz] --out robot/ --robot g1_inspire
"""

from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from roboskills.tools.gvhmr.repair import misread, remake
from roboskills.tools.robokit import feet
from roboskills.tools.wilor.wilor import usable


Y_TO_Z = Rotation.from_euler("x", 90, degrees=True).as_matrix()  # GVHMR's y-up world onto RoboKit's z-up
MIRROR = np.diag([-1.0, 1.0, 1.0])  # MANO's left hand read as a right one: F r F
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SMOOTH = 1.5  # frames of the Gaussian on a seen hand's turn: WiLoR's shakes a few degrees a frame
SIDES = ("left", "right")
BODY_SETTLE = 40  # solves of the body's first frame before the clip: from rest, each moves only so far
LIFTS = 4  # solves at most with the feet planted, each asking a foot that sank up again
SETTLE = 4  # solves of the first frame before the clip: one, from rest, leaves the fingers short of it
COLOUR = (0.06, 0.065, 0.075, 1.0)  # the links' one material (linear): dark grey against the studio's light floor


class Person(NamedTuple):
    poses: np.ndarray  # [T, 55, 7] every SMPL-X joint's world pose (xyz wxyz), z-up over the floor under the person
    relaxed: dict[str, np.ndarray]  # each side's 21 MANO/OpenPose keypoints [T, 21, 3] of SMPL-X's relaxed mean hand
    camera: np.ndarray  # [T, 3, 3] the video camera's turn into that world (Kimodo's: none, the identity)
    betas: np.ndarray  # the person's shape
    height: float  # the T-pose's
    fps: float
    toward: np.ndarray  # [3] `camera_side`
    planted: np.ndarray  # [T, 4] `feet.down` of the heels and toes (`feet.SOLE_JOINTS`)


def camera_turns(d: dict) -> Rotation:
    """GVHMR's camera's turn into its y-up world, per frame."""
    return Rotation.from_rotvec(d["world_global_orient"]) * Rotation.from_rotvec(d["camera_global_orient"]).inv()


def camera_side(d: dict) -> np.ndarray:
    """The horizontal direction from the person toward GVHMR's camera, z-up; for Kimodo, the way the person faces."""
    if "root_orient" in d:  # SMPL-X faces +z in its y-up rest
        toward = (Rotation.from_rotvec(d["root_orient"]).as_matrix() @ [0, 0, 1]).mean(0)
    else:
        toward = Y_TO_Z @ camera_turns(d).apply(-d["camera_transl"]).mean(0)
    return toward * [1, 1, 0] / np.linalg.norm(toward[:2])


def hand_keypoints(joints: np.ndarray, tips: np.ndarray) -> dict[str, np.ndarray]:
    """Each side's 21 MANO/OpenPose keypoints [T, 21, 3]: SMPL-X's wrist and finger `joints` [T, 55, 3] and its
    fingertip landmarks `tips` [T, L, 3]."""
    from robokit.smplx import SMPLX_JOINT_NAMES, SMPLX_STATIC_LANDMARK_NAMES

    out = {}
    for side in SIDES:
        points = [joints[:, SMPLX_JOINT_NAMES.index(f"{side}_wrist")]]
        for f in FINGERS:
            points += [joints[:, SMPLX_JOINT_NAMES.index(f"{side}_{f}{k}")] for k in (1, 2, 3)]
            points.append(tips[:, SMPLX_STATIC_LANDMARK_NAMES.index(f"{side}_{f}")])
        out[side] = np.stack(points, 1)
    return out


def person(smplx: Path, out: Path) -> Person:
    """GVHMR's `smplx` (its misread frames re-made; the world y-up) or Kimodo's (its SMPL-X `_amass.npz`, z-up) as
    RoboKit's SMPL-X, the floor levelled under the person (`feet.level`), as `Person`, the person both the robots here
    and motion_gen's characters retarget. Writes for GVHMR out/amass.npz and out/camera.npz, the files RoboKit's loader
    reads."""
    from robokit.helpers.humanoid_retargeting.loaders.smplx import load_smplx_file
    from robokit.smplx import SMPLX_JOINT_NAMES, BodyModelSpecTensors, body_lbs_warp

    d = dict(np.load(smplx))
    if kimodo := "root_orient" in d:
        fps, up, betas = float(d["mocap_frame_rate"]), np.eye(3), d["betas"]
    else:
        d = remake(d, bad) if (bad := misread(d)).any() else d
        fps, up, betas = float(d["fps"]), Y_TO_Z, d["world_betas"].mean(0)
        np.savez(out / "amass.npz", root_orient=d["world_global_orient"], pose_body=d["world_body_pose"],
                 trans=d["world_transl"], betas=betas, mocap_frame_rate=fps)  # fmt: skip
    clip = load_smplx_file(str(smplx if kimodo else out / "amass.npz"), compute_vertices=False)
    mean = [clip.spec.metadata[f"hands_mean{s}"] for s in "lr"]  # the loader's hands are flat
    clip.state.full_pose_aa[:, SMPLX_JOINT_NAMES.index("left_index1") :] = np.concatenate(mean).reshape(30, 3)
    posed = body_lbs_warp(BodyModelSpecTensors(spec=clip.spec, device=clip.device), clip.state)
    poses, tips = posed["T_world_joint"].numpy().astype(float), posed["landmarks"].numpy() @ up.T
    floor = np.zeros((len(poses), 3))
    floor[:, 2] = feet.level(posed["vertices"].numpy() @ up.T, fps)
    poses[..., :3] = poses[..., :3] @ up.T - floor[:, None]
    turns = Rotation.from_matrix(up) * Rotation.from_quat(poses[..., 3:].reshape(-1, 4), scalar_first=True)
    poses[..., 3:] = turns.as_quat(scalar_first=True).reshape(*poses.shape[:2], 4)
    relaxed = hand_keypoints(poses[..., :3].astype(np.float32), tips - floor[:, None])  # joints in RoboKit's float32
    soles = [SMPLX_JOINT_NAMES.index(j) for j in feet.SOLE_JOINTS]
    if kimodo:  # no camera: the soles seen from far above
        turn, seen, still = np.tile(np.eye(3), (len(poses), 1, 1)), poses[:, soles, :3] + [0, 0, 1e3], None
    else:
        np.savez(out / "camera.npz", root_orient=d["camera_global_orient"], pose_body=d["camera_body_pose"],
                 trans=d["camera_transl"], betas=betas, mocap_frame_rate=fps)  # fmt: skip
        seen = load_smplx_file(str(out / "camera.npz"), compute_vertices=False).T_world_joint_xyz_wxyz[:, soles, :3]
        turn, still = Y_TO_Z @ camera_turns(d).as_matrix(), d.get("feet_still")
    planted = feet.down(poses[:, soles, :3], fps, seen, still)
    return Person(poses, relaxed, turn, betas, clip.human_height, fps, camera_side(d), planted)


def seen_hands(hands: Path, poses: np.ndarray, camera: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each side WiLoR saw usably on two frames or more (`usable`): its 21 keypoints [T, 21, 3] at the SMPL-X wrist of
    `poses` and its world turn [T, 3, 3] (`camera` takes WiLoR's camera into the world; a left hand's read mirrored),
    both interpolated between those frames (held before the first and after the last), the turn smoothed over
    `SMOOTH` frames."""
    from robokit.smplx import SMPLX_JOINT_NAMES

    h, frames, out = np.load(hands), len(poses), {}
    for side, rows in usable(h, frames).items():
        if len(rows) < 2:
            continue
        at = np.array(sorted(rows))
        k = [rows[t] for t in at]
        seen = MIRROR @ h["global_orient"][k] @ MIRROR if side == "Left" else h["global_orient"][k]
        local = np.einsum("tji,tkj->tki", seen, h["keypoints_3d"][k] - h["keypoints_3d"][k][:, :1])  # in the hand
        t = np.clip(np.arange(frames), at[0], at[-1])
        q = Slerp(at, Rotation.from_matrix(camera[at] @ seen))(t).as_quat()
        q *= np.where(np.cumsum(np.r_[0, np.einsum("ti,ti->t", q[1:], q[:-1]) < 0]) % 2, -1.0, 1.0)[:, None]
        turn = Rotation.from_quat(gaussian_filter1d(q, SMOOTH, axis=0, mode="nearest")).as_matrix()
        local = np.stack([np.interp(t, at, c) for c in local.reshape(len(at), -1).T], 1).reshape(frames, -1, 3)
        wrist = poses[:, SMPLX_JOINT_NAMES.index(f"{side.lower()}_wrist"), None, :3]
        out[side.lower()] = wrist + np.einsum("tij,tkj->tki", turn, local), turn
    return out


def glb(robot, out: Path) -> None:
    """`out`: the robot's link meshes as a GLB, one node per link named for it at the world's origin (y-up, as glTF
    is), for Blender to move to the links' poses."""
    import trimesh

    scene, y_up = trimesh.Scene(), np.eye(4)
    y_up[:3, :3] = Y_TO_Z.T
    material = trimesh.visual.material.PBRMaterial(baseColorFactor=COLOUR, metallicFactor=0.0, roughnessFactor=0.6)
    for name, geometry in robot.spec.link_visual_geometries.items():
        mesh = geometry.to_mesh()
        if len(mesh.faces):
            mesh.apply_transform(y_up)
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(out)


HUMANOIDS, HANDS = "robokit.helpers.humanoid_retargeting.presets", "robokit.helpers.hand_retargeting.presets"
EFFECTORS = "robokit.assets.robots.end_effectors"


def module(name: str):
    """RoboKit's module `name`, or None where RoboKit has none."""
    try:
        return import_module(name)
    except ModuleNotFoundError:
        return None


def ancestors(robot, link: str) -> list[str]:
    """`link`, then its parent links up to the root, through RoboKit's joint tree."""
    s = robot.spec
    child = {int(j): k for k, j in enumerate(s.link_parent_joint_indices) if j >= 0}
    out, k = [link], s.link_names.index(link)
    while (j := int(s.link_parent_joint_indices[k])) >= 0 and (p := int(s.parent_joint_indices[j])) in child:
        out.append(s.link_names[k := child[p]])
    return out


def rest(robot) -> np.ndarray:
    """[L, 4, 4] every link's pose at the robot's zero joints, its base at the origin."""
    zero = np.zeros((1, robot.spec.num_actuated_joints), np.float32)
    return robot.forward_kinematics_via_matrix_numpy(zero, device="cpu")[0]


class Hand(NamedTuple):
    robot: object  # the hand alone, a RoboKit `Robot` held at its base
    spec: object  # its hand preset's `HandSpec`, held at its base
    frame: np.ndarray  # [4, 4] its palm frame at rest: palm normal, fingers, their cross; at its mount
    columns: list[int]  # its actuated joints' places among the whole robot's


class Body(NamedTuple):
    mapping: object  # RoboKit's humanoid preset: which link tracks which SMPL-X joint, and how it is solved
    kin: object  # the preset's robot, which the solver moves
    robot: object  # the whole robot (with its hands), its meshes loaded
    link: dict[str, str]  # SMPL-X joint -> the link tracking it
    hands: dict[str, Hand]  # the sides whose hand RoboKit retargets


def humanoid(name: str, fps: float) -> Body:
    """The humanoid `name` from RoboKit's own presets: its body the longest leading part of the name with a humanoid
    preset (`<body>_aligned`, else `<body>`: RoboKit's online retargeting, each frame started from the one before),
    without its clamp of each frame's step, which lags a fast move and then catches up; the rest of the name a hand,
    its whole robot RoboKit's assembly `<name>`, its hands retargeted where RoboKit has the hand's preset and end
    effector (`<hand>_hand` or `<hand>`); a robot without either keeps its hands as they are."""
    from robokit.robo import Robot
    from robokit.utils.ee_coord_utils import HandCoordinateSpec

    parts = name.split("_")
    leading = ("_".join(parts[:k]) for k in range(len(parts), 0, -1))
    if (body := next((b for b in leading if module(f"{HUMANOIDS}.{b}")), None)) is None:
        raise ValueError(f"RoboKit has no humanoid preset for {name!r}")
    preset = module(f"{HUMANOIDS}.{body}_aligned")
    mapping = getattr(preset, f"{body}_aligned") if preset else getattr(module(f"{HUMANOIDS}.{body}"), body)
    one = replace(mapping.solver, num_candidates=[1], select_at=[])  # from the frame before alone: no far restarts
    target, link = deepcopy(mapping.task("target")), {m.human_joint: r for r, m in mapping.link_mapping.items()}
    for side in SIDES:  # a planted foot holds as firmly as the root: the presets let the root pull it off its spot
        foot, root = link.get(f"{side}_foot") or link[f"{side}_ankle"], link[mapping.human_root_name]
        for weights in (target.position_weight, target.orientation_weight):
            weights[foot] = max(weights.get(foot, 0.0), weights.get(root, 0.0))
    terms = [target if t.name == "target" else t for t in mapping.terms]
    mapping = replace(mapping, velocity_limit_dt=1 / fps, velocity_clamp_scale=0.0, solver=one, terms=terms)
    kin = mapping.load_robot()
    if name != body and not hasattr(assembly := module(f"robokit.assets.robots.assembly.{name}"), "URDF_PATH"):
        raise ValueError(f"RoboKit has no whole-body URDF for {name!r}")
    whole, hands = Robot.load(str(assembly.URDF_PATH if name != body else mapping.urdf_path), load_meshes=True), {}
    hand = name[len(body) + 1 :]
    preset, ee = module(f"{HANDS}.{hand}"), module(f"{EFFECTORS}.{hand}_hand") or module(f"{EFFECTORS}.{hand}")
    for side in SIDES if hand and preset and ee and hasattr(ee, "URDF_PATH_RIGHT") else ():
        spheres = getattr(ee, f"COLLISION_SPHERE_PATH_{side.upper()}", None)  # for its preset's self-collision
        robot = Robot.load(str(getattr(ee, f"URDF_PATH_{side.upper()}")), load_collision_spheres=spheres is not None,
                           collision_spheres_path=spheres and str(spheres))  # fmt: skip
        coords: HandCoordinateSpec = getattr(ee, f"EE_COORD_SPEC_{side.upper()}")
        frame = rest(robot)[robot.spec.link_names.index(getattr(ee, f"MOUNT_LINK_NAME_{side.upper()}"))].copy()
        axes = np.stack([coords.palm_forward_vec, coords.four_fingers_up_vec, coords.palm_right_vec], 1)
        frame[:3, :3] = frame[:3, :3] @ axes
        w, below, own = whole.spec, set(), set(kin.spec.actuated_joint_names)
        for k in np.flatnonzero(w.actuated_joint_indices >= 0):  # the joints past this side's wrist the body lacks
            child = w.link_names[list(w.link_parent_joint_indices).index(k)]
            if link[f"{side}_wrist"] in ancestors(whole, child) and w.joint_names[k] not in own:
                below.add(w.joint_names[k])
        names = list(w.actuated_joint_names)  # each named as the hand's own, behind the assembly's prefix
        columns = [names.index(next(j for j in below if j.endswith(n))) for n in robot.spec.actuated_joint_names]
        hands[side] = Hand(robot, replace(preset.spec, floating_base=False), frame, columns)
    return Body(mapping, kin, whole, link, hands)


def hand_config(links: dict[str, str]):
    """RoboKit's `HandRetargeterOnlineConfig` for a hand whose tracked points are `links` (target name -> link): the
    directions between the tracked points along each finger (a finger tracked at one point, from the wrist), so the
    hands' sizes and palm shapes do not matter and a straight finger maps to a straight one; smoothing between frames;
    the joint limits; and a faint pull toward rest, which only keeps what no direction shows (a thumb's twist) from
    wandering."""
    from robokit.helpers.hand_retargeting import HandRetargeterOnlineConfig
    from robokit.helpers.hand_retargeting.config import HandContext as ctx
    from robokit.helpers.hand_retargeting.presets._common import TARGET_CHAINS, correspondences
    from robokit.terms.dense.configuration_task import ConfigurationTask
    from robokit.terms.dense.frame_vector_task import FrameVectorTask
    from robokit.terms.dense.position_limit import PositionLimit
    from robokit.terms.dense.smoothness_task import SmoothnessTask

    tracked = [[p for p in chain if p in links] for chain in TARGET_CHAINS]
    segments = [s for c in tracked for s in (zip(c, c[1:]) if len(c) > 1 else [("wrist", *c)] if c else [])]
    return HandRetargeterOnlineConfig(terms=[
        FrameVectorTask(targets=ctx.DIRECTIONS, target_names=ctx.TARGET_NAMES, weight=correspondences(links, segments),
                        weight_mode="normalized", direction_only=True, name="directions"),
        ConfigurationTask(rest_q=ctx.REST_Q, weight=1e-4, name="rest"),
        SmoothnessTask(prev_var=ctx.PREVIOUS_STATE, warmup_frames=1, frame=ctx.FRAME, weight=0.002, name="smoothness"),
        PositionLimit(weight=1.0, name="position_limit"),
    ])  # fmt: skip


def fit_fingers(hand: Hand, points: np.ndarray, right: bool, device: str | None = None) -> np.ndarray:
    """The `hand`'s joints [T, dofs] fitted per frame by RoboKit's `HandRetargeterOnline` (`hand_config`) to 21
    MANO/OpenPose `points` [T, 21, 3], each frame's put into the hand's palm frame (its end effector's coordinate spec:
    palm normal, fingers; the person's from the wrist and knuckles). The first frame is solved `SETTLE` times."""
    from robokit.helpers.hand_retargeting import HandRetargeterOnline

    solver = HandRetargeterOnline(hand_config(hand.spec.target_link_names), hand.robot, hand.spec, device)
    solver.warmup(1)
    q = np.zeros((len(points), hand.robot.spec.num_actuated_joints), np.float32)
    for t in [0] * SETTLE + list(range(len(points))):
        p = points[t].astype(np.float64) - points[t, 0]
        up = p[9] / np.linalg.norm(p[9])  # the person's: fingers toward the middle knuckle, palm out of the knuckles
        out = np.cross(p[5], p[17]) * (1 if right else -1)
        out = (out - (out @ up) * up) / np.linalg.norm(out - (out @ up) * up)
        p = p @ np.stack([out, up, np.cross(out, up)], 1) @ hand.frame[:3, :3].T + hand.frame[:3, 3]
        q[t] = solver.solve_numpy(p[None].astype(np.float32))[0, 7:]
    return q


def soles(b: Body) -> tuple[list[str], feet.Soles]:
    """The links tracking the person's toes (else ankles) and their `feet.Soles`: the lowest mesh among each and its
    parent links, its points within 1 cm of its lowest at rest in that link's frame, split at their middle along x
    (a robot faces x at rest) into heel and toe."""
    tracked = [b.link.get(f"{side}_foot") or b.link[f"{side}_ankle"] for side in SIDES]
    at, kin_at, parts, heights = rest(b.robot), rest(b.kin), [], []
    for name in tracked:
        meshes = {n: b.robot.spec.link_visual_geometries[n].to_mesh() for n in ancestors(b.kin, name)
                  if n in b.robot.spec.link_visual_geometries}  # fmt: skip
        place = {n: at[b.robot.spec.link_names.index(n)] for n in meshes}
        world = {n: m.vertices @ place[n][:3, :3].T + place[n][:3, 3] for n, m in meshes.items() if len(m.vertices)}
        points = min(world.values(), key=lambda v: v[:, 2].min())
        points = points[points[:, 2] < points[:, 2].min() + 0.01]
        frame = kin_at[b.kin.spec.link_names.index(name)]
        local = (points - frame[:3, 3]) @ frame[:3, :3]
        heights.append(frame[2, 3] - points[:, 2].min())
        parts += [local[points[:, 0] < points[:, 0].mean()], local[points[:, 0] >= points[:, 0].mean()]]
    return tracked, feet.Soles((0, 0, 1, 1), tuple(parts), np.repeat(heights, 2))


def plant(b: Body, tracked: list[str], sole: feet.Soles, solved: np.ndarray, planted: np.ndarray,
          fps: float) -> np.ndarray:  # fmt: skip
    """The `tracked` foot links' world poses [T, 2, 4, 4] for a second solve, from the first, `solved` [T, 7 + dofs]:
    a planted foot (`planted` [T, 4], `feet.down`) level as at rest, its turn about the vertical kept and eased in and
    out over `feet.EASE_S`, then held on its spot with its sole on the floor, and no sole below it (`feet.plant`)."""
    from robokit.xform.numpy import pose7_to_tf_mat

    links = b.kin.forward_kinematics_via_matrix_numpy(solved[:, 7:], pose7_to_tf_mat(solved[:, :7]), device="cpu")
    index = [b.kin.spec.link_names.index(n) for n in tracked]
    at, level_at, frames = links[:, index].copy(), rest(b.kin)[index, :3, :3], np.arange(len(links))
    for k in range(2):
        down = planted[:, 2 * k : 2 * k + 2].any(1)
        if not down.any():
            continue
        near = np.abs(frames[:, None] - frames[down][None]).min(1) / (feet.EASE_S * fps)
        now, heading = Rotation.from_matrix(at[:, k, :3, :3]), at[:, k, :3, :3] @ level_at[k].T
        yaw = Rotation.from_euler("z", np.arctan2(heading[:, 1, 0], heading[:, 0, 0])[:, None])
        ease = (now.inv() * yaw * Rotation.from_matrix(level_at[k])).as_rotvec() * np.clip(1 - near, 0, 1)[:, None]
        at[:, k, :3, :3] = (now * Rotation.from_rotvec(ease)).as_matrix()
    at[..., :3, 3] = feet.plant(sole, at[..., :3, :3], at[..., :3, 3], planted, fps)
    return at


def aim(mapping, names: list[str], frames: np.ndarray, link: str, target: np.ndarray, height: float) -> None:
    """In the person's joint poses `frames` [T, J, 7] (`names`), the joint `link` tracks moved so that RoboKit's target
    for `link` is `target` [T, 4, 4]: its target formula (scaled about the root, then offset) run backwards."""
    entry, s = mapping.link_mapping[link], mapping.human_height_assumption / height
    root, j = frames[:, names.index(mapping.human_root_name), :3], names.index(entry.human_joint)
    offset = Rotation.from_quat(entry.rotation_offset, scalar_first=True) if np.any(entry.rotation_offset) else None
    offset = offset or Rotation.identity()  # a zero offset is none
    place = target[:, :3, 3] + [0, 0, mapping.ground_height] - target[:, :3, :3] @ entry.position_offset
    scale = [mapping.scale_table.get(n, 1.0) for n in (mapping.human_root_name, entry.human_joint)]
    frames[:, j, :3] = root + (place - root * scale[0] * s) / (scale[1] * s)
    frames[:, j, 3:] = (Rotation.from_matrix(target[:, :3, :3]) * offset.inv()).as_quat(scalar_first=True)


def retarget_person(smplx: Path, out: Path, hands: Path | None = None, robot: str = "g1_inspire",
                    device: str | None = None) -> dict:  # fmt: skip
    """out/motion.npz: GVHMR's `smplx` person, its hands from WiLoR's `hands` (else SMPL-X's relaxed hand), on the
    humanoid `robot` (`humanoid`), one row per video frame: `fps`, `qpos` [T, 7 + joints] (root xyz, wxyz, then
    `joint_names`, RoboKit's actuated order), each of `link_names`' world turn `turns` [T, L, 3, 3] and origin
    `positions` [T, L, 3] (the feet on z = 0 where `planted` [T, 4], `feet.down`'s heels and toes) and
    `studio_camera_direction` toward the video's camera; out/robot.glb, the link meshes (`glb`). The body is solved
    frame after frame, then again with the person's feet moved where they plant the robot's (`plant`, `aim`), a foot
    that sank asked up (`feet.lift`). Returns per side whether WiLoR gave the hand and whether the robot's hand was
    retargeted."""
    from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
    from robokit.smplx import SMPLX_JOINT_NAMES
    from robokit.xform.numpy import pose7_to_tf_mat

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    poses, relaxed, camera, _, height, fps, toward, planted = person(smplx, out)
    tracks = {side: (points, None) for side, points in relaxed.items()} | (seen_hands(hands, poses, camera) if hands else {})  # fmt: skip

    b = humanoid(robot, fps)
    body = HumanoidRetargeterOnline(b.mapping, robot=b.kin, device=device)
    body.warmup(1)
    names = list(body.human_joint_names)
    frames = poses[:, [SMPLX_JOINT_NAMES.index(n) for n in names]].astype(np.float32)
    tall = np.array([height], np.float32)

    def solve() -> np.ndarray:
        body.reset()
        for _ in range(BODY_SETTLE):
            body.solve_numpy(frames[:1], tall)
        return np.concatenate([body.solve_numpy(frames[t : t + 1], tall) for t in range(len(frames))])

    tracked, sole = soles(b)
    planted_at, up = plant(b, tracked, sole, solve(), planted, fps), np.zeros((len(poses), 2))
    for _ in range(LIFTS):  # the feet planted, solved again: a foot that sank asked up (`feet.lift`)
        for k, link in enumerate(tracked):
            asked = planted_at[:, k].copy()
            asked[:, 2, 3] += up[:, k]
            aim(b.mapping, names, frames, link, asked, height)
        solved = solve()
        at = b.kin.forward_kinematics_via_matrix_numpy(solved[:, 7:], pose7_to_tf_mat(solved[:, :7]), device="cpu")
        at = at[:, [b.kin.spec.link_names.index(n) for n in tracked]]
        settled = feet.lift(up, feet.lowest(sole, at[..., :3, :3], at[..., :3, 3]).reshape(-1, 2, 2).min(2))
        if np.abs(settled - up).max() < feet.SINK_M:
            break
        up = settled
    names = list(b.robot.spec.actuated_joint_names)
    qpos = np.zeros((len(poses), 7 + len(names)), np.float32)
    qpos[:, :7] = solved[:, :7]
    qpos[:, [7 + names.index(n) for n in b.kin.spec.actuated_joint_names]] = solved[:, 7:]
    for side, hand in b.hands.items():
        limits = b.robot.spec.actuated_joint_limits[hand.columns]
        fingers = fit_fingers(hand, tracks[side][0], side == "right", device)
        qpos[:, [7 + c for c in hand.columns]] = np.clip(fingers, limits[:, 0], limits[:, 1])
    low = feet.lowest(sole, at[..., :3, :3], at[..., :3, 3])  # the floor where planted soles stand, else lowest
    floor = np.median(np.where(planted, low, np.inf).min(1)[planted.any(1)]) if planted.any() else low.min()
    links = b.robot.forward_kinematics_via_matrix_numpy(qpos[:, 7:], pose7_to_tf_mat(qpos[:, :7]), device="cpu")
    qpos[:, 2] -= floor
    links[:, :, 2, 3] -= floor
    np.savez(out / "motion.npz", fps=fps, qpos=qpos, joint_names=np.array(names), robot=robot,
             link_names=np.array(b.robot.spec.link_names), turns=links[..., :3, :3], positions=links[..., :3, 3],
             studio_camera_direction=toward, planted=planted)  # fmt: skip
    glb(b.robot, out / "robot.glb")
    return {side: dict(seen=turn is not None, retargeted=side in b.hands) for side, (_, turn) in tracks.items()}
