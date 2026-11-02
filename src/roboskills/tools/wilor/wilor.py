# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "torch==2.6.0", "torchvision==0.21.0", "numpy==1.26.4", "opencv-python==4.11.0.86", "ultralytics==8.1.34",
#   "smplx==0.1.28", "pytorch-lightning==2.6.6", "timm==1.0.30", "einops==0.8.2", "yacs==0.1.8",
#   "pyrender==0.1.45", "PyOpenGL==3.1.0", "trimesh==5.1.1", "scikit-image==0.25.2", "pandas==2.3.3",
#   "hydra-core==1.3.7", "hydra-colorlog==1.2.0", "hydra-submitit-launcher==1.2.0", "pyrootutils==1.0.4",
#   "rich==15.0.0", "webdataset==1.0.2", "xtcocotools==1.14.3", "dill==0.4.1",  # dill: detector.pt is its pickle
#   "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da",
# ]
#
# [tool.uv]
# extra-build-dependencies = { chumpy = ["pip", "numpy==1.26.4"] }  # chumpy's setup imports both
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# torchvision = { index = "pytorch-cu126" }
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
"""Hands in a video with WiLoR, per frame a detector's boxes and per box MANO's turns and keypoints, in its own
environment (`uv run` of this file: the verified environment's versions). WiLoR's checkout and released weights are
made once in ~/.roboskills/weights/wilor (`prepare`); it is not a package, so put it on PYTHONPATH. `usable(hands,
frames)` reads its hands.npz as one left and one right hand per frame.

    python -m roboskills.tools.wilor --video V --out hands/
"""

import hashlib
import os
import subprocess
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import numpy as np


CHECKOUT = Path.home() / ".roboskills/weights/wilor"
SOURCE = "https://github.com/rolpotamias/WiLoR.git", "fcb911312a38fa8badd30d9656a167485d61b8f9"
RELEASE = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/99fe3d7acff8104ecca1055df7467709506c2fa6"
WEIGHTS = {
    "wilor_final.ckpt": "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2",
    "detector.pt": "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
}
GAP = 10  # frames a hand may go undetected before its track is dropped (a third of a second at 30 fps)


def prepare(mano: Path = Path.home() / ".robokit/body_models/mano") -> Path:
    """~/.roboskills/weights/wilor, made once: WiLoR's checkout at its pinned commit, reading MANO from its own
    mano_data/ from any directory, the released wilor_final.ckpt and detector.pt (SHA-256 checked) in
    pretrained_models/; and mano_data/MANO_RIGHT.pkl linked to `mano`'s (licensed; a left hand is the right one
    mirrored)."""
    if not CHECKOUT.is_dir():
        part = CHECKOUT.with_name("wilor.partial")
        subprocess.run(["git", "init", "-q", part], check=True)
        subprocess.run(["git", "-C", part, "fetch", "-q", "--depth", "1", *SOURCE], check=True)
        subprocess.run(["git", "-C", part, "checkout", "-q", "FETCH_HEAD"], check=True)
        models = part / "wilor/models/__init__.py"
        models.write_text(models.read_text().replace("'./mano_data/", f"'{CHECKOUT}/mano_data/"))
        for name, sha256 in WEIGHTS.items():
            urllib.request.urlretrieve(f"{RELEASE}/pretrained_models/{name}", part / "pretrained_models" / name)
            if hashlib.sha256((part / "pretrained_models" / name).read_bytes()).hexdigest() != sha256:
                raise SystemExit(f"{name}: SHA-256 mismatch")
        part.rename(CHECKOUT)
    link = CHECKOUT / "mano_data/MANO_RIGHT.pkl"
    link.unlink(missing_ok=True)
    link.symlink_to(Path(mano).expanduser().resolve() / "MANO_RIGHT.pkl")
    return CHECKOUT


def detect_hands(video: Path, out: Path, mano: Path = Path.home() / ".robokit/body_models/mano"):
    """out/hands.npz, one row per detected hand: `frame`, `box` [x0, y0, x1, y1] and `right` (the detector's label),
    MANO's `global_orient` (3, 3), `hand_pose` (15, 3, 3, each joint in its parent) and `betas` (10,), `keypoints_3d`
    (21, 3) in WiLoR's camera frame (focal `focal`, centred) and their `keypoints_2d` (21, 2) in image pixels."""
    checkout = prepare(mano)
    Path(out).mkdir(parents=True, exist_ok=True)
    argv = ["uv", "run", "--quiet", __file__, str(Path(video).resolve()), str(Path(out).resolve() / "hands.npz")]
    env = dict(os.environ, PYTHONPATH=str(checkout), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYOPENGL_PLATFORM="egl",
               PYGLET_HEADLESS="1")  # fmt: skip
    subprocess.run(argv, check=True, env=env)


def detect(video, out):
    """`detect_hands`' model side."""
    import cv2
    import torch
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils.renderer import cam_crop_to_full

    weights = CHECKOUT / "pretrained_models"
    model, cfg = load_wilor(str(weights / "wilor_final.ckpt"), str(weights / "model_config.yaml"))
    model, detector = model.cuda().eval(), YOLO(str(weights / "detector.pt"))
    rows, capture, frame, focal = [], cv2.VideoCapture(video), 0, np.nan
    while (got := capture.read())[0]:
        found = detector(got[1], conf=0.3, verbose=False)[0].boxes
        if len(found):
            data = ViTDetDataset(cfg, got[1], found.xyxy.cpu().numpy(), found.cls.cpu().numpy(), rescale_factor=2.0)
            batch = torch.utils.data.default_collate([data[i] for i in range(len(data))])
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                o = model(batch)
            sign, size = 2 * batch["right"] - 1, batch["img_size"].float()
            one = torch.ones_like(sign)
            cam = o["pred_cam"] * torch.stack([one, sign, one], 1)
            focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * size.max()
            cam_t = cam_crop_to_full(cam, batch["box_center"].float(), batch["box_size"].float(), size, focal)
            joints = o["pred_keypoints_3d"] * torch.stack([sign, one, one], 1)[:, None] + cam_t[:, None]
            uv = focal * joints[..., :2] / joints[..., 2:] + size[:, None] / 2
            mano = o["pred_mano_params"]
            for i in range(len(found)):
                rows.append(dict(frame=frame, box=found.xyxy[i].cpu().numpy(), right=bool(batch["right"][i]),
                                 global_orient=mano["global_orient"][i, 0], hand_pose=mano["hand_pose"][i],
                                 betas=mano["betas"][i], keypoints_3d=joints[i], keypoints_2d=uv[i]))  # fmt: skip
        frame += 1
    keys = ["frame", "box", "right", "global_orient", "hand_pose", "betas", "keypoints_3d", "keypoints_2d"]
    as_array = lambda v: v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)  # noqa: E731
    np.savez_compressed(out, focal=float(focal), **{k: np.array([as_array(r[k]) for r in rows]) for k in keys})


def assign(h: Mapping[str, np.ndarray], frames: int) -> dict[str, dict[int, int]]:
    """Which detection row is each side's hand per frame: the detector's left/right label is often wrong, so a hand
    keeps to the detection whose wrist (in the picture) is nearest its last one; the label only breaks ties and seeds
    the track. A detection that jumps more than 1.5 hand-box diagonals from the side's last wrist is dropped; a side
    unassigned for `GAP` frames (turned away, out of view) forgets its last wrist and the label seeds it again."""
    wrist, box, label = h["keypoints_2d"][:, 0], h["box"], h["right"].astype(bool)
    size = np.median(np.linalg.norm(box[:, 2:] - box[:, :2], axis=1))
    by_frame = {}
    for k, f in enumerate(h["frame"]):
        by_frame.setdefault(int(f), []).append(k)
    last, seen = {"Left": None, "Right": None}, {"Left": 0, "Right": 0}
    out = {"Left": {}, "Right": {}}
    for t in range(frames):
        last = {side: w if t - seen[side] <= GAP else None for side, w in last.items()}
        rows = by_frame.get(t, [])[:3]
        best, best_cost = {}, np.inf
        options = [(a, b) for a in [None, *rows] for b in [None, *rows] if a is None or a != b]
        for left, right in options:
            cost = 0.0
            for side, k, want in (("Left", left, False), ("Right", right, True)):
                if k is None:
                    cost += 1.0  # leaving a hand unassigned
                    continue
                cost += 0.3 * (label[k] != want)
                if last[side] is not None:
                    jump = np.linalg.norm(wrist[k] - last[side]) / size
                    cost += jump if jump < 1.5 else np.inf
            if cost < best_cost:
                best, best_cost = {"Left": left, "Right": right}, cost
        for side, k in best.items():
            if k is not None:
                out[side][t], last[side], seen[side] = k, wrist[k], t
    return out


def usable(h: Mapping[str, np.ndarray], frames: int) -> dict[str, dict[int, int]]:
    """`assign`'s detections that can be used as they are: one whose label is not its side was read with the other
    hand's chirality, and one turned more than 60 degrees from most of its detected neighbours within 4 frames was read
    flipped (palm in for palm out); both are dropped."""
    from scipy.spatial.transform import Rotation  # not in WiLoR's own environment

    out = {}
    for side, rows in assign(h, frames).items():
        rows = {t: k for t, k in rows.items() if bool(h["right"][k]) == (side == "Right")}
        turn = {t: Rotation.from_matrix(h["global_orient"][k]) for t, k in rows.items()}
        out[side] = {}
        for t, k in rows.items():
            near = [u for u in rows if u != t and abs(u - t) <= 4]
            if len(near) < 2 or np.median([(turn[t].inv() * turn[u]).magnitude() for u in near]) < np.radians(60):
                out[side][t] = k
    return out


if __name__ == "__main__":
    sys.path.pop(0)  # this file's folder, so `import wilor` is the checkout's (Python 3.10 has no PYTHONSAFEPATH)
    detect(*sys.argv[1:])
