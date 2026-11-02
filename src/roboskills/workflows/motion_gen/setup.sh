#!/usr/bin/env bash
# Make the tools SKILL.md names (Kimodo, GVHMR, Blender, WiLoR, SAM 3) ready, each in its own uv environment (its file's
# inline dependencies in src/roboskills/tools/<tool>/), and write their setup files into runs/motion_gen/env/.
#
#   bash src/roboskills/workflows/motion_gen/setup.sh [ENV_DIR]
#
# Needs uv and the repository environment (uv sync). Licensed or gated, so yours to provide: SMPL-X in
# ~/.robokit/body_models/smplx, MANO's right hand in ~/.robokit/body_models/mano (for WiLoR; without it a video's hands
# are left out), and a Hugging Face token with access to Llama 3 (Kimodo's text encoder) and facebook/sam3. Given the
# SMPL-X and MANO zips you downloaded from their sites (SMPLX_ZIP=..., MANO_ZIP=...), it puts their files in place. The
# script ends by saying which of these is missing. Weights download on first use: Kimodo's and SAM 3's into the Hugging
# Face cache, GVHMR's and WiLoR's into ~/.roboskills/weights/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
E=$(mkdir -p "${1:-$REPO/runs/motion_gen/env}/bin" && cd "${1:-$REPO/runs/motion_gen/env}" && pwd)
TOOLS=$REPO/src/roboskills/tools
BODY=$HOME/.robokit/body_models
CHARACTERS_REVISION=5e6107d0d19df2c3542206437d922560d604320e  # openrobokit/roboskills on Hugging Face
[ -x "$REPO/.venv/bin/python" ] || { echo "run uv sync in $REPO first" >&2; exit 1; }

# SMPL-X and MANO are licensed: download each from its site yourself; given its zip, its model file goes into $BODY
place() {  # place NAME ZIP FILE: FILE ($BODY/<dir>/<name>) out of the ZIP you downloaded, an .npz checked to load
  [ -n "$2" ] && [ ! -f "$BODY/$3" ] || return 0
  "$REPO/.venv/bin/python" -c 'import io, sys, zipfile; from pathlib import Path; import numpy as np
z, out = zipfile.ZipFile(sys.argv[1]), Path(sys.argv[2])
data = z.read(next(n for n in z.namelist() if n.endswith("/" + out.name)))
out.suffix == ".npz" and np.load(io.BytesIO(data))["v_template"]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(data)' "$2" "$BODY/$3" 2> /dev/null && echo "$1: $BODY/$3" || echo "$1: no usable $(basename "$3") in $2"
}
place SMPL-X "${SMPLX_ZIP:-}" smplx/SMPLX_NEUTRAL.npz
place MANO "${MANO_ZIP:-}" mano/MANO_RIGHT.pkl

# Each tool's python: its file's own environment, made once
python_of() {  # python_of SCRIPT
  uv sync --quiet --script "$1" && uv python find --script "$1"
}
KIMODO=$(python_of "$TOOLS/kimodo/kimodo.py")
GVHMR=$(python_of "$TOOLS/gvhmr/gvhmr.py")
BLENDER=$(python_of "$TOOLS/blender/motion_studio.py")
uv sync --quiet --script "$TOOLS/sam3/sam3.py"
# WiLoR's checkout (not a package) and weights, made once, go on its python's path; it needs MANO
WILOR=
rm -f "$E/wilor.sh"
if [ -f "$BODY/mano/MANO_RIGHT.pkl" ]; then
  WILOR=$("$REPO/.venv/bin/python" -c 'from roboskills.tools.wilor.wilor import prepare; print(prepare())')
  printf '#!/bin/sh\nPYTHONPATH=%s TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=1 exec %s "$@"\n' \
    "$WILOR" "$(python_of "$TOOLS/wilor/wilor.py")" > "$E/bin/wilor-python"
  chmod +x "$E/bin/wilor-python"
fi

cat > "$E/kimodo.sh" <<EOF
# Kimodo: text-to-motion, native Unitree G1, SOMA human and SMPL-X human (tools/kimodo: source 1aece8c1; the models
# from the Hugging Face cache). Sourcing this file says whether a text encoder answers at TEXT_ENCODER_URL. Generating
# encodes the prompt with it or, when none answers, in its own process (about 15 GB more GPU memory, and a minute or two
# per call to load it). For several calls start one encoder (ready after about 40 s) on port 9571 and stop it when
# done; another run on the same machine needs another port, GRADIO_SERVER_PORT for the encoder and TEXT_ENCODER_URL to
# match for generate.
#   \$KIMODO_PYTHON -m kimodo.scripts.run_text_encoder_server --tmp-folder /tmp/encoder &
# Generate (writes STEM.npz and STEM.csv; with the SOMA model, --bvh writes STEM.bvh; --model Kimodo-SMPLX-RP-v1 writes
# STEM_amass.npz, an AMASS SMPL-X file the character skill's retarget.retarget_motion takes):
#   \$KIMODO_PYTHON -m kimodo.scripts.generate "PROMPT" --model \$KIMODO_MODEL --duration 4 --seed 0 --output STEM [--constraints c.json]
export KIMODO_PYTHON=$KIMODO KIMODO_MODEL=Kimodo-G1-RP-v1 MOTION_KIMODO_HUMAN_MODEL=Kimodo-SMPLX-RP-v1
export GRADIO_SERVER_NAME=127.0.0.1 GRADIO_TEMP_DIR=/tmp/gradio
export GRADIO_SERVER_PORT=\${GRADIO_SERVER_PORT:-9571} TEXT_ENCODER_URL=\${TEXT_ENCODER_URL:-http://127.0.0.1:9571}
curl -sf -m 3 -o /dev/null "\$TEXT_ENCODER_URL/config" && echo "Kimodo: a text encoder answers at \$TEXT_ENCODER_URL" \\
  || echo "Kimodo: no text encoder at \$TEXT_ENCODER_URL; start one as above"
EOF
cat > "$E/gvhmr.sh" <<EOF
# GVHMR 1.6.1 (tools/gvhmr) for video to SMPL-X motion. For example:
#   \$GVHMR_PYTHON -c 'import gvhmr; r=gvhmr.recover("in.mp4", static_camera=True, device="cuda", output_root="out", output_dir="out/in", progress=False); r.save_npz("out/in/smplx.npz")'
export GVHMR_PYTHON=$GVHMR OMP_NUM_THREADS=\${OMP_NUM_THREADS:-16}
export GVHMR_CHECKPOINTS=$HOME/.roboskills/weights/gvhmr GVHMR_BODY_MODELS=$BODY
EOF
cat > "$E/blender.sh" <<EOF
# Blender 4.5 as the bpy module (tools/blender). Cycles uses the run's GPUs; EEVEE and Workbench open the first. For example:
#   \$BLENDER_PYTHON -c 'import bpy; bpy.ops.import_scene.fbx(filepath="char.fbx"); print([o.name for o in bpy.data.objects])'
export BLENDER_PYTHON=$BLENDER
export MOTION_STUDIO="$REPO/src/roboskills/tools/blender/motion_studio.py"
EOF
cat > "$E/character.sh" <<EOF
# The character retargeting's functions run on the host (this repository's environment).
. "$E/gvhmr.sh"
. "$E/blender.sh"
export ROBOSKILLS_PYTHON="$REPO/.venv/bin/python"
export CHARACTER_SKILL="$REPO/src/roboskills/workflows/motion_gen/character/SKILL.md"
EOF
[ -n "$WILOR" ] && cat > "$E/wilor.sh" <<EOF
# WiLoR (tools/wilor: source fcb91131; its checkout and weights in $WILOR, on WILOR_PYTHON's path, MANO linked) for a
# video's hands: per image, a detector's hand boxes (cls 1 right, 0 left), and per box MANO's global_orient, hand_pose
# (15 joints, each in its parent), pred_keypoints_3d [21, 3] and pred_cam. The model is a 641 M-parameter ViT-H
# (2.6 GB), slow to load: load it once. Batching crops (from any frames) runs the model several times faster per hand
# than one crop at a time, and the detector likewise on lists of frames; cutting each crop out on the CPU
# (ViTDetDataset) then costs the most, so prepare crops in parallel. Save results as you go. For example:
#   "\$WILOR_PYTHON" - frame.jpg <<'PY'
#   import sys, cv2, torch; from ultralytics import YOLO; from wilor.models import load_wilor; from wilor.datasets.vitdet_dataset import ViTDetDataset
#   W = "$WILOR/pretrained_models/"; model, cfg = load_wilor(W + "wilor_final.ckpt", W + "model_config.yaml"); model = model.cuda().eval()
#   image = cv2.imread(sys.argv[1]); found = YOLO(W + "detector.pt")(image, conf=0.3, verbose=False)[0].boxes
#   data = ViTDetDataset(cfg, image, found.xyxy.cpu().numpy(), found.cls.cpu().numpy(), rescale_factor=2.0)
#   batch = {k: v.cuda() for k, v in torch.utils.data.default_collate([data[i] for i in range(len(data))]).items()}
#   with torch.no_grad(): out = model(batch)  # out["pred_mano_params"]["global_orient"] [N,1,3,3], ["hand_pose"] [N,15,3,3]
#   PY
export WILOR_PYTHON=$E/bin/wilor-python WILOR=$WILOR
EOF
cat > "$E/sam3.sh" <<EOF
# SAM 3 (tools/sam3: facebook/sam3 from the Hugging Face cache) for an object's masks from text prompts, per image;
# detect frame by frame. roboskills.tools.sam3.sam3.Sam, from this repository's python, keeps the model loaded in its
# own uv environment's worker process on the run's first GPU; images go in over a pipe.
# For example:
#   "\$ROBOSKILLS_PYTHON" - frame.png <<'PY'
#   import sys, numpy as np; from PIL import Image; from roboskills.tools.sam3.sam3 import Sam
#   sam = Sam()  # loads once: detect every image you need, then close
#   found = sam.detect(np.asarray(Image.open(sys.argv[1]).convert("RGB")), ["umbrella", "parasol"], threshold=0.3)
#   sam.close()  # found: {prompt: [{score, box_px, area_px, mask (H, W) bool}]}, best first
#   PY
export ROBOSKILLS_PYTHON="$REPO/.venv/bin/python"
EOF
# Ready characters: athlete, dancer, hoodie and bob from openrobokit/roboskills on Hugging Face (MIT), pinned
for c in athlete dancer hoodie bob; do
  [ -f "$REPO/runs/motion_gen/characters/$c.glb" ] || "$REPO/.venv/bin/python" -c 'import shutil, sys; from pathlib import Path
from huggingface_hub import hf_hub_download
out = Path(sys.argv[3]); out.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(hf_hub_download("openrobokit/roboskills", f"motion_gen/characters/{sys.argv[1]}.glb", repo_type="dataset", revision=sys.argv[2]), out)' \
    "$c" "$CHARACTERS_REVISION" "$REPO/runs/motion_gen/characters/$c.glb"
done
echo "Tools ready in $E; characters in $REPO/runs/motion_gen/characters"

# What is yours to provide and still missing
token=${HF_TOKEN:-$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}  # none yet: not logged in
ls "$BODY"/smplx/SMPLX_NEUTRAL.* > /dev/null 2>&1 ||
  echo "Missing SMPL-X (a video's person, motion on a character): register at https://smpl-x.is.tue.mpg.de, download its models_smplx_v1_1.zip, rerun this with SMPLX_ZIP=<that zip>"
[ -n "$WILOR" ] ||
  echo "Missing MANO (a video's hands, WiLoR): register at https://mano.is.tue.mpg.de, download its mano_v1_2.zip, rerun this with MANO_ZIP=<that zip>"
gated() {  # gated REPO USE
  curl -sfI -o /dev/null -H "Authorization: Bearer $token" "https://huggingface.co/$1/resolve/main/config.json" ||
    echo "Missing Hugging Face access to $1 ($2): request it at https://huggingface.co/$1, then hf auth login"
}
gated meta-llama/Meta-Llama-3-8B-Instruct "Kimodo's text encoder"
gated facebook/sam3 "SAM 3"
