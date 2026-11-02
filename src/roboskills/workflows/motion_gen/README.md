# Motion Generation

Text, image or video in, rendered motion out. The agent ([SKILL.md](SKILL.md), the `motion-gen` skill) picks the models, writes its own scripts,
renders, looks at the frames and revises. A video's person or a text's motion on a Mixamo character or a RoboKit robot also runs alone as code, agent-free.

## Any request

The agent calls these tools and writes the rest in its run directory.

| Tool                                    | Call                                                     |
| --------------------------------------- | -------------------------------------------------------- |
| [Kimodo](../../tools/kimodo/SKILL.md)   | text to G1 or human motion                               |
| [GVHMR](../../tools/gvhmr/SKILL.md)     | a video's person as SMPL-X body motion                   |
| [WiLoR](../../tools/wilor/SKILL.md)     | a video's hands                                          |
| [SAM 3](../../tools/sam3/SKILL.md)      | an object's masks from text                              |
| [Character](character/SKILL.md)         | motion onto a Mixamo character, its joints and contact   |
| [RoboKit](../../tools/robokit/SKILL.md) | robot kinematics, retargeting and motion planning        |
| [Blender](../../tools/blender/SKILL.md) | `motion_studio.py`: an animated GLB and its saved motion |

Runs from a clone of this repository on Linux with an NVIDIA GPU (we use 24 GB cards), [uv](https://docs.astral.sh/uv/)
and ffmpeg, in about 60 GB of disk; the first calls download about 30 GB of weights. Some models are licensed or gated,
so they come with your own account; get only what you make:

| To make                                  | Needs                                                                                        |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| a video's person on a character or robot | [SMPL-X](https://smpl-x.is.tue.mpg.de)                                                       |
| ... with the hands                       | and [MANO](https://mano.is.tue.mpg.de) (without it, the body alone)                          |
| a text's motion on a character or robot  | SMPL-X and [Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| anything, by the agent                   | what the request needs, and [SAM 3](https://huggingface.co/facebook/sam3) for its objects    |

Download SMPL-X (`models_smplx_v1_1.zip`) and MANO (`mano_v1_2.zip`) from their sites (an account each); setup puts
their files in `~/.robokit/body_models/` and ends by naming what is still missing. Llama 3 and SAM 3 need access
granted on Hugging Face and `uv run hf auth login`. Setup also downloads four ready characters (athlete, dancer,
hoodie, bob) into `runs/motion_gen/characters/`; any Mixamo character works too.

```sh
git clone https://github.com/openrobokit/roboskills && cd roboskills && uv sync  # every command below runs from here
SMPLX_ZIP=<models_smplx_v1_1.zip> MANO_ZIP=<mano_v1_2.zip> bash src/roboskills/workflows/motion_gen/setup.sh  # once per machine
# The agent in this repo (the motion-gen skill). Claude's auto mode has each command checked before it runs;
# --permission-mode bypassPermissions and Codex's flag run every command unasked: use them on a machine you trust
claude  # then: /motion-gen make runs/motion_gen/characters/dancer.glb wave.
claude -p --permission-mode auto "/motion-gen make runs/motion_gen/characters/dancer.glb wave."
claude -p --permission-mode auto "/motion-gen put the person in assets/example_data/tennis.mp4 on runs/motion_gen/characters/dancer.glb."
codex exec --dangerously-bypass-approvals-and-sandbox '$motion-gen make runs/motion_gen/characters/dancer.glb wave.'
codex exec --dangerously-bypass-approvals-and-sandbox '$motion-gen make runs/motion_gen/characters/dancer.glb wave. Deliver within 30 minutes.'  # a time limit only when you want one
# The agent anywhere: it clones and sets up the repository itself
claude -p --permission-mode auto "Follow the motion-gen skill in github.com/openrobokit/roboskills to make its dancer character wave."
```

## Video to character

Recover the person, put the motion on the character, render: [`pipeline.py`](pipeline.py), a folder per tool, each with
a `status.json`. One person, body and hands; on a robot, `robot/motion.npz` and `blender/scene.glb` (its link meshes
animated) line up with the video frame for frame.

| Tool                                             | Call                                                                           | Writes      |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | ----------- |
| [GVHMR](../../tools/gvhmr/SKILL.md)              | `recover_motion`: the person's SMPL-X body per frame                           | `motion/`   |
| [WiLoR](../../tools/wilor/SKILL.md)              | `detect_hands`: the hands per frame (needs MANO, else left out)                | `hands/`    |
| [`character/retarget.py`](character/retarget.py) | `retarget_motion`: the motion on the character's own proportions               | `retarget/` |
| [`character/fingers.py`](character/fingers.py)   | `retarget_fingers`: the hands' turns and fingers on the character's            | `fingers/`  |
| [RoboKit](../../tools/robokit/SKILL.md)          | `retarget_person` (`--robot`, for the two above): the person on a robot        | `robot/`    |
| [Blender](../../tools/blender/SKILL.md)          | `render_motion`: the animated GLB, every frame, source and render side by side | `blender/`  |

```sh
# A video on a Mixamo character (FBX, or a GLB Blender wrote): blender/video.mp4 and compare.mp4, a few minutes
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --video assets/example_data/tennis.mp4 --character runs/motion_gen/characters/dancer.glb
# A moving camera
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/walk --video your_video.mp4 --character runs/motion_gen/characters/dancer.glb --moving-camera
# The same motion on another character
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --character runs/motion_gen/characters/athlete.glb --tools retarget fingers blender --redo retarget fingers blender
# A faster render
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis --tools blender --redo blender --engine eevee
# A RoboKit humanoid instead of a character: g1_inspire's fingers follow the hands; g1, h1, h1_2, gr3 and bhl ran too,
# their hands kept as they are (the robots run: tools/robokit/SKILL.md)
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/tennis-g1 --video assets/example_data/tennis.mp4 --robot g1_inspire
# A text instead of a video: kimodo (Kimodo's SMPL-X body), retarget, fingers (its relaxed hand), blender (video.mp4 alone)
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/wave --text "a person waves with the right hand" --duration 5 --character runs/motion_gen/characters/dancer.glb
# A text on a robot
uv run python -m roboskills.workflows.motion_gen --workspace runs/motion_gen/wave-g1 --text "a person waves with the right hand" --duration 5 --robot g1_inspire
```
