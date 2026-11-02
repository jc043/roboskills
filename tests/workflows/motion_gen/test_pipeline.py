"""The agent-free pipeline's status.json per tool, skipping and its inputs (roboskills.workflows.motion_gen.pipeline), on fake tools."""

import json

import numpy as np
import pytest

from roboskills.workflows.motion_gen import pipeline


status = lambda w, name: json.loads((w / name / "status.json").read_text())  # noqa: E731


@pytest.fixture
def fakes(monkeypatch):
    calls = []

    def tool(name, output):
        def function(w, o, p):
            calls.append(name)
            o.mkdir(parents=True, exist_ok=True)
            (w / output).write_text(json.dumps(p))

        return function, output

    outputs = (("kimodo", "kimodo/motion_amass.npz"), ("gvhmr", "motion/smplx.npz"), ("wilor", "hands/hands.npz"),
               ("retarget", "retarget/motion.npz"), ("fingers", "fingers/motion.npz"), ("robot", "robot/motion.npz"),
               ("blender", "blender/video.mp4"))  # fmt: skip
    tools = {n: tool(n, o) for n, o in outputs}
    monkeypatch.setattr(pipeline, "TOOLS", tools)
    monkeypatch.setattr(pipeline, "MANO", pipeline.Path(__file__))  # any file that exists: MANO is there
    monkeypatch.setattr(
        pipeline, "CHECKS", {"retarget": lambda w: dict(ok=False, reason="a limb strays", limb_mm=120.0)}
    )
    return calls


def test_every_tool_reports_and_a_finished_tool_is_skipped(tmp_path, fakes):
    pipeline.run(tmp_path, tools=["blender", "gvhmr", "retarget"], moving_camera=True, engine="eevee")
    assert fakes == ["gvhmr", "retarget", "blender"]
    assert status(tmp_path, "motion")["ok"] and status(tmp_path, "blender")["seconds"] is not None
    assert json.loads((tmp_path / "motion/smplx.npz").read_text()) == dict(
        moving_camera=True, engine="eevee", duration=5.0
    )
    failed = status(tmp_path, "retarget")  # a check that fails is recorded; the later tools still run
    assert not failed["ok"] and failed["limb_mm"] == 120.0 and failed["reason"] == "a limb strays"
    pipeline.run(tmp_path, redo=["retarget"])  # every video tool: the hands' two run for the first time, retarget again
    assert fakes == ["gvhmr", "retarget", "blender", "wilor", "retarget", "fingers"]


def test_a_text_runs_kimodo_and_its_smplx_file_through_retarget_fingers_and_blender(tmp_path, monkeypatch):
    calls = {}
    for name in ("generate_motion", "retarget_motion", "smplx_fingers", "render_motion"):
        monkeypatch.setattr(pipeline, name, lambda *a, name=name, **k: calls.setdefault(name, a))
    monkeypatch.setattr(pipeline, "CHECKS", {})
    pipeline.run(tmp_path, text="a person waves", duration=3.0)
    assert (tmp_path / pipeline.TEXT).read_text() == "a person waves"
    assert list(calls) == ["generate_motion", "retarget_motion", "smplx_fingers", "render_motion"]
    assert calls["generate_motion"][:3] == ("a person waves", tmp_path / "kimodo", 3.0)
    assert calls["retarget_motion"][0] == calls["smplx_fingers"][2] == tmp_path / "kimodo/motion_amass.npz"
    assert calls["render_motion"][2] is None  # no source video: no compare.mp4


def test_a_video_without_mano_runs_the_body_alone(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(pipeline, "MANO", tmp_path / "MANO_RIGHT.pkl")
    (tmp_path / "input").mkdir()
    (tmp_path / "input/video.mp4").write_text("")
    pipeline.run(tmp_path)
    assert fakes == ["gvhmr", "retarget", "blender"]


def test_a_robot_runs_the_entrys_tools_its_retarget_and_blender(tmp_path, fakes, monkeypatch):
    (tmp_path / "input").mkdir()
    (tmp_path / "input/video.mp4").write_text("")
    pipeline.run(tmp_path, robot="g1_inspire")
    assert fakes == ["gvhmr", "wilor", "robot", "blender"] and (tmp_path / pipeline.ROBOT).read_text() == "g1_inspire"
    assert pipeline.actor(tmp_path) == tmp_path / "robot/robot.glb"
    assert pipeline.posed(tmp_path) == tmp_path / pipeline.ROBOT_MOTION
    fakes.clear()
    pipeline.run(tmp_path / "text", text="a person waves", robot="g1_inspire")
    assert fakes == ["kimodo", "robot", "blender"]
    assert pipeline.body(tmp_path / "text") == tmp_path / "text/kimodo/motion_amass.npz"


def test_a_finished_tool_without_a_status_is_only_checked(tmp_path, fakes):
    (tmp_path / "retarget").mkdir()
    (tmp_path / "retarget/motion.npz").write_text("kept")
    pipeline.run(tmp_path, tools=["retarget"])
    assert fakes == [] and (tmp_path / "retarget/motion.npz").read_text() == "kept"
    assert status(tmp_path, "retarget")["seconds"] is None


def test_a_raise_is_recorded_and_stops_the_run(tmp_path, fakes, monkeypatch):
    def boom(w, o, p):
        o.mkdir(parents=True)
        raise RuntimeError("no person")

    monkeypatch.setitem(pipeline.TOOLS, "gvhmr", (boom, "motion/smplx.npz"))
    with pytest.raises(RuntimeError):
        pipeline.run(tmp_path)
    assert fakes == [] and "no person" in status(tmp_path, "motion")["reason"]


def test_an_fbx_becomes_input_character_glb_and_a_glb_is_copied(tmp_path, fakes, monkeypatch):
    imported = []
    monkeypatch.setattr(pipeline, "import_character", lambda source, out: imported.append((source, out)))
    (tmp_path / "C.FBX").write_text("rig")
    (tmp_path / "v.mp4").write_text("video")
    w = tmp_path / "w"
    pipeline.run(w, video=tmp_path / "v.mp4", character=tmp_path / "C.FBX", tools=[])
    assert imported == [(w / "input/character.fbx", w / pipeline.CHARACTER)]
    assert (w / pipeline.VIDEO).read_text() == "video"
    (tmp_path / "C.glb").write_text("glb")
    pipeline.run(w, character=tmp_path / "C.glb", tools=[])
    assert (w / pipeline.CHARACTER).read_text() == "glb" and len(imported) == 1


def test_check_video_needs_every_frame_at_the_motions_fps(tmp_path, monkeypatch):
    np.savez(tmp_path / "motion.npz", fps=30.0, turns=np.zeros((120, 2, 3, 3)))
    probe = type("Probe", (), dict(stdout='{"streams": [{"nb_read_frames": "119", "avg_frame_rate": "30/1"}]}'))
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: probe)
    report = pipeline.check_video(tmp_path / "v.mp4", tmp_path / "motion.npz")
    assert not report["ok"] and report["frames"] == 119 and "119 frames" in report["reason"]
    probe.stdout = '{"streams": [{"nb_read_frames": "120", "avg_frame_rate": "30/1"}]}'
    assert pipeline.check_video(tmp_path / "v.mp4", tmp_path / "motion.npz")["ok"]
