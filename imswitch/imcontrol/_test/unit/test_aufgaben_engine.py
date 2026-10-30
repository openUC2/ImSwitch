"""The Aufgaben executor on a simulated microscope."""
import json
import time

import pytest
import useq
import zarr

from imswitch.imcontrol.model import aufgaben as A
from imswitch.imcontrol.model.aufgaben.engine import Run
from imswitch.imcontrol.model.aufgaben.simulated import SimulatedMicroscope
from imswitch.imcontrol.model.aufgaben.storage import OmeStore

open_ome_zarr = pytest.importorskip("ome_zarr_models").open_ome_zarr
SOURCES = ("LED", "Laser405", "Laser488", "Laser561", "Laser640")


def _job(tasks, **kw):
    return A.Job.model_validate({"name": "t", **kw, "tasks": tasks})


def _wait_for(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _run(job, hw=None, **kw):
    run = Run(job, hw or SimulatedMicroscope(sources=SOURCES), **kw)
    run.run(confirmed=True)
    return run


# --------------------------------------------------------------------------- #
# The example jobs, end to end
# --------------------------------------------------------------------------- #
def test_hcs_example_writes_a_valid_plate(tmp_path):
    job = A.examples()["hcs_live_timelapse"]
    job.resources["time"].plan = useq.TIntervalLoops(interval=0.05, loops=32)
    run = Run(job, SimulatedMicroscope(sources=SOURCES), parameters={"rounds": 2},
              sink=OmeStore(tmp_path / "data"), run_dir=tmp_path)
    assert run.run() == "finished", run.message
    # 6 wells x 3 sites x 7 planes x 4 channels x 2 rounds
    assert run.frames == 6 * 3 * 7 * 4 * 2
    group = zarr.open_group(str(tmp_path / "data" / "img.ome.zarr"), mode="r")
    assert type(open_ome_zarr(group)).__name__ == "HCS"
    assert group["B/2/0/0"].shape == (2, 4, 7, 32, 48)
    lines = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert lines[0]["event"] == "start" and lines[-1]["state"] == "finished"


def test_stop_flow_example_drives_the_fluidics(tmp_path):
    job = A.examples()["stop_flow_fluidics"]
    for node, _ in job.walk():
        if node.task == "wait":
            node.params["seconds"] = 0
    hw = SimulatedMicroscope(sources=SOURCES)
    run = Run(job, hw, parameters={"cycles": 3, "volume_ul": 20}, sink=OmeStore(tmp_path))
    assert run.run() == "finished", run.message
    assert run.variables["images"] == 3 and run.frames == 3
    assert hw.pumped_ul["peristaltic"] == 3 * 20 + 500
    assert hw.valves["inlet"] == "waste" and hw.target_c is None
    assert zarr.open_group(str(tmp_path / "img.ome.zarr"), mode="r")["0"].shape[0] == 3


def test_tile_scan_example_asks_then_scans(tmp_path):
    job = A.examples()["tile_scan_focus_check"]
    events = []
    run = Run(job, SimulatedMicroscope(sources=SOURCES),
              sink=OmeStore(tmp_path, formats=("ome-zarr", "ome-tiff")), on_event=events.append)
    run.start()
    assert _wait_for(lambda: run.question is not None)
    assert run.status()["question"]["buttons"] == ["Start", "Cancel"]
    run.answer("Start")
    run.join(10)
    assert run.state == "finished", run.message
    assert run.frames == 2 * 9 * 3
    assert any(e["type"] == "question" for e in events)
    assert len(list(tmp_path.glob("*.ome.tif"))) == 18
    open_ome_zarr(zarr.open_group(str(tmp_path / "region_a_0004_img.ome.zarr"), mode="r"))


def test_tile_scan_cancel_finishes_without_frames(tmp_path):
    run = Run(A.examples()["tile_scan_focus_check"], SimulatedMicroscope(sources=SOURCES))
    run.start()
    assert _wait_for(lambda: run.question is not None)
    run.answer("Cancel")
    run.join(5)
    assert run.state == "finished" and run.message == "Cancelled by the operator"
    assert run.frames == 0


# --------------------------------------------------------------------------- #
# Control flow
# --------------------------------------------------------------------------- #
def test_break_continue_every_and_else():
    job = _job([
        {"id": "outer", "task": "repeat", "params": {"count": 3}, "body": [
            {"id": "inner", "task": "repeat", "params": {"count": 10}, "body": [
                {"id": "skip_odd", "task": "condition", "params": {"expr": "loop.inner.index % 2"},
                 "body": [{"id": "c", "task": "continue"}],
                 "orelse": [{"id": "count", "task": "set_variable",
                             "params": {"name": "n", "value": "${n + 1}"}}]},
                {"id": "stop_inner", "task": "condition",
                 "params": {"expr": "loop.inner.index >= 5"},
                 "body": [{"id": "b", "task": "break"}]}]},
            {"id": "every2", "task": "set_variable", "params": {"name": "m", "value": "${m + 1}"},
             "every": {"nth": 2, "of": "outer"}},
            {"id": "leave", "task": "condition", "params": {"expr": "loop.outer.index == 1"},
             "body": [{"id": "bb", "task": "break", "params": {"loop": "outer"}}]}]}],
        variables={"n": 0, "m": 0})
    run = _run(job)
    assert run.state == "finished"
    # odd inner indices continue before the break check, so 0, 2, 4, 6 count, then 6 breaks;
    # two outer rounds; "every 2nd" runs at outer index 0 only
    assert run.variables == {"n": 8, "m": 1}


def test_repeat_until_and_finish_node():
    job = _job([{"id": "r", "task": "repeat", "params": {"until": "n >= 4"}, "body": [
        {"id": "inc", "task": "set_variable", "params": {"name": "n", "value": "${n + 1}"}}]},
        {"id": "done", "task": "finish", "params": {"message": "early"}},
        {"id": "never", "task": "set_variable", "params": {"name": "n", "value": -1}}],
        variables={"n": 0})
    run = _run(job)
    assert (run.state, run.message, run.variables["n"]) == ("finished", "early", 4)


def test_error_policies():
    class Flaky(SimulatedMicroscope):
        fails = 2

        def set_valve(self, valve, position):
            if self.fails:
                self.fails -= 1
                raise IOError("valve did not answer")
            super().set_valve(valve, position)

    hw = Flaky()
    retry = _job([{"id": "v", "task": "set_valve", "params": {"valve": "inlet", "position": "a"},
                   "on_error": {"retry": 2}}])
    assert _run(retry, hw).state == "finished" and hw.valves["inlet"] == "a"

    skip = _job([{"id": "v", "task": "set_valve", "params": {"valve": "x", "position": "a"},
                  "on_error": "skip"}, {"id": "w", "task": "wait", "params": {"seconds": 0}}])
    Flaky.fails = 5
    run = _run(skip, Flaky())
    assert run.state == "finished" and run.results["v"] is None

    abort = _job([{"id": "v", "task": "set_valve", "params": {"valve": "x", "position": "a"}}])
    run = _run(abort, Flaky())
    assert run.state == "failed" and run.message.startswith("v: valve did not answer")


def test_exit_with_error_and_wait_until_timeout():
    job = _job([{"id": "w", "task": "wait_until",
                 "params": {"expr": "device.temperature_c > 100", "timeout_s": 0.3,
                            "poll_s": 0.05}}])
    run = _run(job)
    assert run.state == "failed" and "timed out" in run.message


# --------------------------------------------------------------------------- #
# Run control
# --------------------------------------------------------------------------- #
def _slow_job():
    return _job([{"id": "r", "task": "repeat", "params": {"count": 1000}, "body": [
        {"id": "inc", "task": "set_variable", "params": {"name": "n", "value": "${n + 1}"}},
        {"id": "w", "task": "wait", "params": {"seconds": 0.01}}]}], variables={"n": 0})


def test_pause_resume_and_abort():
    hw = SimulatedMicroscope()
    run = Run(_slow_job(), hw)
    run.start()
    assert _wait_for(lambda: run.variables["n"] > 3)
    run.pause()
    assert _wait_for(lambda: run.state == "paused")
    frozen = run.variables["n"]
    time.sleep(0.2)
    assert run.variables["n"] == frozen
    run.resume()
    assert _wait_for(lambda: run.variables["n"] > frozen + 3)
    run.abort(force=True)
    run.join(5)
    assert run.state == "aborted" and ("stop_stage",) in hw.calls


def test_finish_while_paused():
    run = Run(_slow_job(), SimulatedMicroscope())
    run.start()
    assert _wait_for(lambda: run.variables["n"] > 1)
    run.pause()
    assert _wait_for(lambda: run.state == "paused")
    run.finish()
    run.join(5)
    assert run.state == "finished"


def test_ask_first_steps_need_confirmation():
    job = _job([{"id": "h", "task": "home_axis", "params": {"axis": "X"}}])
    with pytest.raises(PermissionError):
        Run(job, SimulatedMicroscope()).run()
    assert _run(job).state == "finished"


def test_parameters_are_checked():
    job = _job([], parameters={"k": {"type": "integer", "default": 2, "minimum": 1}})
    with pytest.raises(ValueError, match="unknown"):
        Run(job, SimulatedMicroscope(), parameters={"x": 1})
    with pytest.raises(ValueError, match="out of range"):
        Run(job, SimulatedMicroscope(), parameters={"k": 0})
    assert Run(job, SimulatedMicroscope(), parameters={"k": "3"}).params == {"k": 3}


# --------------------------------------------------------------------------- #
# Timing and acquisition details
# --------------------------------------------------------------------------- #
def test_time_lapse_skips_missed_rounds_and_runs_idle():
    job = _job([{"id": "t", "task": "time_lapse",
                 "params": {"plan": {"interval": 0.25, "loops": 6}, "on_overrun": "skip"},
                 "body": [{"id": "inc", "task": "set_variable",
                           "params": {"name": "n", "value": "${n + 1}"}},
                          {"id": "slow", "task": "wait", "params": {"seconds": 0.6},
                           "enabled": "loop.t.index == 0"}],
                 "idle": [{"id": "idle", "task": "set_variable",
                           "params": {"name": "i", "value": "${i + 1}"}}]}],
        variables={"n": 0, "i": 0})
    t0 = time.monotonic()
    run = _run(job)
    # round 0 ends at 0.6 s: the 0.25 s start is missed and 0.5 s has also passed, so
    # round 1 is skipped and the idle body runs until round 2 at 0.75 s (150 ms of
    # slack for a slow runner; the old 50 ms failed on the macOS runner)
    assert run.variables["n"] == 5 and run.variables["i"] >= 2
    assert 1.2 < time.monotonic() - t0 < 2.5


def test_zstack_returns_and_channel_offsets(tmp_path):
    hw = SimulatedMicroscope()
    hw.position["Z"] = 100.0
    job = _job([{"id": "z", "task": "zstack", "params": {"plan": {"range": 4, "step": 2}},
                 "body": [{"id": "img", "task": "capture", "params": {"channels": [
                     {"config": "LED", "intensity": 10, "z_offset": 1.5},
                     {"config": "Laser488", "intensity": 20}]}}]}])
    run = _run(job, hw, sink=OmeStore(tmp_path))
    assert hw.position["Z"] == 100.0 and run.frames == 6
    snaps = [c for c in hw.calls if c[0] == "snap"]
    assert [s[2]["Z"] for s in snaps] == [99.5, 98.0, 101.5, 100.0, 103.5, 102.0]
    data = zarr.open_group(str(tmp_path / "img.ome.zarr"), mode="r")["0"]
    assert data.shape == (1, 2, 3, 32, 48) and int(data[0, 0, 0, 0, 0]) == 10 + 99


def test_capture_inside_a_channel_loop(tmp_path):
    job = _job([{"id": "c", "task": "loop_channels", "params": {"channels": "@cap"}, "body": [
        {"id": "z", "task": "zstack", "params": {"plan": {"range": 2, "step": 2}},
         "body": [{"id": "img", "task": "capture"}]}]}],
        resources={"cap": {"kind": "capture", "channels": [
            {"config": "LED", "intensity": 5}, {"config": "Laser488", "intensity": 7}]}})
    run = _run(job, sink=OmeStore(tmp_path))
    data = zarr.open_group(str(tmp_path / "img.ome.zarr"), mode="r")["0"]
    assert run.frames == 4 and data.shape == (1, 2, 2, 32, 48)
    assert [int(data[0, c, 0, 0, 0]) for c in range(2)] == [5 + 1, 7 + 1]


def test_mda_node_runs_a_useq_sequence(tmp_path):
    seq = useq.MDASequence(stage_positions=[(0, 0, 0), (100, 0, 0)],
                           channels=["LED", "Laser488"], z_plan=useq.ZRangeAround(range=2, step=1))
    job = _job([{"id": "m", "task": "mda",
                 "params": {"sequence": seq.model_dump(mode="json", exclude_unset=True)}}])
    run = _run(job, sink=OmeStore(tmp_path))
    assert run.frames == len(list(seq)) == 12
    group = zarr.open_group(str(tmp_path / "p001_m.ome.zarr"), mode="r")
    assert group["0"].shape == (1, 2, 3, 32, 48)
    open_ome_zarr(group)
