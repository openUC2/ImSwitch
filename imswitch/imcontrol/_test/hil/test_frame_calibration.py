"""The one-button FRAME calibration runs through on the calibration target in the holder.

MOVES REAL HARDWARE: homes X, Y, Z and the turret, moves the stage across the target, focuses,
switches the objective. SWITCHES LIGHT. WRITES THE SETUP: the calibration stores its result
(stage offset, objective offsets, pixel calibration) in the setup file the rig runs with. That
result stays, it is what the calibration is for; the setup as it was before is saved as
setup_before_frame_calibration.json in the report folder, to go back to by hand.

The calibration target (plate or pinhole slides, setting "target") must be in the holder.
Afterwards the settings this test changed are back and every light is off.
"""
import io
import json
import os
import time
from pathlib import Path

import pytest
from PIL import Image

import hil_client as client
import hil_report
from hil_report import attach

RUN_TIMEOUT = 1800  # seconds the whole calibration may take
POLL_S = 2          # seconds between two status requests
STOP_TIMEOUT = 60   # seconds a stopped calibration may take to end
# Without a focus the calibration would wait for someone to focus by hand and press Continue.
# Nobody is there in a test, so the run fails instead.
# The light value is set, not kept ("illuminationValue" 0 = keep its current value): the
# teardown turns every light to 0, so a kept value would be 0 in the next run - a black camera.
TEST_SETTINGS = {"manualFocusFallback": False, "illuminationValue": 500}
BACKUP_NAME = "setup_before_frame_calibration.json"
ENDED = ("finished", "failed", "aborted")


# --- the calibration --------------------------------------------------------------------------

def status():
    return client.get_json("/FrameCalibrationController/getFrameCalibrationStatus")


def settings():
    return client.get_json("/FrameCalibrationController/getFrameCalibrationSettings")


def set_settings(values):
    return client.post_json("/FrameCalibrationController/setFrameCalibrationSettings",
                            json=values)


def running():
    return client.get_json("/FrameCalibrationController/isFrameCalibrationRunning") is True


def stop_and_wait():
    """The terminal status alone does not mean cleanup/report writing has finished."""
    if not running():
        return
    if status()["state"] not in ENDED:
        client.post_json("/FrameCalibrationController/stopFrameCalibration")
    assert client.wait_until(lambda: not running(), STOP_TIMEOUT, interval=POLL_S), (
        f"calibration thread did not stop within {STOP_TIMEOUT}s")


def collect_report(run):
    """Export only the completed calibration started by this test, never an older one."""
    assert not running(), "cannot export while calibration still runs"
    current = status()
    assert run.get("startedAt") and current["startedAt"] == run["startedAt"], (
        "calibration identity changed; refusing to attach another run's report")
    hil_report.calibration.update(state=current["state"], message=current["message"],
                                  startedAt=current["startedAt"])
    folder = Path(os.environ.get("HIL_REPORT_DIR", "."))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "calibration-status.json").write_text(json.dumps(current, indent=2), encoding="utf-8")
    response = client.get("/FrameCalibrationController/getFrameCalibrationReportPdf", timeout=120)
    assert response.status_code == 200 and response.content.startswith(b"%PDF-"), (
        f"calibration PDF export failed: HTTP {response.status_code}")
    # Recheck after export because the endpoint returns the running/last report.
    assert status()["startedAt"] == run["startedAt"], "calibration changed during PDF export"
    from pypdf import PdfReader
    assert len(PdfReader(io.BytesIO(response.content)).pages), "calibration PDF has no pages"
    path = folder / "calibration-report.pdf"
    path.write_bytes(response.content)
    hil_report.calibration["pdf"] = str(path)


def run_calibration(run):
    """Start the calibration and follow it to its end; returns the last status.

    Every step's change is printed once, so the output reads as the calibration's log.
    """
    started = client.post_json("/FrameCalibrationController/startFrameCalibration")
    assert started.get("success"), f"startFrameCalibration refused: {started}"
    run["accepted"] = True
    hil_report.calibration.update(state="running", message="Calibration accepted")
    run["startedAt"] = status()["startedAt"]
    printed = {}
    deadline = time.monotonic() + RUN_TIMEOUT
    while True:
        current = status()
        for step in current["steps"]:
            line = (step["status"], step["detail"], len(step["warnings"]))
            if printed.get(step["name"]) != line and step["status"] != "pending":
                printed[step["name"]] = line
                print(f"[{step['status']}] {step['title']}: {step['detail']}", flush=True)
        assert current["startedAt"] == run["startedAt"], "calibration replaced during polling"
        if current["state"] in ENDED and not running():
            return current
        # The test turned the manual focus off; a wait for the user would never end.
        assert current["state"] != "waiting", (
            f"the calibration waits for the user: {current['message']}")
        assert time.monotonic() < deadline, (
            f"the calibration did not end within {RUN_TIMEOUT}s (step {current['step']!r})")
        time.sleep(POLL_S)


def attach_figures(run):
    """Every report figure of the run (yellow = expected, green = measured) into the report."""
    assert not running() and status()["startedAt"] == run.get("startedAt"), "not this run's completed figures"
    for figure in client.get_json("/FrameCalibrationController/getFrameCalibrationFigures"):
        response = client.get("/FrameCalibrationController/getFrameCalibrationFigure",
                              params={"figureId": figure["id"]})
        if response.headers.get("content-type", "").startswith("image/"):
            mark = "" if figure.get("ok") in (True, None) else " (NOT OK)"
            attach(f"{figure['title']}{mark}", Image.open(io.BytesIO(response.content)))


# --- the rig before and after -----------------------------------------------------------------

def all_lights_off():
    actions = []
    for name in client.setup()["lasers"]:
        actions.extend([
            ("/LaserController/setLaserValue", {"laserName": name, "value": 0}),
            ("/LaserController/setLaserActive", {"laserName": name, "active": False}),
        ])
    if client.setup()["LEDMatrixs"]:
        actions.append(("/LEDMatrixController/setAllLEDOff", None))
    errors = []
    for endpoint, params in actions:
        try:
            client.get_json(endpoint, params)
        except Exception as error:
            errors.append(f"{endpoint} {params}: {error}")
    if errors:
        raise RuntimeError("; ".join(errors))


def save_setup_backup():
    """Write the setup as it is now into the report folder; returns the file."""
    folder = Path(os.environ.get("HIL_REPORT_DIR", "."))
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / BACKUP_NAME
    backup = {"setupFile": client.setup_file(), "setup": client.setup()}
    file.write_text(json.dumps(backup, indent=2), encoding="utf-8")
    print(f"\nsetup before the calibration saved to {file.resolve()}")
    return file


@pytest.fixture(scope="module")
def calibration():
    """Run the calibration once; returns its last status. Puts the settings and light back."""
    setup = client.setup()
    if not setup["positioners"]:
        pytest.skip("the setup has no stage")
    response = client.get("/FrameCalibrationController/getFrameCalibrationSettings")
    if response.status_code == 404:
        pytest.skip("this ImSwitch has no FrameCalibrationController")
    original = client.json_of(response)
    if running():
        pytest.skip(f"a FRAME calibration is already running: {status()['state']}")

    save_setup_backup()
    print(f"target {original['target']!r}, settings changed for the test: {TEST_SETTINGS}")
    run, errors = {}, []
    result = None
    hil_report.calibration.update(state="not started", message="Calibration start not yet accepted")
    try:
        set_settings(TEST_SETTINGS)
        result = run_calibration(run)
    except Exception as error:
        errors.append(("calibration", error))
    finally:
        actions = []
        if run.get("accepted"):
            actions.append(("stop/wait", stop_and_wait))
        actions.extend([
            ("restore settings", lambda: set_settings({key: original[key] for key in TEST_SETTINGS})),
            ("lights off", all_lights_off),
        ])
        if run.get("accepted"):
            actions.extend([("export PDF", lambda: collect_report(run)),
                            ("attach figures", lambda: attach_figures(run))])
        for label, action in actions:
            try:
                action()
            except Exception as error:
                errors.append((label, error))
        if errors:
            detail = "; ".join(f"{label}: {error}" for label, error in errors)
            hil_report.calibration["error"] = detail
            raise RuntimeError(detail) from errors[0][1]
    print(f"\n{result['state']}: {result['message']}")
    return result


# --- tests ------------------------------------------------------------------------------------

def test_calibration_finishes(calibration):
    assert calibration["state"] == "finished", (
        f"the calibration ended as {calibration['state']!r} in step {calibration['step']!r}: "
        f"{calibration['message']} - is the calibration target in the holder?")


def test_every_step_succeeds(calibration):
    """Optional steps may fail without stopping the calibration; this test lists them."""
    failed = [f"{step['title']}: {step['detail']}" for step in calibration["steps"]
              if step["status"] in ("failed", "aborted")]
    assert not failed, "steps that failed:\n" + "\n".join(failed)
    warned = [f"{step['title']}: {'; '.join(step['warnings'])}" for step in calibration["steps"]
              if step["warnings"]]
    if warned:
        print("\nwarnings:\n" + "\n".join(warned))
    # A skipped step did not test its part of the rig, e.g. the turret without a second objective.
    skipped = [step["title"] for step in calibration["steps"] if step["status"] == "skipped"]
    if skipped:
        print("\nskipped, so not tested:\n" + "\n".join(skipped))
