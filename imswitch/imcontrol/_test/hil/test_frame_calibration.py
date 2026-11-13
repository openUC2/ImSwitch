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
from hil_report import attach

RUN_TIMEOUT = 1800  # seconds the whole calibration may take
POLL_S = 2          # seconds between two status requests
STOP_TIMEOUT = 60   # seconds a stopped calibration may take to end
# Without a focus the calibration would wait for someone to focus by hand and press Continue.
# Nobody is there in a test, so the run fails instead.
TEST_SETTINGS = {"manualFocusFallback": False}
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


def stop_and_wait():
    """Stop a calibration that still runs, and wait until it has ended."""
    if status()["state"] in ENDED + ("idle",):
        return
    client.post_json("/FrameCalibrationController/stopFrameCalibration")
    client.wait_until(lambda: status()["state"] in ENDED + ("idle",), STOP_TIMEOUT,
                      interval=POLL_S)


def run_calibration():
    """Start the calibration and follow it to its end; returns the last status.

    Every step's change is printed once, so the output reads as the calibration's log.
    """
    started = client.post_json("/FrameCalibrationController/startFrameCalibration")
    assert started.get("success"), f"startFrameCalibration refused: {started}"
    printed = {}
    deadline = time.monotonic() + RUN_TIMEOUT
    while True:
        current = status()
        for step in current["steps"]:
            line = (step["status"], step["detail"], len(step["warnings"]))
            if printed.get(step["name"]) != line and step["status"] != "pending":
                printed[step["name"]] = line
                print(f"[{step['status']}] {step['title']}: {step['detail']}", flush=True)
        if current["state"] in ENDED:
            return current
        # The test turned the manual focus off; a wait for the user would never end.
        assert current["state"] != "waiting", (
            f"the calibration waits for the user: {current['message']}")
        assert time.monotonic() < deadline, (
            f"the calibration did not end within {RUN_TIMEOUT}s (step {current['step']!r})")
        time.sleep(POLL_S)


def attach_figures():
    """Every report figure of the run (yellow = expected, green = measured) into the report."""
    for figure in client.get_json("/FrameCalibrationController/getFrameCalibrationFigures"):
        response = client.get("/FrameCalibrationController/getFrameCalibrationFigure",
                              params={"figureId": figure["id"]})
        if response.headers.get("content-type", "").startswith("image/"):
            mark = "" if figure.get("ok") in (True, None) else " (NOT OK)"
            attach(f"{figure['title']}{mark}", Image.open(io.BytesIO(response.content)))


# --- the rig before and after -----------------------------------------------------------------

def all_lights_off():
    for name in client.setup()["lasers"]:
        client.get_json("/LaserController/setLaserValue", {"laserName": name, "value": 0})
        client.get_json("/LaserController/setLaserActive", {"laserName": name, "active": False})
    if client.setup()["LEDMatrixs"]:
        client.get_json("/LEDMatrixController/setAllLEDOff")


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
    if status()["state"] not in ENDED + ("idle",):
        pytest.skip(f"a FRAME calibration is already running: {status()['state']}")

    save_setup_backup()
    print(f"target {original['target']!r}, settings changed for the test: {TEST_SETTINGS}")
    try:
        set_settings(TEST_SETTINGS)
        result = run_calibration()
    finally:
        stop_and_wait()
        set_settings({key: original[key] for key in TEST_SETTINGS})
        all_lights_off()
        attach_figures()
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
