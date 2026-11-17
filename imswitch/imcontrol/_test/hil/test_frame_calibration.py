"""The FRAME calibration runs from start to end with the calibration target in the holder,
and none of its steps fails.

MOVES REAL HARDWARE: homes X, Y, Z and the turret, scans the target, focuses, changes objective.
SWITCHES LIGHT.
WRITES THE SETUP: the calibration stores its result in the setup file the rig runs with.
The result is the stage offset, the objective offsets and the pixel calibration.
It stays, it is what the calibration is for.
To go back by hand, the old setup is saved in the report folder
as setup_before_frame_calibration.json.

The calibration target (plate or pinhole slides, setting "target") must be in the holder.
Afterwards the settings this test changed are back and every light is off.
"""
import io
import json
import os
import time
from pathlib import Path

import pytest
from pypdf import PdfReader

import hil_client as client
import hil_report

RUN_TIMEOUT = 1800  # seconds the whole calibration may take
POLL_S = 2          # seconds between two status requests
STOP_TIMEOUT = 60   # seconds a stopped calibration may take to end
# Without a focus the calibration would wait for someone to focus by hand and press Continue.
# Nobody is there in a test, so the run fails instead.
# The light value is set, not kept ("illuminationValue" 0 would keep its current value).
# The teardown turns every light to 0, so a kept value would be 0 in the next run: a black camera.
TEST_SETTINGS = {"manualFocusFallback": False, "illuminationValue": 500}
BACKUP_NAME = "setup_before_frame_calibration.json"
CALIBRATION_PDF = "frame_calibration.pdf"  # the calibration's own report, in the report folder
ENDED = ("finished", "failed", "aborted")
ACTIVE = ("running", "waiting")  # the states the FRAME calibration app shows as running


# --- the calibration --------------------------------------------------------------------------

def status():
    return client.get_json("/FrameCalibrationController/getFrameCalibrationStatus")


def set_settings(values):
    return client.post_json("/FrameCalibrationController/setFrameCalibrationSettings",
                            json=values)


def running():
    """Running as the app shows it, from the state of the status.

    The state ends before the calibration thread does.
    The thread then only resets the camera and writes its report folder.
    It switches no light, moves no motor and leaves the setup file alone.
    So nothing this test does afterwards collides with it.
    """
    return status()["state"] in ACTIVE


def stop_and_wait():
    if not running():
        return
    client.post_json("/FrameCalibrationController/stopFrameCalibration")
    assert client.wait_until(lambda: not running(), STOP_TIMEOUT, interval=POLL_S), (
        f"calibration thread did not stop within {STOP_TIMEOUT}s")


def collect_report(run):
    """Put the calibration's own PDF into the report; never the one of an older run."""
    assert not running(), "cannot export while calibration still runs"
    current = status()
    assert run.get("startedAt") and current["startedAt"] == run["startedAt"], (
        "calibration identity changed; refusing to attach another run's report")
    hil_report.note(f"The calibration ended as {current['state']}: {current['message']}")
    response = client.get("/FrameCalibrationController/getFrameCalibrationReportPdf", timeout=120)
    assert response.status_code == 200 and response.content.startswith(b"%PDF-"), (
        f"calibration PDF export failed: HTTP {response.status_code}")
    # Checked again, because the endpoint gives the report of whatever run is the last one now.
    assert status()["startedAt"] == run["startedAt"], "calibration changed during PDF export"
    assert len(PdfReader(io.BytesIO(response.content)).pages), "calibration PDF has no pages"
    hil_report.append_pdf("FRAME calibration (original report)", response.content)
    # Also on its own, so it is in the report folder even when hil-report.pdf cannot be written.
    folder = Path(os.environ.get("HIL_REPORT_DIR", "."))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / CALIBRATION_PDF).write_bytes(response.content)
    hil_report.note(f"Its own report follows on the last pages, and is saved as {CALIBRATION_PDF}.")


def show_progress(steps, shown, live):
    """Print what changed since the last call: a line when a step starts, one when it ends.

        calibration  4/10  First objective: find via, focus, affine ...
        calibration  4/10  ok (212 s) objective 0: 0.598 x 0.598 um/px

    ``shown`` remembers the lines already printed, as (step name, "start" or "end").
    """
    for number, step in enumerate(steps, 1):
        head = f"calibration {number:>2}/{len(steps)}  "
        if step["status"] != "pending" and (step["name"], "start") not in shown:
            shown.add((step["name"], "start"))
            live(f"{head}{step['title']} ...")
        if step["status"] not in ("pending", "running") and (step["name"], "end") not in shown:
            shown.add((step["name"], "end"))
            took = f" ({step['seconds']:.0f} s)" if step["seconds"] is not None else ""
            live(f"{head}{step['status']}{took} {step['detail']}")
            for warning in step["warnings"]:
                live(f"{' ' * len(head)}warning: {warning}")


def run_calibration(run, live):
    """Start the calibration and follow it to its end; returns the last status.

    ``live`` prints the progress at once (conftest.py), so the log shows which step runs.
    """
    started = client.post_json("/FrameCalibrationController/startFrameCalibration")
    assert started.get("success"), f"startFrameCalibration refused: {started}"
    run["accepted"] = True
    run["startedAt"] = status()["startedAt"]
    shown = set()
    deadline = time.monotonic() + RUN_TIMEOUT
    while True:
        current = status()
        show_progress(current["steps"], shown, live)
        assert current["startedAt"] == run["startedAt"], "calibration replaced during polling"
        if current["state"] in ENDED:
            return current
        # The test turned the manual focus off; a wait for the user would never end.
        assert current["state"] != "waiting", (
            f"the calibration waits for the user: {current['message']}")
        assert time.monotonic() < deadline, (
            f"the calibration did not end within {RUN_TIMEOUT}s (step {current['step']!r})")
        time.sleep(POLL_S)


# --- the rig before and after -----------------------------------------------------------------

def all_lights_off():
    """Every laser, LED and the LED matrix off. One that fails does not stop the others."""
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
    """Write the setup as it is now into the report folder."""
    folder = Path(os.environ.get("HIL_REPORT_DIR", "."))
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / BACKUP_NAME
    backup = {"setupFile": client.setup_file(), "setup": client.setup()}
    file.write_text(json.dumps(backup, indent=2), encoding="utf-8")
    print(f"\nsetup before the calibration saved to {file.resolve()}")
    hil_report.note("The calibration writes its result into the setup. The setup as it was "
                    f"before is saved as {BACKUP_NAME}.")


# pytest ends these fixtures in the reverse order of the arguments of ``calibration``:
# first the light goes off, then the settings go back, then the report is collected.
# Each of them runs also when the one before it failed.

@pytest.fixture(scope="module")
def run():
    """What is known of the calibration this test starts: "accepted" and "startedAt".

    Afterwards its PDF goes into the report, also when the calibration failed.
    """
    run = {}
    yield run
    if run.get("accepted"):
        collect_report(run)


@pytest.fixture(scope="module")
def settings_for_the_test():
    """Skip when there is nothing to calibrate; else set TEST_SETTINGS, and put them back."""
    if not client.setup()["positioners"]:
        pytest.skip("the setup has no stage")
    response = client.get("/FrameCalibrationController/getFrameCalibrationSettings")
    if response.status_code == 404:
        pytest.skip("this ImSwitch has no FrameCalibrationController")
    original = client.json_of(response)
    if running():
        pytest.skip(f"a FRAME calibration is already running: {status()['state']}")

    save_setup_backup()
    print(f"target {original['target']!r}, settings changed for the test: {TEST_SETTINGS}")
    set_settings(TEST_SETTINGS)
    yield
    set_settings({key: original[key] for key in TEST_SETTINGS})


@pytest.fixture(scope="module")
def lights_off_afterwards():
    """The calibration leaves its light on when it ends."""
    yield
    all_lights_off()


@pytest.fixture(scope="module")
def calibration(live, run, settings_for_the_test, lights_off_afterwards):
    """Run the calibration once; returns its last status."""
    try:
        result = run_calibration(run, live)
    finally:
        if run.get("accepted"):
            stop_and_wait()
    live(f"calibration {result['state']}: {result['message']}")
    return result


# --- tests ------------------------------------------------------------------------------------

def test_calibration_finishes(calibration):
    assert calibration["state"] == "finished", (
        f"the calibration ended as {calibration['state']!r} in step {calibration['step']!r}: "
        f"{calibration['message']} - is the calibration target in the holder?")


def test_every_step_succeeds(calibration):
    """Optional steps may fail without stopping the calibration; this test lists them."""
    # Notes, not prints: a print only shows when the test fails.
    for step in calibration["steps"]:
        if step["warnings"]:
            hil_report.note(f"Warning in {step['title']}: {'; '.join(step['warnings'])}")
    # A skipped step did not test its part of the rig, e.g. the turret without a second objective.
    skipped = [step["title"] for step in calibration["steps"] if step["status"] == "skipped"]
    if skipped:
        hil_report.note(f"Skipped by the calibration, so not tested: {', '.join(skipped)}")
    failed = [f"{step['title']}: {step['detail']}" for step in calibration["steps"]
              if step["status"] in ("failed", "aborted")]
    assert not failed, "steps that failed:\n" + "\n".join(failed)
