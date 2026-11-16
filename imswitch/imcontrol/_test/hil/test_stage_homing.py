"""The stage homes Z, X and Y as "Safe Global Homing" does, and every axis reports done.

MOVES REAL HARDWARE: Z, X and Y drive to their end stops.

This is the button "Start homing" under FRAME settings: the endpoint startFrameHoming.
It homes Z and lifts it clear of the bottom, then homes X and Y.
At the end X and Y go to the transport position, and Z goes back to its height from before.
That is where motor_motion_camera and the light tests find the stage.
The turret (A) is not homed here; the FRAME calibration homes it.

The test passes when the homing ends as "done" with Z, X and Y "done", as the button shows it.
The positions at the start and at the end are in the report.
"""
import pytest

import hil_client as client
from hil_report import note

HOME_ORDER = "ZXY"         # the axes startFrameHoming homes, in its order
POLL_S = 0.5               # seconds between two status reads while the stage homes
START_TIMEOUT = 10         # seconds the homing thread may take to start
HOME_EXTRA_S = 120         # seconds the whole homing may take beyond the homeTimeouts of the setup
DEFAULT_HOME_TIMEOUT_MS = 20000  # as ESP32StageManager, when the setup sets no homeTimeout


def positioner_to_home():
    """(name, managerProperties) of the first positioner with homing on for Z, X or Y."""
    for name, positioner in client.setup()["positioners"].items():
        properties = positioner["managerProperties"]
        if any(properties.get(f"home{axis}enabled") for axis in HOME_ORDER):
            return name, properties
    return None, None


def homing_state(positioner):
    return client.get_json("/PositionerController/getFrameHomingState",
                           {"positionerName": positioner})


def positions(positioner):
    return client.get_json("/PositionerController/getPositionerPositions")[positioner]


def test_stage_homes_with_safe_global_homing():
    positioner, properties = positioner_to_home()
    if positioner is None:
        pytest.skip("no positioner has homing switched on for Z, X or Y")
    if homing_state(positioner)["active"]:
        pytest.skip("a homing is already running")
    timeout = HOME_EXTRA_S + sum(
        properties.get(f"homeTimeout{axis}", DEFAULT_HOME_TIMEOUT_MS) / 1000 for axis in HOME_ORDER)

    note(f"positions at the start: {positions(positioner)}")
    started = client.get_json("/PositionerController/startFrameHoming",
                              {"positionerName": positioner, "isBlocking": False})
    assert started.get("success"), f"startFrameHoming refused: {started}"
    # The homing starts in a thread; until then the state is still the one of the last homing.
    assert client.wait_until(lambda: homing_state(positioner)["active"], START_TIMEOUT), (
        f"the homing did not start within {START_TIMEOUT}s")
    ended = client.wait_until(lambda: not homing_state(positioner)["active"], timeout,
                              interval=POLL_S)
    state = homing_state(positioner)
    if not ended:
        client.get_json("/PositionerController/cancelFrameHoming", {"positionerName": positioner})
        pytest.fail(f"the homing did not end within {timeout:.0f}s, in phase "
                    f"{state['phase']!r}; it was cancelled")
    note(f"positions at the end: {positions(positioner)}")

    assert state["phase"] == "done", f"the homing ended as {state['phase']!r}: {state['message']}"
    not_done = [axis for axis in HOME_ORDER if state["axes"].get(axis) != "done"]
    assert not not_done, f"not homed: {', '.join(not_done)} ({state['axes']})"
