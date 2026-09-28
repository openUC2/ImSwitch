"""Drive the stage to its stored transport position, turret on slot 0. Not a test.

MOVES REAL HARDWARE. Named without the test_ prefix so pytest never collects
it; test_motor_motion_camera.py imports move_to_transport() to park the stage
before it starts. Run it through run_move_to_transport.sh, or directly:

    IMSWITCH_URL=http://<pi>:8000/imswitch python3 move_to_transport.py

The target is whatever the setup stores as transportPositionA/X/Y/Z (see
ESP32StageManager.transportPositions). Setups without those keys fall back to
X=65000, Y=40000; FRAME.json sets all of them to 0, which is behind the
endstops rather than the normal position.

moveToTransportPosition switches the Z hard limits off before moving
(override_endstop_z defaults to True and the API does not expose it).

Afterwards the turret goes to the first objective (slot 0), turret only
(skipZ), so the parked Z stays. Skipped on setups without an objective motor.
"""

import os
import time

import requests


BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")

# How long to wait for the stage to stop, in seconds.
TIMEOUT = float(os.environ.get("TRANSPORT_TIMEOUT", "120"))

# Motor speed for the transport move. The endpoint default is 10000.
# Don't go higher than 30000.
SPEED = float(os.environ.get("TRANSPORT_SPEED", "20000"))

POLL = 0.5


def api(method, controller="PositionerController", **params):
    """Call one ImSwitch endpoint and return its JSON."""
    response = requests.get(
        f"{BASE_URL}/api/{controller}/{method}",
        params=params,
        timeout=TIMEOUT,
    )
    assert response.status_code == 200, response.text
    return response.json()


def wait_until_stopped():
    """Positions once two reads in a row agree, or the last read on timeout.

    The endpoint runs on ImSwitch's UI thread, so the HTTP answer is not proof
    the stage has arrived; the position readback is.
    """
    deadline = time.time() + TIMEOUT
    previous = api("getPositionerPositions")

    while time.time() < deadline:
        time.sleep(POLL)

        current = api("getPositionerPositions")

        if current == previous:
            return current

        previous = current

    return previous


def move_to_first_objective():
    """Turn the turret to slot 0 and wait; no-op without an objective motor."""
    try:
        status = api("getstatus", "ObjectiveController")
    except AssertionError:
        return  # no ObjectiveController in this setup

    if not (status.get("isActive") and status.get("hasMotor")
            and (status.get("slotConfigured") or [False])[0]):
        return

    started = api("moveToObjective", "ObjectiveController", slot=0, skipZ=True)
    assert started.get("accepted"), f"objective slot 0 refused: {started}"

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        status = api("getstatus", "ObjectiveController")
        if status.get("lastObjectiveMoveError"):
            raise AssertionError(f"objective move failed: {status['lastObjectiveMoveError']}")
        if (not status.get("isMovingObjective")
                and status.get("lastCompletedObjectiveRequestId") == started["requestId"]):
            return
        time.sleep(POLL)

    raise AssertionError(f"objective move did not finish in {TIMEOUT}s")


def move_to_transport():
    """Drive to the transport position, turret on slot 0; return where it stopped."""
    api("moveToTransportPosition", speed=SPEED, isBlocking=True)
    wait_until_stopped()
    move_to_first_objective()

    return wait_until_stopped()


if __name__ == "__main__":
    if not api("getPositionerNames"):
        raise SystemExit("no positioner loaded in the current setup - nothing to move")

    print(f"transport position: {api('getTransportPosition')}")
    print(f"before:             {api('getPositionerPositions')}")
    print(f"after:              {move_to_transport()}")
