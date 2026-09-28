"""Drive the stage to its stored transport position. Not a test.

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


def api(method, **params):
    """Call one PositionerController endpoint and return its JSON."""
    response = requests.get(
        f"{BASE_URL}/api/PositionerController/{method}",
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


def move_to_transport():
    """Drive to the transport position and return where the stage stopped."""
    api("moveToTransportPosition", speed=SPEED, isBlocking=True)

    return wait_until_stopped()


if __name__ == "__main__":
    if not api("getPositionerNames"):
        raise SystemExit("no positioner loaded in the current setup - nothing to move")

    print(f"transport position: {api('getTransportPosition')}")
    print(f"before:             {api('getPositionerPositions')}")
    print(f"after:              {move_to_transport()}")
