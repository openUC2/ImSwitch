"""The observation camera sees every axis move: the image changes more than its own noise.

MOVES REAL HARDWARE (to the transport position, then 3 to 5 mm per axis and back).
SWITCHES LIGHT.

It runs after stage_homing, so that its absolute move goes to a referenced position.
At the transport position ImSwitch switches the Z hard limits off, and the turret goes to slot 0.
The LED matrix lights the sample.
Then every axis moves out and back once.
The noise is the difference of two frames with nothing moving; the move must change the image more.

The position ImSwitch reports is only a step count.
This file is what shows that the stage really moved.
"""
import base64
import io
import time

import numpy as np
import pytest
from PIL import Image

import hil_client as client
from hil_report import attach

# Distances are in µm, times in seconds.
DISTANCE_UM = 5000             # X and Y
A_DISTANCE_UM = 3000
Z_DISTANCE_UM = 5000
Z_DIRECTION = -1               # away from Z home: at the transport position the limits are off
SPEED = 15000
SETTLE_S = 0.5                 # wait after a move, before the frame
TRANSPORT_SPEED = 20000        # above 30000 the stage loses steps
TRANSPORT_TIMEOUT = 120
TURRET_TIMEOUT = 120
MATRIX_INTENSITY = 255         # per colour, of setAllLED (0..255)
MATRIX_SETTLE_S = 3.5          # wait after switching the matrix on, before the first frame

MIN_RATIO = 1.8                # image change over noise that counts as movement, X and Y
MIN_RATIO_AZ = 1.25            # the same for A and Z, which change the image less

SNAP = {"slot_id": "1", "camera_name": "hil_motion"}  # camera_name: folder ImSwitch saves to


# --- stage ------------------------------------------------------------------------------------

def axes():
    """Every (positioner, axis) pair of the setup, e.g. ("ESP32Stage", "X")."""
    return [(name, axis) for name, positioner in client.setup()["positioners"].items()
            if positioner["forPositioning"] for axis in positioner["axes"]]


def move(positioner, axis, distance):
    """Move one axis by ``distance`` µm and wait until it has arrived and settled."""
    client.get_json("/PositionerController/movePositioner", timeout=60, params={
        "positionerName": positioner, "axis": axis, "dist": distance,
        "isBlocking": True, "speed": SPEED})
    time.sleep(SETTLE_S)


def wait_until_stopped():
    """The positions once two reads agree; some endpoints answer before the stage has arrived."""
    previous = client.get_json("/PositionerController/getPositionerPositions")
    deadline = time.monotonic() + TRANSPORT_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(0.5)
        current = client.get_json("/PositionerController/getPositionerPositions")
        if current == previous:
            return current
        previous = current
    pytest.fail(f"the stage still moves after {TRANSPORT_TIMEOUT}s: {previous}")


def park_at_transport():
    """Move to the transport position and turn the turret to slot 0; returns the positions."""
    client.get_json("/PositionerController/moveToTransportPosition", timeout=TRANSPORT_TIMEOUT,
                    params={"speed": TRANSPORT_SPEED, "isBlocking": True})
    wait_until_stopped()
    if client.setup()["objective"]:
        status = client.get_json("/ObjectiveController/getstatus")
        if status.get("isActive") and status.get("hasMotor") and status["slotConfigured"][0]:
            # skipZ: the turret must not apply the slot's Z offset to the parked stage.
            started = client.get_json("/ObjectiveController/moveToObjective",
                                      {"slot": 0, "skipZ": True})
            assert started.get("accepted"), f"the turret refused slot 0: {started}"

            def turned():
                # moveToObjective only starts the move; the status tells when it is done.
                now = client.get_json("/ObjectiveController/getstatus")
                return (not now.get("isMovingObjective") and
                        now.get("lastCompletedObjectiveRequestId") == started["requestId"])

            assert client.wait_until(turned, TURRET_TIMEOUT), (
                f"the turret did not reach slot 0 in {TURRET_TIMEOUT}s")
    return wait_until_stopped()


# --- observation camera -----------------------------------------------------------------------

def frame():
    """One full-size greyscale frame of the observation camera, as a float array."""
    body = client.post_json("/ExperimentController/snapOverviewImage", params=SNAP, timeout=60)
    image = Image.open(io.BytesIO(base64.b64decode(body["imageBase64"]))).convert("L")
    return np.asarray(image, dtype=np.float32)


def difference(before, after):
    """Mean absolute grey-value difference of two frames."""
    assert before.shape == after.shape
    return float(np.mean(np.abs(after - before)))


# --- the test ---------------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def parked_and_lit():
    """Check the camera before anything moves, then park the stage and switch the matrix on."""
    setup = client.setup()
    cameras = [name for name, detector in setup["detectors"].items()
               if not detector["forAcquisition"]]
    if not cameras:
        pytest.skip("the setup has no observation camera")
    status = client.get_json("/SettingsController/getCameraStatus", {"detectorName": cameras[0]})
    if "error" in status or status.get("status") == "error":
        pytest.skip(f"{cameras[0]}: getCameraStatus failed: {status.get('error')}")
    if status.get("isMock") or status.get("isConnected") is False:
        pytest.skip(f"{cameras[0]} is a mock camera or not connected: {status}")
    response = client.post("/ExperimentController/snapOverviewImage", params=SNAP, timeout=60)
    if response.status_code != 200:
        pytest.skip(f"{cameras[0]} cannot deliver an image: {response.text}")

    print(f"\ntransport position reached: {park_at_transport()}")
    has_matrix = bool(setup["LEDMatrixs"])
    if has_matrix:
        client.get_json("/LEDMatrixController/setAllLED", {
            "intensity_r": MATRIX_INTENSITY, "intensity_g": MATRIX_INTENSITY,
            "intensity_b": MATRIX_INTENSITY})
        time.sleep(MATRIX_SETTLE_S)
    yield
    if has_matrix:
        client.get_json("/LEDMatrixController/setAllLEDOff")


@client.parametrize_from_rig("positioner, axis", axes, "the setup has no positioner axes")
def test_axis_motion_is_visible(positioner, axis):
    if axis.upper() == "Z":
        distance, required = Z_DISTANCE_UM * Z_DIRECTION, MIN_RATIO_AZ
    elif axis.upper() in ("X", "Y"):
        distance, required = DISTANCE_UM, MIN_RATIO
    else:
        distance, required = A_DISTANCE_UM, MIN_RATIO_AZ

    still = frame()
    time.sleep(SETTLE_S)
    before = frame()
    move(positioner, axis, distance)
    try:
        after = frame()
    finally:
        move(positioner, axis, -distance)
    attach(f"{axis} before", before)
    attach(f"{axis} after {distance:+d} um", after)

    noise, change = difference(still, before), difference(before, after)
    ratio = change / max(noise, 1e-6)
    print(f"\n{positioner} {axis} {distance:+d} um: change {change:.3f}, noise {noise:.3f}, "
          f"ratio {ratio:.2f}x")
    assert ratio >= required, (
        f"{positioner} {axis}: movement was not clearly visible. change={change:.3f}, "
        f"noise={noise:.3f}, ratio={ratio:.2f}x, required={required:.2f}x")
