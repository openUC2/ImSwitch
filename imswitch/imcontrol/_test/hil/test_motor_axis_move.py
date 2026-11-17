"""Each motor axis moves 10 µm and back, and ImSwitch reports the position it moved to.

MOVES REAL HARDWARE (10 µm).

The position is the step count of the firmware, not a measurement.
So this only shows that ImSwitch, the board and its bookkeeping follow the command.
motor_motion_camera shows in the camera that the stage really moved.
"""
import pytest

import hil_client as client

STEP_UM = 10
MOVE_TIMEOUT = 60  # seconds a blocking 10 µm move may take


def axes():
    """Every (positioner, axis) pair of the setup, e.g. ("ESP32Stage", "X")."""
    return [(name, axis) for name, positioner in client.setup()["positioners"].items()
            if positioner["forPositioning"] for axis in positioner["axes"]]


def position(positioner, axis):
    return client.get_json("/PositionerController/getPositionerPositions")[positioner][axis]


def move(positioner, axis, distance):
    """Move one axis by ``distance`` µm; returns once it has arrived."""
    client.get_json("/PositionerController/movePositioner", timeout=MOVE_TIMEOUT, params={
        "positionerName": positioner, "axis": axis, "dist": distance, "isBlocking": True})


@client.parametrize_from_rig("positioner, axis", axes, "the setup has no positioner axes")
def test_axis_moves_by_step(positioner, axis):
    before = position(positioner, axis)
    try:
        move(positioner, axis, STEP_UM)
        after = position(positioner, axis)
        assert after == pytest.approx(before + STEP_UM, abs=0.1), (
            f"{positioner} {axis}: {before} -> {after}, expected {before + STEP_UM}")
    finally:
        move(positioner, axis, -STEP_UM)
    back = position(positioner, axis)
    assert back == pytest.approx(before, abs=0.1), (
        f"{positioner} {axis}: back at {back}, expected {before}")
