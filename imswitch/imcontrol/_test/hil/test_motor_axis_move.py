"""Every axis reports a small relative step, then steps back. MOVES REAL HARDWARE (10 µm)."""
import time

import pytest

import hil_client as client

STEP_UM = 10
SETTLE_S = 0.5  # wait before reading the position back


def axes():
    """Every (positioner, axis) pair of the setup, e.g. ("ESP32Stage", "X")."""
    return [(name, axis) for name, positioner in client.setup()["positioners"].items()
            if positioner["forPositioning"] for axis in positioner["axes"]]


def position(positioner, axis):
    return client.get_json("/PositionerController/getPositionerPositions")[positioner][axis]


@client.parametrize_from_rig("positioner, axis", axes, "the setup has no positioner axes")
def test_axis_moves_by_step(positioner, axis):
    step = {"positionerName": positioner, "axis": axis, "dist": STEP_UM}
    back = {"positionerName": positioner, "axis": axis, "dist": -STEP_UM}

    before = position(positioner, axis)
    try:
        client.get_json("/PositionerController/movePositioner", step)
        time.sleep(SETTLE_S)
        after = position(positioner, axis)
        assert after == pytest.approx(before + STEP_UM, abs=0.1), (
            f"{positioner} {axis}: {before} -> {after}, expected {before + STEP_UM}")
    finally:
        client.get_json("/PositionerController/movePositioner", back)
