"""Each motor axis moves 10 µm and back, and ImSwitch reports the position it moved to.
After that the stage drives to its home position.

MOVES REAL HARDWARE (10 µm, then to the end stops).

Every axis with homing switched on homes, Z first.
So the objective is away from the sample before X and Y move.
The stage stays at home, where the slide holder is out of the light path for the light tests.
When a test of this file failed, the axes after it do not home.

homeAxis does not say whether the end stop was found.
So the homing only fails when ImSwitch answers with an error or not at all.
The FRAME calibration homes again and checks each axis.
"""
import time

import pytest

import hil_client as client

STEP_UM = 10
SETTLE_S = 0.5      # wait before reading the position back
HOME_ORDER = "ZXYA"
HOME_EXTRA_S = 30   # seconds homeAxis may take longer than the axis's homeTimeout in the setup


def axes():
    """Every (positioner, axis) pair of the setup, e.g. ("ESP32Stage", "X")."""
    return [(name, axis) for name, positioner in client.setup()["positioners"].items()
            if positioner["forPositioning"] for axis in positioner["axes"]]


def axes_to_home():
    """Every (positioner, axis) with homing switched on, in HOME_ORDER."""
    found = []
    for name, positioner in client.setup()["positioners"].items():
        properties = positioner["managerProperties"]
        for axis in HOME_ORDER:
            # as ESP32StageManager decides it: an end stop, or a fixed number of steps
            homes = properties.get(f"home{axis}enabled") or properties.get(f"homeSteps{axis}")
            if axis in positioner["axes"] and homes:
                found.append((name, axis))
    return found


def position(positioner, axis):
    return client.get_json("/PositionerController/getPositionerPositions")[positioner][axis]


@pytest.fixture(scope="module", autouse=True)
def failures_before(request):
    """How many tests of the run had failed when this file started."""
    return request.session.testsfailed  # pytest counts the failures of the run


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


@client.parametrize_from_rig("positioner, axis", axes_to_home, "no axis has homing switched on")
def test_axis_homes(positioner, axis, failures_before, request):
    if request.session.testsfailed > failures_before:
        pytest.skip("not homing: a step or an axis before this one failed")
    properties = client.setup()["positioners"][positioner]["managerProperties"]
    timeout = properties.get(f"homeTimeout{axis}", 20000) / 1000 + HOME_EXTRA_S
    client.get_json("/PositionerController/homeAxis", timeout=timeout, params={
        "positionerName": positioner, "axis": axis, "isBlocking": True})
