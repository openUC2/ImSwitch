"""Check motor movement over HTTP."""

import os
import time

import pytest
import requests


BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")

STEP = 10
SETTLE = 0.5


def api(method, **params):
    """Call one PositionerController endpoint and return its JSON."""
    response = requests.get(
        f"{BASE_URL}/api/PositionerController/{method}",
        params=params,
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response.json()


def get_axes():
    """Every axis the setup reports, as (positioner, axis) pairs.

    Returns [] rather than raising: parametrize arguments are evaluated at
    collection, so an exception here would abort collection for the whole
    suite instead of skipping this one file.
    """
    try:
        positions = api("getPositionerPositions")
    except Exception:
        return []

    return [
        (positioner, axis)
        for positioner, axes in positions.items()
        for axis in axes
    ]


@pytest.mark.hardware
@pytest.mark.parametrize(
    "positioner,axis",
    get_axes() or [(None, None)],
)
def test_axis_moves_by_step(positioner, axis):
    """Each axis must report the requested step, then move back."""
    if positioner is None:
        pytest.skip("no positioner axes reported — ImSwitch or setup unavailable")

    before = api("getPositionerPositions")[positioner][axis]

    try:
        api(
            "movePositioner",
            positionerName=positioner,
            axis=axis,
            dist=STEP,
        )

        time.sleep(SETTLE)

        after = api("getPositionerPositions")[positioner][axis]

        assert after == pytest.approx(before + STEP, abs=0.1), (
            f"{positioner} {axis}: "
            f"{before} -> {after}, expected {before + STEP}"
        )

    finally:
        api(
            "movePositioner",
            positionerName=positioner,
            axis=axis,
            dist=-STEP,
        )
