"""Check that the LED matrix accepts an off command over HTTP."""

import os

import pytest
import requests


BASE_URL = os.environ.get(
    "IMSWITCH_URL",
    "http://localhost:8001",
)


def get(path):
    """GET an endpoint, skipping the test when ImSwitch is unreachable.

    A refused connection says nothing about the matrix, and the suite is meant
    to be runnable without a rig.
    """
    try:
        return requests.get(f"{BASE_URL}/api/{path}", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"ImSwitch not reachable at {BASE_URL}: {exc}")


def get_available_controllers():
    """Controllers of the active setup; the only way to spot the matrix."""
    response = get("getAvailableControllers")

    assert response.status_code == 200, (
        f"getAvailableControllers -> "
        f"{response.status_code}: {response.text}"
    )

    return response.json()


@pytest.mark.hardware
def test_all_leds_off_is_accepted():
    """setAllLEDOff must be accepted where the setup has a matrix."""
    controllers = get_available_controllers()

    # The response is either a plain list or a dictionary.
    if isinstance(controllers, dict):
        controllers = controllers.get(
            "controllers",
            controllers.get("availableControllers", []),
        )

    if "LEDMatrixController" not in controllers:
        pytest.skip("LEDMatrixController not available in this setup")

    response = get("LEDMatrixController/setAllLEDOff")

    assert response.status_code == 200, (
        f"LEDMatrixController/setAllLEDOff -> "
        f"{response.status_code}: {response.text}"
    )
