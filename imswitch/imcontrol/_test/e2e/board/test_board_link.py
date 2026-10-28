"""Check communication with the UC2 board."""

import os

import pytest
import requests


BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")


def get(path):
    """GET an endpoint, skipping the test when ImSwitch is unreachable.

    A refused connection says nothing about the board, and the suite is meant
    to be runnable without a rig.
    """
    try:
        return requests.get(f"{BASE_URL}/api/{path}", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"ImSwitch not reachable at {BASE_URL}: {exc}")


@pytest.mark.hardware
def test_uc2_board_connected():
    """ImSwitch must report the UC2/ESP32 board as connected."""
    response = get("UC2ConfigController/is_connected")

    assert response.status_code == 200, response.text

    assert response.json() is True, (
        "UC2/ESP32 board is not reported as connected"
    )


@pytest.mark.hardware
def test_correct_esp32_firmware():
    """The connected ESP32 must run the CANopen master firmware."""
    response = get("UC2ConfigController/getFirmwareInfo")

    if response.status_code == 404:
        pytest.skip(
            "getFirmwareInfo endpoint not available in this ImSwitch version"
        )

    assert response.status_code == 200, response.text

    info = response.json()

    assert info, "UC2 board returned no firmware information"

    print(f"\nESP32 firmware: {info}")

    assert info.get("connected") is True, (
        f"ESP32 not connected: {info}"
    )

    assert info.get("isMaster") is True, (
        f"ESP32 is not the CANopen master: {info}"
    )

    assert info.get("pindef") == "UC2_canopen_master", (
        f"wrong firmware type: {info.get('pindef')}"
    )