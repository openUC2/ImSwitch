"""ImSwitch talks to the UC2 board over USB, and the board runs the master firmware,
the one that passes commands on to the motor and LED boards over the CAN bus.

Only reads.
"""
import hil_client as client


def test_uc2_board_connected():
    # Not is_connected: that one only says that ImSwitch answers, it is always true.
    connected = client.get_json("/UC2ConfigController/uc2_board_is_connected", timeout=5)
    assert connected is True, "UC2/ESP32 board is not reported as connected"


def test_correct_esp32_firmware():
    info = client.get_json("/UC2ConfigController/getFirmwareInfo", timeout=5)
    print(f"\nESP32 firmware: {info}")
    assert info.get("connected") is True, f"ESP32 not connected: {info}"
    assert info.get("isMaster") is True, f"ESP32 does not run the master firmware: {info}"
