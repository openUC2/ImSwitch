"""The UC2 board is connected and runs the CANopen master firmware. Only reads."""
import hil_client as client


def test_uc2_board_connected():
    connected = client.get_json("/UC2ConfigController/is_connected", timeout=5)
    assert connected is True, "UC2/ESP32 board is not reported as connected"


def test_correct_esp32_firmware():
    info = client.get_json("/UC2ConfigController/getFirmwareInfo", timeout=5)
    print(f"\nESP32 firmware: {info}")
    assert info.get("connected") is True, f"ESP32 not connected: {info}"
    assert info.get("isMaster") is True, f"ESP32 is not the CANopen master: {info}"
    assert info.get("pindef") == "UC2_canopen_master", f"wrong firmware type: {info.get('pindef')}"
