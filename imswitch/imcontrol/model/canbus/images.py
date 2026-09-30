"""Which firmware image a board needs, and how two firmware versions compare.

Boards built with UC2_FW_IMAGE report the image they were built as (the bus
scan's "fwImage", /state_get "identifier_image"). Older firmware does not;
those boards are matched by their fixed CAN id (CAN_NODES).
"""
import re

# The fixed roles of the UC2 CANopen network: can_id -> (group, name, image).
# Single source for the legacy image lookup and getOTADeviceMapping.
CAN_NODES = {
    1: ("master", "master", "esp32_UC2_canopen_master_release.bin"),
    10: ("motors", "A", "esp32_UC2_canopen_slave_motor_release_motA.bin"),
    11: ("motors", "X", "esp32_UC2_canopen_slave_motor_release_motX.bin"),
    12: ("motors", "Y", "esp32_UC2_canopen_slave_motor_release_motY.bin"),
    13: ("motors", "Z", "esp32_UC2_canopen_slave_motor_release_motZ.bin"),
    14: ("motors", "motor_14", "esp32_UC2_canopen_slave_motor_release.bin"),
    15: ("motors", "motor_15", "esp32_UC2_canopen_slave_motor_release.bin"),
    20: ("laser", "laser_0", "esp32_UC2_canopen_slave_laser_release.bin"),
    21: ("laser", "laser_1", "esp32_UC2_canopen_slave_laser_release.bin"),
    22: ("laser", "laser_2", "esp32_UC2_canopen_slave_laser_release.bin"),
    30: ("led", "led_0", "esp32_UC2_canopen_slave_led_release.bin"),
    31: ("led", "led_1", "esp32_UC2_canopen_slave_led_release.bin"),
    40: ("galvo", "galvo", "esp32_UC2_canopen_slave_galvo_release.bin"),
}


def legacy_image(can_id):
    """Image for a node that does not report its own (None if the id has no fixed role)."""
    node = CAN_NODES.get(can_id)
    return node[2] if node else None


def legacy_images() -> dict:
    """{can_id: image} for every fixed role."""
    return {can_id: image for can_id, (_, _, image) in CAN_NODES.items()}


def device_mapping() -> dict:
    """{"master": 1, "motors": {"A": 10, ...}, "laser": {...}, ...}"""
    mapping = {}
    for can_id, (group, name, _) in CAN_NODES.items():
        if group == "master":
            mapping["master"] = can_id
        else:
            mapping.setdefault(group, {})[name] = can_id
    return mapping


_FW_TIMESTAMP = re.compile(r"-t(\d{14})(?:-|$)")


def update_status(installed, available) -> str:
    """Compare a board's UC2_FW_VERSION with the server's.

    Equal strings = same build. Firmware older than version reporting says
    "UC2-ESP v2.0" or nothing, which never matches -> "update_available".
    "device_newer" only when both carry a -t<timestamp> (non-tag builds) and
    the board's is later, e.g. a developer build that must not be offered a
    downgrade. "unknown" when the server publishes no version.
    Mirrored in frontend/src/components/firmwareStatus.js.
    """
    if not available:
        return "unknown"
    if installed == available:
        return "up_to_date"
    t_installed = _FW_TIMESTAMP.search(installed or "")
    t_available = _FW_TIMESTAMP.search(available)
    if t_installed and t_available and t_installed.group(1) > t_available.group(1):
        return "device_newer"
    return "update_available"
