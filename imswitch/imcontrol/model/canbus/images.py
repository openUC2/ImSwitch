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


# pindefName -> PlatformIO env, where the two differ and the env is unambiguous
# (uc2-ESP main/config/<env>/PinConfig.h). Only firmware built before
# UC2_FW_IMAGE needs this: newer boards report their image.
PINDEF_ENVS = {
    "UC2_esp32s3_xiao": "seeed_xiao_esp32s3",
    "UC2_esp32s3_xiao_ledservo": "seeed_xiao_esp32s3_ledservo",
    "UC2_3_I2CSlaveLaser": "seeed_xiao_esp32s3_ledring",
}


def image_candidates(fw_image=None, pindef=None, can_id=None) -> list:
    """[(image, source)] that fit a board, best first.

    source: "reported" (the board says it was built as this image), "can_id"
    (its fixed CAN role) or "pindef" (the env named after its pin definition).
    A slave's role outranks its pindef: all motor axes share one pindef. The
    master role comes last: several boards act as master (CAN HAT, UC2_4_CAN,
    standalone v4), and only the pindef tells them apart.
    """
    candidates = []
    if fw_image:
        candidates.append((fw_image, "reported"))
    role = CAN_NODES.get(can_id)
    if role and role[0] != "master":
        candidates.append((role[2], "can_id"))
    if pindef:
        for env in dict.fromkeys((PINDEF_ENVS.get(pindef), pindef)):
            if env:
                candidates += [(f"esp32_{env}_release.bin", "pindef"),
                               (f"esp32_{env}.bin", "pindef")]
    if role and role[0] == "master":
        candidates.append((role[2], "can_id"))
    first = {}
    for name, source in candidates:  # one entry per file, its best reason
        first.setdefault(name, source)
    return list(first.items())


def choose_image(fw_image, pindef, can_id, on_server) -> tuple:
    """(image, image_source) for the update check: the reported image as is,
    else the first candidate on the server (else the first candidate)."""
    if fw_image:
        return fw_image, "reported"
    names = [name for name, _ in image_candidates(None, pindef, can_id)]
    image = next((n for n in names if n in on_server), names[0] if names else None)
    return image, "mapping"


def merged_name(image: str) -> str:
    """esp32_X.bin -> esp32_X_merged.bin (bootloader + partitions, flashed at 0x0)."""
    return image[:-len(".bin")] + "_merged.bin" if image.endswith(".bin") else image


def recommend_image(identity: dict, on_server) -> dict:
    """Which file on the firmware server to flash onto the board *identity*
    describes ({fwImage, pindef, canId, ...} as /state_get reports it).

    Returns {filename, merged, source, reason, candidates: [{filename,
    source, on_server}]}. filename is None when no candidate is on the
    server; merged is the _merged twin when the server has it.
    """
    on_server = set(on_server or ())
    candidates = image_candidates(identity.get("fwImage"), identity.get("pindef"),
                                  identity.get("canId"))
    listed = [{"filename": name, "source": source, "on_server": name in on_server}
              for name, source in candidates]
    best = next((c for c in listed if c["on_server"]), None)
    if best is None:
        expected = listed[0]["filename"] if listed else None
        reason = (f"No image on the server matches this board (expected {expected})."
                  if expected else "The board reports neither its image nor a pin definition.")
        return {"filename": None, "merged": None, "source": None, "reason": reason,
                "candidates": listed}
    reasons = {
        "reported": "the board reports it was built as this image",
        "can_id": f"the image for CAN id {identity.get('canId')}",
        "pindef": f"named after the board's pin definition {identity.get('pindef')}",
    }
    reason = reasons[best["source"]]
    if listed[0]["source"] == "reported" and best is not listed[0]:
        reason += f" (the reported {listed[0]['filename']} is not on the server)"
    merged = merged_name(best["filename"])
    return {"filename": best["filename"], "merged": merged if merged in on_server else None,
            "source": best["source"], "reason": reason, "candidates": listed}


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
