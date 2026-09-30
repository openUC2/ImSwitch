"""The UC2 CAN network behind a USB master: discovery, node ids and firmware.

Independent of ImSwitch: it needs a uc2rest client and a few callbacks. Build
everything at once with CanNetwork, or use the parts:

- FirmwareServer: image listing, version.json, verified (sha256) downloads
- CanBus: bus scan, node-id reassignment, restarts, waiting for a version
- CanOta: CAN streaming OTA with retries and post-flash verification
- UsbFlasher: esptool flashing and serial bring-up of fresh boards
- FirmwareUpdater: check all boards / update the outdated ones (UpdateHooks)
- images: which image a board needs, update_status() of two versions

ImSwitch exposes it as UC2ConfigController endpoints
(controllers/uc2config/can_network_api.py). See docs/FIRMWARE_VERSIONING.md.
"""
from .bus import CanBus
from .firmware_server import FirmwareServer
from .guard import SerialPortGuard
from .images import CAN_NODES, device_mapping, legacy_image, update_status
from .network import CanNetwork
from .ota import CanOta
from .updater import FirmwareUpdater, UpdateHooks
from .usb import UsbFlasher

__all__ = ["CanNetwork", "CanBus", "CanOta", "FirmwareServer", "FirmwareUpdater", "UpdateHooks",
           "UsbFlasher", "SerialPortGuard", "CAN_NODES", "device_mapping", "legacy_image",
           "update_status"]
