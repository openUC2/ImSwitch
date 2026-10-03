"""CanNetwork: one object for the UC2 CAN network behind a USB master."""
from pathlib import Path

from .bus import CanBus
from .firmware_server import FirmwareServer
from .guard import SerialPortGuard
from .ota import CanOta
from .updater import FirmwareUpdater, UpdateHooks
from .usb import UsbFlasher


class CanNetwork:
    """Discovery, node ids, firmware delivery (CAN OTA, USB flash) and the
    verified update of all boards.

    :param client: callable returning the uc2rest client of the USB master
                   (or None when no board is connected)
    :param firmware_url: firmware server serving the images + version.json
    :param cache_dir: where downloaded images are kept
    :param link: the host's own serial connection to the master, with
                 release() / restore() / current_port() — esptool needs the
                 port to itself
    :param hooks: what the host contributes to an update (UpdateHooks)
    :param emit_ota: called with every CAN OTA status payload
    :param emit_usb: called with every USB flash status payload
    """

    def __init__(self, client, firmware_url: str, cache_dir: Path, link,
                 hooks: UpdateHooks = None, emit_ota=None, emit_usb=None, logger=None):
        self.guard = SerialPortGuard()
        self.server = FirmwareServer(firmware_url, cache_dir, logger)
        self.bus = CanBus(client, logger)
        self.ota = CanOta(client, self.server, self.bus, self.guard, emit_ota, logger)
        self.usb = UsbFlasher(self.server, self.guard, link, emit_usb, logger)
        self.updater = FirmwareUpdater(self.server, self.bus, self.ota, self.usb, self.guard,
                                       hooks or UpdateHooks(), logger)
