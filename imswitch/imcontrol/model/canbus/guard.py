"""One writer on the master's USB serial port at a time."""
import threading

CAN_BUSY = "A CAN streaming OTA upload is already running — wait for it to finish or cancel it."
USB_BUSY = "A USB flash is using the serial port — wait for it to finish."


class SerialPortGuard:
    """The master's serial port carries either one CAN OTA stream or one
    esptool flash. Two writers steal each other's ACKs and inject commands
    into the other's binary stream."""

    def __init__(self):
        self.can = threading.Lock()
        self.usb = threading.Lock()

    def claim_can(self):
        """Take the CAN OTA lock without waiting: None when taken, else why not."""
        if self.usb.locked():
            return USB_BUSY
        if not self.can.acquire(blocking=False):
            return CAN_BUSY
        return None

    def busy_reasons(self) -> list:
        return [msg for lock, msg in ((self.can, CAN_BUSY), (self.usb, USB_BUSY)) if lock.locked()]
