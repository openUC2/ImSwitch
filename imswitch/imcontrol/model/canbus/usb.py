"""USB side of the CAN network: flashing boards with esptool and bringing
freshly flashed boards up over their own serial port (CAN address, state
probe, test action).

Flash status payloads (``emit``, e.g. ImSwitch's sigUSBFlashStatusUpdate):
{status, progress, message, details, timestamp}; status is one of
disconnecting, downloading, flashing, reconnecting, success, warning, failed,
cancelled. ``link`` is the host's own connection to the master: release()
before esptool takes the port, restore() afterwards, current_port().
"""
import datetime
import json
import logging
import subprocess
import sys
import threading
import time

import serial
from serial.tools import list_ports

from .guard import CAN_BUSY, SerialPortGuard

VID_PID_CHIP = {
    (0x10c4, 0xea60): "esp32",    # CP2102 – UC2 CAN HAT (ESP32)
    (0x303a, 0x1001): "esp32s3",  # USB JTAG/serial – UC2 XIAO (ESP32-S3)
    (0x1a86, 0x55d4): "esp32s3",  # CH9102 – some XIAO boards
    (0x303a, 0x0002): "esp32s2",  # ESP32-S2
}
_MERGED_BOOT_OFFSETS = {"esp32": (0x0, 0x1000), "esp32s2": (0x0,), "esp32s3": (0x0,),
                        "esp32c3": (0x0,)}
_FLASH_MODES = {0: "QIO", 1: "QOUT", 2: "DIO", 3: "DOUT"}
_FLASH_FREQS = {0x0: "40m", 0x1: "26m", 0x2: "20m", 0xF: "80m"}
_FLASH_SIZES = {0x0: "1MB", 0x1: "2MB", 0x2: "4MB", 0x3: "8MB", 0x4: "16MB"}


def _boot_failed(text: str) -> bool:
    """Boot-loop markers of a badly flashed ESP32."""
    return "0xffffffff" in text or "invalid header" in text.lower()


def _read_reply(ser, timeout: float, quiet: float = 0.0) -> str:
    """Read until the device has been silent for *quiet* seconds after
    sending something, at most *timeout*."""
    reply, deadline, last_data = "", time.time() + timeout, time.time()
    while time.time() < deadline:
        if ser.in_waiting:
            reply += ser.read(ser.in_waiting).decode("utf-8", errors="replace")
            last_data = time.time()
            time.sleep(0.05)
        elif reply and time.time() - last_data >= quiet:
            break
        else:
            time.sleep(0.1)
    return reply


def parse_state_reply(text: str) -> dict:
    """The identity in a /state_get reply, flat like uc2rest's
    get_firmware_info(): {name, version, fwVersion, fwImage, date, author,
    pindef, isMaster, canId, gitCommit}. {} when the reply holds no state
    (boot noise, the ++/-- frame and other JSON lines are skipped)."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        state = obj.get("state") if isinstance(obj, dict) else None
        if isinstance(state, dict):
            pindef = str(state.get("pindef") or "")
            can_id = state.get("CAN_SLAVE")
            return {"name": state.get("identifier_name", ""),
                    "version": state.get("identifier_id", ""),
                    "fwVersion": state.get("identifier_version", ""),
                    "fwImage": state.get("identifier_image", ""),
                    "date": state.get("identifier_date", ""),
                    "author": state.get("identifier_author", ""),
                    "pindef": pindef, "isMaster": "master" in pindef.lower(),
                    "canId": can_id if isinstance(can_id, int) and can_id > 0 else None,
                    "gitCommit": state.get("git_commit", "")}
        start = text.find("{", end)
    return {}


def _exchange(ser, command: dict, timeout: float, settle=0.5) -> str:
    """Send one JSON command line and return the reply."""
    ser.reset_input_buffer()
    ser.write((json.dumps(command) + "\n").encode("utf-8"))
    ser.flush()
    time.sleep(settle)
    return _read_reply(ser, timeout)


class UsbFlasher:
    def __init__(self, server, guard: SerialPortGuard, link, emit=None, logger=None):
        self.server = server
        self.guard = guard
        self.link = link
        self._emit_signal = emit or (lambda status: None)
        self._log = logger or logging.getLogger(__name__)
        self.status = {"status": "idle", "progress": 0, "message": "", "details": None}
        self.cancel_event = threading.Event()
        self._proc = None  # running esptool, so cancel can reach it

    def _emit(self, status, progress, message, details=None):
        self.status = {"status": status, "progress": progress, "message": message,
                       "details": details, "timestamp": datetime.datetime.now().isoformat()}
        self._emit_signal(self.status)
        self._log.info(f"USB Flash: {message} ({progress}%)")

    # ── ports / chips / images ──────────────────────────────────────────────

    def list_ports(self) -> list:
        try:
            return [{"device": p.device, "description": p.description or "",
                     "manufacturer": p.manufacturer or "", "product": p.product or "",
                     "hwid": p.hwid or "", "vid": p.vid, "pid": p.pid,
                     "serial_number": p.serial_number} for p in list_ports.comports()]
        except Exception as e:
            self._log.error(f"Failed to list serial ports: {e}")
            return []

    def find_port(self, match="HAT") -> str:
        """Port whose metadata contains *match*, else a USB-looking port;
        the port the host already uses wins among the candidates."""
        ports = self.list_ports()
        if not ports:
            raise RuntimeError("No serial ports found.")
        needle = (match or "").strip().lower()
        fields = ("device", "description", "manufacturer", "product", "hwid")
        candidates = [p for p in ports
                      if needle and needle in " ".join(p.get(f) or "" for f in fields).lower()]
        if not candidates:
            candidates = [p for p in ports
                          if any(x in p["device"].lower()
                                 for x in ("/dev/ttyusb", "/dev/ttyacm", "com"))
                          or "usb" in p["hwid"].lower() or "usb" in p["description"].lower()]
        if not candidates:
            raise RuntimeError(f"No candidate serial ports found for match='{match}'. "
                               f"Ports={ports}")
        current = self.link.current_port()
        if any(c["device"] == current for c in candidates):
            return current

        def rank(p):  # native USB (ttyACM) before USB-UART (ttyUSB)
            device = p["device"].lower()
            return 0 if "/dev/ttyacm" in device else 1 if "/dev/ttyusb" in device else 2
        return sorted(candidates, key=rank)[0]["device"]

    def detect_chip(self, port):
        for p in list_ports.comports():
            if p.device == port:
                chip = VID_PID_CHIP.get((p.vid, p.pid)) if p.vid is not None else None
                if chip:
                    self._log.info(f"Auto-detected chip={chip} from VID:{p.vid:04x} "
                                   f"PID:{p.pid:04x} on {port}")
                return chip
        return None

    def validate(self, path, is_merged: bool, chip: str) -> dict:
        """Pre-flight check of an image: the ESP32 magic byte 0xE9 must exist,
        and for a merged image sit at the chip's boot offset — an image built
        with the wrong BOOT_ADDR flashes fine and then boot-loops with
        "invalid header: 0xffffffff". Returns {valid, magic_offset,
        flash_mode, flash_freq, flash_size, warnings, errors}."""
        result = {"valid": True, "magic_offset": None, "flash_mode": None, "flash_freq": None,
                  "flash_size": None, "warnings": [], "errors": []}
        try:
            with open(path, "rb") as f:
                header = f.read(0x11000)
        except Exception as e:
            result.update(valid=False, errors=[f"Cannot read firmware file: {e}"])
            return result
        if len(header) < 8:
            result.update(valid=False, errors=[f"Firmware file too small ({len(header)} bytes)"])
            return result
        magic = header.find(0xE9)
        if magic < 0:
            result.update(valid=False, errors=[
                "No ESP32 bootloader magic byte (0xE9) found in first 64 KB – "
                "not a valid firmware"])
            return result
        hdr = header[magic:magic + 4]
        result.update(magic_offset=magic,
                      flash_mode=_FLASH_MODES.get(hdr[2], f"unknown(0x{hdr[2]:02x})"),
                      flash_freq=_FLASH_FREQS.get(hdr[3] & 0x0F, f"unknown(0x{hdr[3] & 0x0F:02x})"),
                      flash_size=_FLASH_SIZES.get((hdr[3] & 0xF0) >> 4, "unknown"))
        expected = _MERGED_BOOT_OFFSETS.get(chip, (0x0, 0x1000))
        if is_merged and magic not in expected:
            pretty = " or ".join(f"0x{o:x}" for o in expected)
            result["valid"] = False
            result["errors"].append(
                f"Merged binary has bootloader at 0x{magic:x} instead of {pretty} (expected for "
                f"{chip}). The binary was built with a wrong boot address. Flashing this at 0x0 "
                f"will produce 'invalid header: 0xffffffff'. → Rebuild with the correct BOOT_ADDR "
                f"for {chip}.")
        self._log.info(f"Firmware validation: magic@0x{magic:x}, mode={result['flash_mode']}, "
                       f"freq={result['flash_freq']}, size={result['flash_size']}, "
                       f"valid={result['valid']}")
        for e in result["errors"]:
            self._log.error(f"Firmware error: {e}")
        return result

    # ── flashing ────────────────────────────────────────────────────────────

    def flash(self, port=None, match="HAT", baud=921600, firmware_filename=None,
              reconnect_after=True, chip="auto", erase_flash=False, skip_disconnect=False) -> dict:
        """Flash a board over USB (see run()). Refused while a CAN OTA streams
        through the port; a previous flash still running is cancelled first."""
        if self.guard.can.locked():
            return {"status": "busy", "message": CAN_BUSY}
        if not self.guard.usb.acquire(blocking=False):
            self._log.warning("USB flash requested while a previous flash is still active "
                              "— cancelling the previous run and continuing.")
            self.cancel(reason="superseded by new flash request")
            if not self.guard.usb.acquire(timeout=5.0):
                self._emit("failed", 0,
                           "A previous flash is still running and did not release in time.")
                return {"status": "error",
                        "message": "Previous flash is still running — try again in a few seconds."}
        try:
            self.cancel_event.clear()
            return self.run(port=port, match=match, baud=baud, firmware_filename=firmware_filename,
                            reconnect_after=reconnect_after, chip=chip, erase_flash=erase_flash,
                            skip_disconnect=skip_disconnect)
        except Exception as e:
            self._log.error(f"Unexpected flash error: {e}", exc_info=True)
            self._emit("failed", 0, f"Unexpected error: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self.guard.usb.release()

    def run(self, *, port, match, baud, firmware_filename, reconnect_after, chip, erase_flash,
            skip_disconnect) -> dict:
        """The flash itself; the caller holds guard.usb. Releases the host's
        link, downloads + validates the image, writes it (merged images at 0x0,
        app-only images at 0x10000 keeping the on-device bootloader), then
        restores the link."""
        self._release_link(skip_disconnect)
        path = self._fetch_image(firmware_filename)
        if path is None:
            return {"status": "error", "message": "Firmware not found/downloaded."}
        try:
            flash_port = port or self.find_port(match=match)
        except Exception as e:
            self._emit("failed", 20, f"Failed to find serial port: {e}")
            return {"status": "error", "message": f"Failed to resolve serial port: {e}",
                    "available_ports": self.list_ports()}
        self._emit("flashing", 25, f"Using port: {flash_port}")
        chip = chip if chip and chip != "auto" else (self.detect_chip(flash_port) or "esp32s3")
        is_merged = "_merged" in path.name
        rejected = self._reject_image(path, is_merged, chip)
        if rejected:
            return rejected
        details = {"port": flash_port, "firmware": str(path), "chip": chip}
        self._log.info(f"Flashing {path.name} via {flash_port} (baud={baud}, chip={chip}, "
                       f"erase={erase_flash})")
        failure = self._write(["--port", flash_port, "--baud", str(baud), "--chip", chip],
                              path, is_merged, erase_flash, details)
        if failure:
            return failure
        if reconnect_after and not skip_disconnect:
            self._emit("reconnecting", 90, "Reconnecting to device...")
            try:
                time.sleep(1.0)
                self.link.restore()
            except Exception as e:
                self._emit("warning", 95, "Flashed OK, but reconnect failed", str(e))
                return {"status": "warning", "message": "Flashed OK, but reconnect failed",
                        "details": str(e), **details}
            self._emit("success", 100, "✅ Firmware flashed and reconnected!")
        else:
            self._emit("success", 100, "✅ Firmware flashed successfully!")
        return {"status": "success", "message": "Firmware flashed via USB", "baud": int(baud),
                "reconnect_after": bool(reconnect_after), **details}

    def _release_link(self, skip_disconnect):
        self._emit("flashing", 2, "⚡ Hint: If flashing fails on XIAO boards, turn off 12V "
                                  "power (press the emergency stop button) before flashing.")
        if skip_disconnect:
            self._emit("disconnecting", 5, "Skipping disconnect (non-HAT device)")
            time.sleep(0.2)
        else:
            self._emit("disconnecting", 5, "Disconnecting from ESP32...")
            self.link.release()
            time.sleep(0.5)  # let the OS release the port

    def _fetch_image(self, firmware_filename):
        """The named image, else the master image; verified. None if missing."""
        self._emit("downloading", 10, "Downloading firmware from server...")
        name = firmware_filename or self.server.master_image()
        path = self.server.download(name) if name else None
        if not path:
            self._emit("failed", 10, "Firmware not found on server")
            return None
        self._emit("downloading", 20, f"Firmware downloaded: {path.name}")
        return path

    def _reject_image(self, path, is_merged, chip):
        """None if the image may be flashed, else the error result."""
        check = self.validate(path, is_merged, chip)
        if check["valid"]:
            self._emit("flashing", 29, f"✅ Binary OK: magic@0x{check['magic_offset']:x}, "
                                       f"mode={check['flash_mode']}, freq={check['flash_freq']}")
            return None
        detail = " | ".join(check["errors"])
        self._emit("failed", 28, f"❌ Binary validation failed – will not flash: {detail}")
        magic = check["magic_offset"]
        return {"status": "error", "details": detail, "firmware": str(path),
                "message": "Binary validation failed – flashing aborted to prevent boot-loop",
                "magic_offset": f"0x{magic:x}" if magic is not None else None,
                "flash_mode_in_binary": check["flash_mode"],
                "flash_freq_in_binary": check["flash_freq"],
                "hint": "Rebuild the firmware using the corrected CI YAML (BOOT_ADDR=0x0000, "
                        "flash_mode=qio, flash_freq=80m for ESP32-S3). Then clear the firmware "
                        "cache and retry."}

    def _write(self, common, path, is_merged, erase, details):
        """Optional erase (merged images only) + write. None on success."""
        offset = "0x0" if is_merged else "0x10000"
        self._emit("flashing", 27, f"Chip: {details['chip']}, offset: {offset}")
        if erase and not is_merged:
            erase = False  # would wipe the bootloader/partitions an app image needs
            self._log.warning("Erase flash disabled: non-merged firmware cannot be flashed "
                              "after erasing (bootloader/partitions would be lost).")
            self._emit("flashing", 22, "⚠️ Erase disabled – non-merged firmware needs "
                                       "existing bootloader")
        if erase:
            self._emit("flashing", 28, "Erasing flash...")
            ok, output = self._esptool(common + ["erase_flash"])
            if not ok:
                return self._esptool_failure("Erase", "Flash erase failed", "erase_flash failed",
                                             output, 29, {"port": details["port"]})
            self._emit("flashing", 30, "Flash erased successfully")
        self._emit("flashing", 30, "Writing firmware to device...")
        # A merged image carries its flash settings in the bootloader; an app
        # image keeps the ones compiled into its header ("keep" mirrors `pio
        # run -t upload` — overriding them mismatches the on-device bootloader).
        write = ["write-flash", "0x0", str(path)] if is_merged else [
            "write-flash", "--flash-mode", "keep", "--flash-freq", "keep", "--flash-size", "keep",
            offset, str(path)]
        ok, output = self._esptool(common + write)
        if not ok:
            return self._esptool_failure("Flash", "Firmware write failed", "write-flash failed",
                                         output, 50, details)
        self._emit("flashing", 85, "Firmware written successfully!")
        return None

    def _esptool_failure(self, what, status_message, message, output, progress, details):
        if self.cancel_event.is_set():  # the "cancelled" status was emitted by cancel()
            return {"status": "cancelled", "message": f"{what} cancelled by user", **details}
        self._emit("failed", progress, status_message, output)
        return {"status": "error", "message": message, "details": output, **details}

    def cancel(self, reason="cancelled by user") -> bool:
        """Stop a running esptool; True if a process was terminated."""
        self.cancel_event.set()
        proc, killed = self._proc, False
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                killed = True
                self._log.info(f"esptool subprocess terminated ({reason})")
            except Exception as e:
                self._log.warning(f"Failed to terminate esptool: {e}")
        self._emit("cancelled", -1, reason)  # unblocks the UI even after esptool's last line
        return killed

    def cancel_request(self) -> dict:
        if self._proc is None and not self.guard.usb.locked():
            return {"status": "idle", "message": "No flash in progress."}
        killed = self.cancel()
        return {"status": "cancelled", "killed": killed,
                "message": ("Flash cancel requested — subprocess terminated." if killed
                            else "Flash cancel requested.")}

    def _esptool(self, args) -> tuple:
        """Run esptool, streaming its lines as "flashing" status (the write
        progress mapped to 30–85 %). Returns (ok, output)."""
        lines = []
        try:
            proc = subprocess.Popen([sys.executable, "-m", "esptool", *args],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
        except Exception as e:
            return False, f"Failed to run esptool: {e}"
        self._proc = proc
        try:
            for line in proc.stdout:
                if self.cancel_event.is_set():
                    self._log.warning("Flash cancel requested — terminating esptool")
                    proc.terminate()
                    break
                line = line.rstrip("\r\n")
                if not line:
                    continue
                lines.append(line)
                progress = -1
                if "Writing at" in line and "%" in line:
                    try:
                        progress = 30 + int(int(line.split("(")[1].split("%")[0]) * 0.55)
                    except (IndexError, ValueError):
                        pass
                self._emit("flashing", progress, line)
            try:
                proc.wait(timeout=2 if self.cancel_event.is_set() else None)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        finally:
            self._proc = None
        if self.cancel_event.is_set():
            return False, "cancelled by user"
        return proc.returncode == 0, "\n".join(lines)

    # ── bring-up of a freshly flashed board on its own serial port ─────────

    def send_can_address(self, port, address=1, baud=115200, timeout=2.0) -> dict:
        """Wait for the board to boot (detects boot loops), assign its CAN
        address, then check it answers /state_get."""
        if not port:
            return {"status": "error", "message": "No serial port specified"}
        command = {"task": "/can_act", "address": int(address), "nodeId": int(address),
                   "canMotorAxis": 1}
        self._log.info(f"Sending CAN address {address} to {port} @ {baud} baud")
        self._emit("flashing", 92, f"Assigning CAN address {address}...")
        try:
            with serial.Serial(port, baud, timeout=timeout) as ser:
                boot = _read_reply(ser, 2, quiet=1.0)  # boot log until 1 s of silence
                if boot:
                    self._log.info(f"Boot output from device:\n{boot.strip()[-500:]}")
                if _boot_failed(boot):
                    message = ("Device appears to be in a boot-loop (invalid header: 0xffffffff). "
                               "The firmware was likely not flashed correctly. Try flashing with a "
                               "merged firmware (.._merged.bin) and erase flash enabled.")
                    self._log.error(message)
                    self._emit("failed", 92, "❌ Boot-loop detected!", message)
                    return {"status": "error", "message": message,
                            "boot_output": boot.strip()[-500:], "boot_loop_detected": True}
                response = _exchange(ser, command, timeout)
                state = _exchange(ser, {"task": "/state_get"}, timeout)
            self._log.info(f"CAN address response: {response.strip()}")
            if _boot_failed(state):
                self._emit("failed", 95, "❌ Firmware verification failed "
                                         "(device not responding correctly)")
                return {"status": "error",
                        "message": "Firmware verification failed – device not responding correctly",
                        "response": response.strip(), "state_response": state.strip()[:200],
                        "boot_loop_detected": True}
            self._emit("success", 100, f"✅ CAN address {address} assigned!")
            return {"status": "success", "message": f"CAN address {address} assigned on {port}",
                    "response": response.strip(), "state_response": state.strip()[:200],
                    "firmware_verified": True}
        except Exception as e:
            self._log.error(f"Failed to send CAN address: {e}")
            self._emit("failed", 92, f"CAN address assignment failed: {e}")
            return {"status": "error", "message": str(e)}

    def probe_state(self, port, baud=115200, timeout=2.0) -> dict:
        """/state_get on the board's own port: is the firmware running?"""
        if not port:
            return {"status": "error", "message": "No serial port specified"}
        self._log.info(f"Probing device state on {port} @ {baud} baud")
        try:
            with serial.Serial(port, baud, timeout=timeout) as ser:
                state = _exchange(ser, {"task": "/state_get"}, timeout)
            ok = not _boot_failed(state)
            return {"status": "success" if ok else "warning", "state_response": state.strip()[:500],
                    "firmware_ok": ok}
        except Exception as e:
            self._log.error(f"Failed to probe device state: {e}")
            return {"status": "error", "message": str(e)}

    def identify(self, port, baud=115200, timeout=2.0) -> dict:
        """Which firmware a board on its own *port* runs: /state_get, parsed
        (parse_state_reply), plus the chip guessed from the USB VID:PID.
        Refused while a flash or CAN OTA owns a serial port. Opening the port
        may reset the board (DTR/RTS), so the boot log is waited out first."""
        if not port:
            return {"status": "error", "message": "No serial port specified"}
        busy = self.guard.busy_reasons()
        if busy:
            return {"status": "busy", "message": busy[0]}
        try:
            with serial.Serial(port, baud, timeout=timeout) as ser:
                _read_reply(ser, 2, quiet=1.0)  # boot output after the reset, if any
                reply = _exchange(ser, {"task": "/state_get"}, timeout)
        except Exception as e:
            self._log.error(f"Failed to identify the board on {port}: {e}")
            return {"status": "error", "message": str(e)}
        identity = parse_state_reply(reply)
        if not identity:
            return {"status": "error", "port": port, "state_response": reply.strip()[:500],
                    "message": f"No firmware state from {port} at {baud} baud (no UC2 "
                               f"firmware, another baud rate, or a boot loop)."}
        return {"status": "success", "port": port, "chip": self.detect_chip(port),
                "identity": identity}

    def test_action(self, port, device_type="motor", baud=115200, timeout=2.0, stepperid=1,
                    speed=2000, position=1000, isabs=0, r=25, g=25, b=25, led_action="fill",
                    laserid=1, laserval=118) -> dict:
        """Make a freshly flashed slave do something visible over its own
        port: a motor move, LEDs, or a laser. Moves hardware."""
        if not port:
            return {"status": "error", "message": "No serial port specified"}
        kind = (device_type or "").lower().strip()
        if kind == "motor":
            command = {"task": "/motor_act", "qid": 5, "motor": {"steppers": [{
                "stepperid": int(stepperid), "speed": int(speed), "position": int(position),
                "isabs": int(isabs)}]}}
        elif kind in ("ledarray", "led", "ledarr"):
            command = {"task": "/ledarr_act", "qid": 17, "led": {
                "action": led_action or "fill", "r": int(r), "g": int(g), "b": int(b)}}
        elif kind == "laser":
            command = {"task": "/laser_act", "LASERid": int(laserid), "LASERval": int(laserval),
                       "qid": 1}
        else:
            return {"status": "error", "message": f"Unknown device_type '{device_type}'. "
                                                  f"Use 'motor', 'ledarray', or 'laser'."}
        self._log.info(f"Test device action ({kind}) on {port} @ {baud} baud: {command}")
        try:
            with serial.Serial(port, int(baud), timeout=float(timeout)) as ser:
                response = _exchange(ser, command, float(timeout), settle=0.3).strip()
            self._log.info(f"Test device response ({kind}): {response[:300]}")
            reported_error = '"status":"error"' in response.lower().replace(" ", "")
            ok = not _boot_failed(response) and not reported_error
            return {"status": "success" if ok else "warning", "device_type": kind,
                    "command": command, "response": response[:1000]}
        except Exception as e:
            self._log.error(f"Failed to send test action ({kind}): {e}")
            return {"status": "error", "message": str(e)}
