"""UC2ConfigController's CAN network endpoints (uc2config/can_network_api.py).

The controller is built without __init__ around fake ImSwitch parts (manager,
controller registry, lasers, positioners, signals). The package itself is
covered by test_canbus.py; this pins the ImSwitch side: routes delegate and
keep their shapes, the busy checks, lasers off, homing, the startup prompt.
"""
import logging
import threading
from types import SimpleNamespace

import pytest

from imswitch.imcontrol.controller.controllers.UC2ConfigController import UC2ConfigController
from imswitch.imcontrol.model.SetupInfo import UC2ConfigInfo


class Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, payload):
        self.emitted.append(payload)


class Lasers:
    def __init__(self):
        self.enabled = {"488": True, "635": True}

    def getAllDeviceNames(self):
        return list(self.enabled)

    def __getitem__(self, name):
        return SimpleNamespace(setEnabled=lambda on: self.enabled.__setitem__(name, on))


def make_controller(controllers=None, positioners=None, scan=None):
    c = object.__new__(UC2ConfigController)
    c._logger = c._UC2ConfigController__logger = logging.getLogger("test")
    can = SimpleNamespace(scan=lambda timeout=5, probe_range=False: scan or {},
                          register_callback=lambda i, cb: None)
    esp32 = SimpleNamespace(can=can, serial=SimpleNamespace(baudrate=921600))
    manager = SimpleNamespace(ESP32=esp32, isConnected=lambda: True, serialport="/dev/ttyUSB0")
    registry = controllers or {}
    positioner_map = positioners or {}
    c.lasers = Lasers()
    c._master = SimpleNamespace(
        UC2ConfigManager=manager, getController=registry.get, lasersManager=c.lasers,
        positionersManager=type("PM", (), {
            "getAllDeviceNames": lambda self: list(positioner_map),
            "__getitem__": lambda self, k: positioner_map[k]})())
    c._setupInfo = SimpleNamespace(uc2Config=UC2ConfigInfo(firmwareServerUrl="http://fw/firmware"))
    c._commChannel = SimpleNamespace(sigUpdateCANDevices=Signal())
    c.sigOTAStatusUpdate, c.sigUSBFlashStatusUpdate = Signal(), Signal()
    c.sigFirmwareUpdatesAvailable = Signal()
    c.getFirmwareInfo = lambda: {"connected": True, "pindef": "UC2_canopen_master",
                                 "isMaster": True}
    c._init_can_network()
    return c


def test_routes_delegate_and_keep_their_shapes():
    c = make_controller(scan={"master": {"canId": 1}, "scan": [
        {"canId": 12, "deviceTypeStr": "motor", "statusStr": "idle"}]})
    assert c.getOTAFirmwareServer() == {"firmware_server_url": "http://fw/firmware"}
    assert c.get_canbus_devices() == [12]
    assert c.getOTADeviceMapping()["mapping"]["motors"]["Y"] == 12
    assert c.getUSBFlashStatus()["status"] == "idle"
    assert c.getFirmwareUpdateStatus()["state"] == "idle"
    assert c.startMultipleCANStreamingOTA("12")["status"] == "error"


def test_ota_status_accepts_the_can_id_as_query_string():
    c = make_controller()
    c._can_network.ota.status[12] = {"canId": 12, "status": "success"}
    assert c.getOTAStatus(can_id="12")["ota_status"]["status"] == "success"
    assert c.clearOTAStatus(can_id="12")["status"] == "success"
    assert c.getOTAStatus()["device_count"] == 0


def test_usb_flash_is_refused_during_a_can_upload():
    c = make_controller()
    c._can_network.guard.can.acquire()
    assert c.flashMasterFirmwareUSB()["status"] == "busy"
    c._can_network.guard.can.release()
    c._can_network.guard.usb.acquire()
    assert c.startCANStreamingOTA(12)["status"] == "busy"


def test_busy_hardware_refuses_the_update():
    experiment = SimpleNamespace(getExperimentStatus=lambda: {"status": "running"})
    motor = SimpleNamespace(isBusy=lambda steps, timeout=1: True)
    c = make_controller(controllers={"Experiment": experiment},
                        positioners={"ESP32Stage": SimpleNamespace(_motor=motor)})
    result = c.startFirmwareUpdate()
    assert result["status"] == "refused"
    assert {"An experiment is running.", "A stage is moving or homing."} <= set(result["reasons"])


def test_a_broken_busy_check_does_not_block():
    def broken():
        raise RuntimeError("no workflow manager")
    c = make_controller(controllers={"Recording": SimpleNamespace(isRecording=broken)})
    assert c._hardware_busy_reasons() == []


def test_lasers_off_and_homing_rearmed():
    toggled = []
    positioner = SimpleNamespace(_hasHomedSinceStartup=True, _homingRecommendationDismissed=True)
    laser_controller = SimpleNamespace(setLaserActive=lambda n, on: toggled.append((n, on)))
    c = make_controller(controllers={"Laser": laser_controller, "Positioner": positioner})
    c._lasers_off()
    c._rearm_homing()
    assert toggled == [("488", False), ("635", False)]
    assert not positioner._hasHomedSinceStartup and not positioner._homingRecommendationDismissed
    c._master.getController = {}.get  # no LaserController: straight to the manager
    c._lasers_off()
    assert c.lasers.enabled == {"488": False, "635": False}


def test_startup_check_offers_the_update():
    c = make_controller()
    c.checkFirmwareUpdates = lambda: {"updates_available": 2, "server_version": "v1", "devices": []}
    c._startup_firmware_check()
    assert c.sigFirmwareUpdatesAvailable.emitted[0]["updates_available"] == 2
    assert c.getFirmwareUpdatePrompt()["updates_available"] == 2


def test_startup_check_stays_quiet_while_busy():
    c = make_controller()
    c._can_network.guard.can.acquire()
    c.checkFirmwareUpdates = lambda: {"updates_available": 2}
    c._startup_firmware_check()
    assert c.sigFirmwareUpdatesAvailable.emitted == []
    assert c.getFirmwareUpdatePrompt() == {"updates_available": 0}


def test_startup_timer_only_when_enabled(monkeypatch):
    started = []
    monkeypatch.setattr(threading.Timer, "start", lambda self: started.append(self))
    make_controller()
    assert started == []
    c = object.__new__(UC2ConfigController)
    c._setupInfo = SimpleNamespace(uc2Config=UC2ConfigInfo(checkFirmwareOnConnect=True))
    assert c._check_firmware_on_connect()


def test_check_on_connect_setting_is_saved_from_a_null_uc2config(monkeypatch):
    import imswitch.imcontrol.model.configfiletools as configfiletools
    saved = []
    monkeypatch.setattr(configfiletools, "loadOptions", lambda: ("options", False))
    monkeypatch.setattr(configfiletools, "saveSetupInfo",
                        lambda options, setup: saved.append(setup.uc2Config.checkFirmwareOnConnect))
    c = make_controller()
    c._setupInfo.uc2Config = None  # setup JSONs ship "uc2Config": null
    assert c.setFirmwareCheckOnConnect(True) == {"enabled": True}
    assert saved == [True] and isinstance(c._setupInfo.uc2Config, UC2ConfigInfo)
    assert c.getFirmwareCheckOnConnect() == {"enabled": True}


def test_check_on_connect_save_failure_is_reported(monkeypatch):
    import imswitch.imcontrol.model.configfiletools as configfiletools

    def fail(*args):
        raise OSError("read-only file system")
    monkeypatch.setattr(configfiletools, "loadOptions", fail)
    result = make_controller().setFirmwareCheckOnConnect(True)
    assert result["status"] == "error" and "read-only" in result["message"]


def test_recommended_firmware_for_the_connected_board_and_another_port(monkeypatch):
    c = make_controller()
    usb, server = c._can_network.usb, c._can_network.server
    monkeypatch.setattr(usb, "detect_chip", lambda port: "esp32")
    seen = []
    monkeypatch.setattr(server, "recommend", lambda identity: seen.append(identity) or {
        "status": "success", "recommended": {"filename": "x.bin"}})
    monkeypatch.setattr(usb, "identify", lambda port, **kw: pytest.fail("own port re-opened"))

    own = c.getRecommendedFirmware()            # ImSwitch's board: read over the open link
    assert own["source"] == "imswitch" and own["port"] == "/dev/ttyUSB0"
    assert own["recommended"]["filename"] == "x.bin"
    assert seen[-1]["canId"] == 1               # a master without a reported id is node 1
    assert c.getRecommendedFirmware(port="/dev/ttyUSB0")["source"] == "imswitch"

    monkeypatch.setattr(usb, "identify", lambda port, **kw: {
        "status": "success", "port": port, "chip": "esp32s3",
        "identity": {"pindef": "UC2_canopen_slave_motor", "canId": 12}})
    other = c.getRecommendedFirmware(port="/dev/ttyACM0")
    assert other["source"] == "port" and other["chip"] == "esp32s3"
    assert seen[-1]["canId"] == 12

    monkeypatch.setattr(usb, "identify", lambda port, **kw: {"status": "busy", "message": "m"})
    assert c.getRecommendedFirmware(port="/dev/ttyACM0")["status"] == "busy"
    c.getFirmwareInfo = lambda: {"connected": False}
    assert c.getRecommendedFirmware()["status"] == "error"
