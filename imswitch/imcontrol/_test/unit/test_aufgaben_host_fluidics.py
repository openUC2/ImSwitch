"""The Aufgaben host drives the setup's pumps and valves; a forced abort stops pumps."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import imswitch
from imswitch.imcontrol.controller.controllers.aufgaben.host import ImSwitchMicroscope
from imswitch.imcontrol.model import aufgaben
from imswitch.imcontrol.model.SetupInfo import SetupInfo
from imswitch.imcontrol.model.managers.PumpsManager import PumpsManager
from imswitch.imcontrol.model.managers.ValvesManager import ValvesManager

SETUP = Path(imswitch.__file__).parent / "_data/user_defaults/imcontrol_setups/" \
    "example_virtual_microscope.json"


@pytest.fixture
def host():
    setup = SetupInfo.from_dict(json.loads(SETUP.read_text()))
    master = SimpleNamespace(pumpsManager=PumpsManager(setup.pumps),
                             valvesManager=ValvesManager(setup.valves),
                             getController=lambda name: None)
    return ImSwitchMicroscope(master, logger=None), master


def test_virtual_setup_offers_its_pump_and_valve(host):
    hw, master = host
    assert {"pump", "valve"} <= hw.capabilities()
    assert hw.devices("pump") == ["peristaltic"] and hw.devices("valve") == ["inlet"]
    hw.set_valve("inlet", "sample")
    hw.pump_volume("peristaltic", 50, 1000)         # 3 s of pumping at speedup 60
    assert master.pumpsManager["peristaltic"].pumpedUl == pytest.approx(50)
    assert master.valvesManager["inlet"].position == "sample"
    with pytest.raises(ValueError, match="no pump 'syringe'"):
        hw.pump_volume("syringe", 1, 10)
    with pytest.raises(ValueError, match="no position"):
        hw.set_valve("inlet", "air")
    hw.pump_run("peristaltic", 100)
    assert master.pumpsManager["peristaltic"].running
    hw.stop_pumps()
    assert not master.pumpsManager["peristaltic"].running


def test_stop_flow_example_passes_the_catalogue_checks(host):
    hw, _ = host
    caps = hw.capabilities() | {"illumination", "camera", "stage", "temperature"}
    job = aufgaben.Job.model_validate_json(
        (aufgaben.EXAMPLES_DIR / "stop_flow_fluidics.json").read_text())
    report = aufgaben.validate(job, capabilities=caps)
    assert report.ok, report.as_dict()


def test_forced_abort_stops_the_pumps():
    from imswitch.imcontrol.model.aufgaben.simulated import SimulatedMicroscope
    from imswitch.imcontrol.model.aufgaben.engine import Run
    sim = SimulatedMicroscope()
    job = aufgaben.Job.model_validate({"name": "j", "tasks": [
        {"id": "w", "task": "wait", "params": {"seconds": 30}}]})
    run = Run(job, sim)
    run.start()
    run.abort(force=True)
    run.join(5)
    assert run.state == "aborted"
    assert ("stop_pumps",) in sim.calls and ("stop_stage",) in sim.calls


def test_serial_commands_ask_first_and_reach_plain_serial_devices():
    from imswitch.imcontrol.model.SetupInfo import RS232Info
    from imswitch.imcontrol.model.managers.RS232sManager import RS232sManager
    from imswitch.imcontrol.model.aufgaben.simulated import SimulatedMicroscope
    from imswitch.imcontrol.model.aufgaben.engine import Run

    spec = aufgaben.REGISTRY["send_serial"]
    assert spec.flags.ask_first and spec.flags.external

    rs232 = RS232sManager({
        "syringe": RS232Info(managerName="RS232Manager",
                             managerProperties={"port": "NO_SUCH_PORT", "encoding": "ascii",
                                                "recv_termination": "\r", "send_termination": "\r",
                                                "baudrate": 9600, "bytesize": 8, "parity": "none",
                                                "stopbits": 1, "rtscts": False, "dsrdtr": False,
                                                "xonxoff": False}),
        "ESP32": RS232Info(managerName="VirtualESP32Manager", managerProperties={}),
    })
    hw = ImSwitchMicroscope(SimpleNamespace(rs232sManager=rs232, getController=lambda n: None),
                            logger=None)
    assert hw.devices("serial") == ["syringe"] and "serial" in hw.capabilities()
    hw.send_serial("syringe", "VER", read_reply=False)       # the mock port accepts it
    with pytest.raises(ValueError, match="no serial device 'ESP32'"):
        hw.send_serial("ESP32", "VER")
    rs232.finalize()

    job = aufgaben.Job.model_validate({"name": "j", "tasks": [
        {"id": "ask", "task": "send_serial", "params": {"device": "syringe", "command": "VER"}}]})
    sim = SimulatedMicroscope()
    with pytest.raises(PermissionError):
        Run(job, sim).run()
    run = Run(job, sim)
    assert run.run(confirmed=True) == "finished"
    assert ("serial", "syringe", "VER") in sim.calls and run.results["ask"] == "OK"


def test_laser_channels_that_drive_pumps_or_valves_are_not_light_sources():
    class Devices(dict):
        def getAllDeviceNames(self):
            return list(self)

    laser = SimpleNamespace(valueRangeMax=1023, setValue=lambda v: None,
                            setEnabled=lambda e: None)
    lasers = Devices(LED=laser, PumpPWM=laser, ValvePWM=laser)
    setup = SetupInfo.from_dict({
        "pumps": {"p": {"managerName": "LaserPumpManager",
                        "managerProperties": {"laser": "PumpPWM"}}},
        "valves": {"v": {"managerName": "LaserValveManager",
                         "managerProperties": {"laser": "ValvePWM",
                                               "positions": {"closed": 0, "open": 1}}}}})
    master = SimpleNamespace(lasersManager=lasers, getController=lambda name: None,
                             pumpsManager=PumpsManager(setup.pumps, lasersManager=lasers),
                             valvesManager=ValvesManager(setup.valves, lasersManager=lasers))
    assert ImSwitchMicroscope(master, logger=None).devices("illumination") == ["LED"]
