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
