"""The shared busy check: one place that knows what drives the instrument."""
import types

from imswitch.imcontrol.controller.busy import busy_reasons


def _master(**controllers):
    return types.SimpleNamespace(getController=lambda name: controllers.get(name))


def _flag(method, value):
    return types.SimpleNamespace(**{method: lambda: value})


def test_idle_instrument_has_no_reasons():
    master = _master(StageMap=_flag("isStageMapBusy", False),
                     FocusLock=_flag("isCalibrationRunning", False),
                     Aufgaben=_flag("isAufgabenRunning", False))
    assert busy_reasons(master) == []


def test_stage_map_and_focus_calibration_count_as_busy():
    master = _master(StageMap=_flag("isStageMapBusy", True),
                     FocusLock=_flag("isCalibrationRunning", True),
                     Experiment=types.SimpleNamespace(
                         getExperimentStatus=lambda: {"status": "idle"}))
    assert busy_reasons(master) == ["the stage map or its prescan is running",
                                    "a focus-lock calibration is running"]
    assert busy_reasons(master, ignore=("StageMap",)) == ["a focus-lock calibration is running"]


def test_a_broken_check_is_logged_not_blocking():
    warnings = []
    logger = types.SimpleNamespace(warning=warnings.append)
    master = _master(StageMap=types.SimpleNamespace(isStageMapBusy=lambda: 1 / 0))
    assert busy_reasons(master, logger) == [] and len(warnings) == 1
