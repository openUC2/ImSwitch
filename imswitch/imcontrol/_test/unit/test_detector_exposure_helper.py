"""
getDetectorExposureMs/Seconds: one exposure lookup shared by the snap and
live-view long-exposure logic.

Drivers name the exposure parameter differently. MMCoreDetectorManager uses
"Exposure"; the old exact-key lookup missed it, so a 10-minute Andor exposure
counted as 0 s and the snap timed out after a few seconds.
"""

from types import SimpleNamespace

import pytest

from imswitch.imcontrol.model.managers.detectors.DetectorManager import (
    DetectorListParameter,
    DetectorNumberParameter,
    getDetectorExposureMs,
    getDetectorExposureSeconds,
)


def _number(value, units="ms"):
    return DetectorNumberParameter(group="Misc", value=value, valueUnits=units, editable=True)


def _detector(**parameters):
    return SimpleNamespace(parameters=parameters)


@pytest.mark.parametrize("name", [
    "exposure", "Exposure", "EXPOSURE", "exposure_time", "exposureTime",
    "ExposureTime", "Exposure Time", "exposure-time",
])
def test_exposure_found_under_any_common_name(name):
    det = _detector(**{name: _number(600000.0)})
    assert getDetectorExposureMs(det) == pytest.approx(600000.0)
    assert getDetectorExposureSeconds(det) == pytest.approx(600.0)


def test_mmcore_style_exposure_parameter():
    det = _detector(
        Exposure=DetectorNumberParameter(group="Acquisition", value=600000.0,
                                         editable=True, valueUnits="ms"),
        Gain=_number(1.0, units="arb.u."),
    )
    assert getDetectorExposureSeconds(det) == pytest.approx(600.0)


def test_exposure_mode_is_not_mistaken_for_the_exposure():
    det = _detector(
        exposure_mode=DetectorListParameter(group="Misc", value="manual",
                                            editable=True, options=["auto", "manual"]),
    )
    assert getDetectorExposureMs(det) == 0.0

    det.parameters["exposure"] = _number(20.0)
    assert getDetectorExposureMs(det) == pytest.approx(20.0)


@pytest.mark.parametrize("units, expectedMs", [
    ("ms", 2.0), ("s", 2000.0), ("us", 0.002), ("µs", 0.002), ("arb.u.", 2.0), ("", 2.0),
])
def test_value_units_are_converted_to_ms(units, expectedMs):
    det = _detector(exposure=_number(2.0, units=units))
    assert getDetectorExposureMs(det) == pytest.approx(expectedMs)


def test_hamamatsu_style_set_exposure_time_in_seconds():
    det = _detector(**{
        "Set exposure time": _number(0.5, units="s"),
        "Real exposure time": _number(0.49, units="s"),
    })
    assert getDetectorExposureSeconds(det) == pytest.approx(0.5)


def test_plain_exposure_name_wins_over_other_spellings():
    det = _detector(exposure_time=_number(100.0), exposure=_number(50.0))
    assert getDetectorExposureMs(det) == pytest.approx(50.0)


def test_unusable_values_fall_through_to_the_next_candidate():
    det = _detector(exposure=_number(0.0), ExposureTime=_number(30.0))
    assert getDetectorExposureMs(det) == pytest.approx(30.0)

    det = _detector(exposure=SimpleNamespace(value="n/a"))
    assert getDetectorExposureMs(det) == 0.0


def test_accepts_a_parameter_dict_and_plain_values():
    assert getDetectorExposureMs({"Exposure": 12.5}) == pytest.approx(12.5)
    assert getDetectorExposureMs({"Exposure": SimpleNamespace(value=7)}) == pytest.approx(7.0)


def test_missing_or_broken_parameters_give_zero():
    assert getDetectorExposureMs(_detector()) == 0.0
    assert getDetectorExposureMs(object()) == 0.0

    class Broken:
        @property
        def parameters(self):
            raise RuntimeError("camera gone")

    assert getDetectorExposureMs(Broken()) == 0.0


def test_snap_duration_estimate_uses_mmcore_exposure_name():
    """RecordingController sizes snap waits/estimates from the exposure."""
    from imswitch.imcontrol.controller.controllers.RecordingController import (
        RecordingController,
    )

    ctrl = object.__new__(RecordingController)
    ctrl._master = SimpleNamespace(detectorsManager={"cam": _detector(Exposure=_number(600000.0))})
    assert ctrl._estimateSnapDurationMs(["cam"]) == pytest.approx(600000.0 + 1500.0)


def test_live_view_long_exposure_check_sees_mmcore_exposure():
    from imswitch.imcontrol.controller.controllers.LiveViewController import (
        LiveViewController,
    )

    ctrl = object.__new__(LiveViewController)
    ctrl._master = SimpleNamespace(detectorsManager={"cam": _detector(Exposure=_number(600000.0))})
    ctrl._logger = SimpleNamespace(warning=lambda *a, **k: None)

    info = ctrl.getLongExposureInfo("cam")
    assert info["exposureMs"] == pytest.approx(600000.0)
    assert info["isLongExposure"] is True
