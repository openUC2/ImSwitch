"""getDetectorParameters reports bitDepth from the camera's preview maximum (MTG-3)."""
import types

from imswitch.imcontrol.controller.controllers.SettingsController import bitDepthOf


def _detector(maxValue):
    return types.SimpleNamespace(parameters={"previewMaxValue": types.SimpleNamespace(value=maxValue)})


def test_bit_depth_from_preview_maximum():
    assert [bitDepthOf(_detector(v)) for v in (255, 1023, 4095, 16383, 65535)] == [8, 10, 12, 14, 16]


def test_unknown_without_a_maximum():
    assert bitDepthOf(types.SimpleNamespace(parameters={})) is None
    assert bitDepthOf(_detector(0)) is None
