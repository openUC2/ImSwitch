"""Camera bit depth (MTG-3) and the placeholder test (CAM-LOST) on the detector base class."""
import types

from imswitch.imcontrol.model.managers.detectors.DetectorManager import DetectorManager


def _detector(maxValue=None, model="HIK"):
    params = {} if maxValue is None else {"previewMaxValue": types.SimpleNamespace(value=maxValue)}
    return types.SimpleNamespace(parameters=params, _isMock=False, _camera=types.SimpleNamespace(model=model))


def test_bit_depth_from_preview_maximum():
    bits = DetectorManager.bitDepth.fget
    assert [bits(_detector(v)) for v in (255, 1023, 4095, 16383, 65535)] == [8, 10, 12, 14, 16]
    assert bits(_detector()) is None and bits(_detector(0)) is None


def test_placeholder_is_the_mock_camera():
    placeholder = DetectorManager.isPlaceholder.fget
    assert placeholder(_detector(model="mock"))   # real camera did not open
    assert not placeholder(_detector(model="MV-CE060-10UM"))
    assert not placeholder(types.SimpleNamespace(_isMock=False))   # managers without a _camera
