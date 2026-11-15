"""
SettingsController.setDetectorROI in headless mode.

The ROI is validated against the detector's full frame, widened to the
4-pixel grid sensors want (without leaving the sensor), applied with
DetectorManager.crop(), and the new shape is broadcast so the live view
follows. ``fullFrame`` (or no ROI at all) restores the full sensor.
"""

from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from imswitch.imcontrol.controller.controllers.SettingsController import (
    SettingsController,
    computeCropWindow,
)


# ---------------------------------------------------------------------------
# computeCropWindow
# ---------------------------------------------------------------------------

class TestComputeCropWindow:
    def test_aligned_roi_is_passed_through(self):
        assert computeCropWindow((2048, 2048), (100, 200), (512, 256)) == (100, 200, 512, 256)

    def test_unaligned_roi_is_widened_to_contain_the_request(self):
        hpos, vpos, hsize, vsize = computeCropWindow((2048, 2048), (101, 203), (510, 250))
        assert (hpos, vpos) == (100, 200)
        assert hsize % 4 == 0 and vsize % 4 == 0
        assert hpos + hsize >= 101 + 510 and vpos + vsize >= 203 + 250

    def test_full_frame_request_is_exact_even_if_not_a_multiple_of_4(self):
        assert computeCropWindow((1922, 1081), (0, 0), (1922, 1081)) == (0, 0, 1922, 1081)

    def test_widening_never_leaves_the_sensor(self):
        hpos, vpos, hsize, vsize = computeCropWindow((1922, 1081), (1801, 1001), (121, 80))
        assert hpos + hsize == 1922
        assert vpos + vsize == 1081

    @pytest.mark.parametrize("frameStart, shape", [
        ((0, 0), (0, 10)),        # empty
        ((0, 0), (10, -1)),       # negative size
        ((-4, 0), (10, 10)),      # negative start
        ((2000, 0), (100, 10)),   # runs off the right edge
        ((0, 0), (2048, 2049)),   # taller than the sensor
    ])
    def test_out_of_range_roi_is_rejected(self, frameStart, shape):
        with pytest.raises(ValueError):
            computeCropWindow((2048, 2048), frameStart, shape)

    @pytest.mark.parametrize("frameStart, shape", [
        ((0,), (10, 10)),
        (None, (10, 10)),
        ((0, 0), ("a", 10)),
    ])
    def test_malformed_roi_is_rejected(self, frameStart, shape):
        with pytest.raises(ValueError):
            computeCropWindow((2048, 2048), frameStart, shape)


# ---------------------------------------------------------------------------
# setDetectorROI
# ---------------------------------------------------------------------------

class FakeDetector:
    def __init__(self, fullShape=(2048, 1536), croppable=True):
        self.fullShape = fullShape
        self.shape = fullShape
        self.frameStart = (0, 0)
        self.croppable = croppable
        self.crops = []

    def crop(self, hpos, vpos, hsize, vsize):
        self.crops.append((hpos, vpos, hsize, vsize))
        self.frameStart = (hpos, vpos)
        self.shape = (hsize, vsize)


class FakeDetectorsManager(dict):
    def __init__(self, current, **detectors):
        super().__init__(**detectors)
        self._current = current

    def getCurrentDetectorName(self):
        return self._current


def _make_controller(**detectors):
    current = next(iter(detectors))
    ctrl = object.__new__(SettingsController)
    ctrl._master = SimpleNamespace(detectorsManager=FakeDetectorsManager(current, **detectors))
    ctrl.events = []
    ctrl._commChannel = SimpleNamespace(sigAdjustFrame=SimpleNamespace(
        emit=lambda shape: ctrl.events.append(("sigAdjustFrame", tuple(shape)))))
    ctrl.updateSharedAttrs = lambda: ctrl.events.append(("sharedAttrs",))
    ctrl._emit_detector_parameters_if_changed = (
        lambda force=False: ctrl.events.append(("parameters", force)))
    ctrl._logger = SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None)
    return ctrl


class TestSetDetectorROI:
    def test_crops_and_broadcasts_the_new_shape(self):
        det = FakeDetector()
        ctrl = _make_controller(cam=det)

        result = ctrl.setDetectorROI("cam", (100, 200), (512, 256))

        assert det.crops == [(100, 200, 512, 256)]
        assert result == {
            "status": "ok", "detectorName": "cam",
            "frameStart": [100, 200], "shape": [512, 256], "fullShape": [2048, 1536],
        }
        assert ctrl.events == [
            ("sigAdjustFrame", (512, 256)), ("sharedAttrs",), ("parameters", True),
        ]

    def test_defaults_to_the_current_detector(self):
        det = FakeDetector()
        ctrl = _make_controller(cam=det)
        assert ctrl.setDetectorROI(frameStart=(0, 0), shape=(64, 64))["status"] == "ok"
        assert det.crops == [(0, 0, 64, 64)]

    def test_no_binning_rescale(self):
        """Coordinates are in fullShape pixels, which already reflect binning."""
        det = FakeDetector(fullShape=(1024, 768))
        det.binning = 2
        ctrl = _make_controller(cam=det)
        ctrl.setDetectorROI("cam", (0, 0), (1024, 768))
        assert det.crops == [(0, 0, 1024, 768)]

    @pytest.mark.parametrize("kwargs", [
        {"fullFrame": True},
        {"frameStart": (8, 8), "shape": (16, 16), "fullFrame": True},
        {},
    ])
    def test_full_frame_reset(self, kwargs):
        det = FakeDetector(fullShape=(1922, 1081))
        det.crop(4, 4, 100, 100)
        ctrl = _make_controller(cam=det)

        result = ctrl.setDetectorROI("cam", **kwargs)

        assert det.crops[-1] == (0, 0, 1922, 1081)
        assert result["status"] == "ok"
        assert result["shape"] == [1922, 1081]
        assert ("sigAdjustFrame", (1922, 1081)) in ctrl.events

    def test_out_of_range_roi_is_rejected_without_cropping(self):
        det = FakeDetector()
        ctrl = _make_controller(cam=det)

        result = ctrl.setDetectorROI("cam", (2000, 0), (512, 512))

        assert result["status"] == "error"
        assert "full frame" in result["error"]
        assert det.crops == []
        assert ctrl.events == []

    def test_half_specified_roi_is_rejected(self):
        det = FakeDetector()
        ctrl = _make_controller(cam=det)
        result = ctrl.setDetectorROI("cam", frameStart=(0, 0))
        assert result["status"] == "error"
        assert det.crops == []

    def test_not_croppable_detector_is_rejected(self):
        det = FakeDetector(croppable=False)
        ctrl = _make_controller(cam=det)
        assert ctrl.setDetectorROI("cam", (0, 0), (64, 64))["status"] == "error"
        assert det.crops == []

    def test_unknown_detector_is_reported(self):
        ctrl = _make_controller(cam=FakeDetector())
        assert ctrl.setDetectorROI("nope", (0, 0), (64, 64))["status"] == "error"

    def test_other_detector_does_not_resize_the_current_view(self):
        current, other = FakeDetector(), FakeDetector()
        ctrl = _make_controller(cam=current, side=other)
        ctrl.setDetectorROI("side", (0, 0), (64, 64))
        assert other.crops == [(0, 0, 64, 64)]
        assert not any(e[0] == "sigAdjustFrame" for e in ctrl.events)

    def test_reachable_over_http_with_query_parameters(self):
        """Tuple parameters used to become a JSON body, unusable from a GET."""
        from imswitch.imcontrol.controller.server.ImSwitchServer import includeAPIRoute

        det = FakeDetector()
        ctrl = _make_controller(cam=det)
        router = APIRouter(prefix="/api")
        includeAPIRoute(router, "/SettingsController/setDetectorROI", ctrl.setDetectorROI)
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        spec = app.openapi()["paths"]["/api/SettingsController/setDetectorROI"]["get"]
        assert "requestBody" not in spec

        response = client.get("/api/SettingsController/setDetectorROI", params={
            "detectorName": "cam", "frameStart": [100, 200], "shape": [512, 256]})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert det.crops == [(100, 200, 512, 256)]

        response = client.get("/api/SettingsController/setDetectorROI",
                              params={"fullFrame": "true"})
        assert response.json()["shape"] == [2048, 1536]
