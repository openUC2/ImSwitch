"""
Camera <-> galvo calibration in GalvoScannerController.

The affine transform from the 3-point wizard maps camera pixels to DAC
counts; together with the camera's pixel size it is the only bridge from
0..4095 scanner units to micrometres. These tests pin down that bridge
(``getGalvoCameraCalibration``), the background snapshot the raster tab draws
the scan region on (``snapGalvoCameraBackground``) and the hand-over of the
derived µm/DAC value to a bound FLIM detector.

Everything is driven with fakes: no ESP32, no camera, no FLIM server.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest

from imswitch.imcontrol.controller.controllers.GalvoScannerController import (
    GalvoScannerController,
)
from imswitch.imcontrol.model.managers.galvoscanners.GalvoScannerManager import (
    AffineTransform,
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeCamera:
    """Widefield camera: 640x480, 0.5 µm pixels, one frame ready."""

    def __init__(self, frame=None, pixelSizeUm=(1, 0.5, 0.5)):
        self.shape = (640, 480)
        self.pixelSizeUm = list(pixelSizeUm)
        self._frame = frame
        self._running = False
        self.startCalls = 0

    def getLatestFrame(self):
        return self._frame

    def startAcquisition(self):
        self.startCalls += 1
        self._running = True
        # First frame appears once the stream runs
        self._frame = np.full((480, 640), 1000, dtype=np.uint16)


class FakeFlimDetector:
    """Only what the controller touches: the galvo binding + setUmPerDac."""

    def __init__(self, galvoScannerName=None):
        self.galvoScannerName = galvoScannerName
        self.calls = []
        self.umPerDacInfo = {'umPerDacX': None, 'umPerDacY': None, 'source': None}

    def setGalvoScanner(self, scanner):
        self.scanner = scanner

    def setUmPerDac(self, umX=None, umY=None, source='affine'):
        self.calls.append((umX, umY, source))
        self.umPerDacInfo = {'umPerDacX': umX, 'umPerDacY': umY, 'source': source}
        return self.umPerDacInfo


class FakeDetectorsManager:
    def __init__(self, detectors, current=None):
        self._detectors = detectors
        self._current = current or next(iter(detectors))

    def getAllDeviceNames(self):
        return list(self._detectors)

    def getCurrentDetectorName(self):
        return self._current

    def __getitem__(self, name):
        return self._detectors[name]


class FakeScanner:
    def __init__(self, affine=None):
        self._affine = affine or AffineTransform()
        self.config = SimpleNamespace(nx=256, ny=128, x_min=500, x_max=3500,
                                      y_min=1000, y_max=2000)

    def get_affine_transform_dict(self):
        return self._affine.to_dict()


class FakeGalvoManager:
    def __init__(self, scanners):
        self._scanners = scanners

    def getAllDeviceNames(self):
        return list(self._scanners)

    def __getitem__(self, name):
        return self._scanners[name]


def make_controller(detectors, scanners, current=None):
    """Build a controller around fakes without running the Qt/ImSwitch __init__."""
    ctrl = GalvoScannerController.__new__(GalvoScannerController)
    ctrl._master = SimpleNamespace(
        detectorsManager=FakeDetectorsManager(detectors, current),
        galvoScannersManager=FakeGalvoManager(scanners),
    )
    ctrl._GalvoScannerController__logger = logging.getLogger('test-galvo')
    return ctrl


def rotated_affine(theta_deg, px_per_dac, centre=(320.0, 240.0)):
    """camera = R(theta) * px_per_dac * dac + centre, returned as camera->DAC."""
    theta = math.radians(theta_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    A = np.linalg.inv(R) / px_per_dac
    b = -A @ np.asarray(centre, dtype=float)
    return AffineTransform(a11=A[0, 0], a12=A[0, 1], tx=b[0],
                           a21=A[1, 0], a22=A[1, 1], ty=b[1])


# --------------------------------------------------------------------------
# µm per DAC count
# --------------------------------------------------------------------------

def test_um_per_dac_from_rotated_affine_uses_camera_pixel_size():
    # One DAC count moves the spot 0.25 camera px on a 30° rotated axis;
    # camera pixels are 0.5 µm -> 0.125 µm per count on both axes.
    aff = rotated_affine(30, 0.25).to_dict()
    umX, umY = GalvoScannerController._umPerDacFromAffine(aff, 0.5, 0.5)
    assert umX == pytest.approx(0.125)
    assert umY == pytest.approx(0.125)


def test_um_per_dac_singular_affine_is_none():
    aff = {'a11': 1, 'a12': 2, 'a21': 2, 'a22': 4, 'tx': 0, 'ty': 0}
    assert GalvoScannerController._umPerDacFromAffine(aff, 1, 1) == (None, None)


def test_identity_detection():
    assert GalvoScannerController._isIdentityAffine(AffineTransform().to_dict())
    assert not GalvoScannerController._isIdentityAffine(rotated_affine(0, 0.25).to_dict())


# --------------------------------------------------------------------------
# getGalvoCameraCalibration
# --------------------------------------------------------------------------

def test_calibration_summary_uncalibrated_has_no_physical_values():
    ctrl = make_controller({'cam': FakeCamera()}, {'galvo': FakeScanner()})
    info = ctrl.getGalvoCameraCalibration()
    assert info['calibrated'] is False
    assert info['detectorName'] == 'cam'
    assert info['frameWidth'] == 640 and info['frameHeight'] == 480
    assert info['pixelSizeUmX'] == 0.5
    assert info['umPerDacX'] is None and info['scan'] is None
    assert 'wizard' in info['hint']


def test_calibration_summary_reports_scan_fov_and_pixel_size():
    ctrl = make_controller({'cam': FakeCamera()},
                           {'galvo': FakeScanner(rotated_affine(30, 0.25))})
    info = ctrl.getGalvoCameraCalibration()
    assert info['calibrated'] is True
    assert info['umPerDacX'] == pytest.approx(0.125)
    scan = info['scan']
    # x: 3000 counts * 0.125 = 375 µm over 256 px; y: 1000 counts -> 125 µm / 128 px
    assert scan['fovUmX'] == pytest.approx(375.0)
    assert scan['fovUmY'] == pytest.approx(125.0)
    assert scan['pixelUmX'] == pytest.approx(375.0 / 256)
    assert scan['pixelUmY'] == pytest.approx(125.0 / 128)
    assert scan['fullScaleUmX'] == pytest.approx(0.125 * 4096)
    assert info['rotationDeg'] == pytest.approx(30.0)


def test_flim_detector_is_not_offered_as_a_camera():
    flim = FakeFlimDetector('galvo')
    ctrl = make_controller({'flim': flim, 'cam': FakeCamera()},
                           {'galvo': FakeScanner()}, current='flim')
    info = ctrl.getGalvoCameraCalibration()
    assert info['cameraDetectors'] == ['cam']
    assert info['detectorName'] == 'cam'


def test_calibration_summary_without_any_camera():
    ctrl = make_controller({'flim': FakeFlimDetector()}, {'galvo': FakeScanner()})
    info = ctrl.getGalvoCameraCalibration()
    assert info['detectorName'] is None
    assert info['cameraDetectors'] == []
    assert 'No 2D camera' in info['hint']


# --------------------------------------------------------------------------
# Propagation into the FLIM detector
# --------------------------------------------------------------------------

def test_calibrated_affine_is_pushed_into_bound_flim_detector():
    flim = FakeFlimDetector('galvo')
    other = FakeFlimDetector('otherScanner')
    ctrl = make_controller({'cam': FakeCamera(), 'flim': flim, 'other': other},
                           {'galvo': FakeScanner(rotated_affine(0, 0.25))})
    res = ctrl._propagateUmPerDacToFlim('galvo')
    assert flim.calls == [(pytest.approx(0.125), pytest.approx(0.125), 'affine')]
    assert other.calls == []          # bound to a different scanner
    assert res['flim']['source'] == 'affine'


def test_identity_affine_reverts_flim_detector_to_setup_values():
    flim = FakeFlimDetector(None)     # unbound name -> follows any scanner
    ctrl = make_controller({'cam': FakeCamera(), 'flim': flim},
                           {'galvo': FakeScanner()})
    ctrl._propagateUmPerDacToFlim('galvo')
    assert flim.calls == [(None, None, 'affine')]


# --------------------------------------------------------------------------
# snapGalvoCameraBackground
# --------------------------------------------------------------------------

def _decode_png(dataurl):
    from PIL import Image
    assert dataurl.startswith('data:image/png;base64,')
    raw = base64.b64decode(dataurl.split(',', 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)))


def test_background_is_stretched_and_downsampled_with_frame_geometry():
    # 16-bit gradient, bigger than maxDim on the long edge
    frame = np.tile(np.linspace(100, 60000, 2000, dtype=np.uint16), (1200, 1))
    ctrl = make_controller({'cam': FakeCamera(frame)}, {'galvo': FakeScanner()})
    res = ctrl.snapGalvoCameraBackground(maxDim=500)
    assert 'error' not in res
    assert res['frameWidth'] == 2000 and res['frameHeight'] == 1200
    assert res['subsampling'] == 4
    assert res['width'] == 500 and res['height'] == 300
    assert res['pixelSizeUmX'] == 0.5
    img = _decode_png(res['image'])
    assert img.shape == (300, 500) and img.dtype == np.uint8
    # Percentile stretch: dark left edge, bright right edge, monotone ramp
    assert img[0, 0] == 0 and img[0, -1] == 255
    assert np.all(np.diff(img[0].astype(int)) >= 0)


def test_background_starts_camera_when_no_frame_is_available():
    cam = FakeCamera(frame=None)
    ctrl = make_controller({'cam': cam}, {'galvo': FakeScanner()})
    res = ctrl.snapGalvoCameraBackground()
    assert 'error' not in res
    assert cam.startCalls == 1
    assert res['frameWidth'] == 640 and res['frameHeight'] == 480


def test_background_rgb_frame_becomes_grayscale():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[..., 0] = 255
    ctrl = make_controller({'cam': FakeCamera(frame)}, {'galvo': FakeScanner()})
    res = ctrl.snapGalvoCameraBackground()
    assert res['width'] == 200 and res['height'] == 100
    assert _decode_png(res['image']).ndim == 2


def test_background_errors_without_camera():
    ctrl = make_controller({'flim': FakeFlimDetector()}, {'galvo': FakeScanner()})
    assert 'error' in ctrl.snapGalvoCameraBackground()
