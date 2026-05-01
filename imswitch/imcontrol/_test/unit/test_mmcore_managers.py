"""
Tests for the MMCore* device managers using the Micro-Manager DemoCamera
adapter. The entire module is skipped when ``pymmcore-plus`` is not
installed or when no device adapter library can be located.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest


pymmcore_plus = pytest.importorskip("pymmcore_plus")


def _adapters_available() -> bool:
    """Cross-platform check that delegates to the production discovery code."""
    from imswitch.imcontrol.model.managers import MMCoreManager

    return bool(MMCoreManager.discover_adapter_paths())


pytestmark = pytest.mark.skipif(
    not _adapters_available(),
    reason=(
        "No Micro-Manager 2.0 device adapters found. Install via "
        "`pip install pymmcore-plus[cli] && mmcore install`, or set "
        "MICROMANAGER_PATH to your MM 2.0 install directory."
    ),
)


def _make_detector_info():
    info = MagicMock()
    info.managerProperties = {
        "adapterName": "DemoCamera",
        "deviceName": "DCam",
        "deviceLabel": "Camera",
    }
    info.forAcquisition = True
    info.forFocusLock = False
    return info


def _make_positioner_info():
    info = MagicMock()
    info.managerProperties = {
        "xyAdapterName": "DemoCamera",
        "xyDeviceName": "DXYStage",
        "xyDeviceLabel": "XY",
        "zAdapterName": "DemoCamera",
        "zDeviceName": "DStage",
        "zDeviceLabel": "Z",
    }
    info.axes = ["X", "Y", "Z"]
    info.forPositioning = True
    info.forScanning = True
    info.resetOnClose = False
    return info


def _make_laser_info():
    info = MagicMock()
    info.managerProperties = {
        "adapterName": "DemoCamera",
        "deviceName": "DShutter",
        "deviceLabel": "Shutter",
        "mode": "shutter",
    }
    info.wavelength = 488
    info.valueRangeMin = 0
    info.valueRangeMax = 1
    info.valueRangeStep = 1
    info.freqRangeMin = 0
    info.freqRangeMax = 0
    info.freqRangeInit = 0
    return info


# ---------------------------------------------------------------------------
# MMCoreManager singleton
# ---------------------------------------------------------------------------
class TestMMCoreManager:
    def test_get_core_singleton(self):
        from imswitch.imcontrol.model.managers import MMCoreManager

        c1 = MMCoreManager.get_core()
        c2 = MMCoreManager.get_core()
        assert c1 is c2
        assert c1.getVersionInfo() != ""

    def test_is_available(self):
        from imswitch.imcontrol.model.managers import MMCoreManager

        assert MMCoreManager.is_available() is True

    def test_get_available_adapters(self):
        from imswitch.imcontrol.model.managers import MMCoreManager

        # No explicit path → discover_adapter_paths() handles platform fallbacks.
        adapters = MMCoreManager.get_available_adapters()
        assert "DemoCamera" in adapters, (
            f"DemoCamera adapter not found. Discovered paths: "
            f"{MMCoreManager.discover_adapter_paths()}"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class TestMMCoreDetectorManager:
    def _make_manager(self):
        from imswitch.imcontrol.model.managers.detectors.MMCoreDetectorManager import (
            MMCoreDetectorManager,
        )

        return MMCoreDetectorManager(_make_detector_info(), "TestCamera")

    def test_snap(self):
        mgr = self._make_manager()
        try:
            frame = mgr.getLatestFrame()
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 2
            assert frame.shape[0] > 0 and frame.shape[1] > 0
        finally:
            mgr.finalize()

    def test_continuous_acquisition(self):
        import time

        mgr = self._make_manager()
        try:
            mgr.startAcquisition()
            time.sleep(0.3)
            mgr.stopAcquisition()
        finally:
            mgr.finalize()

    def test_exposure(self):
        mgr = self._make_manager()
        try:
            mgr.setParameter("Exposure", 50.0)
            assert abs(mgr._core.getExposure() - 50.0) < 0.1
        finally:
            mgr.finalize()

    def test_pixel_size_format(self):
        mgr = self._make_manager()
        try:
            ps = mgr.pixelSizeUm
            assert isinstance(ps, list) and len(ps) == 3
        finally:
            mgr.finalize()


# ---------------------------------------------------------------------------
# Positioner
# ---------------------------------------------------------------------------
class TestMMCorePositionerManager:
    def _make_manager(self):
        from imswitch.imcontrol.model.managers.positioners.MMCorePositionerManager import (
            MMCorePositionerManager,
        )

        return MMCorePositionerManager(_make_positioner_info(), "TestStage")

    def test_move_xy(self):
        mgr = self._make_manager()
        try:
            x0 = mgr.getPosition("X")
            mgr.move(10.0, "X")
            assert abs(mgr.getPosition("X") - (x0 + 10.0)) < 1e-3
        finally:
            mgr.finalize()

    def test_move_z(self):
        mgr = self._make_manager()
        try:
            z0 = mgr.getPosition("Z")
            mgr.move(5.0, "Z")
            assert abs(mgr.getPosition("Z") - (z0 + 5.0)) < 1e-3
        finally:
            mgr.finalize()

    def test_set_position(self):
        mgr = self._make_manager()
        try:
            mgr.setPosition(123.0, "X")
            assert abs(mgr.getPosition("X") - 123.0) < 1e-3
        finally:
            mgr.finalize()


# ---------------------------------------------------------------------------
# Laser
# ---------------------------------------------------------------------------
class TestMMCoreLaserManager:
    def _make_manager(self):
        from imswitch.imcontrol.model.managers.lasers.MMCoreLaserManager import (
            MMCoreLaserManager,
        )

        return MMCoreLaserManager(_make_laser_info(), "TestLaser")

    def test_shutter_toggle(self):
        mgr = self._make_manager()
        try:
            mgr.setEnabled(True)
            assert mgr._core.getShutterOpen() is True
            mgr.setEnabled(False)
            assert mgr._core.getShutterOpen() is False
        finally:
            mgr.finalize()
