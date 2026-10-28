"""
Camera selection in CameraHIK._open_camera against a fake MvCamera: the real
ctypes device-info structures, no native library.

Regression: a cameraListIndex serial checksum that matched no connected camera
sent the (only, perfectly working) camera to the mock.
"""

import ctypes

import numpy as np
import pytest

import imswitch.imcontrol.model.interfaces.hikcamera as hik


def _usb_device(serial: bytes):
    info = hik.MV_CC_DEVICE_INFO()
    info.nTLayerType = hik.MV_USB_DEVICE
    for i, b in enumerate(serial):
        info.SpecialInfo.stUsb3VInfo.chSerialNumber[i] = b
    return info


def _checksum(serial: bytes):
    return int(np.sum(np.frombuffer(serial, dtype=np.uint8)))


class FakeMvCamera:
    devices = []
    busy = set()    # serial checksums whose OpenDevice is refused
    opened = []

    @staticmethod
    def MV_CC_EnumDevices(layer, lst):
        infos = [d for d in FakeMvCamera.devices if d.nTLayerType == layer]
        lst.nDeviceNum = len(infos)
        for i, d in enumerate(infos):
            lst.pDeviceInfo[i] = ctypes.pointer(d)
        return 0

    def MV_CC_CreateHandle(self, info):
        self.checksum = int(np.sum(info.SpecialInfo.stUsb3VInfo.chSerialNumber))
        return 0

    def MV_CC_OpenDevice(self, mode, key):
        if self.checksum in FakeMvCamera.busy:
            return 0x80000203  # MV_E_ACCESS_DENIED
        FakeMvCamera.opened.append(self.checksum)
        return 0

    def __getattr__(self, name):
        # every other SDK call succeeds and leaves its output struct zeroed
        return lambda *a, **k: 0


SERIAL_A, SERIAL_B = b"DA0490XYZ", b"DA0777QRS"


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setattr(hik, "MvCamera", FakeMvCamera)
    monkeypatch.setattr(hik.CameraHIK, "_claimedChecksums", set())
    FakeMvCamera.devices = [_usb_device(SERIAL_A), _usb_device(SERIAL_B)]
    FakeMvCamera.busy = set()
    FakeMvCamera.opened = []
    return FakeMvCamera


def test_matching_checksum_opens_that_camera(sdk):
    cam = hik.CameraHIK(cameraNo=_checksum(SERIAL_B))
    assert sdk.opened == [_checksum(SERIAL_B)]
    assert cam._serial_checksum == _checksum(SERIAL_B)


def test_unknown_checksum_falls_back_instead_of_failing(sdk):
    sdk.devices = [_usb_device(SERIAL_A)]
    cam = hik.CameraHIK(cameraNo=507)
    assert cam.is_connected
    assert sdk.opened == [_checksum(SERIAL_A)]


def test_fallback_skips_cameras_held_by_another_detector(sdk):
    first = hik.CameraHIK(cameraNo=_checksum(SERIAL_A))
    second = hik.CameraHIK(cameraNo=507)
    assert second._serial_checksum == _checksum(SERIAL_B)
    # closing frees the camera for the next fallback
    first.close()
    assert hik.CameraHIK._claimedChecksums == {_checksum(SERIAL_B)}


def test_fallback_moves_on_when_a_camera_refuses_to_open(sdk):
    sdk.busy = {_checksum(SERIAL_A)}  # e.g. held by another process
    cam = hik.CameraHIK(cameraNo=507)
    assert cam._serial_checksum == _checksum(SERIAL_B)


def test_no_openable_camera_is_a_clear_error(sdk):
    sdk.busy = {_checksum(SERIAL_A), _checksum(SERIAL_B)}
    with pytest.raises(RuntimeError, match="Could not open a Hik camera"):
        hik.CameraHIK(cameraNo=507)


def test_index_out_of_range_falls_back_to_first(sdk):
    cam = hik.CameraHIK(cameraNo=5)
    assert cam._serial_checksum == _checksum(SERIAL_A)
