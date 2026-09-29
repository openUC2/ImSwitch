"""
HIK trigger path (WP7 of the strobed StageMap sweep) against a fake MvCamera.

Frames are pushed through the real ctypes callback with a real
MV_FRAME_OUT_INFO_EX, so the test covers the path from the SDK frame info to
getChunkWithTriggerIndex. No native library, no camera.
"""

import ctypes
import logging

import numpy as np
import pytest

import imswitch.imcontrol.model.interfaces.hikcamera as hik
from imswitch.imcontrol.model.managers.detectors.HikCamManager import HikCamManager

MV_E_SUPPORT = 0x80000001


class FakeMvCamera:
    """Every SDK call succeeds unless its node is listed in ``refuse``."""

    devices = []
    refuse = set()
    calls = []

    @staticmethod
    def MV_CC_EnumDevices(layer, lst):
        infos = [d for d in FakeMvCamera.devices if d.nTLayerType == layer]
        lst.nDeviceNum = len(infos)
        for i, d in enumerate(infos):
            lst.pDeviceInfo[i] = ctypes.pointer(d)
        return 0

    def _node_call(self, name, node, *args):
        FakeMvCamera.calls.append((name, node) + args)
        return MV_E_SUPPORT if node in FakeMvCamera.refuse else 0

    def MV_CC_SetEnumValueByString(self, node, value):
        return self._node_call("SetEnumValueByString", node, value)

    def MV_CC_SetEnumValue(self, node, value):
        return self._node_call("SetEnumValue", node, value)

    def MV_CC_SetFloatValue(self, node, value):
        return self._node_call("SetFloatValue", node, value)

    def MV_CC_SetBoolValue(self, node, value):
        return self._node_call("SetBoolValue", node, value)

    def __getattr__(self, name):
        return lambda *a, **k: 0


@pytest.fixture
def cam(monkeypatch):
    info = hik.MV_CC_DEVICE_INFO()
    info.nTLayerType = hik.MV_USB_DEVICE
    for i, b in enumerate(b"DA0490XYZ"):
        info.SpecialInfo.stUsb3VInfo.chSerialNumber[i] = b
    monkeypatch.setattr(hik, "MvCamera", FakeMvCamera)
    monkeypatch.setattr(hik.CameraHIK, "_claimedChecksums", set())
    FakeMvCamera.devices = [info]
    FakeMvCamera.refuse = set()
    FakeMvCamera.calls = []
    camera = hik.CameraHIK(cameraNo=0)
    FakeMvCamera.calls = []
    return camera


def push(camera, value, frame_num, trigger_index, w=8, h=4):
    """Deliver one Mono8 frame through the SDK callback."""
    buf = (ctypes.c_ubyte * (w * h))(*([value] * (w * h)))
    info = hik.MV_FRAME_OUT_INFO_EX()
    info.nWidth, info.nHeight = w, h
    info.nFrameLen = w * h
    info.enPixelType = hik.PixelType_Gvsp_Mono8
    info.nFrameNum = frame_num
    info.nTriggerIndex = trigger_index
    camera._sdk_cb(ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)), ctypes.pointer(info), None)


def manager(camera):
    mgr = HikCamManager.__new__(HikCamManager)
    mgr._camera = camera
    mgr._HikCamManager__logger = logging.getLogger("test")
    return mgr


def test_trigger_index_reaches_getChunkWithTriggerIndex_in_order(cam):
    mgr = manager(cam)
    push(cam, 10, frame_num=1, trigger_index=501)
    push(cam, 20, frame_num=2, trigger_index=503)   # pulse 502 produced no frame
    frames, ids, trigs = mgr.getChunkWithTriggerIndex()
    assert list(trigs) == [501, 503]
    assert list(ids) == [1, 2]
    assert [int(f[0, 0]) for f in frames] == [10, 20]
    # drained: the next call only sees what arrived since
    push(cam, 30, frame_num=3, trigger_index=504)
    frames, ids, trigs = mgr.getChunkWithTriggerIndex()
    assert list(trigs) == [504] and int(frames[0][0, 0]) == 30


def test_buffers_stay_in_lock_step_when_the_ring_overflows(cam):
    for k in range(cam.NBuffer + 4):
        push(cam, k, frame_num=k, trigger_index=100 + k)
    frames, ids, trigs = cam.getLastChunkWithTriggerIndex()
    assert len(frames) == len(ids) == len(trigs) == cam.NBuffer
    for f, i, t in zip(frames, ids, trigs):
        assert int(f[0, 0]) == i and t == 100 + i


def test_flush_clears_the_trigger_indices_too(cam):
    push(cam, 1, frame_num=1, trigger_index=7)
    cam.flushBuffer()
    frames, ids, trigs = cam.getLastChunkWithTriggerIndex()
    assert len(frames) == len(ids) == len(trigs) == 0


def test_getChunk_and_getLastChunk_are_unchanged(cam):
    mgr = manager(cam)
    push(cam, 5, frame_num=11, trigger_index=1)
    push(cam, 6, frame_num=12, trigger_index=2)
    chunk = mgr.getChunk()
    assert isinstance(chunk, tuple) and len(chunk) == 2
    frames, ids = chunk
    assert list(ids) == [11, 12] and frames.shape == (2, 4, 8)
    frames, ids = cam.getLastChunk()
    assert len(frames) == 0 and len(ids) == 0


def test_setters_call_the_sdk(cam):
    mgr = manager(cam)
    assert mgr.setTriggerActivation("RisingEdge") is True
    assert mgr.setTriggerDelayUs(12.5) is True
    assert mgr.setShutterMode("GlobalReset") is True
    assert ("SetEnumValueByString", "TriggerActivation", "RisingEdge") in FakeMvCamera.calls
    assert ("SetFloatValue", "TriggerDelay", 12.5) in FakeMvCamera.calls
    assert ("SetEnumValueByString", "SensorShutterMode", "GlobalReset") in FakeMvCamera.calls


def test_setters_return_false_instead_of_raising_when_unsupported(cam):
    FakeMvCamera.refuse = {"TriggerActivation", "TriggerDelay", "SensorShutterMode",
                           "ShutterMode", "FrameSpecInfoSelector"}
    mgr = manager(cam)
    assert mgr.setTriggerActivation("RisingEdge") is False
    assert mgr.setTriggerDelayUs(5) is False
    assert mgr.setShutterMode("GlobalReset") is False
    assert mgr.setTriggerIndexEmbedding(True) is False
    # both shutter node names were tried before giving up
    tried = [c[1] for c in FakeMvCamera.calls if c[0] == "SetEnumValueByString"]
    assert "SensorShutterMode" in tried and "ShutterMode" in tried


def test_setter_exception_becomes_false(cam, monkeypatch):
    def boom(*a, **k):
        raise OSError("SDK gone")
    monkeypatch.setattr(cam.camera, "MV_CC_SetFloatValue", boom, raising=False)
    assert manager(cam).setTriggerDelayUs(3) is False


def test_external_trigger_leaves_the_trigger_edge_alone(cam):
    """Existing setups may rely on an edge saved in the camera; only the
    strobed prescan sets (and restores) TriggerActivation."""
    assert cam.setTriggerSource("External trigger") is True
    assert not any(c[1] == "TriggerActivation" for c in FakeMvCamera.calls if len(c) > 1)


def test_manager_without_trigger_index_support_returns_none():
    class OldCamera:
        model = "mock"

        def getLastChunk(self):
            return np.zeros((0,)), np.zeros((0,))

    mgr = manager(OldCamera())
    assert mgr.getChunkWithTriggerIndex() is None
    assert mgr.setTriggerActivation() is False
