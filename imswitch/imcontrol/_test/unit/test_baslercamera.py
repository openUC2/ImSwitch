"""
Tests for the Basler camera interface and manager against a fake pypylon: no
camera and no pylon runtime involved.

CameraBasler inherits its feature logic from CameraAV (see test_avcamera.py),
so the fake models what is Basler-specific -- typed GenICam nodes that raise
for missing features and reject floats on integer nodes, SFNC 1.x "Raw"/"Abs"
feature names, pylon's event-driven grab loop and the pixel-format converter --
and the tests check that the node adapter drives the whole shared surface.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake genicam
# ---------------------------------------------------------------------------
class LogicalErrorException(Exception):
    pass


class _Node:
    def __init__(self, cam, name, value=None, rng=None, inc=1, entries=None, command=None):
        self.cam, self.name = cam, name
        self._value, self._rng, self._inc = value, rng, inc
        self._entries, self._command = entries, command
        self.available = True
        self.writes = []

    @property
    def Value(self):
        return self._value() if callable(self._value) else self._value

    @Value.setter
    def Value(self, val):
        self._check(val)
        self._value = val
        self.writes.append(val)
        if self.name in ('BinningHorizontal', 'BinningVertical'):
            self.cam._on_binning()

    def _check(self, val):
        if self._rng is not None and not (self.Min <= val <= self.Max):
            raise LogicalErrorException(f"{self.name}: {val} outside [{self.Min}, {self.Max}]")

    def _range(self):
        return self._rng() if callable(self._rng) else self._rng

    @property
    def Min(self):
        return self._range()[0]

    @property
    def Max(self):
        return self._range()[1]

    @property
    def Inc(self):
        return self._inc


class IInteger(_Node):
    def _check(self, val):
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(f"{self.name}: int64 expected, got {val!r}")  # like SWIG
        super()._check(val)


class IFloat(_Node):
    pass


class IBoolean(_Node):
    pass


class IString(_Node):
    pass


class IEnumeration(_Node):
    @property
    def Symbolics(self):
        return tuple(self._entries)

    def _check(self, val):
        if val not in self._entries:
            raise LogicalErrorException(f"{self.name}: invalid entry {val}")


class ICommand(_Node):
    def Execute(self):
        self._command()


# ---------------------------------------------------------------------------
# Fake pylon
# ---------------------------------------------------------------------------
PACKED = ('Mono10p', 'Mono12p', 'Mono12Packed')


class FakeGrabResult:
    def __init__(self, cam, ok=True):
        self.ok = ok
        self.fmt = cam.f['PixelFormat'].Value
        w, h = cam.f['Width'].Value, cam.f['Height'].Value
        dtype = np.uint8 if self.fmt.endswith('8') else np.uint16
        # a gradient, so flips show
        self.data = (np.arange(h * w).reshape(h, w) % 200 + 10).astype(dtype)
        self.TimeStamp = 1000 * cam.frame_counter

    def GrabSucceeded(self):
        return self.ok

    def GetArray(self):
        assert self.fmt not in PACKED and not self.fmt.startswith('Bayer'), \
            f"numpy cannot take {self.fmt} as-is; it needs the converter"
        return self.data.copy()


class FakeImageFormatConverter:
    def __init__(self):
        self.OutputPixelFormat = None
        self.OutputBitAlignment = None

    def Convert(self, res):
        if self.OutputPixelFormat == 'RGB8packed':
            arr = np.repeat(res.data[:, :, None], 3, axis=2).astype(np.uint8)
        else:
            assert (self.OutputPixelFormat, self.OutputBitAlignment) == ('Mono16', 'LsbAligned')
            arr = res.data.astype(np.uint16)
        return SimpleNamespace(GetArray=lambda: arr)


class FakeImageEventHandler:
    def __init__(self):
        pass


class FakeDevice:
    """A pylon DeviceInfo, plus the spec of the camera behind it."""

    def __init__(self, serial="40000001", model="acA2440-75um", user_name="",
                 sensor=(2448, 2048), color=False, sfnc1=False):
        self.serial, self.model, self.user_name = serial, model, user_name
        self.sensor, self.color, self.sfnc1 = sensor, color, sfnc1
        self.camera = None

    def GetSerialNumber(self):
        return self.serial

    def GetUserDefinedName(self):
        return self.user_name

    def GetModelName(self):
        return self.model


class FakeCamera:
    """pylon.InstantCamera: nodes as attributes, event handler, grab loop."""

    def __init__(self, device):
        self.device = device
        device.camera = self
        self.opened = self.grabbing = False
        self.handler = None
        self.registrations = 0
        self.grab_args = None
        self.frame_counter = 0
        sensor = device.sensor
        f = self.f = {}
        f['AcquisitionMode'] = IEnumeration(self, 'AcquisitionMode', 'Continuous',
                                            entries=['Continuous', 'SingleFrame'])
        f['SensorWidth'] = IInteger(self, 'SensorWidth', sensor[0])
        f['SensorHeight'] = IInteger(self, 'SensorHeight', sensor[1])
        f['BinningHorizontal'] = IInteger(self, 'BinningHorizontal', 1, rng=(1, 4))
        f['BinningVertical'] = IInteger(self, 'BinningVertical', 1, rng=(1, 4))
        f['WidthMax'] = IInteger(self, 'WidthMax', lambda: sensor[0] // self._binning())
        f['HeightMax'] = IInteger(self, 'HeightMax', lambda: sensor[1] // self._binning())
        f['OffsetX'] = IInteger(self, 'OffsetX', 0, inc=4,
                                rng=lambda: (0, f['WidthMax'].Value - f['Width'].Value))
        f['OffsetY'] = IInteger(self, 'OffsetY', 0, inc=2,
                                rng=lambda: (0, f['HeightMax'].Value - f['Height'].Value))
        f['Width'] = IInteger(self, 'Width', sensor[0], inc=4,
                              rng=lambda: (16, f['WidthMax'].Value - f['OffsetX'].Value))
        f['Height'] = IInteger(self, 'Height', sensor[1], inc=2,
                               rng=lambda: (16, f['HeightMax'].Value - f['OffsetY'].Value))
        formats = ['Mono8', 'Mono10', 'Mono10p', 'Mono12', 'Mono12p']
        if device.color:
            formats += ['BayerRG8', 'BayerRG12', 'RGB8', 'BGR8']
        f['PixelFormat'] = IEnumeration(self, 'PixelFormat', 'Mono8', entries=formats)
        f['ExposureAuto'] = IEnumeration(self, 'ExposureAuto', 'Off',
                                         entries=['Off', 'Once', 'Continuous'])
        f['AcquisitionFrameRateEnable'] = IBoolean(self, 'AcquisitionFrameRateEnable', False)
        if device.sfnc1:  # ace classic GigE
            f['ExposureTimeAbs'] = IFloat(self, 'ExposureTimeAbs', 3000.0, rng=(35.0, 1e6))
            f['GainRaw'] = IInteger(self, 'GainRaw', 136, rng=(136, 542))
            f['BlackLevelRaw'] = IInteger(self, 'BlackLevelRaw', 32, rng=(0, 255))
            f['AcquisitionFrameRateAbs'] = IFloat(self, 'AcquisitionFrameRateAbs', 30.0,
                                                  rng=(0.1, 100.0))
        else:
            f['ExposureTime'] = IFloat(self, 'ExposureTime', 5000.0, rng=(10.0, 1e7))
            f['Gain'] = IFloat(self, 'Gain', 0.0, rng=(0.0, 36.0))
            f['BlackLevel'] = IFloat(self, 'BlackLevel', 0.0, rng=(0.0, 31.9))
            f['AcquisitionFrameRate'] = IFloat(self, 'AcquisitionFrameRate', 30.0,
                                               rng=(0.1, 1000.0))
        f['TriggerSelector'] = IEnumeration(self, 'TriggerSelector', 'FrameStart',
                                            entries=['FrameStart', 'FrameBurstStart'])
        f['TriggerMode'] = IEnumeration(self, 'TriggerMode', 'Off', entries=['Off', 'On'])
        f['TriggerSource'] = IEnumeration(self, 'TriggerSource', 'Software',
                                          entries=['Software', 'Line1', 'Line3', 'Line4'])
        f['TriggerSoftware'] = ICommand(self, 'TriggerSoftware', command=self._software_trigger)
        f['DeviceSerialNumber'] = IString(self, 'DeviceSerialNumber', device.serial)

    # -- GenICam side effects ------------------------------------------------
    def _binning(self):
        node = self.f['BinningHorizontal']
        return node.Value if node.available else 1

    def _on_binning(self):
        # like a real camera: clamp Width/Height to the new maxima
        for n, m in (('Width', 'WidthMax'), ('Height', 'HeightMax')):
            self.f[n]._value = min(self.f[n]._value, self.f[m].Value)

    def _software_trigger(self):
        assert (self.f['TriggerMode'].Value, self.f['TriggerSource'].Value) == ('On', 'Software')
        if self.grabbing:
            self.deliver()

    # -- InstantCamera API -----------------------------------------------------
    def __getattr__(self, name):
        nodes = self.__dict__.get('f', {})
        if name in nodes:
            return nodes[name]
        raise LogicalErrorException(f"Node not existing: {name}")

    def Open(self):
        self.opened = True

    def Close(self):
        self.opened = False

    def IsOpen(self):
        return self.opened

    def RegisterImageEventHandler(self, handler, mode, cleanup):
        self.handler = handler
        self.registrations += 1

    def StartGrabbing(self, strategy, loop):
        assert self.opened and not self.grabbing
        self.grabbing = True
        self.grab_args = (strategy, loop)

    def StopGrabbing(self):
        self.grabbing = False

    def IsGrabbing(self):
        return self.grabbing

    # test helper: pylon's grab-loop thread delivers one result
    def deliver(self, ok=True):
        self.frame_counter += 1
        res = FakeGrabResult(self, ok)
        self.handler.OnImageGrabbed(self, res)
        return res


class FakeTlFactory:
    devices = []
    busy = set()  # serial numbers held by another process

    @classmethod
    def GetInstance(cls):
        return cls()

    def EnumerateDevices(self):
        return tuple(self.devices)

    def CreateDevice(self, info):
        return info

    def IsDeviceAccessible(self, info):
        return info.serial not in self.busy


@pytest.fixture
def basler(monkeypatch):
    """Import baslercamera against the fake SDK; yields (module, install_devices)."""
    pylon = types.ModuleType("pypylon.pylon")
    pylon.TlFactory = FakeTlFactory
    pylon.InstantCamera = FakeCamera
    pylon.ImageEventHandler = FakeImageEventHandler
    pylon.ImageFormatConverter = FakeImageFormatConverter
    pylon.GrabStrategy_LatestImageOnly = 'LatestImageOnly'
    pylon.GrabLoop_ProvidedByInstantCamera = 'ProvidedByInstantCamera'
    pylon.RegistrationMode_ReplaceAll = 'ReplaceAll'
    pylon.Cleanup_None = 'CleanupNone'
    pylon.PixelType_RGB8packed = 'RGB8packed'
    pylon.PixelType_Mono16 = 'Mono16'
    pylon.OutputBitAlignment_LsbAligned = 'LsbAligned'
    genicam = types.ModuleType("pypylon.genicam")
    genicam.IInteger = IInteger
    genicam.IsAvailable = lambda node: node.available
    genicam.LogicalErrorException = LogicalErrorException
    pkg = types.ModuleType("pypylon")
    pkg.pylon, pkg.genicam = pylon, genicam
    for name, mod in (("pypylon", pkg), ("pypylon.pylon", pylon), ("pypylon.genicam", genicam)):
        monkeypatch.setitem(sys.modules, name, mod)
    modname = "imswitch.imcontrol.model.interfaces.baslercamera"
    sys.modules.pop(modname, None)
    module = importlib.import_module(modname)

    def install(*devices):
        FakeTlFactory.devices = list(devices)
        return devices[0] if devices else None

    FakeTlFactory.busy = set()
    yield module, install
    FakeTlFactory.devices = []
    sys.modules.pop(modname, None)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class TestDeviceSelection:
    def test_by_serial_number_user_name_and_index(self, basler):
        mod, install = basler
        install(FakeDevice("40000001"), FakeDevice("40000002", model="a2A1920-160ucBAS",
                                                  user_name="overview"))
        assert mod.CameraBasler("40000002").model == "a2A1920-160ucBAS"
        assert mod.CameraBasler("overview").model == "a2A1920-160ucBAS"
        assert mod.CameraBasler(40000002).model == "a2A1920-160ucBAS"  # int from JSON
        assert mod.CameraBasler(1).model == "a2A1920-160ucBAS"
        assert mod.CameraBasler(None).model == "acA2440-75um"

    def test_stale_serial_number_falls_back_to_the_first_free_camera(self, basler):
        mod, install = basler
        install(FakeDevice("40000001"), FakeDevice("40000002"))
        FakeTlFactory.busy = {"40000001"}
        cam = mod.CameraBasler("22222222")
        assert cam.is_connected
        assert FakeTlFactory.devices[1].camera.opened

    def test_name_matching_nothing_raises_so_the_manager_mocks(self, basler):
        mod, install = basler
        install(FakeDevice())
        with pytest.raises(RuntimeError, match="No Basler camera 'mock'"):
            mod.CameraBasler("mock")

    def test_no_camera_raises(self, basler):
        mod, install = basler
        install()
        with pytest.raises(RuntimeError, match="No Basler cameras found"):
            mod.CameraBasler()


class TestFrames:
    def test_mono_negotiates_mono12_and_streams_latest_only(self, basler):
        mod, install = basler
        dev = install(FakeDevice(sensor=(64, 32)))
        cam = mod.CameraBasler()
        assert cam.isRGB is False
        assert dev.camera.f['PixelFormat'].Value == 'Mono12' and cam.bitDepth == 12
        cam.start_live()
        assert dev.camera.grab_args == ('LatestImageOnly', 'ProvidedByInstantCamera')
        res = dev.camera.deliver()
        frame, fid = cam.getLast(returnFrameNumber=True)
        assert frame.shape == (32, 64) and frame.dtype == np.uint16
        assert fid == 1
        assert not np.shares_memory(frame, res.data)

    def test_frame_ids_keep_growing_across_stream_restarts(self, basler):
        mod, install = basler
        dev = install(FakeDevice(sensor=(64, 32)))
        cam = mod.CameraBasler()
        cam.start_live()
        dev.camera.deliver()
        cam.stop_live()
        cam.start_live()
        dev.camera.deliver()
        assert cam.getFrameNumber() == 2
        assert dev.camera.registrations == 1  # handler registered once

    def test_packed_mono_goes_through_the_converter(self, basler):
        mod, install = basler
        dev = install(FakeDevice(sensor=(64, 32)))
        cam = mod.CameraBasler(pixel_format="Mono12p")
        cam.start_live()
        dev.camera.deliver()
        frame = cam.getLast()
        assert frame.shape == (32, 64) and frame.dtype == np.uint16

    def test_color_camera_is_detected_and_bayer_is_converted(self, basler):
        mod, install = basler
        dev = install(FakeDevice(sensor=(64, 32), color=True))
        cam = mod.CameraBasler()
        assert cam.isRGB is True
        assert dev.camera.f['PixelFormat'].Value == 'BayerRG8'
        cam.start_live()
        dev.camera.deliver()
        frame = cam.getLast()
        assert frame.shape == (32, 64, 3) and frame.dtype == np.uint8

    def test_failed_grabs_are_counted_not_buffered(self, basler):
        mod, install = basler
        dev = install(FakeDevice(sensor=(64, 32)))
        cam = mod.CameraBasler()
        cam.start_live()
        dev.camera.deliver(ok=False)
        assert cam.getLast(timeout=0.01) is None
        assert cam.getStreamDiagnostics()["incomplete_frames"] == 1

    def test_flip(self, basler):
        mod, install = basler
        dev = install(FakeDevice(sensor=(64, 32)))
        cam = mod.CameraBasler(flipImage=(False, True), pixel_format="Mono8")
        cam.start_live()
        res = dev.camera.deliver()
        np.testing.assert_array_equal(cam.getLast(), np.flip(res.data, axis=1))


class TestFeatures:
    def test_exposure_gain_blacklevel_frame_rate(self, basler):
        mod, install = basler
        dev = install(FakeDevice())
        cam = mod.CameraBasler()
        f = dev.camera.f
        cam.setPropertyValue("exposure", 12.5)
        assert f['ExposureTime'].Value == 12500.0
        assert cam.getPropertyValue("exposure") == 12.5
        cam.setPropertyValue("gain", 50)  # clamped to the camera's range
        assert f['Gain'].Value == 36.0
        cam.setPropertyValue("blacklevel", 4)
        assert cam.getPropertyValue("blacklevel") == 4.0
        assert cam.setPropertyValue("frame_rate", "25") == 25.0
        assert f['AcquisitionFrameRateEnable'].Value is True
        assert cam.set_frame_rate(-1) == -1
        assert f['AcquisitionFrameRateEnable'].Value is False
        cam.setPropertyValue("exposure_mode", "auto")
        assert f['ExposureAuto'].Value == 'Continuous'

    def test_sfnc1_camera_uses_abs_and_raw_features(self, basler):
        mod, install = basler
        dev = install(FakeDevice(model="acA1300-30gm", sfnc1=True))
        cam = mod.CameraBasler()
        f = dev.camera.f
        cam.setPropertyValue("exposure", 2)
        assert f['ExposureTimeAbs'].Value == 2000.0
        assert cam.get_gain() == (136.0, 136.0, 542.0)
        cam.setPropertyValue("gain", 200.4)  # float in, int written
        assert f['GainRaw'].Value == 200
        cam.setPropertyValue("blacklevel", 40)
        assert f['BlackLevelRaw'].Value == 40
        assert cam.set_frame_rate(10) == 10.0
        assert f['AcquisitionFrameRateAbs'].Value == 10.0

    def test_binning_and_hardware_roi(self, basler):
        mod, install = basler
        dev = install(FakeDevice())
        cam = mod.CameraBasler(binning=2)
        assert cam.binning == 2
        assert (cam.SensorWidth, cam.SensorHeight) == (1224, 1024)
        assert cam.getBinningRange() == (1, 4)
        assert cam.setROI(hpos=101, vpos=51, hsize=1003, vsize=501) == (100, 50, 1000, 500)
        assert dev.camera.f['Width'].Value == 1000

    def test_unavailable_or_missing_node_is_none(self, basler):
        mod, install = basler
        dev = install(FakeDevice())
        cam = mod.CameraBasler()
        dev.camera.f['BinningHorizontal'].available = False
        assert cam._feature('BinningHorizontal') is None
        assert cam.getBinningRange() == (1, 1)
        assert cam._feature('NoSuchFeature') is None

    def test_software_trigger_snap_and_external_line(self, basler):
        mod, install = basler
        dev = install(FakeDevice(sensor=(64, 32)))
        cam = mod.CameraBasler()
        f = dev.camera.f
        cam.start_live()
        dev.camera.deliver()  # a stale free-run frame
        assert cam.setTriggerSource("Internal trigger") is True
        assert (f['TriggerMode'].Value, f['TriggerSource'].Value) == ('On', 'Software')
        assert dev.camera.grabbing  # stream restarted after the change
        frame = cam.snapSoftwareTrigger(timeout=0.5)
        assert frame is not None and frame.shape == (32, 64)
        assert dev.camera.frame_counter == 2  # exactly one exposure fired
        assert cam.setTriggerSource("External trigger") is True
        assert f['TriggerSource'].Value == 'Line1'  # Basler's opto-isolated input
        assert cam.setTriggerSource("Continous") is True
        assert f['TriggerMode'].Value == 'Off'

    def test_close_stops_grabbing_and_closes(self, basler):
        mod, install = basler
        dev = install(FakeDevice())
        cam = mod.CameraBasler()
        cam.start_live()
        cam.close()
        assert not dev.camera.grabbing and not dev.camera.opened

    def test_status_snapshot(self, basler):
        mod, install = basler
        install(FakeDevice())
        params = mod.CameraBasler().get_camera_parameters()
        assert params["model_name"] == "acA2440-75um"
        assert params["DeviceSerialNumber"] == "40000001"
        assert params["exposure_min"] == 10.0


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
def _detector_info(**props):
    base = {"cameraListIndex": 0, "basler": {"exposure": 10, "gain": 2}}
    base.update(props)
    return SimpleNamespace(managerProperties=base, forAcquisition=True, forFocusLock=False)


@pytest.fixture
def make_manager(basler):
    from imswitch.imcontrol.model.managers.detectors.BaslerManager import BaslerManager
    mod, install = basler

    def build(device=None, **props):
        dev = install(device or FakeDevice())
        return BaslerManager(_detector_info(**props), "BaslerCam"), dev

    return build


class TestBaslerManager:
    def test_parameters_are_seeded_from_hardware(self, make_manager):
        mgr, dev = make_manager()
        p = mgr.parameters
        assert p['exposure'].value == 10.0
        assert (p['exposure'].valueMin, p['exposure'].valueMax) == (0.01, 10000.0)
        assert p['gain'].value == 2.0
        assert p['previewMaxValue'].value == 4095  # Mono12
        assert mgr.fullShape == (2448, 2048)
        assert mgr.supportedBinnings == [1, 2, 4]
        assert mgr.isRGB is False

    def test_legacy_basler_setup_still_loads(self, make_manager):
        mgr, dev = make_manager(cameraEffPixelsize=0.5,
                                basler={"exposure": 20, "gain": 1, "blacklevel": 2,
                                        "isRGB": 0, "roi_size": 512})
        assert mgr._camera.model == "acA2440-75um"
        assert dev.camera.f['ExposureTime'].Value == 20000.0

    def test_binning_trigger_and_snap_through_the_manager(self, make_manager):
        mgr, dev = make_manager(FakeDevice(sensor=(64, 32)))
        mgr.startAcquisition()
        mgr.setBinning(2)
        assert mgr.fullShape == (32, 16)
        assert mgr.setTriggerSource("software") is True
        frame = mgr.snapSync(timeout=0.5)
        assert frame.shape == (16, 32)

    def test_status_says_basler(self, make_manager):
        mgr, dev = make_manager()
        status = mgr.getCameraStatus()
        assert status['cameraType'] == 'Basler'
        assert status['isConnected'] is True and status['isMock'] is False
        assert status['hardwareParameters']['DeviceSerialNumber'] == "40000001"

    def test_mock_index_loads_the_mocker(self, make_manager):
        mgr, dev = make_manager(cameraListIndex="mock")
        assert mgr._camera.model == "mock"
        assert dev.camera is None
