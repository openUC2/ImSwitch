"""
Tests for the Allied Vision camera interface and manager against a fake VmbPy
SDK: no camera and no native library involved.

The fake models the GenICam behaviour the driver relies on (Width/Height
maxima that shrink with binning and offset, feature increments, enum entries,
frames delivered through the streaming handler), so that binning, hardware
ROI, exposure modes, frame-rate cap and trigger handling can be verified the
same way they are expected to behave on an Alvium.
"""

from __future__ import annotations

import enum
import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake VmbPy
# ---------------------------------------------------------------------------
class FakePixelFormat(enum.IntEnum):
    Mono8 = 1
    Mono10 = 2
    Mono12 = 3
    Mono16 = 4
    Mono12Packed = 5
    BayerRG8 = 6
    Rgb8 = 7
    Bgr8 = 8

    def __str__(self):
        return self.name


class FakeFrameStatus(enum.IntEnum):
    Complete = 0
    Incomplete = 1


class FakeAllocationMode(enum.IntEnum):
    AnnounceFrame = 0


class FakeCameraError(Exception):
    pass


class FakeFeatureError(Exception):
    pass


class FakeFeature:
    def __init__(self, cam, name, value=None, rng=None, inc=1, entries=None, command=None):
        self.cam = cam
        self.name = name
        self._value = value
        self._rng = rng
        self._inc = inc
        self._entries = entries
        self._command = command
        self.writes = []

    def get(self):
        if callable(self._value):
            return self._value()
        return self._value

    def get_range(self):
        return self._rng() if callable(self._rng) else self._rng

    def get_increment(self):
        return self._inc

    def get_available_entries(self):
        return tuple(self._entries or ())

    def set(self, val):
        if self._entries is not None:
            val = str(val)
            if val not in self._entries:
                raise FakeFeatureError(f"{self.name}: invalid entry {val}")
        elif self._rng is not None:
            lo, hi = self.get_range()
            if not (lo <= val <= hi):
                raise FakeFeatureError(f"{self.name}: {val} outside [{lo}, {hi}]")
        self._value = val
        self.writes.append(val)
        if self.name in ('BinningHorizontal', 'BinningVertical'):
            self.cam._on_binning()

    def run(self):
        if self._command:
            self._command()

    def is_done(self):
        return True


class FakeFrame:
    def __init__(self, cam, fid, status=FakeFrameStatus.Complete):
        self.cam = cam
        self.fid = fid
        self.status = status
        self.fmt = cam.pixel_format
        w, h = cam.f['Width'].get(), cam.f['Height'].get()
        bits = 16 if self.fmt in (FakePixelFormat.Mono10, FakePixelFormat.Mono12,
                                  FakePixelFormat.Mono16) else 8
        dtype = np.uint16 if bits == 16 else np.uint8
        ch = 3 if self.fmt in (FakePixelFormat.Rgb8, FakePixelFormat.Bgr8) else 1
        # a gradient so flips are detectable
        base = np.arange(h * w, dtype=dtype).reshape(h, w) % 200 + 10
        self.data = np.repeat(base[:, :, None], ch, axis=2)

    def get_status(self):
        return self.status

    def get_pixel_format(self):
        return self.fmt

    def as_numpy_ndarray(self):
        return self.data

    def get_id(self):
        return self.fid

    def get_timestamp(self):
        return self.fid * 1000

    def convert_pixel_format(self, target):
        assert self.fmt == FakePixelFormat.BayerRG8 and target == FakePixelFormat.Rgb8
        conv = FakeFrame.__new__(FakeFrame)
        conv.cam, conv.fid, conv.status, conv.fmt = self.cam, self.fid, self.status, target
        conv.data = np.repeat(self.data, 3, axis=2)
        return conv


class FakeCamera:
    def __init__(self, name="Alvium 1800 U-500m", sensor=(2592, 1944), color=False,
                 binning_max=4, with_binning=True):
        self.name = name
        self.sensor = sensor
        self.color = color
        self.entered = 0
        self.streaming = False
        self.handler = None
        self.queued = []
        self.frame_counter = 0
        self.formats = [FakePixelFormat.Mono8, FakePixelFormat.Mono10, FakePixelFormat.Mono12,
                        FakePixelFormat.Mono12Packed]
        if color:
            self.formats += [FakePixelFormat.BayerRG8, FakePixelFormat.Rgb8]
        self.pixel_format = FakePixelFormat.Mono8
        f = self.f = {}
        f['AcquisitionMode'] = FakeFeature(self, 'AcquisitionMode', 'Continuous',
                                           entries=['Continuous', 'SingleFrame'])
        f['SensorWidth'] = FakeFeature(self, 'SensorWidth', sensor[0])
        f['SensorHeight'] = FakeFeature(self, 'SensorHeight', sensor[1])
        if with_binning:
            f['BinningHorizontal'] = FakeFeature(self, 'BinningHorizontal', 1, rng=(1, binning_max))
            f['BinningVertical'] = FakeFeature(self, 'BinningVertical', 1, rng=(1, binning_max))
        f['WidthMax'] = FakeFeature(self, 'WidthMax', lambda: sensor[0] // self._binning())
        f['HeightMax'] = FakeFeature(self, 'HeightMax', lambda: sensor[1] // self._binning())
        f['OffsetX'] = FakeFeature(self, 'OffsetX', 0, inc=2,
                                   rng=lambda: (0, f['WidthMax'].get() - f['Width'].get()))
        f['OffsetY'] = FakeFeature(self, 'OffsetY', 0, inc=2,
                                   rng=lambda: (0, f['HeightMax'].get() - f['Height'].get()))
        f['Width'] = FakeFeature(self, 'Width', sensor[0], inc=8,
                                 rng=lambda: (8, f['WidthMax'].get() - f['OffsetX'].get()))
        f['Height'] = FakeFeature(self, 'Height', sensor[1], inc=2,
                                  rng=lambda: (2, f['HeightMax'].get() - f['OffsetY'].get()))
        f['ExposureTime'] = FakeFeature(self, 'ExposureTime', 20000.0, rng=(21.0, 1e7))
        f['ExposureAuto'] = FakeFeature(self, 'ExposureAuto', 'Off',
                                        entries=['Off', 'Once', 'Continuous'])
        f['Gain'] = FakeFeature(self, 'Gain', 0.0, rng=(0.0, 24.0))
        f['BlackLevel'] = FakeFeature(self, 'BlackLevel', 0.0, rng=(0.0, 255.0))
        f['AcquisitionFrameRateEnable'] = FakeFeature(self, 'AcquisitionFrameRateEnable', False)
        f['AcquisitionFrameRate'] = FakeFeature(self, 'AcquisitionFrameRate', 30.0, rng=(1.0, 100.0))
        f['TriggerSelector'] = FakeFeature(self, 'TriggerSelector', 'FrameStart',
                                           entries=['FrameStart', 'AcquisitionStart'])
        f['TriggerMode'] = FakeFeature(self, 'TriggerMode', 'Off', entries=['Off', 'On'])
        f['TriggerSource'] = FakeFeature(self, 'TriggerSource', 'Software',
                                         entries=['Software', 'Line0', 'Line1'])
        f['TriggerSoftware'] = FakeFeature(self, 'TriggerSoftware', command=self._software_trigger)
        f['PixelFormat'] = FakeFeature(self, 'PixelFormat', 'Mono8')
        f['DeviceSerialNumber'] = FakeFeature(self, 'DeviceSerialNumber', '0123')

    # -- GenICam side effects ------------------------------------------------
    def _binning(self):
        return self.f['BinningHorizontal'].get() if 'BinningHorizontal' in self.f else 1

    def _on_binning(self):
        # like a real camera: clamp Width/Height to the new maxima
        for n, m in (('Width', 'WidthMax'), ('Height', 'HeightMax')):
            self.f[n]._value = min(self.f[n]._value, self.f[m].get())

    def _software_trigger(self):
        assert self.f['TriggerMode'].get() == 'On' and self.f['TriggerSource'].get() == 'Software'
        if self.streaming:
            self.deliver()

    # -- VmbPy Camera API ----------------------------------------------------
    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *a):
        self.entered -= 1

    def get_name(self):
        return self.name

    def get_id(self):
        return "DEV_0123"

    def get_streams(self):
        return []

    def get_feature_by_name(self, name):
        try:
            return self.f[name]
        except KeyError:
            raise FakeFeatureError(f"Feature '{name}' not found.")

    def get_pixel_formats(self):
        return tuple(self.formats)

    def get_pixel_format(self):
        return self.pixel_format

    def set_pixel_format(self, fmt):
        if fmt not in self.formats:
            raise ValueError(f"Camera does not support PixelFormat '{fmt}'")
        self.pixel_format = fmt
        self.f['PixelFormat']._value = str(fmt)

    def start_streaming(self, handler, buffer_count=5, allocation_mode=None):
        if self.streaming:
            raise FakeCameraError("already streaming")
        self.handler = handler
        self.streaming = True

    def stop_streaming(self):
        self.streaming = False
        self.handler = None

    def is_streaming(self):
        return self.streaming

    def queue_frame(self, frame):
        self.queued.append(frame)

    # test helper: the SDK delivers one frame
    def deliver(self, status=FakeFrameStatus.Complete):
        self.frame_counter += 1
        frame = FakeFrame(self, self.frame_counter, status)
        self.handler(self, None, frame)
        return frame


class FakeVmbSystem:
    instance = None
    cameras = []

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def get_all_cameras(self):
        return list(self.cameras)

    def get_camera_by_id(self, cam_id):
        for c in self.cameras:
            if c.get_id() == cam_id:
                return c
        raise FakeCameraError(cam_id)


@pytest.fixture
def avcamera(monkeypatch):
    """Import avcamera against the fake SDK; yields (module, install_camera)."""
    fake = types.ModuleType("vmbpy")
    fake.VmbSystem = FakeVmbSystem
    fake.FrameStatus = FakeFrameStatus
    fake.PixelFormat = FakePixelFormat
    fake.AllocationMode = FakeAllocationMode
    fake.VmbCameraError = FakeCameraError
    fake.VmbFeatureError = FakeFeatureError
    monkeypatch.setitem(sys.modules, "vmbpy", fake)
    modname = "imswitch.imcontrol.model.interfaces.avcamera"
    sys.modules.pop(modname, None)
    module = importlib.import_module(modname)
    assert module.isVmbPy
    FakeVmbSystem.instance = None

    def install(*cams):
        FakeVmbSystem.cameras = list(cams)
        return cams[0] if cams else None

    yield module, install
    FakeVmbSystem.cameras = []
    sys.modules.pop(modname, None)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class TestCameraAVInterface:
    def test_mono_camera_negotiates_highest_unpacked_bit_depth(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        assert cam.isRGB is False
        assert fake.pixel_format == FakePixelFormat.Mono12
        assert cam.bitDepth == 12
        assert (cam.SensorWidth, cam.SensorHeight) == (2592, 1944)
        assert fake.entered == 1
        cam.close()
        assert fake.entered == 0

    def test_explicit_pixel_format_wins(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0, pixel_format="Mono8")
        assert fake.pixel_format == FakePixelFormat.Mono8
        assert cam.bitDepth == 8

    def test_color_camera_is_detected_and_bayer_is_demosaiced(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera(color=True))
        cam = mod.CameraAV(0)
        assert cam.isRGB is True
        assert fake.pixel_format == FakePixelFormat.BayerRG8
        cam.start_live()
        fake.deliver()
        frame = cam.getLast()
        assert frame.shape == (1944, 2592, 3)
        assert frame.dtype == np.uint8

    def test_mono_frames_are_squeezed_and_owned(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        cam.start_live()
        sdk_frame = fake.deliver()
        frame, fid = cam.getLast(returnFrameNumber=True)
        assert frame.shape == (1944, 2592)
        assert frame.dtype == np.uint16
        assert fid == 1 and cam.getFrameNumber() == 1
        assert not np.shares_memory(frame, sdk_frame.data)
        assert fake.queued == [sdk_frame]  # buffer handed back to the SDK

    def test_incomplete_frames_are_dropped(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        cam.start_live()
        fake.deliver(status=FakeFrameStatus.Incomplete)
        assert cam.getLast(timeout=0.01) is None
        assert cam.getStreamDiagnostics()["incomplete_frames"] == 1

    def test_flip_is_applied_per_frame(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0, flipImage=(True, False))
        cam.start_live()
        sdk_frame = fake.deliver()
        expected = np.flip(sdk_frame.data[:, :, 0], axis=0)
        np.testing.assert_array_equal(cam.getLast(), expected)
        cam.flipImage = (False, True)
        sdk_frame = fake.deliver()
        np.testing.assert_array_equal(cam.getLast(), np.flip(sdk_frame.data[:, :, 0], axis=1))

    def test_chunk_returns_frames_and_ids_and_empties_buffer(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera(sensor=(64, 32)))
        cam = mod.CameraAV(0)
        cam.start_live()
        for _ in range(5):
            fake.deliver()
        frames, ids = cam.getLastChunk()
        assert frames.shape == (3, 32, 64)  # ring buffer keeps the newest 3
        assert list(ids) == [3, 4, 5]
        assert len(cam.frame_buffer) == 0

    def test_binning_shrinks_the_full_frame(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0, binning=2)
        assert cam.binning == 2
        assert (cam.SensorWidth, cam.SensorHeight) == (1296, 972)
        assert fake.f['Width'].get() == 1296 and fake.f['Height'].get() == 972
        assert cam.getBinningRange() == (1, 4)
        # back to 1 restores the full sensor
        cam.setBinning(1)
        assert (cam.SensorWidth, cam.SensorHeight) == (2592, 1944)
        assert fake.f['Width'].get() == 2592

    def test_binning_is_clamped_to_camera_range(self, avcamera):
        mod, install = avcamera
        install(FakeCamera(binning_max=2))
        cam = mod.CameraAV(0)
        assert cam.setBinning(8) == 2

    def test_camera_without_binning_stays_at_one(self, avcamera):
        mod, install = avcamera
        install(FakeCamera(with_binning=False))
        cam = mod.CameraAV(0, binning=4)
        assert cam.binning == 1
        assert cam.getBinningRange() == (1, 1)

    def test_hardware_roi_is_aligned_and_reported(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        applied = cam.setROI(hpos=101, vpos=51, hsize=1003, vsize=501)
        assert applied == (100, 50, 1000, 500)  # OffsetX inc 2, Width inc 8
        assert fake.f['Width'].get() == 1000 and fake.f['OffsetX'].get() == 100
        assert cam._softwareROI is False
        cam.start_live()
        fake.deliver()
        assert cam.getLast().shape == (500, 1000)
        # growing back to full frame works (offset reset before width)
        assert cam.setROI(0, 0, 2592, 1944) == (0, 0, 2592, 1944)

    def test_roi_falls_back_to_software_crop_when_camera_refuses(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera(sensor=(64, 32)))
        del fake.f['OffsetX']
        cam = mod.CameraAV(0)
        applied = cam.setROI(4, 2, 16, 8)
        assert applied == (4, 2, 16, 8)
        assert cam._softwareROI is True
        cam.start_live()
        fake.deliver()
        assert cam.getLast().shape == (8, 16)

    def test_exposure_is_ms_on_the_outside_and_us_on_the_camera(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        cam.setPropertyValue("exposure", 12.5)
        assert fake.f['ExposureTime'].get() == 12500.0
        assert cam.getPropertyValue("exposure") == 12.5
        assert cam.get_exposuretime() == (12500.0, 21.0, 1e7)
        # out of range is clamped, not rejected
        cam.set_exposure_time(0)
        assert fake.f['ExposureTime'].get() == 21.0

    def test_exposure_mode_maps_to_exposure_auto(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        cam.setPropertyValue("exposure_mode", "auto")
        assert fake.f['ExposureAuto'].get() == 'Continuous'
        assert cam.getPropertyValue("exposure_mode") == 'auto'
        cam.setPropertyValue("exposure_mode", "once")
        assert fake.f['ExposureAuto'].get() == 'Once'
        assert cam.getPropertyValue("exposure_mode") == 'manual'  # transient, like hik
        cam.setPropertyValue("exposure_mode", "manual")
        assert fake.f['ExposureAuto'].get() == 'Off'

    def test_gain_and_blacklevel(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        cam.setPropertyValue("gain", 30)  # above max -> clamped
        assert fake.f['Gain'].get() == 24.0
        assert cam.get_gain() == (24.0, 0.0, 24.0)
        cam.setPropertyValue("blacklevel", 12)
        assert cam.getPropertyValue("blacklevel") == 12.0

    def test_frame_rate_cap_and_free_run(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0, frame_rate=-1)
        assert fake.f['AcquisitionFrameRateEnable'].get() is False
        assert cam.getPropertyValue("frame_rate") == -1
        # strings arrive from the REST API
        assert cam.setPropertyValue("frame_rate", "25") == 25.0
        assert fake.f['AcquisitionFrameRateEnable'].get() is True
        assert fake.f['AcquisitionFrameRate'].get() == 25.0
        assert cam.getPropertyValue("frame_rate") == 25.0
        # above the range -> clamped and the applied value is reported
        assert cam.set_frame_rate(500) == 100.0
        assert cam.set_frame_rate(0) == -1
        assert fake.f['AcquisitionFrameRateEnable'].get() is False

    def test_trigger_modes(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        assert fake.f['TriggerMode'].get() == 'Off'
        assert cam.getTriggerSource() == "Continuous (Free Run)"
        assert cam.setTriggerSource("Internal trigger") is True
        assert fake.f['TriggerMode'].get() == 'On'
        assert fake.f['TriggerSource'].get() == 'Software'
        assert cam.getTriggerSource() == "Software Trigger"
        assert cam.setTriggerSource("External trigger") is True
        assert fake.f['TriggerSource'].get() == 'Line0'
        assert cam.setTriggerSource("line1") is True
        assert fake.f['TriggerSource'].get() == 'Line1'
        assert cam.setTriggerSource("Continous") is True
        assert fake.f['TriggerMode'].get() == 'Off'
        assert cam.setTriggerSource("bogus") is False

    def test_trigger_change_restarts_a_running_stream(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera())
        cam = mod.CameraAV(0)
        cam.start_live()
        cam.setTriggerSource("software")
        assert fake.streaming is True and cam.is_streaming is True

    def test_snap_software_trigger_returns_the_triggered_frame(self, avcamera):
        mod, install = avcamera
        fake = install(FakeCamera(sensor=(64, 32)))
        cam = mod.CameraAV(0)
        cam.start_live()
        fake.deliver()  # a stale free-run frame
        cam.setTriggerSource("software")
        frame = cam.snapSoftwareTrigger(timeout=0.5)
        assert frame is not None and frame.shape == (32, 64)
        assert cam.getFrameNumber() == 2
        assert fake.frame_counter == 2  # exactly one exposure fired

    def test_get_last_waits_then_returns_none(self, avcamera):
        mod, install = avcamera
        install(FakeCamera())
        cam = mod.CameraAV(0)
        assert cam.getLast(timeout=0.02) is None
        assert cam.getLast(returnFrameNumber=True, timeout=0.02) == (None, None)

    def test_camera_by_id_and_bad_index(self, avcamera):
        mod, install = avcamera
        install(FakeCamera(), FakeCamera(name="second"))
        cam = mod.CameraAV("DEV_0123")
        assert cam.model == "Alvium 1800 U-500m"
        assert mod.CameraAV("1").model == "second"
        with pytest.raises(RuntimeError):
            mod.CameraAV(5)

    def test_status_snapshot(self, avcamera):
        mod, install = avcamera
        install(FakeCamera())
        cam = mod.CameraAV(0, binning=2)
        params = cam.get_camera_parameters()
        assert params["binning"] == 2
        assert params["exposure_min"] == 21.0
        assert params["DeviceSerialNumber"] == "0123"
        diag = cam.getDiagnostics()
        assert diag["pixel_format"] == "Mono12" and diag["bit_depth"] == 12


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
def _detector_info(**props):
    base = {"cameraListIndex": 0, "avcam": {"exposure": 10, "gain": 2}}
    base.update(props)
    return SimpleNamespace(managerProperties=base, forAcquisition=True, forFocusLock=False)


@pytest.fixture
def make_manager(avcamera):
    from imswitch.imcontrol.model.managers.detectors.AVManager import AVManager
    mod, install = avcamera

    def build(fake_camera=None, **props):
        fake = install(fake_camera or FakeCamera())
        return AVManager(_detector_info(**props), "AVCam"), fake

    return build


class TestAVManager:
    def test_parameters_are_seeded_from_hardware(self, make_manager):
        mgr, fake = make_manager()
        p = mgr.parameters
        assert p['exposure'].value == 10.0
        assert (p['exposure'].valueMin, p['exposure'].valueMax) == (0.021, 10000.0)
        assert p['gain'].value == 2.0
        assert (p['gain'].valueMin, p['gain'].valueMax) == (0.0, 24.0)
        assert p['frame_rate'].value == -1
        assert p['exposure_mode'].value == 'manual'
        assert p['previewMaxValue'].value == 4095  # Mono12
        assert mgr.fullShape == (2592, 1944)
        assert mgr.shape == (2592, 1944)
        assert mgr.isRGB is False
        assert mgr.supportedBinnings == [1, 2, 4]

    def test_binning_follows_the_camera(self, make_manager):
        mgr, fake = make_manager()
        mgr.startAcquisition()
        mgr.setBinning(2)
        assert mgr.binning == 2
        assert fake.f['BinningHorizontal'].get() == 2
        assert mgr.fullShape == (1296, 972)
        assert mgr.shape == (1296, 972)
        assert mgr._running is True and fake.streaming is True  # stream restarted

    def test_startup_binning_from_config(self, make_manager):
        mgr, fake = make_manager(binning=4)
        assert mgr.binning == 4
        assert mgr.fullShape == (648, 486)
        assert mgr.supportedBinnings == [1, 2, 4]
        # applied once by the camera constructor, not re-written by the manager
        assert fake.f['BinningHorizontal'].writes == [4]

    def test_crop_applies_hardware_roi(self, make_manager):
        mgr, fake = make_manager()
        mgr.crop(101, 51, 1003, 501)
        assert mgr.shape == (1000, 500)
        assert mgr.frameStart == (100, 50)
        assert fake.f['Width'].get() == 1000

    def test_exposure_mode_and_frame_rate_reach_the_camera(self, make_manager):
        mgr, fake = make_manager()
        mgr.setParameter('exposure_mode', 'auto')
        assert fake.f['ExposureAuto'].get() == 'Continuous'
        assert mgr.getParameter('exposure_mode') == 'auto'
        assert mgr.setParameter('frame_rate', "500") == 100.0
        assert mgr.parameters['frame_rate'].value == 100.0
        mgr.setParameter('exposure', 5)
        assert fake.f['ExposureTime'].get() == 5000.0
        assert mgr.getParameter('exposure') == 5.0

    def test_bookkeeping_parameters_do_not_hit_the_camera(self, make_manager):
        mgr, fake = make_manager()
        mgr.setParameter('Camera pixel size', 0.3)
        assert mgr.pixelSizeUm == [1, 0.3, 0.3]
        mgr.setPixelSizeUm(0.5)
        assert mgr.getParameter('Camera pixel size') == 0.5
        mgr.setParameter('previewMaxValue', 1000)
        assert mgr.parameters['previewMaxValue'].value == 1000

    def test_refresh_reads_back_hardware_values(self, make_manager):
        mgr, fake = make_manager()
        fake.f['ExposureTime']._value = 33000.0  # changed by auto exposure
        params = mgr.refreshParameters()
        assert params['exposure'].value == 33.0

    def test_trigger_and_snap_sync(self, make_manager):
        mgr, fake = make_manager(FakeCamera(sensor=(64, 32)))
        mgr.startAcquisition()
        assert mgr.setTriggerSource("software") is True
        assert mgr.parameters['trigger_source'].value == "software"
        assert mgr.getCurrentTriggerType() == "Software Trigger"
        assert "Software Trigger" in mgr.getTriggerTypes()
        frame = mgr.snapSync(timeout=0.5)
        assert frame.shape == (32, 64)
        assert mgr.getFrameNumber() == 1
        mgr.sendSoftwareTrigger()
        assert mgr.getFrameNumber() == 2

    def test_flip_and_latest_frame(self, make_manager):
        mgr, fake = make_manager(FakeCamera(sensor=(64, 32)))
        mgr.setFlipImage(True, False)
        mgr.startAcquisition()
        sdk_frame = fake.deliver()
        frame, fid = mgr.getLatestFrame(returnFrameNumber=True)
        np.testing.assert_array_equal(frame, np.flip(sdk_frame.data[:, :, 0], axis=0))
        assert fid == 1
        frames, ids = mgr.getChunk()
        assert frames.shape == (1, 32, 64) and list(ids) == [1]

    def test_rgb_auto_detection_reaches_the_manager(self, make_manager):
        mgr, fake = make_manager(FakeCamera(color=True))
        assert mgr.isRGB is True
        assert mgr.parameters['previewMaxValue'].value == 255

    def test_status_and_stream_diagnostics(self, make_manager):
        mgr, fake = make_manager()
        status = mgr.getCameraStatus()
        assert status['cameraType'] == 'AlliedVision'
        assert status['isMock'] is False and status['isConnected'] is True
        assert status['hardwareParameters']['model_name'] == "Alvium 1800 U-500m"
        assert status['currentTriggerSource'] == "Continuous (Free Run)"
        assert mgr.getStreamDiagnostics()['is_streaming'] is False

    def test_finalize_closes_the_camera(self, make_manager):
        mgr, fake = make_manager()
        mgr.startAcquisition()
        mgr.finalize()
        assert fake.streaming is False and fake.entered == 0
