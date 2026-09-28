"""
Tests for the ToupTek low-noise / long-exposure configuration:
window heater, conversion gain (HCG), low-noise readout and black level.

They run against a fake SDK handle, so no camera and no native library call is
involved. ``CameraToupcam`` is built with ``object.__new__`` because its
``__init__`` opens a device; the attributes set here are the ones the tested
methods touch.
"""

from __future__ import annotations

import pytest


toupcam = pytest.importorskip(
    "imswitch.imcontrol.model.interfaces.toupcamsdk.toupcam")

from imswitch.imcontrol.model.interfaces.toupcamcamera import (  # noqa: E402
    CameraToupcam,
)


class FakeHcam:
    """Records put_Option writes and serves get_Option from the same store."""

    def __init__(self, options=None, unsupported=()):
        self.options = {
            toupcam.TOUPCAM_OPTION_HEAT_MAX: 5,
            toupcam.TOUPCAM_OPTION_HEAT: 0,
            toupcam.TOUPCAM_OPTION_LOW_NOISE: 0,
            toupcam.TOUPCAM_OPTION_CG: 0,
            toupcam.TOUPCAM_OPTION_BLACKLEVEL: 0,
            toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST: 1,
        }
        if options:
            self.options.update(options)
        # Options this model does not implement -- the real SDK raises for these.
        self.unsupported = set(unsupported)
        self.writes = []

    def put_Option(self, option, value):
        if option in self.unsupported:
            raise toupcam.HRESULTException(0x80004001)  # E_NOTIMPL
        self.writes.append((option, value))
        self.options[option] = value

    def get_Option(self, option):
        if option in self.unsupported:
            raise toupcam.HRESULTException(0x80004001)
        return self.options[option]


class _SilentLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)

    def info(self, msg, *a, **k):
        self.infos.append(msg)

    def debug(self, *a, **k):
        pass

    def error(self, msg, *a, **k):
        self.warnings.append(msg)


def _make_camera(hcam=None, *, bits=16, maxBitDepth=14, heat=True,
                 lowNoise=True, cg=True, cghdr=False, blacklevel=True):
    cam = object.__new__(CameraToupcam)
    cam.hcam = hcam if hcam is not None else FakeHcam()
    cam._CameraToupcam__logger = _SilentLogger()
    cam.deviceName = "FakeToupcam"
    cam.is_streaming = False
    cam.isRGB = False
    cam._bits = bits
    cam._maxBitDepth = maxBitDepth
    cam._hasHeat = heat
    cam._hasLowNoise = lowNoise
    cam._hasCG = cg
    cam._hasCGHDR = cghdr
    cam._hasBlacklevel = blacklevel
    cam._heatMax = 5
    cam.blacklevel = 0
    cam.heat = None
    cam.lowNoise = None
    cam.conversionGain = None
    cam.blacklevelAutoAdjust = None
    return cam


class TestLowNoiseConfiguration:
    def test_hcg_low_noise_and_heater_are_applied(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam)

        cam.set_conversion_gain("HCG")
        cam.set_low_noise(True)
        cam.set_heat(True)

        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_LOW_NOISE] == 1
        # heat=True means the camera's maximum level, not "1".
        assert hcam.options[toupcam.TOUPCAM_OPTION_HEAT] == 5
        assert cam.get_conversion_gain() == "HCG"
        assert cam.get_low_noise() is True
        assert cam.get_heat() == 5

    def test_conversion_gain_accepts_names_and_numbers(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam)

        cam.set_conversion_gain("lcg")
        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 0
        cam.set_conversion_gain(1)
        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1

    def test_hdr_falls_back_to_hcg_without_the_flag(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam, cghdr=False)

        cam.set_conversion_gain("HDR")

        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1
        assert any("HDR" in w for w in cam._CameraToupcam__logger.warnings)

    def test_unsupported_capabilities_are_skipped_not_written(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam, heat=False, lowNoise=False, cg=False)

        cam.set_conversion_gain("HCG")
        cam.set_low_noise(True)
        cam.set_heat(True)

        # Nothing written, nothing raised: the camera simply has no such mode.
        assert hcam.writes == []
        assert cam.get_conversion_gain() is None
        assert cam.get_low_noise() is None
        assert cam.get_heat() is None

    def test_heat_level_is_clamped_to_the_camera_maximum(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam)

        cam.set_heat(99)

        assert hcam.options[toupcam.TOUPCAM_OPTION_HEAT] == 5

    def test_settings_are_reapplied_after_a_reconnect(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam)
        cam.exposure_time = 0
        cam.gain = 0
        cam.binning = 1
        cam.frame_rate = -1
        cam.trigger_source = "Continous"
        cam.conversionGain = "HCG"
        cam.lowNoise = True
        cam.heat = True
        cam.set_frame_rate = lambda *a, **k: None
        cam.setTriggerSource = lambda *a, **k: None

        cam._reapply_settings()

        # A reopened handle comes up with the camera defaults, so all three
        # have to be pushed again.
        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_LOW_NOISE] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_HEAT] == 5


class TestBlackLevel:
    def test_maximum_follows_the_output_bit_depth(self):
        assert _make_camera(bits=8).blacklevel_max() == 31
        assert _make_camera(bits=16, maxBitDepth=12).blacklevel_max() == 31 * 16
        assert _make_camera(bits=16, maxBitDepth=14).blacklevel_max() == 31 * 64
        assert _make_camera(bits=16, maxBitDepth=16).blacklevel_max() == 31 * 256

    def test_out_of_range_value_is_clamped_and_reported(self):
        hcam = FakeHcam()
        # 8-bit output: the maximum is 31, so a requested 100 cannot show up.
        cam = _make_camera(hcam, bits=8)

        cam.set_blacklevel(100)

        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL] == 31
        assert cam.blacklevel == 31
        assert any("outside 0..31" in w for w in cam._CameraToupcam__logger.warnings)

    def test_auto_adjust_is_turned_off_so_the_value_takes_effect(self):
        # Auto-adjust on: the camera keeps re-deriving the offset from its
        # optical-black pixels and overwrites the manual value.
        hcam = FakeHcam({toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST: 1})
        cam = _make_camera(hcam)

        cam.set_blacklevel(100)

        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST] == 0
        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL] == 100
        # ...and the disable happens before the write, or the camera would
        # re-adjust in between.
        order = [opt for opt, _ in hcam.writes]
        assert order.index(toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST) < \
            order.index(toupcam.TOUPCAM_OPTION_BLACKLEVEL)

    def test_in_range_value_is_written_unchanged_at_high_bit_depth(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam, bits=16, maxBitDepth=14)   # max 1984

        cam.set_blacklevel(100)

        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL] == 100
        assert cam.blacklevel == 100
        assert not any("outside" in w for w in cam._CameraToupcam__logger.warnings)

    def test_models_without_the_auto_adjust_option_still_work(self):
        hcam = FakeHcam(unsupported={toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST})
        cam = _make_camera(hcam)

        cam.set_blacklevel(100)

        assert cam.get_blacklevel_autoadjust() is None
        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL] == 100

    def test_read_back_mismatch_is_reported(self):
        class StickyHcam(FakeHcam):
            def put_Option(self, option, value):
                # Camera accepts the write but keeps its own value -- exactly
                # the silent failure this read-back exists to catch.
                if option == toupcam.TOUPCAM_OPTION_BLACKLEVEL:
                    self.writes.append((option, value))
                    return
                super().put_Option(option, value)

        hcam = StickyHcam()
        cam = _make_camera(hcam)

        cam.set_blacklevel(100)

        assert any("reads back" in w for w in cam._CameraToupcam__logger.warnings)
