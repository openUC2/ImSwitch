"""SLM / DMD display: patterns, the camera->display mapping, the framebuffer packing
and the window protocol framing. No display, no camera."""
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from imswitch.imcontrol.model.managers import slm_display
from imswitch.imcontrol.model.managers.SLMManager import (
    SLMManager, calibration_points, fit_frame, make_pattern, spot_centroid)


@pytest.fixture
def slm(tmp_path, monkeypatch):
    from imswitch.imcommon.model import dirtools
    monkeypatch.setattr(dirtools.UserFileDirs, "Root", str(tmp_path))
    info = SimpleNamespace(display="virtual", displayIndex=0, width=200, height=100,
                           settleTimeMs=0, calibrationDetector=None)
    return SLMManager(info, name="TestSLM")


def test_patterns_have_display_shape_and_range():
    for kind in ("blank", "checkerboard", "grating", "disk", "grid"):
        img = make_pattern(kind, 64, 32, value=200, period=8, radius=5)
        assert img.shape == (32, 64) and img.dtype == np.uint8
        assert img.max() <= 200
    assert make_pattern("blank", 8, 8, value=7).min() == 7
    g = make_pattern("grating", 64, 1, period=8, duty=0.5)[0]
    assert g[:4].min() == 255 and g[4:8].max() == 0  # one period = 4 on, 4 off
    with pytest.raises(ValueError):
        make_pattern("nope", 8, 8)


def test_fit_frame_center_keeps_pixels_and_stretch_resizes():
    small = np.full((10, 20), 255, np.uint8)
    centered = fit_frame(small, (40, 20), "center")
    assert centered.shape == (20, 40) and centered[5:15, 10:30].min() == 255
    assert centered[0].max() == 0
    assert fit_frame(small, (40, 20), "stretch").min() == 255
    big = np.arange(50 * 60, dtype=np.uint8).reshape(50, 60)
    assert fit_frame(big, (20, 10), "center").shape == (10, 20)


def test_virtual_show_and_affine_roundtrip(slm):
    seq = slm.show_pattern("disk", cx=50, cy=20, radius=5)
    assert seq == 2 and slm.current.shape == (100, 200)  # init blank was seq 1
    assert slm.current[20, 50] == 255
    # calibration: camera is a scaled, shifted, flipped view of the display
    disp = calibration_points(5, slm.size)
    cam = [(2 * x + 30, -2 * y + 500) for x, y in disp]
    result = slm.compute_affine(cam, disp)
    assert result["calibrated"] and result["calibration"]["residual_px"] < 1e-6
    x, y = slm.camera_to_display(2 * 70 + 30, -2 * 40 + 500)
    assert (round(x), round(y)) == (70, 40)
    cx, cy = slm.display_to_camera(70, 40)
    assert (round(cx), round(cy)) == (170, 420)
    # persisted and reloaded
    again = SLMManager(slm.info, name="TestSLM")
    assert again.affine.to_dict() == slm.affine.to_dict()


def test_camera_mask_lands_at_the_calibrated_display_spot(slm):
    disp = calibration_points(4, slm.size)
    cam = [(2 * x, 2 * y) for x, y in disp]          # camera = 2x display
    slm.compute_affine(cam, disp)
    mask = np.zeros((50, 100), np.uint8)              # mask at 1/4 camera resolution
    # camera (120..128, 40..48) -> display (60..64, 20..24)
    mask[10:12, 30:32] = 255
    frame = slm.camera_mask_to_display(mask, (400, 200), value=200)
    ys, xs = np.nonzero(frame)
    assert frame.max() == 200
    assert 60 <= xs.min() and xs.max() < 64 and 20 <= ys.min() and ys.max() < 24
    inverted = slm.camera_mask_to_display(mask, (400, 200), invert=True)
    assert inverted[22, 62] == 0 and inverted[0, 0] == 255


def test_spot_centroid_finds_the_spot_and_rejects_noise():
    rng = np.random.default_rng(0)
    dark = rng.normal(100, 2, (120, 160)).astype(np.float32)
    frame = dark.copy()
    yy, xx = np.mgrid[0:120, 0:160]
    frame += 800 * np.exp(-((xx - 100.3) ** 2 + (yy - 40.6) ** 2) / (2 * 3 ** 2))
    x, y, snr = spot_centroid(frame, dark)
    assert abs(x - 100.3) < 0.5 and abs(y - 40.6) < 0.5 and snr > 5
    assert spot_centroid(dark, dark) is None


def test_sequences_roundtrip_on_disk(slm):
    frames = [("a.png", np.full((100, 200), 10, np.uint8)),
              ("b", np.full((100, 200), 20, np.uint8))]
    info = slm.save_sequence("My Seq!", frames)
    assert info["name"] == "My_Seq" and info["count"] == 2 and info["size"] == [200, 100]
    assert [n for n, _ in slm.load_sequence("My_Seq")] == ["a", "b"]
    assert slm.load_sequence("My_Seq")[1][1][0, 0] == 20
    assert [s["name"] for s in slm.list_sequences()] == ["My_Seq"]
    slm.delete_sequence("My_Seq")
    assert slm.list_sequences() == []


def test_framebuffer_packing_rgb565_and_xrgb8888():
    fb = slm_display.FramebufferDisplay.__new__(slm_display.FramebufferDisplay)
    fb.size, fb.bpp, fb.stride = (2, 1), 16, 8          # stride wider than the row
    rows = fb.pack(np.array([[255, 0]], np.uint8))
    assert rows.shape == (1, 8)
    assert struct.unpack("<HH", rows[0, :4].tobytes()) == (0xFFFF, 0)
    fb.bpp, fb.stride = 32, 8
    px = fb.pack(np.array([[[255, 128, 0]]], np.uint8))  # R, G, B -> B, G, R, X
    assert px[0, :4].tolist() == [0, 128, 255, 255]


def test_window_protocol_framing(monkeypatch):
    """The parent writes 'frame W H C SEQ' + bytes and expects 'shown SEQ'."""
    import io
    sent = io.BytesIO()
    wd = slm_display.WindowDisplay.__new__(slm_display.WindowDisplay)
    wd.proc = SimpleNamespace(stdin=sent, poll=lambda: None)
    wd._seq, wd.size = 0, (3, 2)
    wd._lock = __import__("threading").Lock()
    wd._readline = lambda: "shown 1"
    frame = np.arange(6, dtype=np.uint8).reshape(2, 3)
    wd.show(frame)
    assert sent.getvalue() == b"frame 3 2 1 1\n" + frame.tobytes()
    wd._readline = lambda: "error boom"
    with pytest.raises(RuntimeError):
        wd.show(frame)
