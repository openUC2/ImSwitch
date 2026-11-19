"""LED geometry -> k-vectors, LED/camera calibration and LED-source DPC (no hardware)."""
import asyncio
import csv
import os
import types

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

from imswitch.imcontrol.model.managers.LEDMatrixs import ledgeometry as lg
from imswitch.imcontrol.controller.controllers.DPCController import (
    DPCController, DPCSolver, DPC_PATTERNS)


def test_geodome_na_matches_csv():
    pos = lg.load_layout("geodome80")
    assert pos.shape == (80, 3)
    with open(os.path.join(lg.LAYOUT_DIR, "geodome80.csv")) as f:
        ref = np.array([[float(r["NAx"]), float(r["NAy"])] for r in csv.DictReader(f)])
    np.testing.assert_allclose(lg.illumination_na(pos), ref, atol=2e-4)


def test_ring_and_grid_layouts():
    ring = lg.load_layout("illumination_ring136")
    na = lg.illumination_na(ring, {"offset_mm": [0, 0, 40]})
    assert ring.shape == (136, 3) and np.hypot(*na.T).max() < 0.5
    grid = lg.load_layout({"grid": [3, 2], "pitch_mm": 2, "serpentine": True})
    np.testing.assert_allclose(grid[:, :2], [[-2, -1], [0, -1], [2, -1], [2, 1], [0, 1], [-2, 1]])


def test_calibration_rotates_and_mirrors():
    pos = np.array([[-10.0, 0, 50]])  # LED at -x lights towards +x
    assert lg.illumination_na(pos)[0, 0] > 0
    na = lg.illumination_na(pos, {"rotation_deg": 90})[0]
    np.testing.assert_allclose(na, [0, 0.196], atol=1e-3)
    na = lg.illumination_na(pos, {"flip": True, "rotation_deg": 90})[0]
    np.testing.assert_allclose(na, [0, -0.196], atol=1e-3)


def test_image_shift_sign():
    rng = np.random.default_rng(0)
    img = gaussian_filter(rng.random((300, 400)), 2)
    moved = nd_shift(img, (-3, 5), mode="wrap")  # rows -3, columns +5
    dx, dy = lg.image_shift(img, moved)
    assert abs(dx - 5) < 0.3 and abs(dy + 3) < 0.3


def test_orientation_from_defocus_shifts():
    """Synthetic defocused sample: every brightfield LED shifts the texture by a*R*F*tan(theta)."""
    pos = lg.load_layout("geodome80")
    na_led = lg.illumination_na(pos)
    use = np.hypot(*na_led.T) < 0.35
    truth = {"rotation_deg": 37.0, "flip": True}
    na_cam = lg.illumination_na(pos, truth)[use]
    t = na_cam / np.sqrt(1 - np.sum(na_cam ** 2, axis=1, keepdims=True))
    rng = np.random.default_rng(1)
    texture = gaussian_filter(rng.random((256, 256)), 1.5)
    stack = [nd_shift(texture, (40 * ty + 2, 40 * tx - 1), mode="wrap") for tx, ty in t]
    ref = stack[int(np.argmin(np.hypot(*na_led[use].T)))]
    shifts = [lg.image_shift(ref, im) for im in stack]
    fit = lg.fit_orientation(na_led[use], shifts)
    assert fit["flip"] is True and abs(fit["rotation_deg"] - 37) < 1.5 and fit["rms_px"] < 0.5


def test_brightfield_split():
    rng = np.random.default_rng(2)
    bright = [0.8 + 0.1 * rng.random((50, 50)) for _ in range(5)]
    dark = [0.02 + 0.5 * (rng.random((50, 50)) > 0.995) for _ in range(20)]
    bf = lg.split_brightfield(lg.brightfield_scores(bright + dark))
    assert bf[:5].all() and not bf[5:].any()


def test_dpc_recovers_phase_from_simulated_led_images():
    """Partially coherent forward model with the dome LEDs (one plane wave each) -> qDPC."""
    shape, px, wl, na_obj = (128, 128), 0.5, 0.53, 0.3
    rng = np.random.default_rng(3)
    phase = gaussian_filter(rng.standard_normal(shape), 3)
    phase = 0.1 * (phase - phase.mean()) / phase.std()
    na = lg.illumination_na(lg.load_layout("geodome80"), {"rotation_deg": 20.0})
    patterns = lg.dpc_patterns(na, na_obj)
    fy = np.fft.fftfreq(shape[0], px)[:, None]
    fx = np.fft.fftfreq(shape[1], px)[None, :]
    pupil = fx ** 2 + fy ** 2 <= (na_obj / wl) ** 2
    yy, xx = np.mgrid[:shape[0], :shape[1]] * px
    obj = np.exp(1j * phase)

    def image(nas):
        total = 0
        for nax, nay in nas:  # plane waves snapped to the FFT grid like the solver's source
            u = np.round(nax / wl * shape[1] * px) / (shape[1] * px)
            v = np.round(nay / wl * shape[0] * px) / (shape[0] * px)
            field = obj * np.exp(2j * np.pi * (u * xx + v * yy))
            total = total + np.abs(np.fft.ifft2(np.fft.fft2(field) * pupil)) ** 2
        return total

    source_na = [na[patterns[name]] for name in DPC_PATTERNS]
    stack = np.array([image(s) for s in source_na])
    solver = DPCSolver(shape, wl, na_obj, px, source_na)
    solver.setTikhonovRegularization(reg_u=1e-1, reg_p=5e-3)
    rec = solver.solve(stack)[0].imag
    # LEDs well inside the objective NA hardly transfer low frequencies, so compare with the
    # phase seen through the solver's Tikhonov filter (sign-blind: a wrong source sign or
    # axis shows up as a negative / zero correlation)
    hp2 = np.sum(np.abs(solver.Hp) ** 2, axis=0)
    expected = np.fft.ifft2(np.fft.fft2(phase) * hp2 / (hp2 + 5e-3)).real
    assert np.corrcoef(rec.ravel(), expected.ravel())[0, 1] > 0.9


def test_dpc_stream_sends_latest_phase_to_every_client():
    c = types.SimpleNamespace(_mjpeg_frame=b"\xff\xd8 phase jpeg", _jpeg_quality=85)

    async def first_two(resp):
        it = resp.body_iterator.__aiter__()
        return await it.__anext__(), await asyncio.wait_for(it.__anext__(), 1.0)

    for _ in range(2):  # a reconnecting page must not find the frame already taken
        resp = DPCController.mjpeg_stream_dpc(c)
        assert resp.media_type == "multipart/x-mixed-replace;boundary=frame"
        part, idle = asyncio.run(first_two(resp))
        jpeg = c._mjpeg_frame
        assert part.endswith(b"Content-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n")
        assert idle == b""  # idle loop yields, so a closed connection ends its generator


def test_darkfield_leds_do_not_alias_into_the_wotf():
    # 1 um pixels at 0.53 um: the grid ends at NA 0.265, an LED at NA 0.5 would wrap to NA -0.03
    bright, dark = np.array([[0.15, 0.05]]), np.array([[0.5, 0.0]])
    clean = DPCSolver((64, 64), 0.53, 0.25, 1.0, [bright] * 4)
    mixed = DPCSolver((64, 64), 0.53, 0.25, 1.0, [np.vstack([bright, dark])] * 4)
    np.testing.assert_allclose(mixed.Hp, clean.Hp)
    np.testing.assert_allclose(mixed.Hu, clean.Hu)
