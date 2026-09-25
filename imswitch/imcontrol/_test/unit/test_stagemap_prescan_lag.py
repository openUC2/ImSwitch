"""Prescan lag calibration: two sweeps of one line, offset by 2 × speed × lag.

The camera delivers a frame some tens of milliseconds after it was exposed,
and the stage keeps moving meanwhile, so a band stamped with its grab time
lands ahead of where it was taken. Forward and return lines are shifted in
opposite directions — the alternating even/odd offset seen on the plate map.
"""

import logging
import types

import numpy as np

from imswitch.imcontrol.controller.controllers.StageMapController import (
    StageMapController, _Sweep, estimateProfileShift)

SCALE = 1.0        # µm per strip pixel
SPAN = 4000.0      # µm
SPEED = 1000.0     # µm/s
BAND = 60          # camera pixels kept per exposure
ROWS = 4


def world(rng, spanPx):
    """A line of tissue: sparse blobs on a bright background, with a margin so
    bands near the ends can be cut without wrapping."""
    x = np.arange(-BAND, spanPx + BAND, dtype=float)
    w = np.full_like(x, 200.0)
    for c in rng.uniform(0, spanPx, 25):
        w -= 150 * np.exp(-((x - c) ** 2) / (2 * rng.uniform(8, 40) ** 2))
    return np.clip(w, 0, 255).astype(np.uint16)


def sweep(w, forward, lagS, dt=0.05, t0=100.0):
    """What ``_sweepLine`` would record: bands grabbed every ``dt`` while the
    stage runs the line at SPEED, each showing where the stage was ``lagS``
    earlier. Camera pixel order is the same on both directions."""
    duration = SPAN / SPEED
    t1 = t0 + duration
    samples = []
    t = t0 + lagS
    while t <= t1 + 0.3:
        along = min(SPAN, max(0.0, SPEED * (t - lagS - t0)))
        x = along if forward else SPAN - along
        c = int(round(x / SCALE)) + BAND
        band = np.tile(w[c - BAND // 2:c + BAND // 2], (ROWS, 1))
        samples.append((t, band.copy()))
        t += dt
    return _Sweep(samples, t0, t1, SPAN, SCALE, forward)


def fakeController(lag=0.0):
    self = types.SimpleNamespace()
    self._logger = logging.getLogger("test")
    self._prescanLagS = lag
    self._shouldStop = types.SimpleNamespace(is_set=lambda: False)
    self._spreadSamples = types.MethodType(StageMapController._spreadSamples, self)
    self._placeSweep = types.MethodType(StageMapController._placeSweep, self)
    self._calibratePrescanLag = types.MethodType(StageMapController._calibratePrescanLag, self)
    return self


def test_profile_shift_recovers_a_known_offset_either_sign():
    rng = np.random.default_rng(1)
    w = world(rng, 3000)[None, :].astype(float)
    for s in (37, -52, 0):
        b = w[:, 100:2900]
        a = np.roll(w, s, axis=1)[:, 100:2900]
        assert estimateProfileShift(a, b) == s


def test_profile_shift_refuses_blank_glass():
    flat = np.full((4, 2000), 180.0)
    assert estimateProfileShift(flat, flat + 0.5) is None


def test_profile_shift_ignores_unfilled_strip_ends():
    rng = np.random.default_rng(2)
    w = world(rng, 3000)[None, :].astype(float)
    b = w[:, 100:2900].copy()
    a = np.roll(w, 40, axis=1)[:, 100:2900].copy()
    # The lag leaves the last columns dark in one strip and the first in the
    # other; that shared "edge" must not drag the estimate towards zero.
    a[:, -60:] = 0
    b[:, :60] = 0
    assert abs(estimateProfileShift(a, b) - 40) <= 1


def test_uncorrected_sweeps_land_2_speed_lag_apart_and_calibration_recovers_it():
    rng = np.random.default_rng(3)
    lag = 0.06
    w = world(rng, int(SPAN / SCALE))
    ctrl = fakeController()
    sweeps = {True: sweep(w, True, lag), False: sweep(w, False, lag)}
    ctrl._sweepLine = lambda startX, endX, y, speedX, dx, subsample: sweeps[endX > startX]

    a = ctrl._placeSweep(sweeps[True], 0.0)
    b = ctrl._placeSweep(sweeps[False], 0.0)
    expected = 2 * SPEED * lag / SCALE
    assert abs(estimateProfileShift(a, b) - expected) <= 2

    lagS, fwd = ctrl._calibratePrescanLag(0.0, SPAN, 0.0, SPEED, 300.0, 4)
    assert fwd is sweeps[True]
    assert abs(lagS - lag) < 0.002


def test_corrected_strips_agree_with_each_other_and_with_the_sample():
    rng = np.random.default_rng(4)
    lag = 0.045
    w = world(rng, int(SPAN / SCALE))
    ctrl = fakeController()
    a = ctrl._placeSweep(sweep(w, True, lag), lag)
    b = ctrl._placeSweep(sweep(w, False, lag), lag)
    assert abs(estimateProfileShift(a, b)) <= 1
    truth = np.tile(w[BAND:BAND + a.shape[1]], (ROWS, 1))
    assert abs(estimateProfileShift(a, truth)) <= 1
    assert abs(estimateProfileShift(b, truth)) <= 1


def test_return_line_bands_keep_camera_pixel_order():
    # A band with a gradient across it: pixel 0 must stay at low X on a return
    # line too; mirroring the strip would flip every field of view inside it.
    ramp = np.tile(np.arange(BAND, dtype=np.uint16), (ROWS, 1))
    t0, t1 = 0.0, 4.0
    samples = [(t0 + k * 0.5, ramp.copy()) for k in range(9)]
    ctrl = fakeController()
    strip = ctrl._spreadSamples(samples, t0, t1, SPAN, SCALE, forward=False)
    filled = np.flatnonzero(strip[0] > 0)
    x = filled[len(filled) // 2]
    assert strip[0, x + 1] > strip[0, x]


def test_frames_before_and_after_the_move_are_clamped_to_the_line_ends():
    band = np.full((ROWS, BAND), 7, dtype=np.uint16)
    t0, t1 = 10.0, 14.0
    samples = [(t0 - 0.2, band), (t0 - 0.1, band), (t0 + 2.0, band), (t1 + 0.2, band)]
    ctrl = fakeController()
    strip = ctrl._spreadSamples(samples, t0, t1, SPAN, SCALE)
    assert strip.shape == (ROWS, int(SPAN / SCALE))
    assert strip[0, 0] == 7 and strip[0, -1] == 7


def test_calibration_falls_back_when_the_line_is_blank():
    ctrl = fakeController(lag=0.03)
    flat = np.full((ROWS, BAND), 100, dtype=np.uint16)
    blank = _Sweep([(k * 0.5, flat) for k in range(9)], 0.0, 4.0, SPAN, SCALE, True)
    ctrl._sweepLine = lambda *a, **k: blank
    lagS, fwd = ctrl._calibratePrescanLag(0.0, SPAN, 0.0, SPEED, 300.0, 4)
    assert lagS == 0.03 and fwd is blank
