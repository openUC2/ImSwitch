#!/usr/bin/env python3
"""WP-17 — prescan strips and the overview must be in stage micrometres.

Everything downstream (drawing on the plate map, planning a 20x acquisition
from a 10x prescan) depends on the strip's width in um matching the distance
the stage actually travelled, and on the overview reporting the extent it
covers. Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.controller.controllers.StageMapController import (  # noqa: E402
    StageMapController,
)


def _controller(pixel_size=1.0):
    stub = types.SimpleNamespace(
        _logger=types.SimpleNamespace(
            debug=lambda *a, **k: None, info=lambda *a, **k: None,
            warning=lambda *a, **k: None, error=lambda *a, **k: None,
        ),
        _getPixelSizeUm=lambda: pixel_size,
        PRESCAN_MAX_STRIP_MB=StageMapController.PRESCAN_MAX_STRIP_MB,
        PRESCAN_SUBSAMPLE=StageMapController.PRESCAN_SUBSAMPLE,
    )
    stub._centreBand = types.MethodType(StageMapController._centreBand, stub)
    stub._resizeGray = StageMapController._resizeGray
    return stub


def test_resize_preserves_the_requested_shape():
    img = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert StageMapController._resizeGray(img, 20, 5).shape == (5, 20)
    assert StageMapController._resizeGray(img, 8, 8) is img  # no-op when it fits


def test_centre_band_is_the_middle_of_the_frame_subsampled():
    stub = _controller()
    frame = np.arange(100 * 400, dtype=np.uint16).reshape(100, 400)
    band = stub._centreBand(frame, widthPx=200, subsample=4)
    assert band.shape == (25, 50)
    assert band.base is None  # a copy — the camera may reuse its buffer
    np.testing.assert_array_equal(band, frame[::4, 100:300:4])


class _FakeClock:
    """time.time() we can step, so the sweep runs in zero real time."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, dt):
        self.now += dt


def _sweep_controller(pixel_size, frame_shape, speed, frames_per_s):
    """A controller whose camera returns a NEW frame every 1/frames_per_s of
    fake time, valued by the X the stage had reached when it was exposed."""
    c = _controller(pixel_size)
    clock = _FakeClock()
    c.clock = clock
    c._shouldStop = types.SimpleNamespace(is_set=lambda: False)
    c._stage = types.SimpleNamespace(move=lambda **kw: None, combinedAxes=["XY"])

    started = {"t": None}

    def latest(returnFrameNumber=False):
        t = clock.now - (started["t"] if started["t"] is not None else clock.now)
        n = int(t * frames_per_s)
        x_when_exposed = speed * (n / frames_per_s)          # µm since the start
        # +1 so that 0 can only mean "never written".
        frame = np.full(frame_shape, min(65535, int(x_when_exposed) + 1), np.uint16)
        return (frame, n) if returnFrameNumber else frame

    c._detector = types.SimpleNamespace(shape=frame_shape, getLatestFrame=latest)
    c._prescanLine = types.MethodType(StageMapController._prescanLine, c)
    c._on_move_start = lambda: started.__setitem__("t", clock.now)
    return c


def test_exposure_k_lands_in_the_columns_slot_k_covers(monkeypatch):
    """The whole point of clock-based sampling: at constant speed, exposure k
    is due k·dx/speed after the start and must fill exactly the strip
    columns [k·dx, (k+1)·dx) — whatever the frame rate."""
    import imswitch.imcontrol.controller.controllers.StageMapController as sm

    speed, dx, span, px = 10000.0, 300.0, 3000.0, 0.12
    c = _sweep_controller(px, (400, 3000), speed, frames_per_s=1000.0)

    class Move:
        def __init__(self, stage, value, axis, speed_):
            self.finished = False
            self.error = None
            self.t_end = c.clock.now + abs(value) / speed_

        def start(self):
            c._on_move_start()

        @property
        def finished(self):
            return c.clock.now >= self.t_end

        @finished.setter
        def finished(self, _):
            pass

        def join(self, timeout=None):
            c.clock.now = max(c.clock.now, self.t_end)

    monkeypatch.setattr(sm, "_NonBlockingMove", Move)
    monkeypatch.setattr(sm.time, "time", c.clock.time)
    monkeypatch.setattr(sm.time, "sleep", c.clock.sleep)

    strip, scale = c._prescanLine(0.0, span, 0.0, speed, dx, 4)

    assert scale == px * 4
    assert strip.shape[1] == int(np.ceil(span / scale))
    slots = int(np.ceil(span / dx))
    for k in range(slots):
        c0 = int(round(k * dx / scale))
        c1 = min(strip.shape[1], int(round((k + 1) * dx / scale)))
        band = strip[0, c0:c1]
        # Every column of slot k holds a frame exposed within one frame period
        # of the moment the stage was at k·dx.
        assert band.min() - 1 >= k * dx - speed / 1000.0 - 1, (k, band.min())
        assert band.max() - 1 <= k * dx + speed / 1000.0 + 1, (k, band.max())
    assert strip.min() > 0  # every column was written by some exposure


def _recording_controller(pixel_size=1.0):
    """A controller stub that records the line sweeps it was asked to run."""
    c = _controller(pixel_size)
    c.sweeps = []
    c.strips = []
    c._shouldStop = types.SimpleNamespace(is_set=lambda: False)
    c._prescanThread = None
    c._emitStatus = lambda: None
    c._master = types.SimpleNamespace(
        detectorsManager=types.SimpleNamespace(
            startAcquisition=lambda: None, stopAcquisition=lambda h: None),
        getController=lambda name: None,
    )
    c._stage = types.SimpleNamespace(
        getPosition=lambda: {"X": 1.0, "Y": 2.0, "Z": 3.0},
        move=lambda **kw: None,
        combinedAxes=["XY"],
    )

    def fake_line(startX, endX, y, speedX, dx, subsample):
        c.sweeps.append((startX, endX, y))
        # The strip comes back in travel order: a return line is descending in
        # X. Model that, or the flip under test has nothing to undo.
        ramp = np.arange(10, dtype=np.uint8)
        if startX > endX:
            ramp = ramp[::-1]
        return np.tile(ramp, (2, 1)), 1.0

    c._prescanLine = fake_line
    c._addStrip = lambda strip, minX, maxX, y, pixelSize: c.strips.append((y, strip.copy()))
    for name in ("_prescanLoop", "_enterPrescanOptics", "_restorePrescanOptics",
                 "_objectiveController"):
        setattr(c, name, types.MethodType(getattr(StageMapController, name), c))
    return c


def test_lines_alternate_direction():
    """The stage is already at the far end, so drive straight back."""
    c = _recording_controller()
    c._prescanLoop(0.0, 1000.0, 0.0, 200.0, 100.0, 5000.0, None, 0.0, 4)

    assert [(s[0], s[1]) for s in c.sweeps] == [
        (0.0, 1000.0), (1000.0, 0.0), (0.0, 1000.0),
    ]
    assert [s[2] for s in c.sweeps] == [0.0, 100.0, 200.0]


def test_reverse_lines_are_stored_left_to_right():
    """Otherwise every other line of the map is mirrored."""
    c = _recording_controller()
    c._prescanLoop(0.0, 1000.0, 0.0, 100.0, 100.0, 5000.0, None, 0.0, 4)

    forward, reverse = c.strips[0][1], c.strips[1][1]
    np.testing.assert_array_equal(forward[0], np.arange(10))
    np.testing.assert_array_equal(reverse[0], np.arange(10))  # un-mirrored


def test_stage_position_is_restored_after_the_sweep():
    c = _recording_controller()
    moves = []
    c._stage.move = lambda **kw: moves.append(kw)
    c._prescanLoop(0.0, 1000.0, 0.0, 0.0, 100.0, 5000.0, None, 0.0, 4)

    # Last moves put X/Y and Z back where they were found.
    assert moves[-2]["value"] == (1.0, 2.0)
    assert moves[-1]["value"] == 3.0 and moves[-1]["axis"] == "Z"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
