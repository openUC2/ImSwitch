#!/usr/bin/env python3
"""Off-axis hologram helpers: sideband finder and fringe trace.

Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import threading
import time
import types
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.controller.controllers.OffAxisHoloController import (  # noqa: E402
    OffAxisHoloController,
    OffAxisHoloParams,
)


def _fringes(h=256, w=256, kx=40, ky=-24):
    """Pure carrier: kx/ky cycles across the frame."""
    yy, xx = np.mgrid[:h, :w]
    return 1 + np.cos(2 * np.pi * (kx * xx / w + ky * yy / h))


def test_locate_sideband_finds_the_carrier():
    cx, cy, size, period = OffAxisHoloController._locate_sideband(_fringes())
    # +1 order sits at (128+40, 128-24), the right half-plane
    assert (cx, cy) == (168, 104)
    dist = np.hypot(40, 24)
    assert abs(size - 2 * dist / 3) <= 2 and size % 2 == 0
    assert abs(period - 256 / dist) < 0.05


def test_locate_sideband_ignores_edge_cross():
    # A strong gradient makes the frequency axes bright without a window.
    img = _fringes(kx=30, ky=30) * 0.2 + np.linspace(0, 50, 256)[None, :]
    cx, cy, _, _ = OffAxisHoloController._locate_sideband(img)
    assert (cx, cy) == (158, 158)


def _trace_stub():
    stub = types.SimpleNamespace(
        _params=OffAxisHoloParams(),
        _trace_lock=threading.Lock(),
        _trace_t=deque(maxlen=5),
        _trace_v=deque(maxlen=5),
        _trace_count=0,
        _trace=None,
        _trace_last={},
    )
    for name in ("_sample_trace", "_extract_color_channel",
                 "get_fringe_trace_offaxisholo"):
        fn = getattr(OffAxisHoloController, name)
        setattr(stub, name, types.MethodType(getattr(fn, "__wrapped__", fn), stub))
    return stub


def test_trace_samples_box_mean_at_fractional_position():
    stub = _trace_stub()
    stub._trace = {"fx": 1.0, "fy": 0.0, "radius": 1, "t0": time.time(),
                   "x": None, "y": None}
    img = np.zeros((10, 20), dtype=np.uint8)
    img[0:2, 18:20] = 90  # 4 px of the clipped 2x2 box at the top-right corner
    stub._sample_trace(img)
    out = stub.get_fringe_trace_offaxisholo(0)
    assert (out["x"], out["y"]) == (19, 0)
    assert out["v"] == [90.0]


def test_trace_since_survives_ring_buffer_wrap():
    stub = _trace_stub()
    stub._trace = {"fx": 0.5, "fy": 0.5, "radius": 0, "t0": time.time(),
                   "x": None, "y": None}
    for k in range(8):  # maxlen 5 -> samples 3..7 kept
        stub._sample_trace(np.full((4, 4), k, dtype=np.uint8))
    assert stub.get_fringe_trace_offaxisholo(0)["v"] == [3, 4, 5, 6, 7]
    out = stub.get_fringe_trace_offaxisholo(6)
    assert out["v"] == [6, 7] and out["next"] == 8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
