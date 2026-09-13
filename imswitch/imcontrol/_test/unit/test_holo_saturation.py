#!/usr/bin/env python3
"""Overexposure metric of the inline-hologram controller.

_measure_saturation drives the "reduce the exposure time" warning in the
frontend. It has to work for plain 8-bit sensors *and* for 10/12/14-bit sensors
that ship their samples in a uint16 container, where the clip level is not
np.iinfo(dtype).max. Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.controller.controllers.InLineHoloController import (  # noqa: E402
    InLineHoloController,
    InLineHoloState,
)


def _stub():
    """Minimal object carrying just what _measure_saturation touches."""
    stub = types.SimpleNamespace(
        _state=InLineHoloState(),
        _max_sample_seen=0,
        _logger=types.SimpleNamespace(debug=lambda *a, **k: None),
    )
    stub._measure_saturation = types.MethodType(
        InLineHoloController._measure_saturation, stub
    )
    return stub


def test_uint8_counts_clipped_pixels():
    stub = _stub()
    roi = np.zeros((10, 10), dtype=np.uint8)
    roi[:2, :] = 255  # 20 % clipped
    stub._measure_saturation(roi)
    assert abs(stub._state.saturated_fraction - 0.2) < 1e-9


def test_uint8_clean_frame_reports_zero():
    stub = _stub()
    stub._measure_saturation(np.full((8, 8), 200, dtype=np.uint8))
    assert stub._state.saturated_fraction == 0.0


def test_12bit_in_uint16_container_uses_4095_not_65535():
    stub = _stub()
    roi = np.zeros((10, 10), dtype=np.uint16)
    roi[0, :] = 4095  # 10 % clipped for a 12-bit sensor
    stub._measure_saturation(roi)
    assert abs(stub._state.saturated_fraction - 0.1) < 1e-9


def test_rgb_roi_counts_across_channels():
    stub = _stub()
    roi = np.zeros((4, 4, 3), dtype=np.uint8)
    roi[:, :, 0] = 255  # red channel clipped -> 1/3 of all samples
    stub._measure_saturation(roi)
    assert abs(stub._state.saturated_fraction - 1 / 3) < 1e-6


def test_float_input_is_ignored():
    stub = _stub()
    stub._state.saturated_fraction = 0.5
    stub._measure_saturation(np.ones((4, 4), dtype=np.float32))
    assert stub._state.saturated_fraction == 0.5  # untouched, no crash


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
