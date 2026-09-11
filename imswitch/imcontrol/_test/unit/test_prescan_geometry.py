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
    )
    for name in ("_composeStrip", "_decodePreview"):
        setattr(stub, name, types.MethodType(getattr(StageMapController, name), stub))
    stub._resizeGray = StageMapController._resizeGray
    return stub


def test_strip_width_matches_the_distance_travelled():
    """1000 um at 1 um/px must be a 1000 px strip, whatever the frame rate."""
    c = _controller(pixel_size=1.0)
    for n_frames in (5, 50, 500):
        frames = [(np.full((16, 16), 100, np.uint8), i * 1000.0 / (n_frames - 1))
                  for i in range(n_frames)]
        strip = c._composeStrip(frames, span=1000.0)
        assert strip.shape == (16, 1000), (n_frames, strip.shape)


def test_strip_width_follows_the_pixel_size():
    """The same sweep through a 2 um/px objective is half as many pixels."""
    frames = [(np.zeros((8, 8), np.uint8), x) for x in (0.0, 500.0, 1000.0)]
    assert _controller(1.0)._composeStrip(frames, 1000.0).shape[1] == 1000
    assert _controller(2.0)._composeStrip(frames, 1000.0).shape[1] == 500


def test_frames_land_in_travel_order_not_frame_order():
    """A frame recorded at 90% of the sweep belongs at 90% of the strip."""
    c = _controller(pixel_size=1.0)
    frames = [
        (np.full((4, 4), 10, np.uint8), 0.0),
        (np.full((4, 4), 200, np.uint8), 900.0),
    ]
    strip = c._composeStrip(frames, span=1000.0)
    assert strip[:, :50].max() == 10
    assert strip[:, 850:950].max() == 200


def test_resize_preserves_the_requested_shape():
    img = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert StageMapController._resizeGray(img, 20, 5).shape == (5, 20)
    assert StageMapController._resizeGray(img, 8, 8) is img  # no-op when it fits


def test_a_single_frame_line_is_rejected_by_the_caller_not_composed():
    """_prescanLine returns None below 2 frames; compose is never called with
    one, but it must not blow up if it is."""
    c = _controller()
    strip = c._composeStrip([(np.ones((4, 4), np.uint8), 0.0)], span=100.0)
    assert strip.shape[0] == 4 and strip.shape[1] == 100




# ---------------------------------------------------------------------------
# Serpentine sweep
# ---------------------------------------------------------------------------

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

    def fake_line(startX, endX, y, speedX):
        c.sweeps.append((startX, endX, y))
        # _composeStrip places frames by DISTANCE TRAVELLED, so a strip always
        # comes back in travel order: a return line is descending in X. Model
        # that, or the flip under test has nothing to undo.
        ramp = np.arange(10, dtype=np.uint8)
        if startX > endX:
            ramp = ramp[::-1]
        return np.tile(ramp, (2, 1))

    c._prescanLine = fake_line
    c._addStrip = lambda strip, minX, maxX, y: c.strips.append((y, strip.copy()))
    for name in ("_prescanLoop", "_enterPrescanOptics", "_restorePrescanOptics",
                 "_objectiveController"):
        setattr(c, name, types.MethodType(getattr(StageMapController, name), c))
    return c


def test_lines_alternate_direction():
    """The stage is already at the far end, so drive straight back."""
    c = _recording_controller()
    c._prescanLoop(0.0, 1000.0, 0.0, 200.0, 100.0, 5000.0, None)

    assert [(s[0], s[1]) for s in c.sweeps] == [
        (0.0, 1000.0), (1000.0, 0.0), (0.0, 1000.0),
    ]
    assert [s[2] for s in c.sweeps] == [0.0, 100.0, 200.0]


def test_reverse_lines_are_stored_left_to_right():
    """Otherwise every other line of the map is mirrored."""
    c = _recording_controller()
    c._prescanLoop(0.0, 1000.0, 0.0, 100.0, 100.0, 5000.0, None)

    forward, reverse = c.strips[0][1], c.strips[1][1]
    np.testing.assert_array_equal(forward[0], np.arange(10))
    np.testing.assert_array_equal(reverse[0], np.arange(10))  # un-mirrored


def test_stage_position_is_restored_after_the_sweep():
    c = _recording_controller()
    moves = []
    c._stage.move = lambda **kw: moves.append(kw)
    c._prescanLoop(0.0, 1000.0, 0.0, 0.0, 100.0, 5000.0, None)

    # Last moves put X/Y and Z back where they were found.
    assert moves[-2]["value"] == (1.0, 2.0)
    assert moves[-1]["value"] == 3.0 and moves[-1]["axis"] == "Z"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
