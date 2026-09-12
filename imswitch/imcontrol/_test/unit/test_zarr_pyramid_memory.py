#!/usr/bin/env python3
"""The OME-Zarr pyramid must not need the whole canvas in RAM.

Franzi's 2026-09-08 log: a 270336 x 172032 RGB slide scan finished, then the
pyramid step asked numpy for 130 GiB three times ("Unable to allocate") and
logged "generated successfully" anyway. Reading the canvas one row band at a
time keeps the working set small however large the scan.
Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.model.io.ome_writers.ome_writer import OMEWriter  # noqa: E402


class _Canvas:
    """A zarr-like array that refuses to be read in one piece.

    Slicing more than `max_rows` rows raises MemoryError, exactly like numpy
    did on the Pi when asked for the whole plane.
    """

    def __init__(self, data, max_rows):
        self._data = data
        self.shape = data.shape
        self.max_rows = max_rows
        self.reads = []

    def __getitem__(self, key):
        rows = key[3]
        if isinstance(rows, slice):
            n = len(range(*rows.indices(self.shape[3])))
        else:
            n = 1
        if n > self.max_rows:
            raise MemoryError(f"Unable to allocate {n} rows")
        self.reads.append(n)
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


def _writer(is_rgb):
    stub = types.SimpleNamespace(
        isRGB=is_rgb,
        logger=types.SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None,
                                     error=lambda *a, **k: None, debug=lambda *a, **k: None),
        PYRAMID_BAND_ROWS=64,
        _pyramid_failures=0,
    )
    stub._downsample_all_dimensions = types.MethodType(OMEWriter._downsample_all_dimensions, stub)
    return stub


def test_pyramid_is_built_from_bands_that_fit():
    h, w = 300, 200
    full = np.arange(h * w, dtype=np.uint16).reshape(1, 1, 1, h, w)
    source = _Canvas(full.copy(), max_rows=64)          # whole-plane reads would fail
    target = np.zeros((1, 1, 1, h // 2, w // 2), dtype=np.uint16)

    _writer(False)._downsample_all_dimensions(source, target, level=1, n_t=1, n_c=1, n_z=1)

    np.testing.assert_array_equal(target[0, 0, 0], full[0, 0, 0, ::2, ::2])
    assert max(source.reads) <= 64                       # never more than a band


def test_rgb_pyramid_is_built_from_bands_that_fit():
    h, w = 130, 90
    full = np.random.default_rng(0).integers(0, 255, (1, 1, 1, h, w, 3), dtype=np.uint8)
    source = _Canvas(full.copy(), max_rows=64)
    target = np.zeros((1, 1, 1, (h + 3) // 4, (w + 3) // 4, 3), dtype=np.uint8)

    _writer(True)._downsample_all_dimensions(source, target, level=2, n_t=1, n_c=1, n_z=1)

    np.testing.assert_array_equal(target[0, 0, 0], full[0, 0, 0, ::4, ::4, :])


def test_a_failed_level_is_counted_not_swallowed():
    source = _Canvas(np.zeros((1, 1, 1, 100, 10), np.uint8), max_rows=5)   # every band too big
    target = np.zeros((1, 1, 1, 50, 5), np.uint8)
    w = _writer(False)
    w.PYRAMID_BAND_ROWS = 64
    w._downsample_all_dimensions(source, target, level=1, n_t=1, n_c=1, n_z=1)
    assert w._pyramid_failures == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
