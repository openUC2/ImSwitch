"""The OME-Zarr pyramid must not need the whole canvas in RAM.

Franzi's 2026-09-08 log: a 270336 x 172032 RGB slide scan finished, then the
pyramid step asked numpy for 130 GiB three times ("Unable to allocate") and
logged "generated successfully" anyway. NgffImage (shared by ExperimentController
and Aufgaben) reads the canvas one row band at a time, and a failure surfaces.
"""
import numpy as np
import pytest

from imswitch.imcontrol.model.aufgaben.storage import ImageSpec, NgffImage


class _Spy:
    """Wraps a zarr array; slicing more than ``max_rows`` rows raises MemoryError."""

    def __init__(self, array, max_rows):
        self._array, self.max_rows, self.reads = array, max_rows, []

    def __getattr__(self, name):
        return getattr(self._array, name)

    def __getitem__(self, key):
        rows = key[3]
        n = len(range(*rows.indices(self._array.shape[3]))) if isinstance(rows, slice) else 1
        if n > self.max_rows:
            raise MemoryError(f"Unable to allocate {n} rows")
        self.reads.append(n)
        return self._array[key]

    def __setitem__(self, key, value):
        self._array[key] = value


def _image(tmp_path, h, w, dtype="uint16", channels=1):
    image = NgffImage.create(tmp_path / "c.ome.zarr", ImageSpec(
        shape=(1, channels, 1, h, w), dtype=dtype, chunks_yx=(32, w)))
    data = np.random.default_rng(0).integers(0, 255, (channels, h, w)).astype(dtype)
    for c in range(channels):
        image.write(data[c], 0, c, 0)
    return image, data


def test_pyramid_is_built_from_bands_that_fit(tmp_path):
    image, data = _image(tmp_path, 300, 200)
    image.BAND_ROWS = 64
    image.array = spy = _Spy(image.array, max_rows=64)      # whole-plane reads would fail
    image.finalize(min_size=50, max_levels=3)
    assert image.levels == 3 and max(spy.reads) <= 64       # never more than a band
    np.testing.assert_array_equal(image.group["1"][0, 0, 0], data[0, ::2, ::2])
    np.testing.assert_array_equal(image.group["2"][0, 0, 0], data[0, ::4, ::4])


def test_rgb_canvas_pyramid_keeps_three_channels(tmp_path):
    image, data = _image(tmp_path, 130, 90, dtype="uint8", channels=3)
    image.BAND_ROWS = 64
    image.finalize(min_size=20, max_levels=3)
    np.testing.assert_array_equal(image.group["2"][0, :, 0], data[:, ::4, ::4])


def test_a_failed_level_is_not_swallowed(tmp_path):
    image, _ = _image(tmp_path, 100, 10, dtype="uint8")
    image.array = _Spy(image.array, max_rows=5)               # every band too big
    with pytest.raises(MemoryError):
        image.finalize(min_size=4)
