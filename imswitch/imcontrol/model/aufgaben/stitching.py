"""Stitch tiles (NGFF images of one field each) into one derived NGFF image.

``stage``: tiles go where the stage says (their NGFF translations).
``ashlar``: the overlaps are registered first (ashlarUC2's EdgeAligner) and
the tiles go to the registered positions, anchored so the mean shift against
the stage positions is zero. Overlaps are last-writer-wins; the raw tiles stay
untouched, the stitched image is derived data.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .storage import NgffImage


def _ashlar_offsets(tiles: Sequence[NgffImage], stage_px: np.ndarray, channel: int,
                    max_shift_um: float) -> np.ndarray:
    """Registered tile offsets (y, x) in pixels, in the frame of ``stage_px``."""
    from ashlarUC2 import reg   # optional at runtime, only for ashlar stitching

    h, w = tiles[0].array.shape[3:]
    px = tiles[0].spec.pixel_size_um

    class Meta(reg.Metadata):
        _num_images = len(tiles)
        num_channels = tiles[0].array.shape[1]
        pixel_size = px
        pixel_dtype = tiles[0].array.dtype

        def tile_position(self, i):
            return stage_px[i]

        def tile_size(self, i):
            return np.array([h, w])

    class Reader(reg.Reader):
        metadata = Meta()

        def read(self, series, c):   # first time point, middle plane
            return np.asarray(tiles[series].array[0, c, tiles[series].array.shape[2] // 2])

    aligner = reg.EdgeAligner(Reader(), channel=channel, max_shift=max_shift_um,
                              do_make_thumbnail=False, verbose=False)
    aligner.run()
    positions = np.asarray(aligner.positions, dtype=float)
    return positions - (positions - stage_px).mean(axis=0)


def stitch(tiles: Sequence[NgffImage], path, name: str, method: str = "stage",
           channel: int = 0, max_shift_um: float = 50.0) -> NgffImage:
    """One image from tiles that share T, C, Z, dtype and pixel size."""
    first = tiles[0]
    px = first.spec.pixel_size_um
    origins = np.array([[img.spec.origin_um[1], img.spec.origin_um[0]] for img in tiles])  # y, x
    offsets = (origins - origins.min(axis=0)) / px
    if method == "ashlar":
        offsets = _ashlar_offsets(tiles, offsets, channel, max_shift_um)
    elif method != "stage":
        raise ValueError(f"unknown stitching method {method!r}")
    corner = offsets.min(axis=0)
    offsets = np.round(offsets - corner).astype(int)
    t, c, z, h, w = first.array.shape
    oy, ox = origins.min(axis=0) + corner * px
    spec = replace(first.spec, shape=(t, c, z, int(offsets[:, 0].max()) + h,
                                      int(offsets[:, 1].max()) + w),
                   origin_um=(float(ox), float(oy), first.spec.origin_um[2]), name=name,
                   chunks_yx=None, exposure_ms=list(first.spec.exposure_ms))
    out = NgffImage.create(Path(path), spec, first.version)
    for tile, (y0, x0) in zip(tiles, offsets):
        for ti, ci, zi in np.ndindex(t, c, z):
            out.write(np.asarray(tile.array[ti, ci, zi]), ti, ci, zi, int(y0), int(x0))
    out.finalize()
    return out
