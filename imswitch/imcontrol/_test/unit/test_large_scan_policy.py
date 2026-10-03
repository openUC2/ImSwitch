#!/usr/bin/env python3
"""A scan too big to stitch on the Pi keeps its tiles and drops the canvas.

Franzi's 2026-09-08 run: 5,726 tiles into one OME-Zarr canvas, then a 130 GiB
pyramid request at the end. The policy lives in OMEWriter.__init__ so every
caller — normal, performance, fast-scan — gets it. Runs standalone.
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.model.io.ome_writers.ome_writer import (  # noqa: E402
    OMEFileStorePaths, OMEWriter, OMEWriterConfig,
)

TILE = (2048, 2048)


def _writer(nx, ny, is_rgb=False, **flags):
    log = []
    logger = types.SimpleNamespace(warning=log.append, info=lambda *a, **k: None,
                                   error=log.append, debug=lambda *a, **k: None)
    cfg = OMEWriterConfig(write_zarr=True, write_stitched_tiff=True, write_tiff=True,
                          write_individual_tiffs=False, **flags)
    paths = OMEFileStorePaths(os.path.join(tempfile.mkdtemp(), "run"))
    w = OMEWriter(paths, TILE, (nx, ny), (0, 0, 2048, 2048), cfg,
                  logger=logger, isRGB=is_rgb)
    return w, log


def test_small_scan_keeps_every_requested_output():
    w, _ = _writer(3, 3)                      # 9 tiles, ~75 MB
    assert not w.large_scan
    assert w.config.write_zarr and w.config.write_stitched_tiff
    assert w.config.write_tiff                # 75 MB mosaic is fine in RAM
    assert not w.config.write_individual_tiffs
    w.finalize()


def test_large_scan_drops_every_canvas_and_keeps_tiles():
    w, log = _writer(60, 40, is_rgb=True)     # 2400 tiles x 12 MB = 29 GiB
    assert w.large_scan
    assert not w.config.write_zarr
    assert not w.config.write_stitched_tiff
    assert not w.config.write_tiff
    assert w.config.write_individual_tiffs   # the one output that scales
    assert w.canvas is None and w.tiff_stitcher is None and w.tiff_mosaic is None
    assert any("Large scan" in m and "GiB" in m for m in log)
    w.finalize()


def test_medium_scan_drops_only_the_in_ram_mosaic():
    w, _ = _writer(12, 12)                    # 144 tiles x 8 MB = 1.1 GiB: under 2 GiB, over 512 MiB
    assert not w.large_scan
    assert w.config.write_zarr and w.config.write_stitched_tiff
    assert not w.config.write_tiff            # would be 1.1 GiB of numpy on an 8 GB Pi
    w.finalize()


def test_canvas_bytes_counts_every_dimension():
    w, _ = _writer(2, 2, n_time_points=3, n_channels=2, n_z_planes=5)
    assert w.canvas_bytes() == 3 * 2 * 5 * 2 * 2048 * 2 * 2048 * 2
    w.finalize()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
