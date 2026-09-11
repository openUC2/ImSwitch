#!/usr/bin/env python3
"""Buffers that hold frames must be bounded.

Every one of these used to grow without limit whenever the consumer fell behind
the producer — which on a Pi (slow storage, compression on) is the normal case,
not the exceptional one. Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import tempfile
import threading
import time
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.model.io.data_store import AcquisitionDataStore  # noqa: E402
from imswitch.imcontrol.model.io.ome_writers import OmeTiffStitcher  # noqa: E402
from imswitch.imcontrol.controller.controllers.StageMapController import (  # noqa: E402
    StageMapController,
)

TILE = np.zeros((32, 32), dtype=np.uint16)


def test_stitcher_queue_does_not_grow_without_limit():
    """A writer that never drains must not swallow unbounded frames."""
    path = os.path.join(tempfile.mkdtemp(), "s.ome.tif")
    stitcher = OmeTiffStitcher(path)
    stitcher.QUEUE_WAIT_S = 0.05         # don't wait 30 s in a test
    stitcher._logger = types.SimpleNamespace(
        error=lambda *a, **k: None, info=lambda *a, **k: None,
        warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    # Deliberately never started: nothing drains the queue.
    for _ in range(OmeTiffStitcher.MAX_QUEUED_FRAMES + 20):
        stitcher.add_image(TILE, 0.0, 0.0, 0, 0, 1.0)
    assert len(stitcher.queue) == OmeTiffStitcher.MAX_QUEUED_FRAMES


def test_stitcher_recovers_room_as_it_writes():
    path = os.path.join(tempfile.mkdtemp(), "s.ome.tif")
    stitcher = OmeTiffStitcher(path)
    stitcher.start()
    try:
        for _ in range(OmeTiffStitcher.MAX_QUEUED_FRAMES * 3):
            stitcher.add_image(TILE, 0.0, 0.0, 0, 0, 1.0)
    finally:
        stitcher.stop()
    assert stitcher.images_written == OmeTiffStitcher.MAX_QUEUED_FRAMES * 3


def test_write_queue_and_error_list_are_capped():
    assert AcquisitionDataStore.MAX_QUEUED_WRITES > 0
    assert AcquisitionDataStore.MAX_KEPT_ERRORS > 0
    # The trim idiom used on the error list keeps the newest entries.
    errors = [f"e{i}" for i in range(200)]
    del errors[: -AcquisitionDataStore.MAX_KEPT_ERRORS]
    assert len(errors) == AcquisitionDataStore.MAX_KEPT_ERRORS
    assert errors[-1] == "e199"


def _stage_map_stub():
    stub = types.SimpleNamespace(_tiles=[], MAX_TILE_PREVIEWS=StageMapController.MAX_TILE_PREVIEWS)
    stub._trimTiles = types.MethodType(StageMapController._trimTiles, stub)
    return stub


def test_only_the_newest_tile_previews_stay_in_memory():
    stub = _stage_map_stub()
    for i in range(StageMapController.MAX_TILE_PREVIEWS + 50):
        stub._tiles.append({"id": i, "preview": "x" * 10})
        stub._trimTiles()

    kept = [t for t in stub._tiles if t["preview"]]
    assert len(kept) == StageMapController.MAX_TILE_PREVIEWS
    # Metadata is kept for every tile; only the image is released.
    assert len(stub._tiles) == StageMapController.MAX_TILE_PREVIEWS + 50
    assert kept[-1]["id"] == StageMapController.MAX_TILE_PREVIEWS + 49


def test_prescan_strip_is_built_coarser_until_it_fits_its_budget():
    """The strip is the only buffer a prescan line holds. A 10 mm line at
    0.12 um/px would be 83k columns at camera resolution; the layout grows the
    subsample factor until the strip fits PRESCAN_MAX_STRIP_MB."""
    from imswitch.imcontrol.controller.controllers.StageMapController import prescanLayout

    budget = StageMapController.PRESCAN_MAX_STRIP_MB * 1024 * 1024
    slots, w, h, sub, dx = prescanLayout(
        spanUm=10000.0, dxUm=300.0, pixelSizeUm=0.12, frameShape=(3000, 3000),
        subsample=1, maxStripBytes=budget)
    assert w * h * 2 <= budget
    assert sub > 1                      # it had to coarsen
    assert slots == int(np.ceil(10000.0 / 300.0))   # sampling is by tile pitch, not frame cap


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
