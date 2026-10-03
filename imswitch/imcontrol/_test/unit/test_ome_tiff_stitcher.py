#!/usr/bin/env python3
"""OmeTiffStitcher must stop when told and write everything it was given.

Both assertions used to fail. The loop exited only after seeing exactly nx*ny
images, so a short run (dropped frame, aborted scan, site with fewer tiles than
the grid) hung stop() forever, and a long one (any Z-stack or second channel)
broke out early and discarded the tail of the queue.

Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import tempfile
import threading

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.model.io.ome_writers import OmeTiffStitcher  # noqa: E402

TILE = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)


def _run(n_images, stop_timeout=15.0):
    """Write n_images through a stitcher; return (path, stopped_cleanly)."""
    path = os.path.join(tempfile.mkdtemp(), "stitched.ome.tif")
    stitcher = OmeTiffStitcher(path, tile_w=16, tile_h=16)
    stitcher.start()
    for i in range(n_images):
        stitcher.add_image(TILE, position_x=i * 16.0, position_y=0.0,
                           index_x=i, index_y=0, pixel_size=0.5)

    # stop() joins the writer thread; run it off-thread so a regression to the
    # old "wait for nx*ny images" loop fails the test instead of hanging it.
    stopper = threading.Thread(target=stitcher.stop, daemon=True)
    stopper.start()
    stopper.join(timeout=stop_timeout)
    return path, not stopper.is_alive()


def test_stop_returns_on_a_short_run():
    """A site that produced fewer tiles than its grid must still finalize."""
    _, stopped = _run(3)
    assert stopped, "stop() did not return — the next site would never start"


def test_stop_returns_with_no_images_at_all():
    """The aborted-before-first-frame case."""
    _, stopped = _run(0)
    assert stopped


def test_every_queued_image_is_written():
    """A Z-stack or second channel sends more frames than the XY grid has."""
    path, stopped = _run(20)
    assert stopped
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 20


def test_file_is_closed_and_readable():
    """The blank-stitched-image complaint: the file must open and hold data."""
    path, stopped = _run(4)
    assert stopped
    assert os.path.getsize(path) > 0
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 4
        np.testing.assert_array_equal(tif.pages[0].asarray(), TILE)


def test_writer_thread_is_not_a_daemon():
    """A daemon thread gets killed inside the TiffWriter context at exit,
    which is what left files unclosed and unreadable."""
    path = os.path.join(tempfile.mkdtemp(), "stitched.ome.tif")
    stitcher = OmeTiffStitcher(path)
    stitcher.start()
    try:
        assert stitcher._thread.daemon is False
    finally:
        stitcher.stop()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
