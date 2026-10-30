"""STORM direct saving: every frame of a (frames, ids) chunk reaches the OME files."""
import logging
from types import SimpleNamespace

import numpy as np
import tifffile

from imswitch.imcommon.model import dirtools
from imswitch.imcontrol.controller.controllers.STORMReconController import STORMReconController


class FakeCamera:
    """getChunk() like Hik and the virtual camera: (frames, ids), then nothing."""

    pixelSizeUm = [1, 0.1, 0.1]

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def startAcquisition(self):
        pass

    def getChunk(self):
        if not self._chunks:
            return np.empty((0, 0, 0)), []
        frames = self._chunks.pop(0)
        return frames, list(range(len(frames)))


def _controller(camera, crop=None):
    c = STORMReconController.__new__(STORMReconController)   # no widget base, no master
    noop = SimpleNamespace(emit=lambda *a: None)
    c.__dict__.update(
        _logger=logging.getLogger("storm-test"), detector=camera, _cropping_params=crop,
        _local_processing_enabled=False, _frame_count=0, _current_session_id="s",
        _ome_zarr_store=None, _zarr_path=None, _tiff_save_path=None,
        sigNSTORMImageAcquired=noop, sigFrameAcquired=noop, sigFrameProcessed=noop,
        sigErrorOccurred=noop)
    return c


def test_every_frame_is_saved_and_exported(tmp_path, monkeypatch):
    monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath",
                        staticmethod(lambda: str(tmp_path)))
    rng = np.random.default_rng(1)
    chunks = [rng.integers(0, 4000, (3, 64, 80), dtype=np.uint16) for _ in range(3)]
    c = _controller(FakeCamera(chunks), crop={"x": 8, "y": 4, "width": 40, "height": 32})
    c._initializeSaving("run", "tiff")
    c._acquisition_active = True
    c._startDirectSavingAcquisition(max_frames=7)
    c._acquisition_thread.join(5)
    assert c._frame_count == 7 and not c._acquisition_active
    c._finalizeSaving()
    tiffs = list(tmp_path.rglob("*.ome.tif"))
    assert len(tiffs) == 1 and not list(tmp_path.rglob(".run.ome.zarr"))
    data = tifffile.imread(tiffs[0])
    assert data.shape[-2:] == (32, 40) and data.reshape(-1, 32, 40).shape[0] == 7
    np.testing.assert_array_equal(data.reshape(-1, 32, 40)[4], chunks[1][1][4:36, 8:48])


def test_frames_from_a_plain_array_or_list():
    c = _controller(FakeCamera([]))
    one = np.zeros((5, 6))
    assert len(c._chunkFrames((np.stack([one, one]), [0, 1]))) == 2
    assert len(c._chunkFrames([one, one, one])) == 3
    assert c._chunkFrames((np.empty((0, 0, 0)), [])) == []
    assert c._chunkFrames(None) == []
