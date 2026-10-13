"""RecordingService streaming: frames in, one valid NGFF image per detector out."""
from pathlib import Path

import numpy as np
import zarr
from ome_zarr_models.v04.image import Image

from imswitch.imcontrol.model.io.recording_service import RecordingService


class _Camera:
    shape = (40, 30)               # width, height in px
    pixelSizeUm = [1, 1, 0.65]
    dtype = "uint16"


class _Detectors:
    def getAllDeviceNames(self):
        return ["cam"]

    def __getitem__(self, name):
        return _Camera()


def test_streaming_writes_valid_ngff(tmp_path):
    service = RecordingService()
    try:
        assert service.start_streaming(str(tmp_path / "rec"), _Detectors(), auto_capture=False)
        assert service.is_streaming and service.is_recording
        assert not service.start_streaming(str(tmp_path / "again"), _Detectors())
        for t in range(3):
            service.write_streaming_frame("cam", np.full((30, 40), t, np.uint16))
        stats = service.stop_streaming()
    finally:
        service.shutdown()
    assert stats["frames"] == 3 and not service.is_streaming
    group_path = next(p.parent for p in Path(tmp_path).rglob(".zattrs")
                      if (p.parent / "0").exists())
    group = zarr.open_group(str(group_path), mode="r")
    Image.from_zarr(group)
    assert group["0"].shape == (3, 1, 1, 30, 40) and int(group["0"][2, 0, 0, 5, 5]) == 2
