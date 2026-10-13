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


def test_snap_tiff_keeps_calibration_position_and_attributes(tmp_path):
    """Snaps are OME-TIFFs: pixel size, top-left position, exposure, all attributes kept."""
    import ome_types
    from imswitch.imcontrol.model.io.recording_service import save_snap_tiff
    metadata = {"Detector": "cam", "Channel": "488nm", "PhysicalSizeX": 0.5,
                "PositionX": 100.0, "PositionY": 50.0, "PositionZ": 3.0, "ExposureTime": 0.02,
                "Gain": 2.0, "AcquisitionAttributes": {"Laser": {"power": 5}}}
    path = tmp_path / "snap.ome.tiff"
    save_snap_tiff(str(path), np.zeros((30, 40), np.uint16), metadata)
    ome = ome_types.from_tiff(path)
    pixels = ome.images[0].pixels
    plane = pixels.planes[0]
    assert pixels.physical_size_x == 0.5 and pixels.channels[0].name == "488nm"
    assert (plane.position_x, plane.position_y, plane.exposure_time) == (90.0, 42.5, 20.0)
    notes = {m.k: m.value for m in ome.structured_annotations.map_annotations[0].value.ms}
    assert notes["Gain"] == "2.0" and notes["AcquisitionAttributes/Laser/power"] == "5"
