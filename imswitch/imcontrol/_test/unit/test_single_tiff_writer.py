"""SingleTiffWriter must keep every frame and produce a readable OME-TIFF.

It declared axes 'ZYX' for 2-D frames; tifffile then refused every write after
the first, so the file held one image and no OME-XML.
"""
import numpy as np
import tifffile

from imswitch.imcontrol.model.io.ome_writers.single_tiff_writer import SingleTiffWriter


def test_all_mono_frames_are_written_as_ome(tmp_path):
    path = tmp_path / "single_tiles.ome.tif"
    writer = SingleTiffWriter(str(path), isRGB=False)
    frames = [np.full((8, 10), i, dtype=np.uint16) for i in range(3)]
    for i, frame in enumerate(frames):
        writer.add_image(frame, {"x": 100.0 * i, "y": 5.0, "pixel_size": 0.5})
    writer.close()

    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome
        assert len(tif.series) == 3
        for i, series in enumerate(tif.series):
            assert series.shape == (8, 10)
            assert int(series.asarray()[0, 0]) == i
        assert 'PositionXUnit="µm"' in tif.ome_metadata
