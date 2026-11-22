"""An OME-TIFF read through tiffzarr gives the pixels that were written, level by level."""
import json

import numpy as np
import pytest
import tifffile

from imswitch.imcontrol.model.aufgaben import tiffzarr
from imswitch.imcontrol.model.aufgaben.storage import save_image


def _level(path, name: str) -> np.ndarray:
    """Assemble one level from its chunks, the way a zarr client does."""
    meta = json.loads(tiffzarr.read(path, f"{name}/.zarray"))
    shape, chunks = meta["shape"], meta["chunks"]
    out = np.zeros([-(-s // c) * c for s, c in zip(shape, chunks)], meta["dtype"])
    for index in np.ndindex(*[-(-s // c) for s, c in zip(shape, chunks)]):
        chunk = np.frombuffer(tiffzarr.read(path, f"{name}/" + ".".join(map(str, index))),
                              meta["dtype"]).reshape(chunks)
        out[tuple(slice(i * c, (i + 1) * c) for i, c in zip(index, chunks))] = chunk
    return out[tuple(slice(0, s) for s in shape)]


def test_pyramid_ome_tiff_reads_back_with_its_metadata(tmp_path):
    data = np.random.default_rng(0).integers(0, 4000, (2, 3, 2, 600, 1100), dtype=np.uint16)
    path = save_image(tmp_path / "a.ome.tif", data, "TCZYX", pixel_size_um=0.5, z_step_um=2.0,
                      channel_names=["DAPI", "GFP", "Cy5"])

    attrs = json.loads(tiffzarr.read(path, ".zattrs"))
    image = attrs["multiscales"][0]
    assert [a["name"] for a in image["axes"]] == list("tczyx")
    assert [d["coordinateTransformations"][0]["scale"] for d in image["datasets"]] == [
        [1.0, 1.0, 2.0, 0.5, 0.5], [1.0, 1.0, 2.0, 1.0, 1.0]]
    assert [c["label"] for c in attrs["omero"]["channels"]] == ["DAPI", "GFP", "Cy5"]
    assert json.loads(tiffzarr.read(path, ".zgroup")) == {"zarr_format": 2}
    np.testing.assert_array_equal(_level(path, "0"), data)
    np.testing.assert_array_equal(_level(path, "1"), data[..., ::2, ::2])

    for key in ("0/0.0.0.9.0", "0/2.0.0.0.0", "0/0.0", "5/.zarray", "0/x", "nothing"):
        with pytest.raises(KeyError):
            tiffzarr.read(path, key)


def test_rgb_becomes_three_channels_and_strips_are_chunks(tmp_path):
    rgb = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    path = save_image(tmp_path / "rgb.ome.tif", rgb, "YXS")
    axes = json.loads(tiffzarr.read(path, ".zattrs"))["multiscales"][0]["axes"]
    assert [a["name"] for a in axes] == ["c", "y", "x"]
    np.testing.assert_array_equal(_level(path, "0"), rgb.transpose(2, 0, 1))

    stack = np.random.default_rng(2).random((3, 70, 50)).astype(np.float32)
    tifffile.imwrite(tmp_path / "strips.tif", stack, photometric="minisblack", rowsperstrip=32,
                     metadata={"axes": "ZYX"})
    np.testing.assert_array_equal(_level(tmp_path / "strips.tif", "0"), stack)


def test_a_layout_it_cannot_read_is_refused(tmp_path):
    tifffile.imwrite(tmp_path / "planar.tif", np.zeros((3, 40, 40), np.uint8), photometric="rgb",
                     planarconfig="separate")
    with pytest.raises(ValueError):
        tiffzarr.read(tmp_path / "planar.tif", ".zattrs")
