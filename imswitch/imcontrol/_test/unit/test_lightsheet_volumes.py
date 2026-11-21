"""Lightsheet volumes: preview of what save_image writes, data-path guard, tile layout."""
import json
import struct

import numpy as np
import pytest

from imswitch.imcontrol.model.aufgaben.storage import save_image
from imswitch.imcontrol.model.io.lightsheet_volumes import (
    fov_um, list_volumes, pack_preview, resolve_data_path, tile_layout, volume_preview,
)


@pytest.mark.parametrize("suffix", [".ome.tif", ".ome.zarr"])
def test_preview_reads_both_formats_with_spacing(tmp_path, suffix):
    stack = np.arange(40 * 600 * 520, dtype=np.uint16).reshape(40, 600, 520)
    path = tmp_path / "run" / f"scan{suffix}"
    path.parent.mkdir()
    save_image(path, stack, "ZYX", pixel_size_um=0.5, z_step_um=4.0)

    header, vol = volume_preview(str(path), max_xy=128, max_z=16)

    assert vol.dtype == np.uint8 and list(vol.shape) == header["shape"]
    assert header["fullShape"] == [40, 600, 520]
    assert header["zStepUm"] == pytest.approx(4.0) and header["pixelSizeUm"] == pytest.approx(0.5)
    sz, sy, sx = header["stride"]
    assert vol.shape[0] == len(range(0, 40, sz)) and vol.shape[0] <= 16
    # voxel spacing of the preview = original spacing x pyramid factor x stride
    assert header["spacingUm"][0] == pytest.approx(4.0 * sz)
    assert header["spacingUm"][2] * vol.shape[2] == pytest.approx(0.5 * 520, rel=0.05)
    assert vol.max() == 255 and vol.min() == 0
    assert [v["name"] for v in list_volumes(str(tmp_path), str(tmp_path))] == [path.name]


def test_pack_preview_round_trip():
    vol = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    blob = pack_preview({"shape": [2, 3, 4]}, vol)
    n = struct.unpack("<I", blob[:4])[0]
    assert json.loads(blob[4:4 + n]) == {"shape": [2, 3, 4]}
    assert np.array_equal(np.frombuffer(blob[4 + n:], np.uint8).reshape(2, 3, 4), vol)


def test_resolve_data_path_stays_inside(tmp_path):
    (tmp_path / "a.tif").write_bytes(b"")
    assert resolve_data_path(str(tmp_path), "/a.tif") == str((tmp_path / "a.tif").resolve())
    with pytest.raises(ValueError):
        resolve_data_path(str(tmp_path), "../../etc/passwd")


def test_tile_layout_follows_the_calibration():
    xy = [(1000, 0), (1900, 0), (2800, 0)]           # 900 µm steps, 1000 px tiles at 1 µm/px
    offsets, canvas = tile_layout(xy, np.eye(2), (500, 1000))
    assert offsets == [(0, 0), (0, 900), (0, 1800)] and canvas == (500, 2800)

    flipped, _ = tile_layout(xy, [[-1, 0], [0, 1]], (500, 1000))   # camera x runs against stage X
    assert flipped == [(0, 1800), (0, 900), (0, 0)]

    turned, canvas = tile_layout(xy, [[0, 1], [1, 0]], (500, 1000))  # stage X moves along rows
    assert turned == [(0, 0), (900, 0), (1800, 0)] and canvas == (2300, 1000)
    assert fov_um([[0, 2], [2, 0]], (500, 1000)) == (1000.0, 2000.0)
