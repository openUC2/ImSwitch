"""Aufgaben storage: what it writes must open as standard OME-Zarr / OME-TIFF."""
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from imswitch.imcontrol.model.aufgaben.storage import (
    FrameInfo, ImageSpec, NgffImage, OmeStore,
)

ome_zarr_models = pytest.importorskip("ome_zarr_models")
ome_types = pytest.importorskip("ome_types")
from ome_zarr_models import open_ome_zarr  # noqa: E402
from ome_zarr_models.v04.image import Image as Image04  # noqa: E402


def _spec(**kw):
    base = dict(shape=(2, 2, 3, 600, 520), pixel_size_um=0.5, z_step_um=2.0, t_step_s=30.0,
                origin_um=(100.0, -50.0, 10.0), channel_names=["LED", "Laser488"], name="field")
    return ImageSpec(**{**base, **kw})


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_image_is_valid_ngff(tmp_path, version):
    image = NgffImage.create(tmp_path / "a.ome.zarr", _spec(), version)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, size=(2, 2, 3, 600, 520), dtype=np.uint16)
    for t, c, z in np.ndindex(2, 2, 3):
        image.write(data[t, c, z], t, c, z)
    image.finalize(min_size=128)

    model = open_ome_zarr(zarr.open_group(str(tmp_path / "a.ome.zarr"), mode="r"))
    assert type(model).__module__.split(".")[1] == ("v04" if version == "0.4" else "v05")
    group = zarr.open_group(str(tmp_path / "a.ome.zarr"), mode="r")
    attrs = group.attrs.asdict() if version == "0.4" else group.attrs["ome"]
    ms = attrs["multiscales"][0]
    assert "coordinateTransformations" not in ms   # no scale on top of the datasets
    scales = [d["coordinateTransformations"][0]["scale"] for d in ms["datasets"]]
    assert scales[0] == [30.0, 1.0, 2.0, 0.5, 0.5] and scales[1][3:] == [1.0, 1.0]
    assert ms["datasets"][0]["coordinateTransformations"][1]["translation"] == [0, 0, 10, -50, 100]
    np.testing.assert_array_equal(group["0"][:], data)
    np.testing.assert_array_equal(group["1"][1, 1, 2], data[1, 1, 2, ::2, ::2])


def test_time_grows_and_tiles_land_in_a_canvas(tmp_path):
    image = NgffImage.create(tmp_path / "c.ome.zarr", _spec(shape=(1, 1, 1, 20, 30)))
    image.write(np.full((10, 15), 5, np.uint16), t=3, c=0, z=0, y0=10, x0=15)
    assert image.array.shape[0] == 4
    assert image.array[3, 0, 0, 10:, 15:].min() == 5 and image.array[3, 0, 0, :10].max() == 0


def test_ome_tiff_export_is_valid_ome(tmp_path):
    image = NgffImage.create(tmp_path / "t.ome.zarr", _spec(shape=(2, 2, 3, 64, 48)))
    for t, c, z in np.ndindex(2, 2, 3):
        image.write(np.full((64, 48), 100 * t + 10 * c + z, np.uint16), t, c, z)
    path = image.export_ome_tiff(tmp_path / "t.ome.tif")
    ome = ome_types.from_tiff(path)
    pixels = ome.images[0].pixels
    assert (pixels.size_t, pixels.size_c, pixels.size_z) == (2, 2, 3)
    assert pixels.physical_size_x == 0.5 and pixels.physical_size_z == 2.0
    assert [c.name for c in pixels.channels] == ["LED", "Laser488"]
    data = tifffile.imread(path)
    assert data.shape == (2, 2, 3, 64, 48) and data[1, 1, 2, 0, 0] == 112


def _info(node="img", t=0, c=0, z=0, pos=None, tile=None, g=None):
    index = {"t": t, "c": c, "z": z}
    if g is not None:
        index["g"] = g
    return FrameInfo(node_id=node, index=index, sizes={"t": 2, "c": 2, "z": 1},
                     channel_names=["LED", "Laser488"], pixel_size_um=1.0,
                     position=pos, tile=tile, z_um=5.0, z_step_um=1.0, t_step_s=10.0)


def test_store_writes_one_image_per_position_and_exports_tiff(tmp_path):
    store = OmeStore(tmp_path, formats=("ome-zarr", "ome-tiff"))
    for name, x in (("a", 0.0), ("b", 500.0)):
        for t, c in np.ndindex(2, 2):
            store.write(np.full((32, 40), t * 10 + c, np.uint16),
                        _info(t=t, c=c, pos={"name": name, "x": x, "y": 0.0, "z": 5.0}))
    tiffs = store.close()
    assert sorted(p.name for p in tiffs) == ["a_img.ome.tif", "b_img.ome.tif"]
    group = zarr.open_group(str(tmp_path / "b_img.ome.zarr"), mode="r")
    Image04.from_zarr(group)
    assert group["0"].shape == (2, 2, 1, 32, 40) and group["0"][1, 1, 0, 0, 0] == 11
    translation = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][1]
    assert translation["translation"] == [0, 0, 5.0, -16.0, 480.0]   # centre minus half a field


def test_store_writes_a_plate_for_wells(tmp_path):
    store = OmeStore(tmp_path)
    for (row, col, name) in ((0, 0, "A1"), (1, 2, "B3")):
        for g in range(2):
            pos = {"name": name, "x": 0.0, "y": 0.0, "z": None, "row": row, "col": col}
            store.write(np.ones((16, 16), np.uint16), _info(pos=pos, g=g,
                        tile={"name": f"{g:04d}", "x": 10.0 * g, "y": 0.0}))
    store.close()
    plate_group = zarr.open_group(str(tmp_path / "img.ome.zarr"), mode="r")
    assert type(open_ome_zarr(plate_group)).__name__ == "HCS"
    attrs = plate_group.attrs["plate"]
    assert [w["path"] for w in attrs["wells"]] == ["A/1", "B/3"] and attrs["field_count"] == 2
    Image04.from_zarr(plate_group["B/3/1"])


def test_rgb_frames_become_three_channels(tmp_path):
    store = OmeStore(tmp_path)
    info = FrameInfo(node_id="rgb", index={"c": 0}, sizes={"c": 1}, channel_names=["BF"],
                     pixel_size_um=1.0)
    frame = np.zeros((8, 8, 3), np.uint8)
    frame[..., 2] = 200
    store.write(frame, info)
    store.close()
    group = zarr.open_group(str(tmp_path / "rgb.ome.zarr"), mode="r")
    assert group["0"].shape == (1, 3, 1, 8, 8) and group["0"][0, 2, 0].max() == 200
    assert [c["color"] for c in group.attrs["omero"]["channels"]] == ["FF0000", "00FF00", "0000FF"]


def test_recording_writer_output_is_valid_ngff(tmp_path):
    """RecordingService's OME-Zarr writer shares NgffImage."""
    from imswitch.imcontrol.model.writers.base import DetectorContext, SessionContext
    from imswitch.imcontrol.model.writers.ome_zarr_writer import OMEZarrWriter

    session = SessionContext(session_id="s1", base_path=str(tmp_path / "data.ome.zarr"),
                             n_time_points=5, time_interval_s=0.5)
    writer = OMEZarrWriter(session)
    writer.open({"cam": DetectorContext(name="cam", shape_px=(40, 30), pixel_size_um=0.65)})
    for t in range(3):
        writer.write("cam", np.full((30, 40), t, np.uint16))
    writer.finalize()
    group = zarr.open_group(str(tmp_path / "data.ome.zarr" / "cam"), mode="r")
    Image04.from_zarr(group)
    assert group["0"].shape == (3, 1, 1, 30, 40) and int(group["0"][2, 0, 0, 0, 0]) == 2
    assert group.attrs["acquisition_complete"] is True


@pytest.mark.parametrize("rgb", [False, True])
def test_experiment_writer_output_is_valid_ngff(tmp_path, rgb):
    """ExperimentController's OMEWriter shares NgffImage: a 2x2 mosaic validates as NGFF 0.4."""
    from imswitch.imcontrol.model.io.ome_writers.ome_writer import (
        OMEFileStorePaths, OMEWriter, OMEWriterConfig,
    )

    cfg = OMEWriterConfig(write_zarr=True, write_tiff=True, pixel_size=0.5, pixel_size_z=3.0,
                          n_z_planes=2,
                          n_channels=2, channel_names=["LED", "Laser488"], time_interval=60.0,
                          z_start=7.0)
    paths = OMEFileStorePaths(str(tmp_path / "run"))
    writer = OMEWriter(paths, (64, 80), (2, 2), (100.0, 200.0, 40.0, 32.0), cfg, isRGB=rgb)
    chunks = []
    for iy, ix, c, z in np.ndindex(2, 2, 2, 2):
        value = 1 + ix + 2 * iy + 4 * c + 8 * z
        frame = np.full((64, 80, 3) if rgb else (64, 80), value, np.uint8 if rgb else np.uint16)
        chunks.append(writer.write_frame(frame, {"x": 100.0 + 40 * ix, "y": 200.0 + 32 * iy,
                                                 "channel_index": c, "z_index": z})["rel_chunk"])
    writer.finalize()

    group = zarr.open_group(writer.store, mode="r")
    Image04.from_zarr(group)
    assert all((Path(writer.store) / key).is_file() for key in chunks)   # keys sent to the UI
    level0 = group["0"]
    assert level0.shape == (1, 6 if rgb else 2, 2, 128, 160)
    c = 3 if rgb else 1                                   # channel 1 starts here
    assert int(level0[0, c, 1, 64, 80]) == 1 + 1 + 2 + 4 + 8   # tile (1, 1), channel 1, z 1
    transforms = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert transforms[0]["scale"] == [60.0, 1.0, 3.0, 0.5, 0.5]
    assert transforms[1]["translation"] == [0.0, 0.0, 7.0, 184.0, 80.0]   # top-left pixel corner

    mosaic = tmp_path / "run" / "run.ome.tif"   # the OME-TIFF hyperstack, same metadata model
    pixels = ome_types.from_tiff(mosaic).images[0].pixels
    assert [c.name for c in pixels.channels] == ["LED", "Laser488"]
    assert pixels.size_c == (6 if rgb else 2) and pixels.physical_size_z == 3.0
    assert pixels.planes[0].position_x == 80.0
    with tifffile.TiffFile(mosaic) as f:
        assert f.series[0].shape[-3 if rgb else -2:] == ((128, 160, 3) if rgb else (128, 160))


def test_large_ome_tiff_export_streams_tiles_and_keeps_the_pyramid(tmp_path):
    """Planes wider than one 512 px TIFF tile: tifffile needs tiles, not planes."""
    data = np.random.default_rng(3).integers(0, 4000, (1, 2, 2, 700, 1100), dtype=np.uint16)
    spec = ImageSpec(shape=data.shape, pixel_size_um=0.5, channel_names=["a", "b"],
                     exposure_ms=[5.0, 20.0], name="big")
    image = NgffImage.create(tmp_path / "big.ome.zarr", spec)
    for t, c, z in np.ndindex(*data.shape[:3]):
        image.write(data[t, c, z], t, c, z)
    image.finalize(min_size=256)
    path = image.export_ome_tiff(tmp_path / "big.ome.tif")
    with tifffile.TiffFile(path) as f:
        series = f.series[0]
        assert series.axes == "CZYX" and len(series.levels) == 2
        assert np.array_equal(series.asarray(), data[0])
        assert np.array_equal(series.levels[1].asarray(), data[0, ..., ::2, ::2])
    planes = ome_types.from_tiff(path).images[0].pixels.planes
    assert [p.exposure_time for p in planes] == [5.0, 5.0, 20.0, 20.0]


@pytest.mark.parametrize("rgb", [False, True])
def test_single_plane_tiff_metadata(tmp_path, rgb):
    """Experiment tiles and snaps: one plane, positions in µm, exposure in ms."""
    from imswitch.imcontrol.model.io.ome_writers.ome_tiff_metadata import (
        build_ome_metadata_from_dict,
    )
    frame = np.zeros((30, 40, 3) if rgb else (30, 40), np.uint8 if rgb else np.uint16)
    xml = build_ome_metadata_from_dict({"height": 30, "width": 40, "dtype": str(frame.dtype),
                                        "pixel_size": 0.5, "x": 1000.0, "y": 2000.0, "z": 3.0,
                                        "channel_name": "LED", "exposure": 12, "rgb": rgb})
    path = tmp_path / "tile.tif"
    tifffile.imwrite(path, frame, description=xml, metadata=None,
                     photometric="rgb" if rgb else None)
    pixels = ome_types.from_tiff(path).images[0].pixels
    plane = pixels.planes[0]
    assert (plane.position_x, plane.position_y, plane.exposure_time) == (990.0, 1992.5, 12.0)
    assert [c.name for c in pixels.channels] == ["LED"]
    assert pixels.channels[0].samples_per_pixel == (3 if rgb else 1)


def test_save_image_axes_formats_and_unknown_metadata(tmp_path):
    from imswitch.imcontrol.model.aufgaben.storage import save_image
    zyx = np.arange(3 * 20 * 30, dtype=np.uint16).reshape(3, 20, 30)
    tif = save_image(tmp_path / "stack.ome.tif", zyx, "ZYX", z_step_um=2.0)
    pixels = ome_types.from_tiff(tif).images[0].pixels
    assert (pixels.size_z, pixels.size_c) == (3, 1) and pixels.physical_size_x is None
    assert pixels.planes[0].position_x is None                   # unknown: left out
    assert np.array_equal(tifffile.imread(tif), zyx)
    yxs = np.zeros((20, 30, 3), np.uint8)
    yxs[..., 1] = 7
    store = save_image(tmp_path / "rgb.ome.zarr", yxs, pixel_size_um=0.5, channel_names=["bf"])
    group = zarr.open_group(str(store), mode="r")
    open_ome_zarr(group)
    assert group["0"].shape == (1, 3, 1, 20, 30) and int(group["0"][0, 1, 0, 0, 0]) == 7
    assert [c["label"] for c in group.attrs["omero"]["channels"]] == ["bf_rgb_r", "bf_rgb_g",
                                                                    "bf_rgb_b"]
    with pytest.raises(ValueError, match="do not describe"):
        save_image(tmp_path / "bad.ome.tif", zyx)                  # 3-D needs axes
    with pytest.raises(ValueError, match="use .tif"):
        save_image(tmp_path / "bad.png", zyx[0])
