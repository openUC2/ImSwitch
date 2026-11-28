"""Flat-field reference: estimate, refuse what is not blank glass, correct tiles."""
import numpy as np
import pytest

from imswitch.imcontrol.model import flatfield as ff

H, W = 96, 128


def _shading(falloff=(0.40, 0.30, 0.50)):
    """Per-channel radial vignetting, 1 in the centre."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    return np.stack([1.0 - f * r2 / 2.0 for f in falloff], axis=-1).astype(np.float32)


def _glass_frames(n=9, level=200.0, dark=4.0, seed=0, debris=True):
    rng = np.random.default_rng(seed)
    shading = _shading()
    frames = []
    for i in range(n):
        img = dark + level * shading + rng.normal(0, 2.0, shading.shape)
        if debris:   # a speck of dirt on the slide, somewhere else in every frame
            y, x = 10 + 7 * i, 15 + 11 * i
            img[y:y + 4, x:x + 4] *= 0.5
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames, shading


def test_reference_from_blank_glass_flattens_a_glass_tile():
    frames, _ = _glass_frames()
    dark = np.full((H, W, 3), 4, np.uint8)
    flat, report = ff.estimate(frames, dark=dark, sigma_px=2.0)
    assert report["ok"], report
    assert flat.shape == (H, W) and flat.channels == 3
    assert 0.3 < report["falloff"] < 0.5          # deepest channel: corners at 1 - 0.5
    assert report["falloff_per_channel"][1] < report["falloff_per_channel"][2]
    tile, _ = _glass_frames(n=1, seed=7, debris=False)
    corrected = flat.apply(tile[0]).astype(np.float32)
    raw = tile[0].astype(np.float32)
    # raw: corners clearly darker than the centre; corrected: flat within noise
    assert raw[..., 2].min() < 0.8 * raw[..., 2].max()
    inner = corrected[4:-4, 4:-4]
    for c in range(3):
        spread = np.percentile(inner[..., c], 99) - np.percentile(inner[..., c], 1)
        assert spread < 16, (c, spread)
    # the centre keeps its value: nothing is brightened beyond what the camera gave there
    assert corrected[H // 2, W // 2].mean() == pytest.approx(raw[H // 2, W // 2].mean(), abs=4)


def test_debris_that_moves_between_frames_is_not_in_the_reference():
    frames, shading = _glass_frames()
    flat, report = ff.estimate(frames, sigma_px=1.0)
    assert report["ok"]
    # the speck of frame 0 sits at (10..14, 15..19): no dip there in the reference
    patch = flat.shading[10:14, 15:19] / (shading[10:14, 15:19] / shading.max(axis=(0, 1)))
    assert np.all(np.abs(patch - 1.0) < 0.08)


def test_tissue_in_the_field_is_refused():
    rng = np.random.default_rng(1)
    texture = rng.uniform(0.3, 1.0, (H // 8, W // 8))
    tissue = np.kron(texture, np.ones((8, 8)))
    frames = []
    for i in range(9):   # the stage moves; the tissue moves with it
        shifted = np.roll(tissue, (3 * i, 5 * i), axis=(0, 1))
        frames.append(np.clip(200 * shifted[..., None] * _shading(), 0, 255).astype(np.uint8))
    flat, report = ff.estimate(frames)
    assert flat is None and not report["ok"]
    assert "blank glass" in report["reason"]


def test_overexposed_and_dark_references_are_refused():
    frames, _ = _glass_frames()
    flat, report = ff.estimate([np.full_like(frames[0], 255)] * 3)
    assert flat is None and "saturated" in report["reason"]
    flat, report = ff.estimate([np.full_like(frames[0], 5)] * 3)
    assert flat is None and "too dark" in report["reason"]


def test_opencv_and_numpy_paths_agree():
    frames, _ = _glass_frames()
    flat, _ = ff.estimate(frames, dark=np.full((H, W, 3), 4, np.uint8), sigma_px=2.0)
    tile = frames[3]
    fast = flat.apply(tile)
    saved = ff.HAS_CV2
    try:
        ff.HAS_CV2 = False
        slow = flat.apply(tile)
    finally:
        ff.HAS_CV2 = saved
    assert fast.dtype == np.uint8 and fast.shape == tile.shape
    assert np.abs(fast.astype(int) - slow.astype(int)).max() <= 1


def test_mono_uint16_frames_and_mismatched_frames():
    rng = np.random.default_rng(2)
    shading = _shading((0.3, 0.3, 0.3))[..., 0]
    frames = [(100 + 3000 * shading + rng.normal(0, 10, shading.shape)).astype(np.uint16)
              for _ in range(5)]
    flat, report = ff.estimate(frames, dark=np.full((H, W), 100, np.uint16))
    assert report["ok"] and flat.channels == 1
    out = flat.apply(frames[0])
    assert out.dtype == np.uint16 and out.shape == (H, W)
    assert np.percentile(out, 99) - np.percentile(out, 1) < 0.1 * 3000
    # another size or a colour frame: returned untouched, never misapplied
    other = np.zeros((H, W + 2), np.uint16)
    assert flat.apply(other) is other


def test_save_and_load_round_trip(tmp_path):
    frames, _ = _glass_frames()
    flat, _ = ff.estimate(frames, dark=np.full((H, W, 3), 4, np.uint8))
    flat.meta["objective"] = "4x"
    flat.save(str(tmp_path), pixel_size_um=0.6)
    back = ff.FlatField.load(str(tmp_path))
    assert back.shape == flat.shape and back.channels == 3
    assert np.allclose(back.shading, flat.shading, atol=1e-5)
    assert np.allclose(back.dark, flat.dark)
    assert back.meta["objective"] == "4x"
    assert np.array_equal(back.apply(frames[0]), flat.apply(frames[0]))


def test_reference_offsets_start_at_the_spot_and_spread_around_it():
    offs = ff.reference_offsets(9, 150.0)
    assert offs[0] == (0.0, 0.0)
    assert len(offs) == 9 and len(set(offs)) == 9
    assert max(abs(x) for x, _ in offs) == 150.0
    assert ff.reference_offsets(1, 150.0) == [(0.0, 0.0)]


def test_reference_key_is_a_safe_folder_name():
    key = ff.reference_key("Widefield Camera", "4x/0.1", (2048, 3072), 1)
    assert "/" not in key and " " not in key
    assert key.endswith("2048x3072__bin1")


def test_writer_corrects_the_stitched_canvas_and_keeps_the_tiles_raw(tmp_path):
    """Two vignetted glass tiles: a flat canvas, raw individual TIFFs, other lights untouched."""
    import glob

    import tifffile
    import zarr

    from imswitch.imcontrol.model.io.ome_writers.ome_writer import (
        OMEFileStorePaths, OMEWriter, OMEWriterConfig)

    frames, shading = _glass_frames(debris=False)
    flat, report = ff.estimate(frames, sigma_px=2.0)
    assert report["ok"]
    positions = [(0.0, 0.0), (W * 0.5, 0.0)]
    cfg = OMEWriterConfig(write_zarr=True, write_individual_tiffs=True, pixel_size=0.5,
                          channel_names=["LED"])
    writer = OMEWriter(OMEFileStorePaths(str(tmp_path / "run")), (H, W), (2, 1),
                       (0.0, 0.0, W * 0.5, H * 0.5), cfg, isRGB=True,
                       tile_positions=positions, flat_fields={"LED": flat})
    tile = (200.0 * shading).astype(np.uint8)          # glass, vignetted like the reference
    for x, y in positions:
        writer.write_frame(tile, {"x": x, "y": y, "illuminationChannel": "LED"})
    writer.finalize()

    canvas = np.moveaxis(zarr.open_group(writer.store, mode="r")["0"][0, :, 0], 0, -1)
    for c in range(3):
        spread = np.percentile(canvas[2:-2, 2:-2, c], 99) - np.percentile(canvas[2:-2, 2:-2, c], 1)
        assert spread < 12, (c, spread)                # flat glass, no tile grid
    raw = [tifffile.imread(f) for f in sorted(glob.glob(str(tmp_path / "run" / "**" / "*.tif"),
                                                        recursive=True))]
    assert raw and any(np.array_equal(np.asarray(r).reshape(tile.shape), tile) for r in raw)

    other = OMEWriter(OMEFileStorePaths(str(tmp_path / "other")), (H, W), (1, 1),
                      (0.0, 0.0, W * 0.5, H * 0.5),
                      OMEWriterConfig(write_zarr=True, pixel_size=0.5, channel_names=["Laser"]),
                      isRGB=True, tile_positions=[(0.0, 0.0)], flat_fields={"LED": flat})
    other.write_frame(tile, {"x": 0.0, "y": 0.0, "illuminationChannel": "Laser"})
    other.finalize()
    untouched = np.moveaxis(zarr.open_group(other.store, mode="r")["0"][0, :, 0], 0, -1)
    assert np.array_equal(untouched, tile)


class _FakeScope:
    """Just enough of ExperimentController for FlatFieldMixin: glass under a vignetting lamp."""

    def __init__(self):
        from imswitch.imcontrol.controller.controllers.experiment_controller.flatfield_api import (
            FlatFieldMixin)
        self.__class__ = type("FakeScope", (_FakeScope, FlatFieldMixin), {})
        quiet = lambda *a, **k: None  # noqa: E731
        self._logger = type("L", (), {"info": quiet, "warning": quiet, "debug": quiet,
                                      "error": quiet})()
        self.rng = np.random.default_rng(5)
        self.light = {}
        self.switched = []
        self.moves = []
        self.mDetector = type("D", (), {"name": "Cam", "parameters": {}, "_running": True,
                                        "getLatestFrame": lambda _self: self.grabCameraFrame()})()
        self.mStage = type("S", (), {"getPosition": lambda _self: {"X": 0.0, "Y": 0.0}})()
        self.detectorPixelSize = [1.0, 0.6, 0.6]

    def grabCameraFrame(self, frameSync=2):
        level = 200.0 if self.light else 0.0
        img = 4 + level * _shading() + self.rng.normal(0, 1.5, (H, W, 3))
        return np.clip(img, 0, 255).astype(np.uint8)

    def move_stage_xy(self, x, y, relative=False):
        self.moves.append((x, y))

    def set_laser_power(self, power, channel):
        self.light[channel] = power
        self.switched.append((channel, power))

    def _switch_off_all_illumination(self):
        self.light.clear()

    def set_exposure_time_gain(self, *args):
        pass

    def _announceDisplay(self, *args):
        pass

    def _cameraSnapshot(self):
        return {"detector": "Cam"}


def _run_context(tmp_path, name, x=None, y=None, sources=("LED",), intensities=(1023,),
                 passthrough=False):
    import types
    p = types.SimpleNamespace(flatField=True, flatFieldX=x, flatFieldY=y, flatFieldFrames=9,
                              flatFieldSpacingUm=150.0)
    run = tmp_path / name
    run.mkdir()
    return types.SimpleNamespace(
        experiment=types.SimpleNamespace(parameterValue=p), illumination_sources=list(sources),
        illumination_kinds=["default"] * len(sources), exposures=[0.3] * len(sources),
        gains=[0.0] * len(sources), illumination_intensities=list(intensities),
        illumination_passthrough=passthrough, dir_path=str(run))


def _patch_store(tmp_path, monkeypatch):
    from imswitch.imcontrol.controller.controllers.experiment_controller import flatfield_api
    from imswitch.imcontrol.model.aufgaben import storage
    monkeypatch.setattr(flatfield_api.FlatFieldMixin, "_flatFieldRoot",
                        staticmethod(lambda: str(tmp_path / "config" / "flatfield")))
    monkeypatch.setattr(storage, "current_objective", lambda: {"name": "4x", "na": 0.1})


def test_a_run_takes_its_reference_on_glass_and_the_next_run_reuses_it(tmp_path, monkeypatch):
    from imswitch.imcontrol.controller.controllers.experiment_controller import flatfield_api
    from imswitch.imcontrol.model.aufgaben import storage
    monkeypatch.setattr(flatfield_api.FlatFieldMixin, "_flatFieldRoot",
                        staticmethod(lambda: str(tmp_path / "config" / "flatfield")))
    monkeypatch.setattr(storage, "current_objective", lambda: {"name": "4x", "na": 0.1})
    scope = _FakeScope()

    first = _run_context(tmp_path, "run1", x=1000.0, y=2000.0)
    scope._prepareRunFlatFields(first)
    info = first.flat_field_info["channels"]["LED"]
    assert first.flat_field_info["applied"] and info["source"] == "taken at the start of this run"
    assert len(scope.moves) == 9 and scope.moves[0] == (1000.0, 2000.0)
    assert max(abs(x - 1000.0) for x, _ in scope.moves) == 150.0
    assert scope.light == {}                                  # the lamp is off again
    flat = first.flat_fields["LED"]
    assert flat.dark == pytest.approx([4, 4, 4], abs=1)       # dark frame with the lamp off
    assert (tmp_path / "run1" / "flatfield_LED" / ff.FILE_NAME).is_file()
    stored = scope.getFlatFieldStatus()
    assert stored["objective"] == "4x" and len(stored["current"]) == 1

    scope.moves.clear()
    second = _run_context(tmp_path, "run2")
    scope._prepareRunFlatFields(second)
    assert second.flat_field_info["channels"]["LED"]["source"] == "stored reference"
    assert scope.moves == [] and "LED" in second.flat_fields

    monkeypatch.setattr(storage, "current_objective", lambda: {"name": "20x", "na": 0.4})
    third = _run_context(tmp_path, "run3")
    scope._prepareRunFlatFields(third)
    assert not third.flat_field_info["applied"]
    assert "no reference for 20x" in third.flat_field_info["channels"]["LED"]["reason"]


def test_the_run_reference_is_taken_with_the_first_light_that_is_on(tmp_path, monkeypatch):
    _patch_store(tmp_path, monkeypatch)
    scope = _FakeScope()
    run = _run_context(tmp_path, "run", x=0.0, y=0.0, sources=("LED", "Laser"),
                       intensities=(0, 700))
    scope._prepareRunFlatFields(run)
    assert list(run.flat_field_info["channels"]) == ["Laser"]   # the LED is not used
    assert scope.switched == [("Laser", 700)] and scope.light == {}
    assert run.flat_field_info["channels"]["Laser"]["source"] == "taken at the start of this run"


def test_a_passthrough_run_keeps_the_light_set_by_hand(tmp_path, monkeypatch):
    # no light in the experiment: the operator lit the sample; power 1 is a sentinel
    _patch_store(tmp_path, monkeypatch)
    scope = _FakeScope()
    scope.light = {"LED": 800}
    run = _run_context(tmp_path, "run", x=0.0, y=0.0, intensities=(1,), passthrough=True)
    scope._prepareRunFlatFields(run)
    assert scope.switched == [] and scope.light == {"LED": 800}   # never dark, never 1
    assert run.flat_field_info["applied"] and "LED" in run.flat_fields
