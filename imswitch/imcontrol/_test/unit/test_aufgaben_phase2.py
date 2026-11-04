"""Aufgaben Phase 2 on the simulated microscope: focus maps, regions, light, scans, useq."""
import numpy as np
import pytest
import useq
import zarr
from ome_zarr_models import open_ome_zarr

from imswitch.imcontrol.model import aufgaben as A
from imswitch.imcontrol.model.aufgaben.engine import Run
from imswitch.imcontrol.model.aufgaben.simulated import SimulatedMicroscope
from imswitch.imcontrol.model.aufgaben.storage import ImageSpec, NgffImage, OmeStore
from imswitch.imcontrol.model.aufgaben.stitching import stitch

SOURCES = ("LED", "Laser488")


def tilt(x, y):
    return 100.0 + 0.01 * x + 0.02 * y


def _job(tasks, **kw):
    return A.Job.model_validate({"name": "t", **kw, "tasks": tasks})


def _run(job, hw=None, **kw):
    run = Run(job, hw or SimulatedMicroscope(sources=SOURCES), **kw)
    run.run(confirmed=True)
    return run


def _snaps(hw):
    return [c for c in hw.calls if c[0] == "snap"]


REGIONS = {"kind": "regions", "regions": [
    {"name": "left", "min_x": 0, "max_x": 400, "min_y": 0, "max_y": 300},
    {"name": "right", "min_x": 1000, "max_x": 1400, "min_y": 0, "max_y": 300}]}
AF = {"kind": "autofocus", "channel": "LED", "intensity": 50, "range_um": 20, "step_um": 2}
CAP = {"kind": "capture", "channels": [{"config": "LED", "intensity": 10, "exposure": 5}]}


# --------------------------------------------------------------------------- #
# Focus maps and regions (2.2)
# --------------------------------------------------------------------------- #
def test_region_tiles_follow_the_measured_focus_surface(tmp_path):
    hw = SimulatedMicroscope(sources=SOURCES, frame_shape=(100, 120), surface=tilt)
    job = _job([
        {"id": "map", "task": "create_focus_map",
         "params": {"focus_map": "@fm", "autofocus": "@af", "regions": "@regions"}},
        {"id": "reg", "task": "loop_regions", "params": {"regions": "@regions"}, "body": [
            {"id": "tiles", "task": "loop_grid", "params": {"overlap": 0.1, "focus_map": "@fm"},
             "body": [{"id": "img", "task": "capture", "params": {"channels": "@cap"}}]}]}],
        resources={"regions": REGIONS, "af": AF, "fm": {"kind": "focus_map", "method": "spline"},
                   "cap": CAP})
    report = A.validate(job)
    assert report.ok and report.moves_z and report.emits_light
    run = _run(job, hw, sink=OmeStore(tmp_path))
    assert run.state == "finished", run.message
    assert run.results["map"]["measured"] == 18 and set(run.results["map"]["groups"]) == \
        {"left", "right"}
    snaps = _snaps(hw)
    assert len(snaps) == 2 * 4 * 4   # per region: 4 x 4 tiles of 120 x 100 µm (10 % overlap)
    for _, _, pos, exposure, lit in snaps:
        assert pos["Z"] == pytest.approx(tilt(pos["X"], pos["Y"]), abs=0.05)
        assert exposure == 5 and lit == ("LED",)
    xs = sorted({round(s[2]["X"]) for s in snaps if s[2]["X"] < 1000})
    assert xs[0] == 60 and xs[-1] - xs[0] == pytest.approx(3 * 108, abs=1)


def test_focus_map_inside_a_well_loop_and_move_to_surface():
    hw = SimulatedMicroscope(sources=SOURCES, surface=tilt)
    plate = {"kind": "well_plate", "plan": {"plate": "96-well", "a1_center_xy": [0, 0],
                                            "selected_wells": [[0, 1], [0, 1]]}}
    job = _job([{"id": "wells", "task": "loop_wells", "params": {"plate": "@plate"}, "body": [
        {"id": "map", "task": "create_focus_map",
         "params": {"focus_map": "@fm", "autofocus": "@af"}},
        {"id": "away", "task": "move_relative", "params": {"dz_um": 30}},
        {"id": "back", "task": "move_to_focus_surface", "params": {"focus_map": "@fm"}}]}],
        resources={"plate": plate, "af": AF, "fm": {"kind": "focus_map", "z_offset_um": 2}})
    run = _run(job, hw)
    assert run.state == "finished", run.message
    assert run.results["back"] == pytest.approx(tilt(9000, -9000) + 2)   # B2, rows go to -Y
    assert set(run.surfaces["fm"]) == {"A1", "B2"}


def test_too_few_focus_points_fail_the_node():
    hw = SimulatedMicroscope(sources=SOURCES)
    hw.autofocus = lambda *a, **k: None
    job = _job([{"id": "map", "task": "create_focus_map", "params": {
        "focus_map": "@fm", "autofocus": "@af", "points": "@pts", "min_points": 2}}],
        resources={"af": AF, "fm": {"kind": "focus_map"},
                   "pts": {"kind": "positions", "positions": [{"x": 0, "y": 0}, {"x": 9, "y": 9}]}})
    run = _run(job, hw)
    assert run.state == "failed" and "0 of 2 focus measurements" in run.message


def test_structural_rules_for_the_new_nodes():
    job = _job([{"id": "g", "task": "loop_grid", "params": {}, "body": [
        {"id": "m", "task": "create_focus_map",
         "params": {"focus_map": "@fm", "autofocus": "@af"}}]}],
        resources={"af": AF, "fm": {"kind": "focus_map"}})
    messages = [i.message for i in A.validate(job).errors]
    assert any("needs an enclosing loop_regions" in m for m in messages)
    assert any("create_focus_map needs points" in m for m in messages)


# --------------------------------------------------------------------------- #
# Capture depth (2.1)
# --------------------------------------------------------------------------- #
def _light_calls(hw):
    return [c[1:] for c in hw.calls if c[0] in ("light", "led_pattern")]


@pytest.mark.parametrize("keep", [False, True])
def test_keep_light_on_between_channels(keep):
    hw = SimulatedMicroscope(sources=SOURCES)
    cap = {"kind": "capture", "keep_light_on_between_channels": keep, "channels": [
        {"config": "LED", "intensity": 10}, {"config": "LED", "intensity": 10},
        {"config": "Laser488", "intensity": 20}]}
    _run(_job([{"id": "img", "task": "capture", "params": {"channels": "@cap"}}],
              resources={"cap": cap}), hw)
    calls = _light_calls(hw)
    if keep:   # same light twice: one switch-on; a new source switches the old one off first
        assert calls == [("LED", 10, True), ("LED", None, False), ("Laser488", 20, True),
                         ("Laser488", None, False)]
    else:
        assert len(calls) == 6
    assert [s[4] for s in _snaps(hw)] == [("LED",), ("LED",), ("Laser488",)]


def test_ring_and_dpc_channels_drive_the_led_matrix():
    hw = SimulatedMicroscope(sources=SOURCES)
    channels = [{"config": "ring", "kind": "ring", "led_pattern": {"radius": 4, "intensity_g": 0}},
                {"config": "dpc_top", "kind": "dpc", "led_pattern": {"direction": "top"}}]
    _run(_job([{"id": "img", "task": "capture", "params": {"channels": channels}}]), hw)
    calls = _light_calls(hw)
    assert calls[0][0] == "ring" and calls[0][1]["radius"] == 4 and calls[0][1]["intensity_g"] == 0
    assert calls[1] == ("off", {}) and calls[2][0] == "dpc"
    assert calls[2][1]["direction"] == "top" and calls[3] == ("off", {})
    with pytest.raises(Exception, match="needs led_pattern.radius"):
        A.CaptureChannel(config="ring", kind="ring")


def test_frames_record_exposure_and_bit_depth(tmp_path):
    cap = {"kind": "capture", "channels": [{"config": "LED", "intensity": 10, "exposure": 7},
                                            {"config": "Laser488", "intensity": 9, "exposure": 30}]}
    _run(_job([{"id": "img", "task": "capture", "params": {"channels": "@cap"}}],
              resources={"cap": cap}, storage={"formats": ["ome-zarr", "ome-tiff"]}),
         sink=OmeStore(tmp_path, formats=("ome-zarr", "ome-tiff")))
    import ome_types
    pixels = ome_types.from_tiff(next(tmp_path.glob("*.ome.tif"))).images[0].pixels
    assert [p.exposure_time for p in pixels.planes] == [7, 30] and pixels.significant_bits == 12
    omero = zarr.open_group(str(tmp_path / "img.ome.zarr"), mode="r").attrs["omero"]
    assert omero["channels"][0]["window"]["end"] == 4095


# --------------------------------------------------------------------------- #
# Scans and stitching (2.4)
# --------------------------------------------------------------------------- #
def test_prescan_overview_is_saved_with_the_run(tmp_path):
    job = _job([{"id": "pre", "task": "stage_prescan", "params": {
        "x_min_um": 100, "x_max_um": 600, "y_min_um": 0, "y_max_um": 200, "source": "LED"}}])
    run = _run(job, run_dir=tmp_path)
    result = run.results["pre"]
    assert result["shape"] == [20, 50] and result["origin_um"] == [100, 0]
    group = zarr.open_group(result["path"], mode="r")
    open_ome_zarr(group)
    translation = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][1]
    assert translation["translation"][3:] == [0, 100]


def test_fast_stage_scan_tiles_the_rectangle_and_needs_the_capability():
    hw = SimulatedMicroscope(sources=SOURCES)
    params = {"x_min_um": 0, "x_max_um": 100, "y_min_um": 0, "y_max_um": 50,
              "sources": ["Laser488"], "intensities": [100], "overlap": 0.5}
    run = _run(_job([{"id": "scan", "task": "fast_stage_scan", "params": params}]), hw)
    call = next(c for c in hw.calls if c[0] == "fast_stage_scan")
    assert call[3:7] == (6, 5, 24.0, 16.0) and run.results["scan"].endswith(".ome.zarr")
    report = A.validate(_job([{"id": "scan", "task": "fast_stage_scan", "params": params}]),
                        capabilities={"stage", "camera"})
    assert not report.ok


def _tiles_from(texture, origins_px, size, px=0.5):
    """NGFF tiles cut from a texture; stage origins may be off by a few pixels."""
    tiles = []
    for i, ((ty, tx), (sy, sx)) in enumerate(origins_px):
        spec = ImageSpec(shape=(1, 1, 1, size, size), pixel_size_um=px,
                         origin_um=(sx * px, sy * px, 0.0), name=f"t{i}")
        image = NgffImage(zarr.group(zarr_format=2), spec)
        image.write(texture[ty:ty + size, tx:tx + size], 0, 0, 0)
        tiles.append(image)
    return tiles


def test_stage_stitching_places_tiles_by_their_stage_positions(tmp_path):
    texture = np.random.default_rng(1).integers(0, 4000, (180, 180), dtype=np.uint16)
    true = [(0, 0), (0, 80), (80, 0), (80, 80)]
    out = stitch(_tiles_from(texture, list(zip(true, true)), 100), tmp_path / "s.ome.zarr", "s")
    assert out.array.shape == (1, 1, 1, 180, 180)
    assert np.array_equal(np.asarray(out.array[0, 0, 0]), texture)
    open_ome_zarr(zarr.open_group(str(tmp_path / "s.ome.zarr"), mode="r"))


def test_ashlar_stitching_corrects_stage_errors(tmp_path):
    pytest.importorskip("ashlarUC2")
    texture = (np.random.default_rng(2).random((180, 180)) * 4000).astype(np.uint16)
    true = [(0, 0), (0, 80), (80, 0), (80, 80)]
    stage = [(0, 0), (3, 78), (78, 2), (81, 83)]   # what the stage reported
    out = stitch(_tiles_from(texture, list(zip(true, stage)), 100), tmp_path / "a.ome.zarr",
                 "a", method="ashlar", max_shift_um=10)
    stitched = np.asarray(out.array[0, 0, 0])
    assert stitched.shape == (180, 180)
    assert np.array_equal(stitched[10:170, 10:170], texture[10:170, 10:170])


# --------------------------------------------------------------------------- #
# useq equivalence (2.6)
# --------------------------------------------------------------------------- #
EQ_SEQ = useq.MDASequence(
    stage_positions=[useq.Position(x=0, y=0, z=10, name="a"),
                     useq.Position(x=500, y=100, z=12, name="b")],
    grid_plan=useq.GridRowsColumns(rows=1, columns=2, fov_width=48, fov_height=32),
    z_plan=useq.ZRangeAround(range=2, step=1),
    channels=[useq.Channel(config="LED", exposure=5), useq.Channel(config="Laser488", exposure=9)],
    time_plan=useq.TIntervalLoops(interval=0, loops=2))


def _acquired(hw):
    return [(round(p["X"], 6), round(p["Y"], 6), round(p["Z"], 6), exposure, lit)
            for _, _, p, exposure, lit in _snaps(hw)]


@pytest.mark.parametrize("order", ["tpgcz", "tpgzc", "ptgzc", "tpzcg"])
@pytest.mark.parametrize("expand", [True, False])
def test_imported_sequences_acquire_the_useq_events_in_order(order, expand):
    seq = EQ_SEQ.model_copy(update={"axis_order": tuple(order)})
    hw = SimulatedMicroscope(sources=SOURCES)
    job = A.job_from_sequence(seq, expand=expand)
    assert any(n.task == "mda" for n, _ in job.walk()) is not expand
    _run(job, hw)
    expected = [(e.x_pos, e.y_pos, e.z_pos, e.exposure, (e.channel.config,)) for e in seq]
    assert _acquired(hw) == expected


def test_region_example_end_to_end(tmp_path):
    hw = SimulatedMicroscope(sources=SOURCES, frame_shape=(240, 320), surface=tilt)
    job = A.examples()["region_overview_focus_map"]
    run = Run(job, hw, run_dir=tmp_path, sink=OmeStore(tmp_path, formats=job.storage.formats,
                                                         stitch="stage"))
    run.run(confirmed=True)
    assert run.state == "finished", run.message
    assert run.results["overview"]["path"].endswith("overview_overview.ome.zarr")
    for region in ("tissue_a", "tissue_b"):
        group = zarr.open_group(str(tmp_path / f"{region}_img_stitched.ome.zarr"), mode="r")
        open_ome_zarr(group)
        assert group.attrs["imswitch"]["derived"] == "stitched"
        assert group["0"].shape[1] == 2              # brightfield and ring
    ring = [c for c in hw.calls if c[0] == "led_pattern" and c[1] == "ring"]
    assert ring and ring[0][2]["radius"] == 4
