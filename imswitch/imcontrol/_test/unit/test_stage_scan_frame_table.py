"""The hardware-triggered stage scan pairs camera frame-id k with row k of a
metadata table built on the Python side. That table used to be written twice
(fast-stage-scan API x-outer, performance mode y-outer, LED before/after the
lasers) while the firmware walks rows outer, snake on odd rows, Z, then lasers
0..4 and the LED. These tests pin the one shared definition; the firmware side
of the same contract is uc2-ESP/test/native/test_stage_scan_order.cpp.
"""
import types

from imswitch.imcontrol.controller.controllers.experiment_controller.scan_plan import (
    stage_scan_channel_sequence, stage_scan_frame_count, stage_scan_frame_table)
from imswitch.imcontrol.controller.controllers.experiment_controller.experiment_mode_base import (
    ExperimentModeBase)
from imswitch.imcontrol.controller.controllers.experiment_controller.experiment_performance_mode import (
    ExperimentPerformanceMode)


def _positions(rows):
    return [(r["ix"], r["iy"], r["z_index"], r["illuminationChannel"]) for r in rows]


def test_grid_order_is_rows_snake_z_then_channels():
    rows = stage_scan_frame_table(nx=2, ny=2, nz=2, xstart=0, ystart=0, zstart=0,
                                  xstep=10, ystep=20, zstep=5,
                                  illumination=[0, 50, 0, 0, 100], led=0)
    # Same sequence the uc2-ESP native test asserts (keep both in sync).
    assert _positions(rows) == [
        (0, 0, 0, "illumination1"), (0, 0, 0, "illumination4"),
        (0, 0, 1, "illumination1"), (0, 0, 1, "illumination4"),
        (1, 0, 0, "illumination1"), (1, 0, 0, "illumination4"),
        (1, 0, 1, "illumination1"), (1, 0, 1, "illumination4"),
        (1, 1, 0, "illumination1"), (1, 1, 0, "illumination4"),   # odd row: X reversed
        (1, 1, 1, "illumination1"), (1, 1, 1, "illumination4"),
        (0, 1, 0, "illumination1"), (0, 1, 0, "illumination4"),
        (0, 1, 1, "illumination1"), (0, 1, 1, "illumination4"),
    ]
    assert [r["x"] for r in rows[:2]] == [0, 0] and rows[8]["x"] == 10 and rows[8]["y"] == 20
    assert rows[2]["z"] == 5


def test_snake_reverses_odd_rows_only_and_can_be_disabled():
    snake = stage_scan_frame_table(3, 3, 1, 0, 0, 0, 1, 1, 0, snake=True)
    raster = stage_scan_frame_table(3, 3, 1, 0, 0, 0, 1, 1, 0, snake=False)
    assert [r["ix"] for r in snake] == [0, 1, 2, 2, 1, 0, 0, 1, 2]
    assert [r["ix"] for r in raster] == [0, 1, 2] * 3


def test_no_light_still_fires_one_default_frame_per_position():
    rows = stage_scan_frame_table(2, 1, 3, 0, 0, 0, 1, 1, 1, illumination=[0, 0, 0, 0, 0], led=None)
    assert len(rows) == 6 == stage_scan_frame_count(2, 1, 3, [0, 0, 0, 0, 0], None)
    assert {r["illuminationChannel"] for r in rows} == {"default"}
    assert {r["illuminationValue"] for r in rows} == {-1}
    assert {r["channel_index"] for r in rows} == {0}


def test_led_fires_after_the_lasers_and_channel_index_follows():
    seq = stage_scan_channel_sequence([30, 0, 0, 0, 0], 255)
    assert [c["name"] for c in seq] == ["illumination0", "led"]
    rows = stage_scan_frame_table(1, 1, 1, 0, 0, 0, 1, 1, 0, illumination=[30, 0, 0, 0, 0], led=255)
    assert [(r["channel_index"], r["illuminationChannel"], r["illuminationValue"]) for r in rows] == [
        (0, "illumination0", 30), (1, "led", 255)]
    # A sixth laser entry is not a firmware channel and must not appear.
    assert stage_scan_channel_sequence([0, 0, 0, 0, 0, 99], 0) == []


def test_rows_carry_what_the_ome_writer_places_by():
    rows = stage_scan_frame_table(2, 1, 2, 5, 7, 9, 1, 1, 2, illumination=[0, 10, 0, 0, 0], led=0)
    assert [r["runningNumber"] for r in rows] == [1, 2, 3, 4]
    assert [r["z_index"] for r in rows] == [0, 1, 0, 1]
    assert {r["time_index"] for r in rows} == {0}
    assert rows[1]["z"] == 11 and rows[3]["x"] == 6


def test_performance_mode_delegates_to_the_shared_table():
    stub = types.SimpleNamespace(_logger=types.SimpleNamespace(info=lambda *a, **k: None))
    rows = ExperimentPerformanceMode._build_scan_metadata(stub, 2, 2, 1, 0, 0, 0, 1, 1, 0, [0, 5, 0, 0, 0], 0)
    assert rows == stage_scan_frame_table(2, 2, 1, 0, 0, 0, 1, 1, 0, [0, 5, 0, 0, 0], 0)


def _mapper(*managers):
    logs = []
    stub = types.SimpleNamespace(
        controller=types.SimpleNamespace(availableIlluminations=list(managers)),
        _logger=types.SimpleNamespace(warning=logs.append, debug=lambda *a, **k: None),
    )
    return (lambda intensities, sources=None:
            ExperimentModeBase.prepare_illumination_parameters(stub, intensities, sources)), logs


def test_illumination_maps_by_source_name_onto_channel_index():
    prep, logs = _mapper(types.SimpleNamespace(name="Laser488", channel_index=2),
                         types.SimpleNamespace(name="LED", channel_index=0))
    # Frontend order differs from config order: names decide, not positions.
    assert prep([20, 10], ["LED", "Laser488"]) == [20, 0, 10, 0, 0]
    # A subset of the lasers must not land on the wrong channel either.
    assert prep([10], ["Laser488"]) == [0, 0, 10, 0, 0]
    assert logs == []


def test_illumination_falls_back_to_config_order_without_names():
    prep, _ = _mapper(types.SimpleNamespace(name="a", channel_index=3),
                      types.SimpleNamespace(name="b"))            # no channel_index -> its position (1)
    assert prep([1, 2]) == [0, 2, 0, 1, 0]


def test_illumination_skips_synthetic_and_out_of_range_channels_with_a_warning():
    prep, logs = _mapper(types.SimpleNamespace(name="Laser", channel_index=1),
                         types.SimpleNamespace(name="Far", channel_index=7))
    assert prep([5, 9, 3], ["Laser", "LED Matrix Ring", "Far"]) == [0, 5, 0, 0, 0]
    assert len(logs) == 2 and "LED Matrix Ring" in logs[0] and "channel_index 7" in logs[1]
    assert prep([0, 0], ["Laser", "Far"]) == [0, 0, 0, 0, 0]   # zero intensity is not an error
