"""Analysis and script steps: synthetic frames, and closed loops on a simulated sample."""
import sys

import numpy as np
import pytest

from imswitch.imcontrol.model import aufgaben as A
from imswitch.imcontrol.model.aufgaben import analysis
from imswitch.imcontrol.model.aufgaben.engine import Run
from imswitch.imcontrol.model.aufgaben.simulated import SimulatedMicroscope

BF = {"kind": "capture", "channels": [{"config": "LED", "intensity": 100, "exposure": 5}]}


def _job(tasks, **kw):
    return A.Job.model_validate({"name": "t", "resources": {"bf": BF}, **kw, "tasks": tasks})


def _run(job, hw):
    run = Run(job, hw)
    run.run(confirmed=True)
    assert run.state == "finished", run.message
    return run


def _disks(shape, centres, radius, value=1000.0, background=100.0):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.full(shape, background)
    for r, c in centres:
        img[(rows - r) ** 2 + (cols - c) ** 2 <= radius ** 2] += value
    return img


# --------------------------------------------------------------------------- #
# Pure helpers on synthetic frames
# --------------------------------------------------------------------------- #
def test_intensity_counts_saturated_pixels():
    frame = np.full((10, 10), 100, np.uint8)
    frame[:2] = 255
    stats = analysis.intensity_stats(frame)
    assert stats["saturated_fraction"] == pytest.approx(0.2)
    assert stats["max"] == 255 and stats["median"] == 100
    assert analysis.intensity_stats(frame, center_fraction=0.5)["saturated_fraction"] == 0


def test_focus_value_prefers_the_sharp_image():
    from scipy import ndimage as ndi
    sharp = _disks((64, 64), [(20, 20), (40, 44)], 4)
    blurred = ndi.gaussian_filter(sharp, 3)
    for method in ("laplacian", "tenengrad", "normalized_variance"):
        assert analysis.focus_value(sharp, method) > analysis.focus_value(blurred, method)


def test_objects_are_found_sized_and_merged():
    img = _disks((80, 100), [(20, 20), (20, 32), (60, 70)], 4).astype(np.float32)
    threshold, objects = analysis.find_objects(img, smooth_px=0)
    assert len(objects) == 3 and 100 < threshold < 1100
    assert objects[0]["area_px"] == objects[1]["area_px"]
    centroids = sorted(o["centroid"] for o in objects)
    assert centroids[0] == pytest.approx((20, 20)) and centroids[2] == pytest.approx((60, 70))
    _, merged = analysis.find_objects(img, smooth_px=0, grow_px=4)   # 12 px apart -> one
    assert len(merged) == 2
    _, none = analysis.find_objects(img, smooth_px=0, min_area_px=1000)
    assert none == []
    dark = 1200 - img
    assert len(analysis.find_objects(dark, smooth_px=0, dark=True)[1]) == 3
    assert analysis.find_objects(np.zeros((5, 5), np.float32))[1] == []


# --------------------------------------------------------------------------- #
# Steps in a run on a simulated sample
# --------------------------------------------------------------------------- #
SAMPLE = [(100.0, 50.0, 6.0), (130.0, 70.0, 6.0), (90.0, 80.0, 6.0)]


def _scope(**kw):
    return SimulatedMicroscope(frame_shape=(100, 120), pixel_size_um=1.0, sample=SAMPLE, **kw)


def test_objects_are_reported_in_stage_coordinates_and_visited():
    hw = _scope()
    job = _job([
        {"id": "go", "task": "move_to_position", "params": {"x_um": 110, "y_um": 65}},
        {"id": "img", "task": "capture", "params": {"channels": "@bf"}},
        {"id": "cells", "task": "count_objects", "params": {"min_area_um2": 20}},
        {"id": "find", "task": "points_from_mask", "params": {"source": "img", "max_points": 2}},
        {"id": "each", "task": "loop_positions",
         "params": {"source": "${result.find.positions}"},
         "body": [{"id": "here", "task": "capture", "params": {"channels": "@bf"}}]},
    ])
    run = _run(job, hw)
    cells = run.results["cells"]
    assert cells["count"] == 3
    found = sorted((o["x"], o["y"]) for o in cells["objects"])
    for (x, y), (sx, sy, _) in zip(found, sorted(SAMPLE)):
        assert x == pytest.approx(sx, abs=0.6) and y == pytest.approx(sy, abs=0.6)
    assert cells["mean_area_um2"] == pytest.approx(np.pi * 36, rel=0.15)
    assert run.results["find"]["count"] == 2 and run.results["find"]["found"] == 3
    visited = [c[2] for c in hw.calls if c[0] == "snap"][1:]
    assert len(visited) == 2
    for pos, point in zip(visited, run.results["find"]["positions"]):
        assert (pos["X"], pos["Y"]) == pytest.approx((point["x"], point["y"]))


def test_skip_empty_fields_and_stop_after_enough_cells():
    hw = _scope()
    fields = [{"x": 0, "y": 0, "name": "empty"}, {"x": 110, "y": 65, "name": "cells"},
              {"x": 400, "y": 400, "name": "far"}]
    job = _job([
        {"id": "field", "task": "loop_positions", "params": {"source": "@fields"}, "body": [
            {"id": "look", "task": "capture", "params": {"channels": "@bf", "save": False}},
            {"id": "bg", "task": "detect_background", "params": {"source": "look"}},
            {"id": "skip", "task": "condition", "params": {"expr": "result.bg.empty"},
             "body": [{"id": "next", "task": "continue"}]},
            {"id": "count", "task": "count_objects", "params": {"source": "look"}},
            {"id": "add", "task": "set_variable",
             "params": {"name": "cells", "value": "${cells + result.count.count}"}},
            {"id": "enough", "task": "condition", "params": {"expr": "cells >= 3"},
             "body": [{"id": "done", "task": "break", "params": {"loop": "field"}}]}]}],
        variables={"cells": 0},
        resources={"bf": BF, "fields": {"kind": "positions", "positions": fields}})
    run = _run(job, hw)
    assert run.variables["cells"] == 3
    assert run.results["bg"]["empty"] is False
    snaps = [c[2] for c in hw.calls if c[0] == "snap"]
    assert [(p["X"], p["Y"]) for p in snaps] == [(0, 0), (110, 65)]   # stopped before "far"


def test_overview_regions_drive_a_region_loop():
    hw = _scope()
    job = _job([
        {"id": "pre", "task": "stage_prescan",
         "params": {"x_min_um": 0, "x_max_um": 400, "y_min_um": 0, "y_max_um": 300,
                    "line_step_um": 50}},
        {"id": "find", "task": "regions_from_overview",
         "params": {"source": "pre", "min_area_um2": 50, "margin_um": 30, "smooth_um": 0}},
        {"id": "each", "task": "loop_regions", "params": {"regions": "${result.find.regions}"},
         "body": [{"id": "img", "task": "capture", "params": {"channels": "@bf"}}]},
    ])
    run = _run(job, hw)
    regions = run.results["find"]["regions"]
    assert len(regions) == 1                       # the three disks lie within the margin
    r = regions[0]
    assert r["min_x"] <= 84 and r["max_x"] >= 136 and r["min_y"] <= 44 and r["max_y"] >= 86


def test_python_script_and_external_process_need_scripting_and_confirmation():
    job = _job([
        {"id": "go", "task": "move_to_position", "params": {"x_um": 110, "y_um": 65}},
        {"id": "img", "task": "capture", "params": {"channels": "@bf"}},
        {"id": "py", "task": "python_script", "params": {"code": (
            "var['peak'] = float(frame('img').data.max())\n"
            "output = {'cells': len(result)}\n")}},
        {"id": "ext", "task": "external_process",
         "params": {"command": [sys.executable, "-c", "print(6 * 7)"]}},
    ], variables={"peak": 0})
    report = A.validate(job, capabilities=set(A.CAPABILITIES) - {"scripting"})
    assert not report.ok
    assert A.validate(job).ask_first == ["py", "ext"]
    hw = _scope()
    with pytest.raises(PermissionError):
        Run(job, hw).run()
    run = _run(job, hw)
    assert run.variables["peak"] > 1000
    assert run.results["py"]["output"] == {"cells": 2}   # results of go and img
    assert run.results["ext"]["stdout"].strip() == "42" and run.results["ext"]["exit_code"] == 0


def test_a_failing_process_fails_the_step():
    job = _job([{"id": "ext", "task": "external_process",
                 "params": {"command": [sys.executable, "-c", "import sys; sys.exit(3)"]}}])
    run = Run(job, _scope())
    run.run(confirmed=True)
    assert run.state == "failed" and "exited with 3" in run.message


def test_sources_must_name_an_earlier_step():
    job = _job([{"id": "m", "task": "measure_intensity", "params": {"source": "nope"}}])
    assert any("not a step" in i.message for i in A.validate(job).errors)
    job = _job([{"id": "m", "task": "measure_intensity"}])
    run = Run(job, _scope())
    run.run()
    assert run.state == "failed" and "no image yet" in run.message


# --------------------------------------------------------------------------- #
# The smart-microscopy templates on a simulated sample
# --------------------------------------------------------------------------- #
def _template(name, hw, **parameters):
    run = Run(A.examples()[name], hw, parameters=parameters)
    run.run(confirmed=True)
    assert run.state == "finished", run.message
    return run


def test_skip_empty_wells_template():
    # A1 at (0, 0), columns towards +X, rows towards -Y, 9 mm pitch (96-well)
    hw = SimulatedMicroscope(frame_shape=(100, 120), sample=[(9000, 0, 8), (18000, -9000, 8)])
    run = _template("skip_empty_wells", hw)
    assert run.variables["skipped"] == 4
    stacks = [c for c in hw.calls if c[0] == "snap" and c[3] == 100]   # Laser488 exposure
    assert len(stacks) == 2 * 6                                       # two wells x 6 planes
    assert {(round(c[2]["X"]), round(c[2]["Y"])) for c in stacks} == {(9000, 0), (18000, -9000)}


def test_stop_after_n_cells_template():
    # the region grid starts at the top-left tile, centred at (60, 2950)
    cells = [(40 + 10 * i, 2940 + 6 * (i % 2), 3) for i in range(5)]
    hw = SimulatedMicroscope(frame_shape=(100, 120), sample=cells, sources=("Laser405",))
    run = _template("stop_after_n_cells", hw, target_cells=4)
    assert run.variables["cells"] == 5 and run.message == "enough cells imaged"
    assert sum(1 for c in hw.calls if c[0] == "snap") == 1          # stopped after one tile


def test_overview_regions_template():
    hw = SimulatedMicroscope(frame_shape=(100, 120),
                             sample=[(5000, 5000, 100), (12000, 9000, 100)])
    run = _template("overview_regions_zstack", hw)
    regions = run.results["find"]["regions"]
    assert len(regions) == 2
    assert ("objective", 2) in hw.calls
    xy = [(c[2]["X"], c[2]["Y"]) for c in hw.calls if c[0] == "snap"]
    assert xy and all(any(r["min_x"] <= x <= r["max_x"] and r["min_y"] <= y <= r["max_y"]
                          for r in regions) for x, y in xy)
