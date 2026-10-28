"""AufgabenController over HTTP on the virtual microscope."""
import json
import time
from pathlib import Path

import pytest

API = "/imswitch/api/AufgabenController"
LED = 20000   # the virtual LED runs 0..32000


def _small_job(name="api check", rounds=3):
    return {"schema": "aufgaben.job/1", "name": name, "variables": {"n": 0},
            "resources": {"bf": {"kind": "capture",
                                 "channels": [{"config": "LED", "intensity": LED, "exposure": 5}]}},
            "tasks": [{"id": "r", "task": "repeat", "params": {"count": rounds}, "body": [
                {"id": "img", "task": "capture", "params": {"channels": "@bf"}},
                {"id": "inc", "task": "set_variable", "params": {"name": "n", "value": "${n + 1}"}}]}]}


def _wait_done(api_server, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = api_server.get(f"{API}/getAufgabenStatus").json()
        if status["state"] not in ("running", "paused"):
            return status
        time.sleep(0.2)
    raise AssertionError(f"job did not finish: {status}")


def test_endpoints_are_exposed(api_server):
    paths = api_server.get("/openapi.json").json()["paths"]
    for name in ("getAufgabenCatalog", "validateAufgabenJob", "startAufgabenJob",
                 "pauseAufgabenJob", "abortAufgabenJob", "getAufgabenStatus"):
        assert any(p.endswith(f"/AufgabenController/{name}") for p in paths), name


def test_catalogue_lists_this_setups_devices(api_server):
    catalogue = api_server.get(f"{API}/getAufgabenCatalog").json()
    ids = {entry["id"] for entry in catalogue}
    assert {"zstack", "capture", "move_to_position", "set_temperature"} <= ids
    light = next(e for e in catalogue if e["id"] == "set_illumination")
    assert "LED" in light["schema"]["properties"]["source"]["enum"]


def test_job_library_round_trip(api_server):
    saved = api_server.post(f"{API}/saveAufgabenJob", json=_small_job("library check")).json()
    names = [j["name"] for j in api_server.get(f"{API}/listAufgabenJobs").json()]
    assert saved["name"] in names and "tile_scan_focus_check" in names
    job = api_server.get(f"{API}/getAufgabenJob", params={"name": saved["name"]}).json()
    assert job["name"] == "library check"
    assert api_server.get(f"{API}/deleteAufgabenJob", params={"name": saved["name"]}).status_code == 200


def test_validation_reports_errors(api_server):
    bad = _small_job()
    bad["tasks"][0]["body"][0]["params"]["channels"] = "@missing"
    report = api_server.post(f"{API}/validateAufgabenJob", json=bad).json()
    assert not report["ok"] and any("missing" in i["message"] for i in report["issues"])


def test_run_writes_valid_ome_zarr(api_server):
    response = api_server.post(f"{API}/startAufgabenJob", json={"job": _small_job()})
    assert response.status_code == 200, response.text
    folder = Path(response.json()["folder"])
    status = _wait_done(api_server)
    assert status["state"] == "finished" and status["frames"] == 3, status
    assert status["variables"]["n"] == 3
    image = folder / "img.ome.zarr"
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(str(image), mode="r")
    assert group["0"].shape[:3] == (3, 1, 1)
    models = pytest.importorskip("ome_zarr_models")
    models.open_ome_zarr(group)
    runs = api_server.get(f"{API}/listAufgabenRuns").json()
    assert runs[0]["state"] == "finished" and runs[0]["job"] == "api check"


def test_ask_first_steps_need_confirmation(api_server):
    job = {"schema": "aufgaben.job/1", "name": "homing",
           "tasks": [{"id": "h", "task": "home_axis", "params": {"axis": "X"}}]}
    response = api_server.post(f"{API}/startAufgabenJob", json={"job": job})
    assert response.status_code == 403
    response = api_server.post(f"{API}/startAufgabenJob", json={"job": job, "confirmed": True})
    assert response.status_code == 200
    assert _wait_done(api_server)["state"] == "finished"


def test_useq_import_and_export(api_server):
    sequence = {"channels": [{"config": "LED", "exposure": 5}],
                "z_plan": {"range": 2, "step": 1}, "axis_order": "tpgzc"}
    job = api_server.post(f"{API}/importAufgabenUseq", json=sequence).json()
    assert job["tasks"][0]["task"] == "zstack"
    back = api_server.post(f"{API}/exportAufgabenUseq", json=job).json()
    assert back["z_plan"]["step"] == 1 and back["channels"][0]["config"] == "LED"


def test_well_plate_plan_from_imswitch_labware(api_server):
    name = "corning_96_wellplate_360ul_flat"
    plan = api_server.get(f"{API}/getAufgabenWellPlate", params={
        "load_name": name, "wells": "A1,B3", "offset_x_um": 1000, "offset_y_um": 500}).json()
    labware = api_server.get("/imswitch/api/ExperimentController/getLabwareDefinition", params={
        "load_name": name, "offset_x_um": 1000, "offset_y_um": 500}).json()
    a1, b3 = labware["wells"]["A1"], labware["wells"]["B3"]
    assert plan["a1_center_xy"] == [a1["x"], a1["y"]] and plan["selected_wells"] == [[0, 1], [0, 2]]
    spacing = plan["plate"]["well_spacing"]
    assert b3["x"] - a1["x"] == pytest.approx(2 * spacing[0] * 1000)
    assert a1["y"] - b3["y"] == pytest.approx(spacing[1] * 1000)       # rows run towards -Y
    missing = api_server.get(f"{API}/getAufgabenWellPlate", params={"load_name": "nope"})
    assert missing.status_code == 404


def test_region_job_with_focus_map_prescan_and_stitching(api_server):
    zarr = pytest.importorskip("zarr")
    models = pytest.importorskip("ome_zarr_models")
    job = {"schema": "aufgaben.job/1", "name": "regions",
           "resources": {
               "regions": {"kind": "regions", "regions": [
                   {"name": "ra", "min_x": 0, "max_x": 600, "min_y": 0, "max_y": 400}]},
               "af": {"kind": "autofocus", "range_um": 40, "step_um": 2, "channel": "LED",
                      "intensity": LED},
               "surface": {"kind": "focus_map", "method": "constant"},
               "bf": {"kind": "capture",
                      "channels": [{"config": "LED", "intensity": LED, "exposure": 5}]}},
           "storage": {"formats": ["ome-zarr"], "stitched_preview": True},
           "tasks": [
               {"id": "overview", "task": "stage_prescan", "params": {
                   "x_min_um": 0, "x_max_um": 600, "y_min_um": 0, "y_max_um": 400,
                   "line_step_um": 200, "speed_um_s": 2000, "timeout_s": 120}},
               {"id": "map", "task": "create_focus_map", "params": {
                   "focus_map": "@surface", "autofocus": "@af", "regions": "@regions",
                   "nx": 2, "ny": 1}},
               {"id": "reg", "task": "loop_regions", "params": {"regions": "@regions"}, "body": [
                   {"id": "tile", "task": "loop_grid", "params": {"overlap": 0.2,
                                                                  "focus_map": "@surface"},
                    "body": [{"id": "img", "task": "capture", "params": {"channels": "@bf"}}]}]}]}
    response = api_server.post(f"{API}/startAufgabenJob", json={"job": job})
    assert response.status_code == 200, response.text
    folder = Path(response.json()["folder"])
    status = _wait_done(api_server, timeout=180)
    assert status["state"] == "finished" and status["frames"] == 4, status
    for name in ("ra_0000_img", "ra_img_stitched", "overview_overview"):
        models.open_ome_zarr(zarr.open_group(str(folder / f"{name}.ome.zarr"), mode="r"))
    stitched = zarr.open_group(str(folder / "ra_img_stitched.ome.zarr"), mode="r")["0"]
    assert stitched.shape[3:] == (540, 720)   # 2 x 2 tiles of 300 x 400 px, 20 % overlap


def test_stop_flow_template_runs_on_the_virtual_fluidics(api_server):
    """FlowStop as a job: heater, valve, pump volume, capture, flush."""
    catalogue = {e["id"]: e for e in api_server.get(f"{API}/getAufgabenCatalog").json()}
    assert catalogue["pump_volume"]["schema"]["properties"]["pump"]["enum"] == ["peristaltic"]
    assert catalogue["set_valve"]["schema"]["properties"]["valve"]["enum"] == ["inlet"]
    job = api_server.get(f"{API}/getAufgabenJob?name=stop_flow_fluidics").json()
    assert api_server.post(f"{API}/validateAufgabenJob", json=job).json()["ok"]
    response = api_server.post(f"{API}/startAufgabenJob", json={
        "job": job, "parameters": {"cycles": 2, "volume_ul": 20, "target_c": 20}})
    assert response.status_code == 200, response.text
    status = _wait_done(api_server)
    assert status["state"] == "finished", status
    assert status["frames"] == 2 and status["variables"]["images"] == 2


def test_smart_microscopy_demo_on_the_virtual_sample(api_server):
    """Count objects in a brightfield frame of the virtual sample, then visit three of them."""
    job = {"schema": "aufgaben.job/1", "name": "find and visit",
           "resources": {"bf": {"kind": "capture", "channels": [
               {"config": "LED", "intensity": 200, "exposure": 5}]}},
           "tasks": [   # a bright part of the virtual sample, where the stage starts
               {"id": "start", "task": "move_to_position",
                "params": {"x_um": 0, "y_um": -1000, "z_um": 0}},
               {"id": "look", "task": "capture", "params": {"channels": "@bf"}},
               {"id": "light", "task": "measure_intensity", "params": {"source": "look"}},
               {"id": "cells", "task": "count_objects",
                "params": {"source": "look", "min_area_um2": 50, "smooth_um": 2}},
               {"id": "find", "task": "points_from_mask",
                "params": {"source": "look", "min_area_um2": 50, "smooth_um": 2, "max_points": 3}},
               {"id": "visit", "task": "loop_positions",
                "params": {"source": "${result.find.positions}"},
                "body": [{"id": "zoom", "task": "capture", "params": {"channels": "@bf"}}]}]}
    response = api_server.post(f"{API}/startAufgabenJob", json={"job": job})
    assert response.status_code == 200, response.text
    status = _wait_done(api_server)
    assert status["state"] == "finished", status
    assert status["frames"] == 4                     # the look and three visits
    folder = Path(response.json()["folder"])
    results = {line["node"]: line["result"] for line in map(json.loads, (folder / "run.jsonl")
               .read_text().splitlines()) if line.get("event") == "node"}
    assert results["light"]["mean"] > 20             # the LED lights the sample
    assert results["cells"]["count"] >= 3 and results["find"]["count"] == 3
