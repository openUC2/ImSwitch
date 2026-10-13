"""AufgabenController over HTTP on the virtual microscope."""
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
