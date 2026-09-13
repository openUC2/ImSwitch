"""WP-13 — multi-site acquisition gate on the virtual microscope.

Every other backend fix in the Frame Scan Stability backlog (WP-05..WP-09) is
about what happens *between* scan areas, and nothing in the suite ran more than
one. This runs four sites in a single acquisition — 4 sites x 3x3 tiles x
2 channels x 3 Z planes, focus map on — and asserts the things that were
silently wrong in the field:

* exactly one experiment folder for one run   (WP-07)
* one OME-Zarr store per site, all four present
* a protocol JSON that parses and names all four areas   (WP-09)
* a stitched OME-TIFF per site that actually opens       (WP-08)
* the whole thing finishes inside a wall-clock ceiling   (WP-08: stop() hung)

The abort variant stops the run mid-flight, which is the field failure mode:
a site ends up with fewer tiles than its grid, and the stitcher used to block
in "Finalize OME writer" forever, so the next site never started.
"""

import json
import os
import re
import time

import pytest
import tifffile

from imswitch.imcommon.model import dirtools

# The app is mounted under /imswitch and its router adds /api (ImSwitchServer.py),
# so every controller endpoint lives below this prefix — same base URL the
# frontend's axios instance uses.
API = "/imswitch/api"

# --- scan shape (WP-13 spec) ------------------------------------------------
SITES = 4
GRID = 3            # 3x3 tiles per site
CHANNELS = ["LED", "LASER"]
Z_PLANES = 3        # zStackMin..zStackMax inclusive, step 2 -> -2, 0, +2
FOV_UM = 100.0
SITE_PITCH_UM = 1000.0

# Ceiling for the whole acquisition. Generous enough for a loaded laptop, tight
# enough that a hung stitcher (the WP-08 bug) fails instead of running forever.
ACQUISITION_CEILING_S = 300.0
POLL_S = 1.0


def _experiment_root() -> str:
    """<data path>/ExperimentController — where every run drops its folder."""
    return os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), "ExperimentController")


# Run folders are named <YYYYmmdd>_<HHMMSS>; the same parent also holds shared
# artefacts such as focus_maps/, which must not be mistaken for a run.
RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def _snapshot_runs() -> set:
    root = _experiment_root()
    if not os.path.isdir(root):
        return set()
    return {name for name in os.listdir(root) if RUN_DIR_RE.match(name)}


def _scan_area(site_index: int) -> dict:
    """One site: a GRID x GRID snake of positions, offset from its neighbours."""
    x0 = site_index * SITE_PITCH_UM
    y0 = site_index * SITE_PITCH_UM
    positions = []
    for iy in range(GRID):
        columns = range(GRID) if iy % 2 == 0 else reversed(range(GRID))
        for ix in columns:
            positions.append({
                "index": len(positions),
                "x": x0 + ix * FOV_UM,
                "y": y0 + iy * FOV_UM,
                "iX": ix,
                "iY": iy,
            })
    span = (GRID - 1) * FOV_UM
    return {
        "areaId": f"area_{site_index}",
        "areaName": f"Site{site_index}",
        "areaType": "free_scan",
        "centerPosition": {"x": x0 + span / 2, "y": y0 + span / 2},
        "bounds": {
            "minX": x0, "maxX": x0 + span,
            "minY": y0, "maxY": y0 + span,
            "width": span, "height": span,
        },
        "scanPattern": "snake",
        "positions": positions,
    }


def _experiment_payload(name: str) -> dict:
    return {
        "name": name,
        "parameterValue": {
            "illumination": CHANNELS,
            "illuIntensities": [100] * len(CHANNELS),
            "exposureTimes": [1.0] * len(CHANNELS),
            "gains": [0.0] * len(CHANNELS),
            "timeLapsePeriod": 0.0,
            "numberOfImages": 1,
            # Autofocus per point is WP-06's problem; the focus map below is
            # what WP-13 needs to exercise.
            "autoFocus": False,
            "autoFocusMin": -10.0,
            "autoFocusMax": 10.0,
            "autoFocusStepSize": 5.0,
            "zStack": True,
            "zStackMin": -2.0,
            "zStackMax": 2.0,
            "zStackStepSize": 2.0,
            "performanceMode": False,
            "ome_write_zarr": True,
            "ome_write_stitched_tiff": True,
            "ome_write_tiff": False,
            "returnToOrigin": False,
        },
        "scanAreas": [_scan_area(i) for i in range(SITES)],
        "focusMap": {
            "enabled": True,
            "rows": 2,
            "cols": 2,
            "fit_by_region": True,
            "method": "constant",
            "apply_during_scan": True,
            # Keep the mapping phase cheap — this test is about multi-site
            # bookkeeping, not about autofocus quality.
            "af_range": 20.0,
            "af_resolution": 10.0,
            "af_settle_time": 0.0,
            "store_debug_artifacts": False,
        },
        "timepoints": 1,
    }


def _wait_until_idle(api_server, ceiling_s: float) -> float:
    """Poll until the workflow leaves running/paused/stopping. Returns seconds."""
    started = time.time()
    while time.time() - started < ceiling_s:
        response = api_server.get(f"{API}/ExperimentController/getExperimentStatus")
        assert response.status_code == 200
        status = (response.json() or {}).get("status")
        if status not in ("running", "paused", "stopping"):
            return time.time() - started
        time.sleep(POLL_S)
    pytest.fail(
        f"acquisition still {status!r} after {ceiling_s:.0f}s — the run never "
        "finalized (a stitcher that blocks in stop() looks exactly like this)"
    )


def _new_run_dir(before: set) -> str:
    """The single experiment folder this run created."""
    created = sorted(_snapshot_runs() - before)
    assert len(created) == 1, (
        f"expected exactly one experiment folder for one run, got {created}"
    )
    return os.path.join(_experiment_root(), created[0])


def _stores(run_dir: str) -> list:
    return sorted(n for n in os.listdir(run_dir) if n.endswith(".ome.zarr"))


def _stitched_tiffs(run_dir: str) -> list:
    return sorted(
        os.path.join(run_dir, entry, "stitched.ome.tif")
        for entry in os.listdir(run_dir)
        if os.path.isfile(os.path.join(run_dir, entry, "stitched.ome.tif"))
    )


def test_focus_map_summary_before_any_run(api_server):
    """The focus-map endpoints must answer on a fresh server.

    getFocusMapSummary read _last_scan_areas, which only existed once a run had
    set it, so the first call after a restart was a 500.
    """
    response = api_server.get(f"{API}/ExperimentController/getFocusMapSummary")
    assert response.status_code == 200, response.text[:300]
    body = response.json()
    assert body["regions"] == []
    assert body["focus_map_active"] is False


@pytest.fixture(scope="module")
def multisite_run(api_server):
    """Run the four-site acquisition once; share the result across assertions."""
    before = _snapshot_runs()

    response = api_server.post(
        f"{API}/ExperimentController/startWellplateExperiment",
        json=_experiment_payload("wp13_multisite"),
        timeout=60,
    )
    assert response.status_code == 200, f"start rejected: {response.text[:500]}"

    elapsed = _wait_until_idle(api_server, ACQUISITION_CEILING_S)
    # Finalization (writer close, stitcher drain) trails the workflow status.
    time.sleep(5)
    return {"dir": _new_run_dir(before), "elapsed": elapsed}


def test_one_experiment_folder(multisite_run):
    """Two folders for one run is WP-07; one run must leave one folder."""
    assert os.path.isdir(multisite_run["dir"])


def test_finishes_within_the_ceiling(multisite_run):
    assert multisite_run["elapsed"] < ACQUISITION_CEILING_S


def test_one_store_per_site(multisite_run):
    stores = _stores(multisite_run["dir"])
    assert len(stores) == SITES, (
        f"expected {SITES} OME-Zarr stores (one per site), got {stores}"
    )


def test_protocol_json_is_written_and_complete(multisite_run):
    """WP-09: a truncated protocol used to be the only sign of a failed run."""
    protocols = [
        os.path.join(multisite_run["dir"], n)
        for n in os.listdir(multisite_run["dir"])
        if n.endswith("_protocol.json")
    ]
    assert protocols, "no protocol JSON written"

    protocol = json.loads(open(protocols[0]).read())  # parses => not truncated
    assert protocol.get("workflow_steps"), "protocol has no workflow steps"
    assert len(protocol["snake_tiles"]) == SITES

    area_ids = {point["region_id"] for tile in protocol["snake_tiles"] for point in tile}
    assert area_ids == {f"area_{i}" for i in range(SITES)}, (
        f"scan areas lost their identity between request and protocol: {area_ids}"
    )


def test_every_stitched_tiff_opens(multisite_run):
    """WP-08: the field symptom was a blank/unopenable stitched image."""
    tiffs = _stitched_tiffs(multisite_run["dir"])
    assert len(tiffs) == SITES, f"expected {SITES} stitched TIFFs, got {tiffs}"

    expected_pages = GRID * GRID * len(CHANNELS) * Z_PLANES
    for path in tiffs:
        assert os.path.getsize(path) > 0, f"{path} is empty"
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == expected_pages, (
                f"{os.path.basename(os.path.dirname(path))}: "
                f"{len(tif.pages)} pages, expected {expected_pages} "
                "(the stitcher used to stop at nx*ny and drop the rest)"
            )
            assert tif.pages[0].asarray().size > 0


def test_aborted_run_still_finalizes(api_server):
    """The dropped-frame / abort path: a site with fewer tiles than its grid.

    stop() blocked forever on exactly this, so the run never released the
    workflow and the next acquisition could not start.
    """
    before = _snapshot_runs()

    response = api_server.post(
        f"{API}/ExperimentController/startWellplateExperiment",
        json=_experiment_payload("wp13_abort"),
        timeout=60,
    )
    assert response.status_code == 200, f"start rejected: {response.text[:500]}"

    # Let the first site produce a few tiles, then pull the rug.
    time.sleep(15)
    stop = api_server.get(f"{API}/ExperimentController/stopExperiment")
    assert stop.status_code == 200

    _wait_until_idle(api_server, 90.0)
    time.sleep(5)

    run_dir = _new_run_dir(before)
    for path in _stitched_tiffs(run_dir):
        with tifffile.TiffFile(path) as tif:
            pass  # opening a partial file is the assertion

    # The workflow must be free again, or the next run is dead on arrival.
    status = api_server.get(f"{API}/ExperimentController/getExperimentStatus").json()
    assert status.get("status") not in ("running", "paused", "stopping")
