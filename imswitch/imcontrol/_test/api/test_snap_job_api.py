"""
API tests for the asynchronous snap job endpoints.

A snap costs at least one full exposure, so the blocking snapImageToPath call
cannot be used for long-exposure work. These tests cover the non-blocking
replacement: startSnap returns immediately with a jobId, getSnapStatus reports
progress, and cancelSnap aborts an exposure that is still running.
"""
import time


def _wait_for_terminal_status(api_server, jobId, timeout=30.0):
    """Poll getSnapStatus until the job leaves pending/running."""
    deadline = time.time() + timeout
    state = None
    while time.time() < deadline:
        response = api_server.get(
            "/imswitch/api/RecordingController/getSnapStatus", params={"jobId": jobId}
        )
        assert response.status_code == 200, response.text
        state = response.json()
        if state["status"] not in ("pending", "running"):
            return state
        time.sleep(0.1)
    raise AssertionError(f"Snap job did not finish within {timeout}s: {state}")


def test_snap_job_endpoints_exposed(api_server):
    """The three job endpoints must be in the OpenAPI spec."""
    # The spec is served un-prefixed even though requests go through
    # /imswitch/api, so match on the route suffix only.
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})

    for name in ("startSnap", "getSnapStatus", "cancelSnap"):
        matches = [p for p in paths if p.endswith(f"/RecordingController/{name}")]
        assert matches, f"{name} endpoint missing from the API"


def test_start_snap_returns_immediately_with_job(api_server):
    """startSnap must hand back a job id without waiting for the exposure."""
    started = time.time()
    response = api_server.get(
        "/imswitch/api/RecordingController/startSnap",
        params={"fileName": "pytest_async", "saveFormat": 1, "returnPreview": True},
    )
    elapsed = time.time() - started

    assert response.status_code == 200, response.text
    job = response.json()
    assert job.get("jobId"), f"No jobId in response: {job}"
    assert job["status"] in ("pending", "running")
    # The whole point: the request returns without integrating a frame.
    assert elapsed < 5.0, f"startSnap blocked for {elapsed:.1f}s"
    # The UI counts down against this estimate.
    assert job["expectedDurationMs"] > 0

    state = _wait_for_terminal_status(api_server, job["jobId"])
    assert state["status"] == "done", state
    assert state["progress"] == 100.0
    assert state["result"], "Finished job carries no result"


def test_snap_status_defaults_to_latest_job(api_server):
    """Omitting jobId polls the most recent job, so simple UIs need no ids."""
    response = api_server.get(
        "/imswitch/api/RecordingController/startSnap", params={"fileName": "pytest_latest"}
    )
    assert response.status_code == 200
    jobId = response.json()["jobId"]

    response = api_server.get("/imswitch/api/RecordingController/getSnapStatus")
    assert response.status_code == 200
    assert response.json()["jobId"] == jobId

    _wait_for_terminal_status(api_server, jobId)


def test_snap_status_unknown_job(api_server):
    """An unknown id reports 'unknown' rather than erroring out."""
    response = api_server.get(
        "/imswitch/api/RecordingController/getSnapStatus", params={"jobId": "does-not-exist"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "unknown"


def test_cancel_snap_is_accepted(api_server):
    """cancelSnap must be callable and always answer with the job state.

    The virtual camera exposes for milliseconds, so the job may well have
    finished before the cancel lands; both outcomes are valid, and what matters
    is that the endpoint reports the state instead of hanging or 500ing.
    """
    response = api_server.get(
        "/imswitch/api/RecordingController/startSnap", params={"fileName": "pytest_cancel"}
    )
    assert response.status_code == 200
    jobId = response.json()["jobId"]

    response = api_server.get(
        "/imswitch/api/RecordingController/cancelSnap", params={"jobId": jobId}
    )
    assert response.status_code == 200, response.text
    cancelState = response.json()
    assert cancelState["jobId"] == jobId
    assert cancelState["status"] in ("cancelling", "done", "cancelled", "error")

    state = _wait_for_terminal_status(api_server, jobId)
    assert state["status"] in ("cancelled", "done", "error"), state


def test_cancel_unknown_job(api_server):
    """Cancelling something that does not exist is reported, not raised."""
    response = api_server.get(
        "/imswitch/api/RecordingController/cancelSnap", params={"jobId": "does-not-exist"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "unknown"


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
