"""
How ImSwitchServer registers APIExport functions as FastAPI routes.

A synchronous GET export runs on the server's asyncio event loop unless it is
flagged ``runInThreadpool``. While such a handler blocks (e.g. a snap waiting
out a ten-minute exposure) no other request is served. These tests pin down
that a flagged GET runs in the threadpool, so a second request still gets
through, and that unflagged GETs keep running on the loop thread.
"""

import asyncio
import threading
import time

import httpx
from fastapi import APIRouter, FastAPI

from imswitch.imcommon.model import APIExport
from imswitch.imcontrol.controller.server.ImSwitchServer import includeAPIRoute


class _Controller:
    def __init__(self):
        self.release = threading.Event()

    @APIExport(runInThreadpool=True)
    def slowSnap(self, timeout: float = 5.0) -> dict:
        # Stands in for a snap: blocks until released or the "exposure" ends.
        released = self.release.wait(timeout)
        return {"released": released, "thread": threading.get_ident()}

    @APIExport()
    def releaseSnap(self) -> dict:
        self.release.set()
        return {"thread": threading.get_ident()}

    @APIExport()
    def blockingGetOnLoop(self, timeout: float = 5.0) -> dict:
        released = self.release.wait(timeout)
        return {"released": released, "thread": threading.get_ident()}

    @APIExport(requestType="POST")
    def releaseViaPost(self) -> dict:
        self.release.set()
        return {"thread": threading.get_ident()}

    @APIExport(asyncExecution=True)
    async def asyncGet(self) -> dict:
        return {"thread": threading.get_ident()}


def _make_app(controller):
    router = APIRouter(prefix="/api")
    for name in ("slowSnap", "releaseSnap", "blockingGetOnLoop", "releaseViaPost", "asyncGet"):
        includeAPIRoute(router, f"/Test/{name}", getattr(controller, name))
    app = FastAPI()
    app.include_router(router)
    return app


def _run(coro):
    return asyncio.run(coro)


async def _start_then_release(app, blockingPath, releaseMethod, releasePath):
    """Start the blocking request, then send the release request while it runs."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        loopThread = threading.get_ident()
        t0 = time.monotonic()
        blocking = asyncio.create_task(client.get(blockingPath, params={"timeout": 3.0}))
        await asyncio.sleep(0.2)  # let the blocking handler start
        release = await client.request(releaseMethod, releasePath)
        blockingResponse = await blocking
        return loopThread, time.monotonic() - t0, blockingResponse.json(), release.json()


def test_flagged_sync_get_runs_off_the_event_loop():
    controller = _Controller()
    app = _make_app(controller)

    loopThread, elapsed, snap, release = _run(
        _start_then_release(app, "/api/Test/slowSnap", "GET", "/api/Test/releaseSnap"))

    # The release request was served while the snap was still blocking.
    assert snap["released"] is True
    assert elapsed < 2.5
    assert snap["thread"] != loopThread
    # Unflagged sync GETs still run on the loop thread, as before.
    assert release["thread"] == loopThread


def test_unflagged_sync_get_still_runs_on_the_event_loop():
    """Documents the default the flag opts out of: the loop is blocked."""
    controller = _Controller()
    app = _make_app(controller)

    loopThread, elapsed, result, release = _run(
        _start_then_release(app, "/api/Test/blockingGetOnLoop", "POST",
                            "/api/Test/releaseViaPost"))

    # The POST could only run after the blocking GET timed out.
    assert result["released"] is False
    assert result["thread"] == loopThread
    assert release["thread"] != loopThread


def test_async_export_is_awaited_on_the_loop():
    controller = _Controller()
    app = _make_app(controller)

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return threading.get_ident(), (await client.get("/api/Test/asyncGet")).json()

    loopThread, result = _run(call())
    assert result["thread"] == loopThread


def test_query_parameters_still_come_from_the_function_signature():
    controller = _Controller()
    app = _make_app(controller)
    parameters = app.openapi()["paths"]["/api/Test/slowSnap"]["get"]["parameters"]
    assert [p["name"] for p in parameters] == ["timeout"]


def test_recording_snap_endpoints_run_in_the_threadpool():
    from imswitch.imcontrol.controller.controllers.RecordingController import (
        RecordingController,
    )

    for name in ("snapImageToPath", "snapImage", "snapNumpyToFastAPI",
                 "cancelSnap", "stopRecording"):
        func = getattr(RecordingController, name)
        assert func._APIRunInThreadpool is True, name
    # Quick status polls stay on the loop.
    assert RecordingController.getSnapStatus._APIRunInThreadpool is False
