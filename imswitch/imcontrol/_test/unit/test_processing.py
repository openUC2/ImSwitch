"""ProcessingController and the worker: preview, whole-stack jobs on the Pi and on a worker."""

import base64
import json
import threading
import time
import types
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import zarr
from fastapi import HTTPException

from imswitch.imcommon.model import dirtools
from imswitch.imcontrol.controller.controllers.ProcessingController import (
    ProcessingController,
    PreviewRequest,
    ProcessingJobRequest,
    _Source,
)
from imswitch.imcontrol.model.aufgaben import deconvolution as recipe
from imswitch.imcontrol.model.aufgaben import worker
from imswitch.imcontrol.model.aufgaben.storage import save_image

FAST = {"iterations": 4}


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A 2-channel 8 x 96 x 96 stack as OME-Zarr and OME-TIFF in <data>/run."""
    monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath", lambda: str(tmp_path))
    o = recipe.Optics(pixel_um=0.12, z_step_um=0.5)
    stack = np.stack([recipe.synthetic((8, 96, 96), o, seed=s)[1] for s in (1, 2)]).astype(
        np.uint16
    )
    for name in ("a.ome.zarr", "a.ome.tif"):
        save_image(
            tmp_path / "run" / name,
            stack,
            "CZYX",
            pixel_size_um=0.12,
            z_step_um=0.5,
            channel_names=["LED 488", "LED 635"],
        )
    return tmp_path, stack


@pytest.fixture
def worker_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), worker.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def controller(url="", experiment="idle"):
    ctl = ProcessingController.__new__(ProcessingController)
    ctl.__dict__.update(
        _logger=types.SimpleNamespace(error=print, warning=print),
        _lock=threading.Lock(),
        _job={"state": "idle"},
        _stop=threading.Event(),
        _preview=0,
        _previewing=0,
        _setupInfo=types.SimpleNamespace(processing={"workerUrl": url}),
    )
    busy = types.SimpleNamespace(getExperimentStatus=lambda: {"status": experiment})
    ctl._master = types.SimpleNamespace(
        getController=lambda name: busy if name == "Experiment" else None
    )
    return ctl


def wait(ctl, timeout=60):
    end = time.monotonic() + timeout
    while ctl.getProcessingStatus()["state"] == "running" and time.monotonic() < end:
        time.sleep(0.05)
    return ctl.getProcessingStatus()


def job(ctl, defaults, **kw):
    return ProcessingJobRequest(
        path="/run/a.ome.zarr", optics=defaults["optics"], parameters=FAST, **kw
    )


def test_preview_and_a_job_on_the_pi(data):
    root, stack = data
    ctl = controller()
    defaults = ctl.getProcessingDefaults("/run/a.ome.zarr")
    assert defaults["shape"] == {"t": 1, "c": 2, "z": 8, "y": 96, "x": 96}
    assert [o["emission_um"] for o in defaults["optics"]] == [0.52, 0.67]
    assert defaults["optics"][0]["pixel_um"] == 0.12 and defaults["optics"][0]["z_step_um"] == 0.5

    preview = ctl.previewProcessing(
        PreviewRequest(
            path="/run/a.ome.zarr",
            c=1,
            z=3,
            y=90,
            x=10,
            size=64,
            planes=6,
            optics=defaults["optics"][1],
            parameters=FAST,
        )
    )
    crop = preview["crop"]
    assert (crop["z0"], crop["y0"], crop["x0"], crop["planes"]) == (
        0,
        32,
        0,
        6,
    )  # clamped to the image
    before = np.frombuffer(base64.b64decode(preview["before"]["xy"]), "<f4").reshape(64, 64)
    np.testing.assert_array_equal(before, stack[1, 3, 32:96, 0:64])
    assert len(base64.b64decode(preview["after"]["xz"])) == 6 * 64 * 4 and preview["background"] > 0

    ctl.startProcessingJob(job(ctl, defaults, runner="pi"))
    status = wait(ctl)
    assert status["state"] == "done" and status["output"] == "/run/a_deconvolved.ome.zarr"
    out = zarr.open_group(str(root / "run" / "a_deconvolved.ome.zarr"), mode="r")
    assert out["0"].shape == (1, 2, 8, 96, 96) and out["0"].dtype == np.float32
    assert [ch["window"]["end"] > 1 for ch in out.attrs["omero"]["channels"]] == [True, True]
    assert (
        out.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][-1]
        == 0.12
    )
    saved = json.loads((root / "run" / "a_deconvolved.ome.zarr" / "recipe.json").read_text())
    assert saved["request"]["path"] == "/run/a.ome.zarr" and saved["runner"] == "pi"
    expected = recipe.run(stack[0], {"optics": defaults["optics"][0], "parameters": FAST})
    np.testing.assert_allclose(out["0"][0, 0], expected, rtol=1e-5)


def test_a_job_on_a_worker_gives_the_pi_result(data, worker_url):
    root, stack = data
    ctl = controller(worker_url)
    assert ctl.getProcessingWorker()["ok"]
    defaults = ctl.getProcessingDefaults("/run/a.ome.tif")  # a TIFF source, read page by page
    assert _Source(root / "run" / "a.ome.tif").read(0, 1).tolist() == stack[1].tolist()
    ctl.startProcessingJob(
        ProcessingJobRequest(
            path="/run/a.ome.tif", optics=defaults["optics"], parameters=FAST, channels=[1]
        )
    )
    status = wait(ctl)
    assert status["state"] == "done" and status["runner"] == "worker", status
    out = zarr.open_group(str(root / "run" / "a_deconvolved.ome.zarr"), mode="r")["0"]
    expected = recipe.run(stack[1], {"optics": defaults["optics"][1], "parameters": FAST})
    np.testing.assert_allclose(out[0, 0], expected, rtol=1e-5)


def test_worker_reports_failures_and_cancels(worker_url):
    volume = np.ones((4, 32, 32), np.uint16)
    bad = {"optics": {"pixel_um": 0.12, "na": 1.1}}
    with pytest.raises(RuntimeError, match="NA 1.1"):
        list(worker.run_remote(worker_url, volume, volume.shape, volume.dtype, bad, poll_s=0.01))
    slow = {"optics": {"pixel_um": 0.12}, "parameters": {"iterations": 500}}
    big = np.ones((8, 128, 128), np.uint16)  # still running at the first poll, however busy
    with pytest.raises(recipe.Cancelled):
        list(
            worker.run_remote(
                worker_url, big, big.shape, big.dtype, slow, progress=lambda f: True, poll_s=0.01
            )
        )
    with pytest.raises(RuntimeError, match="unknown recipe"):
        list(worker.run_remote(worker_url, volume, volume.shape, volume.dtype, slow, recipe="nope"))
    assert worker.info(worker_url)["jobs"] == 0  # every job above was deleted


def test_refusals_and_cancel(data):
    _, _ = data
    ctl = controller(experiment="running")
    defaults = ctl.getProcessingDefaults("/run/a.ome.zarr")
    for call in (
        lambda: ctl.startProcessingJob(job(ctl, defaults, runner="pi")),
        lambda: ctl.previewProcessing(
            PreviewRequest(path="/run/a.ome.zarr", z=0, y=0, x=0, optics=defaults["optics"][0])
        ),
    ):
        with pytest.raises(HTTPException) as e:
            call()
        assert e.value.status_code == 409 and "experiment" in e.value.detail
    with pytest.raises(HTTPException) as e:
        controller().startProcessingJob(job(ctl, defaults, runner="worker"))  # none registered
    assert e.value.status_code == 409
    with pytest.raises(HTTPException) as e:
        controller().getProcessingDefaults("/../outside")
    assert e.value.status_code in (400, 404)

    ctl = controller()
    ctl.startProcessingJob(
        ProcessingJobRequest(
            path="/run/a.ome.zarr", optics=defaults["optics"], parameters={"iterations": 500}
        )
    )
    ctl.cancelProcessingJob()
    assert wait(ctl)["state"] == "cancelled"
    assert not (data[0] / "run" / "a_deconvolved.ome.zarr").exists()
