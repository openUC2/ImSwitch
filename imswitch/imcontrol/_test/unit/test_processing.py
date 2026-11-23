"""ProcessingController: preview, whole-stack jobs on the Pi and on a worker (uc2-deconvolution
tests the recipe and the worker themselves)."""

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
    DistillRequest,
    ProcessingController,
    PreviewRequest,
    ProcessingJobRequest,
    PSFViewRequest,
    _Source,
)
from imswitch.imcontrol.model.aufgaben import storage
from imswitch.imcontrol.model.aufgaben.storage import save_image
from uc2deconvolution import recipe, worker

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


def test_worker_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("IMSWITCH_PROCESSING_WORKER_URL", "http://processing-worker:8090/")
    assert controller()._worker_url() == "http://processing-worker:8090"
    assert controller("http://lab-pc:8090")._worker_url() == "http://lab-pc:8090"  # setup wins


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


def test_the_worker_beside_imswitch_needs_no_setting(worker_url, monkeypatch):
    from imswitch.imcontrol.controller.controllers import ProcessingController as module
    monkeypatch.setenv(module.WORKER_ENV, worker_url + "/")
    assert controller().getProcessingWorker()["ok"]                 # no worker in the setup
    assert controller("http://lab-pc:8090").getProcessingWorker()["url"] == "http://lab-pc:8090"


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


def test_runs_on_the_pi_need_the_memory(data, monkeypatch):
    from imswitch.imcontrol.controller.controllers import ProcessingController as module
    ctl = controller()
    defaults = ctl.getProcessingDefaults("/run/a.ome.zarr")
    little = types.SimpleNamespace(available=1 << 20)
    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: little)
    for call in (
        lambda: ctl.startProcessingJob(job(ctl, defaults, runner="pi")),
        lambda: ctl.previewProcessing(
            PreviewRequest(path="/run/a.ome.zarr", z=0, y=0, x=0, optics=defaults["optics"][0])
        ),
    ):
        with pytest.raises(HTTPException) as e:
            call()
        assert e.value.status_code == 409 and "memory" in e.value.detail
    assert ctl.getProcessingStatus()["state"] == "idle"


def test_psf_from_beads_and_the_datasets_objective(data, worker_url, monkeypatch):
    root, _ = data
    o = recipe.Optics(pixel_um=0.12, z_step_um=0.5, na=0.75)
    points = np.zeros((16, 128, 128), np.float32)
    points[8, 16::32, 16::32] = 1.0  # 16 beads in focus at plane 8
    kernel = recipe.psf(o, points.shape)
    blurred = np.fft.irfftn(np.fft.rfftn(points) * np.fft.rfftn(kernel), s=points.shape)
    beads = np.random.default_rng(0).poisson(np.maximum(blurred, 0) * 3000 / kernel.max() + 100)
    taken_with = {"name": "20x", "na": 0.75, "magnification": 20, "immersion": "Air"}
    save_image(
        root / "run" / "beads.ome.zarr",
        beads.astype(np.uint16),
        "ZYX",
        pixel_size_um=0.12,
        z_step_um=0.5,
        channel_names=["LED 488"],
        objective=taken_with,
    )
    # today's objective is another one: files of earlier data keep theirs
    monkeypatch.setattr(storage, "current_objective", lambda: {"name": "4x", "na": 0.1})
    ctl = controller(worker_url)
    defaults = ctl.getProcessingDefaults("/run/beads.ome.zarr")
    assert defaults["optics"][0]["na"] == 0.75 and defaults["opticsFrom"]["na"] == "dataset"
    assert "measured" in defaults["psfModels"]

    made = ctl.distillProcessingPSF(DistillRequest(path="/run/beads.ome.zarr", window_um=2.0))
    assert made["path"] == "/run/beads_psf.ome.zarr" and (made["found"], made["used"]) == (16, 16)
    psf_attrs = zarr.open_group(str(root / "run" / "beads_psf.ome.zarr"), mode="r").attrs
    assert storage.objective_of(psf_attrs["acquisition"]) == taken_with
    optics = {**defaults["optics"][0], "psf_model": "measured", "psf_path": made["path"]}
    measured = ctl.previewProcessingPSF(PSFViewRequest(optics=optics, planes=16))
    model = ctl.previewProcessingPSF(PSFViewRequest(optics=defaults["optics"][0], planes=16))
    assert abs(measured["fwhm_xy_um"] / model["fwhm_xy_um"] - 1) < 0.1
    assert len(base64.b64decode(model["xz"])) == model["planes"] * model["width"] * 4

    outputs = []
    for runner in ("worker", "pi"):  # the measured PSF travels to the worker in the spec
        ctl.startProcessingJob(
            ProcessingJobRequest(
                path="/run/beads.ome.zarr", optics=[optics], parameters=FAST, runner=runner
            )
        )
        status = wait(ctl)
        assert status["state"] == "done", status
        outputs.append(zarr.open_group(str(root / status["output"].lstrip("/")), mode="r"))
    np.testing.assert_allclose(outputs[0]["0"][:], outputs[1]["0"][:], rtol=1e-4, atol=1e-6)
    assert storage.objective_of(outputs[0].attrs["acquisition"]) == taken_with
    with pytest.raises(HTTPException) as e:
        ctl.previewProcessingPSF(PSFViewRequest(optics={**optics, "psf_path": ""}))
    assert e.value.status_code == 422
