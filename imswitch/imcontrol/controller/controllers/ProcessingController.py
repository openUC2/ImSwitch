"""Processing of stored datasets: 3D deconvolution, previewed on a crop (docs/DECONVOLUTION.md).

The recipe (model/aufgaben/deconvolution.py) runs here, on the Pi, for previews and small stacks.
Larger stacks go to a processing worker (model/aufgaben/worker.py, docker/processing-worker)
registered by URL; the Pi then only reads the planes and writes the result. The result is an
OME-Zarr next to the source, written through storage.py, with the request in recipe.json inside
it. Nothing computes on the Pi while the microscope is busy (controller/busy.py).
"""

import base64
import json
import shutil
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import numpy as np
import tifffile
import zarr
from fastapi import HTTPException
from pydantic import BaseModel, Field

from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcontrol.model.aufgaben import deconvolution as recipe
from imswitch.imcontrol.model.aufgaben import tiffzarr, worker
from imswitch.imcontrol.model.aufgaben.storage import ImageSpec, NgffImage
from ..basecontrollers import ImConWidgetController
from ..busy import busy_reasons

PI_MAX_VOXELS = 10_000_000  # one volume on the Pi; 32 x 512 x 512 takes about 1.5 min there
PREVIEW_MAX_VOXELS = 64 * 256 * 256
RUNNERS = ("auto", "pi", "worker")
_SUFFIXES = (".ome.zarr", ".zarr", ".ome.tiff", ".ome.tif", ".tiff", ".tif")


class PreviewRequest(BaseModel):
    path: str = Field(..., description="Dataset below the data folder: a .zarr store or a TIFF")
    image: str = Field("", description="Image inside the store, e.g. a plate field A/1/0")
    t: int = 0
    c: int = 0
    z: int = Field(..., description="Plane shown; the crop is centred on it")
    y: int = Field(..., description="Crop centre, full-resolution pixels")
    x: int = Field(..., description="Crop centre, full-resolution pixels")
    size: int = Field(256, description="Crop width and height in pixels")
    planes: int = Field(32, description="Planes in the crop")
    optics: Dict[str, Any] = Field(..., description="recipe Optics: pixel_um, z_step_um, na, ...")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="recipe Parameters")


class ProcessingJobRequest(BaseModel):
    path: str = Field(..., description="Dataset below the data folder: a .zarr store or a TIFF")
    image: str = Field("", description="Image inside the store, e.g. a plate field A/1/0")
    channels: List[int] = Field(default_factory=list, description="Channels to process; empty: all")
    optics: List[Dict[str, Any]] = Field(
        ..., description="One per channel, as getProcessingDefaults"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict, description="recipe Parameters")
    runner: str = Field("auto", description="auto (the worker if one is registered), pi or worker")


class _Source:
    """One image below the data folder: an OME-Zarr image (or a field of a plate) or a TIFF.

    ``shape`` is always (t, c, z, y, x); ``read`` returns (z, y, x) from the full-resolution level.
    """

    def __init__(self, path: Path, image: str = ""):
        self.path, self.image = path, image.strip("/")
        self.tiff = path.suffix.lower() in (".tif", ".tiff")
        self.acquisition = {}  # flat key/values that storage.py writes; zarr only
        if self.tiff:
            attrs = json.loads(tiffzarr.read(path, ".zattrs"))
            level = json.loads(tiffzarr.read(path, "0/.zarray"))
            shape, self.dtype = level["shape"], np.dtype(level["dtype"])
        else:
            group = zarr.open_group(str(path / self.image), mode="r")
            attrs = group.attrs.asdict()
            self.acquisition = attrs.get("acquisition") or {}
            attrs = attrs.get("ome", attrs)  # NGFF 0.5 nests under "ome"
            self.array = group[attrs["multiscales"][0]["datasets"][0]["path"]]
            shape, self.dtype = self.array.shape, self.array.dtype
        ms = attrs["multiscales"][0]
        self.axes = [
            a if isinstance(a, str) else a["name"] for a in ms.get("axes") or "tczyx"[-len(shape) :]
        ]
        size = dict(zip(self.axes, shape))
        self.shape = tuple(size.get(a, 1) for a in "tczyx")
        transforms = ms["datasets"][0].get("coordinateTransformations") or []
        scale = dict(
            zip(
                self.axes,
                next(
                    (t["scale"] for t in transforms if t["type"] == "scale"), [1.0] * len(self.axes)
                ),
            )
        )
        units = [
            a.get("unit") for a in ms.get("axes") or [] if isinstance(a, dict) and a["name"] == "x"
        ]
        self.pixel_um = scale.get("x") if units and units[0] == "micrometer" else None
        self.z_step_um, self.t_step_s = scale.get("z", 1.0), scale.get("t", 1.0)
        shift = dict(
            zip(
                self.axes,
                next(
                    (t["translation"] for t in transforms if t["type"] == "translation"),
                    [0.0] * len(self.axes),
                ),
            )
        )
        self.origin_um = (shift.get("x", 0.0), shift.get("y", 0.0), shift.get("z", 0.0))
        omero = (attrs.get("omero") or {}).get("channels") or []
        self.channels = [
            (omero[i].get("label") if i < len(omero) else None) or f"C{i}"
            for i in range(self.shape[1])
        ]
        self.colors = [
            (omero[i].get("color") if i < len(omero) else None) or "FFFFFF"
            for i in range(self.shape[1])
        ]

    def read(self, t: int, c: int, zs=slice(None), ys=slice(None), xs=slice(None)) -> np.ndarray:
        if not self.tiff:
            index = {"t": t, "c": c, "z": zs, "y": ys, "x": xs}
            data = np.asarray(self.array[tuple(index.get(a, 0) for a in self.axes)])
            return data if "z" in self.axes else data[None]
        with tifffile.TiffFile(self.path) as tif:  # one TIFF page per plane
            series = tif.series[0]
            lead, rgb = series.axes.split("Y")[0], series.axes.endswith("S")
            counts = series.shape[: len(lead)]
            planes = []
            for z in range(*zs.indices(self.shape[2])):
                at = {"T": t, "C": c, "Z": z}
                page = series.pages[
                    int(np.ravel_multi_index([at.get(a, 0) for a in lead], counts)) if lead else 0
                ]
                plane = page.asarray()
                planes.append(plane[ys, xs, c] if rgb else plane[ys, xs])
            return np.stack(planes)

    def planes(self, t: int, c: int):
        for z in range(self.shape[2]):
            yield self.read(t, c, slice(z, z + 1))[0]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, "<f4").tobytes()).decode()


def _parse(cls, data):
    try:
        return recipe.parse(cls, data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


class ProcessingController(ImConWidgetController):
    """Deconvolution of stored stacks: preview on the Pi, whole stacks on the Pi or a worker."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._lock = threading.Lock()
        self._job: Dict[str, Any] = {"state": "idle"}
        self._stop = threading.Event()
        self._preview = 0  # number of the newest preview; older ones stop
        self._previewing = 0  # previews computing now

    def isProcessingBusy(self) -> bool:
        """A deconvolution computes on the Pi (a preview, or a job without a worker)."""
        return self._previewing > 0 or (
            self._job.get("state") == "running" and self._job.get("runner") == "pi"
        )

    # -- helpers ---------------------------------------------------------------------------------
    def _worker_url(self) -> str:
        return (getattr(self._setupInfo, "processing", None) or {}).get("workerUrl") or ""

    def _source(self, path: str, image: str = "") -> _Source:
        base = Path(dirtools.UserFileDirs.getValidatedDataPath()).resolve()
        target = (base / path.lstrip("/")).resolve()
        inner = (target / image.strip("/")).resolve()
        if not target.is_relative_to(base) or not inner.is_relative_to(target) or ":" in path:
            raise HTTPException(status_code=400, detail="Invalid path")
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"{path} not found")
        try:
            return _Source(target, image)
        except Exception as e:
            raise HTTPException(status_code=415, detail=f"Cannot read {path} {image}: {e}") from e

    def _optics(self, source: _Source):
        """Optics per channel from the dataset, the current objective, the channel names."""
        na = next(
            (v for k, v in source.acquisition.items() if k.lower().endswith("objective_na")), None
        )
        na_from = "dataset"
        if na is None:
            try:
                na, na_from = (
                    self._master.getController("Objective").getstatus().get("NA"),
                    "current objective",
                )
            except Exception:
                na = None
        try:
            na = float(na)
        except (TypeError, ValueError):
            na, na_from = recipe.Optics.na, "default"
        immersion = 1.0 if na < 0.95 else 1.33 if na <= 1.25 else 1.515
        optics = []
        for name in source.channels:
            try:
                optics.append(
                    recipe.Optics(
                        pixel_um=source.pixel_um or 1.0,
                        z_step_um=source.z_step_um or 1.0,
                        na=na,
                        immersion_n=immersion,
                        emission_um=recipe.emission_um(name),
                    )
                )
            except ValueError:
                optics.append(recipe.Optics(pixel_um=source.pixel_um or 1.0))
        return optics, {
            "na": na_from,
            "pixel_um": "dataset" if source.pixel_um else "unknown",
            "z_step_um": "dataset",
            "emission_um": "channel name",
        }

    def _refuse_on_pi(self) -> None:
        reasons = busy_reasons(self._master, self._logger, ignore=("Processing",))
        if self._job.get("state") == "running" and self._job.get("runner") == "pi":
            reasons.append("a processing job runs on the Pi")
        if reasons:
            raise HTTPException(status_code=409, detail="Not on the Pi while " + ", ".join(reasons))

    def _runner(self, requested: str, voxels: int) -> str:
        if requested not in RUNNERS:
            raise HTTPException(status_code=422, detail=f"runner must be one of {RUNNERS}")
        if requested == "worker" or (requested == "auto" and self._worker_url()):
            if not self._worker_url():
                raise HTTPException(status_code=409, detail="No processing worker is registered")
            return "worker"
        if voxels > PI_MAX_VOXELS:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"One volume has {voxels / 1e6:.0f} M voxels; the Pi takes "
                    f"{PI_MAX_VOXELS / 1e6:.0f} M at most. Register a processing worker, "
                    f"or crop or bin the stack."
                ),
            )
        self._refuse_on_pi()
        return "pi"

    @staticmethod
    def _output(source: _Source) -> Path:
        stem = source.path.name
        stem = next((stem[: -len(s)] for s in _SUFFIXES if stem.lower().endswith(s)), stem)
        if source.image:
            stem += "_" + source.image.replace("/", "-")
        for n in range(1, 1000):
            path = source.path.parent / f"{stem}_deconvolved{'' if n == 1 else f'_{n}'}.ome.zarr"
            if not path.exists():
                return path
        raise HTTPException(
            status_code=409, detail="Too many deconvolved copies next to the source"
        )

    # -- API -----------------------------------------------------------------------------------
    @APIExport(runInThreadpool=True)
    def getProcessingDefaults(self, path: str, image: str = "") -> Dict[str, Any]:
        """What the deconvolution panel needs for one image of a dataset.

        ``path``: dataset below the data folder (a .zarr store or a TIFF); ``image``: an image
        inside the store (a plate field such as A/1/0). Returns the recipe's default parameters
        and limits, optics per channel (µm: pixel size and z step from the dataset, NA from the
        dataset or the current objective, emission wavelength from the channel name) and where
        each optics value came from, the image shape (t, c, z, y, x), the largest volume the Pi
        processes, and the registered worker URL.
        """
        source = self._source(path, image)
        optics, origin = self._optics(source)
        return {
            "recipe": {"name": recipe.NAME, "version": recipe.VERSION},
            "parameters": asdict(recipe.Parameters()),
            "limits": recipe.LIMITS,
            "optics": [asdict(o) for o in optics],
            "opticsFrom": origin,
            "channels": source.channels,
            "shape": dict(zip("tczyx", source.shape)),
            "dtype": str(source.dtype),
            "piMaxVoxels": PI_MAX_VOXELS,
            "previewMaxVoxels": PREVIEW_MAX_VOXELS,
            "workerUrl": self._worker_url(),
        }

    @APIExport(requestType="POST")
    def previewProcessing(self, request: PreviewRequest) -> Dict[str, Any]:
        """Deconvolve a crop on the Pi and return it before and after.

        The crop is ``size`` x ``size`` pixels around (``y``, ``x``) of the full-resolution level
        and ``planes`` planes around ``z``, at most PREVIEW_MAX_VOXELS. Returns the crop's origin,
        the plane ``z`` (xy) and the centre row (xz) before and after as base64 float32
        (little endian), the background used (counts) and the time taken. A newer preview stops
        this one ({"cancelled": true}). 409 while the microscope acquires.
        """
        self._refuse_on_pi()
        source = self._source(request.path, request.image)
        T, C, Z, Y, X = source.shape
        if not (0 <= request.t < T and 0 <= request.c < C and 0 <= request.z < Z):
            raise HTTPException(status_code=422, detail="t, c or z is outside the image")
        nz, sy, sx = (
            max(1, min(request.planes, Z)),
            max(1, min(request.size, Y)),
            max(1, min(request.size, X)),
        )
        if nz * sy * sx > PREVIEW_MAX_VOXELS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The crop has {nz * sy * sx} voxels; a preview takes "
                    f"{PREVIEW_MAX_VOXELS} at most"
                ),
            )
        z0 = min(max(request.z - nz // 2, 0), Z - nz)
        y0, x0 = min(max(request.y - sy // 2, 0), Y - sy), min(max(request.x - sx // 2, 0), X - sx)
        optics = _parse(recipe.Optics, request.optics)
        parameters = _parse(recipe.Parameters, request.parameters)
        crop = source.read(
            request.t, request.c, slice(z0, z0 + nz), slice(y0, y0 + sy), slice(x0, x0 + sx)
        ).astype(np.float32)
        if parameters.background is None:
            parameters.background = recipe.estimate_background(crop)
        with self._lock:
            self._preview += 1
            number = self._preview
            self._previewing += 1
        started = time.monotonic()
        try:
            after = recipe.deconvolve(
                crop, optics, parameters, progress=lambda _: self._preview != number
            )
        except recipe.Cancelled:
            return {"cancelled": True}
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        finally:
            with self._lock:
                self._previewing -= 1
        zi, row = request.z - z0, sy // 2
        return {
            "crop": {"z0": z0, "y0": y0, "x0": x0, "planes": nz, "height": sy, "width": sx},
            "z": zi,
            "row": row,
            "background": parameters.background,
            "seconds": round(time.monotonic() - started, 2),
            "before": {"xy": _b64(crop[zi]), "xz": _b64(crop[:, row])},
            "after": {"xy": _b64(after[zi]), "xz": _b64(after[:, row])},
        }

    @APIExport(requestType="POST")
    def startProcessingJob(self, request: ProcessingJobRequest) -> Dict[str, Any]:
        """Deconvolve every volume (time point x channel) of an image in the background.

        Runs on the registered worker (``runner`` auto or worker) or on the Pi (volumes up to
        PI_MAX_VOXELS, refused while the microscope acquires). The result is
        <source>_deconvolved.ome.zarr next to the source (float32, same pixel size and z step),
        with the request in recipe.json inside it. Poll getProcessingStatus; cancelProcessingJob
        stops it and removes the unfinished result.
        """
        source = self._source(request.path, request.image)
        T, C, Z, Y, X = source.shape
        channels = request.channels or list(range(C))
        if len(set(channels)) != len(channels) or not all(0 <= c < C for c in channels):
            raise HTTPException(status_code=422, detail=f"channels must be distinct, 0 to {C - 1}")
        if len(request.optics) != C:
            raise HTTPException(status_code=422, detail=f"optics needs one entry per channel ({C})")
        optics = [_parse(recipe.Optics, o) for o in request.optics]
        parameters = _parse(recipe.Parameters, request.parameters)
        runner = self._runner(request.runner, Z * Y * X)
        with self._lock:
            if self._job.get("state") == "running":
                raise HTTPException(status_code=409, detail="A processing job is running")
            output = self._output(source)
            base = Path(dirtools.UserFileDirs.getValidatedDataPath()).resolve()
            self._stop.clear()
            self._job = {
                "state": "running",
                "runner": runner,
                "worker": self._worker_url() if runner == "worker" else None,
                "source": request.path,
                "image": request.image,
                "output": "/" + output.relative_to(base).as_posix(),
                "progress": 0.0,
                "volume": 0,
                "volumes": T * len(channels),
                "error": None,
                "started": time.time(),
                "seconds": None,
            }
        threading.Thread(
            target=self._run,
            args=(source, channels, optics, parameters, output, request),
            daemon=True,
            name="processing-job",
        ).start()
        return self.getProcessingStatus()

    def _run(self, source, channels, optics, parameters, output, request):
        T, _, Z, Y, X = source.shape
        job, started = self._job, time.monotonic()
        try:
            image = NgffImage.create(
                output,
                ImageSpec(
                    shape=(T, len(channels), Z, Y, X),
                    dtype="float32",
                    pixel_size_um=source.pixel_um,
                    z_step_um=source.z_step_um,
                    t_step_s=source.t_step_s,
                    origin_um=source.origin_um,
                    channel_names=[source.channels[c] for c in channels],
                    channel_colors=[source.colors[c] for c in channels],
                    name=output.name.split(".")[0],
                    annotations={
                        "processing": {
                            "recipe": recipe.NAME,
                            "version": recipe.VERSION,
                            "source": request.path,
                            "image": request.image,
                        }
                    },
                ),
            )
            volumes = [(t, i, c) for t in range(T) for i, c in enumerate(channels)]
            ends = [1e-6] * len(channels)
            for k, (t, i, c) in enumerate(volumes):

                def progress(fraction, k=k):
                    job.update(progress=round((k + fraction) / len(volumes), 4), volume=k + 1)
                    return self._stop.is_set()

                spec = {"optics": asdict(optics[c]), "parameters": asdict(parameters)}
                if job["runner"] == "worker":
                    planes = worker.run_remote(
                        job["worker"],
                        source.planes(t, c),
                        (Z, Y, X),
                        source.dtype,
                        spec,
                        progress=progress,
                    )
                else:
                    planes = recipe.run(source.read(t, c), spec, progress)
                for z, plane in enumerate(planes):
                    image.write(plane, t, i, z)
                    if z == Z // 2:  # display window: the middle plane's 99.9th percentile
                        ends[i] = max(ends[i], float(np.percentile(plane, 99.9)))
                if self._stop.is_set():
                    raise recipe.Cancelled()
            image.spec.display_end = ends
            image.finalize()
            (output / "recipe.json").write_text(
                json.dumps(
                    {
                        "recipe": recipe.NAME,
                        "version": recipe.VERSION,
                        "request": request.model_dump(),
                        "runner": job["runner"],
                        "worker": job["worker"],
                        "seconds": round(time.monotonic() - started, 1),
                        "created": datetime.now().isoformat(timespec="seconds"),
                    },
                    indent=2,
                )
            )
            job.update(state="done", progress=1.0)
        except Exception as e:
            cancelled = isinstance(e, recipe.Cancelled) or self._stop.is_set()
            if not cancelled:
                self._logger.error(f"Processing job failed: {e}")
            job.update(
                state="cancelled" if cancelled else "failed", error=None if cancelled else str(e)
            )
            shutil.rmtree(output, ignore_errors=True)  # only the store this job created
        job["seconds"] = round(time.monotonic() - started, 1)

    @APIExport()
    def getProcessingStatus(self) -> Dict[str, Any]:
        """The current or last job: state (idle, running, done, failed, cancelled), runner (pi or
        worker), progress 0..1 over all volumes, volume k of volumes, output (path below the data
        folder), error, seconds."""
        job = dict(self._job)
        if job.get("state") == "running":
            job["seconds"] = round(time.time() - job["started"], 1)
        return job

    @APIExport()
    def cancelProcessingJob(self) -> Dict[str, Any]:
        """Stop the running job; its unfinished result is removed."""
        if self._job.get("state") != "running":
            raise HTTPException(status_code=409, detail="No processing job is running")
        self._stop.set()
        return self.getProcessingStatus()

    @APIExport(runInThreadpool=True)
    def getProcessingWorker(self) -> Dict[str, Any]:
        """The registered processing worker and whether it answers (3 s timeout)."""
        url = self._worker_url()
        if not url:
            return {"url": "", "ok": False, "message": "No worker registered"}
        try:
            return {"url": url, "ok": True, **worker.info(url)}
        except Exception as e:
            return {"url": url, "ok": False, "message": str(e)}

    @APIExport(runInThreadpool=True)
    def setProcessingWorker(self, url: str = "") -> Dict[str, Any]:
        """Register the processing worker, e.g. http://lab-pc:8090; an empty url removes it.

        Saved in the setup file (``processing.workerUrl``). Returns getProcessingWorker.
        """
        url = url.strip().rstrip("/")
        parsed = urlparse(url)
        if url and (parsed.scheme not in ("http", "https") or not parsed.netloc):
            raise HTTPException(
                status_code=422, detail="The worker URL must look like http://host:8090"
            )
        self._setupInfo.processing = {
            **(getattr(self._setupInfo, "processing", None) or {}),
            "workerUrl": url,
        }
        try:
            import imswitch.imcontrol.model.configfiletools as configfiletools

            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
        except Exception as e:
            self._logger.warning(f"Worker URL not saved to the setup file: {e}")
        return self.getProcessingWorker()
