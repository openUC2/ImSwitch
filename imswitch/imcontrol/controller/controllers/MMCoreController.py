"""Controller for the MMCore (Micro-Manager) frontend widget.

Exposes the full property tree of any MMCoreDetectorManager detector through a
small REST API and supports long-exposure software-triggered snaps that run in
a background thread so the request doesn't block on multi-minute exposures.
"""
import datetime
import io
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import tifffile
from fastapi import HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import APIExport, dirtools, initLogger
from ..basecontrollers import ImConWidgetController


_MMCORE_MANAGER_NAME = "MMCoreDetectorManager"

# Cap preview size so the PNG conversion is cheap even for 4k+ frames.
_PREVIEW_MAX_DIM = 1024


class SetParameterRequest(BaseModel):
    detectorName: Optional[str] = None
    name: str
    value: Any


class SetParametersRequest(BaseModel):
    detectorName: Optional[str] = None
    values: Dict[str, Any]


class SnapRequest(BaseModel):
    detectorName: Optional[str] = None
    exposureMs: Optional[float] = None
    fileName: Optional[str] = None
    saveFormat: str = "tiff"


def _is_mmcore_detector(detector) -> bool:
    return type(detector).__name__ == _MMCORE_MANAGER_NAME


class MMCoreController(ImConWidgetController):
    """Read/write MMCore detector parameters and drive long-exposure snaps."""

    sigSnapJobUpdate = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._jobsLock = threading.Lock()
        # Cache the captured numpy arrays per job for preview rendering.
        # Bounded to the most recent few jobs to avoid leaking large frames.
        self._snapImages: Dict[str, np.ndarray] = {}
        self._snapImageOrder: List[str] = []
        self._maxCachedSnaps = 4

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolveDetectorName(self, detectorName: Optional[str]) -> Optional[str]:
        if detectorName:
            return detectorName
        mmcore = self.getMMCoreDetectors()
        return mmcore[0] if mmcore else None

    def _getDetector(self, detectorName: Optional[str]):
        name = self._resolveDetectorName(detectorName)
        if name is None:
            raise ValueError("No MMCore detector available in this setup")
        try:
            detector = self._master.detectorsManager[name]
        except Exception as exc:
            raise ValueError(f"Detector '{name}' not found") from exc
        if not _is_mmcore_detector(detector):
            raise ValueError(
                f"Detector '{name}' is not an MMCore detector "
                f"(got {type(detector).__name__})"
            )
        return name, detector

    def _serialize_parameters(self, detector) -> Dict[str, Any]:
        status = detector.getCameraStatus()
        # Reshape into a simple ordered list per group so the frontend can
        # render groups without re-sorting.
        params = status.get("parameters", {})
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for name, info in params.items():
            group = info.get("group") or "Other"
            entry = {"name": name, **info}
            groups.setdefault(group, []).append(entry)
        # Stable group order: Acquisition first, MMCore second, others after.
        order = []
        if "Acquisition" in groups:
            order.append("Acquisition")
        if "MMCore" in groups:
            order.append("MMCore")
        for g in groups:
            if g not in order:
                order.append(g)
        return {
            "detectorName": detector.name,
            "model": status.get("model"),
            "sensorWidth": status.get("sensorWidth"),
            "sensorHeight": status.get("sensorHeight"),
            "pixelSizeUm": status.get("pixelSizeUm"),
            "binning": status.get("binning"),
            "supportedBinnings": status.get("supportedBinnings"),
            "groups": [
                {"name": g, "parameters": groups[g]} for g in order
            ],
        }

    # ------------------------------------------------------------------
    # Discovery / parameter read-write
    # ------------------------------------------------------------------
    @APIExport()
    def getMMCoreDetectors(self) -> List[str]:
        """Return the names of all detectors backed by MMCoreDetectorManager."""
        names = []
        try:
            for name in self._master.detectorsManager.getAllDeviceNames():
                try:
                    if _is_mmcore_detector(self._master.detectorsManager[name]):
                        names.append(name)
                except Exception:
                    continue
        except Exception:
            self._logger.error("Failed to enumerate MMCore detectors", exc_info=True)
        return names

    @APIExport()
    def getMMCoreParameters(self, detectorName: Optional[str] = None) -> Dict[str, Any]:
        """Return the full parameter tree of the (named or first) MMCore detector."""
        _, detector = self._getDetector(detectorName)
        return self._serialize_parameters(detector)

    @APIExport(requestType="POST")
    def setMMCoreParameter(self, body: SetParameterRequest) -> Dict[str, Any]:
        """Set a single MMCore parameter and return the updated parameter tree.

        Body schema (JSON): ``{"detectorName": str|null, "name": str, "value": any}``.

        The returned tree contains the device's view of the value, which may
        have been clamped to an allowed range or list entry by MMCore.
        """
        if not body.name:
            raise HTTPException(status_code=400, detail="'name' is required")
        if body.value is None:
            raise HTTPException(status_code=400, detail="'value' is required")

        _, detector = self._getDetector(body.detectorName)
        detector.setParameter(body.name, body.value)
        return self._serialize_parameters(detector)

    @APIExport(requestType="POST")
    def setMMCoreParameters(self, body: SetParametersRequest) -> Dict[str, Any]:
        """Batch-set MMCore parameters. Stops on first failure.

        Body schema (JSON): ``{"detectorName": str|null, "values": {name: value, ...}}``.
        """
        if not body.values:
            raise HTTPException(
                status_code=400, detail="'values' must be a non-empty mapping"
            )
        _, detector = self._getDetector(body.detectorName)
        for name, value in body.values.items():
            detector.setParameter(name, value)
        return self._serialize_parameters(detector)

    # ------------------------------------------------------------------
    # Long-exposure snap
    # ------------------------------------------------------------------
    @APIExport(requestType="POST")
    def snapMMCoreToDisk(self, body: SnapRequest) -> Dict[str, Any]:
        """Run a single software-triggered snap and write it to the recordings folder.

        Body schema (JSON):
            ``{"detectorName": str|null, "exposureMs": float|null,
                "fileName": str|null, "saveFormat": "tiff"}``

        Returns immediately with a job id; the actual MMCore snap runs in a
        background thread. The frontend can poll :py:meth:`getMMCoreSnapStatus`
        for progress and the final file path, and fetch the captured frame
        via :py:meth:`getLastSnapPreview` once the job is done.
        """
        resolvedName, detector = self._getDetector(body.detectorName)

        if body.saveFormat and body.saveFormat.lower() != "tiff":
            raise HTTPException(
                status_code=400,
                detail=f"saveFormat='{body.saveFormat}' not implemented; only 'tiff' is supported",
            )

        # Resolve the effective exposure (used for the status display and the
        # MMCore TimeOut bump).
        try:
            effectiveExposureMs = (
                float(body.exposureMs)
                if body.exposureMs is not None
                else float(detector._core.getExposure())
            )
        except Exception:
            effectiveExposureMs = 0.0

        jobId = uuid.uuid4().hex
        job = {
            "jobId": jobId,
            "detectorName": resolvedName,
            "state": "pending",
            "exposureMs": effectiveExposureMs,
            "startedAt": None,
            "finishedAt": None,
            "elapsedMs": 0,
            "filePath": None,
            "relativeFilePath": None,
            "hasPreview": False,
            "error": None,
            "fileName": body.fileName,
        }
        with self._jobsLock:
            self._jobs[jobId] = job

        thread = threading.Thread(
            target=self._runSnapJob,
            args=(jobId, detector, body.exposureMs, body.fileName),
            daemon=True,
            name=f"MMCoreSnap-{jobId[:8]}",
        )
        thread.start()

        return dict(job)

    @APIExport()
    def getMMCoreSnapStatus(self, jobId: str) -> Dict[str, Any]:
        """Return the current state of a snap job."""
        with self._jobsLock:
            job = self._jobs.get(jobId)
            if job is None:
                return {"jobId": jobId, "state": "unknown"}
            snapshot = dict(job)
        # Live-update elapsed time while still running
        if snapshot["state"] == "running" and snapshot["startedAt"] is not None:
            snapshot["elapsedMs"] = int((time.time() - snapshot["startedAt"]) * 1000)
        return snapshot

    @APIExport()
    def listMMCoreSnapJobs(self) -> List[Dict[str, Any]]:
        """Return all snap jobs (running + recent)."""
        with self._jobsLock:
            return [dict(j) for j in self._jobs.values()]

    @APIExport()
    def getLastSnapPreview(self, jobId: str) -> Response:
        """Return the captured frame for ``jobId`` as a PNG image.

        The PNG is contrast-stretched (1st–99th percentile) and downsampled
        to ``_PREVIEW_MAX_DIM`` on the long edge to stay snappy for big
        sensors. The full-resolution data is on disk at
        ``getMMCoreSnapStatus(jobId)['filePath']``.
        """
        with self._jobsLock:
            image = self._snapImages.get(jobId)
        if image is None:
            raise HTTPException(status_code=404, detail="No preview for this job")

        try:
            img = np.asarray(image)
            if img.ndim != 2:
                img = img.squeeze()
            # Percentile contrast stretch — long exposures often have a long
            # tail of cosmic-ray hot pixels that wipe out a min/max stretch.
            lo, hi = np.percentile(img, (1, 99))
            if hi <= lo:
                hi = lo + 1
            scaled = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
            scaled = scaled.astype(np.uint8)

            # Downsample for the browser
            h, w = scaled.shape[:2]
            longest = max(h, w)
            if longest > _PREVIEW_MAX_DIM:
                stride = int(np.ceil(longest / _PREVIEW_MAX_DIM))
                scaled = scaled[::stride, ::stride]

            buf = io.BytesIO()
            Image.fromarray(scaled, mode="L").save(buf, format="PNG")
            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            self._logger.error("Failed to render snap preview", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Preview render failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------
    def _updateJob(self, jobId: str, **fields):
        with self._jobsLock:
            job = self._jobs.get(jobId)
            if job is None:
                return None
            job.update(fields)
            snapshot = dict(job)
        try:
            self.sigSnapJobUpdate.emit(snapshot)
        except Exception:
            pass
        return snapshot

    def _runSnapJob(
        self,
        jobId: str,
        detector,
        exposureMs: Optional[float],
        fileName: Optional[str],
    ):
        core = detector._core
        label = detector._label
        started = time.time()
        self._updateJob(jobId, state="running", startedAt=started)

        previousTimeout = None
        wasSequenceRunning = False
        try:
            # Halt any continuous acquisition before snapping — the MMCore
            # circular buffer and the live loop will otherwise race with our
            # single-frame call.
            try:
                wasSequenceRunning = bool(core.isSequenceRunning())
                if wasSequenceRunning:
                    core.stopSequenceAcquisition()
            except Exception:
                self._logger.debug("Could not query sequence state", exc_info=True)

            if exposureMs is not None:
                detector.setParameter("Exposure", float(exposureMs))

            try:
                currentExposure = float(core.getExposure())
            except Exception:
                currentExposure = float(exposureMs or 0.0)

            # Make sure MMCore doesn't time out before the exposure finishes.
            # The TimeOut property is in milliseconds; pad generously.
            try:
                previousTimeout = core.getProperty(label, "TimeOut")
            except Exception:
                previousTimeout = None
            try:
                newTimeoutMs = int(max(currentExposure * 2.0 + 30000.0, 30000.0))
                core.setProperty(label, "TimeOut", str(newTimeoutMs))
            except Exception:
                self._logger.debug("Could not raise MMCore TimeOut", exc_info=True)

            self._updateJob(jobId, exposureMs=currentExposure)

            # Blocking snap — this is the long part.
            core.snap()
            image = np.asarray(core.getImage())

            # Save to disk using the same convention as RecordingController.snap
            data_root = dirtools.UserFileDirs.getValidatedDataPath()
            day = datetime.datetime.now().strftime("%Y-%m-%d")
            folder = os.path.join(data_root, "recordings", day)
            os.makedirs(folder, exist_ok=True)

            now = datetime.datetime.now()
            iso = now.strftime("%Y-%m-%dT%H-%M-%S")
            micro = f"{now.microsecond:06d}"
            safeDetector = detector.name.replace(" ", "_").replace("/", "_")
            descPart = ""
            if fileName and fileName.strip():
                desc = fileName.strip().replace(" ", "_").replace("/", "_")
                descPart = f"_{desc}"
            base = f"{iso}-{micro}_{safeDetector}{descPart}"
            fullPath = os.path.join(folder, base + ".tif")

            tifffile.imwrite(
                fullPath,
                image,
                description=(
                    f"MMCore long-exposure snap: exposure_ms={currentExposure:.3f}, "
                    f"detector={detector.name}"
                ),
            )

            try:
                relativePath = "/" + os.path.relpath(fullPath, data_root).replace("\\", "/")
            except Exception:
                relativePath = None

            # Push the frame to the live viewer so the user sees it immediately.
            try:
                detector.sigImageUpdated.emit(image, True, detector.scale)
            except Exception:
                self._logger.debug("Could not emit sigImageUpdated", exc_info=True)

            # Cache the captured frame for the preview endpoint.
            with self._jobsLock:
                self._snapImages[jobId] = image
                self._snapImageOrder.append(jobId)
                while len(self._snapImageOrder) > self._maxCachedSnaps:
                    oldest = self._snapImageOrder.pop(0)
                    self._snapImages.pop(oldest, None)

            finished = time.time()
            self._updateJob(
                jobId,
                state="done",
                finishedAt=finished,
                elapsedMs=int((finished - started) * 1000),
                filePath=fullPath,
                relativeFilePath=relativePath,
                hasPreview=True,
            )

        except Exception as exc:
            self._logger.error("MMCore snap job failed", exc_info=True)
            finished = time.time()
            self._updateJob(
                jobId,
                state="error",
                finishedAt=finished,
                elapsedMs=int((finished - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            # Restore previous MMCore TimeOut value
            if previousTimeout is not None:
                try:
                    core.setProperty(label, "TimeOut", str(previousTimeout))
                except Exception:
                    self._logger.debug("Could not restore MMCore TimeOut", exc_info=True)


# Copyright (C) 2020-2026 ImSwitch developers
# This file is part of ImSwitch and licensed under GPL-3.0-or-later.
