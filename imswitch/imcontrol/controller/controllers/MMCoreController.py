"""Controller for the MMCore (Micro-Manager) frontend widget.

Exposes the full property tree of any MMCoreDetectorManager detector through a
small REST API and supports long-exposure software-triggered snaps that run in
a background thread so the request doesn't block on multi-minute exposures.
"""
import datetime
import io
import json
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

# How many finished jobs to keep in the registry before pruning the oldest.
_JOB_HISTORY = 20

# Fixed overhead added to the exposure for the duration estimate the UI counts
# down against: arming, readout, file write.
_SNAP_OVERHEAD_MS = 2000.0

# Job states that mean "no longer touching the camera".
_TERMINAL_STATES = ("done", "error", "cancelled", "timeout")


class SetParameterRequest(BaseModel):
    detectorName: Optional[str] = None
    name: str
    value: Any


class SetParametersRequest(BaseModel):
    detectorName: Optional[str] = None
    values: Dict[str, Any]


class SaveSettingsRequest(BaseModel):
    detectorName: Optional[str] = None
    values: Optional[Dict[str, Any]] = None


class DetectorNameRequest(BaseModel):
    detectorName: Optional[str] = None


class SnapRequest(BaseModel):
    detectorName: Optional[str] = None
    exposureMs: Optional[float] = None
    fileName: Optional[str] = None
    saveFormat: str = "tiff"


class CancelSnapRequest(BaseModel):
    jobId: Optional[str] = None
    detectorName: Optional[str] = None


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
        # Per-job cancel events so the frontend can abort in-flight snaps.
        self._cancelEvents: Dict[str, threading.Event] = {}
        # Most recently started job, so a single-user UI can poll and cancel
        # without tracking ids (mirrors RecordingController's snap jobs).
        self._lastJobId: Optional[str] = None

        # Re-apply any camera settings persisted from a previous session.
        try:
            self._applySavedSettingsOnStartup()
        except Exception:
            self._logger.error("Failed to apply saved MMCore settings", exc_info=True)

    # ------------------------------------------------------------------
    # Persistence (dedicated `mmcoreSettings` section in the setup JSON)
    # ------------------------------------------------------------------
    def _savedPropertiesFor(self, detectorName: str) -> Dict[str, Any]:
        """Return the persisted {property: value} dict for a detector (or {})."""
        settings = getattr(self._setupInfo, "mmcoreSettings", None)
        if settings is None:
            return {}
        try:
            return dict(settings.savedProperties.get(detectorName, {}) or {})
        except Exception:
            return {}

    def _hasSavedSettings(self, detectorName: str) -> bool:
        return bool(self._savedPropertiesFor(detectorName))

    def _applySavedSettingsOnStartup(self) -> None:
        settings = getattr(self._setupInfo, "mmcoreSettings", None)
        if settings is None:
            return
        for detectorName in self.getMMCoreDetectors():
            saved = self._savedPropertiesFor(detectorName)
            if not saved:
                continue
            try:
                _, detector = self._getDetector(detectorName)
                applied = detector.applySavedProperties(saved)
                self._logger.info(
                    f"Applied {len(applied)} saved MMCore setting(s) to '{detectorName}'"
                )
            except Exception:
                self._logger.warning(
                    f"Could not apply saved settings to '{detectorName}'", exc_info=True
                )

    def _persistSetupInfo(self) -> None:
        import imswitch.imcontrol.model.configfiletools as configfiletools
        mOptions, _ = configfiletools.loadOptions()
        configfiletools.saveSetupInfo(mOptions, self._setupInfo)

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
            "currentWidth": status.get("currentWidth"),
            "currentHeight": status.get("currentHeight"),
            "pixelSizeUm": status.get("pixelSizeUm"),
            "binning": status.get("binning"),
            "supportedBinnings": status.get("supportedBinnings"),
            "temperatureC": status.get("temperatureC"),
            "temperatureSetpointC": status.get("temperatureSetpointC"),
            "exposureMs": status.get("exposureMs"),
            "hasSavedSettings": self._hasSavedSettings(detector.name),
            # Each parameter entry now carries advanced/min/max metadata that
            # getCameraStatus() injected, so the frontend can render limits and
            # collapse expert parameters.
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
    # Temperature
    # ------------------------------------------------------------------
    @APIExport()
    def getMMCoreTemperature(self, detectorName: Optional[str] = None) -> Dict[str, Any]:
        """Return the current sensor temperature and setpoint in °C."""
        _, detector = self._getDetector(detectorName)
        return {
            "detectorName": detector.name,
            "temperatureC": detector.getTemperatureC(),
            "temperatureSetpointC": detector.getTemperatureSetpointC(),
        }

    # ------------------------------------------------------------------
    # Persisted settings (save / reset to default)
    # ------------------------------------------------------------------
    @APIExport()
    def getMMCoreSettings(self, detectorName: Optional[str] = None) -> Dict[str, Any]:
        """Return the persisted settings for the detector plus its factory defaults."""
        _, detector = self._getDetector(detectorName)
        return {
            "detectorName": detector.name,
            "savedProperties": self._savedPropertiesFor(detector.name),
            "factoryDefaults": detector.getFactoryDefaults(),
        }

    @APIExport(requestType="POST")
    def saveMMCoreSettings(self, body: SaveSettingsRequest) -> Dict[str, Any]:
        """Persist the camera's current editable parameters to the setup JSON.

        Body schema (JSON): ``{"detectorName": str|null, "values": {name: value}|null}``.
        When ``values`` is omitted, the current device state (all editable
        parameters) is captured. The settings are re-applied automatically on
        the next ImSwitch startup.
        """
        from imswitch.imcontrol.model.SetupInfo import MMCoreSettingsInfo

        _, detector = self._getDetector(body.detectorName)

        if body.values:
            toSave = dict(body.values)
        else:
            # Snapshot editable parameters that differ from the device's own
            # factory defaults — keeps the persisted set minimal and avoids
            # re-applying noise (and failure-prone selectors) on startup.
            status = detector.getCameraStatus()
            defaults = detector.getFactoryDefaults()
            toSave = {}
            for name, info in status.get("parameters", {}).items():
                if not info.get("editable"):
                    continue
                value = info.get("value")
                default = defaults.get(name)
                if default is not None and str(value) == str(default):
                    continue
                toSave[name] = value

        if getattr(self._setupInfo, "mmcoreSettings", None) is None:
            self._setupInfo.mmcoreSettings = MMCoreSettingsInfo()
        self._setupInfo.mmcoreSettings.savedProperties[detector.name] = toSave

        try:
            self._persistSetupInfo()
        except Exception as exc:
            self._logger.error("Failed to persist MMCore settings", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Could not save settings: {exc}"
            ) from exc

        return {
            "detectorName": detector.name,
            "saved": True,
            "savedProperties": toSave,
        }

    @APIExport(requestType="POST")
    def resetMMCoreSettings(self, body: DetectorNameRequest = None) -> Dict[str, Any]:
        """Reset the camera to factory defaults and drop any persisted settings.

        Restores every editable property to the value captured at manager init
        and removes the detector's entry from the persisted ``mmcoreSettings``.
        Accepts an optional ``{"detectorName": str}`` body.
        """
        detectorName = getattr(body, "detectorName", None) if body else None
        _, detector = self._getDetector(detectorName)

        applied = detector.resetToFactoryDefaults()

        settings = getattr(self._setupInfo, "mmcoreSettings", None)
        if settings is not None and detector.name in settings.savedProperties:
            settings.savedProperties.pop(detector.name, None)
            try:
                self._persistSetupInfo()
            except Exception:
                self._logger.error(
                    "Failed to persist cleared MMCore settings", exc_info=True
                )

        return self._serialize_parameters(detector)

    # ------------------------------------------------------------------
    # Long-exposure snap
    # ------------------------------------------------------------------
    def _jobPublicState(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """A JSON-friendly view of a job, with live timing fields.

        ``elapsedMs``/``remainingMs``/``progress`` are recomputed on every read
        so the UI can count down against ``expectedDurationMs`` even though the
        worker thread only writes the job dict at state transitions.
        """
        snapshot = dict(job)
        startedAt = snapshot.get("startedAt")
        finishedAt = snapshot.get("finishedAt")
        expectedMs = float(snapshot.get("expectedDurationMs") or 0.0)

        if startedAt is None:
            elapsedMs = 0.0
        else:
            elapsedMs = ((finishedAt or time.time()) - startedAt) * 1000.0
        snapshot["elapsedMs"] = int(elapsedMs)

        if snapshot["state"] in _TERMINAL_STATES:
            snapshot["progress"] = 100.0
            snapshot["remainingMs"] = 0
        elif expectedMs > 0:
            # Cap at 99 %: only the worker may declare the job finished, so a
            # camera that overruns the estimate shows "99 %", never "done".
            snapshot["progress"] = round(min(99.0, elapsedMs / expectedMs * 100.0), 1)
            snapshot["remainingMs"] = int(max(0.0, expectedMs - elapsedMs))
        else:
            snapshot["progress"] = 0.0
            snapshot["remainingMs"] = 0

        # Alias for parity with RecordingController's job API, whose clients
        # read "status" rather than "state".
        snapshot["status"] = snapshot["state"]
        return snapshot

    def _pruneJobs(self) -> None:
        """Drop the oldest finished jobs so the registry cannot grow forever.

        Caller must hold ``_jobsLock``.
        """
        if len(self._jobs) <= _JOB_HISTORY:
            return
        finished = sorted(
            (job.get("finishedAt") or 0.0, jobId)
            for jobId, job in self._jobs.items()
            if job["state"] in _TERMINAL_STATES
        )
        for _, jobId in finished[: len(self._jobs) - _JOB_HISTORY]:
            self._jobs.pop(jobId, None)
            self._cancelEvents.pop(jobId, None)

    def _runningJobsFor(self, detectorName: Optional[str]) -> List[str]:
        """Ids of pending/running jobs, optionally filtered by detector."""
        with self._jobsLock:
            return [
                jobId for jobId, job in self._jobs.items()
                if job["state"] not in _TERMINAL_STATES
                and (detectorName is None or job["detectorName"] == detectorName)
            ]

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

        # One acquisition at a time per detector. Starting a second one would
        # have both jobs fighting over the same sensor, which is exactly the
        # collision that used to leave a job waiting forever for a frame that
        # the other one had already popped.
        busy = self._runningJobsFor(resolvedName)
        if busy:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Detector '{resolvedName}' is already running snap job "
                    f"{busy[0]}. Cancel it first (cancelMMCoreSnap / "
                    f"stopMMCoreAcquisition)."
                ),
            )

        jobId = uuid.uuid4().hex
        job = {
            "jobId": jobId,
            "detectorName": resolvedName,
            "state": "pending",
            "exposureMs": effectiveExposureMs,
            # What the UI counts down against. Display only — the real state
            # always comes from the job, never from this estimate running out.
            "expectedDurationMs": effectiveExposureMs + _SNAP_OVERHEAD_MS,
            "startedAt": time.time(),
            "finishedAt": None,
            "elapsedMs": 0,
            "filePath": None,
            "relativeFilePath": None,
            "hasPreview": False,
            "error": None,
            "fileName": body.fileName,
        }
        cancelEvent = threading.Event()
        with self._jobsLock:
            self._jobs[jobId] = job
            self._cancelEvents[jobId] = cancelEvent
            self._lastJobId = jobId
            self._pruneJobs()
            state = self._jobPublicState(job)

        thread = threading.Thread(
            target=self._runSnapJob,
            args=(jobId, detector, body.exposureMs, body.fileName, cancelEvent),
            daemon=True,
            name=f"MMCoreSnap-{jobId[:8]}",
        )
        thread.start()

        return state

    @APIExport()
    def getMMCoreSnapStatus(self, jobId: Optional[str] = None) -> Dict[str, Any]:
        """Return the current state of a snap job.

        ``jobId`` defaults to the most recently started job, so a single-user
        UI does not have to track ids. Besides the job fields, the reply
        carries ``progress`` (0-100), ``elapsedMs`` and ``remainingMs``, all
        recomputed at read time so a countdown keeps ticking while the exposure
        integrates.
        """
        with self._jobsLock:
            if jobId is None:
                jobId = self._lastJobId
            job = self._jobs.get(jobId) if jobId else None
            if job is None:
                return {"jobId": jobId, "state": "unknown", "status": "unknown"}
            return self._jobPublicState(job)

    @APIExport()
    def listMMCoreSnapJobs(self) -> List[Dict[str, Any]]:
        """Return all snap jobs (running + recent)."""
        with self._jobsLock:
            return [self._jobPublicState(j) for j in self._jobs.values()]

    @APIExport(requestType="POST")
    def cancelMMCoreSnap(self, body: CancelSnapRequest) -> Dict[str, Any]:
        """Cancel a pending or running snap job.

        Sets the job's cancel flag *and* stops the acquisition on the device,
        which ends the integration in the driver rather than waiting out the
        remaining exposure — a 10-minute snap therefore aborts within one poll
        cycle (~50 ms) instead of ten minutes. Jobs that already finished are
        returned unchanged.

        Body schema (JSON): ``{"jobId": str|null, "detectorName": str|null}``.
        ``jobId`` defaults to the most recently started job.
        """
        jobId = body.jobId
        with self._jobsLock:
            if jobId is None:
                jobId = self._lastJobId
            job = self._jobs.get(jobId) if jobId else None
            if job is None:
                raise HTTPException(
                    status_code=404, detail=f"Job '{jobId}' not found"
                )
            if job["state"] in _TERMINAL_STATES:
                return {**self._jobPublicState(job), "cancelled": False,
                        "message": "Job already finished"}
            event = self._cancelEvents.get(jobId)
            detectorName = job["detectorName"]
            job["state"] = "cancelling"

        if event is not None:
            event.set()

        # Stop the sensor now. The worker is parked in its poll loop waiting
        # for a frame; ending the integration makes that wait return
        # immediately with a "cancelled" status.
        self._abortDetector(detectorName)
        self._logger.info(f"MMCore snap job {jobId} cancelled by request")

        return {**self.getMMCoreSnapStatus(jobId), "cancelled": True}

    @APIExport(requestType="POST")
    def stopMMCoreAcquisition(self, body: DetectorNameRequest = None) -> Dict[str, Any]:
        """Stop everything the detector is doing — the "stop acquisition" button.

        Cancels every pending/running snap job on the detector, aborts the
        exposure in the driver, and drops any exclusive claim left behind by a
        worker that died, so the camera is usable again without a restart.

        Body schema (JSON): ``{"detectorName": str|null}``.
        """
        detectorName = body.detectorName if body is not None else None
        resolvedName, detector = self._getDetector(detectorName)

        cancelled = self._runningJobsFor(resolvedName)
        with self._jobsLock:
            for jobId in cancelled:
                event = self._cancelEvents.get(jobId)
                if event is not None:
                    event.set()
                job = self._jobs.get(jobId)
                if job is not None and job["state"] not in _TERMINAL_STATES:
                    job["state"] = "cancelling"

        self._abortDetector(resolvedName)

        # A worker that was killed mid-flight would leave the claim held and
        # every later acquisition would fail with "camera is busy".
        owner = getattr(detector, "exclusiveOwner", None)
        if owner is not None and not cancelled:
            try:
                detector.releaseExclusive(owner)
                self._logger.warning(
                    f"Released stale camera claim by '{owner}' on "
                    f"'{resolvedName}'"
                )
            except Exception:
                self._logger.error("Could not release camera claim", exc_info=True)

        return {
            "detectorName": resolvedName,
            "cancelledJobs": cancelled,
            "stopped": True,
        }

    def _abortDetector(self, detectorName: str) -> None:
        """Stop acquisition on a detector, ending an in-flight exposure."""
        try:
            detector = self._master.detectorsManager[detectorName]
        except Exception:
            return
        try:
            if hasattr(detector, "abortAcquisition"):
                detector.abortAcquisition()
            else:
                detector.stopAcquisition()
        except Exception:
            self._logger.warning(
                f"Could not stop acquisition on '{detectorName}'", exc_info=True
            )

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
            snapshot = self._jobPublicState(job)
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
        cancelEvent: threading.Event,
    ):
        core = detector._core
        label = detector._label
        started = time.time()
        self._updateJob(jobId, state="running", startedAt=started)

        previousTimeout = None
        claimed = False
        try:
            # Claim the sensor for the whole job. While the claim is held the
            # live-view poll replays its cached frame instead of driving
            # MMCore, so nothing else can pop our frame out of the circular
            # buffer or fire a competing snap (the Andor 20072/20073 errors).
            # acquireExclusive also stops a running live sequence and restarts
            # it on release.
            if not detector.acquireExclusive(f"snapJob:{jobId[:8]}"):
                raise RuntimeError(
                    f"Camera is busy — owned by '{detector.exclusiveOwner}'"
                )
            claimed = True

            if cancelEvent.is_set():
                self._finishJob(jobId, started, state="cancelled")
                return

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

            self._updateJob(
                jobId,
                exposureMs=currentExposure,
                expectedDurationMs=currentExposure + _SNAP_OVERHEAD_MS,
            )

            # The manager owns the acquisition protocol: it arms a
            # single-frame sequence and polls, instead of calling the blocking
            # core.snap(). snapImage() holds MMCore's device lock for the whole
            # integration, which stalls every other MMCore call in the process
            # — including the status endpoint this job's own progress display
            # polls, which is why a long exposure looked like a frozen backend
            # with a frozen timer. It also gives us cancellation and a timeout
            # instead of an unbounded wait.
            image, status = detector.acquireSingleFrame(
                cancelEvent=cancelEvent,
                timeout=max(30.0, currentExposure / 1000.0 * 2.0 + 30.0),
            )

            if status == "cancelled" or cancelEvent.is_set():
                self._finishJob(jobId, started, state="cancelled")
                return
            if status == "timeout":
                self._finishJob(
                    jobId, started, state="timeout",
                    error=(
                        f"The camera did not deliver a frame within the "
                        f"timeout for a {currentExposure:.0f} ms exposure."
                    ),
                )
                return
            if status != "done" or image is None:
                self._finishJob(
                    jobId, started, state="error",
                    error="The camera aborted the acquisition without "
                          "delivering a frame.",
                )
                return

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

            # Embed the full MMCore property snapshot as JSON metadata so the
            # capture is fully reproducible from the file alone.
            try:
                metadata = detector.getMetadataSnapshot()
            except Exception:
                metadata = {}
            metadata.update(
                {
                    "exposure_ms": currentExposure,
                    "detector": detector.name,
                    "timestamp": now.isoformat(),
                }
            )
            try:
                description = json.dumps(metadata, default=str)
            except Exception:
                description = (
                    f"MMCore long-exposure snap: exposure_ms={currentExposure:.3f}, "
                    f"detector={detector.name}"
                )

            tifffile.imwrite(
                fullPath,
                image,
                description=description,
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

            self._finishJob(
                jobId, started,
                state="done",
                filePath=fullPath,
                relativeFilePath=relativePath,
                hasPreview=True,
            )

        except Exception as exc:
            self._logger.error("MMCore snap job failed", exc_info=True)
            self._finishJob(
                jobId, started, state="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            # Restore previous MMCore TimeOut value
            if previousTimeout is not None:
                try:
                    core.setProperty(label, "TimeOut", str(previousTimeout))
                except Exception:
                    self._logger.debug("Could not restore MMCore TimeOut", exc_info=True)
            # Hand the camera back (and restart live view if we stopped it).
            # This must happen even on the error paths, or the detector stays
            # claimed forever and every later acquisition reports "busy".
            if claimed:
                try:
                    detector.releaseExclusive(f"snapJob:{jobId[:8]}")
                except Exception:
                    self._logger.error(
                        "Could not release the camera claim", exc_info=True)
            # Drop the cancel event so the dict doesn't grow unbounded.
            with self._jobsLock:
                self._cancelEvents.pop(jobId, None)

    def _finishJob(self, jobId: str, started: float, state: str, **fields):
        """Mark a job as finished in ``state`` and stamp its timing."""
        finished = time.time()
        return self._updateJob(
            jobId,
            state=state,
            finishedAt=finished,
            elapsedMs=int((finished - started) * 1000),
            **fields,
        )


# Copyright (C) 2020-2026 ImSwitch developers
# This file is part of ImSwitch and licensed under GPL-3.0-or-later.
