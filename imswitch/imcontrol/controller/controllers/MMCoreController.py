"""Controller for the MMCore (Micro-Manager) frontend widget.

Exposes the full property tree of any MMCoreDetectorManager detector through a
small REST API and supports long-exposure software-triggered snaps that run in
a background thread so the request doesn't block on multi-minute exposures.
"""
from __future__ import annotations

import datetime
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import tifffile

from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import APIExport, dirtools, initLogger
from ..basecontrollers import ImConWidgetController


_MMCORE_MANAGER_NAME = "MMCoreDetectorManager"


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
    def setMMCoreParameter(
        self,
        detectorName: Optional[str] = None,
        name: str = None,
        value: Any = None,
    ) -> Dict[str, Any]:
        """Set a single MMCore parameter and return the updated parameter tree.

        The returned tree contains the device's view of the value, which may
        have been clamped to an allowed range or list entry by MMCore.
        """
        if not name:
            raise ValueError("'name' is required")
        if value is None:
            raise ValueError("'value' is required")

        resolvedName, detector = self._getDetector(detectorName)
        detector.setParameter(name, value)
        return self._serialize_parameters(detector)

    @APIExport(requestType="POST")
    def setMMCoreParameters(
        self,
        detectorName: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Batch-set MMCore parameters. Stops on first failure."""
        if not values:
            raise ValueError("'values' must be a non-empty mapping")
        _, detector = self._getDetector(detectorName)
        for name, value in values.items():
            detector.setParameter(name, value)
        return self._serialize_parameters(detector)

    # ------------------------------------------------------------------
    # Long-exposure snap
    # ------------------------------------------------------------------
    @APIExport(requestType="POST")
    def snapMMCoreToDisk(
        self,
        detectorName: Optional[str] = None,
        exposureMs: Optional[float] = None,
        fileName: Optional[str] = None,
        saveFormat: str = "tiff",
    ) -> Dict[str, Any]:
        """Run a single software-triggered snap and write it to the recordings folder.

        Returns immediately with a job id; the actual MMCore snap runs in a
        background thread. The frontend can poll :py:meth:`getMMCoreSnapStatus`
        for progress and the final file path.

        Args:
            detectorName: MMCore detector name. Defaults to the first one.
            exposureMs: Exposure time in milliseconds. ``None`` keeps the
                current device setting.
            fileName: Optional description suffix for the saved filename.
            saveFormat: ``"tiff"`` (default). Other formats not yet implemented.
        """
        resolvedName, detector = self._getDetector(detectorName)

        if saveFormat and saveFormat.lower() != "tiff":
            raise NotImplementedError(
                f"saveFormat='{saveFormat}' not implemented; only 'tiff' is supported"
            )

        # Resolve the effective exposure (used for the status display and the
        # MMCore TimeOut bump).
        try:
            effectiveExposureMs = (
                float(exposureMs)
                if exposureMs is not None
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
            "error": None,
            "fileName": fileName,
        }
        with self._jobsLock:
            self._jobs[jobId] = job

        thread = threading.Thread(
            target=self._runSnapJob,
            args=(jobId, detector, exposureMs, fileName),
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

            finished = time.time()
            self._updateJob(
                jobId,
                state="done",
                finishedAt=finished,
                elapsedMs=int((finished - started) * 1000),
                filePath=fullPath,
                relativeFilePath=relativePath,
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
