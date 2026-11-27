"""Flat-field (shading) references of the scan camera.

Taken on blank glass, kept per camera, objective, frame size, binning and
light under ``<config>/flatfield/<key>/``, and applied to the stitched image of
every run that asks for it (``ParameterValue.flatField``). The individual
tiles stay raw; the reference used is copied into the run folder, so the
correction can be redone or undone later.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

from imswitch.imcommon.model import APIExport, dirtools
from imswitch.imcontrol.model import flatfield as ff
from imswitch.imcontrol.model.aufgaben import storage
from imswitch.imcontrol.controller.busy import busy_reasons


class FlatFieldCaptureRequest(BaseModel):
    """Where and with which light a flat-field reference is taken."""
    x: Optional[float] = None          # blank glass, stage µm; None = where the stage is
    y: Optional[float] = None
    frames: int = 9                    # a centred grid of this many fields...
    spacingUm: float = 150.0           # ...this far apart (debris moves, shading stays)
    illumination: str = ""             # light to switch on; "" = leave the light as it is
    intensity: float = -1.0            # -1: the light's current value
    exposure: float = -1.0             # ms; -1: as set
    gain: float = -1.0                 # -1: as set
    darkFrame: bool = True             # also a frame with the light off (needs illumination)


class FlatFieldMixin:
    """Take, keep and apply flat-field references (see module docstring)."""

    # ------------------------------------------------------------------ #
    # Identity of a reference
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flatFieldRoot() -> str:
        return os.path.join(dirtools.UserFileDirs.Root, "flatfield")

    @staticmethod
    def _objectiveLabel() -> str:
        """The current objective as a short name ("4x" when it has no name)."""
        try:
            info = storage.current_objective() if storage.current_objective else None
        except Exception:
            info = None
        info = info or {}
        if info.get("name"):
            return str(info["name"])
        if info.get("magnification"):
            return f"{info['magnification']:g}x"
        return "objective"

    def _detectorBinning(self) -> Any:
        params = getattr(self.mDetector, "parameters", {}) or {}
        binning = params.get("binning")
        return getattr(binning, "value", 1) if binning is not None else 1

    def _flatFieldKey(self, shape, channel: str) -> str:
        return ff.reference_key(getattr(self.mDetector, "name", "camera"), self._objectiveLabel(),
                                shape[:2], self._detectorBinning(), channel)

    def _currentFrameShape(self, timeout: float = 2.0) -> Optional[Tuple[int, ...]]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.mDetector.getLatestFrame()
            if frame is not None and getattr(frame, "size", 0):
                return tuple(frame.shape)
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------ #
    # Taking one
    # ------------------------------------------------------------------ #

    def _grabStill(self) -> Optional[np.ndarray]:
        frame = self.grabCameraFrame(frameSync=2)
        return None if frame is None else np.array(frame, copy=True)

    def _takeFlatField(self, x: Optional[float], y: Optional[float], frames: int,
                       spacing: float, illumination: str, intensity: float,
                       dark: bool) -> Tuple[Optional[ff.FlatField], Dict[str, Any]]:
        """Frames of blank glass on a small grid around (x, y), and the reference from them.

        Moves the stage (at most ``spacing * (sqrt(frames) - 1) / 2`` from the
        spot) and leaves it on the last field; switches ``illumination`` on and
        leaves it on. The caller restores both.
        """
        pos = self.mStage.getPosition()
        x0 = float(pos.get("X", 0.0) if x is None else x)
        y0 = float(pos.get("Y", 0.0) if y is None else y)
        dark_frame = None
        if dark and illumination:
            self._switch_off_all_illumination()
            time.sleep(0.1)
            dark_frame = self._grabStill()
        if illumination:
            if intensity is None or intensity < 0:
                intensity = float(getattr(self._master.lasersManager[illumination], "power",
                                          0) or 0)
            self.set_laser_power(intensity, illumination)
            time.sleep(0.15)
        shots: List[np.ndarray] = []
        for dx, dy in ff.reference_offsets(frames, spacing):
            self.move_stage_xy(x0 + dx, y0 + dy, relative=False)
            time.sleep(0.1)
            img = self._grabStill()
            if img is not None:
                shots.append(img)
        flat, report = ff.estimate(shots, dark=dark_frame)
        report["position"] = {"x": x0, "y": y0}
        report["illumination"] = illumination
        return flat, report

    def _storeFlatField(self, flat: ff.FlatField, channel: str) -> str:
        """Keep a reference for this camera, objective and light (on disk and in memory)."""
        key = self._flatFieldKey(flat.shape, channel)
        flat.meta.update({"key": key, "objective": self._objectiveLabel(), "channel": channel,
                          "detector": getattr(self.mDetector, "name", None),
                          "binning": self._detectorBinning(), "camera": self._cameraSnapshot()})
        flat.save(os.path.join(self._flatFieldRoot(), key), pixel_size_um=self._pixelSizeOrNone())
        self._flatFieldCache()[key] = flat
        return key

    def _pixelSizeOrNone(self) -> Optional[float]:
        try:
            return float(self.detectorPixelSize[-1])
        except Exception:
            return None

    def _flatFieldCache(self) -> Dict[str, ff.FlatField]:
        if not hasattr(self, "_flatFields"):
            self._flatFields = {}
        return self._flatFields

    def _storedFlatField(self, shape, channel: str) -> Optional[ff.FlatField]:
        key = self._flatFieldKey(shape, channel)
        cache = self._flatFieldCache()
        if key not in cache:
            folder = os.path.join(self._flatFieldRoot(), key)
            if not os.path.isfile(os.path.join(folder, ff.META_NAME)):
                return None
            try:
                cache[key] = ff.FlatField.load(folder)
            except Exception as e:
                self._logger.warning(f"Flat-field reference {key} could not be read: {e}")
                return None
        return cache[key]

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #

    @APIExport(requestType="POST")
    def captureFlatField(self, request: FlatFieldCaptureRequest) -> Dict[str, Any]:
        """Take a flat-field reference on blank glass and keep it for this objective and light.

        **Moves the stage** to a small grid around (x, y) (``frames`` fields
        ``spacingUm`` apart) and **switches ``illumination`` on**; afterwards
        the light is off and the stage on the last field. The frames must show
        clean glass: the reference is refused when it is saturated, dark, or
        when structure moves between the frames (tissue, a slide edge).
        Returns ``key``, the quality ``report`` (falloff = how much darker the
        corners are) and a ``thumbnail`` of the shading.
        """
        others = busy_reasons(self._master, self._logger, camera=True)
        if others:
            raise HTTPException(status_code=409, detail=f"Refused: {', '.join(others)}.")
        started = not self.mDetector._running
        if started:
            self.mDetector.startAcquisition()
        try:
            self.set_exposure_time_gain(request.exposure if request.exposure > 0 else None,
                                        request.gain if request.gain >= 0 else None, None, {})
            flat, report = self._takeFlatField(request.x, request.y, request.frames,
                                               request.spacingUm, request.illumination,
                                               request.intensity, request.darkFrame)
        finally:
            if request.illumination:
                self._switch_off_all_illumination()
            if started:
                self.mDetector.stopAcquisition()
        if flat is None:
            self._logger.warning(f"Flat-field reference refused: {report['reason']}")
            return {"success": False, "error": report["reason"], "report": report}
        key = self._storeFlatField(flat, request.illumination)
        self._logger.info(f"Flat-field reference {key}: corners {report['falloff'] * 100:.0f} % "
                          f"darker than the centre, {len(report.get('warnings', []))} warning(s)")
        return {"success": True, "key": key, "report": report, "thumbnail": flat.thumbnail()}

    @APIExport()
    def getFlatFieldStatus(self) -> Dict[str, Any]:
        """The stored flat-field references, and which belong to the current objective."""
        root = self._flatFieldRoot()
        refs = []
        for key in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            meta_path = os.path.join(root, key, ff.META_NAME)
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
            refs.append({"key": key, **{k: meta.get(k) for k in (
                "created", "objective", "channel", "detector", "falloff", "frames", "shape",
                "moved_share", "level")}})
        objective = self._objectiveLabel()
        detector = getattr(self.mDetector, "name", None)
        return {"objective": objective, "detector": detector, "references": refs,
                "current": [r for r in refs
                            if r.get("objective") == objective and r.get("detector") == detector]}

    @APIExport()
    def deleteFlatField(self, key: str) -> Dict[str, Any]:
        """Forget one stored flat-field reference (``key`` from ``getFlatFieldStatus``)."""
        if not key or os.sep in key or "/" in key or key.startswith("."):
            raise HTTPException(status_code=400, detail="Not a flat-field reference key.")
        folder = os.path.join(self._flatFieldRoot(), key)
        if not os.path.isfile(os.path.join(folder, ff.META_NAME)):
            raise HTTPException(status_code=404, detail=f"No flat-field reference {key}.")
        shutil.rmtree(folder)
        self._flatFieldCache().pop(key, None)
        return {"success": True, "key": key}

    # ------------------------------------------------------------------ #
    # Runs
    # ------------------------------------------------------------------ #

    def _prepareRunFlatFields(self, ctx) -> None:
        """Resolve the references of a run into ``ctx.flat_fields`` (by light) and record them.

        With ``flatFieldX/Y`` a fresh reference of the run's first light that
        is on is taken there with the run's own exposure, gain and intensity
        (the stage moves there first); otherwise the stored reference of this
        objective is used. A light without one is scanned uncorrected, and the
        run's protocol says so. In a passthrough run (no light set, the sample
        lit by hand) the reference is taken under that light: no light is
        switched, there is no dark frame.
        """
        p = ctx.experiment.parameterValue
        ctx.flat_fields, ctx.flat_field_info = {}, {"requested": bool(p.flatField), "channels": {}}
        if not p.flatField:
            return
        shape = self._currentFrameShape()
        if shape is None:
            ctx.flat_field_info["error"] = "No camera frame to size the reference by"
            self._logger.warning("Flat-field: no frame from the camera; the scan is not corrected")
            return
        kinds = list(ctx.illumination_kinds or [])
        intensities = list(ctx.illumination_intensities or [])
        fresh = p.flatFieldX is not None and p.flatFieldY is not None
        for index, channel in enumerate(ctx.illumination_sources):
            if index < len(intensities) and not (intensities[index] or 0) > 0:
                continue                      # a light this run does not acquire with
            kind = kinds[index] if index < len(kinds) else "default"
            entry = self._resolveRunFlatField(ctx, p, index, channel, kind, shape, fresh)
            fresh = fresh and kind != "default"     # one fresh reference: the first light
            ctx.flat_field_info["channels"][channel] = entry
            if entry.get("applied"):
                falloff = (entry.get("falloff") or 0) * 100
                self._logger.info(f"Flat-field for {channel}: {entry['source']}, corners "
                                  f"{falloff:.0f} % darker than the centre")
        ctx.flat_field_info["applied"] = bool(ctx.flat_fields)
        if not ctx.flat_fields:
            self._logger.warning("Flat-field requested, but there is no reference for "
                                 f"{self._objectiveLabel()}: the scan is not corrected")

    def _freshRunFlatField(self, ctx, p, index: int, channel: str):
        """Take the run's reference at flatFieldX/Y; returns (flat or None, report)."""
        def at(values, default):
            return values[index] if index < len(values or []) else default
        passthrough = bool(getattr(ctx, "illumination_passthrough", False))
        self.set_exposure_time_gain(at(ctx.exposures, None), at(ctx.gains, None), None, {})
        self._announceDisplay("Flat-field reference")
        flat, report = self._takeFlatField(p.flatFieldX, p.flatFieldY, p.flatFieldFrames,
                                           p.flatFieldSpacingUm, "" if passthrough else channel,
                                           at(ctx.illumination_intensities, -1), True)
        if not passthrough:
            self._switch_off_all_illumination()
        return flat, report

    def _resolveRunFlatField(self, ctx, p, index: int, channel: str, kind: str,
                             shape, fresh: bool = False) -> Dict[str, Any]:
        if kind != "default":
            return {"applied": False, "reason": f"not for {kind} illumination"}
        flat, source = None, ""
        if fresh:
            flat, report = self._freshRunFlatField(ctx, p, index, channel)
            if flat is None:
                self._logger.warning(f"Flat-field reference refused: {report['reason']}")
                stored = self._storedFlatField(shape, channel)
                if stored is None:
                    return {"applied": False, "reason": report["reason"]}
                flat, source = stored, f"stored reference (new one refused: {report['reason']})"
            else:
                self._storeFlatField(flat, channel)
                source = "taken at the start of this run"
        if flat is None:
            flat = self._storedFlatField(shape, channel)
            source = "stored reference"
        if flat is None:
            return {"applied": False, "reason": f"no reference for {self._objectiveLabel()}"}
        channels = shape[2] if len(shape) == 3 else 1
        if flat.shape != tuple(shape[:2]) or flat.channels != channels:
            return {"applied": False, "reason": f"reference is {flat.shape}, frames are {shape}"}
        ctx.flat_fields[channel] = flat
        folder = os.path.join(ctx.dir_path, f"flatfield_{ff.safe_name(channel)}")
        flat.save(folder, pixel_size_um=self._pixelSizeOrNone())
        return {"applied": True, "source": source, "file": os.path.relpath(folder, ctx.dir_path),
                **{k: flat.meta.get(k) for k in ("created", "falloff", "frames", "key")}}
