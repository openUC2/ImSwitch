"""HTTP API of the SLM / DMD display (``SLMManager``): static patterns, images,
masks drawn over the camera image, the camera<->display calibration, pattern
sequences and sequence acquisitions synchronised with the camera.

Coordinates: display pixels unless a name says camera; camera pixels are full-
sensor pixels (the live stream may be subsampled; multiply by its factor).
All ``show*`` calls return after the frame is on the display and the settle
time has passed. Nothing here moves a stage.
"""
import base64
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import HTTPException, Response
from pydantic import BaseModel, Field

from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcontrol.model.aufgaben.storage import save_image
from imswitch.imcontrol.model.managers.SLMManager import (
    PATTERNS, calibration_points, decode_image, encode_png, spot_centroid)
from ..basecontrollers import ImConWidgetController
from ..busy import busy_reasons


class SLMImageRequest(BaseModel):
    data_b64: str = Field(description="Image file (PNG/BMP/TIFF/JPEG) as base64")
    fit: str = Field("center", description="'center' keeps 1:1 pixels, 'stretch' resizes")


class SLMCameraMaskRequest(BaseModel):
    mask_b64: str = Field(description="PNG mask drawn over the camera image, nonzero = light on")
    camera_width: int = Field(description="Camera width in sensor pixels the mask spans")
    camera_height: int = Field(description="Camera height in sensor pixels the mask spans")
    value: int = Field(255, ge=0, le=255)
    invert: bool = Field(False, description="Light everywhere except the mask")


class SLMAffineRequest(BaseModel):
    camera_points: List[List[float]]
    display_points: List[List[float]]


class SLMSequenceFrame(BaseModel):
    name: str = ""
    data_b64: str


class SLMSequenceUpload(BaseModel):
    name: str
    frames: List[SLMSequenceFrame]


class SLMAcquisitionRequest(BaseModel):
    sequence: str
    detectorName: str = Field("", description="Empty = current detector")
    settleMs: Optional[float] = Field(None, description="Overrides the configured settle time")
    repeats: int = Field(1, ge=1, le=10000)
    save: bool = True
    fileName: str = Field("", description="Stem under <data>/SLM; empty = timestamp_sequence")


class SLMController(ImConWidgetController):
    """See the module docstring. ``isSLMBusy`` feeds the shared busy check."""

    sigSLMSequenceProgress = Signal(object)  # {"kind": "acquisition"|"play"|"calibration", ...}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)
        self._slm = self._master.slmManager
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._acq: Dict = {"running": False}
        self._cal: Dict = {"running": False}
        self._play: Dict = {"running": False}
        self._acquired: List[np.ndarray] = []

    # ───────────────────────── status / output ─────────────────────────
    @APIExport()
    def getSLMStatus(self) -> dict:
        """Display, affine and run states in one call."""
        return {**self._slm.status(), "patterns": sorted(PATTERNS),
                "acquisition": self._acq, "calibration_run": self._cal, "play": self._play}

    @APIExport()
    def getSLMDisplays(self) -> list:
        """Monitors (window backend: [w, h] per index) or framebuffer devices."""
        try:
            return self._slm.list_displays()
        except Exception as e:
            raise HTTPException(500, str(e))

    @APIExport()
    def openSLMDisplay(self, display: str = "", displayIndex: int = -1) -> dict:
        """(Re)open the output; empty/-1 keep the configured backend/index."""
        self._refuse_if_running()
        self._slm.open(display or None, None if displayIndex < 0 else displayIndex)
        return self._slm.status()

    @APIExport()
    def closeSLMDisplay(self) -> dict:
        """Close the output (window backend: gives the monitor back)."""
        self._refuse_if_running()
        self._slm.close()
        return self._slm.status()

    @APIExport()
    def setSLMSettleTime(self, settleTimeMs: float) -> dict:
        """Wait after a new frame before the camera is triggered (ms)."""
        self._slm.settleTimeS = max(0.0, float(settleTimeMs)) / 1000.0
        return self._slm.status()

    @APIExport()
    def showSLMPattern(self, kind: str = "blank", value: int = 255, period: float = 32,
                       angle: float = 0, duty: float = 0.5, phase: float = 0,
                       sinusoidal: bool = False, cx: float = -1, cy: float = -1,
                       radius: float = 20, nx: int = 3, ny: int = 3) -> dict:
        """Show a built-in pattern: blank(value), checkerboard(period, value),
        grating(period px, angle deg, duty, phase in periods, sinusoidal, value),
        disk(cx, cy, radius; -1 = centre), grid(nx, ny, radius). Display pixels."""
        self._refuse_if_running()
        try:
            self._slm.show_pattern(kind, value=value, period=period, angle=angle, duty=duty,
                                   phase=phase, sinusoidal=sinusoidal, cx=cx, cy=cy,
                                   radius=radius, nx=nx, ny=ny)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return self._slm.status()

    @APIExport(requestType="POST")
    def showSLMImage(self, request: SLMImageRequest) -> dict:
        """Show an uploaded image (converted to 8-bit grey)."""
        self._refuse_if_running()
        try:
            img = decode_image(base64.b64decode(request.data_b64.split(",")[-1]))
        except Exception as e:
            raise HTTPException(400, f"bad image: {e}")
        self._slm.show(img, source="image", fit=request.fit)
        return self._slm.status()

    @APIExport(requestType="POST")
    def showSLMCameraMask(self, request: SLMCameraMaskRequest) -> dict:
        """Light the sample where a mask drawn over the camera image is set: the mask
        is warped into display pixels with the calibrated affine."""
        self._refuse_if_running()
        try:
            mask = decode_image(base64.b64decode(request.mask_b64.split(",")[-1]))
        except Exception as e:
            raise HTTPException(400, f"bad mask: {e}")
        frame = self._slm.camera_mask_to_display(
            mask, (request.camera_width, request.camera_height), request.value, request.invert)
        self._slm.show(frame, source="camera_mask")
        return {**self._slm.status(), "on_pixels": int(np.count_nonzero(frame))}

    @APIExport()
    def getSLMPreview(self, maxSize: int = 512) -> Response:
        """The frame currently on the display as PNG (downscaled to maxSize)."""
        return Response(self._slm.preview_png(maxSize), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    # ───────────────────────── affine calibration ─────────────────────────
    @APIExport()
    def getSLMAffine(self) -> dict:
        """Camera px -> display px: x' = a11 x + a12 y + tx, y' = a21 x + a22 y + ty."""
        return self._slm.affine_dict()

    @APIExport()
    def setSLMAffine(self, a11: Optional[float] = None, a12: Optional[float] = None,
                     tx: Optional[float] = None, a21: Optional[float] = None,
                     a22: Optional[float] = None, ty: Optional[float] = None) -> dict:
        """Set coefficients by hand (unset ones keep their value); saved."""
        return self._slm.set_affine(a11=a11, a12=a12, tx=tx, a21=a21, a22=a22, ty=ty)

    @APIExport()
    def resetSLMAffine(self) -> dict:
        """Back to identity (camera pixel = display pixel)."""
        return self._slm.reset_affine()

    @APIExport(requestType="POST")
    def computeSLMAffine(self, request: SLMAffineRequest) -> dict:
        """Fit the affine from >=3 (camera, display) point pairs, e.g. from the
        manual wizard: showSLMPattern(disk, cx, cy) then click the spot on the camera."""
        try:
            return self._slm.compute_affine(request.camera_points, request.display_points)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @APIExport()
    def getSLMCalibrationPoints(self, nPoints: int = 5) -> list:
        """Display positions [[x, y], ...] the calibration uses (3..9 points)."""
        return [list(p) for p in calibration_points(nPoints, self._slm.size)]

    @APIExport()
    def runSLMAutoCalibration(self, nPoints: int = 5, spotRadius: float = 15,
                              detectorName: str = "") -> dict:
        """Show nPoints spots one after another, find each in the camera image and
        fit the affine. Needs the DMD light path imaged onto the camera; the
        display is blanked afterwards. Follow getSLMCalibrationStatus."""
        self._refuse_if_running()
        det = self._detector(detectorName or self._slm.calibrationDetector or "")
        pts = calibration_points(nPoints, self._slm.size)
        self._cal = {"running": True, "step": 0, "n_points": len(pts), "detector": det.name,
                     "points": [{"display": list(p), "camera": None, "snr": None} for p in pts],
                     "residual_px": None, "error": "", "finished": False}
        self._start(self._calibrate, det, pts, float(spotRadius))
        return self._cal

    @APIExport()
    def getSLMCalibrationStatus(self) -> dict:
        return self._cal

    @APIExport()
    def stopSLMCalibration(self) -> dict:
        self._stop.set()
        return self._cal

    def _calibrate(self, det, pts, radius):
        try:
            with self._camera(det) as grab:
                dark = grab(self._slm.show_pattern("blank", value=0))
                cam, disp = [], []
                for i, (x, y) in enumerate(pts):
                    if self._stop.is_set():
                        raise InterruptedError("stopped")
                    self._cal["step"] = i
                    frame = grab(self._slm.show_pattern("disk", cx=x, cy=y, radius=radius))
                    hit = spot_centroid(frame, dark)
                    entry = self._cal["points"][i]
                    if hit is None:
                        self._logger.warning(f"SLM calibration: no spot found for display {x},{y}")
                    else:
                        entry["camera"], entry["snr"] = [hit[0], hit[1]], hit[2]
                        cam.append(hit[:2])
                        disp.append((x, y))
                    self.sigSLMSequenceProgress.emit({"kind": "calibration", "step": i,
                                                      "total": len(pts)})
                if len(cam) < 3:
                    raise RuntimeError(f"only {len(cam)} of {len(pts)} spots found; check that "
                                       "the display light reaches the camera and the exposure")
                self._cal["residual_px"] = self._slm.compute_affine(cam, disp)["calibration"][
                    "residual_px"]
        except Exception as e:
            self._cal["error"] = str(e)
        finally:
            self._slm.show_pattern("blank", value=0)
            self._cal.update(running=False, finished=True)

    # ───────────────────────── sequences ─────────────────────────
    @APIExport(requestType="POST")
    def uploadSLMSequence(self, request: SLMSequenceUpload) -> dict:
        """Store a named sequence of pattern images (any format; kept as 8-bit PNG);
        an existing sequence of that name is replaced."""
        try:
            frames = [(f.name, decode_image(base64.b64decode(f.data_b64.split(",")[-1])))
                      for f in request.frames]
            return self._slm.save_sequence(request.name, frames)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @APIExport()
    def listSLMSequences(self) -> list:
        return self._slm.list_sequences()

    @APIExport()
    def deleteSLMSequence(self, name: str) -> list:
        self._refuse_if_running()
        self._slm.delete_sequence(name)
        return self._slm.list_sequences()

    def _frames(self, name: str) -> list:
        try:
            frames = self._slm.load_sequence(name)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        if not frames:
            raise HTTPException(404, f"no frames in {name!r}")
        return frames

    @APIExport()
    def getSLMSequenceFrame(self, name: str, index: int = 0, maxSize: int = 256) -> Response:
        """One stored frame as PNG."""
        frames = self._frames(name)
        if not 0 <= index < len(frames):
            raise HTTPException(404, f"frame {index} of {len(frames)}")
        return Response(encode_png(frames[index][1], maxSize), media_type="image/png")

    @APIExport()
    def showSLMSequenceFrame(self, name: str, index: int = 0) -> dict:
        """Show one stored frame."""
        self._refuse_if_running()
        frames = self._frames(name)
        if not 0 <= index < len(frames):
            raise HTTPException(404, f"frame {index} of {len(frames)}")
        self._slm.show(frames[index][1], source=f"sequence:{name}[{index}]")
        return self._slm.status()

    @APIExport()
    def playSLMSequence(self, name: str, intervalMs: float = 500, loop: bool = False) -> dict:
        """Display a sequence frame by frame (no camera); stop with stopSLMSequence."""
        self._refuse_if_running()
        frames = self._frames(name)
        self._play = {"running": True, "sequence": name, "index": 0, "total": len(frames),
                      "loop": loop, "error": ""}
        self._start(self._play_loop, frames, max(0.0, intervalMs) / 1000.0, loop)
        return self._play

    @APIExport()
    def stopSLMSequence(self) -> dict:
        """Stop playback or a sequence acquisition (the current frame stays)."""
        self._stop.set()
        return {"play": self._play, "acquisition": self._acq}

    def _play_loop(self, frames, interval, loop):
        try:
            while not self._stop.is_set():
                for i, (_, img) in enumerate(frames):
                    if self._stop.is_set():
                        return
                    self._play["index"] = i
                    self._slm.show(img, source="play", settle=interval)
                    self.sigSLMSequenceProgress.emit({"kind": "play", "index": i,
                                                      "total": len(frames)})
                if not loop:
                    return
        except Exception as e:
            self._play["error"] = str(e)
        finally:
            self._play["running"] = False

    # ───────────────────────── synchronised acquisition ─────────────────────────
    @APIExport(requestType="POST")
    def startSLMSequenceAcquisition(self, request: SLMAcquisitionRequest) -> dict:
        """For each frame: show it, wait the settle time, take exactly one camera frame
        (software trigger when the camera supports it, otherwise flush + next frame).
        Saved as OME-TIFF under <data>/SLM with one channel per pattern (and T =
        repeats). Refused while anything else drives the microscope."""
        self._refuse_if_running()
        reasons = busy_reasons(self._master, self._logger, ignore=("SLM",))
        if reasons:
            raise HTTPException(409, f"Refused: {'; '.join(reasons)}.")
        frames = self._frames(request.sequence)
        det = self._detector(request.detectorName)
        self._acquired = []
        self._acq = {"running": True, "sequence": request.sequence, "detector": det.name,
                     "index": 0, "repeat": 0, "total": len(frames) * request.repeats,
                     "done": 0, "triggered": None, "path": "", "error": "", "finished": False,
                     "t_start": time.time(), "fps": None}
        self._start(self._acquire, det, frames, request)
        return self._acq

    @APIExport()
    def getSLMAcquisitionStatus(self) -> dict:
        return self._acq

    @APIExport()
    def stopSLMSequenceAcquisition(self) -> dict:
        self._stop.set()
        return self._acq

    @APIExport()
    def getSLMAcquiredFrame(self, index: int = -1, maxSize: int = 512) -> Response:
        """A frame of the last sequence acquisition as PNG (-1 = latest), auto-scaled."""
        if not self._acquired:
            raise HTTPException(404, "nothing acquired yet")
        return Response(encode_png(self._acquired[index], maxSize), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    def _acquire(self, det, frames, req: SLMAcquisitionRequest):
        settle = None if req.settleMs is None else req.settleMs / 1000.0
        try:
            with self._camera(det) as grab:
                self._acq["triggered"] = self._triggered
                for r in range(req.repeats):
                    for i, (_, img) in enumerate(frames):
                        if self._stop.is_set():
                            raise InterruptedError("stopped")
                        self._acq.update(index=i, repeat=r)
                        self._acquired.append(grab(self._slm.show(img, source="acquisition",
                                                                  settle=settle)))
                        self._acq["done"] += 1
                        elapsed = time.time() - self._acq["t_start"]
                        self._acq["fps"] = self._acq["done"] / elapsed if elapsed > 0 else None
                        self.sigSLMSequenceProgress.emit({"kind": "acquisition",
                                                          "done": self._acq["done"],
                                                          "total": self._acq["total"]})
            if req.save and self._acquired:
                self._acq["path"] = self._save(det, frames, req)
        except Exception as e:
            self._acq["error"] = str(e)
            self._logger.error(f"SLM sequence acquisition: {e}")
        finally:
            self._acq.update(running=False, finished=True)

    def _save(self, det, frames, req: SLMAcquisitionRequest) -> str:
        folder = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), "SLM")
        os.makedirs(folder, exist_ok=True)
        stem = req.fileName.strip() or f"{time.strftime('%Y%m%d_%H%M%S')}_{req.sequence}"
        path = os.path.join(folder, f"{stem}.ome.tif")
        n = len(frames)
        reps = len(self._acquired) // n
        stack = np.stack(self._acquired[:reps * n]) if reps else np.stack(self._acquired)
        axes = "CYX"
        if reps > 1:
            stack = stack.reshape(reps, n, *stack.shape[1:])
            axes = "TCYX"
        elif reps == 0:  # stopped inside the first repeat
            n = len(stack)
        if stack.ndim == len(axes) + 1:  # RGB camera
            axes += "S"
        names = [name for name, _ in frames][:n]
        try:
            px = det.pixelSizeUm[-1]
        except Exception:
            px = None
        settle_ms = req.settleMs if req.settleMs is not None else self._slm.settleTimeS * 1000
        save_image(path, stack, axes, pixel_size_um=px, channel_names=names, name=stem,
                   annotations={"slm_sequence": req.sequence,
                                "slm_display": f"{self._slm.size[0]}x{self._slm.size[1]}",
                                "settle_ms": settle_ms,
                                "software_trigger": bool(self._triggered)})
        return path

    # ───────────────────────── camera helpers ─────────────────────────
    def _detector(self, name: str):
        dm = self._master.detectorsManager
        try:
            return dm[name] if name else dm[dm.getCurrentDetectorName()]
        except Exception:
            raise HTTPException(404, f"detector {name!r} not found")

    def _camera(self, det):
        """Context manager: acquisition running, software trigger when supported;
        yields ``grab(seq) -> frame`` returning a frame exposed after the display
        frame ``seq`` was shown (the ExperimentController's deterministic grab)."""
        controller = self

        class _Cam:
            def __enter__(self_):
                self_.handle = controller._master.detectorsManager.startAcquisition()
                controller._triggered = controller._begin_triggered(det)
                return controller._grab_factory(det)

            def __exit__(self_, *exc):
                controller._end_triggered(det)
                try:
                    controller._master.detectorsManager.stopAcquisition(self_.handle)
                except Exception:
                    pass
                return False
        return _Cam()

    def _begin_triggered(self, det) -> bool:
        if not (hasattr(det, "setTriggerSource") and hasattr(det, "snapSync")):
            return False
        try:
            ok = bool(det.setTriggerSource("software"))
        except Exception as e:
            self._logger.warning(f"SLM: software trigger unavailable ({e}); free-run grabs")
            ok = False
        return ok

    def _end_triggered(self, det) -> None:
        if getattr(self, "_triggered", False):
            try:
                det.setTriggerSource("continuous")
            except Exception as e:
                self._logger.warning(f"SLM: could not restore continuous mode: {e}")
        self._triggered = False

    def _exposure_s(self, det) -> float:
        try:
            return max(0.001, float(det.getParameter("exposure")) / 1000.0)
        except Exception:
            return 0.1

    def _grab_factory(self, det):
        exposure = self._exposure_s(det)

        def grab(_seq=None) -> np.ndarray:
            if self._triggered:
                frame = det.snapSync(timeout=max(2.0, exposure * 4 + 1.0))
                if frame is not None:
                    return np.asarray(frame)
            return self._next_free_run_frame(det, exposure)
        return grab

    def _next_free_run_frame(self, det, exposure: float) -> np.ndarray:
        """Flush, then the second new frame (the first may have started exposing
        before the display changed)."""
        flush = getattr(det, "flushBuffer", None) or getattr(det, "flushBuffers", None)
        if flush is not None:
            try:
                flush()
            except Exception:
                pass
        deadline = time.time() + max(1.0, 4 * exposure + 0.5)
        first = None
        while time.time() < deadline:
            try:
                frame, n = det.getLatestFrame(returnFrameNumber=True)
            except TypeError:
                return np.asarray(det.getLatestFrame())
            if first is None:
                first = n
            elif n >= first + 2 and frame is not None:
                return np.asarray(frame)
            time.sleep(0.005)
        self._logger.warning("SLM grab: no new frame in time, using the latest one")
        return np.asarray(det.getLatestFrame())

    # ───────────────────────── run bookkeeping ─────────────────────────
    def isSLMBusy(self) -> bool:
        return bool(self._acq.get("running") or self._cal.get("running"))

    def _running(self) -> bool:
        return self.isSLMBusy() or bool(self._play.get("running"))

    def _refuse_if_running(self) -> None:
        if self._running():
            raise HTTPException(
                409, "SLM is busy (acquisition, calibration or playback); stop it first")

    def _start(self, target, *args) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=target, args=args, daemon=True,
                                        name="SLMController")
        self._thread.start()

    def __del__(self):
        try:
            self._stop.set()
        except Exception:
            pass


# Copyright (C) 2020-2026 ImSwitch developers
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
