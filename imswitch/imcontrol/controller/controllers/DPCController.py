import io
import json
import os
import queue

import numpy as np
import time
import threading
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import cv2
import tifffile
from scipy.ndimage import uniform_filter

from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.framework import Signal
from imswitch.imcontrol.model import configfiletools
from imswitch.imcontrol.model.aufgaben.storage import save_image
from imswitch.imcontrol.model.managers.LEDMatrixs import ledgeometry


F     = lambda x: np.fft.fft2(x)
IF    = lambda x: np.fft.ifft2(x)

# Order of the images in every DPC stack: LEDs whose illumination NA points along +x / -x /
# +y / -y of the camera image (x = column, y = row), see ledgeometry.dpc_patterns.
DPC_PATTERNS = ("+x", "-x", "+y", "-y")

# =========================
# Dataclasses (API-stable)
# =========================
@dataclass
class DPCParams:
    """DPC / FPM parameters"""
    pixelsize: float = 0.2  # micrometers, sample plane
    wavelength: float = 0.53  # micrometers
    na: float = 0.3  # objective NA
    nai: float = 0.3  # outer illumination NA of the DPC half patterns
    nai_inner: float = 0.0  # inner illumination NA (annular DPC)
    n: float = 1.0
    led_intensity_r: int = 0  # Red channel intensity
    led_intensity_g: int = 255  # Green channel intensity (default)
    led_intensity_b: int = 0  # Blue channel intensity
    wait_time: float = 0.2  # seconds between LED changes
    frame_sync: int = 2  # number of frames to wait for fresh frame
    save_images: bool = False
    save_directory: str = ""
    reg_u: float = 1e-1  # Tikhonov regularization for absorption
    reg_p: float = 5e-3  # Tikhonov regularization for phase
    fpm_na_max: float = 1.0  # FPM scans every LED up to this illumination NA
    display: str = "phase"  # streamed image: phase, absorption, dpc_x, dpc_y or a raw image +x/-x/+y/-y

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DPCState:
    """DPC processing state"""
    is_processing: bool = False
    is_paused: bool = False
    frame_count: int = 0
    processed_count: int = 0
    last_process_time: float = 0.0
    processing_fps: float = 0.0
    task: str = ""  # "fpm" / "calibration" / "snap" while an LED scan runs
    task_progress: float = 0.0
    task_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DPCController(ImConWidgetController):
    """
    DPC and FPM with any LED source (geodesic dome, ring board, LED matrix).

    LEDs are chosen by their illumination NA in camera frame (LEDMatrixManager.illuminationNA,
    ledgeometry.py), so the DPC half patterns, the DPC transfer functions and the FPM k-vectors
    all come from the real LED positions. start_led_calibration measures the LED -> camera
    rotation / mirror from a defocused sample.

    - live DPC (MJPEG stream of the phase), snap_dpc saves one raw stack + reconstruction
    - start_fpm_acquisition: one image per LED, TIFF stack + JSON with the k-vectors
    """

    sigImageReceived = Signal()
    sigDPCProcessorImageComputed = Signal(np.ndarray, str)
    sigDPCStateChanged = Signal(object)  # state_dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # switch to detect if a recording is in progress
        self.isRecording = False

        # dpc parameters from setup (optional)
        dpcManager = self._master.dpcManager
        self._params = DPCParams(
            pixelsize=getattr(dpcManager, 'pixelsize', 0.2),
            wavelength=getattr(dpcManager, 'wavelength', 0.53),
            na=getattr(dpcManager, 'NA', 0.3),
            nai=getattr(dpcManager, 'NAi', 0.3),
            n=getattr(dpcManager, 'n', 1.0),
            save_directory=str(os.path.join(dirtools.UserFileDirs.Data, "DPC"))
        )
        self._state = DPCState()
        self._processing_lock = threading.Lock()
        self._stop_processing_event = threading.Event()
        self._stop_task_event = threading.Event()

        # Get LEDMatrix manager
        allLEDMatrixNames = self._master.LEDMatrixsManager.getAllDeviceNames()
        if len(allLEDMatrixNames) == 0:
            self._logger.error("No LEDMatrix found in setup")
            return
        self.ledMatrixName = allLEDMatrixNames[0]
        self.ledMatrix = self._master.LEDMatrixsManager[self.ledMatrixName]

        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]
        self.detector.startAcquisition()

        # initialize DPC processor
        self.DPCProcessor = DPCProcessor(self._params)

        # MJPEG streaming: latest phase JPEG, sent to every connected client
        self._mjpeg_frame = None
        self._jpeg_quality = 85

        # Processing queue - decouples capture from reconstruction
        self._processing_queue = queue.Queue(maxsize=2)

        # Processing threads
        self._capture_thread = None
        self._reconstruction_thread = None
        self._task_thread = None

        # Performance monitoring
        self._perf_process_times = []
        self._perf_window_size = 30

        # last LED scan analysis (start_led_calibration)
        self._brightfield = {}  # LED id -> bool
        self._last_calibration = None
        self._views = {}  # images of the last DPC result by display name

        self._logger.info("DPCController initialized successfully")

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_dpc_processing()
        self._stop_task_event.set()

    # =========================
    # LED patterns
    # =========================
    def _rgb(self):
        return (self._params.led_intensity_r, self._params.led_intensity_g,
                self._params.led_intensity_b)

    def _dpc_patterns(self) -> Dict[str, np.ndarray]:
        """{pattern: LED ids} of the four DPC half patterns at the current params / calibration."""
        masks = ledgeometry.dpc_patterns(self.ledMatrix.illuminationNA(),
                                         self._params.nai, self._params.nai_inner)
        return {name: np.flatnonzero(masks[name]) for name in DPC_PATTERNS}

    def _check_dpc_patterns(self, patterns) -> str:
        """Error message if a half pattern has no LED inside the objective NA, else ''."""
        r = np.hypot(*self.ledMatrix.illuminationNA().T)
        empty = [n for n, ids in patterns.items() if not np.any(r[ids] <= self._params.na)]
        if empty:
            return (f"DPC patterns {empty} have no LED inside the objective NA {self._params.na}: "
                    f"raise 'nai' / 'na' or check the LED calibration (offset_mm)")
        return ""

    # ────────────────────────────────────────────────────────────────────────
    # Deterministic frame-acquisition helpers (same pattern as ExperimentController)
    # In software-trigger mode every DPC exposure is fired *after* the LED
    # pattern is set, guaranteeing no stale pre-pattern frame can slip through.
    # ────────────────────────────────────────────────────────────────────────

    def _getExposureTimeSec(self) -> float:
        """Return current detector exposure time in seconds (best effort)."""
        try:
            exp_ms = self.detector.getParameter("exposure")
            if exp_ms is not None and float(exp_ms) > 0:
                return float(exp_ms) / 1000.0
        except Exception:
            pass
        try:
            return self.detector._camera.exposure_time / 1e6
        except Exception:
            return 0.1

    def _beginTriggeredAcquisition(self) -> bool:
        """Switch the detector to software-trigger mode for deterministic post-LED-change grabs.

        In software-trigger mode every exposure is initiated *after* the LED pattern is set,
        so no pre-pattern photons reach the sensor. Falls back transparently to the
        freerun frame-number polling path when the detector does not support it.

        Returns True when the camera was successfully switched.
        """
        if not hasattr(self.detector, "setTriggerSource") or not hasattr(self.detector, "snapSync"):
            self._useTriggeredGrab = False
            return False
        try:
            ok = bool(self.detector.setTriggerSource("software"))
        except Exception as e:
            self._logger.warning(f"Could not enable software trigger: {e}")
            ok = False
        self._useTriggeredGrab = ok
        if ok:
            self._logger.info("DPCController: camera in software-trigger mode (deterministic grabs)")
        else:
            self._logger.warning("DPCController: software trigger not supported, using freerun fallback")
        return ok

    def _endTriggeredAcquisition(self) -> None:
        """Restore continuous (free-run) acquisition — safe to call unconditionally."""
        if not getattr(self, "_useTriggeredGrab", False):
            return
        self._useTriggeredGrab = False
        try:
            if hasattr(self.detector, "setTriggerSource"):
                self.detector.setTriggerSource("continuous")
                self._logger.info("DPCController: camera restored to continuous mode")
        except Exception as e:
            self._logger.warning(f"Could not restore continuous mode: {e}")

    def grabCameraFrame(self, frameSync: int = 2, returnFrameNumber: bool = False):
        """Return a single camera frame guaranteed to be captured after this call.

        Fast path (triggered): fires one software trigger via detector.snapSync()
        and returns exactly the resulting frame — ideal for DPC because the LED
        pattern is already stable before the trigger fires.

        Fallback (freerun): flushes ring-buffer, then polls by frame-number until
        the counter advances by frameSync counts.  Timeout is exposure-time-scaled.
        """
        # ── Fast path ────────────────────────────────────────────────────────
        if getattr(self, "_useTriggeredGrab", False) and hasattr(self.detector, "snapSync"):
            mFrame = self.detector.snapSync(timeout=max(2.0, self._getExposureTimeSec() * 4 + 1.0))
            if returnFrameNumber:
                fn = self.detector.getFrameNumber() if hasattr(self.detector, "getFrameNumber") else -1
                return mFrame, fn
            return mFrame

        # ── Fallback path ────────────────────────────────────────────────────
        try:
            if hasattr(self.detector, "flushBuffer"):
                self.detector.flushBuffer()
        except Exception:
            pass

        exposure_s = self._getExposureTimeSec()
        timeoutFrameRequest = max(1.0, (frameSync + 2) * exposure_s + 0.5)
        cTime = time.time()
        lastFrameNumber = -1
        currentFrameNumber = -1
        mFrame = None

        while True:
            mFrame, currentFrameNumber = self.detector.getLatestFrame(returnFrameNumber=True)
            if lastFrameNumber == -1:
                lastFrameNumber = currentFrameNumber
            if time.time() - cTime > timeoutFrameRequest:
                if mFrame is None:
                    mFrame = self.detector.getLatestFrame(returnFrameNumber=False)
                self._logger.warning(
                    f"grabCameraFrame: timed out after {timeoutFrameRequest:.1f}s "
                    f"(exposure={exposure_s*1000:.0f}ms, frameSync={frameSync})"
                )
                break
            if currentFrameNumber <= lastFrameNumber + frameSync:
                time.sleep(0.01)
            else:
                break

        if returnFrameNumber:
            return mFrame, currentFrameNumber
        return mFrame

    def _grab(self) -> np.ndarray:
        """Fresh 2-D frame in the camera's dtype (colour cameras: channel mean)."""
        frame = self.grabCameraFrame(frameSync=self._params.frame_sync)
        if frame is None:
            raise RuntimeError("no camera frame")
        frame = np.asarray(frame)
        return frame.mean(axis=2).astype(frame.dtype) if frame.ndim == 3 else frame

    def _capture_dpc_stack(self):
        """(4, H, W) stack in DPC_PATTERNS order and the LED ids lit for each image,
        None when stopped."""
        patterns = self._dpc_patterns()
        stack = []
        for pattern_name in DPC_PATTERNS:
            if self._stop_processing_event.is_set() or self._stop_task_event.is_set():
                return None
            self.ledMatrix.setLEDs(patterns[pattern_name], self._rgb())
            time.sleep(self._params.wait_time)
            stack.append(self._grab())
            self._state.frame_count += 1
        return np.array(stack), patterns

    def _process(self, stack, patterns, save):
        """Reconstruct, emit and stream one DPC stack."""
        na = self.ledMatrix.illuminationNA()
        source_na = [na[patterns[name]] for name in DPC_PATTERNS]
        dpc_x, dpc_y, qdpc = self.DPCProcessor.process_stack(stack, source_na, save=save)
        self.sigDPCProcessorImageComputed.emit(dpc_x, "DPC x")
        self.sigDPCProcessorImageComputed.emit(dpc_y, "DPC y")
        self._views = {**dict(zip(DPC_PATTERNS, stack)), "dpc_x": dpc_x, "dpc_y": dpc_y,
                       "absorption": qdpc.real, "phase": qdpc.imag}
        self._show()
        self._state.processed_count += 1

    def _show(self):
        """Stream the selected image (params.display) of the last DPC result."""
        views = self._views
        if self._params.display in views:
            self._add_to_mjpeg_stream(views[self._params.display])

    def _capture_loop(self):
        """Capture loop - captures DPC stacks and adds to processing queue"""
        self._logger.info("DPC capture loop started")

        while not self._stop_processing_event.is_set():
            try:
                captured = self._capture_dpc_stack()
                if captured is None:
                    continue

                # Add to processing queue (non-blocking, drop old if full)
                try:
                    self._processing_queue.put_nowait(captured)
                except queue.Full:
                    # Drop oldest stack and add new one
                    try:
                        self._processing_queue.get_nowait()
                        self._processing_queue.put_nowait(captured)
                    except queue.Empty:
                        pass

            except Exception as e:
                self._logger.error(f"Error in DPC capture loop: {e}", exc_info=True)
                time.sleep(0.1)

        # Turn off LEDs when done
        self.ledMatrix.setLEDs([], getReturn=False)
        self._logger.info("DPC capture loop stopped")

    def _reconstruction_loop(self):
        """Reconstruction loop - processes stacks from queue"""
        self._logger.info("DPC reconstruction loop started")

        while not self._stop_processing_event.is_set():
            try:
                # Get stack from queue with timeout
                try:
                    stack, patterns = self._processing_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                t_start = time.time()
                self._process(stack, patterns, save=self._params.save_images)

                # Update performance metrics
                process_time = time.time() - t_start
                self._state.last_process_time = process_time

                self._perf_process_times.append(process_time)
                if len(self._perf_process_times) > self._perf_window_size:
                    self._perf_process_times.pop(0)

                avg_time = np.mean(self._perf_process_times)
                self._state.processing_fps = 1.0 / avg_time if avg_time > 0 else 0.0

                self._emit_state_changed()

            except Exception as e:
                self._logger.error(f"Error in DPC reconstruction loop: {e}", exc_info=True)
                time.sleep(0.1)

        self._logger.info("DPC reconstruction loop stopped")

    def _add_to_mjpeg_stream(self, image):
        """Add processed image to MJPEG stream (1-99 percentile contrast)"""
        try:
            img = np.nan_to_num(np.asarray(image, np.float64))
            lo, hi = np.percentile(img, (1, 99))
            img_u8 = np.uint8(np.clip((img - lo) / (hi - lo + 1e-12), 0, 1) * 255)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            _, buffer = cv2.imencode('.jpg', img_u8, encode_param)
            self._mjpeg_frame = buffer.tobytes()
        except Exception as e:
            self._logger.error(f"Error adding image to MJPEG stream: {e}")

    def _emit_state_changed(self):
        """Emit state changed signal"""
        self.sigDPCStateChanged.emit(self._state.to_dict())

    # =========================
    # LED scans (FPM, calibration, DPC snap) in a background thread
    # =========================
    def _start_task(self, name: str, body) -> Dict[str, Any]:
        with self._processing_lock:
            if self._state.is_processing or self._state.task:
                return {"status": "busy", "state": self._state.to_dict()}
            self._state.task, self._state.task_progress, self._state.task_message = name, 0.0, "starting"
            self._stop_task_event.clear()
            self._task_thread = threading.Thread(target=self._run_task, args=(name, body),
                                                 daemon=True, name=f"DPC-{name}")
            self._task_thread.start()
        return {"status": "started", "state": self._state.to_dict()}

    def _run_task(self, name, body):
        try:
            self._ensure_camera_running()
            self._beginTriggeredAcquisition()
            self._state.task_message = body()
        except Exception as e:
            self._logger.error(f"{name} failed: {e}", exc_info=True)
            self._state.task_message = f"{name} failed: {e}"
        finally:
            self._endTriggeredAcquisition()
            self.ledMatrix.setLEDs([], getReturn=False)
            self._state.task = ""
            self._emit_state_changed()

    def _scan_leds(self, ids):
        """One frame per LED, lit one at a time: (N, H, W), or None when stopped."""
        frames = []
        for k, i in enumerate(ids):
            if self._stop_task_event.is_set():
                return None
            self.ledMatrix.setLEDs([i], self._rgb())
            time.sleep(self._params.wait_time)
            frames.append(self._grab())
            self._state.task_progress = (k + 1) / len(ids)
            self._state.task_message = f"LED {i} ({k + 1}/{len(ids)})"
            self._emit_state_changed()
        return np.array(frames)

    def _fpm(self) -> str:
        na = self.ledMatrix.illuminationNA()
        ids = np.flatnonzero(np.hypot(*na.T) <= self._params.fpm_na_max)
        stack = self._scan_leds(ids)
        if stack is None:
            return "FPM aborted"
        p = self._params
        os.makedirs(p.save_directory, exist_ok=True)
        stem = os.path.join(p.save_directory, datetime.now().strftime("%Y_%m_%d-%H-%M-%S") + "_FPM")
        tifffile.imwrite(stem + ".tif", stack)
        cal = self.ledMatrix.calibration
        na_led = ledgeometry.illumination_na(self.ledMatrix.positions, {"offset_mm": cal["offset_mm"]})
        meta = {
            "images": "one per LED, in led_ids order",
            "led_ids": ids.tolist(),
            "na_camera": na[ids].tolist(),
            "na_camera_axes": "x = image column, y = image row",
            "na_led_frame": na_led[ids].tolist(),
            "led_positions_mm": self.ledMatrix.positions[ids].tolist(),
            "led_layout": self.ledMatrix.ledLayout,
            "led_calibration": cal,
            "led_rgb": list(self._rgb()),
            "wavelength_um": p.wavelength,
            "pixelsize_um": p.pixelsize,
            "na_objective": p.na,
            "n": p.n,
            "exposure_ms": self._getExposureTimeSec() * 1e3,
        }
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        return f"saved {len(ids)} images to {stem}.tif"

    def _calibrate(self) -> str:
        """Scan all LEDs on a defocused sample: brightfield / darkfield map (objective NA
        estimate) and the image shift per brightfield LED -> rotation / mirror (applied)."""
        ids = np.arange(len(self.ledMatrix.positions))
        stack = self._scan_leds(ids)
        if stack is None:
            return "calibration aborted"
        cal = self.ledMatrix.calibration
        na_led = ledgeometry.illumination_na(self.ledMatrix.positions, {"offset_mm": cal["offset_mm"]})
        r = np.hypot(*na_led.T)
        bf = ledgeometry.split_brightfield(ledgeometry.brightfield_scores(stack))
        self._brightfield = dict(zip(ids.tolist(), bf.tolist()))
        use = bf if bf.sum() >= 3 else np.ones_like(bf)
        ref = stack[use][np.argmin(r[use])]
        shifts = np.array([ledgeometry.image_shift(ref, img) for img in stack[use]])
        fit = ledgeometry.fit_orientation(na_led[use], shifts)
        spread = float(np.sqrt(np.mean(np.sum((shifts - shifts.mean(axis=0)) ** 2, axis=1))))
        na_obj = float((r[bf].max() + r[~bf].min()) / 2) if bf.any() and (~bf).any() else None
        ok = spread > 1.0 and fit["rms_px"] < 0.5 * spread
        self._last_calibration = {**fit, "shift_spread_px": spread, "n_brightfield": int(bf.sum()),
                                  "na_objective_estimate": na_obj, "applied": ok,
                                  "shifts_px": {int(i): s.tolist() for i, s in zip(ids[use], shifts)}}
        msg = f"{int(bf.sum())} brightfield LEDs" + (f", objective NA ~{na_obj:.2f}" if na_obj else "")
        if not ok:
            return msg + f"; shifts too small / inconsistent ({spread:.1f} px, rms {fit['rms_px']:.1f} px): defocus more"
        self._save_calibration({**cal, "rotation_deg": round(fit["rotation_deg"], 1), "flip": fit["flip"]})
        return msg + f"; applied rotation {fit['rotation_deg']:.1f} deg, flip {fit['flip']}"

    def _snap(self) -> str:
        msg = self._check_dpc_patterns(self._dpc_patterns())
        if msg:
            return msg
        captured = self._capture_dpc_stack()
        if captured is None:
            return "DPC snap aborted"
        self._process(*captured, save=True)
        return f"DPC saved to {self._params.save_directory}"

    def _save_calibration(self, cal):
        """Apply the LED -> camera calibration and store it in the setup file."""
        self.ledMatrix.calibration = cal
        self._setupInfo.LEDMatrixs[self.ledMatrixName].managerProperties["ledCalibration"] = cal
        try:
            configfiletools.saveSetupInfo(configfiletools.loadOptions()[0], self._setupInfo)
        except Exception as e:
            self._logger.warning(f"LED calibration applied but not saved to the setup file: {e}")

    # =========================
    # API: Parameter Control
    # =========================
    @APIExport(runOnUIThread=True)
    def get_dpc_params(self) -> Dict[str, Any]:
        """Get current DPC parameters"""
        return self._params.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def set_dpc_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set DPC parameters

        Args:
            params: Dictionary of parameters to update

        Returns:
            Updated parameter dictionary
        """
        with self._processing_lock:
            for key, value in params.items():
                if hasattr(self._params, key):
                    setattr(self._params, key, value)
                    self._logger.debug(f"Updated DPC parameter: {key} = {value}")
            self.DPCProcessor.update_parameters(self._params)
        self._show()  # a new display choice shows the last result at once

        return self.get_dpc_params()

    @APIExport(runOnUIThread=True)
    def get_dpc_state(self) -> Dict[str, Any]:
        """Get current DPC processing state"""
        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def get_led_layout(self) -> Dict[str, Any]:
        """Every LED with its illumination NA in camera frame (x = column, y = row), the DPC
        patterns it belongs to and its brightfield flag from the last calibration scan."""
        na = self.ledMatrix.illuminationNA()
        member = {}
        for name, ids in self._dpc_patterns().items():
            for i in ids:
                member.setdefault(int(i), []).append(name)
        leds = [{"id": i, "na_x": float(n[0]), "na_y": float(n[1]), "dpc": member.get(i, []),
                 "brightfield": self._brightfield.get(i)} for i, n in enumerate(na)]
        return {"layout": self.ledMatrix.ledLayout, "calibration": self.ledMatrix.calibration,
                "last_calibration": self._last_calibration, "na_objective": self._params.na,
                "leds": leds}

    @APIExport(runOnUIThread=True)
    def set_led_calibration(self, rotation_deg: float = 0.0, flip: bool = False,
                            offset_x_mm: float = 0.0, offset_y_mm: float = 0.0,
                            offset_z_mm: float = 0.0) -> Dict[str, Any]:
        """Set the LED -> camera calibration by hand (rotation of the LED NA vectors in the
        image, mirror in x first; offset = position of the LED structure relative to its
        layout coordinates, for a planar board offset_z_mm is its height above the sample)."""
        self._save_calibration({"rotation_deg": float(rotation_deg), "flip": bool(flip),
                                "offset_mm": [float(offset_x_mm), float(offset_y_mm), float(offset_z_mm)]})
        return self.get_led_layout()

    # =========================
    # API: Processing Control
    # =========================
    @APIExport(runOnUIThread=True)
    def start_dpc_processing(self) -> Dict[str, Any]:
        """Start DPC processing"""
        with self._processing_lock:
            if self._state.is_processing:
                return {"status": "already_running", "state": self._state.to_dict()}
            if self._state.task:
                return {"status": "busy", "state": self._state.to_dict()}
            error = self._check_dpc_patterns(self._dpc_patterns())
            if error:
                self._state.task_message = error
                return {"status": "error", "message": error, "state": self._state.to_dict()}

            # Reset state
            self._state.is_processing = True
            self._state.is_paused = False
            self._state.frame_count = 0
            self._state.processed_count = 0
            self._state.task_message = ""
            self._perf_process_times = []

            # Ensure camera is running
            self._ensure_camera_running()

            # Activate software-trigger mode so each grabCameraFrame() fires
            # exactly one exposure *after* the LED pattern is settled.
            self._beginTriggeredAcquisition()

            # Clear processing queue
            while not self._processing_queue.empty():
                try:
                    self._processing_queue.get_nowait()
                except queue.Empty:
                    break

            # Start capture and reconstruction threads
            self._stop_processing_event.clear()

            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="DPC-Capture"
            )
            self._capture_thread.start()

            self._reconstruction_thread = threading.Thread(
                target=self._reconstruction_loop,
                daemon=True,
                name="DPC-Reconstruction"
            )
            self._reconstruction_thread.start()

            self._emit_state_changed()
            self._logger.info("DPC processing started")

            return {"status": "started", "state": self._state.to_dict()}

    @APIExport(runOnUIThread=True)
    def stop_dpc_processing(self) -> Dict[str, Any]:
        """Stop DPC processing"""
        with self._processing_lock:
            if not self._state.is_processing:
                return {"status": "not_running", "state": self._state.to_dict()}

            # Stop processing
            self._stop_processing_event.set()

            if self._capture_thread is not None:
                self._capture_thread.join(timeout=2.0)
                self._capture_thread = None

            if self._reconstruction_thread is not None:
                self._reconstruction_thread.join(timeout=2.0)
                self._reconstruction_thread = None

            # Restore continuous acquisition mode after DPC session.
            self._endTriggeredAcquisition()

            self._state.is_processing = False
            self._state.is_paused = False

            self._emit_state_changed()
            self._logger.info("DPC processing stopped")

            return {"status": "stopped", "state": self._state.to_dict()}

    @APIExport(runOnUIThread=True)
    def snap_dpc(self) -> Dict[str, Any]:
        """One DPC stack: reconstruct, stream and save raw images + phase / absorption."""
        return self._start_task("snap", self._snap)

    @APIExport(runOnUIThread=True)
    def start_fpm_acquisition(self) -> Dict[str, Any]:
        """One image per LED up to fpm_na_max, saved as TIFF stack + JSON with the LED
        k-vectors (illumination NA in camera frame) for FPM reconstruction."""
        return self._start_task("fpm", self._fpm)

    @APIExport(runOnUIThread=True)
    def start_led_calibration(self) -> Dict[str, Any]:
        """Scan all LEDs on a sample defocused by ~50-100 um: finds the brightfield LEDs and
        the LED -> camera rotation / mirror (applied when the image shifts are consistent).
        The defocus direction is unknown, so a 180 deg rotation flips the phase sign."""
        return self._start_task("calibration", self._calibrate)

    @APIExport(runOnUIThread=True)
    def stop_task(self) -> Dict[str, Any]:
        """Abort a running FPM / calibration / snap scan."""
        self._stop_task_event.set()
        return {"status": "stopping", "state": self._state.to_dict()}

    @APIExport(runOnUIThread=False)
    def get_dpc_wotf(self, wavelength: Optional[float] = None, na: Optional[float] = None,
                     nai: Optional[float] = None, nai_inner: Optional[float] = None):
        """PNG of the weak-object transfer functions of the four DPC patterns, absorption (Hu,
        real) and phase (Hp, imaginary), in NA units, camera frame (x right, y down). Unset
        arguments use the current parameters; the LEDs follow the current calibration."""
        from fastapi.responses import Response
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle
        p = self._params
        wavelength = p.wavelength if wavelength is None else wavelength
        na = p.na if na is None else na
        nai = p.nai if nai is None else nai
        nai_inner = p.nai_inner if nai_inner is None else nai_inner
        fig = Figure(figsize=(10, 5.4), layout="constrained")
        try:
            led_na = self.ledMatrix.illuminationNA()
            masks = ledgeometry.dpc_patterns(led_na, nai, nai_inner)
            lim = 2.2 * na  # frequency grid out to 2.2 NA / wavelength, the WOTFs end at 2 NA
            solver = DPCSolver((256, 256), wavelength, na, wavelength / (2 * lim),
                               [led_na[masks[name]] for name in DPC_PATTERNS])
            axes = fig.subplots(2, 4)
            for row, (wotf, label) in enumerate([(solver.Hu.real, "absorption, real"),
                                                 (solver.Hp.imag, "phase, imaginary")]):
                vmax = np.abs(wotf).max()
                for ax, h, name in zip(axes[row], wotf, DPC_PATTERNS):
                    im = ax.imshow(np.fft.fftshift(h), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                   extent=(-lim, lim, lim, -lim))
                    ax.add_patch(Circle((0, 0), na, fill=False, ls="--", lw=0.7))
                    ax.set_title(f"{name}: {label}", fontsize=9)
                    ax.tick_params(labelsize=7)
                axes[row][0].set_ylabel("NA y", fontsize=8)
                fig.colorbar(im, ax=axes[row], shrink=0.85)
            for ax in axes[1]:
                ax.set_xlabel("NA x", fontsize=8)
            fig.suptitle(f"WOTF  \u03bb {wavelength:g} \u00b5m, objective NA {na:g} (dashed), "
                         f"illumination NA {nai_inner:g}-{nai:g}", fontsize=10)
        except Exception as e:  # e.g. a pattern without LED inside the objective NA
            fig.clear()
            fig.text(0.5, 0.5, f"No WOTF: {e}.\nRaise the illumination NA or check the LED calibration.",
                     ha="center", va="center", wrap=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        return Response(buf.getvalue(), media_type="image/png")

    @APIExport(runOnUIThread=False)
    def mjpeg_stream_dpc(self, startStream: bool = True, jpeg_quality: int = 85):
        """MJPEG stream (multipart/x-mixed-replace) of the reconstructed phase. A new client
        gets the latest image at once, then every new one."""
        from fastapi.responses import StreamingResponse
        if not startStream:
            return {"status": "success", "message": "stream stopped"}
        self._jpeg_quality = max(1, min(100, jpeg_quality))

        def generate():
            sent = None
            while True:
                frame = self._mjpeg_frame
                if frame is sent:
                    time.sleep(0.1)
                    yield b""  # back to the server loop, so a closed connection ends here
                    continue
                sent = frame
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii") + frame + b"\r\n")

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame",
                                 headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                                          "X-Accel-Buffering": "no"})

    def _ensure_camera_running(self):
        """Ensure camera is running"""
        try:
            if not self.detector.isAcquiring():
                self.detector.startAcquisition()
        except Exception:
            pass


'''#####################################
# DPC PROCESSOR
#####################################'''

class DPCProcessor(object):
    """Reconstructs DPC stacks; the solver is rebuilt when the frame shape, the optics or the
    lit LEDs change."""

    def __init__(self, params: DPCParams):
        self._logger = initLogger(self, tryInheritParent=False)
        self.update_parameters(params)

    def update_parameters(self, params: DPCParams):
        """Update processor parameters from DPCParams"""
        self.params = params
        self._solver = None
        self._key = None

    def process_stack(self, stack: np.ndarray, source_na, save: bool = False):
        """
        Process a 4-image DPC stack (DPC_PATTERNS order).

        Args:
            stack: (4, H, W) DPC images
            source_na: per image, (M, 2) illumination NA of the lit LEDs (camera frame)
            save: Whether to save raw images and results

        Returns:
            (dpc_x, dpc_y, qdpc) with qdpc = absorption + 1j * phase
        """
        p = self.params
        key = (stack.shape[1:], p.wavelength, p.na, p.pixelsize, p.reg_u, p.reg_p,
               tuple(np.asarray(s).tobytes() for s in source_na))
        if key != self._key:
            self._solver = DPCSolver(stack.shape[1:], p.wavelength, p.na, p.pixelsize, source_na)
            self._solver.setTikhonovRegularization(reg_u=p.reg_u, reg_p=p.reg_p)
            self._key = key

        imgs = stack.astype(np.float64)
        qdpc = self._solver.solve(dpc_imgs=imgs)[0]
        dpc_x = (imgs[0] - imgs[1]) / (imgs[0] + imgs[1] + 1e-10)
        dpc_y = (imgs[2] - imgs[3]) / (imgs[2] + imgs[3] + 1e-10)

        if save and p.save_directory:
            os.makedirs(p.save_directory, exist_ok=True)
            stem = os.path.join(p.save_directory, datetime.now().strftime("%Y_%m_%d-%H-%M-%S"))
            save_image(stem + "_DPC_raw.tif", stack, axes="CYX", channel_names=DPC_PATTERNS,
                       pixel_size_um=p.pixelsize, name="DPC raw (+x, -x, +y, -y)")
            save_image(stem + "_DPC_phase.tif", qdpc.imag.astype(np.float32),
                       pixel_size_um=p.pixelsize, name="qDPC phase")
            save_image(stem + "_DPC_absorption.tif", qdpc.real.astype(np.float32),
                       pixel_size_um=p.pixelsize, name="qDPC absorption")
            self._logger.info(f"Saved DPC results to {stem}_DPC_*.tif")

        return dpc_x, dpc_y, qdpc

# (C) Wallerlab 2019
# https://github.com/Waller-Lab/DPC/blob/master/python_code/dpc_algorithm.py
# Source = the lit LEDs (one delta per LED) instead of ideal half discs. WOTFGen's correlation
# puts a source point at -NA / wavelength for an illumination exp(+i 2 pi NA / wavelength r),
# hence the minus sign below (checked against a forward simulation in test_led_geometry.py).

class DPCSolver:
    def __init__(self, shape, wavelength, na, pixelsize, source_na):
        """shape: (H, W); source_na: per DPC image, (M, 2) illumination NA of its LEDs
        in camera frame (x = column, y = row)."""
        self.shape = tuple(shape)
        self.wavelength = wavelength
        self.na = na
        self.pixel_size = pixelsize
        self.dpc_num = len(source_na)
        fy = np.fft.fftfreq(self.shape[0], d=pixelsize)[:, None]
        fx = np.fft.fftfreq(self.shape[1], d=pixelsize)[None, :]
        self.pupil = (fx ** 2 + fy ** 2 <= (na / wavelength) ** 2).astype(np.complex128)
        self.source = np.zeros((self.dpc_num,) + self.shape)
        for src, nas in zip(self.source, source_na):
            nas = -np.reshape(nas, (-1, 2))
            nas = nas[np.hypot(nas[:, 0], nas[:, 1]) <= na]  # darkfield LEDs add nothing, but alias on a coarse grid
            ix = np.round(nas[:, 0] / wavelength * self.shape[1] * pixelsize).astype(int) % self.shape[1]
            iy = np.round(nas[:, 1] / wavelength * self.shape[0] * pixelsize).astype(int) % self.shape[0]
            np.add.at(src, (iy, ix), 1.0)
        self.WOTFGen()

    def setTikhonovRegularization(self, reg_u = 1e-6, reg_p = 1e-6):
        self.reg_u      = reg_u
        self.reg_p      = reg_p

    def normalization(self):
        for img in self.dpc_imgs:
            img          /= uniform_filter(img, size=img.shape[0]//2)
            meanIntensity = img.mean()
            img          /= meanIntensity        # normalize intensity with DC term
            img          -= 1.0                  # subtract the DC term

    def WOTFGen(self):
        self.Hu = []
        self.Hp = []
        for rotIdx in range(self.source.shape[0]):
            FSP_cFP  = F(self.source[rotIdx]*self.pupil)*F(self.pupil).conj()
            I0       = (self.source[rotIdx]*self.pupil*self.pupil.conj()).sum().real
            if I0 == 0:
                raise ValueError(f"DPC image {rotIdx} has no LED inside the objective NA")
            self.Hu.append(2.0*IF(FSP_cFP.real)/I0)
            self.Hp.append(2.0j*IF(1j*FSP_cFP.imag)/I0)
        self.Hu = np.asarray(self.Hu)
        self.Hp = np.asarray(self.Hp)

    def solve(self, dpc_imgs, **kwargs):
        self.dpc_imgs = dpc_imgs.astype('float64')
        self.normalization()

        dpc_result  = []
        AHA         = [(self.Hu.conj()*self.Hu).sum(axis=0)+self.reg_u,            (self.Hu.conj()*self.Hp).sum(axis=0),\
                       (self.Hp.conj()*self.Hu).sum(axis=0)           , (self.Hp.conj()*self.Hp).sum(axis=0)+self.reg_p]
        determinant = AHA[0]*AHA[3]-AHA[1]*AHA[2]

        # Avoid division by zero - add small epsilon where determinant is zero
        determinant = np.where(np.abs(determinant) < 1e-10, 1e-10, determinant)

        for frame_index in range(self.dpc_imgs.shape[0]//self.dpc_num):
            fIntensity = np.asarray([F(self.dpc_imgs[frame_index*self.dpc_num+image_index]) for image_index in range(self.dpc_num)])
            AHy        = np.asarray([(self.Hu.conj()*fIntensity).sum(axis=0), (self.Hp.conj()*fIntensity).sum(axis=0)])

            # Compute with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                absorption = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
                phase      = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real

            # Replace NaN/inf with 0
            absorption = np.nan_to_num(absorption, nan=0.0, posinf=0.0, neginf=0.0)
            phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

            dpc_result.append(absorption+1.0j*phase)

        return np.asarray(dpc_result)




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
