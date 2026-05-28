"""OpticalFlowController.

Camera <-> stage rotation alignment using optical flow.

Moves the stage in a straight line at constant slow velocity and computes
optical flow between consecutive camera frames. The angle between the stage
axis and the camera axis is reported live to the frontend so the user can
rotate the camera until the angle is zero -- making downstream stitching
trivial.

Architecture mirrors :mod:`AutofocusController`:
  * a ``MovementController`` runs the stage move in a background thread so
    we can poll frames without blocking on the move;
  * a thread-safe state machine guards the lifecycle;
  * per-frame results are pushed to the frontend via a Signal that the
    WebSocket bridge forwards into Redux for a live Plotly chart.

Recommendations for the user (documented here so we can show them in the
UI as hints):
  * use a feature-rich sample -- clear water / uniform plastic give no flow;
  * make sure exposure time < inter-frame interval, otherwise consecutive
    frames are too similar for the Lucas-Kanade tracker;
  * at the default speed (100 um/s) and a typical 0.5 um/px pixel size the
    stage moves ~200 px/s -- the detector must deliver >= 5 fps.
"""

import threading
import time
from enum import Enum
from typing import Optional

import numpy as np

from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 should be available in ImSwitch envs
    cv2 = None  # type: ignore
    _HAS_CV2 = False


# Default axis used when none provided
gAxis = "X"


class FlowState(Enum):
    """Lifecycle of an optical-flow measurement."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    ABORTED = "aborted"
    FINISHED = "finished"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Reused asynchronous mover (kept local so the controller stays self-contained).
# Mirrors ``AutofocusController.MovementController`` line-for-line.
# ---------------------------------------------------------------------------
class _MovementController:
    """Asynchronous mover with abort capability."""

    def __init__(self, stages):
        self.stages = stages
        self._lock = threading.Lock()
        self.target_reached = True
        self.target_position = None
        self.axis = None
        self.speed = None
        self.is_absolute = True
        self._thread: Optional[threading.Thread] = None
        self._abort_flag = False

    def move_to_position(self, value, axis=gAxis, speed=None, is_absolute=True):
        with self._lock:
            self._abort_flag = False
            self.target_reached = False
            self.target_position = value
            self.axis = axis
            self.speed = speed
            self.is_absolute = is_absolute
        self._thread = threading.Thread(target=self._move, daemon=True)
        self._thread.start()

    def _move(self):
        with self._lock:
            value = self.target_position
            axis = self.axis
            speed = self.speed
            is_absolute = self.is_absolute
        try:
            if self._abort_flag:
                return
            self.stages.move(
                value=value,
                axis=axis,
                speed=speed,
                is_absolute=is_absolute,
                is_blocking=True,
            )
        finally:
            with self._lock:
                self.target_reached = True

    def is_target_reached(self) -> bool:
        with self._lock:
            return bool(self.target_reached)

    def abort(self):
        with self._lock:
            self._abort_flag = True
        if hasattr(self.stages, "forceStop"):
            try:
                if self.axis:
                    self.stages.forceStop(self.axis)
                else:
                    self.stages.forceStop(gAxis)
            except Exception:
                pass
        with self._lock:
            self.target_reached = True


def _circular_mean_std_deg(angles_deg: np.ndarray) -> tuple:
    """Return circular mean and circular std (degrees) over [-180, 180]."""
    if angles_deg.size == 0:
        return float("nan"), float("nan")
    rad = np.deg2rad(angles_deg.astype(float))
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    mean = float(np.rad2deg(np.arctan2(s, c)))
    # Circular standard deviation (Mardia)
    r = float(np.sqrt(s * s + c * c))
    r = max(min(r, 1.0), 1e-12)
    std = float(np.rad2deg(np.sqrt(-2.0 * np.log(r))))
    return mean, std


class OpticalFlowController(ImConWidgetController):
    """Linked to OpticalFlowWidget."""

    # Per-sample live update: (time_s, angle_deg). Single-sample emission
    # mirrors the FocusLock pattern so the frontend can append to a stable
    # Redux array without re-rendering the whole plot on every update.
    sigUpdateFlowAngle = Signal(object, object)
    # IDLE / RUNNING / FINISHED / ABORTED / ERROR
    sigFlowStateChanged = Signal(str)
    # {meanAngle, std, n, distanceUm, speedUmS, axis}
    sigFlowResult = Signal(object)

    # ----------------------------- init --------------------------------------

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        self._stateLock = threading.Lock()
        self._state = FlowState.IDLE

        # Toggle to dump per-iteration optical-flow debug plots from
        # ``_flowVector``. Off by default so we don't spam the filesystem.
        self.DEBUG = False

        # Worker handle and result cache for getstatus()
        self._worker: Optional[threading.Thread] = None
        self._lastResult: Optional[dict] = None
        self._lastTimes: list[float] = []
        self._lastAngles: list[float] = []

        # Pick the first available detector + positioner 
        self.cameraName = self._master.detectorsManager.getAllDeviceNames()[0]
        self.stageName = self._master.positionersManager.getAllDeviceNames()[0]
        self.camera = self._master.detectorsManager[self.cameraName]
        self.stages = self._master.positionersManager[self.stageName]

        self._moveController = _MovementController(self.stages)

    def __del__(self):
        try:
            self._setState(FlowState.IDLE)
            if self._worker is not None and self._worker.is_alive():
                self._worker.join(timeout=1.0)
        except Exception:
            pass
        if hasattr(super(), "__del__"):
            super().__del__()

    # ------------------------- state helpers ---------------------------------

    def _setState(self, state: FlowState):
        with self._stateLock:
            self._state = state
        try:
            self.sigFlowStateChanged.emit(state.value)
        except Exception:
            pass

    def _getState(self) -> FlowState:
        with self._stateLock:
            return self._state

    # ----------------------- frame grabbing ----------------------------------

    def _grabFreshFrame(
        self,
        last_frame_number: int = -1,
        timeout_s: float = 1.0,
    ) -> tuple:
        """Return (frame, frame_number) waiting for a *new* frame.

        Mirrors ``AutofocusController.grabCameraFrame`` but tuned for a
        live-flow loop: we always want the very next frame after the one we
        last processed.
        """
        t0 = time.time()
        while True:
            try:
                frame, frame_number = self.camera.getLatestFrame(returnFrameNumber=True)
            except TypeError:
                # Some detectors do not accept the kwarg
                frame = self.camera.getLatestFrame()
                frame_number = -1
            if frame is not None and frame_number != last_frame_number:
                return frame, frame_number
            if time.time() - t0 > timeout_s:
                # Best-effort fallback so the loop keeps making progress
                if frame is None:
                    try:
                        frame = self.camera.getLatestFrame()
                    except Exception:
                        frame = None
                return frame, frame_number
            time.sleep(0.005)

    # ----------------------- optical flow ------------------------------------

    @staticmethod
    def _toGray(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        img = frame
        if img.ndim == 3:
            img = np.mean(img, axis=-1)
        # Normalise to 8-bit for cv2 trackers
        if img.dtype == np.uint8:
            return img
        img = img.astype(np.float32)
        vmin, vmax = float(img.min()), float(img.max())
        if vmax - vmin < 1e-9:
            return np.zeros(img.shape, dtype=np.uint8)
        out = (img - vmin) * (255.0 / (vmax - vmin))
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _flowVector(prev_gray: np.ndarray, curr_gray: np.ndarray, DEBUG: bool = False) -> Optional[tuple]:
        """Return (dx, dy) median displacement in pixels, or None on failure."""
        if not _HAS_CV2 or prev_gray is None or curr_gray is None:
            return None
        if prev_gray.shape != curr_gray.shape:
            return None
        try:
            # Sparse LK on Shi-Tomasi corners -- fast enough for live updates.
            corners = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=8,
                blockSize=7,
            )
            if corners is None or len(corners) < 5:
                # Fall back to dense Farneback on featureless frames
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    0.5, 3, 21, 3, 5, 1.2, 0,
                )
                # Use median of dense field for robustness
                dx = float(np.median(flow[..., 0]))
                dy = float(np.median(flow[..., 1]))
                return dx, dy

            next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, corners, None,
                winSize=(21, 21), maxLevel=3,
            )
            if next_pts is None or status is None:
                return None
            mask = status.flatten() == 1
            if mask.sum() < 5:
                return None
            disp = (next_pts[mask] - corners[mask]).reshape(-1, 2)
            
            # plot and save the plot in case of debugging
            if DEBUG: 
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 6))
                plt.quiver(corners[mask][:, 0, 0], corners[mask][:, 0, 1], disp[:, 0], disp[:, 1], angles='xy', scale_units='xy', scale=1)
                plt.xlim(0, prev_gray.shape[1])
                plt.ylim(prev_gray.shape[0], 0)
                plt.title("Optical Flow Vectors")
                plt.xlabel("X (pixels)")
                plt.ylabel("Y (pixels)")
                plt.grid()
                plt.savefig("optical_flow_debug.png")
                plt.close()
            # Median is robust to mismatched tracks
            dx = float(np.median(disp[:, 0]))
            dy = float(np.median(disp[:, 1]))
            return dx, dy
        except Exception:
            return None

    # ----------------------------- API ---------------------------------------

    @APIExport(runOnUIThread=True)
    def startMeasurement(
        self,
        distance_um: float = 2000.0,
        speed_um_s: float = 100.0,
        axis: str = "X",
        warmup_frames: int = 3,
        min_displacement_px: float = 0.5,
    ) -> dict:
        """Run a calibration sweep and stream live angle samples.

        Args:
            distance_um: total relative move in micrometres along ``axis``.
            speed_um_s: stage speed (um/s). Must be > 0.
            axis: ``"X"`` or ``"Y"``.
            warmup_frames: number of frames to discard after the move starts
                so the very first samples (still showing zero displacement)
                do not pollute the aggregate.
            min_displacement_px: samples with |displacement| below this are
                dropped from the angle aggregate (low signal-to-noise).

        Returns:
            ``{"status": "started"|"error", ...}``
        """
        if not _HAS_CV2:
            return {"status": "error", "message": "OpenCV (cv2) is not available."}

        try:
            distance_um = float(distance_um)
            speed_um_s = float(speed_um_s)
            warmup_frames = int(warmup_frames)
            min_displacement_px = float(min_displacement_px)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid numeric parameter."}

        axis = (axis or "X").upper()
        if axis not in ("X", "Y"):
            return {"status": "error", "message": f"Unsupported axis '{axis}'."}
        if speed_um_s <= 0:
            return {"status": "error", "message": "speed_um_s must be > 0."}
        if abs(distance_um) < 1.0:
            return {"status": "error", "message": "distance_um is too small."}
        if warmup_frames < 0:
            warmup_frames = 0

        with self._stateLock:
            if self._state in (FlowState.STARTING, FlowState.RUNNING):
                return {
                    "status": "error",
                    "message": f"Measurement already running (state: {self._state.value})",
                }
            self._state = FlowState.STARTING

        # Reset live caches before launching worker
        self._lastResult = None
        self._lastTimes = []
        self._lastAngles = []

        self._worker = threading.Thread(
            target=self._runMeasurement,
            args=(distance_um, speed_um_s, axis, warmup_frames, min_displacement_px),
            daemon=True,
        )
        self._worker.start()

        return {
            "status": "started",
            "distance_um": distance_um,
            "speed_um_s": speed_um_s,
            "axis": axis,
        }

    @APIExport(runOnUIThread=True)
    def abortMeasurement(self) -> dict:
        """Abort the active sweep and stop the stage immediately."""
        self.__logger.info("Abort optical-flow measurement requested")
        with self._stateLock:
            if self._state not in (FlowState.STARTING, FlowState.RUNNING):
                return {"status": "idle", "state": self._state.value}
            self._state = FlowState.ABORTED
        try:
            self._moveController.abort()
        except Exception as e:
            self.__logger.error(f"Error aborting movement: {e}")
        try:
            self.sigFlowStateChanged.emit(FlowState.ABORTED.value)
        except Exception:
            pass
        return {"status": "aborted", "state": FlowState.ABORTED.value}

    @APIExport(runOnUIThread=True)
    def getstatusOpticalFlow(self) -> dict:
        """Return current state, last result and the latest angle samples."""
        return {
            "state": self._getState().value,
            "isRunning": self._getState() in (FlowState.STARTING, FlowState.RUNNING),
            "result": self._lastResult,
            "times": list(self._lastTimes),
            "angles": list(self._lastAngles),
            "camera": self.cameraName,
            "stage": self.stageName,
        }

    # ------------------------- worker thread ---------------------------------

    def _runMeasurement(
        self,
        distance_um: float,
        speed_um_s: float,
        axis: str,
        warmup_frames: int,
        min_displacement_px: float,
    ):
        try:
            # Capture starting position so we issue an absolute target rather
            # than a relative one -- absolute is robust against partial moves.
            try:
                pos = self.stages.getPosition()
                start_pos = float(pos.get(axis, 0.0)) if isinstance(pos, dict) else 0.0
            except Exception:
                start_pos = 0.0

            direction = 1.0 if distance_um >= 0 else -1.0
            target = start_pos + distance_um

            self._setState(FlowState.RUNNING)

            # Prime the optical-flow loop with a baseline frame.
            prev_frame, prev_fn = self._grabFreshFrame(last_frame_number=-1)
            prev_gray = self._toGray(prev_frame)

            # Kick off the asynchronous main move so we can poll frames.
            self._moveController.move_to_position(
                value=target, axis=axis, speed=speed_um_s, is_absolute=True,
            )

            # Skip a few frames after the move starts so the stage actually
            # has begun accelerating before we start collecting samples.
            for _ in range(int(warmup_frames)):
                if self._getState() == FlowState.ABORTED:
                    break
                if self._moveController.is_target_reached():
                    break
                prev_frame, prev_fn = self._grabFreshFrame(last_frame_number=prev_fn)
                prev_gray = self._toGray(prev_frame)

            t_start = time.time()
            times: list[float] = []
            angles_deg: list[float] = []
            disp_norms: list[float] = []

            # Safety: never run longer than (distance / speed) * 5 + 5 s
            max_runtime_s = max(5.0, (abs(distance_um) / max(speed_um_s, 1e-6)) * 5.0 + 5.0)

            while True:
                # State / completion / safety checks
                if self._getState() == FlowState.ABORTED:
                    break
                if self._moveController.is_target_reached():
                    break
                if time.time() - t_start > max_runtime_s:
                    self.__logger.warning("Optical-flow loop hit safety timeout")
                    break

                curr_frame, curr_fn = self._grabFreshFrame(last_frame_number=prev_fn)
                curr_gray = self._toGray(curr_frame)
                if curr_gray is None or prev_gray is None:
                    prev_frame, prev_fn, prev_gray = curr_frame, curr_fn, curr_gray
                    continue

                vec = self._flowVector(prev_gray, curr_gray, DEBUG=self.DEBUG)
                prev_frame, prev_fn, prev_gray = curr_frame, curr_fn, curr_gray
                if vec is None:
                    continue

                dx, dy = vec
                norm = float(np.hypot(dx, dy))

                # angle of the camera-observed motion vector. We flip the
                # sign so that a stage move along +X with no rotation yields
                # ~0 degrees (cv2 image y axis points down).
                angle = float(np.degrees(np.arctan2(-dy, dx)))

                # When the user runs along +Y, subtract 90 deg so a perfectly
                # aligned camera also yields ~0 for the Y measurement.
                if axis == "Y":
                    angle = ((angle - 90.0 + 180.0) % 360.0) - 180.0

                # Flip 180 deg if the stage moves in the negative direction
                # so the reported angle does not jump by pi.
                if direction < 0:
                    angle = ((angle + 180.0 + 180.0) % 360.0) - 180.0

                t_now = time.time() - t_start
                times.append(t_now)
                angles_deg.append(angle)
                disp_norms.append(norm)

                # Keep snapshot for getstatusOpticalFlow() callers.
                self._lastTimes = list(times)
                self._lastAngles = list(angles_deg)
                # Emit just the new sample (FocusLock-style streaming) so the
                # frontend can append to a stable Redux array. This avoids the
                # whole-plot redraw / blink that came from replacing the array
                # reference on every update.
                try:
                    self.sigUpdateFlowAngle.emit(float(t_now), float(angle))
                except Exception:
                    pass

            # Aggregate result -- only use samples with enough displacement
            arr_angles = np.asarray(angles_deg, dtype=float)
            arr_disp = np.asarray(disp_norms, dtype=float)
            if arr_angles.size and arr_disp.size == arr_angles.size:
                mask = arr_disp >= min_displacement_px
                useful = arr_angles[mask] if mask.any() else arr_angles
            else:
                useful = arr_angles

            mean_angle, std_angle = _circular_mean_std_deg(useful)

            self._lastResult = {
                "meanAngle": mean_angle,
                "std": std_angle,
                "n": int(useful.size),
                "nTotal": int(arr_angles.size),
                "distanceUm": float(distance_um),
                "speedUmS": float(speed_um_s),
                "axis": axis,
                "minDisplacementPx": float(min_displacement_px),
            }
            try:
                self.sigFlowResult.emit(dict(self._lastResult))
            except Exception:
                pass

            if self._getState() == FlowState.ABORTED:
                self.__logger.info(
                    f"Optical-flow measurement aborted ({len(angles_deg)} samples)"
                )
            else:
                self._setState(FlowState.FINISHED)
                self.__logger.info(
                    f"Optical-flow done: mean={mean_angle:.2f} deg, "
                    f"std={std_angle:.2f}, n={useful.size}"
                )

        except Exception as e:
            self.__logger.error(f"Optical-flow measurement error: {e}")
            self._setState(FlowState.ERROR)
            try:
                self._moveController.abort()
            except Exception:
                pass


# Copyright (C) 2020-2026 ImSwitch developers
