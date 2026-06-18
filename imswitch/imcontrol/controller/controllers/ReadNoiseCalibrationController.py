"""
ReadNoiseCalibrationController
==============================

Camera read-noise / gain calibration as a guided wizard, exposed over FastAPI.

Ports the analysis of NanoImagingPack's ``cal_readnoise`` (a photon-transfer-curve
fit) into ImSwitch. The frontend wizard (FRAME Settings → "Read Noise Calibration")
walks the user through:

1. **Setup** — name the session, pick a detector, choose frame counts.
2. **Bright frames** — tune exposure/gain against a live histogram so the bright
   frames span the detector's gray range, then capture a bright stack.
3. **Dark frames** — switch off all illumination (lasers + LED matrices) via the
   managers, confirm darkness, capture a dark stack, then restore the lights.
4. **Compute** — run ``cal_readnoise`` and report gain (e⁻/ADU), read noise (e⁻),
   offset (ADU) plus interactive charts (photon-transfer curve, linearity,
   dark/bright histograms).
5. **Notes** — store a free-text comment.

Everything is stored per session under
``<dataRoot>/recordings/readnoise_calibration/<sessionId>/``::

    bright/bright_stack.tif   dark/dark_stack.tif          # captured stacks
    figures/*.png             calibration_results.txt      # cal_readnoise output
    definitions.txt           result.json  session.json    # structured results
    comment.txt                                            # user notes

The whole ``recordings`` tree is served statically at ``/data`` by the server, so
the figures and result files are directly viewable / downloadable from the UI.

Scope is **store / display / browse only** — the computed numbers are recorded but
not pushed into live camera behaviour.
"""
import datetime
import glob
import json
import math
import os
import re
import shutil
import threading
import time
from typing import Annotated, Optional

import numpy as np
import tifffile
from fastapi import Body

import NanoImagingPack as nip

from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import APIExport, initLogger, dirtools

from ..basecontrollers import ImConWidgetController


def _convert_to_native(obj):
    """Recursively convert numpy types to plain Python so values are JSON-safe.

    Also replaces NaN / +-Inf with ``None`` so that FastAPI / json.dumps does not
    raise ``ValueError: Out of range float values are not JSON compliant``.
    """
    if isinstance(obj, np.ndarray):
        return _convert_to_native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    return obj


def _sanitize_name(name: str) -> str:
    """Make a filesystem-safe, compact slug out of a user-supplied session name."""
    if not name:
        return ""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
    slug = slug.strip("_")
    return slug[:48]


# Common detector saturation caps used to flag clipping in the live histogram.
_SATURATION_CAPS = (255, 1023, 4095, 16383, 65535)


class _AcquisitionAborted(Exception):
    """Raised inside the capture loop when the user requests a stop."""


class ReadNoiseCalibrationController(ImConWidgetController):
    """Controller exposing camera read-noise calibration as a wizard backend."""

    sigImageReceived = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # Active (in-flight) calibration session, or None.
        self._session: Optional[dict] = None
        # Snapshot of which illumination sources were on before we forced them off.
        self._illuminationBackup: dict = {}
        # Pollable progress for the background acquisition loop.
        self._progress: dict = {
            "running": False,
            "phase": "",
            "step": 0,
            "total": 0,
            "message": "",
        }
        self._abort = threading.Event()
        self._acqThread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Path helpers                                                        #
    # ------------------------------------------------------------------ #

    def _data_root(self) -> str:
        return dirtools.UserFileDirs.getValidatedDataPath()

    def _calibration_root(self) -> str:
        root = os.path.join(self._data_root(), "recordings", "readnoise_calibration")
        os.makedirs(root, exist_ok=True)
        return root

    def _session_dir(self, session_id: str) -> str:
        return os.path.join(self._calibration_root(), session_id)

    def _data_rel(self, path: str) -> str:
        """Path relative to the ``/data`` static-mount root, with forward slashes."""
        try:
            return os.path.relpath(path, self._data_root()).replace(os.sep, "/")
        except Exception:
            return path

    @staticmethod
    def _read_json(path: str):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    # Detector helpers                                                    #
    # ------------------------------------------------------------------ #

    def _resolve_detector(self, name: Optional[str] = None):
        dm = self._master.detectorsManager
        if name:
            try:
                return dm[name]
            except Exception:
                self._logger.debug(f"Detector '{name}' not found, using current detector")
        return dm.getCurrentDetector()

    def _detector_name(self, detector, fallback: Optional[str] = None) -> Optional[str]:
        name = getattr(detector, "name", None)
        if name:
            return name
        if fallback:
            return fallback
        names = list(self._master.detectorsManager.getAllDeviceNames())
        return names[0] if names else None

    def _ensure_acquiring(self, detector):
        try:
            detector.startAcquisition()
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.debug(f"startAcquisition failed (continuing): {exc}")

    def _read_settings(self, detector) -> dict:
        out = {}
        for key in ("exposure", "gain", "blacklevel"):
            val = None
            try:
                val = detector.getParameter(key)
            except Exception:
                try:
                    val = detector.parameters[key].value
                except Exception:
                    val = None
            if val is not None:
                out[key] = val
        return out

    def _grab_one(self, detector):
        """Grab a single frame as a 2D float-able array (RGB averaged to gray).

        Returns ``None`` if the detector has no frame available (e.g. not
        streaming), so callers can degrade gracefully instead of raising.
        """
        frame = None
        try:
            frame, _ = detector.getLatestFrame(returnFrameNumber=True)
        except TypeError:
            frame = detector.getLatestFrame()
        if frame is None:
            frame = detector.getLatestFrame()
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim > 2:
            arr = arr.mean(axis=2)
        return arr

    def _grab_stack(self, detector, count: int, phase: str) -> np.ndarray:
        """Capture ``count`` frame-synced frames, abortable, updating progress."""
        frames = []
        last_fn = None
        supports_fn = True
        per_frame_timeout = 5.0
        while len(frames) < count:
            if self._abort.is_set():
                raise _AcquisitionAborted()
            deadline = time.time() + per_frame_timeout
            new_frame = None
            while True:
                try:
                    if supports_fn:
                        got, fn = detector.getLatestFrame(returnFrameNumber=True)
                    else:
                        got, fn = detector.getLatestFrame(), None
                except TypeError:
                    supports_fn = False
                    got, fn = detector.getLatestFrame(), None
                if got is not None and (fn is None or fn != last_fn):
                    last_fn = fn
                    new_frame = got
                    break
                if time.time() > deadline:
                    new_frame = got  # accept a duplicate rather than hang forever
                    break
                time.sleep(0.005)
            if new_frame is None:
                raise RuntimeError(
                    "Detector returned no frames - is the camera streaming/acquiring?"
                )
            arr = np.asarray(new_frame)
            if arr.ndim > 2:
                arr = arr.mean(axis=2)
            frames.append(arr)
            self._progress.update(
                {"step": len(frames), "total": count,
                 "message": f"Captured {len(frames)}/{count} {phase} frames"}
            )
        return np.array(frames)

    # ------------------------------------------------------------------ #
    # Session helpers                                                     #
    # ------------------------------------------------------------------ #

    def _write_session(self):
        if not self._session:
            return
        self._write_session_to(self._session["folder"], self._session)

    def _write_session_to(self, folder: str, sess: dict):
        payload = {k: v for k, v in sess.items() if k != "folder"}
        try:
            with open(os.path.join(folder, "session.json"), "w") as f:
                json.dump(_convert_to_native(payload), f, indent=2)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error(f"Could not write session.json: {exc}")

    def _public_session(self) -> Optional[dict]:
        if not self._session:
            return None
        s = {k: v for k, v in self._session.items() if k != "folder"}
        s["dataRelPath"] = self._data_rel(self._session["folder"])
        return s

    def _load_session_folder(self, session_id: Optional[str]):
        """Return ``(folder, sessionDict)`` for the given id or the active session."""
        if session_id:
            folder = self._session_dir(session_id)
            sess = self._read_json(os.path.join(folder, "session.json")) or {"sessionId": session_id}
            sess["folder"] = folder
            return folder, sess
        if self._session:
            return self._session["folder"], self._session
        raise RuntimeError("No active session and no sessionId provided")

    # ================================================================== #
    # API: status & detector settings                                    #
    # ================================================================== #

    @APIExport()
    def getStatus(self) -> dict:
        """Detectors, illumination sources, active session and progress."""
        dm = self._master.detectorsManager
        names = list(dm.getAllDeviceNames())
        try:
            current = self._detector_name(dm.getCurrentDetector(), names[0] if names else None)
        except Exception:
            current = names[0] if names else None

        illum = []
        try:
            for n in self._master.lasersManager.getAllDeviceNames():
                laser = self._master.lasersManager[n]
                illum.append({"type": "laser", "name": n,
                              "enabled": bool(getattr(laser, "enabled", False))})
        except Exception as exc:
            self._logger.debug(f"Could not enumerate lasers: {exc}")
        try:
            if hasattr(self._master, "LEDMatrixsManager"):
                for n in self._master.LEDMatrixsManager.getAllDeviceNames():
                    led = self._master.LEDMatrixsManager[n]
                    enabled = bool(getattr(led, "enabled", bool(getattr(led, "intensity", 0))))
                    illum.append({"type": "led", "name": n, "enabled": enabled})
        except Exception as exc:
            self._logger.debug(f"Could not enumerate LED matrices: {exc}")

        return _convert_to_native({
            "detectors": names,
            "currentDetector": current,
            "illumination": illum,
            "session": self._public_session(),
            "progress": self._progress,
            "calibrationRelPath": self._data_rel(self._calibration_root()),
        })

    @APIExport()
    def getProgress(self) -> dict:
        """Poll the background acquisition progress."""
        return _convert_to_native(self._progress)

    @APIExport()
    def getDetectorSettings(self, detectorName: str = None) -> dict:
        """Current exposure / gain / blacklevel for the chosen (or current) detector."""
        detector = self._resolve_detector(detectorName)
        return _convert_to_native({
            "detector": self._detector_name(detector, detectorName),
            "settings": self._read_settings(detector),
        })

    @APIExport(requestType="POST")
    def setDetectorSetting(self, detectorName: str = None, name: str = None, value: float = None) -> dict:
        """Set a single numeric detector parameter (e.g. ``exposure`` or ``gain``)."""
        if name is None:
            return {"status": "error", "message": "parameter 'name' is required"}
        detector = self._resolve_detector(detectorName)
        v = value
        try:
            v = float(value)
        except (TypeError, ValueError):
            pass
        try:
            detector.setParameter(name, v)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        return _convert_to_native({"status": "ok", "settings": self._read_settings(detector)})

    @APIExport()
    def getHistogram(self, detectorName: str = None, numBins: int = 128) -> dict:
        """Grab one frame and return its histogram plus saturation diagnostics."""
        detector = self._resolve_detector(detectorName)
        self._ensure_acquiring(detector)
        frame = self._grab_one(detector)
        if frame is None:
            return {"status": "error", "message": "no frame available (is the camera streaming?)"}
        a = np.asarray(frame, dtype=float)
        if a.size == 0 or a.ndim < 2:
            return {"status": "error", "message": "empty frame from detector"}
        h, w = int(a.shape[-2]), int(a.shape[-1])
        amin, amax = float(a.min()), float(a.max())
        nb = int(numBins) if numBins else 128
        counts, edges = np.histogram(a, bins=nb)
        sat_cap = next((c for c in _SATURATION_CAPS if int(round(amax)) == c), None)
        clip_frac = float(np.mean(a >= amax)) if amax > 0 else 0.0
        sat_frac = float(np.mean(a >= sat_cap)) if sat_cap else 0.0
        return _convert_to_native({
            "status": "ok",
            "counts": counts,
            "binEdges": edges,
            "min": amin, "max": amax,
            "mean": float(a.mean()), "std": float(a.std()),
            "width": w, "height": h,
            "saturationCap": sat_cap,
            "saturationFraction": sat_frac,
            "clipFraction": clip_frac,
        })

    # ================================================================== #
    # API: illumination control                                          #
    # ================================================================== #

    @APIExport(requestType="POST")
    def setIllumination(self, state: str = "off") -> dict:
        """Switch all illumination ``off`` (recording prior state) or ``restore`` it."""
        if state == "off":
            backup = {}
            try:
                for n in self._master.lasersManager.getAllDeviceNames():
                    laser = self._master.lasersManager[n]
                    backup[f"laser:{n}"] = bool(getattr(laser, "enabled", False))
                    try:
                        laser.setEnabled(False)
                    except Exception as exc:
                        self._logger.debug(f"Could not disable laser {n}: {exc}")
            except Exception as exc:
                self._logger.debug(f"Laser enumeration failed: {exc}")
            try:
                if hasattr(self._master, "LEDMatrixsManager"):
                    for n in self._master.LEDMatrixsManager.getAllDeviceNames():
                        led = self._master.LEDMatrixsManager[n]
                        backup[f"led:{n}"] = bool(getattr(led, "enabled", bool(getattr(led, "intensity", 0))))
                        try:
                            led.setAll((0, 0, 0))
                        except Exception:
                            pass
                        try:
                            led.setEnabled(False)
                        except Exception:
                            pass
            except Exception as exc:
                self._logger.debug(f"LED matrix enumeration failed: {exc}")
            self._illuminationBackup = backup
            return _convert_to_native({"status": "ok", "state": "off", "backup": backup})

        if state == "restore":
            restored = []
            for key, was_on in (self._illuminationBackup or {}).items():
                if not was_on:
                    continue
                typ, _, n = key.partition(":")
                try:
                    if typ == "laser":
                        self._master.lasersManager[n].setEnabled(True)
                        restored.append(n)
                    elif typ == "led":
                        self._master.LEDMatrixsManager[n].setEnabled(True)
                        restored.append(n)
                except Exception as exc:
                    self._logger.debug(f"Could not restore {key}: {exc}")
            self._illuminationBackup = {}
            return _convert_to_native({"status": "ok", "state": "restore", "restored": restored})

        return {"status": "error", "message": "state must be 'off' or 'restore'"}

    # ================================================================== #
    # API: session lifecycle & acquisition                               #
    # ================================================================== #

    @APIExport(requestType="POST")
    def startSession(self, name: str = "", detectorName: str = None,
                     nBright: int = 20, nDark: int = 20, numBins: int = 100) -> dict:
        """Create a new calibration session folder and make it the active session."""
        detector = self._resolve_detector(detectorName)
        det_name = self._detector_name(detector, detectorName)
        ts = datetime.datetime.now()
        session_id = ts.strftime("%Y%m%d_%H%M%S")
        slug = _sanitize_name(name)
        if slug:
            session_id = f"{session_id}_{slug}"
        folder = self._session_dir(session_id)
        os.makedirs(folder, exist_ok=True)
        settings = self._read_settings(detector)
        self._session = {
            "sessionId": session_id,
            "name": name,
            "detector": det_name,
            "nBright": int(nBright),
            "nDark": int(nDark),
            "numBins": int(numBins),
            "exposure": settings.get("exposure"),
            "gain": settings.get("gain"),
            "created": ts.isoformat(timespec="seconds"),
            "status": "created",
            "brightCount": 0,
            "darkCount": 0,
            "folder": folder,
        }
        self._write_session()
        return _convert_to_native({"status": "ok", "session": self._public_session()})

    @APIExport(requestType="POST")
    def acquireFrames(self, kind: str, count: int = None) -> dict:
        """Capture a ``bright`` or ``dark`` stack in a background thread (poll progress)."""
        if self._progress.get("running"):
            return {"status": "error", "message": "An acquisition is already running"}
        if self._session is None:
            return {"status": "error", "message": "No active session - call startSession first"}
        if kind not in ("bright", "dark"):
            return {"status": "error", "message": "kind must be 'bright' or 'dark'"}
        default = self._session.get("nBright" if kind == "bright" else "nDark", 20)
        n = int(count) if count else int(default)
        self._abort.clear()
        self._progress = {"running": True, "phase": kind, "step": 0, "total": n,
                          "message": f"Starting {kind} capture…"}
        self._acqThread = threading.Thread(target=self._acquire_worker, args=(kind, n), daemon=True)
        self._acqThread.start()
        return {"status": "started", "kind": kind, "count": n}

    def _acquire_worker(self, kind: str, count: int):
        try:
            detector = self._resolve_detector(self._session["detector"])
            self._ensure_acquiring(detector)
            stack = self._grab_stack(detector, count, kind)
            subdir = os.path.join(self._session["folder"], kind)
            os.makedirs(subdir, exist_ok=True)
            path = os.path.join(subdir, f"{kind}_stack.tif")
            save_stack = stack.astype(np.float32) if np.issubdtype(stack.dtype, np.floating) else stack
            tifffile.imwrite(path, save_stack)
            self._session[f"{kind}Count"] = int(stack.shape[0])
            self._session["status"] = f"{kind}_done"
            self._write_session()
            self._progress.update({"running": False, "done": True,
                                   "message": f"Saved {stack.shape[0]} {kind} frames"})
        except _AcquisitionAborted:
            self._progress.update({"running": False, "aborted": True,
                                   "message": f"{kind.capitalize()} capture aborted"})
        except Exception as exc:
            self._logger.error(f"{kind} acquisition failed: {exc}")
            self._progress.update({"running": False, "error": str(exc),
                                   "message": f"Error: {exc}"})

    @APIExport(requestType="POST")
    def stopAcquisition(self) -> dict:
        """Request the running capture loop to abort."""
        self._abort.set()
        return {"status": "ok"}

    # ================================================================== #
    # API: compute                                                       #
    # ================================================================== #

    @APIExport(requestType="POST")
    def computeCalibration(self, sessionId: str = None, numBins: int = None,
                           validRangeLow: float = None, validRangeHigh: float = None,
                           saturationImage: bool = False) -> dict:
        """Run cal_readnoise on the session's bright+dark stacks and store the result."""
        try:
            folder, sess = self._load_session_folder(sessionId)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        bright = os.path.join(folder, "bright", "bright_stack.tif")
        dark = os.path.join(folder, "dark", "dark_stack.tif")
        if not os.path.exists(bright) or not os.path.exists(dark):
            return {"status": "error",
                    "message": "Both a bright and a dark stack are required before computing"}
        try:
            return self._run_cal_readnoise(folder, sess, bright, dark, numBins,
                                           validRangeLow, validRangeHigh, saturationImage)
        except AssertionError as exc:
            return {"status": "error", "message": f"Analysis failed: {exc}"}
        except Exception as exc:
            self._logger.error(f"computeCalibration failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def _run_cal_readnoise(self, folder, sess, bright, dark, numBins,
                           validRangeLow, validRangeHigh, saturationImage):
        # Headless: force a non-interactive backend before cal_readnoise plots.
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fg = tifffile.imread(bright)
        bg = tifffile.imread(dark)
        nb = int(numBins) if numBins else int(sess.get("numBins", 100))
        valid_range = None
        if validRangeLow is not None and validRangeHigh is not None:
            valid_range = (float(validRangeLow), float(validRangeHigh))
        fig_dir = os.path.join(folder, "figures")

        (bg_total_mean, gain, readnoise, mean_el, _validmap, figures, doc_dict) = nip.cal_readnoise(
            fg, bg, numBins=nb, validRange=valid_range, doPlot=True, exportpath=fig_dir,
            exportFormat="png", brightness_blurring=True, plotHist=False,
            check_bg=False, saturationImage=bool(saturationImage))

        charts = self._harvest_charts(figures, fg, bg)
        try:
            plt.close("all")
        except Exception:
            pass

        fig_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(fig_dir, "*.png")))
        metrics = {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in doc_dict.items()}
        definitions = {k: (v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else "")
                       for k, v in doc_dict.items()}

        result = {
            "computedAt": datetime.datetime.now().isoformat(timespec="seconds"),
            "offset": bg_total_mean,
            "gain": gain,
            "readnoise": readnoise,
            "meanElectronsPerExposure": mean_el,
            "numBins": nb,
            "validRange": valid_range,
            "saturationImage": bool(saturationImage),
            "metrics": metrics,
            "definitions": definitions,
            "charts": charts,
            "figures": fig_files,
        }
        result = _convert_to_native(result)
        try:
            with open(os.path.join(folder, "result.json"), "w") as f:
                json.dump(result, f, indent=2)
        except Exception as exc:
            self._logger.error(f"Could not write result.json: {exc}")

        sess["status"] = "computed"
        sess["gain"] = float(gain)
        sess["readnoise"] = float(readnoise)
        sess["offset"] = float(bg_total_mean)
        self._write_session_to(folder, sess)

        return _convert_to_native({
            "status": "ok",
            "sessionId": sess.get("sessionId"),
            "dataRelPath": self._data_rel(folder),
            "result": result,
        })

    def _harvest_charts(self, figures, fg, bg) -> dict:
        """Pull chart-ready arrays out of cal_readnoise's matplotlib figures.

        Reading the plotted ``Line2D`` data (rather than re-implementing the
        ``RWLSPoisson`` fit) keeps the interactive charts numerically identical to
        cal_readnoise. The dark/bright histograms are recomputed directly from the
        raw stacks (trivial and robust).
        """
        fig_map = {}
        for item in figures or []:
            try:
                fig, name = item
                fig_map[name] = fig
            except Exception:
                continue

        ptc = {}
        fig = fig_map.get("Photon_Calibration")
        if fig is not None and fig.axes:
            for line in fig.axes[0].get_lines():
                label = line.get_label()
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                if label == "Brightness bins":
                    ptc["meanADU"], ptc["varADU"] = x, y
                elif label == "Gain fit":
                    ptc["fitX"], ptc["fitY"] = x, y
                elif label == "Error":
                    ptc["errUpperX"], ptc["errUpper"] = x, y
            if "fitY" in ptc and "errUpper" in ptc and len(ptc["fitY"]) == len(ptc["errUpper"]):
                ptc["errLower"] = (2 * np.asarray(ptc["fitY"]) - np.asarray(ptc["errUpper"]))

        linearity = {}
        fig = fig_map.get("linearity_error")
        if fig is not None and fig.axes and fig.axes[0].get_lines():
            line = fig.axes[0].get_lines()[0]
            linearity["x"] = np.asarray(line.get_xdata(), dtype=float)
            linearity["devPercent"] = np.asarray(line.get_ydata(), dtype=float)

        return _convert_to_native({
            "ptc": ptc,
            "linearity": linearity,
            "histograms": self._compute_histograms(fg, bg),
        })

    @staticmethod
    def _compute_histograms(fg, bg, max_bins: int = 512) -> dict:
        def _hist(a):
            a = np.asarray(a)
            n_frames = a.shape[0] if a.ndim >= 3 else 1
            amax = float(a.max())
            nbins = int(min(max_bins, max(16, amax)))
            counts, edges = np.histogram(a, bins=nbins, range=(0.0, max(amax, 1.0)))
            counts = counts / max(n_frames, 1)
            return edges, counts

        dark_edges, dark_counts = _hist(bg)
        bright_edges, bright_counts = _hist(fg)
        return {
            "darkEdges": dark_edges,
            "darkCounts": dark_counts,
            "brightEdges": bright_edges,
            "brightCounts": bright_counts,
        }

    # ================================================================== #
    # API: browse previous sessions                                      #
    # ================================================================== #

    @APIExport()
    def listSessions(self) -> dict:
        """Summaries of all stored calibration sessions (most recent first)."""
        root = self._calibration_root()
        rows = []
        for session_id in sorted(os.listdir(root), reverse=True):
            d = os.path.join(root, session_id)
            if not os.path.isdir(d):
                continue
            sess = self._read_json(os.path.join(d, "session.json")) or {"sessionId": session_id}
            res = self._read_json(os.path.join(d, "result.json"))
            comment = self._read_text(os.path.join(d, "comment.txt"))
            rows.append({
                "sessionId": sess.get("sessionId", session_id),
                "name": sess.get("name", ""),
                "detector": sess.get("detector", ""),
                "created": sess.get("created", ""),
                "status": sess.get("status", ""),
                "brightCount": sess.get("brightCount", 0),
                "darkCount": sess.get("darkCount", 0),
                "gain": (res or {}).get("gain", sess.get("gain")),
                "readnoise": (res or {}).get("readnoise", sess.get("readnoise")),
                "offset": (res or {}).get("offset", sess.get("offset")),
                "hasResult": res is not None,
                "comment": (comment or "").strip()[:200],
                "dataRelPath": self._data_rel(d),
            })
        return _convert_to_native({"sessions": rows})

    @APIExport()
    def getSession(self, sessionId: str) -> dict:
        """Full session + result + comment for one stored session (for browsing)."""
        d = self._session_dir(sessionId)
        if not os.path.isdir(d):
            return {"status": "error", "message": "Session not found"}
        return _convert_to_native({
            "status": "ok",
            "session": self._read_json(os.path.join(d, "session.json")) or {"sessionId": sessionId},
            "result": self._read_json(os.path.join(d, "result.json")),
            "comment": self._read_text(os.path.join(d, "comment.txt")),
            "dataRelPath": self._data_rel(d),
        })

    @APIExport(requestType="POST")
    def saveComment(self, sessionId: str,
                    comment: Annotated[str, Body(embed=True)] = "") -> dict:
        """Write/replace the free-text comment for a session (comment.txt)."""
        d = self._session_dir(sessionId)
        if not os.path.isdir(d):
            return {"status": "error", "message": "Session not found"}
        try:
            with open(os.path.join(d, "comment.txt"), "w") as f:
                f.write(comment or "")
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok"}

    @APIExport(requestType="POST")
    def deleteSession(self, sessionId: str) -> dict:
        """Delete a stored session folder (guarded to the calibration directory)."""
        root = os.path.realpath(self._calibration_root())
        d = os.path.realpath(self._session_dir(sessionId))
        if os.path.dirname(d) != root or not os.path.isdir(d):
            return {"status": "error", "message": "Invalid session"}
        try:
            shutil.rmtree(d)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        if self._session and self._session.get("sessionId") == sessionId:
            self._session = None
        return {"status": "ok"}
