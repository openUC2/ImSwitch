import time
import threading
import numpy as np
try:
    from skimage.filters import gaussian as _gaussian_filter
except Exception:  # pragma: no cover
    _gaussian_filter = None
import requests
from imswitch import IS_HEADLESS
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController
from imswitch.imcontrol.model import SaveMode


class DMDController(ImConWidgetController):
    """Controller for DMDWidget: send commands to Raspberry Pi FastAPI and
    orchestrate 3-shot capture with the camera.

    Expected FastAPI endpoints on the Raspberry Pi:
      - GET {host}/display/{pattern_id}
      - GET {host}/health (optional)

    Workflow for 3-shot:
      1) display 0 -> snap
      2) wait 0.2s -> display 1 -> snap
      3) wait 0.2s -> display 2 -> snap
      4) reconstruct IOS and show in napari
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        self._last_images = []  # store last 3 single-frame numpy images

        if not IS_HEADLESS:
            # Connect signals from the widget
            self._widget.sigDisplayPattern.connect(self.displayPattern)
            self._widget.sigRunThreeShot.connect(self.runThreeShot)
            self._widget.sigReconstruct.connect(self.reconstructIOS)
            self._widget.sigCheckStatus.connect(self.checkStatus)

    # --- HTTP helpers ---
    def _get_host(self) -> str:
        # Priority: value in the widget field; otherwise infer from setup ossim host/port; else default.
        if not IS_HEADLESS:
            txt = (self._widget.getHost() or '').strip()
            if txt:
                return txt

        # Try to infer from setup: many users keep a single FastAPI for both OSSIM and DMD
        try:
            cfg = getattr(self._setupInfo, "_catchAll", None)
            if isinstance(cfg, dict) and isinstance(cfg.get("ossim"), dict):
                host = cfg["ossim"].get("host")
                port = cfg["ossim"].get("port")
                if host and port:
                    return f"http://{host}:{port}"
        except Exception:
            pass

        # Fallback default
        return "http://192.168.137.2:8000"

    def _http_get(self, path: str, timeout: float = 2.0) -> dict:
        url = f"{self._get_host().rstrip('/')}{path}"
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return {"status": "ok"}
        except Exception as e:
            self.__logger.error(f"HTTP GET failed {url}: {e}")
            if not IS_HEADLESS:
                self._widget.setStatus(f"HTTP error: {e}")
            return {"status": "error", "message": str(e)}

    # --- DMD actions ---
    @APIExport(runOnUIThread=False)
    def displayPattern(self, pattern_id: int):
        self.__logger.info(f"Display DMD pattern {pattern_id}")
        res = self._http_get(f"/display/{pattern_id}")
        if not IS_HEADLESS:
            self._widget.setStatus(f"Display {pattern_id}: {res.get('status')}")
        return res

    @APIExport(runOnUIThread=False)
    def checkStatus(self):
        """Check DMD health via /health and show result in UI."""
        res = self._http_get("/health")
        if not IS_HEADLESS:
            if res.get("status") == "healthy":
                total = res.get("patterns_loaded", "?")
                self._widget.setStatus(f"DMD healthy. Patterns loaded: {total}")
            else:
                self._widget.setStatus(f"DMD status: {res}")
        return res

    # --- Camera and workflow ---
    def _snap_single(self) -> np.ndarray:
        """Acquire a single image from the current detector as numpy array via RecordingManager."""
        det_name = self._master.detectorsManager.getCurrentDetectorName()
        # Minimal HDF5 attrs from shared attributes
        attrs = {det_name: self._commChannel.sharedAttrs.getHDF5Attributes()}
        images = self._master.recordingManager.snap(
            [det_name], savename="", saveMode=SaveMode.Numpy, saveFormat=None, attrs=attrs
        )
        return images[det_name]

    @APIExport(runOnUIThread=False)
    def runThreeShot(self):
        """Run 3-shot synchronized capture with DMD 0-1-2 patterns and collect images."""
        def worker():
            try:
                if not IS_HEADLESS:
                    self._widget.setStatus("Starting 3-shot...")
                imgs = []

                # Track if live view was active to restore afterwards
                dm = self._master.detectorsManager
                was_live = dm.checkIfIsLiveView()

                # Use liveView acquisition if it was already on
                handle = dm.startAcquisition(liveView=was_live)
                det_name = self._master.detectorsManager.getCurrentDetectorName()
                det = self._master.detectorsManager[det_name]

                # Get user-defined delay (seconds) from widget (default 0.2 if fails)
                try:
                    delay_s = self._widget.getDelaySeconds()
                except Exception:
                    delay_s = 0.2

                def next_frame(timeout=2.0):
                    # Wait for the next available frame from the detector buffer
                    deadline = time.time() + timeout
                    while time.time() < deadline:
                        try:
                            frames, idx = det.getChunk()
                            if isinstance(frames, np.ndarray) and frames.size > 0:
                                # frames shape: (n, H, W) -> take last
                                return frames[-1]
                            elif isinstance(frames, (list, tuple)) and len(frames) > 0:
                                return np.asarray(frames[-1])
                        except Exception:
                            pass
                        time.sleep(0.01)
                    # Fallback
                    try:
                        return det.getLatestFrame()
                    except Exception:
                        raise RuntimeError("No frame available from detector")

                # Step 1: display 0 -> wait -> flush -> snap
                self._http_get("/display/0")
                time.sleep(delay_s)
                try:
                    det.flushBuffers()
                except Exception:
                    pass
                img0 = next_frame()
                imgs.append(img0)

                # Step 2: display 1 -> wait -> flush -> snap
                self._http_get("/display/1")
                time.sleep(delay_s)
                try:
                    det.flushBuffers()
                except Exception:
                    pass
                img1 = next_frame()
                imgs.append(img1)

                # Step 3: display 2 -> wait -> flush -> snap
                self._http_get("/display/2")
                time.sleep(delay_s)
                try:
                    det.flushBuffers()
                except Exception:
                    pass
                img2 = next_frame()
                imgs.append(img2)

                self._last_images = imgs
                if not IS_HEADLESS:
                    self._widget.setStatus("3-shot done. Ready to reconstruct.")
                # Stop acquisition only if it wasn't live before (so we don't kill user's live view)
                try:
                    if not was_live:
                        dm.stopAcquisition(handle)
                    else:
                        # If live view, we started with liveView=True so keep it running
                        pass
                except Exception:
                    pass
            except Exception as e:
                self.__logger.error(f"3-shot failed: {e}")
                if not IS_HEADLESS:
                    self._widget.setStatus(f"3-shot error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    @APIExport(runOnUIThread=False)
    def reconstructIOS(self):
        """Compute Classic IOS sqrt((I1-I2)^2 + (I1-I3)^2 + (I2-I3)^2) and show on napari.
        Adds optional constant offset subtraction (from widget) to reduce squared background noise.
        """
        if len(self._last_images) != 3:
            if not IS_HEADLESS:
                self._widget.setStatus("No images captured. Run 3-shot first.")
            return {"status": "error", "message": "need 3 images"}

        try:
            # Preserve raw frames and dtypes for bit-depth report/export
            raw1, raw2, raw3 = [np.asarray(im) for im in self._last_images]
            dtypes = [raw1.dtype, raw2.dtype, raw3.dtype]

            # Work in float32 for math
            I1, I2, I3 = [arr.astype(np.float32, copy=False) for arr in (raw1, raw2, raw3)]

            # Optional offset subtraction (units: ADU counts). If widget lacks control, offset=0.
            offset_val = 0.0
            if not IS_HEADLESS:
                try:
                    offset_val = float(getattr(self._widget, 'getOffset', lambda: 0.0)())
                except Exception:
                    offset_val = 0.0
            if offset_val != 0.0:
                I1 = np.clip(I1 - offset_val, 0, None)
                I2 = np.clip(I2 - offset_val, 0, None)
                I3 = np.clip(I3 - offset_val, 0, None)

            ios = np.sqrt((I1 - I2) ** 2 + (I1 - I3) ** 2 + (I2 - I3) ** 2)
            widefield = (I1 + I2 + I3) / 3.0

            # Optional Gaussian filtering (post-reconstruction) to suppress artifacts
            sigma = 0.0
            if not IS_HEADLESS:
                try:
                    sigma = float(self._widget.getSigma())
                except Exception:
                    sigma = 0.0
            # UI toggle: gaussian enable
            gaussian_enabled = True
            widefield_enabled = True
            export_enabled = False
            export_raw_enabled = False
            export_base = "dmd_recon"
            if not IS_HEADLESS:
                try:
                    gaussian_enabled = self._widget.isGaussianEnabled()
                except Exception:
                    gaussian_enabled = True
                try:
                    widefield_enabled = self._widget.isWidefieldEnabled()
                except Exception:
                    widefield_enabled = True
                try:
                    export_enabled = self._widget.isExportEnabled()
                except Exception:
                    export_enabled = False
                try:
                    export_raw_enabled = self._widget.isExportRawEnabled()
                except Exception:
                    export_raw_enabled = False
                try:
                    export_base = self._widget.getExportBaseName()
                except Exception:
                    export_base = "dmd_recon"

            if gaussian_enabled and sigma > 0 and _gaussian_filter is not None:
                try:
                    ios = _gaussian_filter(ios, sigma=sigma, preserve_range=True)
                except Exception:
                    pass

            # Normalize to 0..1 for viewing (both ios and widefield)
            def _norm(a: np.ndarray):
                m, M = float(np.min(a)), float(np.max(a))
                if M > m:
                    return (a - m) / (M - m)
                return np.zeros_like(a)

            ios_norm = _norm(ios)
            wf_norm = _norm(widefield)

            if not IS_HEADLESS:
                # Bit-depth heuristic from raw dtypes
                try:
                    base_dtype = max(dtypes, key=lambda dt: np.dtype(dt).itemsize)
                    if np.issubdtype(base_dtype, np.integer):
                        bits = np.dtype(base_dtype).itemsize * 8
                        bit_info = "16-bit" if bits >= 16 else "8-bit"
                    else:
                        bit_info = str(base_dtype)
                except Exception:
                    bit_info = "unknown"

                self._widget.showImage(ios_norm, name="DMD Reconstruction")
                if widefield_enabled:
                    try:
                        self._widget.showImage(wf_norm, name="DMD Widefield")
                    except Exception:
                        pass

                # Optional export
                export_msg = ""
                if export_enabled or export_raw_enabled:
                    try:
                        import tifffile, os, datetime, pathlib
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Determine base export directory: user provided overrides everything
                        export_dir = None
                        try:
                            user_dir = self._widget.getExportDirectory()
                            if user_dir:
                                export_dir = user_dir
                        except Exception:
                            pass
                        if not export_dir:
                            try:
                                recman = getattr(self._master, 'recordingManager', None)
                                if recman is not None:
                                    for attr in ('_targetDir', 'baseDirectory', 'outputDir', 'saveDir'):
                                        p = getattr(recman, attr, None)
                                        if p:
                                            export_dir = p
                                            break
                            except Exception:
                                pass
                        if not export_dir:
                            # Fallback: Documents/ImSwitchConfig/recordings or CWD
                            try:
                                from qtpy import QtCore as _QtCore
                                docs = _QtCore.QStandardPaths.writableLocation(_QtCore.QStandardPaths.DocumentsLocation)
                                if docs:
                                    export_dir = os.path.join(docs, 'ImSwitchConfig', 'recordings')
                            except Exception:
                                pass
                        if not export_dir:
                            export_dir = os.getcwd()
                        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
                        base = os.path.join(export_dir, f"{export_base}_{timestamp}")
                        ios_path = base + "_ios.tif"
                        tifffile.imwrite(ios_path, (ios_norm * 65535).astype(np.uint16))
                        if widefield_enabled:
                            wf_path = base + "_wf.tif"
                            tifffile.imwrite(wf_path, (wf_norm * 65535).astype(np.uint16))
                            export_msg = f" Exported: {os.path.basename(ios_path)}, {os.path.basename(wf_path)}"
                        else:
                            export_msg = f" Exported: {os.path.basename(ios_path)}"

                        if export_raw_enabled:
                            # Save raw (original dtype) frames
                            try:
                                i1_path = base + "_i1_raw.tif"
                                i2_path = base + "_i2_raw.tif"
                                i3_path = base + "_i3_raw.tif"
                                tifffile.imwrite(i1_path, raw1)
                                tifffile.imwrite(i2_path, raw2)
                                tifffile.imwrite(i3_path, raw3)
                                export_msg += f" + raw frames (i1,i2,i3)"
                            except Exception as eraw:
                                export_msg += f" (raw export failed: {eraw})"
                    except Exception as ee:
                        export_msg = f" Export failed: {ee}"  # keep going

                self._widget.setStatus(
                    f"Reconstruction displayed. dtype~{bit_info} offset={offset_val if offset_val != 0.0 else 'off'} "
                    f"sigma={sigma if gaussian_enabled else 'off'} wf={'on' if widefield_enabled else 'off'} "
                    f"raw={'on' if export_raw_enabled else 'off'}{export_msg}"
                )
            return {"status": "ok"}
        except Exception as e:
            self.__logger.error(f"Reconstruction failed: {e}")
            if not IS_HEADLESS:
                self._widget.setStatus(f"Reconstruction error: {e}")
            return {"status": "error", "message": str(e)}


# Copyright (C) 2020-2025 ImSwitch developers
# GPLv3 License