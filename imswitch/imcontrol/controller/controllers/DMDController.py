import time
import threading
import numpy as np
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

                # Ensure acquisition is running during the sequence
                handle = self._master.detectorsManager.startAcquisition()
                det_name = self._master.detectorsManager.getCurrentDetectorName()
                det = self._master.detectorsManager[det_name]

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

                # Step 1: pattern 0 -> snap
                try:
                    det.flushBuffers()
                except Exception:
                    pass
                self._http_get("/display/0")
                img0 = next_frame()
                imgs.append(img0)

                # Step 2: wait 0.2s -> pattern 1 -> snap
                time.sleep(0.2)
                try:
                    det.flushBuffers()
                except Exception:
                    pass
                self._http_get("/display/1")
                img1 = next_frame()
                imgs.append(img1)

                # Step 3: wait 0.2s -> pattern 2 -> snap
                time.sleep(0.2)
                try:
                    det.flushBuffers()
                except Exception:
                    pass
                self._http_get("/display/2")
                img2 = next_frame()
                imgs.append(img2)

                self._last_images = imgs
                if not IS_HEADLESS:
                    self._widget.setStatus("3-shot done. Ready to reconstruct.")
                # Stop acquisition
                try:
                    self._master.detectorsManager.stopAcquisition(handle)
                except Exception:
                    pass
            except Exception as e:
                self.__logger.error(f"3-shot failed: {e}")
                if not IS_HEADLESS:
                    self._widget.setStatus(f"3-shot error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    @APIExport(runOnUIThread=False)
    def reconstructIOS(self):
        """Compute Classic IOS sqrt((I1-I2)^2 + (I1-I3)^2 + (I2-I3)^2) and show on napari."""
        if len(self._last_images) != 3:
            if not IS_HEADLESS:
                self._widget.setStatus("No images captured. Run 3-shot first.")
            return {"status": "error", "message": "need 3 images"}

        try:
            I1, I2, I3 = [np.asarray(im, dtype=np.float32) for im in self._last_images]
            ios = np.sqrt((I1 - I2) ** 2 + (I1 - I3) ** 2 + (I2 - I3) ** 2)
            # Normalize to 0..1 for viewing
            m, M = float(np.min(ios)), float(np.max(ios))
            if M > m:
                ios_norm = (ios - m) / (M - m)
            else:
                ios_norm = np.zeros_like(ios)

            if not IS_HEADLESS:
                self._widget.showImage(ios_norm, name="DMD Reconstruction")
                self._widget.setStatus("Reconstruction displayed.")
            return {"status": "ok"}
        except Exception as e:
            self.__logger.error(f"Reconstruction failed: {e}")
            if not IS_HEADLESS:
                self._widget.setStatus(f"Reconstruction error: {e}")
            return {"status": "error", "message": str(e)}


# Copyright (C) 2020-2025 ImSwitch developers
# GPLv3 License
