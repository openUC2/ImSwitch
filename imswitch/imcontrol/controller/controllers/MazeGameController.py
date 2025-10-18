# imswitch/controllers/MazeGameController.py
import time
import base64
import threading
import queue
import numpy as np
import cv2
from fastapi import FastAPI, Response
from PIL import Image
import io

from imswitch import IS_HEADLESS
from imswitch.imcommon.model import initLogger, APIExport
from imswitch.imcommon.framework import Signal
from ..basecontrollers import ImConWidgetController

class MazeGameController(ImConWidgetController):
    """
    Vision-only maze 'wall-crossing' game:
      - Central crop (CxC px) is analysed per frame.
      - Maintain a 5-frame rolling mean in the crop.
      - Detect LOW→HIGH jump -> increment counter.
      - Expose REST + Signals; provide downscaled latest frame and processed preview.
    """

    # Signals for the React UI via sockets
    sigGameState = Signal(object)          # {"running": bool, "counter": int, "elapsed_s": float}
    sigCounterUpdated = Signal(int)        # live counter
    sigPreviewUpdated = Signal(object)     # {"jpeg_b64": "...", "ts": float}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        # Devices
        self.cameraName = self._master.detectorsManager.getAllDeviceNames()[0]
        self.camera = self._master.detectorsManager[self.cameraName]
        self.positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self.positioner = self._master.positionersManager[self.positionerName]

        # Game state
        self._running = False
        self._t0 = None
        self._counter = 0

        # Processing params (API settable)
        self._crop = 30                        # central crop size (pixels)
        self._edge = 4                         # margin ignored at crop edges (px)
        self._jump_low = 0.05                  # "low" normalized intensity threshold
        self._jump_high = 0.15                 # "high" normalized intensity threshold
        self._m_smooth = 0.0                  # smoothed mean intensity of trace
        self._history = 3                      # rolling window size
        self._min_hold_low = 2                 # require ≥2 low frames before a jump
        self._downscale = 2                    # integer downscale for live frame
        self._poll_dt = 0.03                   # seconds between grabs
        self._jpeg_quality = 80
        self._start_pos_x = 0               # default start position X
        self._start_pos_y = 0               # default start position Y

        # Worker infra
        self._q = queue.Queue(maxsize=4)
        self._worker = None
        self._grabber = None
        self._lock = threading.Lock()

    # ------------------------- API (REST) -------------------------
    @APIExport()
    def moveToStartPosition(self, x: int = None, y: int = None):
        """Move stage to specified (x,y) position."""
        if x is None:
            x = self._start_pos_x
        if y is None:
            y = self._start_pos_y
        try:
            self.positioner.move(value=x, speed=20000, axis="X", is_absolute=True, is_blocking=False)
            self.positioner.move(value=y, speed=20000, axis="Y", is_absolute=True, is_blocking=False)
            return {"ok": True, "x": x, "y": y}
        except Exception as e:
            self.__logger.error(f"MazeGameController: could not move to position ({x}, {y}): {e}")
            return {"ok": False, "error": str(e)}


    @APIExport()
    def startGame(self, startX: int = None, startY: int = None):
        """Reset counter + timer and start background grab/process."""
        if startX is not None and startX != 0:
            # move to initial position if requested
            try:
                self.positioner.move(value=startX, speed=20000, axis="X", is_absolute=True, is_blocking=True)
            except Exception as e:
                self.__logger.error(f"MazeGameController: could not move to start position ({startX}, {startY}): {e}")  
        if startY is not None and startY != 0:
            try:
                self.positioner.move(value=startY, speed=20000, axis="Y", is_absolute=True, is_blocking=True)
            except Exception as e:
                self.__logger.error(f"MazeGameController: could not move to start position ({startX}, {startY}): {e}")
             
        with self._lock:
            if self._running:
                return {"ok": True, "already_running": True}
            self._counter = 0
            self._t0 = time.time()
            self._running = True
            self._q.queue.clear()
            self._worker = threading.Thread(target=self._process_loop, daemon=True)
            self._grabber = threading.Thread(target=self._grab_loop, daemon=True)
            self._worker.start()
            self._grabber.start()
        self._emit_state()
        return {"ok": True}

    @APIExport()
    def stopGame(self):
        """Stop background threads; keep last state."""
        with self._lock:
            self._running = False
        self._emit_state()
        return {"ok": True}

    @APIExport()
    def resetGame(self):
        """Stop + reset counter and timer."""
        self.stopGame()
        with self._lock:
            self._counter = 0
            self._t0 = None
        self._emit_state()
        return {"ok": True}

    @APIExport()
    def getState(self):
        return self._state_dict()

    @APIExport()
    def getCounter(self) -> int:
        return self._counter

    @APIExport()
    def getElapsedSeconds(self) -> float:
        return (time.time() - self._t0) if (self._t0 is not None) else 0.0

    @APIExport()
    def setCropSize(self, size: int = 30):
        self._crop = int(max(8, size))
        return {"ok": True, "crop": self._crop}

    @APIExport()
    def setJumpThresholds(self, low: float = 0.20, high: float = 0.55):
        """Normalized [0..1]; require low < high."""
        self._jump_low = float(low)
        self._jump_high = float(high)
        return {"ok": True, "low": self._jump_low, "high": self._jump_high}

    @APIExport()
    def setHistory(self, n: int = 5, min_hold_low: int = 2):
        self._history = max(3, int(n))
        self._min_hold_low = max(1, int(min_hold_low))
        return {"ok": True, "history": self._history, "min_hold_low": self._min_hold_low}

    @APIExport()
    def setDownscale(self, factor: int = 2):
        self._downscale = max(1, int(factor))
        return {"ok": True, "downscale": self._downscale}

    @APIExport()
    def setPollInterval(self, seconds: float = 0.03):
        self._poll_dt = max(0.005, float(seconds))
        return {"ok": True, "poll_dt": self._poll_dt}

    @APIExport(runOnUIThread=False)
    def getLatestProcessedPreview(self) -> Response:
        """Return latest processed preview with overlay as PNG image."""
        try:
            was_high = False
            self._last_preview = draw_overlay(gray, box, color=(0, 255, 0) if was_high else (255, 0, 0))
            if hasattr(self, '_last_preview') and self._last_preview is not None:
                im = Image.fromarray(self._last_preview)
            else:
                # fallback
                frame = self.camera.getLatestFrame()  # get fresh frame for preview
                gray = to_gray_float(frame)
                crop, box = center_crop(gray, self._crop)
                gray = draw_overlay(gray, box, color=(0, 255, 0))
                im = Image.fromarray(to_uint8(gray))

            # save image to an in-memory bytes buffer
            with io.BytesIO() as buf:
                im = im.convert("L")  # convert image to 'L' mode
                im.save(buf, format="PNG")
                im_bytes = buf.getvalue()

            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            return Response(im_bytes, headers=headers, media_type="image/png")
        except Exception as e:
            self.__logger.error(f"Error generating preview image: {e}")
            return Response(status_code=500, content="Error generating preview image")


    # ------------------------- Loops -------------------------

    def _grab_loop(self):
        while self._running:
            try:
                print("polling frame...")
                frame = self.camera.getLatestFrame()  # np.ndarray (H,W) or (H,W,C)
                if frame is None:
                    time.sleep(self._poll_dt)
                    continue

                # push to processor
                if not self._q.full():
                    self._q.put(frame, block=False)
                time.sleep(self._poll_dt)
            except Exception as e:
                self.__logger.error(f"Maze grab loop error: {e}")
                time.sleep(0.05)

    def _process_loop(self):
        roll = []           # rolling mean history
        low_run = 0         # count consecutive "low" frames for debounce
        was_high = False    # track last state for LOW→HIGH edge detection

        while self._running:
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            gray = to_gray_float(frame)
            crop, box = center_crop(gray, self._crop)
            # ignore edges for robustness
            if self._edge > 0:
                crop_eval = crop[self._edge:-self._edge, self._edge:-self._edge]
            else:
                crop_eval = crop

            m = float(np.mean(crop_eval))
            roll.append(m)
            if len(roll) > self._history:
                roll.pop(0)
            self._m_smooth = float(np.mean(roll))

            # debounced LOW→HIGH
            print(f"  m={m:.3f}  self._m_smooth={self._m_smooth:.3f}  low_run={low_run}  was_high={was_high}")
            if self._m_smooth < self._jump_low:
                low_run += 1
                was_high = False
            elif (low_run >= self._min_hold_low) and (self._m_smooth > self._jump_high) and (not was_high):
                self._increment_counter()
                low_run = 0
                was_high = True
            else:
                was_high = self._m_smooth > self._jump_high

            # preview: draw crop box + current state color
            # self._last_preview = draw_overlay(gray, box, color=(0, 255, 0) if was_high else (255, 0, 0))
            # encode to jpeg (base64) and send via socket as payload
            # use helper that encodes via OpenCV + base64 for smaller payloads
            #img_uint8 = to_uint8(self._last_preview)
            #jpeg_b64 = encode_jpeg_b64(img_uint8, quality=self._jpeg_quality)
            #payload = {"jpeg_b64": jpeg_b64, "ts": time.time()}
            #self.sigPreviewUpdated.emit(payload)
            self.sigGameState.emit(self._state_dict())
    # ------------------------- Helpers -------------------------

    def _increment_counter(self):
        with self._lock:
            self._counter += 1
            c = self._counter
        print(f"MazeGameController: couxnter = {c}")
        self.sigCounterUpdated.emit(c)
        self._emit_state()

    def _state_dict(self):
        return {
            "running": self._running,
            "counter": int(self._counter),
            "elapsed_s": (time.time() - self._t0) if (self._t0 is not None) else 0.0,
            "crop": int(self._crop),
            "thresholds": {"low": self._jump_low, "high": self._jump_high},
            "smooth_mean": float(self._m_smooth),
            "history": int(self._history),
        }

    def _emit_state(self):
        self.sigGameState.emit(self._state_dict())

# ------------------------- image utils -------------------------

def to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[..., 0]
    img = img.astype(np.float32)
    return img

def to_uint8(imgf: np.ndarray) -> np.ndarray:
    # normalize the image to [0..255]
    out = np.uint8(255. * (imgf/np.max(imgf)))
    return out

def center_crop(img: np.ndarray, size: int):
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    s = max(8, int(size))
    y0 = max(0, cy - s // 2)
    x0 = max(0, cx - s // 2)
    y1 = min(H, y0 + s)
    x1 = min(W, x0 + s)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)

def draw_overlay(gray: np.ndarray, box, color=(0, 255, 0)):
    """Return RGB image with crop rectangle."""
    x0, y0, x1, y1 = box
    rgb = cv2.cvtColor(to_uint8(gray), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(rgb, (x0, y0), (x1, y1), color, 2)
    return rgb

def downscale_frame(frame: np.ndarray, factor: int = 2):
    if factor <= 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w // factor, h // factor), interpolation=cv2.INTER_AREA)

def encode_jpeg_b64(img_uint8: np.ndarray, quality=70) -> str:
    ok, buf = cv2.imencode(".jpg", img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
