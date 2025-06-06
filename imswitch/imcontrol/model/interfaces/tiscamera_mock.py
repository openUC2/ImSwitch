import threading
import time
import collections
from enum import Enum
from pathlib import Path

import numpy as np
import tifffile as tif
from scipy.stats import multivariate_normal

# -----------------------------------------------------------------------------
#  Simple trigger emulation helpers
# -----------------------------------------------------------------------------
class TriggerSource(str, Enum):
    CONTINUOUS        = "Continuous"        # free‑run
    SOFTWARE          = "Internal trigger"  # SW trigger in SDK
    EXTERNAL          = "External trigger"  # emulated hardware pulse


class MockCameraTIS:
    """Drop‑in replacement for *CameraHIK* when no hardware is present.

    It mimics trigger modes, keeps a ring‑buffer and delivers deterministic
    synthetic images (noise, peaks, STORM stack, off‑axis hologram,…)
    so the entire acquisition pipeline—including callback‑based buffering—
    can be tested headless.
    """

    # ---------------------------------------------------------------------
    #  Construction / basic attributes
    # ---------------------------------------------------------------------

    def __init__(self, mocktype: str = "normal", mockstackpath: str | None = None,
                 isRGB: bool = False, width: int = 512, height: int = 512,
                 nbuffer: int = 256):

        # public metadata (CameraHIK exposes the same)
        self.model          = "mock"
        self.SensorWidth    = width
        self.SensorHeight   = height
        self.pixelSize      = 1.0  # [µm] defaults; configurable outside

        # acquisition parameters
        self.exposure_ms    = 50.0
        self.gain           = 1.0
        self.trigger_source = TriggerSource.CONTINUOUS

        # internal state
        self._frame_counter = 0
        self._is_running    = False
        self.isRGB         = isRGB
        self._mocktype      = mocktype
        self._stackpath     = Path(mockstackpath) if mockstackpath else None
        self._stack_reader  = None  # tifffile.TiffFile when needed

        # buffers equivalent to CameraHIK implementation
        self.frame_buffer   = collections.deque(maxlen=nbuffer)
        self.frameid_buffer = collections.deque(maxlen=nbuffer)

        # background thread for continuous mode
        self._thread        = None
        self._stop_evt      = threading.Event()

        # prepare data source depending on mocktype
        self._prepare_source()

    # ---------------------------------------------------------------------
    #  Public API (subset of CameraHIK)
    # ---------------------------------------------------------------------

    # — acquisition control — ------------------------------------------------
    def start_live(self):
        if self._is_running:
            return
        self._stop_evt.clear()
        if self.trigger_source == TriggerSource.CONTINUOUS:
            self._thread = threading.Thread(target=self._continuous_loop,
                                            daemon=True)
            self._thread.start()
        self._is_running = True

    def stop_live(self):
        if not self._is_running:
            return
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._is_running = False

    suspend_live = stop_live  # alias for compatibility

    def getTriggerSource(self):
        """Match CameraHIK interface."""
        return self.trigger_source.value

    def getTriggerTypes(self):
        """Match CameraHIK interface."""
        return [TriggerSource.CONTINUOUS.value, TriggerSource.SOFTWARE.value, TriggerSource.EXTERNAL.value]

    def flushBuffer(self):
        self.frame_buffer.clear()
        self.frameid_buffer.clear()

    # — trigger handling — ---------------------------------------------------
    def setTriggerSource(self, source: str):
        """Match CameraHIK interface."""
        self.trigger_source = TriggerSource(source)
        # if we change to continuous while running, restart worker
        if self._is_running:
            if self.trigger_source == TriggerSource.CONTINUOUS and not self._thread:
                self.start_live()
            elif self.trigger_source != TriggerSource.CONTINUOUS and self._thread:
                self.stop_live()

    def send_trigger(self):
        """Software trigger (internal)."""
        if self.trigger_source != TriggerSource.SOFTWARE:
            return False
        self._emit_frame()
        return True

    def external_pulse(self):
        """Emulate a hardware rising‑edge when in EXT trigger mode."""
        if self.trigger_source != TriggerSource.EXTERNAL:
            return False
        self._emit_frame()
        return True

    # — frame access — -------------------------------------------------------
    def getLast(self, returnFrameNumber: bool = False):
        if not self.frame_buffer:
            return (None, -1) if returnFrameNumber else None
        if returnFrameNumber:
            return np.asarray(self.frame_buffer[-1]), self.frameid_buffer[-1]
        return np.asarray(self.frame_buffer[-1])

    def getLastChunk(self):
        ids   = np.asarray(self.frameid_buffer)
        stack = np.asarray(self.frame_buffer)
        self.flushBuffer()
        return stack, ids

    # maintain compatibility with CameraHIK configuration interface ----------
    def setPropertyValue(self, name, value):
        if name == "exposure":
            self.exposure_ms = value
        elif name == "gain":
            self.gain = value
        elif name == "trigger_source":
            self.setTriggerSource(value)
        return value

    def getPropertyValue(self, name):
        if name == "image_width":
            return self.SensorWidth
        if name == "image_height":
            return self.SensorHeight
        if name == "frame_number":
            return self._frame_counter
        return None

    # — dummy/irrelevant API stubs — ----------------------------------------
    def openPropertiesGUI(self):
        pass

    def close(self):
        self.stop_live()

    def flushBuffer(self):
        self.frame_buffer.clear()
        self.frameid_buffer.clear()

    # ---------------------------------------------------------------------
    #  Internal helpers
    # ---------------------------------------------------------------------

    def _prepare_source(self):
        """Set up any heavy data needed by the chosen *mocktype*."""
        if self._mocktype == "STORM" and self._stackpath and self._stackpath.exists():
            self._stack_reader = tif.TiffFile(self._stackpath)
        elif self._mocktype == "OffAxisHolo":
            self._holo_intensity_image = self._generate_hologram()

    # — continuous acquisition loop — ---------------------------------------
    def _continuous_loop(self):
        interval = self.exposure_ms / 1000.0
        while not self._stop_evt.wait(interval):
            self._emit_frame()

    # — frame emission — -----------------------------------------------------
    def _emit_frame(self):
        img = self._simulate_frame()
        fid = self._frame_counter
        self._frame_counter += 1

        self.frame_buffer.append(img)
        self.frameid_buffer.append(fid)

    # — synthetic frame generators — ----------------------------------------

    def _simulate_frame(self):
        """Return synthetic image depending on *mocktype* and *isRGB*."""
        if self._mocktype == "focus_lock":
            img = np.zeros((self.SensorHeight, self.SensorWidth), dtype=np.uint16)
            cy = int(np.random.randn() * 1  + self.SensorHeight/2)
            cx = int(np.random.randn() * 30 + self.SensorWidth/2)
            img[max(cy-10,0):cy+10, max(cx-10,0):cx+10] = 4095
        elif self._mocktype == "random_peak":
            img = self._random_peak_frame()
        elif self._mocktype == "STORM" and self._stack_reader:
            idx = self._frame_counter % len(self._stack_reader.pages)
            img = self._stack_reader.pages[idx].asarray()
        elif self._mocktype == "OffAxisHolo":
            img = self._holo_intensity_image
        else:  # "normal" default
            img = np.zeros((self.SensorHeight, self.SensorWidth), dtype=np.uint16)
            idx = np.random.choice(self.SensorHeight * self.SensorWidth, 200, replace=False)
            img.flat[idx] = np.random.randint(500, 4000, size=200)

        if self.isRGB:
            return np.stack([img, img, img], axis=-1)
        return img

    # — specialised generators — -------------------------------------------
    def _random_peak_frame(self):
        imgsize = (self.SensorHeight, self.SensorWidth)
        img = np.zeros(imgsize, dtype=np.float32)
        # add gaussian blob occasionally
        if np.random.rand() > 0.8:
            x, y = np.meshgrid(np.arange(imgsize[1]), np.arange(imgsize[0]))
            xc = (np.random.rand()*2-1)*imgsize[0]/2 + imgsize[0]/2
            yc = (np.random.rand()*2-1)*imgsize[1]/2 + imgsize[1]/2
            rv = multivariate_normal([yc, xc], [[60, 0], [0, 60]])
            img += 2000 * rv.pdf(np.dstack((y, x)))
        img += np.random.poisson(lam=15, size=imgsize)
        return img.astype(np.uint16)

    def _generate_hologram(self):
        """Off‑axis hologram once→ reused for every frame."""
        width, height = self.SensorWidth, self.SensorHeight
        wavelength    = 0.6328e-6
        k             = 2 * np.pi / wavelength
        angle         = np.pi / 10

        x = np.linspace(-np.pi, np.pi, width)
        y = np.linspace(-np.pi, np.pi, height)
        X, Y = np.meshgrid(x, y)
        pupil = (X**2 + Y**2) < np.pi
        phase_sample = np.exp(1j * ((X**2 + Y**2) < 50))

        tilt_x = k * np.sin(angle)
        tilt_y = k * np.sin(angle)
        Xpix, Ypix = np.meshgrid(np.arange(width), np.arange(height))
        plane_wave = np.exp(1j * (tilt_x * Xpix + tilt_y * Ypix))

        filtered = np.fft.ifft2(np.fft.fftshift(pupil) * np.fft.fft2(phase_sample))
        holo     = filtered + plane_wave
        return np.real(holo * np.conjugate(holo)).astype(np.float32)
