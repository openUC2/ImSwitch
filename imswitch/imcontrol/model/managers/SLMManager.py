"""Spatial light modulator / DMD shown as a display in a plane conjugate to the sample.

The manager owns the output (see ``slm_display``), the camera->display affine
calibration, the built-in test patterns and the pattern sequences on disk.
Everything is in *display pixels* (x right, y down, origin top-left) unless a
name says camera. Camera pixels are full-sensor pixels of the calibrated camera.

Files: ``~/ImSwitchConfig/slm/slm_affine_<name>.json`` and
``~/ImSwitchConfig/slm/sequences/<sequence>/NNNN_<frame>.png``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import dirtools, initLogger
from .galvoscanners.GalvoScannerManager import AffineTransform
from .slm_display import FramebufferDisplay, VirtualDisplay, WindowDisplay, open_display

# Display positions (fractions of width/height) for the calibration spots, in
# the order they are used; nPoints=5 gives the centre plus the inner corners.
_CAL_FRACTIONS = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
                  (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)]


# ───────────────────────────── patterns ─────────────────────────────

def pattern_blank(w: int, h: int, value: int = 0, **_) -> np.ndarray:
    return np.full((h, w), np.clip(int(value), 0, 255), np.uint8)


def pattern_checkerboard(w: int, h: int, period: float = 64, value: int = 255,
                         **_) -> np.ndarray:
    half = max(1, int(round(period / 2)))
    y, x = np.mgrid[0:h, 0:w]
    return (((x // half) + (y // half)) % 2 * int(value)).astype(np.uint8)


def pattern_grating(w: int, h: int, period: float = 32, angle: float = 0, duty: float = 0.5,
                    value: int = 255, phase: float = 0, sinusoidal: bool = False,
                    **_) -> np.ndarray:
    """Binary (or sinusoidal) line grating; ``angle`` in degrees, ``phase`` in periods."""
    period = max(2.0, float(period))
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.deg2rad(float(angle))
    proj = (x * np.cos(t) + y * np.sin(t)) / period + float(phase)
    if sinusoidal:
        img = 0.5 + 0.5 * np.cos(2 * np.pi * proj)
    else:
        img = (proj % 1.0) < float(duty)
    return np.round(img * int(value)).astype(np.uint8)


def pattern_disk(w: int, h: int, cx: float = -1, cy: float = -1, radius: float = 20,
                 value: int = 255, **_) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    cx = w / 2 if cx < 0 else cx
    cy = h / 2 if cy < 0 else cy
    cv2.circle(img, (int(round(cx)), int(round(cy))), max(1, int(round(radius))),
               int(value), -1, lineType=cv2.LINE_8)
    return img


def pattern_grid(w: int, h: int, nx: int = 3, ny: int = 3, radius: float = 10,
                 value: int = 255, **_) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    for i in range(max(1, int(nx))):
        for j in range(max(1, int(ny))):
            cx = int(round((i + 0.5) * w / max(1, int(nx))))
            cy = int(round((j + 0.5) * h / max(1, int(ny))))
            cv2.circle(img, (cx, cy), max(1, int(round(radius))), int(value), -1)
    return img


PATTERNS = {"blank": pattern_blank, "checkerboard": pattern_checkerboard,
            "grating": pattern_grating, "disk": pattern_disk, "grid": pattern_grid}


def make_pattern(kind: str, w: int, h: int, **params) -> np.ndarray:
    if kind not in PATTERNS:
        raise ValueError(f"unknown pattern {kind!r}; one of {sorted(PATTERNS)}")
    return PATTERNS[kind](int(w), int(h), **params)


def fit_frame(frame: np.ndarray, size: Tuple[int, int], mode: str = "center") -> np.ndarray:
    """Bring a uint8 image to the display size: ``center`` keeps 1:1 pixels (pad/crop),
    ``stretch`` resizes with nearest neighbour (binary stays binary)."""
    w, h = size
    f = np.asarray(frame)
    if f.dtype != np.uint8:
        f = np.clip(f, 0, 255).astype(np.uint8)
    if f.shape[0] == h and f.shape[1] == w:
        return f
    if mode == "stretch":
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h, w) + f.shape[2:], np.uint8)
    fh, fw = f.shape[:2]
    sy, sx = max(0, (fh - h) // 2), max(0, (fw - w) // 2)
    dy, dx = max(0, (h - fh) // 2), max(0, (w - fw) // 2)
    ch, cw = min(h, fh), min(w, fw)
    out[dy:dy + ch, dx:dx + cw] = f[sy:sy + ch, sx:sx + cw]
    return out


def decode_image(data: bytes) -> np.ndarray:
    """Any image file (PNG, BMP, TIFF, JPEG, ...) as uint8 grey."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("not a decodable image")
    return img


def encode_png(frame: np.ndarray, max_size: int = 0) -> bytes:
    f = np.asarray(frame)
    if f.dtype != np.uint8:
        lo, hi = float(np.min(f)), float(np.max(f))
        f = ((f - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo \
            else np.zeros(f.shape, np.uint8)
    if max_size and max(f.shape[:2]) > max_size:
        s = max_size / max(f.shape[:2])
        f = cv2.resize(f, (max(1, int(f.shape[1] * s)), max(1, int(f.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", f)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def spot_centroid(frame: np.ndarray, dark: Optional[np.ndarray] = None,
                  min_snr: float = 5.0) -> Optional[Tuple[float, float, float]]:
    """(x, y, snr) of one bright spot, or None when there is no clear spot.

    ``dark`` is a frame with the display blank; subtracting it removes ambient
    light and hot pixels. The spot is the connected bright region above half
    the peak after a Gaussian blur; its intensity-weighted centroid is returned.
    """
    f = np.asarray(frame, np.float32)
    if f.ndim == 3:
        f = f.mean(axis=2)
    if dark is not None and np.asarray(dark).size == f.size:
        f = f - np.asarray(dark, np.float32).reshape(f.shape)
    f = cv2.GaussianBlur(f, (0, 0), 3)
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med))) * 1.4826 + 1e-6
    peak = float(f.max())
    snr = (peak - med) / mad
    if snr < min_snr:
        return None
    mask = (f > med + 0.5 * (peak - med)).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask)
    if n > 2:  # keep the component holding the peak
        py, px = np.unravel_index(int(np.argmax(f)), f.shape)
        mask = (labels == labels[py, px]).astype(np.uint8)
    weights = (f - med) * mask
    m = cv2.moments(weights, binaryImage=False)
    if m["m00"] <= 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"], snr


def calibration_points(n: int, size: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Display positions of the calibration spots (3..9)."""
    n = max(3, min(len(_CAL_FRACTIONS), int(n)))
    w, h = size
    return [(int(round(fx * w)), int(round(fy * h))) for fx, fy in _CAL_FRACTIONS[:n]]


_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    name = _NAME_RE.sub("_", str(name or "")).strip("._")
    if not name:
        raise ValueError("empty name")
    return name[:80]


# ───────────────────────────── manager ─────────────────────────────

class SLMManager(SignalInterface):
    """One SLM/DMD display plus its camera mapping. See the module docstring."""

    sigSLMFrameShown = Signal(object)  # {"seq": int, "source": str}

    def __init__(self, slmInfo, name: str = "SLM", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self.name = name
        self.info = slmInfo
        self.backend = getattr(slmInfo, "display", "virtual") or "virtual"
        self.displayIndex = int(getattr(slmInfo, "displayIndex", 0) or 0)
        self._requested = (int(getattr(slmInfo, "width", 1920) or 1920),
                           int(getattr(slmInfo, "height", 1080) or 1080))
        self.settleTimeS = float(getattr(slmInfo, "settleTimeMs", 50.0) or 0) / 1000.0
        self.calibrationDetector = getattr(slmInfo, "calibrationDetector", None)
        self._lock = threading.Lock()
        self._seq = 0
        self.current: Optional[np.ndarray] = None
        self.currentSource = ""
        self.display = None
        self.error = ""
        self.affine = AffineTransform()
        self.calibration: Dict = {}
        self._load_affine()
        self.open()

    # ── output ──
    @property
    def size(self) -> Tuple[int, int]:
        return self.display.size if self.display is not None else self._requested

    @property
    def connected(self) -> bool:
        return self.display is not None and not isinstance(self.display, VirtualDisplay) \
            or self.backend == "virtual" and self.display is not None

    def open(self, backend: Optional[str] = None, index: Optional[int] = None) -> bool:
        """(Re)open the output. On failure the manager keeps working on a virtual
        display of the configured size and ``error`` says why."""
        self.close()
        if backend:
            self.backend = backend
        if index is not None:
            self.displayIndex = int(index)
        try:
            self.display = open_display(self.backend, *self._requested, self.displayIndex)
            self.error = ""
            self._logger.info(f"SLM display {self.backend}[{self.displayIndex}] "
                              f"{self.size[0]}x{self.size[1]}")
            if self.current is None:
                self.show(pattern_blank(*self.size), source="init", settle=0)
            return True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self._logger.warning(f"SLM display {self.backend} unavailable, virtual fallback: {e}")
            self.display = VirtualDisplay(*self._requested)
            return False

    def close(self) -> None:
        if self.display is not None:
            try:
                self.display.close()
            except Exception as e:
                self._logger.debug(f"closing SLM display: {e}")
            self.display = None

    def list_displays(self) -> list:
        if self.backend == "window":
            return WindowDisplay.list_displays()
        if self.backend == "framebuffer":
            return FramebufferDisplay.list_devices()
        return []

    def show(self, frame: np.ndarray, source: str = "pattern", fit: str = "center",
             settle: Optional[float] = None) -> int:
        """Put ``frame`` on the display, wait ``settle`` seconds (default the
        configured settle time) and return the frame sequence number."""
        f = fit_frame(frame, self.size, fit)
        with self._lock:
            self.display.show(f)
            self._seq += 1
            self.current, self.currentSource = f, source
            seq = self._seq
        time.sleep(self.settleTimeS if settle is None else max(0.0, settle))
        self.sigSLMFrameShown.emit({"seq": seq, "source": source})
        return seq

    def show_pattern(self, kind: str, **params) -> int:
        return self.show(make_pattern(kind, *self.size, **params), source=f"pattern:{kind}")

    def preview_png(self, max_size: int = 512) -> bytes:
        frame = self.current if self.current is not None else pattern_blank(*self.size)
        return encode_png(frame, max_size)

    # ── camera mapping ──
    def camera_mask_to_display(self, mask: np.ndarray, camera_size: Tuple[int, int],
                               value: int = 255, invert: bool = False) -> np.ndarray:
        """Warp a mask drawn over the camera image (any resolution, nonzero = on)
        into display pixels with the calibrated affine. ``camera_size`` is the
        (width, height) the mask's extent corresponds to, in camera pixels."""
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m.max(axis=2)
        mh, mw = m.shape
        cw, ch = camera_size
        sx, sy = cw / mw, ch / mh
        # pixel centres: mask pixel i spans camera [i*s, (i+1)*s), centre (i+0.5)*s-0.5
        scale = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], float)
        A = np.vstack([self.affine.to_matrix(), [0, 0, 1]])
        M = (A @ scale)[:2]
        w, h = self.size
        warped = cv2.warpAffine((m > 0).astype(np.uint8), M, (w, h),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        on = warped == 0 if invert else warped > 0
        return (on * np.clip(int(value), 0, 255)).astype(np.uint8)

    def camera_to_display(self, x: float, y: float) -> Tuple[float, float]:
        return self.affine.transform_point(x, y)

    def display_to_camera(self, x: float, y: float) -> Tuple[float, float]:
        inv = cv2.invertAffineTransform(self.affine.to_matrix())
        return (float(inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]),
                float(inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]))

    def compute_affine(self, camera_points: Sequence[Sequence[float]],
                       display_points: Sequence[Sequence[float]]) -> Dict:
        """Fit camera->display from >=3 pairs, keep it and save it. Returns the
        transform with the RMS residual in display pixels."""
        cam = [tuple(map(float, p)) for p in camera_points]
        disp = [tuple(map(float, p)) for p in display_points]
        if len(cam) != len(disp) or len(cam) < 3:
            raise ValueError("need at least 3 camera/display point pairs")
        self.affine = AffineTransform.compute_from_point_pairs(cam, disp)
        err = [np.hypot(*(np.subtract(self.affine.transform_point(*c), d)))
               for c, d in zip(cam, disp)]
        self.calibration = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_points": len(cam),
                            "residual_px": float(np.sqrt(np.mean(np.square(err)))),
                            "camera_points": cam, "display_points": disp}
        self.save_affine()
        return self.affine_dict()

    def affine_dict(self) -> Dict:
        return {"affine": self.affine.to_dict(), "calibrated": bool(self.calibration),
                "calibration": self.calibration}

    def set_affine(self, **coeffs) -> Dict:
        d = self.affine.to_dict()
        d.update({k: float(v) for k, v in coeffs.items() if v is not None and k in d})
        self.affine = AffineTransform.from_dict(d)
        self.calibration = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "manual": True}
        self.save_affine()
        return self.affine_dict()

    def reset_affine(self) -> Dict:
        self.affine = AffineTransform()
        self.calibration = {}
        self.save_affine()
        return self.affine_dict()

    def _slm_dir(self) -> str:
        path = os.path.join(dirtools.UserFileDirs.Root, "slm")
        os.makedirs(path, exist_ok=True)
        return path

    def _affine_path(self) -> str:
        return os.path.join(self._slm_dir(), f"slm_affine_{safe_name(self.name)}.json")

    def save_affine(self) -> str:
        path = self._affine_path()
        with open(path, "w") as f:
            json.dump({"affine_transform": self.affine.to_dict(), "calibration": self.calibration,
                       "display_size": list(self.size)}, f, indent=2)
        return path

    def _load_affine(self) -> None:
        try:
            with open(self._affine_path()) as f:
                data = json.load(f)
            self.affine = AffineTransform.from_dict(data.get("affine_transform", {}))
            self.calibration = data.get("calibration", {}) or {}
        except FileNotFoundError:
            pass
        except Exception as e:
            self._logger.warning(f"Could not load SLM affine: {e}")

    # ── sequences on disk ──
    def _sequences_dir(self) -> str:
        path = os.path.join(self._slm_dir(), "sequences")
        os.makedirs(path, exist_ok=True)
        return path

    def save_sequence(self, name: str, frames: Sequence[Tuple[str, np.ndarray]]) -> Dict:
        """Store frames as ``NNNN_<name>.png`` (grey uint8), replacing the sequence."""
        if not frames:
            raise ValueError("a sequence needs at least one frame")
        folder = os.path.join(self._sequences_dir(), safe_name(name))
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        for i, (fname, img) in enumerate(frames):
            stem = os.path.splitext(os.path.basename(fname or f"frame{i}"))[0]
            cv2.imwrite(os.path.join(folder, f"{i:04d}_{safe_name(stem)}.png"),
                        np.asarray(img, np.uint8))
        return self.sequence_info(safe_name(name))

    def sequence_info(self, name: str) -> Dict:
        folder = os.path.join(self._sequences_dir(), safe_name(name))
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        size = None
        if files:
            first = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_GRAYSCALE)
            size = [int(first.shape[1]), int(first.shape[0])] if first is not None else None
        return {"name": safe_name(name), "count": len(files), "size": size,
                "frames": [f[5:-4] for f in files]}

    def list_sequences(self) -> List[Dict]:
        root = self._sequences_dir()
        return [self.sequence_info(d) for d in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, d))]

    def load_sequence(self, name: str) -> List[Tuple[str, np.ndarray]]:
        folder = os.path.join(self._sequences_dir(), safe_name(name))
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"no SLM sequence {name!r}")
        out = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".png"):
                img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    out.append((f[5:-4], img))
        return out

    def delete_sequence(self, name: str) -> None:
        shutil.rmtree(os.path.join(self._sequences_dir(), safe_name(name)), ignore_errors=True)

    # ── status ──
    def status(self) -> Dict:
        w, h = self.size
        return {"name": self.name, "backend": self.backend, "displayIndex": self.displayIndex,
                "width": w, "height": h, "connected": self.connected, "error": self.error,
                "seq": self._seq, "source": self.currentSource,
                "settleTimeMs": self.settleTimeS * 1000.0,
                "calibrationDetector": self.calibrationDetector, **self.affine_dict()}


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
