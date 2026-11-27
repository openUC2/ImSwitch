"""Flat-field (shading) correction for transmitted-light tile scans.

A brightfield tile is darker towards its corners (vignetting) and, on a colour
camera, a little differently so in every channel. In a stitched slide that shows
as a grid of darker tile borders with a colour cast at the seams. The reference
is blank glass seen through the same optics; dividing by it makes the glass flat.

Model, per channel c: ``frame = dark + signal * F``, with ``F`` the shading
normalised to 1 at the brightest part of the field (its 99.5th percentile).
The correction ``dark + (frame - dark) / F`` keeps the centre of the field as
the camera gave it and lifts the corners to it, so exposure and colour balance
are unchanged and nothing that was not saturated saturates.

The reference is the median of several frames of blank glass taken a little
apart: debris on the slide moves between them and drops out, while dust on the
optics and the vignetting stay. Frames that differ more than that are not
blank glass, and the reference is refused rather than imprinted on every tile.

Pure numpy (cv2 when available for speed); no ImSwitch imports.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover - cv2 is a dependency of ImSwitch
    cv2 = None
    HAS_CV2 = False

FILE_NAME = "flatfield.ome.tif"
META_NAME = "flatfield.json"

# Gaussian sigma (px) for the reference: removes its shot noise (which would
# otherwise be stamped onto every tile as fixed-pattern noise) but keeps dust
# shadows of more than a few pixels.
SMOOTH_SIGMA_PX = 4.0
# The shading is never allowed below this (a corner at 0.2 would gain 5x noise).
MIN_SHADING = 0.2
# A pixel of a reference frame "moved" when it differs this much from the median.
MOVE_TOLERANCE = 0.10
# Share of moved pixels (mean over frames) above which the reference is refused,
# and above which it is accepted with a warning.
MOVED_REFUSE = 0.05
MOVED_WARN = 0.01
# Exposure limits as a fraction of full scale (median of the reference).
TOO_DARK = 0.08
SATURATED_SHARE = 0.002


def full_scale(dtype) -> float:
    """Largest value of an integer image type (1.0 for float images)."""
    dtype = np.dtype(dtype)
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def data_full_scale(data: np.ndarray) -> float:
    """Full scale of camera data: a 12-bit camera delivers uint16 that never exceeds 4095.

    The smallest of 2^8, 2^10, 2^12, 2^14, 2^16 (minus one) that holds the data;
    the type's own maximum for anything else.
    """
    dtype = np.asarray(data).dtype
    if not np.issubdtype(dtype, np.integer):
        return full_scale(dtype)
    top = float(np.max(data)) if np.size(data) else 0.0
    for bits in (8, 10, 12, 14, 16):
        if top <= 2 ** bits - 1 and 2 ** bits - 1 <= full_scale(dtype):
            return float(2 ** bits - 1)
    return full_scale(dtype)


def _as_hwc(frame: np.ndarray) -> np.ndarray:
    """(H, W) -> (H, W, 1); (H, W, C) unchanged."""
    return frame[..., None] if frame.ndim == 2 else frame


def _smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of an (H, W, C) float32 image, channel by channel."""
    if sigma <= 0:
        return img
    if HAS_CV2:
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[..., c] = cv2.GaussianBlur(img[..., c], (0, 0), sigma,
                                           borderType=cv2.BORDER_REFLECT)
        return out
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=(sigma, sigma, 0), mode="reflect").astype(np.float32)


@dataclass
class FlatField:
    """A shading reference and the correction it implies.

    ``shading`` is F (H, W, C) float32, 1 at the brightest part of the field;
    ``dark`` the per-channel offset that is not light. ``meta`` records how and
    where it was taken (objective, camera settings, quality numbers).
    """

    shading: np.ndarray
    dark: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.shading = _as_hwc(np.asarray(self.shading, dtype=np.float32))
        self.shading = np.clip(self.shading, MIN_SHADING, None)
        self.dark = np.asarray(self.dark, dtype=np.float32).reshape(-1)
        if self.dark.size == 1 and self.shading.shape[2] > 1:
            self.dark = np.repeat(self.dark, self.shading.shape[2])
        self._gain = (1.0 / self.shading).astype(np.float32)
        self._gain2d = np.ascontiguousarray(self._gain[..., 0])

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.shading.shape[0]), int(self.shading.shape[1])

    @property
    def channels(self) -> int:
        return int(self.shading.shape[2])

    def matches(self, frame: np.ndarray) -> bool:
        """True when the frame has the reference's size and number of channels."""
        channels = 1 if frame.ndim == 2 else frame.shape[2]
        return tuple(frame.shape[:2]) == self.shape and channels == self.channels

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """The corrected frame, same shape and type; frames that do not match come back as is."""
        if not self.matches(frame):
            return frame
        if HAS_CV2 and frame.dtype == np.uint8 and self.channels <= 4:
            return self._apply_cv2(frame)
        work = _as_hwc(frame).astype(np.float32)
        work -= self.dark
        np.maximum(work, 0, out=work)
        work *= self._gain
        work += self.dark
        np.clip(work, 0, full_scale(frame.dtype), out=work)
        if np.issubdtype(frame.dtype, np.integer):
            np.rint(work, out=work)
        return work.astype(frame.dtype).reshape(frame.shape)

    def _apply_cv2(self, frame: np.ndarray) -> np.ndarray:
        """uint8 in OpenCV (several times faster than numpy on a 6 MP RGB tile).

        Saturating subtract of the dark level, float multiply by 1/F, add the
        dark level back, then round and saturate to 0..255.
        """
        gain = self._gain if frame.ndim == 3 else self._gain2d
        dark = [float(d) for d in self.dark]
        if any(dark):
            scalar = tuple(dark + [0.0] * (4 - len(dark)))
            scaled = cv2.multiply(cv2.subtract(frame, scalar), gain, dtype=cv2.CV_32F)
            scaled = cv2.add(scaled, scalar)
        else:
            scaled = cv2.multiply(frame, gain, dtype=cv2.CV_32F)
        return cv2.convertScaleAbs(scaled).reshape(frame.shape)

    # ------------------------------------------------------------------ #
    # Files
    # ------------------------------------------------------------------ #

    def save(self, folder: str, pixel_size_um: Optional[float] = None) -> str:
        """Write ``flatfield.ome.tif`` (F, one plane per channel) and ``flatfield.json``."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, FILE_NAME)
        names = ["R", "G", "B"] if self.channels == 3 else [f"C{i}" for i in range(self.channels)]
        planes = np.moveaxis(self.shading, 2, 0)   # CYX
        try:
            from imswitch.imcontrol.model.aufgaben.storage import save_image
            save_image(path, planes, "CYX", pixel_size_um=pixel_size_um,
                       channel_names=names, name="flat-field shading",
                       annotations={"flatfield": {"model": "dark + signal * F",
                                                  "dark": [float(d) for d in self.dark]}},
                       objective={})
        except ImportError:  # outside ImSwitch (tests of this module alone)
            import tifffile
            tifffile.imwrite(path, planes, metadata={"axes": "CYX"})
        meta = dict(self.meta)
        meta.update({"dark": [float(d) for d in self.dark], "shape": list(self.shape),
                     "channels": self.channels, "file": FILE_NAME})
        with open(os.path.join(folder, META_NAME), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, default=str)
        return path

    @classmethod
    def load(cls, folder: str) -> "FlatField":
        import tifffile
        with open(os.path.join(folder, META_NAME), encoding="utf-8") as f:
            meta = json.load(f)
        planes = np.asarray(tifffile.imread(os.path.join(folder, FILE_NAME)), dtype=np.float32)
        planes = planes.reshape((-1,) + planes.shape[-2:])
        return cls(np.moveaxis(planes, 0, 2), np.asarray(meta.get("dark", [0.0])), meta)

    def thumbnail(self, width: int = 160) -> Optional[str]:
        """Base64 JPEG of F (white = 1) for a UI; None without cv2."""
        if not HAS_CV2:
            return None
        import base64
        img = np.clip(self.shading.mean(axis=2) * 255.0, 0, 255).astype(np.uint8)
        h = max(1, int(round(img.shape[0] * width / img.shape[1])))
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _dark_levels(dark: Optional[np.ndarray], channels: int) -> np.ndarray:
    """Per-channel dark level from a dark frame (0 without one)."""
    if dark is None:
        return np.zeros(channels, np.float32)
    d = _as_hwc(np.asarray(dark)).astype(np.float32)
    levels = np.median(d.reshape(-1, d.shape[2]), axis=0).astype(np.float32)
    return levels if levels.size == channels else np.full(channels, float(np.median(d)), np.float32)


def _exposure_problem(frames: List[np.ndarray], fs: float, dark_c: np.ndarray,
                      report: Dict[str, Any]) -> str:
    """Why the frames cannot be a reference (saturated, dark), or ''."""
    saturated = float(np.mean([(f >= fs * 0.99).mean() for f in frames])) if fs > 1 else 0.0
    report["saturated_share"] = round(saturated, 5)
    if saturated > SATURATED_SHARE:
        return (f"{saturated * 100:.1f} % of the reference is saturated: lower the exposure "
                f"or the light and take it again.")
    level = float(np.median([np.median(f[::4, ::4]) for f in frames])) - float(dark_c.mean())
    report["level"] = round(level / fs, 3) if fs > 1 else round(level, 3)
    if level < TOO_DARK * fs:
        return (f"The reference is too dark ({level / fs * 100:.0f} % of full scale): is the "
                f"light on and the stage over glass?")
    return ""


def _median_signal(frames: List[np.ndarray], dark_c: np.ndarray) -> np.ndarray:
    """Per-pixel median minus dark, a band of rows at a time.

    Never more than a few MB in flight, where stacking 9 RGB frames of 6 MP
    as float is 700 MB.
    """
    height, width, channels = frames[0].shape
    signal = np.empty((height, width, channels), np.float32)
    for y0 in range(0, height, 128):
        band = np.stack([f[y0:y0 + 128] for f in frames])
        signal[y0:y0 + 128] = np.median(band, axis=0) if len(frames) > 1 else band[0]
    signal -= dark_c
    np.maximum(signal, 1.0, out=signal)
    return signal


def _moved_share(frames: List[np.ndarray], signal: np.ndarray, dark_c: np.ndarray) -> float:
    """Share of the field that changes between the frames (mean over frames).

    Structure that moves is not shading: tissue, debris on the slide, a slide
    edge. Judged at about 512 px across, plenty for "how much moved".
    """
    step = max(1, int(round(max(signal.shape[:2]) / 512)))

    def small(img):
        return _as_hwc(img[::step, ::step].astype(np.float32))

    ref = np.maximum(_smooth(small(signal), 1.5), 1.0)
    moved = []
    for f in frames:
        ratio = _smooth(np.maximum(small(f) - dark_c, 0.0), 1.5) / ref
        moved.append(float((np.abs(ratio - 1.0) > MOVE_TOLERANCE).mean()))
    return float(np.mean(moved))


def _normalised_shading(signal: np.ndarray, sigma_px: float, report: Dict[str, Any]) -> np.ndarray:
    """Smoothed signal scaled to 1 at its brightest (99.5th percentile) per channel."""
    channels = signal.shape[2]
    shading = _smooth(signal, sigma_px)
    top = np.percentile(shading[::4, ::4].reshape(-1, channels), 99.5, axis=0)
    shading /= np.maximum(top, 1e-6).astype(np.float32)
    sample = shading[::4, ::4].reshape(-1, channels)
    low = float(np.percentile(sample, 0.5))
    report["falloff"] = round(1.0 - low, 3)   # 0.3 = corners 30 % darker than the centre
    report["falloff_per_channel"] = [round(1.0 - float(np.percentile(sample[:, c], 0.5)), 3)
                                     for c in range(channels)]
    if low < MIN_SHADING:
        report["warnings"].append(f"Strong vignetting (darkest part at {low * 100:.0f} % of the "
                                  f"centre): the corners gain noise when corrected.")
    return shading


def estimate(frames: Sequence[np.ndarray], dark: Optional[np.ndarray] = None,
             sigma_px: float = SMOOTH_SIGMA_PX,
             max_value: Optional[float] = None) -> Tuple[Optional[FlatField], Dict[str, Any]]:
    """A flat-field from frames of blank glass (and optionally a dark frame).

    ``max_value`` is the camera's full scale (4095 for 12 bit); by default it
    is inferred from the data (``data_full_scale``).

    Returns ``(flat, report)``; ``flat`` is None when the frames are no
    usable reference, and ``report["reason"]`` says why in words an
    operator can act on. ``report["warnings"]`` lists what was accepted
    but is worth knowing.
    """
    report: Dict[str, Any] = {"ok": False, "reason": "", "warnings": [], "frames": len(frames)}
    if not frames:
        report["reason"] = "No frames were taken."
        return None, report
    shapes = {tuple(np.asarray(f).shape) for f in frames}
    if len(shapes) != 1:
        report["reason"] = f"The frames differ in size: {sorted(shapes)}."
        return None, report
    frames = [_as_hwc(np.asarray(f)) for f in frames]
    fs = float(max_value) if max_value else max(data_full_scale(f) for f in frames)
    dark_c = _dark_levels(dark, frames[0].shape[2])
    report["dark"] = [round(float(v), 2) for v in dark_c]
    report["reason"] = _exposure_problem(frames, fs, dark_c, report)
    if report["reason"]:
        return None, report

    signal = _median_signal(frames, dark_c)
    if len(frames) > 1:
        share = report["moved_share"] = round(_moved_share(frames, signal, dark_c), 4)
        if share > MOVED_REFUSE:
            report["reason"] = (f"Not blank glass: {share * 100:.0f} % of the field changes "
                                f"between the reference frames (tissue, a slide edge or dirt). "
                                f"Move to clean glass and take it again.")
            return None, report
        if share > MOVED_WARN:
            report["warnings"].append(f"{share * 100:.1f} % of the field changed between the "
                                      f"frames (debris); the median removes most of it.")
    else:
        report["warnings"].append("One frame only: debris on the glass ends up in the reference.")

    shading = _normalised_shading(signal, sigma_px, report)
    flat = FlatField(shading, dark_c, {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "frames": len(frames),
        "sigma_px": sigma_px, "level": report["level"], "falloff": report["falloff"],
        "moved_share": report.get("moved_share")})
    report["ok"] = True
    return flat, report


def reference_offsets(n_frames: int, spacing_um: float) -> List[Tuple[float, float]]:
    """Stage offsets (µm) for the reference frames: a centred grid, nearest first.

    9 frames -> 3 x 3 at ``spacing_um``; 1 -> the spot itself. Debris on the
    slide moves by the spacing between frames, so it drops out of the median.
    """
    n = max(1, int(n_frames))
    side = int(np.ceil(np.sqrt(n)))
    half = (side - 1) / 2.0
    grid = [((ix - half) * spacing_um, (iy - half) * spacing_um)
            for iy in range(side) for ix in range(side)]
    grid.sort(key=lambda p: (p[0] ** 2 + p[1] ** 2, p[1], p[0]))
    return grid[:n]


def safe_name(text: Any) -> str:
    """``text`` usable as a file or folder name."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text)) or "none"


def reference_key(detector: str, objective: str, shape: Sequence[int], binning: Any = 1,
                  channel: str = "") -> str:
    """The folder a reference is stored under: per camera, objective, size, binning and light."""
    key = (f"{safe_name(detector)}__{safe_name(objective)}__{int(shape[0])}x{int(shape[1])}"
           f"__bin{safe_name(binning)}")
    return f"{key}__{safe_name(channel)}" if channel else key
