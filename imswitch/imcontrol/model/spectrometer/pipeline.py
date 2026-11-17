"""Camera frame(s) -> spectrum: region of interest, dark, averaging, spikes, smile, collapse.

Rows run along the slit, columns along the dispersion. All row and column numbers are
full-sensor indices. Plain numpy; nothing here touches hardware.
"""
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .peaks import dominant_feature


@dataclass
class ProcessingSettings:
    rowStart: int = 0
    rowStop: int = 0              # exclusive; 0 = to the last row
    colStart: int = 0
    colStop: int = 0              # exclusive; 0 = to the last column
    collapse: str = "mean"        # "mean" or "sum" over the rows of the region
    nAverage: int = 1             # frames averaged per spectrum
    removeSpikes: bool = True     # median of >= 3 frames, else a 3 x 3 median test
    spikeThreshold: float = 6.0   # in robust standard deviations
    subtractDark: bool = True
    smileCorrection: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "ProcessingSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def to_dict(self) -> Dict:
        return asdict(self)

    def bounds(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """(row0, row1, col0, col1) clipped to the frame; empty ranges become full ones."""
        h, w = int(shape[0]), int(shape[1])
        r0 = int(np.clip(self.rowStart, 0, h - 1))
        r1 = int(np.clip(self.rowStop or h, r0 + 1, h))
        c0 = int(np.clip(self.colStart, 0, w - 1))
        c1 = int(np.clip(self.colStop or w, c0 + 1, w))
        return r0, r1, c0, c1


def as_mono(frame) -> np.ndarray:
    """2-D float32 view of a frame; colour frames are averaged over their channels."""
    a = np.asarray(frame)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., :3].mean(axis=-1)
    elif a.ndim == 3 and a.shape[0] in (3, 4):
        a = a[:3].mean(axis=0)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {a.shape}")
    return a.astype(np.float32, copy=False)


def _clipped_var(d: np.ndarray, n_iter: int = 3, k: float = 4.0) -> float:
    """Variance after iterative k-sigma clipping (not quantised like the MAD of integers)."""
    v = float(np.var(d))
    for _ in range(n_iter):
        kept = d[np.abs(d) <= k * np.sqrt(v) + 1e-12]
        if kept.size < 10:
            break
        v = float(np.var(kept))
    return v


def noise_model(level: np.ndarray, dev: np.ndarray,
                n_bins: int = 16) -> Tuple[float, float, float]:
    """Noise variance against signal level, var = max(a + b * level, floor), from the
    data itself (photon transfer: read noise and camera offset in a, shot noise b).
    Level bins are spaced evenly from the 1st percentile to the maximum, so a few bright
    line pixels still set the slope. Returns (a, b, floor)."""
    lv, d = np.ravel(level), np.ravel(dev)
    if lv.size > 400_000:                       # a regular subsample is plenty
        step = lv.size // 400_000 + 1
        lv, d = lv[::step], d[::step]
    lo, hi = float(np.percentile(lv, 1)), float(np.max(lv))
    xs, vs, ns = [], [], []
    min_n = max(30, lv.size // 20000)
    if hi > lo:
        edges = np.linspace(lo, hi, n_bins + 1)
        idx = np.clip(np.digitize(lv, edges) - 1, 0, n_bins - 1)
        for i in range(n_bins):
            m = idx == i
            if m.sum() >= min_n:
                xs.append(float(np.median(lv[m])))
                vs.append(_clipped_var(d[m]))
                ns.append(float(m.sum()))
    if len(xs) < 2:
        v = max(_clipped_var(d) if d.size else 1.0, 1e-6)
        return v, 0.0, v
    floor = max(vs[0], 1e-6)                    # the quietest well-filled level
    b, a = np.polyfit(xs, vs, 1, w=np.sqrt(ns) / np.maximum(vs, 1e-6))
    if b <= 0:
        a, b = float(np.average(vs, weights=ns)), 0.0
    return float(a), float(b), float(floor)


def _sigma(level: np.ndarray, a: float, b: float, floor: float) -> np.ndarray:
    return np.sqrt(np.maximum(a + b * level, floor))


def _spikes(dev: np.ndarray, med: np.ndarray, model, threshold: float,
            rel: float) -> np.ndarray:
    """Spike test: far above the noise AND above the local level by the fraction rel
    (a sparse bright line cannot fool the noise model into removing itself)."""
    return (dev > threshold * _sigma(med, *model)) & (dev > rel * np.clip(med, 0, None))


def stack_mean(frames: Sequence[np.ndarray], remove_spikes: bool = True,
               threshold: float = 6.0, rel: float = 0.3) -> Tuple[np.ndarray, int]:
    """Mean of frames; with >= 3 frames, values far above the per-pixel median are
    replaced by that median first (cosmic rays, flicker). Returns (mean, n replaced)."""
    stack = np.stack([as_mono(f) for f in frames]).astype(np.float32)
    if stack.shape[0] < 3 or not remove_spikes:
        return stack.mean(axis=0), 0
    med = np.median(stack, axis=0)
    dev = stack - med
    model = noise_model(np.broadcast_to(med, dev.shape), dev)
    bad = _spikes(dev, med[None], model, threshold, rel)
    if bad.any():
        stack = np.where(bad, med[None], stack)
    return stack.mean(axis=0), int(bad.sum())


def despike_2d(img: np.ndarray, threshold: float = 6.0, half: int = 2,
               rel: float = 0.3) -> Tuple[np.ndarray, int]:
    """Replace spikes and hot pixels by the median of their column neighbours.

    The median runs along the slit (rows): a spectral line is long in that direction
    and survives whatever its width; a spike is not. The noise threshold follows the
    signal (noise_model)."""
    a = np.asarray(img, dtype=np.float32)
    h = a.shape[0]
    if h < 2 * half + 1:
        return a, 0
    p = np.pad(a, ((half, half), (0, 0)), mode="edge")
    med = np.median(np.stack([p[i:i + h] for i in range(2 * half + 1)]), axis=0)
    dev = a - med
    inner = slice(half, h - half) if h > 4 * half + 2 else slice(None)   # padded rows biased
    model = noise_model(med[inner], dev[inner])
    bad = _spikes(dev, med, model, threshold, rel)
    return (np.where(bad, med, a), int(bad.sum())) if bad.any() else (a, 0)


def smile_shifts(rows, smile: Optional[Dict]) -> np.ndarray:
    """Column offset of a line at each row relative to the reference row (px).

    smile: {"coeffs": [c0, c1, ...] ascending in u = (row - rowCenter) / rowScale}.
    """
    rows = np.asarray(rows, dtype=float)
    if not smile or not smile.get("coeffs"):
        return np.zeros_like(rows)
    u = (rows - float(smile["rowCenter"])) / float(smile.get("rowScale") or 1.0)
    c = np.asarray(smile["coeffs"], dtype=float)
    return np.polynomial.polynomial.polyval(u, c) - c[0]


def apply_shifts(img: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """out[r, x] = img[r, x + shifts[r]] with linear interpolation (edges repeated)."""
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape
    xs = np.arange(w, dtype=np.float32)[None, :] + np.asarray(shifts, np.float32)[:, None]
    xs = np.clip(xs, 0, w - 1)
    i0 = np.floor(xs).astype(np.int64)
    i1 = np.minimum(i0 + 1, w - 1)
    f = xs - i0
    v0 = np.take_along_axis(a, i0, axis=1)
    v1 = np.take_along_axis(a, i1, axis=1)
    return (1 - f) * v0 + f * v1


def collapse_rows(roi: np.ndarray, mode: str = "mean") -> np.ndarray:
    return (roi.sum(axis=0) if mode == "sum" else roi.mean(axis=0)).astype(np.float64)


def find_slit_rows(frame, frac: float = 0.5, pad: float = 0.0) -> Dict:
    """Rows lit by the slit image: the contiguous band around the brightest row whose
    mean is above base + frac x (peak - base). pad widens it by that fraction."""
    mono = as_mono(frame)
    prof = mono.mean(axis=1).astype(np.float64)
    base, peak = np.percentile(prof, 5), np.percentile(prof, 99.5)
    if peak - base <= 1e-9:
        return {"rowStart": 0, "rowStop": int(prof.size), "found": False,
                "profile": prof.tolist()}
    lit = prof >= base + frac * (peak - base)
    i = int(np.argmax(prof))
    r0 = i
    while r0 > 0 and lit[r0 - 1]:
        r0 -= 1
    r1 = i + 1
    while r1 < prof.size and lit[r1]:
        r1 += 1
    extra = int(round(pad * (r1 - r0)))
    return {"rowStart": max(0, r0 - extra), "rowStop": min(prof.size, r1 + extra),
            "found": True, "profile": prof.tolist()}


def measure_smile(roi: np.ndarray, row0: int, n_bins: int = 16, order: int = 2,
                  window_px: Optional[int] = None) -> Dict:
    """Curvature of a narrow line along the slit, from a dark-subtracted region.

    The line is the brightest feature of the collapsed region; in each of n_bins row
    bands its centroid is measured inside +-window_px. Returns the smile polynomial
    (see smile_shifts) with the per-band positions and the bow (max - min, px).
    """
    a = np.asarray(roi, dtype=np.float64)
    h, w = a.shape
    n_bins = int(max(3, min(n_bins, h)))
    line = dominant_feature(a.mean(axis=0))
    half = int(window_px or max(8, 2 * line["fwhm"]))
    c = int(round(line["center"]))
    lo, hi = max(0, c - half), min(w, c + half + 1)
    edges = np.linspace(0, h, n_bins + 1).astype(int)
    rows, pos = [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        prof = a[b0:b1, lo:hi].mean(axis=0)
        f = dominant_feature(prof)
        if f["snr"] > 5:
            rows.append(row0 + 0.5 * (b0 + b1 - 1))
            pos.append(lo + f["center"])
    if len(rows) < order + 2:
        raise ValueError("line too weak or too short along the slit to measure the smile")
    rows_a, pos_a = np.asarray(rows), np.asarray(pos)
    rc, rs = row0 + (h - 1) / 2.0, max((h - 1) / 2.0, 1.0)
    coeffs = np.polynomial.polynomial.polyfit((rows_a - rc) / rs, pos_a, order)
    fit = np.polynomial.polynomial.polyval((rows_a - rc) / rs, coeffs)
    full = np.polynomial.polynomial.polyval(np.linspace(-1, 1, 101), coeffs)
    return {"coeffs": coeffs.tolist(), "rowCenter": rc, "rowScale": rs,
            "rows": [int(row0), int(row0 + h)], "lineColumn": float(coeffs[0]),
            "binRows": rows_a.tolist(), "binPositions": pos_a.tolist(),
            "residualRmsPx": float(np.sqrt(np.mean((pos_a - fit) ** 2))),
            "bowPx": float(np.ptp(full))}


def process_roi(roi: np.ndarray, rows: Sequence[int], cols: Sequence[int],
                settings: ProcessingSettings, dark_roi: Optional[np.ndarray] = None,
                smile: Optional[Dict] = None, despiked: bool = False) -> Dict:
    """Region (already cut from the frame at rows/cols) -> spectrum.

    despiked=True skips the single-frame spike test (frames were median-cleaned)."""
    roi = as_mono(roi)
    if dark_roi is not None and settings.subtractDark:
        if dark_roi.shape != roi.shape:
            raise ValueError(f"dark {dark_roi.shape} does not match the region {roi.shape}")
        roi = roi - dark_roi
    n_spikes = 0
    if settings.removeSpikes and not despiked:
        roi, n_spikes = despike_2d(roi, settings.spikeThreshold)
    if smile and settings.smileCorrection:
        roi = apply_shifts(roi, smile_shifts(np.arange(rows[0], rows[1]), smile))
    counts = collapse_rows(roi, settings.collapse)
    return {"px": np.arange(cols[0], cols[1], dtype=np.float64), "counts": counts, "roi": roi,
            "rows": [int(rows[0]), int(rows[1])], "cols": [int(cols[0]), int(cols[1])],
            "spikesRemoved": n_spikes}


def process(frame: np.ndarray, settings: ProcessingSettings, dark: Optional[np.ndarray] = None,
            smile: Optional[Dict] = None, despiked: bool = False) -> Dict:
    """One (averaged) full frame -> spectrum. Returns px, counts and the corrected region."""
    mono = as_mono(frame)
    if dark is not None and dark.shape != mono.shape:
        raise ValueError(f"dark {dark.shape} does not match the frame {mono.shape}")
    r0, r1, c0, c1 = settings.bounds(mono.shape)
    dark_roi = dark[r0:r1, c0:c1] if dark is not None else None
    return process_roi(mono[r0:r1, c0:c1], (r0, r1), (c0, c1), settings, dark_roi, smile,
                       despiked)


def saturation(frame, bits: int, rows=None, cols=None) -> Dict:
    """Raw-frame peak and the fraction of pixels at the top of the range."""
    a = np.asarray(frame)
    if rows is not None and cols is not None:
        a = a[rows[0]:rows[1], cols[0]:cols[1]]
    top = (1 << int(bits)) - 1
    peak = float(np.max(a)) if a.size else 0.0
    return {"peak": peak, "fullScale": top, "peakFraction": peak / top if top else 0.0,
            "saturatedPixels": int(np.count_nonzero(a >= top)), "saturated": bool(peak >= top)}


def decimate(x: np.ndarray, y: np.ndarray, max_points: int):
    """Downsample for display keeping each bin's maximum (narrow lines survive)."""
    n = len(y)
    if not max_points or n <= max_points:
        return np.asarray(x), np.asarray(y)
    k = int(np.ceil(n / max_points))
    m = (n // k) * k
    yy = np.asarray(y[:m]).reshape(-1, k)
    xx = np.asarray(x[:m]).reshape(-1, k)
    j = np.argmax(yy, axis=1)
    rows = np.arange(yy.shape[0])
    return xx[rows, j], yy[rows, j]
