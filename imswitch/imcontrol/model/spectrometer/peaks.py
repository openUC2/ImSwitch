"""Peaks in a 1-D spectrum: detection, sub-pixel centre, width, matching to line lists."""
from typing import Dict, List, Optional, Sequence

import numpy as np


def noise_sigma(y) -> float:
    """Robust noise from the first difference (insensitive to smooth features)."""
    d = np.diff(np.asarray(y, dtype=float))
    if d.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def baseline_level(y) -> float:
    """Median of the lowest 20 % of the values."""
    v = np.sort(np.asarray(y, dtype=float))
    return float(np.median(v[:max(1, v.size // 5)]))


def _smooth3(y: np.ndarray) -> np.ndarray:
    if y.size < 3:
        return y.copy()
    p = np.pad(y, 1, mode="edge")
    return (p[:-2] + p[1:-1] + p[2:]) / 3.0


def _prominence(y: np.ndarray, i: int):
    """Prominence of the maximum at i and the two bases (scipy's definition)."""
    h = y[i]
    left = np.nonzero(y[:i] > h)[0]
    lo = left[-1] + 1 if left.size else 0
    right = np.nonzero(y[i + 1:] > h)[0]
    hi = i + 1 + right[0] if right.size else y.size
    lmin = float(np.min(y[lo:i + 1]))
    rmin = float(np.min(y[i:hi]))
    return h - max(lmin, rmin), lmin, rmin


def find_peaks(y, min_prominence: Optional[float] = None, min_distance: int = 5,
               max_peaks: int = 40) -> List[Dict]:
    """Local maxima with their prominence, strongest first.

    Default threshold: 6 x the noise or 3 % of the spectrum's range, whichever is larger.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return []
    ys = _smooth3(y)
    cand = np.nonzero((ys[1:-1] > ys[:-2]) & (ys[1:-1] >= ys[2:]))[0] + 1
    if min_prominence is None:
        min_prominence = max(6.0 * noise_sigma(y), 0.03 * float(np.ptp(ys)))
    found = []
    for i in cand:
        prom, lmin, rmin = _prominence(ys, int(i))
        if prom >= min_prominence:
            found.append({"index": int(i), "prominence": float(prom),
                          "base": float(max(lmin, rmin))})
    found.sort(key=lambda p: -p["prominence"])
    kept: List[Dict] = []
    for p in found:
        if all(abs(p["index"] - q["index"]) >= min_distance for q in kept):
            kept.append(p)
        if len(kept) >= max_peaks:
            break
    return kept


def _crossing(y: np.ndarray, i: int, level: float, step: int) -> float:
    """Sub-pixel position where y falls below level, walking from i in direction step."""
    j = i
    while 0 <= j + step < y.size and y[j + step] >= level:
        j += step
    k = j + step
    if not 0 <= k < y.size:
        return float(j)
    y0, y1 = y[j], y[k]
    frac = (y0 - level) / (y0 - y1) if y0 != y1 else 0.0
    return float(j + step * frac)


def peak_properties(y, index: int, base: Optional[float] = None, x=None) -> Dict:
    """Centre (centroid above half maximum), FWHM, height above base; in x if given.

    x must be ascending or descending and evenly spaced if widths in x are used; with
    x=None everything is in samples.
    """
    y = np.asarray(y, dtype=float)
    i = int(index)
    if base is None:
        base = _prominence(_smooth3(y), i)[1:]
        base = float(max(base))
    height = float(y[i] - base)
    half = base + 0.5 * height
    left, right = _crossing(y, i, half, -1), _crossing(y, i, half, +1)
    lo, hi = int(np.ceil(left)), int(np.floor(right))
    seg = np.arange(lo, hi + 1)
    w = np.clip(y[seg] - half, 0, None)
    center = float(np.sum(seg * w) / np.sum(w)) if np.sum(w) > 0 else float(i)
    props = {"index": i, "center": center, "height": height, "base": float(base),
             "fwhm": float(right - left), "left": left, "right": right,
             "area": float(np.sum(np.clip(y[lo:hi + 1] - base, 0, None))),
             "snr": float(height / max(noise_sigma(y), 1e-12))}
    if x is not None:
        x = np.asarray(x, dtype=float)
        idx = np.arange(x.size)
        props["centerX"] = float(np.interp(center, idx, x))
        props["fwhmX"] = float(abs(np.interp(right, idx, x) - np.interp(left, idx, x)))
    return props


def dominant_feature(y, frac: float = 0.5) -> Dict:
    """Centre and width of the strongest broad feature (an LED band).

    Centroid of the contiguous region around the maximum that lies above
    base + frac x (max - base), weighted by the height above that level.
    """
    y = np.asarray(y, dtype=float)
    ys = _smooth3(y)
    i = int(np.argmax(ys))
    base = baseline_level(ys)
    level = base + frac * (ys[i] - base)
    left, right = _crossing(ys, i, level, -1), _crossing(ys, i, level, +1)
    seg = np.arange(int(np.ceil(left)), int(np.floor(right)) + 1)
    w = np.clip(ys[seg] - level, 0, None)
    center = float(np.sum(seg * w) / np.sum(w)) if np.sum(w) > 0 else float(i)
    half = base + 0.5 * (ys[i] - base)
    fwhm = _crossing(ys, i, half, +1) - _crossing(ys, i, half, -1)
    return {"index": i, "center": center, "height": float(ys[i] - base), "base": base,
            "fwhm": float(fwhm), "left": left, "right": right,
            "snr": float((ys[i] - base) / max(noise_sigma(y), 1e-12))}


def reference_feature(nm, intensity, frac: float = 0.5) -> Dict:
    """Same measure as dominant_feature on a reference spectrum, in nm."""
    nm = np.asarray(nm, dtype=float)
    f = dominant_feature(intensity, frac)
    idx = np.arange(nm.size)
    return {"centerNm": float(np.interp(f["center"], idx, nm)),
            "fwhmNm": float(abs(np.interp(f["index"] + f["fwhm"] / 2, idx, nm)
                                - np.interp(f["index"] - f["fwhm"] / 2, idx, nm)))}


def cross_correlation_shift(a, b, max_shift: int) -> float:
    """Sub-pixel shift s that best maps b onto a (a[i] ~ b[i - s])."""
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    shifts = np.arange(-max_shift, max_shift + 1)
    cc = np.array([np.sum(a[max(0, s):a.size + min(0, s)] * b[max(0, -s):b.size - max(0, s)])
                   for s in shifts])
    k = int(np.argmax(cc))
    if 0 < k < cc.size - 1:
        d = cc[k - 1] - 2 * cc[k] + cc[k + 1]
        if d != 0:
            return float(shifts[k] + 0.5 * (cc[k - 1] - cc[k + 1]) / d)
    return float(shifts[k])


def match_lines(peak_px: Sequence[float], predicted_px: Sequence[float], tol_px: float):
    """Pair each predicted line with the nearest detected peak within tol_px (one-to-one)."""
    peaks = np.asarray(peak_px, dtype=float)
    pairs, used = [], set()
    for j, p in sorted(enumerate(predicted_px), key=lambda t: t[1]):
        if not np.isfinite(p) or peaks.size == 0:
            continue
        d = np.abs(peaks - p)
        k = int(np.argmin(d))
        if d[k] <= tol_px and k not in used:
            used.add(k)
            pairs.append((j, k, float(peaks[k] - p)))
    return pairs


def best_offset_scale(peak_px: Sequence[float], predicted_px: Sequence[float], center: float,
                      search_px: float, tol_px: float, max_scale_error: float = 0.15,
                      scale_step: float = 0.005):
    """Coarse alignment of a line list: offset and dispersion scale about ``center``
    that pair the most predicted lines with detected peaks.

    pred' = center + s * (pred - center) + offset. Returns (offset, scale, n paired).
    """
    peaks = np.asarray(peak_px, dtype=float)
    pred = np.asarray([p for p in predicted_px if np.isfinite(p)], dtype=float)
    if peaks.size == 0 or pred.size == 0:
        return 0.0, 1.0, 0
    best = (0.0, 1.0, -1, np.inf)
    for s in np.arange(1 - max_scale_error, 1 + max_scale_error + 1e-9, scale_step):
        ps = center + s * (pred - center)
        cands = (peaks[:, None] - ps[None, :]).ravel()
        cands = cands[np.abs(cands) <= search_px]
        if cands.size == 0:
            continue
        # (candidate, peak, line) distances; a line counts if any peak is within tol
        d = np.abs(peaks[None, :, None] - (ps[None, None, :] + cands[:, None, None]))
        dmin = d.min(axis=1)
        hit = dmin <= tol_px
        n = hit.sum(axis=1)
        err = np.where(hit, dmin, 0).sum(axis=1) / np.maximum(n, 1)
        k = int(np.lexsort((err, -n))[0])
        if n[k] > best[2] or (n[k] == best[2] and err[k] < best[3]):
            best = (float(cands[k]), float(s), int(n[k]), float(err[k]))
    return best[0], best[1], best[2]


def best_offset(peak_px: Sequence[float], predicted_px: Sequence[float], search_px: float,
                tol_px: float) -> float:
    """Global offset that pairs the most predicted lines with peaks (coarse alignment)."""
    peaks = np.asarray(peak_px, dtype=float)
    pred = np.asarray([p for p in predicted_px if np.isfinite(p)], dtype=float)
    if peaks.size == 0 or pred.size == 0:
        return 0.0
    cands = (peaks[:, None] - pred[None, :]).ravel()
    cands = cands[np.abs(cands) <= search_px]
    best, best_n, best_err = 0.0, -1, np.inf
    for c in np.concatenate([[0.0], cands]):
        pairs = match_lines(peaks, pred + c, tol_px)
        err = float(np.mean([abs(d) for *_, d in pairs])) if pairs else np.inf
        if len(pairs) > best_n or (len(pairs) == best_n and err < best_err):
            best, best_n, best_err = float(c), len(pairs), err
    return best
