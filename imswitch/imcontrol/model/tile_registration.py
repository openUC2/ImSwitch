"""Tile registration for a finished tile scan: where did the tiles really go?

Neighbouring tiles are registered by phase correlation (coarse full-tile shift
unwrapped towards the commanded offset, refined on the overlap), a weighted
least-squares solve gives every tile's real position, and the result is
compared with the commanded grid. Ported from the openUC2 PCB-stage bench
(bench/e6.py), where it recovered tile positions to ~3 µm on a stage that
misplaced them by up to ~0.4 mm.

Used by the ShitScope scan (model/shitscope_scan.py). No ImSwitch imports,
so it can be tested and reused standalone.
"""
from __future__ import annotations

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage.registration import phase_cross_correlation

OUTLIER_PX = 15          # pair residual after the solve that marks a bad match
MIN_QUALITY = 0.2        # normalised overlap correlation below which a pair is ignored

# ---------------------------------------------------------------- registration
def _pcc(a, b):
    s, _, _ = phase_cross_correlation(a, b, upsample_factor=10, normalization=None)
    return np.array(s, float)


def register_pair(a: np.ndarray, b: np.ndarray, expected) -> Tuple[np.ndarray, float]:
    """Offset (dy, dx) in px of tile b relative to tile a (sign convention of
    ``phase_cross_correlation(a, b)``), given the commanded offset `expected`.
    Returns (offset, quality); quality is the normalised overlap correlation."""
    expected = np.asarray(expected, float)
    n = np.array(a.shape, float)
    s = _pcc(a, b)
    cand = s[None, :] + np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)]) * n
    coarse = cand[np.argmin(np.hypot(*(cand - expected).T))]
    p = np.round(coarse).astype(int)
    h, w = a.shape
    if abs(p[0]) >= h - 32 or abs(p[1]) >= w - 32:
        return coarse, 0.0
    # b shifted by +p aligns with a: a[y, x] ~ b[y - p0, x - p1]
    A = a[max(0, p[0]):h + min(0, p[0]), max(0, p[1]):w + min(0, p[1])]
    B = b[max(0, -p[0]):h - max(0, p[0]), max(0, -p[1]):w - max(0, p[1])]
    r = _pcc(A, B)
    ri = np.round(r).astype(int)
    A2 = A[max(0, ri[0]):A.shape[0] + min(0, ri[0]), max(0, ri[1]):A.shape[1] + min(0, ri[1])]
    B2 = B[max(0, -ri[0]):B.shape[0] - max(0, ri[0]), max(0, -ri[1]):B.shape[1] - max(0, ri[1])]
    A2 = A2 - A2.mean(); B2 = B2 - B2.mean()
    q = float((A2 * B2).sum() / (np.sqrt((A2 ** 2).sum() * (B2 ** 2).sum()) + 1e-9))
    return p + r, q


def solve_positions(n_tiles: int, pairs, anchor: int = 0):
    """Weighted least-squares positions from pairs [(i, j, offset, weight)]."""
    rows, rhs, wts = [], [], []
    for i, j, off, w in pairs:
        r = np.zeros(n_tiles); r[j] = 1; r[i] = -1
        rows.append(r); rhs.append(off); wts.append(w)
    r0 = np.zeros(n_tiles); r0[anchor] = 1
    A = np.vstack(rows + [r0]); b = np.vstack(rhs + [np.zeros(2)])
    w = np.sqrt(np.array(wts + [1e3]))
    pos, *_ = np.linalg.lstsq(A * w[:, None], b * w[:, None], rcond=None)
    res = np.array([np.hypot(*(pos[j] - pos[i] - off)) for i, j, off, _ in pairs])
    return pos, res


def robust_solve(n_tiles: int, pairs, thresh: float = OUTLIER_PX, anchor: int = 0):
    """Solve, drop the worst pair while its residual exceeds `thresh`, re-solve."""
    keep = list(pairs)
    while True:
        pos, res = solve_positions(n_tiles, keep, anchor)
        if not len(res):
            return pos, res, keep
        k = int(np.argmax(res))
        if res[k] <= thresh or len(keep) <= 1:
            return pos, res, keep
        keep.pop(k)


def connected(n_tiles: int, pairs, start: int = 0) -> List[int]:
    comp, grow = {start}, True
    while grow:
        grow = False
        for i, j, *_ in pairs:
            if (i in comp) != (j in comp):
                comp |= {i, j}; grow = True
    return sorted(comp)


# ---------------------------------------------------------------- grid analysis
def grid_indices(xy_um: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Integer grid indices from commanded positions; returns (idx, step_x, step_y)."""
    idx = np.zeros((len(xy_um), 2), int); steps = []
    for ax in (0, 1):
        v = np.sort(np.unique(np.round(xy_um[:, ax], 1)))
        d = np.diff(v)
        step = float(np.median(d[d > 1.0])) if np.any(d > 1.0) else 1.0
        idx[:, ax] = np.round((xy_um[:, ax] - v.min()) / step).astype(int)
        steps.append(step)
    return idx, steps[0], steps[1]


def analyse_tiles(tiles: Sequence[np.ndarray], xy_um: np.ndarray, um_per_px: float,
                  order: Optional[Sequence[int]] = None) -> Dict:
    """Register a tile grid. tiles: 2-D arrays (same shape); xy_um: commanded
    (x, y) per tile in µm; order: scan order (defaults to list order).
    Returns summary + per-tile commanded/measured positions (µm)."""
    n = len(tiles)
    order = list(order) if order is not None else list(range(n))
    xy_um = np.asarray(xy_um, float)
    tiles = [np.asarray(t, float) for t in tiles]
    # flat field from <= 32 tiles spread over the scan (median over all is slow and no better)
    flat = np.median(np.stack(tiles[::max(1, n // 32)]), axis=0); flat /= max(flat.mean(), 1e-9)
    tiles = [t / np.maximum(flat, 1e-3) for t in tiles]
    from scipy.ndimage import gaussian_filter
    tex = np.array([np.std(t - gaussian_filter(t, 8)) for t in tiles])   # fine texture after flat-field
    bright = np.array([t.mean() for t in tiles])
    usable = (tex > 0.3 * np.median(tex)) & (bright > 0.3 * np.median(bright))

    idx, step_x, step_y = grid_indices(xy_um)
    cmd_px = np.c_[xy_um[:, 1], xy_um[:, 0]] / um_per_px          # (y, x)
    at = {tuple(g): k for k, g in enumerate(idx)}
    horiz = [(k, at[(g[0] + 1, g[1])]) for k, g in enumerate(idx) if (g[0] + 1, g[1]) in at]
    vert = [(k, at[(g[0], g[1] + 1)]) for k, g in enumerate(idx) if (g[0], g[1] + 1) in at]

    # image orientation per axis: +stage may move the image either way (camera flip)
    def best_sign(pairs_, axis):
        test = [(i, j) for i, j in pairs_ if usable[i] and usable[j]][:6]
        if not test:
            return 1.0
        score = {}
        for s in (1.0, -1.0):
            e = lambda i, j: (cmd_px[j] - cmd_px[i]) * (np.array([1, s]) if axis == "x" else np.array([s, 1]))
            score[s] = np.mean([register_pair(tiles[i], tiles[j], e(i, j))[1] for i, j in test])
        return max(score, key=score.get)

    sx, sy = best_sign(horiz, "x"), best_sign(vert, "y")
    sign = np.array([sy, sx])
    cmd_img = cmd_px * sign                                         # commanded, in image orientation

    todo = [(i, j) for i, j in horiz + vert if usable[i] and usable[j]]
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:    # FFTs release the GIL
        found = ex.map(lambda ij: register_pair(tiles[ij[0]], tiles[ij[1]], cmd_img[ij[1]] - cmd_img[ij[0]]), todo)
        pairs = [(i, j, off, q) for (i, j), (off, q) in zip(todo, found) if q > MIN_QUALITY]
    anchor = int(np.argmax(usable)) if usable.any() else 0
    comp = connected(n, pairs, anchor)
    pairs = [p for p in pairs if p[0] in comp]
    pos_img, res, kept = robust_solve(n, pairs, anchor=anchor)
    # back to stage orientation, anchored to the commanded position of the anchor tile
    pos = (pos_img - pos_img[anchor]) * sign + cmd_px[anchor]
    inc = np.array(comp)
    err_um = (pos - cmd_px)[inc] * um_per_px                         # (y, x) µm

    C, P = cmd_px[inc] * um_per_px, pos[inc] * um_per_px
    A = np.c_[C, np.ones(len(C))]
    summary = dict(tiles=n, tiles_usable=int(usable.sum()), tiles_solved=len(inc),
                   pairs_used=len(kept), step_x_um=step_x, step_y_um=step_y,
                   image_sign_x=int(sx), image_sign_y=int(sy),
                   pair_rms_um=float(np.sqrt(np.mean(res ** 2)) * um_per_px) if len(res) else None,
                   max_err_um=float(np.hypot(*err_um.T).max()) if len(inc) else None,
                   rms_err_um=float(np.sqrt(np.mean(np.sum(err_um ** 2, 1)))) if len(inc) else None)
    if len(inc) >= 4:
        M, *_ = np.linalg.lstsq(A, P, rcond=None)
        r = P - A @ M
        summary.update(scale_x=float(np.hypot(*M[1])), scale_y=float(np.hypot(*M[0])),
                       rotation_deg=float(np.degrees(np.arctan2(M[1][0], M[1][1]))),
                       non_orthogonality_deg=float(np.degrees(np.arctan2(M[1][0], M[1][1]) + np.arctan2(M[0][1], M[0][0]))),
                       affine_resid_rms_um=float(np.sqrt(np.mean(np.sum(r ** 2, 1)))))
    # per-step statistics in scan order, and the minimum overlap / holes
    H, W = tiles[0].shape
    rank = {k: r for r, k in enumerate(order)}
    solved = set(inc.tolist())
    sx_steps, sy_steps = [], []
    for a, b in horiz:
        if a in solved and b in solved:
            sx_steps.append(abs(pos[b][1] - pos[a][1]) * um_per_px)
    for a, b in vert:
        if a in solved and b in solved:
            sy_steps.append(abs(pos[b][0] - pos[a][0]) * um_per_px)
    if sx_steps:
        summary.update(measured_step_x_um=float(np.mean(sx_steps)), measured_step_x_sd=float(np.std(sx_steps)),
                       min_overlap_x=float(1 - max(sx_steps) / (W * um_per_px)))
    if sy_steps:
        summary.update(measured_step_y_um=float(np.mean(sy_steps)), measured_step_y_sd=float(np.std(sy_steps)),
                       min_overlap_y=float(1 - max(sy_steps) / (H * um_per_px)))
    if len(inc):
        g = 4
        lo = pos[inc].min(0); hi = pos[inc].max(0) + [H, W]
        cov = np.zeros((int((hi[0] - lo[0]) // g) + 1, int((hi[1] - lo[1]) // g) + 1), bool)
        for p in pos[inc]:
            y, x = ((p - lo) // g).astype(int); cov[y:y + H // g, x:x + W // g] = True
        summary["holes_pct"] = float(100 * (1 - cov.mean()))
    return dict(summary=summary, solved=inc.tolist(), usable=usable.tolist(),
                commanded_um=(cmd_px * um_per_px)[:, ::-1].tolist(),     # (x, y)
                measured_um=(pos * um_per_px)[:, ::-1].tolist(),
                order=[rank.get(k, k) for k in range(n)], _tiles=tiles, _pos_px=pos, _cmd_px=cmd_px,
                _sign=sign)


# ---------------------------------------------------------------- rendering
def _png(fig) -> str:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mosaic(tiles, pos_px, sign, scale: int = 4) -> np.ndarray:
    """Place tiles at stage positions (px); sign maps stage to image direction."""
    pos = np.asarray(pos_px) * sign
    pos = pos - pos.min(0)
    H, W = tiles[0].shape
    h, w = int(pos[:, 0].max() + H) // scale + 1, int(pos[:, 1].max() + W) // scale + 1
    acc, cnt = np.zeros((h, w)), np.zeros((h, w))
    for t, p in zip(tiles, pos):
        s = t[::scale, ::scale]; y, x = (p // scale).astype(int)
        acc[y:y + s.shape[0], x:x + s.shape[1]] += s; cnt[y:y + s.shape[0], x:x + s.shape[1]] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def render(result: Dict, um_per_px: float) -> Dict[str, str]:
    """Error map + mosaics (commanded vs measured) as PNG data URLs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inc = np.array(result["solved"], int)
    cmd = np.array(result["commanded_um"]); meas = np.array(result["measured_um"])
    out = {}
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if len(inc):
        e = meas[inc] - cmd[inc]
        q = ax.quiver(cmd[inc, 0] / 1000, cmd[inc, 1] / 1000, e[:, 0], e[:, 1], np.hypot(*e.T),
                      angles="xy", scale_units="xy", scale=1000, cmap="viridis")
        plt.colorbar(q, label="|measured − commanded| (µm)")
    miss = np.setdiff1d(np.arange(len(cmd)), inc)
    if len(miss):
        ax.plot(cmd[miss, 0] / 1000, cmd[miss, 1] / 1000, "x", color="0.6", label="not registered")
        ax.legend(fontsize=7)
    ax.set(aspect="equal", xlabel="X (mm)", ylabel="Y (mm)", title="Tile error (arrows 1:1)")
    ax.invert_yaxis()
    out["error_map"] = _png(fig); plt.close(fig)
    tiles, sign = result["_tiles"], result["_sign"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for a, t, p, title in ((axs[0], tiles, result["_cmd_px"], "commanded positions"),
                           (axs[1], [tiles[k] for k in inc], result["_pos_px"][inc], "measured positions")):
        if len(t):
            m = mosaic(t, p, sign)
            lo, hi = np.nanpercentile(m, [1, 99])
            a.imshow(m, cmap="gray", vmin=lo, vmax=hi)
        a.set_title(title); a.axis("off")
    out["mosaic"] = _png(fig); plt.close(fig)
    return out
