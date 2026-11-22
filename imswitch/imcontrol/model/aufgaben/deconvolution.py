"""3D deconvolution: a stack, the optics and the parameters in, the deconvolved stack out.

The recipe of docs/DECONVOLUTION.md. numpy and scipy only, no ImSwitch imports: the same file runs
in ImSwitch (preview and small stacks on the Pi), in the processing worker (``worker.py``,
docker/processing-worker) and in scripts/deconvolution/benchmark.py.

    out = deconvolve(stack, Optics(pixel_um=0.12, z_step_um=0.5, na=0.75), Parameters())

Two solvers. ``rl`` is Richardson-Lucy. ``map`` ports DeconvOptim.jl's map_cj_lean (branch
new_methods, src/map_cj_lean.jl, MIT licence): Poisson deviance plus Good's roughness,
object = x**2, Fletcher-Reeves conjugate gradients, one Newton step per iteration with Armijo
backtracking, three FFT convolutions per iteration. test_deconvolution.py checks it against stored
DeconvOptim.jl output.

Arrays are (z, y, x) float32 and a PSF is centred at index (0, 0, 0). ``padding`` is one fraction
per axis, as in DeconvOptim.jl: the grid grows by that fraction and the border is left unconstrained
by the data, which removes FFT wrap-around (wide-field light leaves the stack along z). Planes wider
than ``Parameters.tile`` are deconvolved in tiles that overlap by the PSF's reach.
"""
import itertools
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import fft, ndimage

NAME, VERSION = "deconvolution", "1"
WORKERS = os.cpu_count() or 4        # FFT threads
EPS = np.float32(1e-7)
PADDING = (0.25, 0, 0)               # z only: light from planes outside the stack
METHODS = ("map", "rl")
MODALITIES = ("widefield", "lightsheet")
LIMITS = {"iterations": (1, 500), "regularization": (0.0, 10.0), "tile": (64, 16384)}


class Cancelled(Exception):
    """The progress callback asked deconvolve() to stop."""


def _check(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass
class Optics:
    """What the PSF is computed from. Lengths in µm, in the sample."""
    pixel_um: float
    z_step_um: float = 1.0
    na: float = 0.75
    emission_um: float = 0.52
    immersion_n: float = 1.0          # air 1.0, water 1.33, oil 1.515
    modality: str = "widefield"       # or "lightsheet"
    sheet_um: float = 2.0             # light-sheet thickness (FWHM along z); light sheet only

    def __post_init__(self):
        for name in ("pixel_um", "z_step_um", "na", "emission_um", "immersion_n", "sheet_um"):
            value = float(getattr(self, name))
            _check(math.isfinite(value) and value > 0, f"{name} must be a positive number")
            setattr(self, name, value)
        _check(1.0 <= self.immersion_n <= 2.0, "immersion_n must be between 1 and 2")
        _check(self.na < self.immersion_n, f"NA {self.na} must be below the immersion index "
                                           f"{self.immersion_n}")
        _check(0.2 <= self.emission_um <= 2.0, "emission_um must be between 0.2 and 2 µm")
        _check(self.modality in MODALITIES, f"modality must be one of {MODALITIES}")


@dataclass
class Parameters:
    method: str = "map"                   # "map" (regularised) or "rl" (Richardson-Lucy, fast)
    iterations: int = 30
    regularization: float = 1e-4          # weight of Good's roughness in measured counts; map only
    background: Optional[float] = None    # camera offset + dark counts; None: estimated
    tile: int = 512                       # planes wider than this are split into tiles; a
                                          # 64-plane tile then needs about 3 GB

    def __post_init__(self):
        _check(self.method in METHODS, f"method must be one of {METHODS}")
        self.iterations, self.tile = int(self.iterations), int(self.tile)
        self.regularization = float(self.regularization)
        for name, (lo, hi) in LIMITS.items():
            _check(lo <= getattr(self, name) <= hi, f"{name} must be between {lo} and {hi}")
        if self.background is not None:
            self.background = float(self.background)
            _check(math.isfinite(self.background), "background must be a number")


def parse(cls, data):
    """Optics or Parameters from a dict (a request body); ValueError for unknown or bad fields."""
    try:
        return cls(**(data or {}))
    except TypeError as e:
        raise ValueError(str(e)) from e


def emission_um(channel: str) -> float:
    """A guess at the emission wavelength from a channel name (excitation line or dye)."""
    name = (channel or "").lower()
    for keys, um in ((("405", "dapi", "hoechst"), 0.46), (("488", "gfp", "fitc", "alexa488"), 0.52),
                     (("561", "532", "rfp", "mcherry", "tritc", "cy3"), 0.60),
                     (("635", "640", "cy5", "alexa647"), 0.67)):
        if any(k in name for k in keys):
            return um
    return 0.52


# --- point spread functions -------------------------------------------------------------------

def widefield_psf(shape, px_um, dz_um, na, wavelength_um=0.52, n=1.0):
    """Scalar wide-field PSF |ifft2(pupil * defocus)|**2 per plane, sum 1."""
    nz, ny, nx = shape
    fy, fx = np.meshgrid(fft.fftfreq(ny, px_um), fft.fftfreq(nx, px_um), indexing="ij")
    kr2 = fx ** 2 + fy ** 2
    pupil = kr2 <= (na / wavelength_um) ** 2
    kz = np.sqrt(np.maximum((n / wavelength_um) ** 2 - kr2, 0))
    z = fft.fftfreq(nz, 1 / nz) * dz_um                       # planes 0, 1, .., -1
    field = fft.ifft2(pupil * np.exp(2j * np.pi * kz * z[:, None, None]), workers=WORKERS)
    psf = np.abs(field) ** 2
    return (psf / psf.sum()).astype(np.float32)


def lightsheet_psf(shape, px_um, dz_um, na, sheet_fwhm_um, **kw):
    """Detection PSF times a Gaussian light sheet of the given thickness (FWHM along z), sum 1."""
    psf = widefield_psf(shape, px_um, dz_um, na, **kw).astype(np.float64)
    z = fft.fftfreq(shape[0], 1 / shape[0]) * dz_um
    psf *= np.exp(-4 * np.log(2) * (z / sheet_fwhm_um) ** 2)[:, None, None]
    return (psf / psf.sum()).astype(np.float32)


def psf(optics: Optics, shape) -> np.ndarray:
    """The PSF of ``optics`` on a (z, y, x) grid of ``shape``, centred at (0, 0, 0), sum 1."""
    o = optics
    if o.modality == "lightsheet":
        return lightsheet_psf(shape, o.pixel_um, o.z_step_um, o.na, o.sheet_um,
                              wavelength_um=o.emission_um, n=o.immersion_n)
    return widefield_psf(shape, o.pixel_um, o.z_step_um, o.na, o.emission_um, o.immersion_n)


def overlap_px(optics: Optics, nz: int) -> int:
    """How far (pixels) light spreads sideways within the stack: the margin a tile needs.

    The defocus cone over half the stack depth (half the sheet for a light sheet) plus two Airy
    radii.
    """
    depth = nz * optics.z_step_um / 2
    if optics.modality == "lightsheet":
        depth = min(depth, optics.sheet_um)
    cone = math.tan(math.asin(min(optics.na / optics.immersion_n, 0.99)))
    return int(math.ceil((depth * cone + optics.emission_um / optics.na) / optics.pixel_um))


def estimate_background(stack) -> float:
    """Camera offset plus dark counts: the 5th percentile after 8 x 8 lateral averaging.

    The averaging beats the noise. Too high when the sample fills the whole field; set the value
    by hand then.
    """
    planes = np.asarray(stack[::max(1, len(stack) // 8)], np.float32)
    return float(np.percentile(ndimage.uniform_filter(planes, size=(1, 8, 8)), 5))


# --- grid helpers -----------------------------------------------------------------------------

def _padded_shape(shape, padding):
    return tuple(n if (n == 1 or p == 0)
                 else fft.next_fast_len(n + max(4, 2 * round(n * abs(p))), real=True)
                 for n, p in zip(shape, padding))


def _centre(big, small):
    return tuple(slice(b // 2 - s // 2, b // 2 - s // 2 + s) for b, s in zip(big, small))


def _setup(measured, psf, padding):
    """Data on the padded grid (border = -1), its mask, and conv(a, adjoint) with the PSF."""
    g = np.asarray(measured, np.float32)
    big = _padded_shape(g.shape, padding)
    data = np.full(big, -1, np.float32)
    data[_centre(big, g.shape)] = g
    kernel = np.zeros(big, np.float32)
    kernel[_centre(big, psf.shape)] = fft.fftshift(psf)
    otf = fft.rfftn(fft.ifftshift(kernel), workers=WORKERS) / np.float32(psf.sum())

    def conv(a, adjoint=False):
        return fft.irfftn(fft.rfftn(a, workers=WORKERS) * (np.conj(otf) if adjoint else otf),
                          s=big, workers=WORKERS)
    return g, data, data >= 0, conv


def _tiles(n, tile, margin):
    """(window start, window stop, core start, core stop) along one axis; windows of one length.

    Cores tile [0, n). Each window holds its core plus up to ``margin`` on both sides, shifted
    inward at the ends, so every tile uses the same PSF.
    """
    if n <= tile:
        return [(0, n, 0, n)]
    size = min(n, tile + 2 * margin)
    out = []
    for c0 in range(0, n, tile):
        w0 = min(max(c0 - margin, 0), n - size)
        out.append((w0, w0 + size, c0, min(c0 + tile, n)))
    return out


# --- solvers ----------------------------------------------------------------------------------

def richardson_lucy(measured, psf, iterations=30, background=0.0, padding=(0, 0, 0), callback=None):
    """Classic Richardson-Lucy with a constant background and an unconstrained padding border."""
    g, data, mask, conv = _setup(measured, psf, padding)
    b = np.float32(background)
    ratio = np.zeros_like(data)
    norm = np.maximum(conv(mask.astype(np.float32), True), EPS)
    est = np.full(data.shape, max(float(g.mean()) - background, 1e-7), np.float32)
    centre = _centre(data.shape, g.shape)
    for it in range(1, iterations + 1):
        np.divide(data, np.maximum(conv(est) + b, EPS), out=ratio, where=mask)
        est *= conv(ratio, True) / norm
        if callback is not None and callback(it, est[centre]):
            break
    return est[centre]


def _good(x, weights):
    """Good's roughness 4 * sum_d w_d * sum (x[n+1] - x[n])**2."""
    return 4 * sum(w * float(np.sum(np.diff(x, axis=d) ** 2, dtype=np.float64))
                   for d, w in enumerate(weights))


def _laplacian(x, weights):
    """Weighted Laplacian with Neumann borders."""
    lap = np.zeros_like(x)
    for d, w in enumerate(weights):
        diff = np.float32(w) * np.diff(x, axis=d)
        hi = [slice(None)] * x.ndim
        lo = [slice(None)] * x.ndim
        hi[d], lo[d] = slice(1, None), slice(None, -1)
        lap[tuple(lo)] += diff
        lap[tuple(hi)] -= diff
    return lap


def map_lean(measured, psf, gamma=0.0, iterations=60, background=0.0, padding=(0, 0, 0),  # noqa: C901
             weights=(1, 1, 1), saddle_escape=1.0, escape_threshold=1e-4, callback=None):
    # one function, as in map_cj_lean.jl, so the two can be read side by side
    """MAP deconvolution, Poisson noise + gamma * Good's roughness, object = x**2 >= 0.

    gamma is in units of the measured counts (the data are not normalised). Returns the object.
    callback(iteration, object) may return True to stop early.
    """
    g, data, mask, conv = _setup(measured, psf, padding)
    b = np.float32(background)
    centre = _centre(data.shape, g.shape)
    x = np.full(data.shape, np.sqrt(max(float(g.mean()) - background, 1e-7)), np.float32)
    logdata = np.log(np.where(data > 0, data, 1))

    def objective(x, mu):
        mt = np.maximum(mu + b, EPS)
        deviance = np.where(data > 0, mt - data + data * (logdata - np.log(mt)), mt)
        return (float(np.sum(deviance, where=mask, dtype=np.float64))
                + (gamma * _good(x, weights) if gamma else 0.0))

    mu = conv(x * x)
    psi = objective(x, mu)
    direction = np.zeros_like(x)
    gnorm_prev, alpha_prev, restart = 0.0, 1.0, True
    for it in range(1, iterations + 1):
        mt = np.maximum(mu + b, EPS)
        # gradient in object space
        g_f = conv(np.where(mask, 1 - data / mt, 0).astype(np.float32), True)
        grad = 2 * x * g_f
        if gamma:
            grad -= np.float32(8 * gamma) * _laplacian(x, weights)
        gnorm = float(np.sum(grad * grad, dtype=np.float64))
        beta = 0.0 if (restart or gnorm_prev == 0) else gnorm / gnorm_prev
        direction = np.float32(beta) * direction - grad
        steepest = beta == 0
        if float(np.sum(grad * direction, dtype=np.float64)) >= 0:
            direction, steepest = -grad, True
        gnorm_prev, restart = gnorm, False

        if saddle_escape > 0:       # voxels stuck at x = 0 although the data ask them to grow
            shift = 8 * gamma * sum(weights)
            pull = np.where((x * x <= escape_threshold * float((x * x).max())) & (g_f + shift < 0),
                            -(g_f + shift), 0)
            rms = float(np.sqrt(np.mean(direction * direction, dtype=np.float64)))
            if pull.max() > 0 and rms > 0:
                direction += (np.where(x < 0, -1, 1) * np.float32(saddle_escape * rms)
                              * np.sqrt(pull / pull.max()))

        dphi = float(np.sum(grad * direction, dtype=np.float64))
        h = conv(2 * x * direction)
        curvature = float(np.sum(2 * g_f * direction * direction, dtype=np.float64))
        curvature += float(np.sum(np.where(mask, data / (mt * mt), 0) * h * h, dtype=np.float64))
        if gamma:
            curvature += gamma * 2 * _good(direction, weights)
        alpha = -dphi / curvature if (np.isfinite(curvature) and curvature > 0) else 2 * alpha_prev
        if not (np.isfinite(alpha) and alpha > 0):
            alpha = alpha_prev

        x += np.float32(alpha) * direction        # Armijo backtracking, one convolution per trial
        mu = conv(x * x)
        psi_new = objective(x, mu)
        for _ in range(20):
            if np.isfinite(psi_new) and psi_new <= psi + 1e-4 * alpha * dphi:
                break
            alpha /= 2
            x -= np.float32(alpha) * direction
            mu = conv(x * x)
            psi_new = objective(x, mu)
        if np.isfinite(psi_new) and psi_new < psi:
            psi, alpha_prev = psi_new, alpha
        else:                                     # no decrease: undo, restart with steepest descent
            x -= np.float32(alpha) * direction
            mu = conv(x * x)
            if steepest:
                break
            restart = True
        if callback is not None and callback(it, (x * x)[centre]):
            break
    return (x * x)[centre]


# --- the recipe -------------------------------------------------------------------------------

def deconvolve(stack, optics: Optics, parameters: Optional[Parameters] = None, progress=None):
    """Deconvolve a (z, y, x) stack, or one (y, x) plane; float32 of the same shape.

    ``progress(fraction)`` is called after every iteration; when it returns True the run stops
    and Cancelled is raised. ValueError for data or settings it cannot use.
    """
    p = parameters or Parameters()
    data = np.asarray(stack, np.float32)
    plane = data.ndim == 2
    data = data[None] if plane else data
    _check(data.ndim == 3 and data.size > 0, f"expected a (z, y, x) stack, got shape {data.shape}")
    _check(bool(np.isfinite(data).all()), "the stack holds NaN or infinite values")
    background = estimate_background(data) if p.background is None else p.background
    nz, ny, nx = data.shape
    margin = overlap_px(optics, nz)
    ys, xs = _tiles(ny, p.tile, margin), _tiles(nx, p.tile, margin)
    kernel = psf(optics, (nz, ys[0][1] - ys[0][0], xs[0][1] - xs[0][0]))
    out = np.empty_like(data)
    tiles = list(itertools.product(ys, xs))
    for k, ((y0, y1, cy0, cy1), (x0, x1, cx0, cx1)) in enumerate(tiles):
        def step(it, _, k=k):
            if progress is not None and progress((k + it / p.iterations) / len(tiles)):
                raise Cancelled()
        window = data[:, y0:y1, x0:x1]
        if p.method == "rl":
            result = richardson_lucy(window, kernel, p.iterations, background, PADDING, step)
        else:
            result = map_lean(window, kernel, p.regularization, p.iterations, background, PADDING,
                              callback=step)
        out[:, cy0:cy1, cx0:cx1] = result[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out[0] if plane else out


def run(stack, spec: dict, progress=None) -> np.ndarray:
    """The recipe signature every runner calls: spec = {"optics": {..}, "parameters": {..}}."""
    return deconvolve(stack, parse(Optics, spec.get("optics")),
                      parse(Parameters, spec.get("parameters")), progress)


# --- test data --------------------------------------------------------------------------------

def synthetic(shape, optics: Optics, seed=1, photons=300.0, background=5.0):
    """(truth, measured): beads, filaments, a hollow shell and line pairs 3, 5 and 8 px apart,
    blurred with the PSF of ``optics`` and Poisson-sampled. The benchmark's and the Julia
    comparison's test object: keep it as it is, or the stored Julia output no longer matches.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    obj = np.zeros(shape, np.float32)
    for _ in range(40 * nx // 256):                                  # beads
        obj[rng.integers(nz // 4, 3 * nz // 4), rng.integers(8, ny - 8),
            rng.integers(8, nx - 8)] += rng.uniform(40, 120)
    t = np.linspace(0, 1, 4 * nx)
    for _ in range(6 * nx // 256):                       # filaments, slightly tilted in z
        p0 = np.array([rng.uniform(0.3, 0.7) * nz, rng.uniform(0.1, 0.9) * ny,
                       rng.uniform(0.1, 0.9) * nx])
        p1 = np.array([rng.uniform(0.3, 0.7) * nz, rng.uniform(0.1, 0.9) * ny,
                       rng.uniform(0.1, 0.9) * nx])
        z, y, x = (p0[:, None] + (p1 - p0)[:, None] * t).round().astype(int)
        obj[z, y, x] = 6
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]                             # hollow shell ("cell")
    r = np.sqrt(((zz - nz / 2) * optics.z_step_um / optics.pixel_um / 6) ** 2 + (yy - 0.3 * ny) ** 2
                + (xx - 0.7 * nx) ** 2)
    obj[np.abs(r - 0.12 * nx) < 1] += 2
    for i, gap in enumerate((3, 5, 8)):                              # line pairs
        x0 = int(0.2 * nx) + i * nx // 10
        obj[nz // 2, int(0.6 * ny):int(0.85 * ny), [x0, x0 + gap]] = 8
    big = (2 * nz, ny, nx)                               # light may leave the stack along z
    padded = np.zeros(big, np.float32)
    padded[nz // 2:nz // 2 + nz] = obj
    blurred = fft.irfftn(fft.rfftn(padded, workers=WORKERS)
                         * fft.rfftn(psf(optics, big), workers=WORKERS),
                         s=big, workers=WORKERS)[nz // 2:nz // 2 + nz]
    scale = photons / blurred.max()
    measured = rng.poisson(np.maximum(blurred * scale, 0) + background).astype(np.float32)
    return obj * scale, measured
