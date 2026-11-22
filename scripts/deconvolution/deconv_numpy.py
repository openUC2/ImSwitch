"""3D deconvolution in plain numpy/scipy: Richardson-Lucy and a port of DeconvOptim.jl's map_cj_lean.

No ImSwitch imports. Arrays are (z, y, x) float32. A PSF is centred at index (0, 0, 0) and may be
smaller than the data. `padding` is one fraction per axis, as in DeconvOptim.jl: the grid grows by
that fraction and the border is left unconstrained by the data, which removes FFT wrap-around
(a wide-field stack needs it along z, e.g. (0.25, 0, 0)).

map_lean follows DeconvOptim.jl (branch new_methods, src/map_cj_lean.jl, MIT licence): Poisson
deviance, Good's roughness, object = x**2, Fletcher-Reeves conjugate gradients, one Newton step
per iteration with Armijo backtracking. Three FFT convolutions per iteration.
"""
import numpy as np
from scipy import fft

WORKERS = 4       # FFT threads; a Raspberry Pi 5 has four cores
EPS = np.float32(1e-7)


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


# --- grid helpers -----------------------------------------------------------------------------

def _padded_shape(shape, padding):
    return tuple(n if (n == 1 or p == 0) else fft.next_fast_len(n + max(4, 2 * round(n * abs(p))), real=True)
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


def map_lean(measured, psf, gamma=0.0, iterations=60, background=0.0, padding=(0, 0, 0),
             weights=(1, 1, 1), saddle_escape=1.0, escape_threshold=1e-4, callback=None):
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
        return float(np.sum(deviance, where=mask, dtype=np.float64)) + (gamma * _good(x, weights) if gamma else 0.0)

    mu = conv(x * x)
    psi = objective(x, mu)
    direction = np.zeros_like(x)
    gnorm_prev, alpha_prev, restart = 0.0, 1.0, True
    for it in range(1, iterations + 1):
        mt = np.maximum(mu + b, EPS)
        g_f = conv(np.where(mask, 1 - data / mt, 0).astype(np.float32), True)      # object-space gradient
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
                direction += np.where(x < 0, -1, 1) * np.float32(saddle_escape * rms) * np.sqrt(pull / pull.max())

        dphi = float(np.sum(grad * direction, dtype=np.float64))
        h = conv(2 * x * direction)
        curvature = float(np.sum(2 * g_f * direction * direction, dtype=np.float64))
        curvature += float(np.sum(np.where(mask, data / (mt * mt), 0) * h * h, dtype=np.float64))
        if gamma:
            curvature += gamma * 2 * _good(direction, weights)
        alpha = -dphi / curvature if (np.isfinite(curvature) and curvature > 0) else 2 * alpha_prev
        if not (np.isfinite(alpha) and alpha > 0):
            alpha = alpha_prev

        x += np.float32(alpha) * direction                 # Armijo backtracking, one convolution per trial
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
        else:                                              # no decrease: undo, restart with steepest descent
            x -= np.float32(alpha) * direction
            mu = conv(x * x)
            if steepest:
                break
            restart = True
        if callback is not None and callback(it, (x * x)[centre]):
            break
    return (x * x)[centre]


def demo():
    """Self-check: both solvers sharpen a blurred point pair and keep it non-negative."""
    shape = (16, 48, 48)
    psf = widefield_psf(shape, 0.12, 0.5, 0.75)
    truth = np.zeros(shape, np.float32)
    truth[8, 24, 20] = truth[8, 24, 28] = 5000
    blurred = fft.irfftn(fft.rfftn(truth) * fft.rfftn(psf), s=shape) + 2
    measured = np.random.default_rng(0).poisson(np.maximum(blurred, 0)).astype(np.float32)

    def dip(a):                                                    # contrast between the two points
        return a[8, 24, 24] / max(a[8, 24, 20], 1e-9)

    for solve in (richardson_lucy, lambda m, p, **k: map_lean(m, p, gamma=1e-3, **k)):
        out = solve(measured, psf, iterations=30, background=2.0)
        assert out.shape == shape and out.min() >= 0 and np.isfinite(out).all()
        assert dip(out) < 0.7 * dip(measured), (dip(out), dip(measured))
    padded = map_lean(measured, psf, iterations=5, background=2.0, padding=(0.25, 0, 0))
    assert padded.shape == shape
    print("deconv_numpy self-check passed")


if __name__ == "__main__":
    demo()
