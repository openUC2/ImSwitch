"""The deconvolution recipe: same result as DeconvOptim.jl, tiles, PSF from the optics, refusals.

deconvolution_julia_reference.npz holds the z and y maximum projections (scaled to 1, float16) of
DeconvOptim.jl's map_cj_lean (branch new_methods, c72a2c8) on synthetic((32, 256, 256), FRAME)
with gamma 1e-4, 30 iterations, background 5 and z padding 0.25 (scripts/deconvolution).
"""
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from imswitch.imcontrol.model.aufgaben import deconvolution as d

FRAME = d.Optics(pixel_um=0.12, z_step_um=0.5, na=0.75)     # 20x/0.75, IMX178
REFERENCE = Path(__file__).with_name("deconvolution_julia_reference.npz")


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum(dtype=np.float64) / np.sqrt((a * a).sum(dtype=np.float64)
                                                         * (b * b).sum(dtype=np.float64)))


def test_matches_deconvoptim_jl():
    _, measured = d.synthetic((32, 256, 256), FRAME)
    result = d.deconvolve(measured, FRAME, d.Parameters(regularization=1e-4, background=5.0))
    julia = np.load(REFERENCE)
    # 0.995 / 0.996 on macOS and on a Raspberry Pi; the raw stack only reaches about 0.3
    assert ncc(result.max(0), julia["mip_z"].astype(np.float32)) > 0.99
    assert ncc(result.max(1), julia["mip_y"].astype(np.float32)) > 0.99
    assert ncc(measured.max(0), julia["mip_z"].astype(np.float32)) < 0.6


def test_tiles_give_the_whole_plane_result():
    truth, measured = d.synthetic((8, 256, 256), FRAME, seed=3)
    whole = d.deconvolve(measured, FRAME, d.Parameters(iterations=20))
    tiled = d.deconvolve(measured, FRAME, d.Parameters(iterations=20, tile=96))   # 3 x 3 tiles
    assert tiled.shape == whole.shape and ncc(tiled, whole) > 0.99
    assert ncc(tiled, truth) > ncc(measured, truth) + 0.3
    assert abs(d.estimate_background(measured) - 5.0) < 1.0


def test_psf_follows_the_optics():
    wide = d.psf(FRAME, (32, 64, 64))
    sheet = d.psf(d.Optics(pixel_um=0.12, z_step_um=0.5, na=0.75, modality="lightsheet",
                           sheet_um=1.0), (32, 64, 64))
    high_na = d.psf(d.Optics(pixel_um=0.12, z_step_um=0.5, na=1.2, immersion_n=1.33), (32, 64, 64))
    for p in (wide, sheet, high_na):
        assert abs(float(p.sum()) - 1) < 1e-4 and np.unravel_index(p.argmax(), p.shape) == (0, 0, 0)
    in_focus = lambda p: p[0].sum() / p.sum()     # noqa: E731
    assert in_focus(sheet) > 2 * in_focus(wide)   # the sheet keeps the light near the focal plane
    assert high_na[0, 0, 0] > wide[0, 0, 0]       # higher NA, tighter spot
    assert d.overlap_px(FRAME, 32) > d.overlap_px(FRAME, 8)
    assert d.emission_um("LED 488") == 0.52 and d.emission_um("Cy5") == 0.67


def centred(p):
    return np.fft.fftshift(p)


def bead_stack(optics, shape, beads, photons=3000.0, background=100.0, seed=0):
    """Point beads (z, y, x, brightness) at sub-pixel positions, blurred with the optics' PSF
    (``photons`` at the peak of a bead of brightness 1) and Poisson-sampled."""
    k = np.meshgrid(*(np.fft.fftfreq(n) for n in shape), indexing="ij", sparse=True)
    spectrum = sum(b * np.exp(-2j * np.pi * (k[0] * z + k[1] * y + k[2] * x))
                   for z, y, x, b in beads)
    kernel = d.psf(optics, shape)
    blurred = np.fft.ifftn(spectrum * np.fft.fftn(kernel)).real * photons / kernel.max()
    rng = np.random.default_rng(seed)
    return rng.poisson(np.maximum(blurred, 0) + background).astype(np.float32)


def test_psf_models():
    low = d.Optics(pixel_um=0.2, z_step_um=0.5, na=0.3)
    assert ncc(d.psf(low, (16, 64, 64)), d.psf(replace(low, psf_model="vectorial"),
                                                (16, 64, 64))) > 0.99   # equal at low NA
    vectorial = replace(FRAME, psf_model="vectorial")
    lateral, axial = d.widths(centred(d.psf(vectorial, (32, 96, 96))), vectorial)
    expected = d.expected_fwhm(FRAME)
    assert abs(lateral / expected[0] - 1) < 0.15 and abs(axial / expected[1] - 1) < 0.2
    gauss = replace(FRAME, psf_model="gaussian", fwhm_xy_um=0.6, fwhm_z_um=2.0)
    assert np.allclose(d.widths(centred(d.psf(gauss, (32, 64, 64))), gauss), (0.6, 2.0), rtol=0.05)
    with pytest.raises(ValueError):
        d.Optics(pixel_um=0.1, psf_model="airy")


def test_measured_psf_is_placed_resampled_and_sent():
    voxel = (0.5, 0.12, 0.12)
    true = centred(d.psf(FRAME, (33, 65, 65)))             # odd sizes: the centre is a voxel
    m = replace(FRAME, psf_model="measured")
    placed = d.psf(m, (32, 96, 96), measured=(true, voxel))
    assert abs(float(placed.sum()) - 1) < 1e-5
    assert np.unravel_index(placed.argmax(), placed.shape) == (0, 0, 0)
    fine = centred(d.psf(replace(FRAME, z_step_um=0.25), (65, 65, 65)))   # beads at 0.25 µm
    coarse = d.psf(m, (32, 64, 64), measured=(fine, (0.25, 0.12, 0.12)))
    assert ncc(coarse, d.psf(FRAME, (32, 64, 64))) > 0.95
    _, measured = d.synthetic((8, 64, 64), FRAME)
    spec = {"optics": asdict(m), "parameters": {"iterations": 3},
            "psf": d.encode_psf(true, voxel)}                 # what a worker gets
    np.testing.assert_allclose(d.run(measured, spec), d.deconvolve(
        measured, m, d.Parameters(iterations=3), measured=(true, voxel)), rtol=1e-5)
    with pytest.raises(ValueError):
        d.deconvolve(measured, m)                            # the measured model needs a PSF


def test_distilled_psf_is_the_beads_psf():
    rng = np.random.default_rng(4)
    beads = [(12, 20 + 30 * i + rng.uniform(-0.5, 0.5), 20 + 30 * j + rng.uniform(-0.5, 0.5), 1.0)
             for i in range(5) for j in range(5)]
    beads[0] = beads[0][:3] + (2.5,)                         # a clump
    beads.append((12, beads[6][1] + 4, beads[6][2] + 3, 1.0))   # a neighbour too close
    psf, info = d.distill_psf(bead_stack(FRAME, (24, 160, 160), beads), window_px=17)
    assert (info["found"], info["bright"], info["crowded"], info["used"]) == (26, 1, 2, 23)
    assert psf.shape[1:] == (17, 17) and abs(float(psf.sum()) - 1) < 1e-5
    assert np.unravel_index(psf.argmax(), psf.shape) == tuple(n // 2 for n in psf.shape)
    truth = d.widths(centred(d.psf(FRAME, (24, 64, 64))), FRAME)
    assert np.allclose(d.widths(psf, FRAME), truth, rtol=0.1)
    with pytest.raises(ValueError, match="no beads|none of"):
        d.distill_psf(np.full((8, 64, 64), 100, np.float32))


def test_plane_cancel_and_refusals():
    _, measured = d.synthetic((4, 64, 64), FRAME)
    plane = d.deconvolve(measured[2], FRAME, d.Parameters(iterations=5))
    assert plane.shape == (64, 64) and np.isfinite(plane).all() and plane.min() >= 0
    with pytest.raises(d.Cancelled):
        d.deconvolve(measured, FRAME, d.Parameters(iterations=5), progress=lambda f: f > 0.3)
    spec = {"optics": {"pixel_um": 0.12, "z_step_um": 0.5},
            "parameters": {"method": "rl", "iterations": 3}}
    assert d.run(measured, spec).shape == measured.shape
    for bad in ({"optics": {"pixel_um": 0.12, "na": 1.1}},                    # NA above air
                {"optics": {"pixel_um": 0.12, "colour": 1}},                  # unknown field
                {"optics": {"pixel_um": 0.12}, "parameters": {"iterations": 0}},
                {"optics": {"pixel_um": 0.12}, "parameters": {"method": "magic"}}):
        with pytest.raises(ValueError):
            d.run(measured, bad)
    with pytest.raises(ValueError):
        d.deconvolve(np.full((4, 8, 8), np.nan), FRAME)
