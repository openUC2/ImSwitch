"""The deconvolution recipe: same result as DeconvOptim.jl, tiles, PSF from the optics, refusals.

deconvolution_julia_reference.npz holds the z and y maximum projections (scaled to 1, float16) of
DeconvOptim.jl's map_cj_lean (branch new_methods, c72a2c8) on synthetic((32, 256, 256), FRAME)
with gamma 1e-4, 30 iterations, background 5 and z padding 0.25 (scripts/deconvolution).
"""
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
