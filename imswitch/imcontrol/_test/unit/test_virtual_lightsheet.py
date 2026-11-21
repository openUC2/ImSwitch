"""Virtual lightsheet: X/Y/A move the sample, Z (objective) defocuses, the illumination NA sets the sheet."""
import cv2
import numpy as np
import pytest

from imswitch.imcontrol.model.managers.rs232.virtual_lightsheet import LightsheetSimulator


@pytest.fixture(scope="module")
def sim():
    return LightsheetSimulator(width=256, height=256, nSpheroids=3, nBeads=0, readNoise=0, background=0)


def test_frame_is_a_12bit_camera_image(sim):
    frame = sim.render({}, 1024, 10)
    assert frame.shape == (256, 256) and frame.dtype == np.uint16
    assert 0 < frame.max() <= 4095


def test_stage_x_moves_the_content_right(sim):
    a = sim.render({"X": 0}, 1024, 10).astype(np.float32)
    b = sim.render({"X": 5}, 1024, 10).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    assert dx == pytest.approx(5 / sim.pixelUm, abs=1.0) and abs(dy) < 1.0


def test_objective_out_of_focus_blurs(sim):
    def sharpness(frame):   # edge energy above the shot noise
        smooth = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 2)
        return cv2.Laplacian(smooth, cv2.CV_32F).var()
    assert sharpness(sim.render({"Z": 0}, 1024, 10)) > 3 * sharpness(sim.render({"Z": 80}, 1024, 10))   # Z: objective


def test_illumination_na_sets_the_sheet():
    s = LightsheetSimulator(width=64, height=64, nSpheroids=1, nBeads=0)
    s.setIlluminationNA(0.05)
    thick, zr_long = s.optics()["sheetWaistUm"], s.optics()["sheetRayleighRangeUm"]
    s.setIlluminationNA(0.10)
    assert s.optics()["sheetWaistUm"] == pytest.approx(thick / 2)
    assert s.optics()["sheetRayleighRangeUm"] == pytest.approx(zr_long / 4)
    edge, centre = s._sheetThickness(np.array([88.0, 0.0]))
    assert edge > 2 * centre                     # NA 0.1: the sheet widens across a 177 µm field
