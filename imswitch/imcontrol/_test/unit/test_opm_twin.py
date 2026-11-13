"""dvOPM twin: stage <-> model axes, frame positions from the measured speed, deskew."""
import math

import numpy as np
import pytest

from imswitch.imcontrol.model.opm import twin
from imswitch.imcontrol.model.opm.kinematics import Detection


def test_axis_map_and_round_trip():
    links = twin.parse_axis_map({"sample.X": "-X", "launch.z": "Launch.Z", "launch.x": ""}, "Stage")
    assert links == {"sample.X": ("Stage", "X", -1.0), "launch.z": ("Launch", "Z", 1.0)}
    # reference: stage X = 500 µm stands for sample X = 0.2 mm; the axis runs backwards
    x = twin.to_model("sample.X", 1500.0, -1.0, 500.0, 0.2)
    assert x == pytest.approx(0.2 - 1.0)
    assert twin.to_raw("sample.X", x, -1.0, 500.0, 0.2) == pytest.approx(1500.0)
    assert twin.unit("launch.phi_deg") == 1.0
    with pytest.raises(ValueError):
        twin.parse_axis_map({"sample.Q": "X"}, "Stage")


def test_frame_positions_use_the_measured_move():
    xs = twin.frame_positions([9.0, 10.0, 12.0, 14.0, 15.0], 10.0, 14.0, -0.1, 1.1)
    assert xs == pytest.approx([-0.1, -0.1, 0.5, 1.1, 1.1])
    s = twin.scan_summary([0.0, 0.05, 0.10, 0.15], [0.0, 0.01, 0.02, 0.03], [7, 8, 10, 11])
    assert s["fps"] == pytest.approx(20.0)
    assert s["velocity_mm_s"] == pytest.approx(0.2)
    assert s["step_um"] == pytest.approx(10.0)
    assert s["dropped_frames"] == 1
    assert s["camera_fps"] == pytest.approx(80 / 3)
    # polling jitter on top of a 25 fps camera; frame 2 was missed
    t = twin.frame_times([1.002, 1.041, 1.120, 1.163], [0, 1, 3, 4])
    assert np.diff(t) == pytest.approx([0.04, 0.08, 0.04], abs=0.004)


def test_deskew_puts_a_bead_where_it_is():
    det = Detection(pixel_um=4.4)
    n, N, R, C = 1.45, 120, 48, 6
    p, a = det.pixel_um * 1e-3, det.alpha
    kx = p * math.cos(a) / det.M
    xs = np.linspace(0.0, 0.6, N)                           # sample X per frame, mm
    off = kx * (np.arange(R) - (R - 1) / 2)
    rb, cb, xb = 30, 2, -0.31                               # bead row (depth), column, sample x
    shown = -off[rb] - xs                                   # sample x that row rb sees per frame
    stack = np.zeros((N, R, C), np.uint16)
    stack[:, rb, cb] = (1000 * np.exp(-((shown - xb) / 0.006) ** 2)).astype(np.uint16)
    vol, vox = twin.deskew(stack, xs, n, det)
    r, c, j = np.unravel_index(np.argmax(vol), vol.shape)
    assert (r, c) == (rb, cb)
    assert vox["x0_mm"] + j * vox["dx_um"] * 1e-3 == pytest.approx(xb, abs=vox["dx_um"] * 1e-3)
    assert vox["dz_um"] == pytest.approx(p * n * math.sin(a) / det.M ** 2 * 1e3)
    flipped, _ = twin.deskew(stack, xs, n, det, flip=True)
    assert np.unravel_index(np.argmax(flipped), flipped.shape)[0] == R - 1 - rb
