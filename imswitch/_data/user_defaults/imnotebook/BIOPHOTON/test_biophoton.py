"""Synthetic-data checks of the BIOPHOTON analysis. Run: python test_biophoton.py (or pytest)."""
import csv
import json
import math
import tempfile

import numpy as np
import tifffile

import biophoton as bp

RNG = np.random.default_rng(1)


def emccd_frame(lam, shape=(256, 256), bias=400.0, read_noise=6.0, em_gain_adu=300.0):
    """Bias with an amplifier-glow ramp, read noise, and Poisson photo-electrons through the EM
    register (each one an exponential burst of mean `em_gain_adu`)."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    img = bias + 40 * x / w + 113 * y / h + RNG.normal(0, read_noise, shape)
    n = RNG.poisson(lam, shape)
    img += np.where(n > 0, RNG.gamma(np.maximum(n, 1), em_gain_adu), 0)
    return img


def test_photon_events():
    lam, sigma, g = 0.05, 6.0, 300.0
    ev = bp.photon_events(emccd_frame(lam, read_noise=sigma, em_gain_adu=g), nsigma=5)
    assert abs(ev["sigma"] - sigma) / sigma < 0.05, ev["sigma"]
    est = bp.event_stats(ev["mask"], None)["lam"]
    expected = lam * math.exp(-5 * sigma / g)      # threshold efficiency
    assert abs(est - expected) / expected < 0.1, (est, expected)


def test_camera_temperature_keys():
    class S:
        def __init__(self, status):
            self.status = status

        def getCameraStatus(self, detector):
            return self.status

    class C:
        def __init__(self, status):
            self.settingsManager = S(status)

    toupcam = {"hardwareParameters": {"temperature_c": -29.8, "tec_target_c": -30}}
    mmcore = {"temperatureC": -94.0, "temperatureSetpointC": -95.0}
    assert bp.camera_temperature(C(toupcam), "x") == (-29.8, -30)
    assert bp.camera_temperature(C(mmcore), "x") == (-94.0, -95.0)


def write_frames(run_dir, rows):
    with open(run_dir / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, bp.FRAME_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_run(tmp, cfg, frames):
    """frames: [(label, exposure_s, image)] -> run folder as Acquisition would leave it."""
    run = bp.Run("TEST", "synthetic", tmp)
    rows = []
    for i, (label, exp_s, img) in enumerate(frames):
        name = f"raw/{label}_{i:04d}.tif"
        tifffile.imwrite(run.dir / name, img.astype(np.uint16))
        rows.append({"index": i, "step": 0, "label": label, "start_iso": "2026-10-23T10:00:00",
                     "start_unix": 1e9 + 2000 * i, "end_unix": 1e9 + 2000 * i + exp_s,
                     "exposure_ms": exp_s * 1000, "file": name, "median": float(np.median(img))})
    write_frames(run.dir, rows)
    run.update_config(**cfg)
    return run.dir


def test_photon_counting_run():
    with tempfile.TemporaryDirectory() as tmp:
        def frame():
            img = emccd_frame(0.002)
            img[100:150, 100:150] = emccd_frame(0.05, shape=(50, 50))[...]   # glowing leaf
            return img
        run_dir = make_run(tmp, {"kind": "timecourse", "analysis": {"mode": "photon_counting"},
                                 "rois": {"sample": [100, 100, 50, 50], "background": [10, 10, 60, 60]}},
                           [("frame", 90, frame()) for _ in range(5)])
        s = bp.redraw(run_dir)
        assert s["frames"] == 5 and s["rate"] > 0 and s["significance_sigma"] > 5, s
        assert (run_dir / "README.md").exists() and (run_dir / "event_map.tif").exists()
        assert (run_dir / "figures" / "timecourse.png").exists()


def test_integrated_run():
    with tempfile.TemporaryDirectory() as tmp:
        def frame():
            img = 1000 + RNG.normal(0, 2, (200, 200))
            img[50:100, 50:100] += 36.0                     # 36 ADU over 1800 s = 0.02 ADU/px/s
            return img
        run_dir = make_run(tmp, {"kind": "timecourse", "analysis": {"mode": "integrated"},
                                 "rois": {"sample": [50, 50, 50, 50], "background": [120, 120, 60, 60]}},
                           [("frame", 1800, frame()) for _ in range(3)])
        s = bp.redraw(run_dir)
        assert abs(s["rate"] - 0.02) < 0.002, s


def test_ptc_and_dark_ladder():
    k, rn_e, offset, dark_e = 0.5, 1.5, 1000.0, 0.005      # ADU/e-, e-, ADU, e-/px/s
    with tempfile.TemporaryDirectory() as tmp:
        signal = np.linspace(100, 5000, 120 * 120).reshape(120, 120)
        bright = np.stack([offset + k * RNG.poisson(signal) + RNG.normal(0, rn_e * k, signal.shape)
                           for _ in range(20)])
        dark = offset + RNG.normal(0, rn_e * k, (20, 120, 120))
        ladder = [(f"dark_{t}s", t, offset + k * RNG.poisson(dark_e * t, (300, 300))
                   + RNG.normal(0, rn_e * k, (300, 300))) for t in (60, 600, 3000) for _ in range(3)]
        run_dir = make_run(tmp, {"kind": "ptc", "dark_ladder_gain": 23, "datasheet_dark_e_px_s": 5e-4},
                           [(lab, t, np.round(img)) for lab, t, img in ladder])
        sess = run_dir / "ptc" / "s1"
        sess.mkdir(parents=True)
        tifffile.imwrite(sess / "bright_stack.tif", np.round(bright).astype(np.uint16))
        tifffile.imwrite(sess / "dark_stack.tif", np.round(dark).astype(np.uint16))
        (sess / "result.json").write_text(json.dumps({"gain": 1.9, "readnoise": 1.4, "offset": 1000}))
        cfg = json.loads((run_dir / "config.json").read_text())
        cfg["ptc_sessions"] = [{"gain": 23, "dir": "ptc/s1"}]
        (run_dir / "config.json").write_text(json.dumps(cfg))

        tp = bp.temporal_ptc(sess / "bright_stack.tif", sess / "dark_stack.tif", step=1)
        assert abs(tp["e_per_adu"] - 1 / k) / (1 / k) < 0.05, tp["e_per_adu"]
        assert abs(tp["read_noise_e"] - rn_e) / rn_e < 0.25, tp["read_noise_e"]
        s = bp.redraw(run_dir)
        assert abs(s["dark_current_e_px_s_from_mean"] - dark_e) / dark_e < 0.2, s
        assert abs(s["dark_current_e_px_s_from_variance"] - dark_e) / dark_e < 0.3, s


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
