"""Spectrometer model (no hardware): grating design, pipeline, peaks, fits, wizard, IO."""
import threading
import time

import numpy as np
import pytest

from imswitch.imcontrol.model.spectrometer import pipeline as P
from imswitch.imcontrol.model.spectrometer import peaks as K
from imswitch.imcontrol.model.spectrometer import references as R
from imswitch.imcontrol.model.spectrometer import spectrum_io
from imswitch.imcontrol.model.spectrometer.calibration import (
    Calibration, GratingDesign, alignment_plan, fit_polynomial, make_calibration,
    nm_to_raman_shift, raman_shift_to_nm, resample_linear_k, turn_from_line)
from imswitch.imcontrol.model.spectrometer.simulator import (SimulationSettings,
                                                             SpectrometerSimulator)
from imswitch.imcontrol.model.spectrometer.wizard import (CalibrationSource, CalibrationWizard,
                                                          WizardSettings)

SLIT = (57, 456)          # rows lit in the default simulation


def make_sim(**kw):
    """A short-lens instrument (true f 30 mm, 555 nm on the axis; the nominal design says
    32 mm and 550 nm): 488, 520 and 635 nm and the R, G, B LEDs land on the sensor."""
    design = GratingDesign(cameraFocalLengthMm=32.0, centerWavelengthNm=550.0)
    return SpectrometerSimulator(design=design, settings=SimulationSettings(**kw)), design


def lines(*nm, b=0.5):
    return [{"type": "line", "nm": v, "brightness": b} for v in nm]


# ----------------------------------------------------------------------------- design
def test_design_reproduces_openraman_numbers():
    """Block numbers from openuc2-openraman (spectro_block.py, docs)."""
    d = GratingDesign()
    assert d.turn_for_axis(614.4) == pytest.approx(0.0, abs=0.01)      # OpenRAMAN position
    assert d.turn_for_axis(592.4) == pytest.approx(-0.90, abs=0.01)    # Raman centred
    s = Calibration(design=d).summary()
    assert s["nmPerPxAtCenter"] == pytest.approx(0.020, abs=0.0005)     # 0.020 nm/px
    assert s["nmAtLastPx"] == pytest.approx(561.7, abs=0.5)             # 562-623 nm window
    assert s["nmAtFirstPx"] == pytest.approx(623.1, abs=0.5)
    assert d.slit_image_px(592.4, -0.90) == pytest.approx(41, abs=1.0)  # measured 41 px


def test_raman_shift_and_k_axis():
    assert nm_to_raman_shift(562.0, 532.0) == pytest.approx(1003.4, abs=0.1)
    assert raman_shift_to_nm(nm_to_raman_shift(600.0, 532.0), 532.0) == pytest.approx(600.0)
    nm = np.linspace(800, 900, 500)
    k, v = resample_linear_k(nm, np.cos(2 * np.pi * 1000 / nm))
    assert np.allclose(np.diff(k), np.diff(k)[0]) and k[0] < k[-1]


def test_grating_model_round_trip():
    cal = Calibration(design=GratingDesign())
    px = np.array([0.0, 1000.0, 3071.0])
    assert np.allclose(cal.nm_to_px(cal.px_to_nm(px)), px, atol=1e-6)
    poly = Calibration(model="polynomial", design=GratingDesign(),
                       polynomial=fit_polynomial([{"px": p, "nm": float(cal.px_to_nm(p))}
                                                  for p in (0, 700, 1500, 2300, 3071)], 3, 3072))
    assert np.allclose(poly.nm_to_px(poly.px_to_nm(px)), px, atol=0.05)


# ----------------------------------------------------------------------------- pipeline
def test_slit_rows_smile_and_line_centres():
    sim, _ = make_sim()
    frame = sim.render(lines(488.0, 520.0, 635.0))
    dark = sim.render([]).astype(np.float32)
    rows = P.find_slit_rows(frame)
    assert abs(rows["rowStart"] - SLIT[0]) <= 3 and abs(rows["rowStop"] - SLIT[1]) <= 3
    s = P.ProcessingSettings(rowStart=rows["rowStart"], rowStop=rows["rowStop"])
    raw = P.process(frame, s, dark=dark)
    smile = P.measure_smile(raw["roi"], rows["rowStart"])
    assert smile["bowPx"] == pytest.approx(16.0, abs=1.0)
    out = P.process(frame, s, dark=dark, smile=smile)
    centres = sorted(K.peak_properties(out["counts"], p["index"], p["base"])["center"]
                     for p in K.find_peaks(out["counts"]))
    assert np.allclose(centres, sorted(sim.nm_to_px([488.0, 520.0, 635.0])), atol=0.15)
    widths = [K.peak_properties(out["counts"], p["index"], p["base"])["fwhm"]
              for p in K.find_peaks(out["counts"])]
    widths_raw = [K.peak_properties(raw["counts"], p["index"], p["base"])["fwhm"]
                  for p in K.find_peaks(raw["counts"])]
    assert max(widths) < min(widths_raw)          # the smile correction sharpens the lines


def test_spikes_removed_but_narrow_lines_kept():
    sim, _ = make_sim(lineFwhmPx=2.0, hotPixels=0, spikesPerFrame=0)
    frame = sim.render(lines(520.0))
    s = P.ProcessingSettings(rowStart=SLIT[0], rowStop=SLIT[1])
    assert P.process(frame, s)["spikesRemoved"] == 0               # a 2 px line survives
    spiky = frame.copy()
    spiky[200, 1000] += 2000
    spiky[300, 2357] += 2500                                         # on top of the line
    clean = P.process(spiky, s)
    assert clean["spikesRemoved"] >= 2
    assert np.max(np.abs(clean["counts"] - P.process(frame, s)["counts"])) < 3.0
    sim2, _ = make_sim(spikesPerFrame=3.0)
    frames = [sim2.render(lines(520.0)) for _ in range(5)]
    _, n = P.stack_mean(frames)
    assert 5 <= n <= 30                                              # ~15 spikes, no lines


def test_saturation_and_decimate():
    sat = P.saturation(np.array([[4095, 10], [5, 7]], dtype=np.uint16), 12)
    assert sat["saturated"] and sat["saturatedPixels"] == 1
    x, y = P.decimate(np.arange(3000), np.r_[np.zeros(1500), 9.0, np.zeros(1499)], 300)
    assert len(y) == 300 and y.max() == 9.0


# ----------------------------------------------------------------------------- fits
def test_three_lasers_recover_the_true_axis():
    sim, design = make_sim()
    frame = sim.render(lines(488.0, 520.0, 635.0))
    s = P.ProcessingSettings(rowStart=SLIT[0], rowStop=SLIT[1])
    dark = sim.render([]).astype(np.float32)
    smile = P.measure_smile(P.process(frame, s, dark=dark)["roi"], SLIT[0])
    out = P.process(frame, s, dark=dark, smile=smile)
    pts = []
    for p in K.find_peaks(out["counts"]):
        c = K.peak_properties(out["counts"], p["index"], p["base"])["center"]
        nm = min((488.0, 520.0, 635.0), key=lambda v: abs(float(sim.nm_to_px(v)) - c))
        pts.append({"px": c, "nm": nm, "sigmaNm": 0.05})
    cal = make_calibration(pts, design)
    assert cal.model == "grating" and cal.rmsNm < 0.02
    grid = np.linspace(0, 3071, 50)
    assert np.max(np.abs(cal.px_to_nm(grid) - sim.px_to_nm(grid))) < 0.1   # whole sensor


def test_one_line_fixes_the_offset_only():
    design = GratingDesign()
    truth = Calibration(design=design, grating={"turnDeg": -0.7, "axisPx": 1560.0,
                                                "focalLengthMm": 100.0})
    pt = {"px": float(truth.nm_to_px(600.0)), "nm": 600.0, "sigmaNm": 0.1}
    cal = make_calibration([pt], design)
    assert float(cal.px_to_nm(pt["px"])) == pytest.approx(600.0, abs=0.01)
    assert cal.grating["axisPx"] == pytest.approx(design.axisPx, abs=30)    # prior holds


# ----------------------------------------------------------------------------- alignment
def test_alignment_plan_for_the_qbox_520_laser():
    """520 nm is outside the Raman window: park it on the axis, then turn by +2.92 deg."""
    d = GratingDesign(leverArmMm=25.0)
    plan = alignment_plan(d, 520.0, 592.4)
    assert not plan["lineOnSensorAtTarget"]
    assert plan["windowAtTargetNm"] == pytest.approx([561.7, 623.1], abs=0.5)
    assert plan["turnAtParkDeg"] == pytest.approx(-3.82, abs=0.01)
    assert plan["deltaTurnFromParkDeg"] == pytest.approx(2.92, abs=0.01)
    assert plan["leverTravelFromParkMm"] == pytest.approx(1.275, abs=0.01)
    seen = alignment_plan(d, 520.0, 592.4, measured_px=1000.0)
    assert seen["turnNowDeg"] == pytest.approx(turn_from_line(d, 520.0, 1000.0))
    near = alignment_plan(d, 600.0, 592.4, measured_px=2000.0)
    assert near["lineOnSensorAtTarget"] and 0 < near["linePxAtTarget"] < 3072


# ----------------------------------------------------------------------------- wizard
class Rig:
    """Simulator behind the wizard's three callables."""

    def __init__(self, sim, emission_of):
        self.sim, self.on, self.emission_of, self.dark = sim, set(), emission_of, None
        self.log = []

    def switch(self, src, on):
        self.log.append((src.id, on))
        (self.on.add if on else self.on.discard)(src.id)

    def take_dark(self):
        self.dark = np.mean([self.sim.render([]) for _ in range(3)], axis=0).astype(np.float32)

    def measure(self):
        em = [c for sid in sorted(self.on) for c in self.emission_of[sid]]
        frames = [self.sim.render(em) for _ in range(3)]
        mean, _ = P.stack_mean(frames)
        roi = mean[SLIT[0]:SLIT[1]] - (self.dark[SLIT[0]:SLIT[1]] if self.dark is not None else 0)
        return {"roi": roi, "rows": list(SLIT), "cols": [0, self.sim.settings.cols]}


def run_wizard(sources, rig, design, manual_on=None, **settings):
    wiz = CalibrationWizard(sources, rig.measure, rig.switch, design,
                            settings=WizardSettings(settleS=0.0, **settings),
                            take_dark=rig.take_dark)
    wiz.start()
    t0 = time.time()
    while wiz.state not in ("review", "error", "cancelled") and time.time() - t0 < 60:
        if wiz.state == "waiting":
            src = sources[wiz.status()["stepIndex"]]
            if manual_on:
                manual_on(src)
            wiz.proceed()
        time.sleep(0.02)
    return wiz


def test_wizard_lasers_and_manual_qbox_laser():
    sim, design = make_sim()
    srcs = [CalibrationSource("laser488", "488 nm laser", control="laser", wavelengthNm=488.0,
                              sigmaNm=0.05),
            CalibrationSource("qbox520", "Q-Box 520 nm laser", control="manual",
                              wavelengthNm=520.0, sigmaNm=0.05),
            CalibrationSource("laser635", "635 nm laser", control="laser", wavelengthNm=635.0,
                              sigmaNm=0.05)]
    em = {"laser488": lines(488.0), "qbox520": lines(520.0), "laser635": lines(635.0)}
    rig = Rig(sim, em)
    wiz = run_wizard(srcs, rig, design, manual_on=lambda s: rig.on.add(s.id))
    st = wiz.status()
    assert st["state"] == "review", st["message"]
    assert [s["status"] for s in st["steps"]] == ["done", "done", "done"]
    assert st["smile"]["bowPx"] == pytest.approx(16.0, abs=1.0)
    assert len(st["resolution"]) == 3 and st["resolution"][0]["fwhmPx"] < 20
    grid = np.linspace(0, 3071, 50)
    assert np.max(np.abs(wiz.result.px_to_nm(grid) - sim.px_to_nm(grid))) < 0.15
    assert ("laser488", False) in rig.log and ("laser635", False) in rig.log   # left off


def test_wizard_neopixel_leds_with_references():
    sim, design = make_sim()
    refs = R.placeholder_led_spectra()
    srcs = [CalibrationSource(f"ring_{c}", f"NeoPixel {c}", kind="led", control="ledmatrix",
                              channel=c, reference=refs[c], sigmaNm=1.5) for c in "rgb"]
    em = {f"ring_{c}": [{"type": "spectrum", **refs[c], "brightness": 0.6}] for c in "rgb"}
    wiz = run_wizard(srcs, Rig(sim, em), design)
    st = wiz.status()
    assert st["state"] == "review", st["message"]
    assert all(p["dispersionHintNmPerPx"] > 0 for p in st["points"])
    assert any("placeholder" in h for h in st["hints"])
    grid = np.linspace(300, 2800, 20)
    assert np.max(np.abs(wiz.result.px_to_nm(grid) - sim.px_to_nm(grid))) < 1.0


def test_wizard_rejects_a_band_off_the_sensor():
    sim, design = make_sim(focalLengthMm=40.0, centerNm=565.0)     # window ~490-640 nm
    refs = R.placeholder_led_spectra()
    srcs = [CalibrationSource(f"ring_{c}", f"NeoPixel {c}", kind="led", control="ledmatrix",
                              channel=c, reference=refs[c], sigmaNm=1.5) for c in "rgb"]
    em = {f"ring_{c}": [{"type": "spectrum", **refs[c], "brightness": 0.6}] for c in "rgb"}
    st = run_wizard(srcs, Rig(sim, em), design).status()
    assert [s["status"] for s in st["steps"]] == ["done", "done", "failed"]
    assert "off the sensor" in st["steps"][2]["detail"] and st["state"] == "review"


def test_wizard_neon_lamp_matches_lines_from_a_wrong_start():
    sim, design = make_sim()
    lo, hi = sorted(sim.px_to_nm([0, 3071]))
    neon = [v for v in R.NEON_LINES_NM if lo + 2 < v < hi - 2]
    src = CalibrationSource("neon", "neon lamp", kind="lamp", control="manual",
                            lines=R.NEON_LINES_NM, lineList="neon")
    em = {"neon": [{"type": "lines", "nm": neon, "weights": list(np.linspace(1, 0.4, len(neon))),
                    "brightness": 0.5}]}
    rig = Rig(sim, em)
    wiz = run_wizard([src], rig, design, manual_on=lambda s: rig.on.add(s.id),
                     model="polynomial", order=3)
    st = wiz.status()
    assert st["state"] == "review", st["message"]
    assert len(st["points"]) >= 8
    assert wiz.result.rmsNm < 0.05


def test_wizard_cancel_switches_everything_off():
    sim, design = make_sim()
    srcs = [CalibrationSource("qbox520", "Q-Box", control="manual", wavelengthNm=520.0),
            CalibrationSource("laser488", "488", control="laser", wavelengthNm=488.0)]
    rig = Rig(sim, {"qbox520": lines(520.0), "laser488": lines(488.0)})
    wiz = CalibrationWizard(srcs, rig.measure, rig.switch, design,
                            settings=WizardSettings(settleS=0.0))
    wiz.start()
    t0 = time.time()
    while wiz.state != "waiting" and time.time() - t0 < 10:
        time.sleep(0.01)
    wiz.cancel()
    wiz.join(5)
    assert wiz.state == "cancelled" and ("laser488", False) in rig.log
    assert "laser488" not in rig.on


def test_wizard_reports_a_dark_source():
    sim, design = make_sim()
    srcs = [CalibrationSource("laser488", "488", control="laser", wavelengthNm=488.0)]
    wiz = run_wizard(srcs, Rig(sim, {"laser488": []}), design)
    assert wiz.state == "error" and wiz.status()["steps"][0]["status"] == "failed"


# ----------------------------------------------------------------------------- references, io
def test_reference_files(tmp_path):
    p = tmp_path / "ring.csv"
    nm = np.arange(400, 701, 1.0)
    with open(p, "w") as fh:
        fh.write("wavelength_nm,r,g,b\n")
        for v in nm:
            fh.write(f"{v},{np.exp(-((v - 630) / 10) ** 2)},{np.exp(-((v - 525) / 15) ** 2)},"
                     f"{np.exp(-((v - 465) / 12) ** 2)}\n")
    refs = R.load_reference_spectra(str(p))
    assert set(refs) == {"r", "g", "b"} and max(refs["g"]["intensity"]) == 1.0
    assert K.reference_feature(refs["g"]["wavelengthNm"], refs["g"]["intensity"])["centerNm"] \
        == pytest.approx(525.0, abs=0.2)
    j = tmp_path / "ring.json"
    j.write_text(spectrum_io.to_json({"sources": {"g": refs["g"]}}))
    assert R.load_reference_spectra(str(j))["g"]["placeholder"] is False


def test_save_and_load(tmp_path):
    rec = {"px": np.arange(5.0), "nm": np.linspace(500, 504, 5), "counts": np.arange(5.0) ** 2,
           "roi": np.ones((3, 5)), "metadata": {"exposureMs": 10.0, "nAverage": np.int64(4)}}
    h5 = spectrum_io.save_hdf5(str(tmp_path / "a.h5"), rec)
    back = spectrum_io.load_hdf5(h5)
    assert np.allclose(back["counts"], rec["counts"]) and back["metadata"]["nAverage"] == 4
    assert back["roi"].shape == (3, 5)
    paths = spectrum_io.save_csv(str(tmp_path / "a.csv"), rec)
    back = spectrum_io.load_csv(paths[0])
    assert np.allclose(back["nm"], rec["nm"]) and back["metadata"]["exposureMs"] == 10.0


def test_wizard_is_thread_safe_to_poll():
    sim, design = make_sim()
    srcs = [CalibrationSource("laser488", "488", control="laser", wavelengthNm=488.0)]
    rig = Rig(sim, {"laser488": lines(488.0)})
    wiz = CalibrationWizard(srcs, rig.measure, rig.switch, design,
                            settings=WizardSettings(settleS=0.0))
    stop = threading.Event()
    polled = []

    def poll():
        while not stop.is_set():
            polled.append(wiz.status()["state"])

    t = threading.Thread(target=poll)
    t.start()
    wiz.start()
    wiz.join(30)
    stop.set()
    t.join()
    assert wiz.state == "review" and polled
