"""Shared code for the BIOPHOTON notebooks (E0-E4).

Every run gets one folder that holds everything needed to redraw it later::

    data/<experiment>/<YYYY-MM-DD_HH-MM-SS>_<name>/
        README.md         summary, settings and figures (rewritten by redraw())
        config.json       notebook settings, sample, camera read-back
        frames.csv        one row per acquired frame
        analysis.csv      per-frame results of the last redraw()
        environment.csv   camera sensor + enclosure temperature, humidity, light
        events.log        what happened when, incl. every hand-done step
        raw/              full bit-depth TIFFs as ImSwitch wrote them
        figures/          PNGs from redraw()
        report.ipynb      redraws everything from this folder alone
        biophoton.py      the version of this file the run was taken with

No figure or number lives only in a notebook's memory. Acquisition writes each
frame and each environment sample to disk as it arrives, and every figure is
drawn from those files, so clearing or re-running a cell loses nothing.
"""
import csv
import datetime
import json
import math
import shutil
import threading
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent

FRAME_COLUMNS = ["index", "step", "label", "start_iso", "start_unix", "end_unix", "exposure_ms",
                 "camera_temperature_c", "file", "mean", "median", "std", "max", "server_path"]
ENV_COLUMNS = ["iso_time", "unix_time", "source", "camera_temperature_c", "camera_target_c",
               "enclosure_temperature_c", "humidity_pct", "lux", "note"]
TERMINAL_MMCORE_STATES = ("done", "error", "cancelled", "timeout", "unknown")


# --------------------------------------------------------------------------------------------
# ImSwitch access
# --------------------------------------------------------------------------------------------
def connect(url=None):
    """ImSwitchClient for `url` (None: $IMSWITCH_API_URL, else localhost:8001)."""
    import imswitchclient.ImSwitchClient as imc
    client = imc.ImSwitchClient(url)
    if not hasattr(client.recordingManager, "snapAndWait"):
        raise ImportError("imswitchclient is too old (no recordingManager.snapAndWait), update it: "
                          "pip install -U https://github.com/openUC2/imswitchclient/archive/"
                          "refs/heads/main.zip")
    return client


def _flat(tree):
    return {p["name"]: p for g in tree.get("groups", []) for p in g.get("parameters", [])}


def parameters(client, detector):
    """({name: info}, full tree) of the detector: value, type, min/max, options."""
    tree = client.settingsManager.getDetectorParameterTree(detector)
    if tree.get("error"):
        raise RuntimeError(f"{detector}: {tree['error']}")
    return _flat(tree), tree


def apply_settings(client, detector, settings):
    """Set each {name: value} in order and return {name: value read back}.

    Unknown names raise before anything is changed, so a typo cannot cost a 12 h run.
    The read-back shows what the camera clamped.
    """
    known, _ = parameters(client, detector)
    unknown = [n for n in settings if n not in known]
    if unknown:
        raise ValueError(f"{detector} has no parameter(s) {unknown}; available: {sorted(known)}")
    readback = {}
    for name, value in settings.items():
        tree = client.settingsManager.setDetectorParameterValue(name, value, detector_name=detector)
        if tree.get("error"):
            raise RuntimeError(f"setting {name}={value!r} failed: {tree['error']}")
        readback[name] = _flat(tree).get(name, {}).get("value")
    return readback


def camera_temperature(client, detector):
    """(sensor °C, set point °C). ToupTek reports both under hardwareParameters, MMCore at top level."""
    status = client.settingsManager.getCameraStatus(detector)
    hw = status.get("hardwareParameters") or {}
    return (hw.get("temperature_c", status.get("temperatureC")),
            hw.get("tec_target_c", status.get("temperatureSetpointC")))


def enclosure_values(client):
    """SHT45 + TSL2591 reading of I2CSensorController, or None when the server has none."""
    try:
        return client.get_json(f"{client.base_uri}/I2CSensorController/getLatestI2CSensorValues")
    except Exception:
        return None


def env_sources(client, detector, label, others=()):
    """EnvLogger sources: this camera plus other servers [{"label", "url", "detector"}, ...]
    (detector None = that server's first detector)."""
    sources = [{"label": label, "client": client, "detector": detector}]
    for other in others:
        c = connect(other["url"])
        sources.append({"label": other["label"], "client": c,
                        "detector": other.get("detector") or c.settingsManager.getDetectorNames()[0]})
    return sources


def fetch_ptc_session(run, client, result, stacks=True):
    """Copy a computed ReadNoiseCalibration session (results, figures, optionally the stacks)
    into <run>/ptc/<sessionId>/ and return that folder."""
    rel, dest = result["dataRelPath"], run.dir / "ptc" / result["sessionId"]
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    names = ["result.json", "session.json", "figures/calibration_results.txt",
             "figures/definitions.txt"]
    names += [f"figures/{n}" for n in result.get("result", {}).get("figures", [])]
    if stacks:
        names += ["bright/bright_stack.tif", "dark/dark_stack.tif"]
    for name in names:
        target = dest / (Path(name).name if name.endswith("_stack.tif") else name)
        try:
            client.recordingManager.downloadFile(f"{rel}/{name}", target)
        except Exception as e:
            run.log(f"could not fetch {name} of session {result['sessionId']}: {e}")
    return dest


def show_exposure_check(img):
    """Histogram of a test frame: brightest part near 70 % of full scale, nothing saturated?"""
    img = np.squeeze(np.asarray(img))
    full = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else float(img.max())
    p50, p99, p999 = np.percentile(img, [50, 99, 99.9])
    print(f"median {p50:.0f} | 99 % {p99:.0f} | 99.9 % {p999:.0f} | max {img.max():.0f} "
          f"of {full} ADU | at max: {np.mean(img >= img.max()):.3%} of pixels")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4))
    a0.imshow(img[::4, ::4], cmap="gray", **_stretch(img))
    a0.axis("off")
    a1.hist(img.ravel()[::7], bins=256, log=True)
    a1.axvline(0.7 * full, color="g", ls="--", label="70 % of the container")
    a1.axvline(img.max(), color="r", lw=1, label="frame max")
    a1.set_xlabel("ADU")
    a1.legend()
    fig.tight_layout()
    plt.show()


def wait_for_temperature(client, detector, target_c, tolerance_c=1.0, timeout_s=1800, poll_s=10,
                         log=print):
    """Block until the sensor is within `tolerance_c` of `target_c` (or `timeout_s` passed)."""
    t_end = time.time() + timeout_s
    while True:
        temp, _ = camera_temperature(client, detector)
        if temp is None:
            log("camera reports no temperature, not waiting for the cooler")
            return None
        print(f"\rsensor {temp:.1f} °C, target {target_c} °C   ", end="")
        if abs(temp - target_c) <= tolerance_c:
            log(f"sensor at {temp:.1f} °C")
            return temp
        if time.time() > t_end:
            log(f"WARNING cooldown timed out at {temp:.1f} °C (target {target_c} °C), continuing")
            return temp
        time.sleep(poll_s)


def prompt(message, dry_run=False, log=None):
    """Ask the operator for a step by hand and wait for Enter (not in a dry run)."""
    if log:
        log(f"operator: {message}")
    if dry_run:
        print(f"[dry run, not waiting] {message}")
        return
    input(f">>> {message}\n    Press Enter when done: ")
    if log:
        log("operator: done")


def snap_recording(client, name, stop_event=None, poll_s=5, on_progress=None):
    """One fresh exposure through RecordingController (ToupTek, Hik, virtual); server path."""
    result = client.recordingManager.snapAndWait(name, return_preview=False, poll=poll_s,
                                                 on_progress=on_progress, stop_event=stop_event)
    return result.get("relativeFilePath")


def snap_mmcore(client, detector, name, exposure_ms, stop_event=None, poll_s=5, on_progress=None):
    """One exposure through MMCoreController (Andor). It owns the sensor for the whole exposure
    and polls the driver, so temperature reads keep working while it integrates."""
    base = f"{client.base_uri}/MMCoreController"
    job = client.post_json(f"{base}/snapMMCoreToDisk",
                           payload={"detectorName": detector, "exposureMs": exposure_ms,
                                    "fileName": name, "saveFormat": "tiff"})
    job_id = job["jobId"]
    stop_event = stop_event or threading.Event()
    try:
        while True:
            status = client.get_json(f"{base}/getMMCoreSnapStatus", payload={"jobId": job_id})
            if on_progress:
                on_progress(status)
            if status.get("state") == "done":
                return status.get("relativeFilePath")
            if status.get("state") in TERMINAL_MMCORE_STATES:
                raise RuntimeError(f"MMCore snap {job_id} {status.get('state')}: {status.get('error')}")
            if stop_event.wait(poll_s):
                raise InterruptedError(f"MMCore snap {job_id} stopped")
    except (InterruptedError, KeyboardInterrupt):
        client.post_json(f"{base}/cancelMMCoreSnap", payload={"jobId": job_id})
        raise


# --------------------------------------------------------------------------------------------
# Run folder
# --------------------------------------------------------------------------------------------
class Run:
    """One new run folder. An existing folder is never reused or overwritten."""

    def __init__(self, experiment, name, root="data"):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = Path(root).resolve() / experiment / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "raw").mkdir()
        (self.dir / "figures").mkdir()
        shutil.copy2(Path(__file__), self.dir / "biophoton.py")
        if (HERE / "report_template.ipynb").exists():
            shutil.copy2(HERE / "report_template.ipynb", self.dir / "report.ipynb")
        self._lock = threading.Lock()
        self.log(f"run folder {self.dir}")

    def log(self, msg):
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line)
        with self._lock, open(self.dir / "events.log", "a") as f:
            f.write(line + "\n")

    def save_json(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=str))

    def update_config(self, **values):
        path = self.dir / "config.json"
        cfg = json.loads(path.read_text()) if path.exists() else {}
        cfg.update(values)
        self.save_json("config.json", cfg)
        return cfg

    def append_frame(self, row):
        path = self.dir / "frames.csv"
        with self._lock:
            new = not path.exists()
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, FRAME_COLUMNS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)


# --------------------------------------------------------------------------------------------
# Environment log
# --------------------------------------------------------------------------------------------
class EnvLogger:
    """Polls camera temperature and enclosure sensors of one or more servers into a CSV.

    sources: [{"label": "toupcam", "client": c, "detector": "ToupCam", "enclosure": True}, ...]
    Each source has its own thread, so a server that answers slowly does not hold up the others.
    """

    def __init__(self, path, sources, period_s=30):
        self.path, self.sources, self.period_s = Path(path), sources, period_s
        self.samples = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._loop, args=(s,), daemon=True,
                                          name=f"env-{s['label']}") for s in sources]

    def read(self, src):
        now = time.time()
        row = {"iso_time": datetime.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
               "unix_time": now, "source": src["label"]}
        notes = []
        if src.get("detector"):
            try:
                row["camera_temperature_c"], row["camera_target_c"] = camera_temperature(
                    src["client"], src["detector"])
            except Exception as e:      # one missed sample must not end the log
                notes.append(f"camera: {e}")
        if src.get("enclosure", True):
            values = enclosure_values(src["client"])
            if values:
                row.update(enclosure_temperature_c=values.get("temperature_c"),
                           humidity_pct=values.get("humidity_pct"), lux=values.get("lux"))
        row["note"] = "; ".join(notes)
        return row

    def _loop(self, src):
        while not self._stop.is_set():
            row = self.read(src)
            with self._lock:
                self.samples.append(row)
                new = not self.path.exists()
                with open(self.path, "a", newline="") as f:
                    w = csv.DictWriter(f, ENV_COLUMNS, extrasaction="ignore")
                    if new:
                        w.writeheader()
                    w.writerow(row)
            self._stop.wait(self.period_s)

    def latest(self, label):
        for row in reversed(self.samples):
            if row["source"] == label:
                return row
        return {}

    def start(self):
        for t in self._threads:
            t.start()
        return self

    def stop(self):
        self._stop.set()

    @property
    def running(self):
        return any(t.is_alive() for t in self._threads)


# --------------------------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------------------------
class Acquisition:
    """Snaps frames in a background thread and stores each one in the run folder.

    steps: [{"label": "frame", "exposure_ms": 1800000, "n_frames": None, "duration_s": 43200}, ...]
    A step ends after `n_frames` frames or, at the start of the next frame, once `duration_s`
    has passed. backend "recording" for ToupTek (and the virtual camera), "mmcore" for the Andor.
    Interrupting the dashboard cell does not stop it; stop() does.
    """

    def __init__(self, run, client, detector, steps, backend="recording", env=None,
                 env_label=None, poll_s=5, max_errors=3):
        self.run_folder, self.client, self.detector = run, client, detector
        self.steps, self.backend = steps, backend
        self.env, self.env_label = env, env_label
        self.poll_s, self.max_errors = poll_s, max_errors
        self.stop_event = threading.Event()
        self.state = {"phase": "idle", "detail": "", "step": None, "frame": None, "error": None,
                      "started": None, "finished": None}
        self._thread = None

    def _progress(self, status):
        elapsed = status.get("elapsedMs", 0) / 60000
        total = status.get("expectedDurationMs", 0) / 60000
        self.state["detail"] = f"{status.get('state', status.get('status'))} {elapsed:.1f} / {total:.1f} min"

    def _set_exposure(self, step):
        if self.backend == "mmcore":       # exposure goes with every MMCore snap request
            return step["exposure_ms"]
        s = self.client.settingsManager
        s.setDetectorExposureTime(self.detector, step["exposure_ms"])
        actual = s.getDetectorParameter("exposure", self.detector)
        if actual is not None and abs(float(actual) - step["exposure_ms"]) > 1:
            self.run_folder.log(f"WARNING camera runs at {actual} ms, not {step['exposure_ms']} ms")
        return actual

    def _snap(self, step_index, step, index):
        name = f"{step['label']}_{index:04d}"
        server_name = f"{self.run_folder.dir.name}_{name}"
        temp = self.env.latest(self.env_label).get("camera_temperature_c") if self.env else None
        t0 = time.time()
        if self.backend == "mmcore":
            rel = snap_mmcore(self.client, self.detector, server_name, step["exposure_ms"],
                              self.stop_event, self.poll_s, self._progress)
        else:
            rel = snap_recording(self.client, server_name, self.stop_event, self.poll_s,
                                 self._progress)
        if not rel:
            raise RuntimeError("snap returned no file path")
        local = self.run_folder.dir / "raw" / f"{name}.tif"
        self.client.recordingManager.downloadFile(rel, local)
        img = load_image(local)
        self.run_folder.append_frame({
            "index": index, "step": step_index, "label": step["label"],
            "start_iso": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "start_unix": t0, "end_unix": time.time(), "exposure_ms": step["exposure_ms"],
            "camera_temperature_c": temp, "file": f"raw/{name}.tif",
            "mean": float(img.mean()), "median": float(np.median(img)), "std": float(img.std()),
            "max": float(img.max()), "server_path": rel})
        self.run_folder.log(f"saved {name} ({step['exposure_ms'] / 1000:g} s, median "
                            f"{np.median(img):.1f})")

    def _run_step(self, step_index, step, index):
        self.state.update(step=step_index, phase=step["label"])
        self._set_exposure(step)
        n, duration = step.get("n_frames"), step.get("duration_s")
        self.run_folder.log(f"step {step_index} '{step['label']}': {step['exposure_ms'] / 1000:g} s "
                            f"frames, {f'{n} of them' if n else f'for {duration} s'}")
        t_step, done, errors = time.time(), 0, 0
        while (n is None or done < n) and (duration is None or time.time() - t_step < duration):
            if self.stop_event.is_set():
                raise InterruptedError("stopped")
            self.state["frame"] = index
            try:
                self._snap(step_index, step, index)
            except InterruptedError:
                raise
            except Exception as e:          # one failed frame must not end a 12 h run
                errors += 1
                self.run_folder.log(f"frame {index} failed ({errors}/{self.max_errors}): {e}")
                if errors >= self.max_errors:
                    raise
                self.stop_event.wait(10)
                continue
            errors, done, index = 0, done + 1, index + 1
        return index

    def run(self):
        self.state.update(started=datetime.datetime.now().isoformat(timespec="seconds"), error=None)
        try:
            index = 0
            for step_index, step in enumerate(self.steps):
                index = self._run_step(step_index, step, index)
            self.state.update(phase="done", detail=f"{index} frames")
        except InterruptedError:
            self.state.update(phase="stopped", detail="stopped by user")
        except Exception as e:
            self.state.update(phase="error", error=str(e))
            self.run_folder.log(traceback.format_exc())
        finally:
            self.state["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
            self.run_folder.log(f"acquisition {self.state['phase']}: {self.state['detail']}")

    def start(self):
        if self.running:
            raise RuntimeError("already running")
        self.stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True, name="Acquisition")
        self._thread.start()
        return self

    def stop(self, wait_s=120):
        """Cancel the running exposure and end the acquisition."""
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=wait_s)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()


# --------------------------------------------------------------------------------------------
# Reading back
# --------------------------------------------------------------------------------------------
def read_table(path):
    """CSV -> {column: array}; numeric columns as float (blank = nan), others as str."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        try:
            out[key] = np.array([float(v) if v not in ("", "None") else np.nan for v in values])
        except ValueError:
            out[key] = np.array(values, dtype=object)
    return out


def load_config(run_dir):
    return json.loads((Path(run_dir) / "config.json").read_text())


def load_image(path):
    """One frame as a 2-D float array."""
    img = np.squeeze(tifffile.imread(path)).astype(np.float64)
    if img.ndim != 2:
        raise ValueError(f"{path}: expected one 2-D frame, got shape {img.shape}")
    return img


def read_stack(path, step=1):
    """(n, y, x) float stack, every `step`-th pixel only (keeps a 9 MP x 20 stack small)."""
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) > 1:
            arr = np.stack([p.asarray()[::step, ::step] for p in tif.pages])
        else:
            arr = tif.asarray()[..., ::step, ::step]
    arr = np.asarray(arr, np.float64)
    return arr.reshape(-1, *arr.shape[-2:])


def time_zero(cfg, frames=None, env=None):
    """Unix time the plots count from: sample preparation if known, else the first record."""
    prepared = (cfg.get("sample") or {}).get("prepared_at")
    if prepared:
        return datetime.datetime.fromisoformat(prepared).timestamp(), "since sample preparation"
    starts = [a[0] for a in ((frames or {}).get("start_unix"), (env or {}).get("unix_time"))
              if a is not None and len(a)]
    return (min(starts) if starts else time.time()), "since run start"


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------
def roi_view(img, roi):
    """img[roi] for roi = (x, y, w, h) in pixels; None = whole frame."""
    if roi is None:
        return img
    x, y, w, h = (int(v) for v in roi)
    return img[y:y + h, x:x + w]


def clipped_mean(a, k=5.0):
    """Mean without cosmic-ray and hot-pixel outliers (> k robust sigma from the median)."""
    a = np.asarray(a, np.float64).ravel()
    m = np.median(a)
    s = 1.4826 * np.median(np.abs(a - m))
    return float(a[np.abs(a - m) <= k * s].mean()) if s > 0 else float(m)


def background_surface(img, order=2, clip=3.0, iterations=3, max_samples=250_000):
    """Polynomial surface through the bias of a frame, refitted without outliers.

    The EMCCD baseline is not flat (amplifier glow ramps towards the readout corner, June
    data: +40 counts in x, +113 in y), so a single pedestal would bias the threshold.
    """
    h, w = img.shape
    step = max(1, int(math.sqrt(h * w / max_samples)))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    powers = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    xn, yn = xs.ravel() / w - 0.5, ys.ravel() / h - 0.5
    a = np.stack([xn ** i * yn ** j for i, j in powers], axis=1)
    z = img[::step, ::step].ravel()
    keep = np.isfinite(z)
    for _ in range(iterations):
        coef = np.linalg.lstsq(a[keep], z[keep], rcond=None)[0]
        r = z - a @ coef
        keep = np.abs(r) < clip * 1.4826 * np.median(np.abs(r[keep]))
    xg, yg = (np.arange(w) / w - 0.5)[None, :], (np.arange(h) / h - 0.5)[:, None]
    return sum(c * xg ** i * yg ** j for c, (i, j) in zip(coef, powers))


def photon_events(img, nsigma=5.0):
    """Threshold one EMCCD frame: {"mask", "sigma"} with sigma = read noise in ADU.

    Photon events only add to a pixel, so the half of the residual below zero is pure read
    noise; the median of a half-normal |x| is 0.6745 sigma.
    """
    resid = img - background_surface(img)
    low = -resid[resid < 0]
    sigma = float(np.median(low) / 0.6745) if low.size else float("nan")
    return {"mask": resid > nsigma * sigma, "sigma": sigma}


def event_stats(mask, roi):
    """Events in a ROI; lam = -ln(1 - p) undoes pixels hit twice (Poisson), p = hit fraction."""
    sub = roi_view(mask, roi)
    n, pixels = int(sub.sum()), int(sub.size)
    p = n / pixels
    return {"events": n, "pixels": pixels, "p": p, "lam": -math.log(1 - p) if p < 1 else math.inf}


def temporal_ptc(bright_path, dark_path, step=4, nbins=40):
    """Mean-variance fit per pixel over the frames of a stack (fixed pattern drops out).

    var_bright - var_dark = K * (mean_bright - mean_dark), K in ADU per e-.
    Returns {"adu_per_e", "e_per_adu", "read_noise_adu", "read_noise_e", "bins_mean", "bins_var"}.
    Cross-check for cal_readnoise, which reads ~20 % low on ideal synthetic data.
    """
    b, d = read_stack(bright_path, step), read_stack(dark_path, step)
    mean = (b.mean(0) - d.mean(0)).ravel()
    var = (b.var(0, ddof=1) - d.var(0, ddof=1)).ravel()
    read_noise_adu = float(np.sqrt(np.median(d.var(0, ddof=1))))
    lo, hi = np.percentile(mean, [1, 99])
    edges = np.linspace(lo, lo + 0.8 * (hi - lo), nbins + 1)   # top 20 % may saturate
    idx = np.digitize(mean, edges) - 1
    bm, bv = [], []
    for i in range(nbins):
        sel = idx == i
        if sel.sum() >= 20:
            bm.append(np.median(mean[sel]))
            bv.append(np.median(var[sel]))
    bm, bv = np.array(bm), np.array(bv)
    k = float(np.polyfit(bm, bv, 1)[0]) if len(bm) >= 3 else float("nan")
    return {"adu_per_e": k, "e_per_adu": 1 / k if k > 0 else float("nan"),
            "read_noise_adu": read_noise_adu,
            "read_noise_e": read_noise_adu / k if k > 0 else float("nan"),
            "bins_mean": bm.tolist(), "bins_var": bv.tolist()}


def _write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _hours(unix, t0):
    return (np.asarray(unix, np.float64) - t0) / 3600


def _window_mean(t, y, window_h):
    """Mean of y in consecutive windows of `window_h` hours: (centres, means)."""
    if not window_h or len(t) == 0:
        return np.array([]), np.array([])
    edges = np.arange(np.nanmin(t), np.nanmax(t) + window_h, window_h)
    idx = np.digitize(t, edges) - 1
    keep = [i for i in range(len(edges) - 1) if np.any(idx == i)]
    return (np.array([edges[i] + window_h / 2 for i in keep]),
            np.array([np.nanmean(y[idx == i]) for i in keep]))


def analyse_timecourse(run_dir, cfg):
    """Per-frame signal of a time course. Returns (figures, summary)."""
    run_dir = Path(run_dir)
    frames = read_table(run_dir / "frames.csv")
    a = cfg.get("analysis", {})
    label = a.get("label", "frame")
    if not frames:
        return {}, {"frames": 0}
    sel = np.flatnonzero(frames["label"].astype(str) == label)
    if not len(sel):
        return {}, {"frames": 0}
    rois = cfg.get("rois") or {}
    sample, background = rois.get("sample"), rois.get("background")
    mode = a.get("mode", "integrated")
    t0, t_label = time_zero(cfg, frames)
    exp_s = frames["exposure_ms"][sel] / 1000
    t = _hours((frames["start_unix"][sel] + frames["end_unix"][sel]) / 2, t0)

    rows, acc, first = [], None, None
    for k, i in enumerate(sel):
        img = load_image(run_dir / frames["file"][i])
        first = img if first is None else first
        row = {"index": int(frames["index"][i]), "t_h": float(t[k]), "exposure_s": float(exp_s[k])}
        if mode == "photon_counting":
            ev = photon_events(img, a.get("nsigma", 5.0))
            s = event_stats(ev["mask"], sample)
            row.update(sigma_adu=ev["sigma"], events_sample=s["events"], pixels_sample=s["pixels"],
                       p_sample=s["p"], lam_sample=s["lam"])
            bg_lam = 0.0
            if background is not None:
                b = event_stats(ev["mask"], background)
                row.update(events_background=b["events"], pixels_background=b["pixels"],
                           lam_background=b["lam"])
                bg_lam = b["lam"]
            row["rate"] = (s["lam"] - bg_lam) / exp_s[k]
            plane = ev["mask"].astype(np.float64)
        else:
            sig = clipped_mean(roi_view(img, sample))
            bg = (clipped_mean(roi_view(img, background)) if background is not None
                  else float(a.get("offset_adu") or 0.0))
            row.update(sample_adu=sig, background_adu=bg, rate=(sig - bg) / exp_s[k])
            if a.get("e_per_adu"):
                row["rate_e"] = row["rate"] * a["e_per_adu"] / a.get("em_gain", 1)
            plane = img - bg
        acc = plane if acc is None else acc + plane
        rows.append(row)
    _write_rows(run_dir / "analysis.csv", rows)
    acc_name = "event_map" if mode == "photon_counting" else "mean_image"
    acc_img = acc if mode == "photon_counting" else acc / len(rows)
    tifffile.imwrite(run_dir / f"{acc_name}.tif", acc_img.astype(np.float32))

    rate = np.array([r["rate"] for r in rows])
    use_e = mode == "integrated" and a.get("e_per_adu")
    unit = ("events / px / s" if mode == "photon_counting"
            else "e- / px / s" if use_e else "ADU / px / s")
    y = np.array([r["rate_e"] for r in rows]) if use_e else rate
    summary = {"frames": len(rows), "mode": mode, "unit": unit,
               "total_exposure_h": round(float(exp_s.sum()) / 3600, 3),
               "time_axis": t_label, "rois": rois, "warnings": []}
    if sample is None:
        summary["warnings"].append("no sample ROI set: whole frame used")
    if background is None and mode == "integrated" and not a.get("offset_adu"):
        summary["warnings"].append("no background ROI or offset: rates include the camera offset")

    if mode == "photon_counting":
        p_max = max(r["p_sample"] for r in rows)
        summary["max_hit_fraction"] = round(p_max, 4)
        if p_max > 0.2:
            summary["warnings"].append(
                f"up to {p_max:.0%} of sample pixels fire per frame: above ~20 % thresholding "
                "undercounts, shorten the exposure")
        n_s = sum(r["events_sample"] for r in rows)
        exposure_px_s = sum(r["pixels_sample"] * r["exposure_s"] for r in rows)
        rate_s, err = n_s / exposure_px_s, math.sqrt(n_s) / exposure_px_s
        if background is not None:
            n_b = sum(r["events_background"] for r in rows)
            exposure_b = sum(r["pixels_background"] * r["exposure_s"] for r in rows)
            rate_s -= n_b / exposure_b
            err = math.hypot(err, math.sqrt(n_b) / exposure_b)
        summary.update(events_sample=n_s, rate=rate_s, rate_error=err)
        # ponytail: threshold efficiency exp(-T/g) is not corrected (needs the EM gain g in ADU
        # per e-); events undercount by that factor, ~10 % at 5 sigma when g is ~50x the read noise.
        summary["note"] = "events above threshold, not corrected for threshold efficiency"
    else:
        summary.update(rate=float(np.mean(y)),
                       rate_error=float(np.std(y, ddof=1) / math.sqrt(len(y))) if len(y) > 1
                       else float("nan"))
    has_background = background is not None or (mode == "integrated" and a.get("offset_adu"))
    err = summary["rate_error"]
    if has_background and err and math.isfinite(err) and err > 0:
        summary["significance_sigma"] = round(summary["rate"] / err, 2)

    figs = {"timecourse": _fig_timecourse(t, y, unit, t_label, rows, mode, a.get("window_min")),
            "images": _fig_images(first, load_image(run_dir / frames["file"][sel[-1]]), acc_img,
                                  acc_name, rois)}
    return figs, summary


def _fig_timecourse(t, y, unit, t_label, rows, mode, window_min):
    n = 2 if mode == "photon_counting" else 1
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(t, y, "o", ms=3, alpha=0.6, label="per frame")
    tw, yw = _window_mean(t, y, (window_min or 0) / 60)
    if len(tw):
        ax.step(tw, yw, where="mid", lw=2, label=f"{window_min:g} min mean")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel(f"sample - background\n[{unit}]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    if mode == "photon_counting":
        ax = axes[1, 0]
        ax.plot(t, [r["p_sample"] for r in rows], "o", ms=3)
        ax.axhline(0.2, color="r", ls="--", lw=1, label="20 %: thresholding undercounts above")
        ax.set_ylabel("hit fraction p")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(f"time {t_label} [h]")
    fig.tight_layout()
    return fig


def _stretch(img):
    lo, hi = np.percentile(img, [0.5, 99.7])
    return dict(vmin=lo, vmax=hi if hi > lo else lo + 1)


def _draw_rois(ax, rois):
    colors = {"sample": "lime", "background": "orange"}
    for name, roi in (rois or {}).items():
        if roi is None:
            continue
        x, y, w, h = roi
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=colors.get(name, "cyan"), lw=1.5))
        ax.text(x, y, name, color=colors.get(name, "cyan"), fontsize=8, va="bottom")


def _fig_images(first, last, acc, acc_name, rois):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, (first, last, acc), ("first frame", "last frame",
                                                          acc_name.replace("_", " "))):
        im = ax.imshow(img, cmap="gray", **_stretch(img))
        _draw_rois(ax, rois)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def analyse_ptc(run_dir, cfg):
    """PTC sessions (controller + temporal cross-check) and the dark-current ladder."""
    run_dir = Path(run_dir)
    table, ptc_curves = [], []
    for s in cfg.get("ptc_sessions", []):
        d = run_dir / s["dir"]
        res = json.loads((d / "result.json").read_text())
        row = {"gain_ui": s["gain"], "exposure_ms": s.get("exposure_ms"),
               "e_per_adu": res.get("gain"), "read_noise_e": res.get("readnoise"),
               "offset_adu": res.get("offset")}
        if (d / "bright_stack.tif").exists() and (d / "dark_stack.tif").exists():
            tp = temporal_ptc(d / "bright_stack.tif", d / "dark_stack.tif")
            row.update(e_per_adu_temporal=tp["e_per_adu"], read_noise_e_temporal=tp["read_noise_e"])
            ptc_curves.append((s["gain"], tp))
        table.append(row)
    _write_rows(run_dir / "analysis.csv", table)
    summary = {"ptc": table, "warnings": []}
    figs = {}
    if ptc_curves:
        fig, ax = plt.subplots(figsize=(7, 5))
        for gain, tp in ptc_curves:
            bm = np.array(tp["bins_mean"])
            ax.plot(bm, tp["bins_var"], "o", ms=4, label=f"gain {gain}")
            ax.plot(bm, tp["adu_per_e"] * bm, "-", lw=1,
                    label=f"fit: {tp['e_per_adu']:.3g} e-/ADU")
        ax.set_xlabel("mean signal above dark [ADU]")
        ax.set_ylabel("temporal variance above dark [ADU²]")
        ax.set_title("Photon transfer (per-pixel, over the frames of the stack)")
        ax.legend()
        ax.grid(alpha=0.3)
        figs["ptc_temporal"] = fig

    frames = read_table(run_dir / "frames.csv")
    if frames:
        dark = np.flatnonzero(np.char.startswith(frames["label"].astype(str), "dark_"))
        if len(dark):
            f, s = _dark_ladder(run_dir, cfg, frames, dark, table)
            figs.update(f)
            summary.update(s)
    return figs, summary


def _dark_ladder(run_dir, cfg, frames, dark, table):
    """Dark current from the median and from the variance of frame pairs vs exposure."""
    groups = {}
    for i in dark:
        groups.setdefault(float(frames["exposure_ms"][i]) / 1000, []).append(
            load_image(run_dir / frames["file"][i]))
    t, med, var = [], [], []
    for exp_s, imgs in sorted(groups.items()):
        t.append(exp_s)
        med.append(np.mean([np.median(im) for im in imgs]))
        # (a - b) / sqrt(2): offset and fixed pattern cancel, temporal noise stays
        pairs = [(imgs[k] - imgs[k + 1]) for k in range(len(imgs) - 1)]
        var.append(np.mean([(1.4826 * np.median(np.abs(p - np.median(p)))) ** 2 / 2
                            for p in pairs]) if pairs else np.nan)
    t, med, var = np.array(t), np.array(med), np.array(var)
    gain = cfg.get("dark_ladder_gain")
    row = next((r for r in table if r["gain_ui"] == gain), table[0] if table else {})
    usable = [v for v in (row.get("e_per_adu_temporal"), row.get("e_per_adu"))
              if v is not None and math.isfinite(v) and v > 0]
    e_per_adu = usable[0] if usable else None
    summary = {"dark_ladder_exposures_s": t.tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t, med, "o")
    axes[1].plot(t, var, "o")
    if len(t) >= 2:
        slope_med, icpt_med = np.polyfit(t, med, 1)
        ok = np.isfinite(var)
        slope_var, icpt_var = np.polyfit(t[ok], var[ok], 1) if ok.sum() >= 2 else (np.nan, np.nan)
        tt = np.linspace(0, t.max(), 50)
        axes[0].plot(tt, slope_med * tt + icpt_med, "-")
        axes[1].plot(tt, slope_var * tt + icpt_var, "-")
        summary.update(dark_slope_adu_per_s=float(slope_med),
                       dark_variance_slope_adu2_per_s=float(slope_var))
        if e_per_adu:
            d_mean = slope_med * e_per_adu
            d_var = slope_var * e_per_adu ** 2
            summary.update(e_per_adu_used=e_per_adu,
                           dark_current_e_px_s_from_mean=float(d_mean),
                           dark_current_e_px_s_from_variance=float(d_var))
            sheet = cfg.get("datasheet_dark_e_px_s")
            if sheet:
                summary["ratio_to_datasheet_mean"] = round(float(d_mean / sheet), 2)
                summary["ratio_to_datasheet_variance"] = round(float(d_var / sheet), 2)
            axes[0].set_title(f"dark current {d_mean:.2g} e-/px/s (from the median)")
            axes[1].set_title(f"dark current {d_var:.2g} e-/px/s (from the variance)")
    axes[0].set_ylabel("median dark level [ADU]")
    axes[1].set_ylabel("temporal variance [ADU²]")
    for ax in axes:
        ax.set_xlabel("exposure [s]")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return {"dark_ladder": fig}, summary


# --------------------------------------------------------------------------------------------
# Figures and report
# --------------------------------------------------------------------------------------------
def fig_environment(env, t0, t_label="since run start"):
    """Camera sensor temperature, enclosure temperature + humidity, and light over time."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    if not env:
        axes[0].text(0.5, 0.5, "no environment samples yet", ha="center",
                     transform=axes[0].transAxes)
        return fig
    hours = _hours(env["unix_time"], t0)
    rh_ax = axes[1].twinx()
    for label in dict.fromkeys(env["source"].astype(str)):
        m = env["source"].astype(str) == label
        axes[0].plot(hours[m], env["camera_temperature_c"][m], label=f"{label} sensor")
        if np.isfinite(env["camera_target_c"][m]).any():
            axes[0].plot(hours[m], env["camera_target_c"][m], "--", lw=1, label=f"{label} set point")
        axes[1].plot(hours[m], env["enclosure_temperature_c"][m], label=f"{label} °C")
        rh_ax.plot(hours[m], env["humidity_pct"][m], ":", label=f"{label} % RH")
        axes[2].plot(hours[m], env["lux"][m], label=label)
    axes[0].set_ylabel("camera sensor [°C]")
    axes[1].set_ylabel("enclosure [°C]")
    rh_ax.set_ylabel("humidity [% RH]")
    axes[2].set_ylabel("light [lux]")
    axes[2].set_xlabel(f"time {t_label} [h]")
    for ax in axes:
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=8)
    if rh_ax.get_legend_handles_labels()[0]:
        rh_ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _summary_lines(summary):
    lines = []
    for key, value in summary.items():
        if key == "ptc" and value:
            cols = list(dict.fromkeys(k for row in value for k in row))
            lines += ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            lines += ["| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |" for row in value]
            lines.append("")
        elif key == "warnings":
            lines += [f"- **warning:** {w}" for w in value]
        else:
            lines.append(f"- **{key}:** {_fmt(value)}")
    return lines


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def write_readme(run_dir, cfg, summary):
    run_dir = Path(run_dir)
    frames = read_table(run_dir / "frames.csv")
    sample = cfg.get("sample") or {}
    lines = [f"# {cfg.get('title', run_dir.name)}", "", cfg.get("description", ""), "",
             "| | |", "|---|---|",
             f"| Run | `{run_dir.name}` |",
             f"| Experiment | {cfg.get('experiment', '')} |",
             f"| Server / detector | {cfg.get('server', '')} / {cfg.get('detector', '')} |",
             f"| Frames | {len(frames.get('file', []))} |"]
    if frames:
        lines.append(f"| First / last frame | {frames['start_iso'][0]} / {frames['start_iso'][-1]} |")
    lines += [f"| Sample {k} | {v} |" for k, v in sample.items() if v not in (None, "")]
    if cfg.get("dry_run"):
        lines.append("| **Dry run** | short test exposures, not data |")
    lines += ["", "## Summary", ""] + _summary_lines(summary)
    lines += ["", "## Figures", ""]
    lines += [f"![{p.stem}](figures/{p.name})" for p in sorted((run_dir / "figures").glob("*.png"))]
    for s in cfg.get("ptc_sessions", []):
        figs = sorted((run_dir / s["dir"] / "figures").glob("*.png"))
        if figs:
            lines += ["", f"### ImSwitch PTC figures, gain {s['gain']}", ""]
            lines += [f"![{p.stem}]({p.relative_to(run_dir).as_posix()})" for p in figs]
    lines += ["", "## Settings", "", "```json",
              json.dumps({k: v for k, v in cfg.items() if k not in ("title", "description")},
                         indent=2, default=str), "```", "",
              "## Files", "",
              "- `config.json` settings, `events.log` timeline, `frames.csv` one row per frame",
              "- `environment.csv` camera and enclosure log, `analysis.csv` per-frame results",
              f"- `raw/` {len(list((run_dir / 'raw').glob('*.tif')))} full bit-depth TIFFs",
              "- `report.ipynb` redraws every figure from this folder (run all cells)", ""]
    (run_dir / "README.md").write_text("\n".join(lines))


def redraw(run_dir, show=False):
    """Recompute every figure, `summary.json` and `README.md` from the files of a run."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    frames, env = read_table(run_dir / "frames.csv"), read_table(run_dir / "environment.csv")
    t0, t_label = time_zero(cfg, frames, env)
    figs, summary = {}, {}
    if cfg.get("kind") == "timecourse":
        figs, summary = analyse_timecourse(run_dir, cfg)
    elif cfg.get("kind") == "ptc":
        figs, summary = analyse_ptc(run_dir, cfg)
    if env:
        figs["environment"] = fig_environment(env, t0, t_label)
        for col in ("camera_temperature_c", "enclosure_temperature_c", "humidity_pct", "lux"):
            v = env[col][np.isfinite(env[col])]
            if v.size:
                summary[f"{col}_min_max"] = [round(float(v.min()), 3), round(float(v.max()), 3)]
    for name, fig in figs.items():
        fig.savefig(run_dir / "figures" / f"{name}.png", dpi=120)
        if not show:
            plt.close(fig)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_readme(run_dir, cfg, summary)
    if show:
        plt.show()
    return summary


def show_frame(run_dir, index=0, grid=True):
    """Show one saved frame with pixel axes and the current ROIs, to choose ROIs."""
    run_dir = Path(run_dir)
    frames = read_table(run_dir / "frames.csv")
    img = load_image(run_dir / frames["file"][index])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray", **_stretch(img))
    _draw_rois(ax, load_config(run_dir).get("rois"))
    ax.set_title(f"{frames['file'][index]}  ({img.shape[1]} x {img.shape[0]} px)")
    if grid:
        ax.grid(color="c", alpha=0.3)
    plt.show()
    return img


def set_rois(run_dir, rois):
    """Store ROIs {"sample": (x, y, w, h), "background": (x, y, w, h)} in config.json."""
    path = Path(run_dir) / "config.json"
    cfg = json.loads(path.read_text())
    cfg["rois"] = rois
    path.write_text(json.dumps(cfg, indent=2, default=str))


_image_cache = {}


def show_live(run_dir, acq=None):
    """One dashboard refresh, drawn from the files on disk."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    frames, env = read_table(run_dir / "frames.csv"), read_table(run_dir / "environment.csv")
    t0, t_label = time_zero(cfg, frames, env)
    if acq is not None:
        st = acq.state
        print(f"{st['phase']} | step {st['step']} frame {st['frame']} | {st['detail']}"
              + (f" | ERROR {st['error']}" if st["error"] else ""))
    fig_environment(env, t0, t_label)
    plt.show()
    if frames:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [2, 1]})
        for label in dict.fromkeys(frames["label"].astype(str)):
            m = frames["label"].astype(str) == label
            ax0.plot(_hours(frames["start_unix"][m], t0), frames["median"][m], "o-", ms=3,
                     label=label)
        ax0.set_xlabel(f"time {t_label} [h]")
        ax0.set_ylabel("frame median [ADU]")
        ax0.legend(fontsize=8)
        ax0.grid(alpha=0.3)
        path = run_dir / frames["file"][-1]
        if path not in _image_cache:
            _image_cache.clear()
            img = load_image(path)
            _image_cache[path] = img[::max(1, img.shape[0] // 1000), ::max(1, img.shape[1] // 1000)]
        img = _image_cache[path]
        ax1.imshow(img, cmap="gray", **_stretch(img))
        ax1.set_title(frames["file"][-1], fontsize=9)
        ax1.axis("off")
        fig.tight_layout()
        plt.show()
    log = run_dir / "events.log"
    if log.exists():
        print("\n".join(log.read_text().splitlines()[-6:]))


def dashboard(run_dir, acq=None, refresh_s=10):
    """Refresh show_live() until the acquisition ends. Interrupting (■) stops only the display."""
    from IPython.display import clear_output
    try:
        while True:
            clear_output(wait=True)
            show_live(run_dir, acq)
            if acq is not None and not acq.running:
                print("acquisition finished:", acq.state["phase"])
                break
            time.sleep(refresh_s)
    except KeyboardInterrupt:
        print("Dashboard stopped, the acquisition keeps running.")
