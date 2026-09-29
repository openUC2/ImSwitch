"""In-situ stage calibration for the ShitScope (PCB planar stage), using the camera.

About a minute, all measured by registering camera frames before and after
each move (pixel size assumed calibrated). Per axis:

1. probe     +N +N -N -N µstep moves at the configured drive (escalating only
             if nothing moves): image direction of the axis, first µm/µstep
2. threshold drive stepped down 20 % per level with the same moves; only the
             second move of each direction counts (no reversal dead band). The lowest level
             where every move reaches >= `travel_ok` of the expected travel
             is the threshold, recommended drive = `margin` x threshold
3. steps     runs of +, -, + moves at the recommended drive: µm/µstep and
             the travel lost on each reversal (-> approach overshoot)
then the stage returns to the start and the last frame is compared with the
first: placement error and sled rotation (the split magnet array can yaw).

Findings behind the defaults: openUC2-PCB-XY-Stage docs E1-E6 (bare PCB:
X threshold ~4000-5000, Y ~12000, reversal loss ~230 µm, 130 µm/µstep).
No ImSwitch imports: `grab()` returns a fresh frame taken after the last move,
`move_steps(dx, dy)` is a blocking raw move in µsteps (no approach logic),
`set_power(px, py)` sets the per-axis drive (0..32767).
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import tile_registration
from .shitscope_scan import downsample, to_gray

REGISTER_MAX_PX = 1024      # frames are downsampled to this size for registration

AXES = ("X", "Y")


def rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation of b relative to a (degrees, small angles): phase correlation of
    the log-polar Fourier magnitudes, which do not depend on translation."""
    from skimage.registration import phase_cross_correlation
    from skimage.transform import warp_polar

    def spectrum(im):
        im = im - im.mean()
        win = np.outer(np.hanning(im.shape[0]), np.hanning(im.shape[1]))
        return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(im * win))))

    r = min(a.shape) // 2
    n_ang = 1440                                                # 0.25° per row
    pa = warp_polar(spectrum(a), radius=r, output_shape=(n_ang, r))[:, r // 8:]
    pb = warp_polar(spectrum(b), radius=r, output_shape=(n_ang, r))[:, r // 8:]
    shift, _, _ = phase_cross_correlation(pa, pb, upsample_factor=10, normalization=None)
    ang = shift[0] * 360.0 / n_ang
    return float((ang + 90) % 180 - 90)                         # spectra repeat every 180°


class StageCalibration:
    """Run with start()/run(); poll status()."""

    def __init__(self, grab: Callable[[], np.ndarray], move_steps: Callable[[int, int], None],
                 set_power: Callable[[int, int], None], um_per_px: float,
                 start_power: Tuple[int, int] = (18000, 24000), max_power: int = 30000,
                 probe_steps: int = 4, trials: int = 2, travel_ok: float = 0.8, margin: float = 1.5,
                 min_power: int = 1500, logger: Optional[Callable[[str], None]] = None):
        self._grab, self._move, self._set_power = grab, move_steps, set_power
        self.um_per_px = float(um_per_px)
        self.start_power = {"X": int(start_power[0]), "Y": int(start_power[1])}
        self.max_power, self.min_power = int(max_power), int(min_power)
        self.n, self.trials = int(probe_steps), int(trials)
        self.travel_ok, self.margin = float(travel_ok), float(margin)
        self._log_fn = logger or (lambda m: None)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.power = dict(self.start_power)          # currently set drive per axis
        self.state, self.phase, self.progress, self.error = "idle", "", 0.0, None
        self.log: List[str] = []
        self.trials_log: List[Dict] = []
        self.result: Optional[Dict] = None
        self._net = [0, 0]                            # commanded µsteps since start
        self._ds = 1
        self._last = None

    # ------------------------------------------------------------ control
    def start(self):
        self.state = "running"
        self._thread = threading.Thread(target=self.run, daemon=True, name="ShitScopeCalibration")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def status(self) -> Dict:
        return {"state": self.state, "phase": self.phase, "progress": round(self.progress, 3),
                "error": self.error, "log": self.log[-40:], "result": self.result}

    def _say(self, msg: str):
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
        self._log_fn(msg)

    # ------------------------------------------------------------ primitives
    def _frame(self) -> np.ndarray:
        f = to_gray(np.asarray(self._grab(), float))
        if self._last is None:
            self._ds = max(1, int(np.ceil(max(f.shape) / REGISTER_MAX_PX)))
        return downsample(f, self._ds).astype(float)

    def _set(self, axis: str, power: int):
        self.power[axis] = int(power)
        self._set_power(self.power["X"], self.power["Y"])

    def _step(self, axis: str, n: int, direction: Optional[np.ndarray] = None,
              um_per_step: Optional[float] = None) -> Dict:
        """Move `n` µsteps on `axis`, register the frames before/after.
        Returns the image shift (px, (dy, dx)), travel along `direction` (µm)."""
        if self._stop.is_set():
            raise InterruptedError("stopped")
        dx, dy = (n, 0) if axis == "X" else (0, n)
        self._move(dx, dy)
        self._net[0] += dx; self._net[1] += dy
        cur = self._frame()
        px = self.um_per_px * self._ds
        expected = (direction * n * um_per_step / px) if (direction is not None and um_per_step) else np.zeros(2)
        shift, q = tile_registration.register_pair(self._last, cur, expected)
        self._last = cur
        along = float(np.dot(shift, direction) * px) if direction is not None else None
        rec = {"axis": axis, "power": self.power[axis], "steps": n, "shift_px": [float(v) for v in shift],
               "travel_um": along, "quality": q}
        self.trials_log.append(rec)
        return rec

    # ------------------------------------------------------------ phases
    def _pairs(self, axis, direction=None, ups=None) -> List[Tuple[int, Dict]]:
        """+n, +n, -n, -n (x trials). Only the second move of each direction is
        returned: the first one after a reversal loses the friction dead band
        (measured separately in _steps), which would bias travel low."""
        out = []
        for _ in range(self.trials):
            for sign in (+1, -1):
                self._step(axis, sign * self.n, direction, ups)
                out.append((sign, self._step(axis, sign * self.n, direction, ups)))
        return out

    def _probe(self, axis: str) -> Optional[Dict]:
        """Direction of the axis in the image and a first µm/µstep. Escalates
        the drive if nothing moves; None if the axis never moves."""
        levels = sorted({self.power[axis], min(24000, self.max_power), self.max_power})
        levels = [p for p in levels if p >= self.power[axis]]
        for p in levels:
            self._set(axis, p)
            vecs = [sign * np.array(r["shift_px"]) for sign, r in self._pairs(axis)]
            v = np.mean(vecs, axis=0)
            mag = np.hypot(*v)
            consistent = all(np.dot(w, v) > 0.5 * np.hypot(*w) * mag for w in vecs if np.hypot(*w) > 1)
            self._say(f"{axis} probe @ POWER {p}: {mag * self.um_per_px * self._ds / self.n:.0f} µm/µstep"
                      f"{'' if consistent else ' (inconsistent directions)'}")
            if mag * self.um_per_px * self._ds > 20 and consistent:
                return {"power": p, "direction": v / mag,
                        "um_per_step": mag * self.um_per_px * self._ds / self.n}
        return None

    def _passes(self, axis: str, power: int, direction, ups: float) -> Tuple[bool, List[float]]:
        self._set(axis, power)
        travels = [sign * r["travel_um"] for sign, r in self._pairs(axis, direction, ups)]
        ok = all(t >= self.travel_ok * self.n * ups for t in travels)
        return ok, travels

    def _threshold(self, axis: str, probe: Dict) -> Dict:
        d, ups = probe["direction"], probe["um_per_step"]
        # the probe level must pass the stricter travel test; otherwise go up
        p, levels = probe["power"], []
        ok, tr = self._passes(axis, p, d, ups); levels.append((p, ok, tr))
        while not ok and p < self.max_power:
            p = min(self.max_power, int(p * 1.25)); ok, tr = self._passes(axis, p, d, ups); levels.append((p, ok, tr))
        if not ok:
            return {"threshold": None, "recommended": self.max_power, "levels": levels}
        lowest = p
        while p > self.min_power:
            p = max(self.min_power, int(round(p * 0.8 / 250) * 250))
            ok, tr = self._passes(axis, p, d, ups); levels.append((p, ok, tr))
            self._say(f"{axis} POWER {p}: {'ok' if ok else 'too weak'} ({np.mean(tr):.0f} µm of {self.n * ups:.0f})")
            if not ok:
                break
            lowest = p
        rec = int(min(self.max_power, max(lowest, round(lowest * self.margin / 500) * 500)))
        return {"threshold": lowest, "recommended": rec, "levels": levels}

    def _steps(self, axis: str, probe: Dict, power: int) -> Dict:
        """+4, -4, +4 runs of n µsteps: µm/µstep from steps that continue the
        direction, reversal loss from the first step after each reversal."""
        self._set(axis, power)
        d, ups = probe["direction"], probe["um_per_step"]
        steady, first = [], []
        for run, sign in enumerate((+1, -1, +1)):
            for i in range(4):
                t = sign * self._step(axis, sign * self.n, d, ups)["travel_um"]
                if i > 0:
                    steady.append(t)
                elif run > 0:            # run 0 follows unknown moves: neither steady nor a clean reversal
                    first.append(t)
        self._step(axis, -4 * self.n, d, ups)          # back to where the run started
        ups2 = float(np.mean(steady)) / self.n
        loss = max(0.0, float(np.mean(steady) - np.mean(first)))
        overshoot = int(np.clip(np.ceil(loss / ups2) + 2 if ups2 > 0 else 8, 4, 16))
        return {"um_per_step": ups2, "spread_pct": float(100 * np.std(steady) / max(np.mean(steady), 1e-9)),
                "reversal_loss_um": loss, "overshoot_steps": overshoot}

    # ------------------------------------------------------------ run
    def run(self) -> Dict:
        t0 = time.time()
        warnings, axes, flags = [], {}, set()
        try:
            self.state = "running"
            self._set_power(self.power["X"], self.power["Y"])
            self._last = first_frame = self._frame()
            # texture: two still frames of a textured sample correlate, pure noise does not
            _, texture = tile_registration.register_pair(first_frame, self._frame(), np.zeros(2))
            if texture < 0.5:
                raise RuntimeError("Too little texture in the image to measure motion: "
                                   "move to a structured area of the sample and retry.")
            for i, axis in enumerate(AXES):
                self.phase, self.progress = f"{axis}: probe", i / 2
                probe = self._probe(axis)
                if probe is None:
                    warnings.append(f"{axis} does not move even at POWER {self.max_power}: the sled may be "
                                    "rotated or stuck, the magnets too far from the traces (spacer/glass too "
                                    "thick), or friction too high. Check the sled, then retry.")
                    axes[axis] = {"moves": False}
                    flags.add("no_motion")
                    self._set(axis, self.start_power[axis])
                    continue
                self.phase, self.progress = f"{axis}: threshold", i / 2 + 0.15
                thr = self._threshold(axis, probe)
                if thr["threshold"] is None:
                    flags.add("weak")
                    warnings.append(f"{axis} moves, but never reaches {self.travel_ok:.0%} of the commanded "
                                    f"travel even at POWER {self.max_power}: too little force margin.")
                elif thr["recommended"] >= 0.9 * self.max_power:
                    warnings.append(f"{axis} needs POWER {thr['recommended']} for a {self.margin}x margin: "
                                    "close to the maximum (heat, little reserve).")
                self.phase, self.progress = f"{axis}: step size", i / 2 + 0.35
                steps = self._steps(axis, probe, thr["recommended"])
                angle = float(np.degrees(np.arctan2(probe["direction"][0], probe["direction"][1])))
                axes[axis] = {"moves": True, "image_angle_deg": angle, "probe_power": probe["power"],
                              "threshold_power": thr["threshold"], "recommended_power": thr["recommended"],
                              "levels": [{"power": p, "ok": ok, "travel_um": [float(v) for v in tr]}
                                         for p, ok, tr in thr["levels"]], **steps}
                self._say(f"{axis}: threshold {thr['threshold']}, recommended POWER {thr['recommended']}, "
                          f"{steps['um_per_step']:.1f} µm/µstep, reversal loss {steps['reversal_loss_um']:.0f} µm")
            # back to the start: every phase returns except for accumulated commanded steps
            self.phase, self.progress = "health check", 0.9
            if self._net != [0, 0]:
                self._move(-self._net[0], -self._net[1]); self._net = [0, 0]
            last = self._frame()
            closure, q = tile_registration.register_pair(first_frame, last, np.zeros(2))
            rot = rotation_deg(first_frame, last)
            if abs(rot) > 0.5:
                flags.add("rotated")
                warnings.append(f"The sample rotated by {rot:.1f}° during calibration: the sled yawed "
                                "(the split magnet array can slip by a cycle). Re-seat the sled.")
            if all(a.get("moves") for a in axes.values()):
                ang = abs(((axes["Y"]["image_angle_deg"] - axes["X"]["image_angle_deg"]) + 180) % 360 - 180)
                if abs(ang - 90) > 10:
                    flags.add("rotated")
                    warnings.append(f"X and Y move at {ang:.0f}° to each other in the image (expected 90°): "
                                    "the sled is rotated on the stator.")
            rec = {}
            for axis in AXES:
                a = axes.get(axis, {})
                if a.get("moves"):
                    rec[f"power{axis}"] = a["recommended_power"]
                    rec[f"umPerStep{axis}"] = a["um_per_step"]
            if rec and all(a.get("moves") for a in axes.values()):
                rec["approachOvershootSteps"] = int(max(a["overshoot_steps"] for a in axes.values()))
            self.result = {"ok": not flags, "flags": sorted(flags),
                           "axes": axes, "recommended": rec, "warnings": warnings,
                           "rotation_deg": rot, "closure_um": [float(closure[1] * self.um_per_px * self._ds),
                                                               float(closure[0] * self.um_per_px * self._ds)],
                           "texture": texture, "duration_s": time.time() - t0, "moves": len(self.trials_log)}
            self.state, self.phase, self.progress = "done", "done", 1.0
        except InterruptedError:
            self.state, self.phase = "stopped", "stopped"
        except Exception as exc:
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
            self._say(f"calibration failed: {self.error}")
        finally:
            try:   # leave the drive as found; applying the result is a separate step
                self._set_power(self.start_power["X"], self.start_power["Y"])
                if self._net != [0, 0]:
                    self._move(-self._net[0], -self._net[1])
            except Exception as exc:
                self._say(f"restoring the stage failed: {exc}")
        return self.result
