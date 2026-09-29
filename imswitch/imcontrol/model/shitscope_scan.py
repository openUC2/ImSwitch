"""ShitScope tile scan: move, fresh frame, save, live preview, then register and stitch.

Built for the openUC2 PCB planar stage (see the PCB-stage bench, E0-E6):
the stage misplaces tiles by up to ~0.4 mm but tiles overlap, so tile
positions are recovered from the images (tile_registration, ~3 µm) and the
mosaic is stitched at the measured positions. Stage-specific behaviour
(one approach direction, settle, speed cap) lives in the stage manager.

No ImSwitch imports: `detector` needs getLatestFrame([returnFrameNumber]) and
optionally flushBuffer(s); `stage` needs move(value=(x, y), axis="XY",
is_absolute=True, is_blocking=True) and getPosition() -> {"X", "Y"}.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from . import tile_registration

PREVIEW_MAX_PX = 1200          # live preview width/height budget
ANALYSIS_MAX_PX = 768          # registration works on tiles no larger than this (sub-px anyway)


def plan_grid(x0: float, y0: float, nx: int, ny: int, step_x: float, step_y: float,
              pattern: str = "raster", centered: bool = True) -> List[Dict]:
    """Tile positions in scan order. raster: every row in +X (preferred by the
    PCB stage, one approach direction). snake: every other row reversed."""
    if centered:
        x0 -= (nx - 1) * step_x / 2
        y0 -= (ny - 1) * step_y / 2
    tiles = []
    for iy in range(ny):
        cols = range(nx) if (pattern != "snake" or iy % 2 == 0) else range(nx - 1, -1, -1)
        for ix in cols:
            tiles.append({"ix": ix, "iy": iy, "x": x0 + ix * step_x, "y": y0 + iy * step_y})
    return tiles


def suggested_step(fov_um: float, full_step_um: Optional[float], min_overlap: float) -> float:
    """Largest whole number of stage full steps that keeps `min_overlap`."""
    max_step = fov_um * (1 - min_overlap)
    if not full_step_um or full_step_um <= 0:
        return max_step
    return max(full_step_um, np.floor(max_step / full_step_um) * full_step_um)


def to_gray(frame) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a.mean(axis=-1) if a.shape[-1] in (3, 4) else a[0]
    return a


def downsample(a: np.ndarray, f: int) -> np.ndarray:
    """Block mean by an integer factor (cv2 INTER_AREA is ~10x faster than numpy)."""
    a = np.asarray(a, np.float32)
    if f <= 1:
        return a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    try:
        import cv2
        return cv2.resize(a[:h, :w], (w // f, h // f), interpolation=cv2.INTER_AREA)
    except ImportError:
        a = a[:h, :w]
        return a.reshape(h // f, f, w // f, f, *a.shape[2:]).mean(axis=(1, 3))


def auto_downsample(shape) -> int:
    return max(1, int(np.ceil(max(shape) / ANALYSIS_MAX_PX)))


def fresh_frame(detector, timeout_s: float = 3.0):
    """A frame exposed after the last move finished: flush, then wait for the
    frame number to advance past the first one read (that one may overlap
    the end of the move)."""
    for name in ("flushBuffer", "flushBuffers"):
        if hasattr(detector, name):
            try:
                getattr(detector, name)()
            except Exception:
                pass
            break
    t0, first, frame = time.time(), None, None
    while time.time() - t0 < timeout_s:
        try:
            frame, fid = detector.getLatestFrame(returnFrameNumber=True)
        except TypeError:                      # detector without frame numbers
            time.sleep(0.1)
            return detector.getLatestFrame()
        if frame is not None and fid is not None:
            if first is None:
                first = fid
            elif fid > first:
                return frame
        time.sleep(0.01)
    if frame is None:
        raise TimeoutError("camera delivered no frame")
    return frame


class ShitScopeScan:
    """One scan in a background thread. Poll `status()`; `preview_png()` and
    `result` become available as it goes."""

    def __init__(self, detector, stage, out_dir: str, um_per_px: float, tiles: List[Dict],
                 frame_timeout_s: float = 3.0, return_to_start: bool = True,
                 analysis_downsample: Optional[int] = None, stitch_downsample: int = 2,
                 logger: Optional[Callable[[str], None]] = None,
                 on_done: Optional[Callable[[], None]] = None,
                 home: Optional[Callable[[], None]] = None,
                 speed: Optional[float] = None, settle_s: float = 0.0):
        self.detector, self.stage = detector, stage
        self.out_dir, self.um_per_px, self.plan = out_dir, float(um_per_px), tiles
        self.frame_timeout_s, self.return_to_start = frame_timeout_s, return_to_start
        # None: chosen from the first frame so tiles are <= ANALYSIS_MAX_PX
        self.analysis_ds = max(1, analysis_downsample) if analysis_downsample else None
        self.stitch_ds = max(1, stitch_downsample)
        self._log = logger or (lambda m: None)
        self._on_done = on_done   # e.g. release the camera acquisition handle
        self._home = home         # optional: re-home (-X/-Y stop) before the first tile
        self.speed = speed        # stage speed (steps/s for the PCB stage); None = stage default
        self.settle_s = max(0.0, float(settle_s))   # extra wait after each move, before the frame
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.state, self.error, self.done_tiles = "idle", None, 0
        self.t_start = self.t_end = None
        self.records: List[Dict] = []           # per tile: plan + stage position + file
        self._small: List[np.ndarray] = []       # analysis-resolution tiles
        self._preview_ds = 8
        self.result: Optional[Dict] = None

    # ------------------------------------------------------------ control
    def start(self):
        os.makedirs(os.path.join(self.out_dir, "tiles"), exist_ok=True)
        self.state, self.t_start = "running", time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ShitScopeScan")
        self._thread.start()

    def stop(self, join_s: float = 10.0):
        self._stop.set()
        if self._thread:
            self._thread.join(join_s)

    def status(self) -> Dict:
        n, k = len(self.plan), self.done_tiles
        el = (self.t_end or time.time()) - self.t_start if self.t_start else 0.0
        eta = el / k * (n - k) if k and self.state == "running" else None
        return {"state": self.state, "tile": k, "total": n, "elapsed_s": el, "eta_s": eta,
                "outDir": self.out_dir, "error": self.error,
                "summary": (self.result or {}).get("summary")}

    # ------------------------------------------------------------ helpers
    def _fresh_frame(self):
        return fresh_frame(self.detector, self.frame_timeout_s)

    def _move(self, x: float, y: float):
        kw = {"speed": self.speed} if self.speed else {}
        try:
            self.stage.move(value=(x, y), axis="XY", is_absolute=True, is_blocking=True, **kw)
        except (TypeError, ValueError):            # positioners without combined XY
            self.stage.move(value=x, axis="X", is_absolute=True, is_blocking=True, **kw)
            self.stage.move(value=y, axis="Y", is_absolute=True, is_blocking=True, **kw)

    def _save_json(self, name: str, obj):
        with open(os.path.join(self.out_dir, name), "w") as fh:
            json.dump(obj, fh, indent=1)

    # ------------------------------------------------------------ scan
    def _run(self):
        import tifffile
        try:
            start = self.stage.getPosition()
            if self._home is not None:
                # The plan is in homed coordinates; re-homing clears the drift
                # collected since the last home, and from the -X/-Y stop every
                # move to the first tile is in +X/+Y (the approach direction).
                self.state = "homing"
                self._home()
                self.state = "running"
            meta = {"um_per_px": self.um_per_px, "plan": self.plan,
                    "start": {"X": start.get("X"), "Y": start.get("Y")},
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._save_json("scan.json", meta)
            for k, t in enumerate(self.plan):
                if self._stop.is_set():
                    self.state = "stopped"
                    break
                self._move(t["x"], t["y"])
                if self.settle_s:
                    time.sleep(self.settle_s)
                raw = np.asarray(self._fresh_frame())       # saved as delivered (colour stays colour)
                path = os.path.join("tiles", f"tile_{k:04d}.tif")
                tifffile.imwrite(os.path.join(self.out_dir, path), raw)
                frame = to_gray(raw)                         # registration and preview work on gray
                pos = self.stage.getPosition()
                with self._lock:
                    if k == 0:   # analysis and preview scale from the first frame
                        self.analysis_ds = self.analysis_ds or auto_downsample(frame.shape)
                        n_side = max(max(t_["ix"], t_["iy"]) for t_ in self.plan) + 1
                        self._preview_ds = max(1, int(np.ceil(max(frame.shape) * n_side / PREVIEW_MAX_PX)))
                    self._small.append(downsample(frame, self.analysis_ds))
                    self.records.append({**t, "k": k, "file": path,
                                         "stage_x": pos.get("X"), "stage_y": pos.get("Y")})
                    self.done_tiles = k + 1
                self._save_json("tiles.json", self.records)
            if self.return_to_start and start.get("X") is not None:
                self._move(start["X"], start["Y"])
            if self.state == "running":
                self.state = "analysing"
            if len(self._small) >= 2:
                self.analyse()
            if self.state == "analysing":
                self.state = "done"
        except Exception as exc:                  # keep the tiles, report the error
            self.error, self.state = f"{type(exc).__name__}: {exc}", "error"
            self._log(f"ShitScope scan failed: {self.error}")
        finally:
            self.t_end = time.time()
            if self._on_done:
                try:
                    self._on_done()
                except Exception as exc:
                    self._log(f"ShitScope on_done failed: {exc}")

    # ------------------------------------------------------------ results
    def analyse(self):
        xy = np.array([(r["x"], r["y"]) for r in self.records])
        res = tile_registration.analyse_tiles(self._small, xy, self.um_per_px * self.analysis_ds)
        images = tile_registration.render(res, self.um_per_px * self.analysis_ds)
        public = {k: v for k, v in res.items() if not k.startswith("_")}
        public.update(images, outDir=self.out_dir)
        self._save_json("scan_quality.json", {k: v for k, v in public.items() if k not in ("error_map", "mosaic")})
        if res["solved"]:
            stitch(self.out_dir, self.records, stitch_positions(public), res["_sign"],
                   self.um_per_px, ds=self.stitch_ds)
        with self._lock:
            self.result, self._registered = public, res

    def preview_png(self) -> Optional[str]:
        """Mosaic as a PNG data URL: at measured positions once registered,
        at commanded positions while scanning."""
        with self._lock:
            if not self._small:
                return None
            reg = getattr(self, "_registered", None)
            f = max(1, self._preview_ds // self.analysis_ds)
            if reg is not None and reg["solved"]:
                inc = reg["solved"]
                tiles = [self._small[k][::f, ::f] for k in inc]
                pos, sign = reg["_pos_px"][inc] / f, reg["_sign"]
            else:
                tiles = [t[::f, ::f] for t in self._small]
                px = self.um_per_px * self.analysis_ds * f
                pos = np.array([(r["y"] / px, r["x"] / px) for r in self.records])
                sign = np.array([1.0, 1.0])
        m = tile_registration.mosaic(tiles, pos, sign, scale=1)
        lo, hi = np.nanpercentile(m, [1, 99]) if np.isfinite(m).any() else (0, 1)
        img = np.clip((np.nan_to_num(m, nan=lo) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        return _png_gray(img)


def _png_gray(img: np.ndarray) -> str:
    import base64, io
    try:
        from PIL import Image
        buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
    except ImportError:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        buf = io.BytesIO(); plt.imsave(buf, img, cmap="gray", format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stitch_positions(result: Dict) -> np.ndarray:
    """(x, y) µm per tile: measured where registered; the others at their
    commanded position shifted by the median error of the registered ones."""
    cmd, meas = np.array(result["commanded_um"], float), np.array(result["measured_um"], float)
    solved = np.zeros(len(cmd), bool); solved[result["solved"]] = True
    if not solved.any():
        return cmd
    return np.where(solved[:, None], meas, cmd + np.median(meas[solved] - cmd[solved], axis=0))


def stitch(out_dir: str, records: List[Dict], xy_um, sign, um_per_px: float, ds: int = 1,
           name: str = "stitched.tif") -> Dict:
    """Paste the saved raw tiles (colour stays colour) at `xy_um` into one image,
    downsampled by `ds`; ds=1 is full resolution. `sign` (y, x) maps stage to
    image direction (from the registration)."""
    # ponytail: later tiles simply cover earlier ones in the overlap (no blending,
    # no blur from averaging); add feathering if seams become a problem.
    import tifffile
    px = np.asarray(xy_um, float)[:, ::-1] / (um_per_px * ds) * np.asarray(sign, float)   # (y, x)
    px = np.round(px - px.min(0)).astype(int)
    read = lambda r: np.asarray(tifffile.imread(os.path.join(out_dir, r["file"])))
    first = read(records[0])
    tile = lambda a: a if ds == 1 else downsample(a, ds).astype(first.dtype)
    H, W = tile(first).shape[:2]
    out = np.zeros((px[:, 0].max() + H, px[:, 1].max() + W) + first.shape[2:], first.dtype)
    for r, (y, x) in zip(records, px):
        out[y:y + H, x:x + W] = tile(read(r))
    path = os.path.join(out_dir, name)
    size_um = um_per_px * ds
    tifffile.imwrite(path, out, bigtiff=out.nbytes > 2 ** 31, photometric="rgb" if out.ndim == 3 else None,
                     resolution=(1e4 / size_um, 1e4 / size_um), resolutionunit="CENTIMETER",
                     metadata={"PhysicalSizeX": size_um, "PhysicalSizeY": size_um})
    return {"path": path, "shape": list(out.shape), "umPerPx": size_um, "sizeMB": round(out.nbytes / 1e6, 1)}


def export_full_res(out_dir: str) -> Dict:
    """Full-resolution stitch of a saved, analysed scan -> stitched_full.tif."""
    meta = json.load(open(os.path.join(out_dir, "scan.json")))
    records = json.load(open(os.path.join(out_dir, "tiles.json")))
    quality = json.load(open(os.path.join(out_dir, "scan_quality.json")))
    s = quality["summary"]
    return stitch(out_dir, records, stitch_positions(quality), (s["image_sign_y"], s["image_sign_x"]),
                  meta["um_per_px"], ds=1, name="stitched_full.tif")


def analyse_saved_scan(out_dir: str, analysis_downsample: Optional[int] = None) -> Dict:
    """Re-run registration on a saved ShitScope scan (scan.json + tiles.json)."""
    import tifffile
    meta = json.load(open(os.path.join(out_dir, "scan.json")))
    records = json.load(open(os.path.join(out_dir, "tiles.json")))
    scan = ShitScopeScan(None, None, out_dir, meta["um_per_px"], meta["plan"],
                         analysis_downsample=analysis_downsample)
    scan.records = records
    first = tifffile.imread(os.path.join(out_dir, records[0]["file"]))
    scan.analysis_ds = scan.analysis_ds or auto_downsample(to_gray(first).shape)
    scan._small = [downsample(to_gray(tifffile.imread(os.path.join(out_dir, r["file"]))), scan.analysis_ds)
                   for r in records]
    scan.analyse()
    return scan.result
