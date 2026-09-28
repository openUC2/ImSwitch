"""ShitScope: dedicated tile scan with live preview, registration and stitching.

Loaded when "ShitScope" is in the setup's availableWidgets. The scan itself
(move -> fresh frame -> save -> preview, then register + stitch) lives in
imswitch.imcontrol.model.shitscope_scan and is tested standalone; this
controller only wires it to the detector, the stage and the REST API.

Results of each scan: <data>/ShitScope/<timestamp>/{tiles/, scan.json,
tiles.json, scan_quality.json, stitched.tif}.
"""
import os
import time
from typing import Dict, Optional

from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcontrol.model import shitscope_scan
from ..basecontrollers import ImConWidgetController


class ShitScopeController(ImConWidgetController):
    """Tile scans for the ShitScope (PCB planar stage)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._scan: Optional[shitscope_scan.ShitScopeScan] = None
        self._last_dir: str = ""

    # ------------------------------------------------------------ hardware
    def _detector(self):
        name = self._master.detectorsManager.getCurrentDetectorName()
        return self._master.detectorsManager[name]

    def _stage(self):
        """First positioner, or None when none initialised (e.g. serial port missing)."""
        names = self._master.positionersManager.getAllDeviceNames()
        return self._master.positionersManager[names[0]] if names else None

    def _um_per_px(self) -> float:
        try:
            v = float(self._detector().pixelSizeUm[-1])
            return v if v > 0 else 1.0
        except Exception:
            return 1.0

    def _hints(self) -> Dict:
        stage = self._stage()
        if stage is None:
            return {}
        try:
            return stage.getScanHints() if hasattr(stage, "getScanHints") else {}
        except Exception as exc:
            self._logger.warning(f"Stage scan hints unavailable: {exc}")
            return {}

    # ------------------------------------------------------------ API
    @APIExport()
    def getShitScopeInfo(self) -> Dict:
        """Pixel size, field of view, stage hints and suggested tile steps."""
        det, px = self._detector(), self._um_per_px()
        w, h = (list(det.shape) + [0, 0])[:2]
        hints = self._hints()
        min_overlap = float(hints.get("recommendedMinOverlap", 0.25))
        full = hints.get("fullStepUm") or {}
        fov_x, fov_y = w * px, h * px
        return {
            "umPerPx": px, "fovXUm": fov_x, "fovYUm": fov_y,
            "minOverlap": min_overlap,
            "pattern": hints.get("recommendedPattern", "snake"),
            "fullStepUm": full or None,
            "suggestedStepXUm": shitscope_scan.suggested_step(fov_x, full.get("X"), min_overlap),
            "suggestedStepYUm": shitscope_scan.suggested_step(fov_y, full.get("Y"), min_overlap),
            "stageHints": hints,
            "stageAvailable": self._stage() is not None,
            "status": self.getShitScopeStatus(),
        }

    @APIExport()
    def startShitScopeScan(self, nx: int = 3, ny: int = 3, stepXUm: float = 0.0, stepYUm: float = 0.0,
                           pattern: str = "", centered: bool = True, returnToStart: bool = True,
                           homeFirst: bool = False) -> Dict:
        """Start a tile scan around (centered=True) or from the current position.
        homeFirst: drive into the -X/-Y stop first (position -> 0), then scan the
        same grid; the grid only makes sense if the stage was homed before.
        Steps of 0 use the suggested steps (whole stage full steps, min overlap);
        an empty pattern uses the stage's recommendation (raster for the PCB stage)."""
        if self._scan is not None and self._scan.state in ("homing", "running", "analysing"):
            return {"success": False, "error": "A scan is already running"}
        if self._stage() is None:
            return {"success": False, "error": "No stage available (check the stage's serial connection)"}
        info = self.getShitScopeInfo()
        sx = float(stepXUm) if stepXUm > 0 else info["suggestedStepXUm"]
        sy = float(stepYUm) if stepYUm > 0 else info["suggestedStepYUm"]
        if sx <= 0 or sy <= 0:
            return {"success": False, "error": "No step size (field of view unknown); pass stepXUm/stepYUm"}
        nx, ny = max(1, int(nx)), max(1, int(ny))
        pos = self._stage().getPosition()
        plan = shitscope_scan.plan_grid(pos["X"], pos["Y"], nx, ny, sx, sy,
                                        pattern or info["pattern"], centered=bool(centered))
        out_dir = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), "ShitScope",
                               time.strftime("%Y%m%d_%H%M%S"))
        det_mgr = self._master.detectorsManager
        handle = det_mgr.startAcquisition()
        self._scan = shitscope_scan.ShitScopeScan(
            self._detector(), self._stage(), out_dir, info["umPerPx"], plan,
            return_to_start=bool(returnToStart), logger=self._logger.error,
            on_done=lambda: det_mgr.stopAcquisition(handle),
            home=self._stage().home if homeFirst and hasattr(self._stage(), "home") else None)
        self._last_dir = out_dir
        self._scan.start()
        self._logger.info(f"ShitScope scan {nx}x{ny}, step {sx:.0f}x{sy:.0f} µm -> {out_dir}")
        return {"success": True, "outDir": out_dir, "tiles": len(plan), "stepXUm": sx, "stepYUm": sy,
                "plan": plan, "homeFirst": bool(homeFirst),
                "pattern": pattern or info["pattern"]}

    @APIExport()
    def stopShitScopeScan(self) -> Dict:
        """Stop after the current tile; the tiles taken so far are kept and registered."""
        if self._scan is None:
            return {"success": False, "error": "No scan"}
        self._scan.stop(join_s=0)
        return {"success": True}

    @APIExport()
    def getShitScopeStatus(self) -> Dict:
        """state (idle/homing/running/analysing/done/stopped/error), tile k of n, ETA, summary when done."""
        if self._scan is None:
            return {"state": "idle", "tile": 0, "total": 0, "outDir": self._last_dir}
        return self._scan.status()

    @APIExport()
    def getShitScopePreview(self) -> Dict:
        """Live mosaic (PNG data URL): commanded positions while scanning,
        measured positions after registration."""
        if self._scan is None:
            return {"image": None}
        return {"image": self._scan.preview_png(), "registered": self._scan.result is not None}

    @APIExport()
    def getShitScopeResult(self) -> Dict:
        """Registration result of the last scan: summary, per-tile commanded and
        measured positions (µm), error map and mosaic (PNG data URLs)."""
        if self._scan is None or self._scan.result is None:
            return {"success": False, "error": "No registered scan yet"}
        return {"success": True, **self._scan.result}

    @APIExport()
    def analyzeShitScopeScan(self, scanDir: str = "") -> Dict:
        """Re-run registration and stitching on a saved scan (default: the last one)."""
        target = (scanDir or "").strip() or self._last_dir
        if not target or not os.path.isdir(target):
            return {"success": False, "error": f"Scan directory not found: '{target}'"}
        try:
            return {"success": True, **shitscope_scan.analyse_saved_scan(target)}
        except Exception as exc:
            self._logger.warning(f"ShitScope re-analysis failed: {exc}")
            return {"success": False, "error": str(exc)}
