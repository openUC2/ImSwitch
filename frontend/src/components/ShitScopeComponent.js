import React, { useState, useEffect, useRef, useCallback } from "react";
import { useSelector } from "react-redux";
import {
  Button,
  Box,
  Typography,
  LinearProgress,
  Paper,
  Chip,
  Alert,
  CircularProgress,
  TextField,
  Divider,
  FormControlLabel,
  Checkbox,
  MenuItem,
} from "@mui/material";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Home as HomeIcon,
  FolderOpen as FolderOpenIcon,
  Refresh as RefreshIcon,
  SaveAlt as SaveAltIcon,
} from "@mui/icons-material";

import * as positionSlice from "../state/slices/PositionSlice.js";
import apiExperimentControllerHomeAllAxes from "../backendapi/apiExperimentControllerHomeAllAxes.js";
import {
  apiShitScopeGetInfo,
  apiShitScopeStartScan,
  apiShitScopeStopScan,
  apiShitScopeGetStatus,
  apiShitScopeGetPreview,
  apiShitScopeGetResult,
  apiShitScopeAnalyzeScan,
  apiShitScopeExportFullRes,
} from "../backendapi/apiShitScopeController.js";

import ShitScopeStageMap from "../axon/ShitScopeStageMap.js";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";
import InfoPopup from "../axon/InfoPopup.js";
import ShitScopeCalibrationPanel from "./ShitScopeCalibrationPanel.js";

// Preset area for "Fill area" (µm)
const PRESET_AREA_X = 15000;
const PRESET_AREA_Y = 7000;
const BUSY_STATES = ["homing", "running", "analysing"];

const fmt = (v, digits = 0) => (typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "–");
// Largest whole number of stage full steps that keeps `overlap` (mirrors shitscope_scan.suggested_step)
const suggestStep = (fov, full, overlap) => {
  const max = fov * (1 - overlap);
  return full > 0 ? Math.max(full, Math.floor(max / full) * full) : max;
};
const fmtTime = (s) => (typeof s === "number" && isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : "–");

/**
 * ShitScope - dedicated tile scanning app, driven by the ShitScopeController.
 *
 * The backend does everything (see ShitScopeController / model/shitscope_scan.py):
 * move with one approach direction, grab a frame exposed after the move, save
 * the tile, live mosaic; afterwards register the tiles against each other,
 * stitch at the measured positions and report the placement errors.
 */
const ShitScopeComponent = ({ onOpenFileManager }) => {
  const infoPopupRef = useRef(null);
  const pos = useSelector(positionSlice.getPositionState);

  const [info, setInfo] = useState(null);
  const [tilesX, setTilesX] = useState(5);
  const [tilesY, setTilesY] = useState(5);
  const [stepX, setStepX] = useState(0); // 0 = suggested by the backend
  const [stepY, setStepY] = useState(0);
  const [centered, setCentered] = useState(true);
  const [homeFirst, setHomeFirst] = useState(false);
  const [returnToStart, setReturnToStart] = useState(true);
  const [pattern, setPattern] = useState(""); // "" = stage recommendation
  const [overlapPct, setOverlapPct] = useState(0); // 0 = stage recommendation
  const [speed, setSpeed] = useState(0); // steps/s, 0 = stage default
  const [settleMs, setSettleMs] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const [scanPlan, setScanPlan] = useState(null); // tiles of the running/last scan (µm)
  const [status, setStatus] = useState({ state: "idle", tile: 0, total: 0 });
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isHoming, setIsHoming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const busy = BUSY_STATES.includes(status.state);
  const notify = (msg) => infoPopupRef.current && infoPopupRef.current.showMessage(msg);

  const loadInfo = useCallback(() => {
    apiShitScopeGetInfo()
      .then((i) => {
        setInfo(i);
        if (i.status) setStatus(i.status);
      })
      .catch(() => setInfo(null));
  }, []);
  useEffect(loadInfo, [loadInfo]);

  const effOverlap = overlapPct > 0 ? overlapPct / 100 : info?.minOverlap || 0.25;
  const effStepX = stepX > 0 ? stepX : info ? suggestStep(info.fovXUm, info.fullStepUm?.X, effOverlap) : 0;
  const effStepY = stepY > 0 ? stepY : info ? suggestStep(info.fovYUm, info.fullStepUm?.Y, effOverlap) : 0;
  const effPattern = pattern || info?.pattern || "raster";

  // ── Polling: status every second; preview every 2 s while busy ─────────
  const lastStateRef = useRef(status.state);
  useEffect(() => {
    let tick = 0;
    const id = setInterval(async () => {
      tick += 1;
      try {
        const s = await apiShitScopeGetStatus();
        setStatus(s);
        const wasBusy = BUSY_STATES.includes(lastStateRef.current);
        lastStateRef.current = s.state;
        if (BUSY_STATES.includes(s.state) && tick % 2 === 0) {
          const p = await apiShitScopeGetPreview();
          if (p.image) setPreview(p.image);
        }
        if (wasBusy && !BUSY_STATES.includes(s.state)) {
          // finished: final (registered) preview + result
          const p = await apiShitScopeGetPreview();
          if (p.image) setPreview(p.image);
          const r = await apiShitScopeGetResult();
          setResult(r.success ? r : null);
          notify(s.state === "error" ? `Scan failed: ${s.error}` : `Scan ${s.state}: ${s.tile}/${s.total} tiles`);
        }
      } catch (err) {
        /* backend unreachable: keep last state */
      }
    }, 1000);
    return () => clearInterval(id);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleStart = async () => {
    try {
      const r = await apiShitScopeStartScan({
        nx: tilesX, ny: tilesY, stepXUm: stepX, stepYUm: stepY, centered, homeFirst, returnToStart,
        pattern: effPattern, overlap: overlapPct / 100, speed, settleMs,
      });
      if (!r.success) throw new Error(r.error);
      setResult(null);
      setPreview(null);
      setScanPlan(r.plan || null);
      setStatus({ state: homeFirst ? "homing" : "running", tile: 0, total: r.tiles });
      lastStateRef.current = "running";
      notify(`Scan started: ${r.tiles} tiles, ${fmt(r.stepXUm)} × ${fmt(r.stepYUm)} µm steps (${r.pattern})`);
    } catch (err) {
      notify("Failed to start scan: " + (err.message || "unknown error"));
    }
  };

  const handleStop = () => apiShitScopeStopScan().catch(() => notify("Stop failed"));

  const handleReanalyze = async () => {
    setIsAnalyzing(true);
    try {
      const r = await apiShitScopeAnalyzeScan();
      if (!r.success) throw new Error(r.error);
      setResult(r);
      const p = await apiShitScopeGetPreview();
      if (p.image) setPreview(p.image);
    } catch (err) {
      notify("Analysis failed: " + (err.message || "unknown error"));
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const r = await apiShitScopeExportFullRes();
      if (!r.success) throw new Error(r.error);
      notify(`Full-resolution mosaic saved: ${r.shape.join(" × ")} px, ${r.sizeMB} MB → ${r.path}`);
    } catch (err) {
      notify("Export failed: " + (err.message || "unknown error"));
    } finally {
      setIsExporting(false);
    }
  };

  const fillPresetArea = () => {
    if (effStepX > 0 && info?.fovXUm) setTilesX(Math.max(1, Math.ceil((PRESET_AREA_X - info.fovXUm) / effStepX) + 1));
    if (effStepY > 0 && info?.fovYUm) setTilesY(Math.max(1, Math.ceil((PRESET_AREA_Y - info.fovYUm) / effStepY) + 1));
  };

  // Grid the next scan would take (mirrors shitscope_scan.plan_grid), drawn on the PCB map
  const plannedTiles = [];
  if (info && effStepX > 0 && effStepY > 0) {
    const x0 = (pos.x || 0) - (centered ? ((tilesX - 1) * effStepX) / 2 : 0);
    const y0 = (pos.y || 0) - (centered ? ((tilesY - 1) * effStepY) / 2 : 0);
    for (let iy = 0; iy < tilesY; iy++)
      for (let ix = 0; ix < tilesX; ix++) plannedTiles.push({ x: x0 + ix * effStepX, y: y0 + iy * effStepY });
  }

  const areaX = info ? (tilesX - 1) * effStepX + info.fovXUm : 0;
  const areaY = info ? (tilesY - 1) * effStepY + info.fovYUm : 0;
  const overlapX = info?.fovXUm ? 1 - effStepX / info.fovXUm : 0;
  const overlapY = info?.fovYUm ? 1 - effStepY / info.fovYUm : 0;
  const progress = status.total ? (100 * status.tile) / status.total : 0;
  const q = result?.summary || null;
  const gaps = q && ((q.min_overlap_x ?? 1) < 0 || (q.min_overlap_y ?? 1) < 0);

  return (
    <Box sx={{ width: "100%", p: 1 }}>
      <Typography variant="h5" sx={{ fontWeight: "bold", mb: 1 }}>
        ShitScope
      </Typography>

      {!info && (
        <Alert severity="warning" sx={{ mb: 1 }}>
          ShitScopeController not reachable. Add "ShitScope" to availableWidgets in the setup file.
        </Alert>
      )}
      {info && !info.stageAvailable && (
        <Alert severity="error" sx={{ mb: 1 }}>
          No stage available: check the stage's serial connection and restart ImSwitch.
        </Alert>
      )}
      {info && (
        <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
          <Chip size="small" variant="outlined" color="info"
                label={`FOV: ${fmt(info.fovXUm / 1000, 2)} × ${fmt(info.fovYUm / 1000, 2)} mm`} />
          <Chip size="small" variant="outlined" color="info" label={`Pixel: ${fmt(info.umPerPx, 3)} µm`} />
          {info.fullStepUm && (
            <Chip size="small" variant="outlined"
                  label={`Full step: ${fmt(info.fullStepUm.X)} / ${fmt(info.fullStepUm.Y)} µm`} />
          )}
          <Chip size="small" variant="outlined" color="secondary"
                label={`Scan area: ${fmt(areaX / 1000, 1)} × ${fmt(areaY / 1000, 1)} mm`} />
          <Chip size="small" color={overlapX < info.minOverlap - 1e-6 || overlapY < info.minOverlap - 1e-6 ? "warning" : "default"}
                label={`Overlap: ${fmt(overlapX * 100)} / ${fmt(overlapY * 100)} %`} />
        </Box>
      )}

      <Box sx={{ display: "flex", gap: 2 }}>
        {/* Left: overview (click to move) + live view */}
        <Box sx={{ flex: 3, display: "flex", gap: 1 }}>
          <Box sx={{ flex: 1, minHeight: 220 }}>
            <Typography variant="caption" color="text.secondary">
              PCB stage – click the slide to move · <b>right-click where the camera really is</b> to re-align the slide ("we are here")
            </Typography>
            <ShitScopeStageMap tiles={busy && scanPlan ? scanPlan : plannedTiles}
                               doneTiles={busy ? status.tile : 0} disabled={busy}
                               fovUm={{ x: info?.fovXUm || 0, y: info?.fovYUm || 0 }} />
          </Box>
          <Box sx={{ flex: 1, minHeight: 220 }}>
            <Typography variant="caption" color="text.secondary">Live View</Typography>
            <LiveViewControlWrapper />
          </Box>
        </Box>

        {/* Right: scan control */}
        <Box sx={{ flex: 1, minWidth: 260 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Scan</Typography>
            <Box sx={{ display: "flex", gap: 1, mb: 1.5 }}>
              <TextField label="Tiles X" type="number" size="small" value={tilesX} disabled={busy}
                         onChange={(e) => setTilesX(Math.max(1, parseInt(e.target.value) || 1))} />
              <TextField label="Tiles Y" type="number" size="small" value={tilesY} disabled={busy}
                         onChange={(e) => setTilesY(Math.max(1, parseInt(e.target.value) || 1))} />
            </Box>
            <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
              <TextField label="Step X (µm)" type="number" size="small" value={stepX || ""} disabled={busy}
                         placeholder={fmt(effStepX)} helperText="empty = suggested"
                         onChange={(e) => setStepX(parseFloat(e.target.value) || 0)} />
              <TextField label="Step Y (µm)" type="number" size="small" value={stepY || ""} disabled={busy}
                         placeholder={fmt(effStepY)} helperText="empty = suggested"
                         onChange={(e) => setStepY(parseFloat(e.target.value) || 0)} />
            </Box>
            <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
              <TextField select label="Pattern" size="small" value={effPattern} disabled={busy} sx={{ minWidth: 110 }}
                         helperText={info?.pattern === "raster" ? "raster: one approach dir." : " "}
                         onChange={(e) => setPattern(e.target.value)}>
                <MenuItem value="raster">Raster</MenuItem>
                <MenuItem value="snake">Snake</MenuItem>
              </TextField>
              <TextField label="Overlap (%)" type="number" size="small" value={overlapPct || ""} disabled={busy}
                         placeholder={fmt((info?.minOverlap || 0.25) * 100)} helperText="sets the steps"
                         inputProps={{ min: 0, max: 90 }}
                         onChange={(e) => setOverlapPct(Math.min(90, Math.max(0, parseFloat(e.target.value) || 0)))} />
            </Box>
            <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
              <TextField label="Speed (steps/s)" type="number" size="small" value={speed || ""} disabled={busy}
                         placeholder="default"
                         helperText={info?.maxSpeedSteps ? `stage cap ${info.maxSpeedSteps}` : "firmware max 2000"}
                         onChange={(e) => setSpeed(Math.max(0, parseFloat(e.target.value) || 0))} />
              <TextField label="Extra settle (ms)" type="number" size="small" value={settleMs || ""} disabled={busy}
                         placeholder="0" helperText={`+ stage ${fmt(info?.defaultSettleMs)} ms, then a fresh frame`}
                         onChange={(e) => setSettleMs(Math.max(0, parseFloat(e.target.value) || 0))} />
            </Box>
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
              <FormControlLabel
                control={<Checkbox size="small" checked={centered} disabled={busy} onChange={(e) => setCentered(e.target.checked)} />}
                label={<Typography variant="body2">Centred on position</Typography>} />
              <Button size="small" onClick={fillPresetArea} disabled={busy || !info}>
                Fill {PRESET_AREA_X / 1000}×{PRESET_AREA_Y / 1000} mm
              </Button>
            </Box>

            <FormControlLabel sx={{ mb: 1 }}
              control={<Checkbox size="small" checked={homeFirst} disabled={busy} onChange={(e) => setHomeFirst(e.target.checked)} />}
              label={<Typography variant="body2">Home before scan (−X/−Y stop, then scan in +X/+Y)</Typography>} />
            <FormControlLabel sx={{ mb: 1 }}
              control={<Checkbox size="small" checked={returnToStart} disabled={busy} onChange={(e) => setReturnToStart(e.target.checked)} />}
              label={<Typography variant="body2">Move back to the start position after the scan</Typography>} />

            <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
              <Button variant="contained" size="large" fullWidth startIcon={<PlayArrowIcon />}
                      onClick={handleStart} disabled={busy || !info?.stageAvailable}>
                Start Scan
              </Button>
              <Button variant="contained" color="error" size="large" startIcon={<StopIcon />}
                      onClick={handleStop} disabled={status.state !== "running"}>
                Stop
              </Button>
            </Box>

            <Alert severity={status.state === "error" ? "error" : busy ? "info" : "success"} sx={{ mb: 1 }}>
              {status.state === "error" ? status.error : busy ? `${status.state}…` : status.state === "idle" ? "Ready" : `Last scan: ${status.state}`}
            </Alert>
            <Box sx={{ display: "flex", justifyContent: "space-between" }}>
              <Typography variant="body2" color="text.secondary">
                Tile {status.tile} / {status.total || "–"}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {fmtTime(status.elapsed_s)}{status.eta_s ? ` · ETA ${fmtTime(status.eta_s)}` : ""}
              </Typography>
            </Box>
            <LinearProgress variant={status.state === "analysing" ? "indeterminate" : "determinate"}
                            value={progress} sx={{ height: 8, borderRadius: 1, mb: 2 }} />

            <Divider sx={{ mb: 2 }} />
            <Button variant="outlined" fullWidth sx={{ mb: 1 }} disabled={busy || isHoming}
                    startIcon={isHoming ? <CircularProgress size={16} color="inherit" /> : <HomeIcon />}
                    onClick={async () => {
                      setIsHoming(true);
                      try {
                        await apiExperimentControllerHomeAllAxes();
                        notify("Homing complete.");
                      } catch (err) {
                        notify("Homing failed.");
                      } finally {
                        setIsHoming(false);
                      }
                    }}>
              Home stage (−X/−Y stop)
            </Button>
            {onOpenFileManager && (
              <Button variant="outlined" fullWidth startIcon={<FolderOpenIcon />}
                      onClick={() => onOpenFileManager("/ShitScope")}>
                Open Scans Folder
              </Button>
            )}
          </Paper>
        </Box>
      </Box>

      {/* Mosaic + scan quality */}
      <Paper sx={{ p: 2, mt: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1 }}>
          <Typography variant="h6">Mosaic &amp; Scan Quality</Typography>
          <Button size="small" variant="outlined" onClick={handleReanalyze} disabled={busy || isAnalyzing || !status.outDir}
                  startIcon={isAnalyzing ? <CircularProgress size={14} color="inherit" /> : <RefreshIcon />}>
            Re-analyse last scan
          </Button>
          <Button size="small" variant="outlined" onClick={handleExport} disabled={busy || isExporting || !result}
                  startIcon={isExporting ? <CircularProgress size={14} color="inherit" /> : <SaveAltIcon />}>
            Export full resolution
          </Button>
          <Typography variant="caption" color="text.secondary">
            {busy ? "Tiles at commanded positions (live)" : result ? "Tiles at measured positions (registered) · scroll/pinch to zoom, double-click to reset" : ""}
          </Typography>
        </Box>
        {q && (
          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1 }}>
            <Chip size="small" label={`Registered: ${q.tiles_solved}/${q.tiles}`}
                  color={q.tiles_solved === q.tiles ? "success" : "warning"} />
            <Chip size="small" label={`Max error: ${fmt(q.max_err_um)} µm`} />
            <Chip size="small" label={`RMS error: ${fmt(q.rms_err_um)} µm`} />
            <Chip size="small" label={`Step X: ${fmt(q.measured_step_x_um)} ± ${fmt(q.measured_step_x_sd)} µm (cmd ${fmt(q.step_x_um)})`} />
            <Chip size="small" label={`Step Y: ${fmt(q.measured_step_y_um)} ± ${fmt(q.measured_step_y_sd)} µm (cmd ${fmt(q.step_y_um)})`} />
            <Chip size="small" color={gaps ? "error" : "default"}
                  label={`Min overlap: ${fmt((q.min_overlap_x ?? NaN) * 100)} / ${fmt((q.min_overlap_y ?? NaN) * 100)} %`} />
            <Chip size="small" label={`Holes: ${fmt(q.holes_pct, 1)} %`} />
            <Chip size="small" variant="outlined" label={`Registration: ${fmt(q.pair_rms_um, 1)} µm rms`} />
            <Chip size="small" variant="outlined" label={`Rotation: ${fmt(q.rotation_deg, 2)}°`} />
          </Box>
        )}
        <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", alignItems: "flex-start" }}>
          {preview && (
            <Box sx={{ maxWidth: 700, width: "100%", border: 1, borderColor: "divider", cursor: "grab" }}>
              <TransformWrapper minScale={1} maxScale={16} wheel={{ step: 0.15 }} doubleClick={{ mode: "reset" }}>
                <TransformComponent wrapperStyle={{ width: "100%" }} contentStyle={{ width: "100%" }}>
                  <img src={preview} alt="Mosaic" style={{ width: "100%", imageRendering: "pixelated" }} />
                </TransformComponent>
              </TransformWrapper>
            </Box>
          )}
          {result?.error_map && <img src={result.error_map} alt="Tile position error" style={{ maxWidth: 420, width: "100%" }} />}
          {!preview && <Typography variant="body2" color="text.secondary">No scan yet.</Typography>}
        </Box>
        {status.outDir && (
          <Typography variant="caption" color="text.secondary">
            {status.outDir} · tiles/, stitched.tif, scan_quality.json
          </Typography>
        )}
      </Paper>

      <ShitScopeCalibrationPanel disabled={busy} />

      <InfoPopup ref={infoPopupRef} />
    </Box>
  );
};

export default ShitScopeComponent;
