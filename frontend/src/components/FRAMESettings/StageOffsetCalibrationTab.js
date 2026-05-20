// src/components/FRAMESettings/StageOffsetCalibrationTab.js
//
// Two-step stage-offset calibration replacing the legacy StageCenterStep1..6
// wizard.
//
// Step 1: pick the inserted reference layout. A miniature layout map shows
//         the slot rectangles + current stage position; clicking the map
//         moves the stage to that XY in absolute coordinates (similar to the
//         wellplate component). Scan parameters (step, radius) are
//         pre-filled adaptively from the active detector's pixel size and
//         frame shape - the user does not deal with exposure here.
//
// Step 2: a heatmap canvas renders the (x, y, intensity) raster samples.
//         Clicking the heatmap moves the stage to that location (similar to
//         the focus map). The detected brightest point is shown next to the
//         expected (known) calibration point - the difference of the two is
//         what gets persisted as the stage offset.
//
import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stepper,
  Step,
  StepLabel,
  TextField,
  Typography,
} from "@mui/material";

import LiveViewControlWrapper from "../../axon/LiveViewControlWrapper";
import * as connectionSettingsSlice from "../../state/slices/ConnectionSettingsSlice";
import * as positionSlice from "../../state/slices/PositionSlice";
import apiStageCenterCalibrationPerformCalibration from "../../backendapi/apiStageCenterCalibrationPerformCalibration";
import apiStageCenterCalibrationStopCalibration from "../../backendapi/apiStageCenterCalibrationStopCalibration";
import apiStageCenterCalibrationGetHeatmap from "../../backendapi/apiStageCenterCalibrationGetHeatmap";
import apiStageCenterCalibrationGetIsRunning from "../../backendapi/apiStageCenterCalibrationGetIsRunning";
import apiStageCenterCalibrationGetRecommendedScanParameters from "../../backendapi/apiStageCenterCalibrationGetRecommendedScanParameters";
import apiExperimentControllerGetKnownCalibrationLayouts from "../../backendapi/apiExperimentControllerGetKnownCalibrationLayouts";
import apiPositionerControllerMovePositioner from "../../backendapi/apiPositionerControllerMovePositioner";

const STEPS = ["Insert slide & start scan", "Review heatmap & accept"];

// Built-in fallback layouts in case the backend has no entry yet (e.g. first
// boot before the user has registered the openUC2 chart). The same dict
// shape as ``ExperimentController._KNOWN_CALIBRATION_POINTS``.
const FALLBACK_LAYOUTS = [
  {
    name: "Heidstar 4x Histosample",
    x: 20000,
    y: 40000,
    description: "Heidstar 4x slide carrier - centre of the calibration pinhole (slot 1).",
    bounds: { width: 127000, height: 84000 },
    slots: [
      { x: 18400, y: 40600, w: 27000, h: 74000, name: "Slot 1" },
      { x: 48400, y: 40600, w: 27000, h: 74000, name: "Slot 2" },
      { x: 78400, y: 40600, w: 27000, h: 74000, name: "Slot 3" },
      { x: 108400, y: 40600, w: 27000, h: 74000, name: "Slot 4" },
    ],
  },
  {
    name: "openUC2 96-Well Calibration Chart",
    x: 14380,
    y: 11240,
    description: "openUC2 96-well calibration chart - well A1 (slot 1).",
    bounds: { width: 127000, height: 86000 },
    slots: [{ x: 14380, y: 11240, w: 9000, h: 9000, name: "A1" }],
  },
];

// ----------------------------------------------------------------------
// LayoutMapMini - small canvas rendering layout slots + the current stage
// position. Click moves the stage to that XY (absolute, blocking).
// ----------------------------------------------------------------------
const LayoutMapMini = ({
  layout,
  stageX,
  stageY,
  knownX,
  knownY,
  brightestX,
  brightestY,
  width = 380,
  height = 240,
  onClickMove,
}) => {
  const canvasRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#101820";
    ctx.fillRect(0, 0, w, h);

    const lw = layout.bounds?.width || 127000;
    const lh = layout.bounds?.height || 84000;
    const sx = w / lw;
    const sy = h / lh;
    const s = Math.min(sx, sy);
    const offX = (w - lw * s) / 2;
    const offY = (h - lh * s) / 2;

    // Outer carrier rectangle.
    ctx.strokeStyle = "#666";
    ctx.lineWidth = 1;
    ctx.strokeRect(offX, offY, lw * s, lh * s);

    // Slot rectangles.
    ctx.fillStyle = "rgba(80, 140, 200, 0.25)";
    ctx.strokeStyle = "#5aa";
    (layout.slots || []).forEach((slot, idx) => {
      const px = offX + (slot.x - slot.w / 2) * s;
      const py = offY + (slot.y - slot.h / 2) * s;
      ctx.fillRect(px, py, slot.w * s, slot.h * s);
      ctx.strokeRect(px, py, slot.w * s, slot.h * s);
      ctx.fillStyle = "#bcd";
      ctx.font = "10px sans-serif";
      ctx.fillText(slot.name || `Slot ${idx + 1}`, px + 3, py + 12);
      ctx.fillStyle = "rgba(80, 140, 200, 0.25)";
    });

    // Known calibration point (expected centre).
    if (knownX != null && knownY != null) {
      const kx = offX + knownX * s;
      const ky = offY + knownY * s;
      ctx.strokeStyle = "#ffcc33";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(kx, ky, 6, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(kx - 8, ky);
      ctx.lineTo(kx + 8, ky);
      ctx.moveTo(kx, ky - 8);
      ctx.lineTo(kx, ky + 8);
      ctx.stroke();
    }

    // Brightest point (detected).
    if (brightestX != null && brightestY != null) {
      const bx = offX + brightestX * s;
      const by = offY + brightestY * s;
      ctx.fillStyle = "#ff5577";
      ctx.beginPath();
      ctx.arc(bx, by, 4, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Current stage position (red dot - "we are here").
    if (stageX != null && stageY != null) {
      const px = offX + stageX * s;
      const py = offY + stageY * s;
      ctx.fillStyle = "#ff3333";
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [layout, stageX, stageY, knownX, knownY, brightestX, brightestY]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleClick = (event) => {
    if (!layout || !onClickMove) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const cy = ((event.clientY - rect.top) / rect.height) * canvas.height;

    const lw = layout.bounds?.width || 127000;
    const lh = layout.bounds?.height || 84000;
    const w = canvas.width;
    const h = canvas.height;
    const s = Math.min(w / lw, h / lh);
    const offX = (w - lw * s) / 2;
    const offY = (h - lh * s) / 2;
    const stageXClick = (cx - offX) / s;
    const stageYClick = (cy - offY) / s;
    if (
      stageXClick < 0 ||
      stageXClick > lw ||
      stageYClick < 0 ||
      stageYClick > lh
    ) {
      return;
    }
    onClickMove(stageXClick, stageYClick);
  };

  return (
    <Box
      sx={{
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 1,
        background: "#000",
      }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: "100%", height: "auto", display: "block", cursor: "crosshair" }}
        onClick={handleClick}
      />
    </Box>
  );
};

// ----------------------------------------------------------------------
// HeatmapCanvas - intensity grid + click-to-move stage.
// ----------------------------------------------------------------------
const HeatmapCanvas = ({ data, onClickMove, width = 420, height = 420 }) => {
  const canvasRef = useRef(null);
  // Cache the projection so the click handler can invert it.
  const projectionRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    if (!data?.samples?.length) {
      projectionRef.current = null;
      return;
    }

    const xs = data.samples.map((s) => s.x);
    const ys = data.samples.map((s) => s.y);
    const is = data.samples.map((s) => s.intensity);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const iMin = Math.min(...is);
    const iMax = Math.max(...is);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const iRange = iMax - iMin || 1;

    // Estimate grid spacing in stage units, then in pixels.
    const stepUm = data.meta?.step_um || 250;
    const cellW = Math.max(1, Math.round(w / Math.max(1, Math.round(xRange / stepUm) + 1)));
    const cellH = Math.max(1, Math.round(h / Math.max(1, Math.round(yRange / stepUm) + 1)));

    data.samples.forEach((s) => {
      const px = ((s.x - xMin) / xRange) * (w - cellW);
      const py = h - cellH - ((s.y - yMin) / yRange) * (h - cellH);
      const t = (s.intensity - iMin) / iRange;
      const r = Math.round(255 * Math.max(0, t - 0.5) * 2);
      const g = Math.round(255 * Math.min(1, t * 1.5));
      const b = Math.round(255 * Math.max(0, 1 - t * 1.5));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(px, py, cellW + 1, cellH + 1);
    });

    if (data.brightest) {
      const bx = ((data.brightest.x - xMin) / xRange) * (w - cellW);
      const by = h - cellH - ((data.brightest.y - yMin) / yRange) * (h - cellH);
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(bx + cellW / 2, by + cellH / 2, Math.max(cellW, cellH), 0, 2 * Math.PI);
      ctx.stroke();
    }

    projectionRef.current = { xMin, xMax, yMin, yMax, w, h, cellW, cellH };
  }, [data]);

  const handleClick = (event) => {
    const proj = projectionRef.current;
    if (!proj || !onClickMove) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cx = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const cy = ((event.clientY - rect.top) / rect.height) * canvas.height;
    const { xMin, xMax, yMin, yMax, w, h, cellW, cellH } = proj;
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const stageX = xMin + (cx / (w - cellW)) * xRange;
    const stageY = yMin + ((h - cellH - cy) / (h - cellH)) * yRange;
    onClickMove(stageX, stageY);
  };

  return (
    <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1, background: "#111" }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: "100%", height: "auto", display: "block", cursor: "crosshair" }}
        onClick={handleClick}
      />
    </Box>
  );
};

// ----------------------------------------------------------------------
// Main tab
// ----------------------------------------------------------------------
const StageOffsetCalibrationTab = () => {
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const positionState = useSelector(positionSlice.getPositionState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  const [activeStep, setActiveStep] = useState(0);
  const [layouts, setLayouts] = useState(FALLBACK_LAYOUTS);
  const [layoutName, setLayoutName] = useState(FALLBACK_LAYOUTS[0].name);

  // Scan parameters - exposure intentionally removed (managed in detector).
  const [stepUm, setStepUm] = useState(250);
  const [maxRadiusUm, setMaxRadiusUm] = useState(5000);
  const [recommended, setRecommended] = useState(null);

  // Derived "running" state - driven by polling, NOT by the local action.
  // performCalibration returns immediately because the controller spawns its
  // own raster thread, so toggling local state here would never flip back to
  // "Stop" properly.
  const [pollingRunning, setPollingRunning] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [heatmap, setHeatmap] = useState(null);

  // Manual override applied to the *known* point (gold cross).
  const [overrideKnownX, setOverrideKnownX] = useState("");
  const [overrideKnownY, setOverrideKnownY] = useState("");

  const pollRef = useRef(null);

  const selectedLayout = useMemo(
    () => layouts.find((l) => l.name === layoutName) || layouts[0],
    [layouts, layoutName]
  );

  const knownX =
    overrideKnownX !== ""
      ? Number(overrideKnownX)
      : selectedLayout?.x;
  const knownY =
    overrideKnownY !== ""
      ? Number(overrideKnownY)
      : selectedLayout?.y;

  // ----- bootstrap -------------------------------------------------------

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiExperimentControllerGetKnownCalibrationLayouts();
        if (cancelled) return;
        if (res?.layouts?.length) {
          // Merge backend-known layouts with our fallbacks (fallback wins on
          // missing 'slots'/'bounds' since the backend only ships x/y).
          const merged = res.layouts.map((b) => {
            const fb = FALLBACK_LAYOUTS.find((f) => f.name === b.name);
            return { ...(fb || {}), ...b };
          });
          // Also append any fallback layouts the backend does not know about.
          FALLBACK_LAYOUTS.forEach((f) => {
            if (!merged.find((m) => m.name === f.name)) merged.push(f);
          });
          setLayouts(merged);
        }
      } catch (e) {
        // Use fallback only.
      }
      try {
        const rec = await apiStageCenterCalibrationGetRecommendedScanParameters();
        if (cancelled) return;
        if (rec?.success) {
          setRecommended(rec);
          setStepUm(Math.round(rec.recommendedStepUm));
          setMaxRadiusUm(Math.round(rec.recommendedMaxRadiusUm));
        }
      } catch (e) {
        // Keep static defaults.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // ----- polling ---------------------------------------------------------

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => () => stopPolling(), []);

  const startPolling = () => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const data = await apiStageCenterCalibrationGetHeatmap();
        if (data && data.samples) setHeatmap(data);
        const running = await apiStageCenterCalibrationGetIsRunning();
        const isRun = !!running;
        setPollingRunning(isRun);
        if (!isRun) {
          stopPolling();
          // Auto-advance to step 2 once the scan is done.
          setActiveStep(1);
        }
      } catch (e) {
        // Keep polling - one tick failing is non-fatal.
      }
    }, 1000);
  };

  // ----- actions ---------------------------------------------------------

  const handleStart = async () => {
    setError("");
    setInfo("");
    setHeatmap(null);
    setActiveStep(0);
    setPollingRunning(true);
    startPolling();
    try {
      const result = await apiStageCenterCalibrationPerformCalibration({
        // Start at the *current* stage position; user is supposed to move
        // roughly above the calibration hole using the layout map first.
        start_x: positionState?.x ?? 0,
        start_y: positionState?.y ?? 0,
        step_um: Number(stepUm) || 250,
        max_radius_um: Number(maxRadiusUm) || 5000,
        speed: 5000,
      });
      if (result && result.success === false) {
        setError(result.error || "Calibration failed");
        setPollingRunning(false);
        stopPolling();
      }
    } catch (e) {
      setError(`Calibration failed: ${e.message || e}`);
      setPollingRunning(false);
      stopPolling();
    }
  };

  const handleStop = async () => {
    try {
      await apiStageCenterCalibrationStopCalibration();
    } catch (e) {
      // ignore
    }
    setPollingRunning(false);
    stopPolling();
  };

  const moveStage = async (x, y) => {
    try {
      await apiPositionerControllerMovePositioner({
        axis: "X",
        dist: x,
        isAbsolute: true,
        isBlocking: false,
      });
      await apiPositionerControllerMovePositioner({
        axis: "Y",
        dist: y,
        isAbsolute: true,
        isBlocking: false,
      });
    } catch (e) {
      setError(`Stage move failed: ${e.message || e}`);
    }
  };

  const acceptOffset = async () => {
    setError("");
    setInfo("");
    const actualX = heatmap?.brightest?.x;
    const actualY = heatmap?.brightest?.y;
    if (
      actualX == null ||
      actualY == null ||
      knownX == null ||
      knownY == null
    ) {
      setError("Need both a known and a measured X/Y position before saving.");
      return;
    }
    // setStageOffsetAxis(knownPosition=known, currentPosition=actual) makes the
    // stage report the known coordinate when it is mechanically at "actual",
    // i.e. it bakes the (known - actual) deviation into the stored offset.
    try {
      const base = `${hostIP}:${hostPort}/imswitch/api/PositionerController/setStageOffsetAxis`;
      await fetch(
        `${base}?knownPosition=${encodeURIComponent(knownX)}` +
          `&currentPosition=${encodeURIComponent(actualX)}&axis=X`
      );
      await fetch(
        `${base}?knownPosition=${encodeURIComponent(knownY)}` +
          `&currentPosition=${encodeURIComponent(actualY)}&axis=Y`
      );
      const dx = knownX - actualX;
      const dy = knownY - actualY;
      setInfo(
        `Stage offset stored. Computed deltas: dX=${dx.toFixed(1)} um, ` +
          `dY=${dy.toFixed(1)} um.`
      );
    } catch (e) {
      setError(`Failed to store offset: ${e.message || e}`);
    }
  };

  // ----- render ----------------------------------------------------------

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Stage Offset Calibration
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Two-step calibration: scan a small XY area for the reference slide&apos;s
        well-defined hole, then store the difference between where the stage
        thinks it is and where the pinhole really sits as the stage offset.
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {info && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {info}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Live view + reference layout selector + layout map */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Live view
          </Typography>
          <Box sx={{ mb: 2, maxHeight: 320, overflow: "hidden" }}>
            <LiveViewControlWrapper useFastMode={true} />
          </Box>

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="ref-layout-label">Reference Layout</InputLabel>
            <Select
              labelId="ref-layout-label"
              label="Reference Layout"
              value={layoutName}
              onChange={(e) => {
                setLayoutName(e.target.value);
                setOverrideKnownX("");
                setOverrideKnownY("");
              }}
            >
              {layouts.map((l) => (
                <MenuItem key={l.name} value={l.name}>
                  {l.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {selectedLayout?.description && (
            <Alert severity="info" sx={{ mb: 2 }}>
              {selectedLayout.description}
            </Alert>
          )}

          <Typography variant="subtitle2" gutterBottom>
            Layout map (click to move)
          </Typography>
          <LayoutMapMini
            layout={selectedLayout}
            stageX={positionState?.x}
            stageY={positionState?.y}
            knownX={knownX}
            knownY={knownY}
            brightestX={heatmap?.brightest?.x}
            brightestY={heatmap?.brightest?.y}
            onClickMove={moveStage}
          />
          <Typography variant="caption" color="textSecondary">
            Red = current stage position, gold = expected pinhole, pink =
            detected brightest spot.
          </Typography>
        </Grid>

        {/* Scan parameters + actions */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Scan parameters (adaptive to detector FOV)
          </Typography>
          {recommended && (
            <Typography
              variant="caption"
              color="textSecondary"
              sx={{ display: "block", mb: 1 }}
            >
              Detector: {recommended.frameWidth}×{recommended.frameHeight} px;
              Pixel size {recommended.pixelSizeUm.toFixed(3)} um/px;
              FOV {Math.round(recommended.fovXUm)} × {Math.round(recommended.fovYUm)} um
              (= {recommended.frameWidth}px × {recommended.pixelSizeUm.toFixed(3)} um/px);
              recommended step {Math.round(recommended.recommendedStepUm)} um.
            </Typography>
          )}
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                size="small"
                label="Step (um)"
                type="number"
                value={stepUm}
                onChange={(e) => setStepUm(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                size="small"
                label="Max radius (um)"
                type="number"
                value={maxRadiusUm}
                onChange={(e) => setMaxRadiusUm(e.target.value)}
              />
            </Grid>
          </Grid>

          <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
            {!pollingRunning ? (
              <Button variant="contained" color="primary" onClick={handleStart}>
                Start scan
              </Button>
            ) : (
              <Button variant="contained" color="warning" onClick={handleStop}>
                Stop
              </Button>
            )}
            {pollingRunning && (
              <CircularProgress size={24} sx={{ alignSelf: "center" }} />
            )}
            <Box sx={{ flexGrow: 1 }} />
            <Typography variant="caption" color="textSecondary" sx={{ alignSelf: "center" }}>
              Stage at: X={positionState?.x?.toFixed?.(1) ?? "?"} um,
              Y={positionState?.y?.toFixed?.(1) ?? "?"} um
            </Typography>
          </Box>
          <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: "block" }}>
            Requires a valid pixel calibration. The backend will refuse to
            start otherwise.
          </Typography>
        </Grid>

        {/* Heatmap + accept controls */}
        <Grid item xs={12}>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1" gutterBottom>
            Heatmap & result
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <HeatmapCanvas data={heatmap} onClickMove={moveStage} />
              <Typography variant="caption" color="textSecondary">
                {heatmap?.samples?.length
                  ? `${heatmap.samples.length} samples; click anywhere to move stage there.`
                  : "No data yet - run a scan to populate."}
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Detected brightest position
              </Typography>
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Actual X (um)"
                    value={heatmap?.brightest?.x?.toFixed?.(1) ?? ""}
                    InputProps={{ readOnly: true }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Actual Y (um)"
                    value={heatmap?.brightest?.y?.toFixed?.(1) ?? ""}
                    InputProps={{ readOnly: true }}
                  />
                </Grid>
              </Grid>
              <Typography variant="subtitle2" gutterBottom>
                Expected (known) calibration point
              </Typography>
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Known X (um)"
                    type="number"
                    value={overrideKnownX !== "" ? overrideKnownX : selectedLayout?.x ?? ""}
                    onChange={(e) => setOverrideKnownX(e.target.value)}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Known Y (um)"
                    type="number"
                    value={overrideKnownY !== "" ? overrideKnownY : selectedLayout?.y ?? ""}
                    onChange={(e) => setOverrideKnownY(e.target.value)}
                  />
                </Grid>
              </Grid>
              {heatmap?.brightest && knownX != null && knownY != null && (
                <Typography variant="caption" color="textSecondary" sx={{ display: "block", mb: 2 }}>
                  Pending offset: dX={(knownX - heatmap.brightest.x).toFixed(1)} um,
                  dY={(knownY - heatmap.brightest.y).toFixed(1)} um.
                </Typography>
              )}
              <Button
                variant="contained"
                color="success"
                fullWidth
                disabled={!heatmap?.brightest}
                onClick={acceptOffset}
              >
                Accept and store stage offset
              </Button>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default StageOffsetCalibrationTab;
