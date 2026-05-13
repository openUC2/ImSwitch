// src/components/FRAMESettings/StageOffsetCalibrationTab.js
import React, { useEffect, useRef, useState } from "react";
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
import apiStageCenterCalibrationPerformCalibration from "../../backendapi/apiStageCenterCalibrationPerformCalibration";
import apiStageCenterCalibrationStopCalibration from "../../backendapi/apiStageCenterCalibrationStopCalibration";
import apiStageCenterCalibrationGetHeatmap from "../../backendapi/apiStageCenterCalibrationGetHeatmap";
import apiStageCenterCalibrationGetIsRunning from "../../backendapi/apiStageCenterCalibrationGetIsRunning";

// Reference layouts shown to the user (purely informational - just hints on
// which calibration target to insert into slot 1).
const REFERENCE_LAYOUTS = [
  {
    id: "heidstar4",
    label: "Heidstar 4x Histosample (4 slides)",
    description:
      "Insert the openUC2 calibration slide into slot 1, then move roughly above the central hole.",
  },
  {
    id: "openuc2-96",
    label: "openUC2 Calibration Chart (96-well)",
    description:
      "Use the openUC2 96-well calibration chart and move roughly above the well-defined central hole.",
  },
];

const STEPS = ["Insert slide & start scan", "Review heatmap & accept"];

const POSITIONER_AXIS_API =
  "/PositionerController/getStageOffsetAxis";

/**
 * StageOffsetCalibrationTab - simplified 2-step wizard that replaces the
 * previous StageCenterStep1..6 flow.
 *
 * Step 1: Pick reference layout, configure scan, run XY raster (+/- 5 mm).
 * Step 2: Render heatmap, show brightest spot, allow manual override, accept
 *         and persist as stage offset (X / Y) via PositionerController.
 */
const StageOffsetCalibrationTab = () => {
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  const [activeStep, setActiveStep] = useState(0);
  const [layoutId, setLayoutId] = useState(REFERENCE_LAYOUTS[0].id);

  // Scan parameters
  const [startX, setStartX] = useState(0);
  const [startY, setStartY] = useState(0);
  const [stepUm, setStepUm] = useState(250);
  const [maxRadiusUm, setMaxRadiusUm] = useState(5000);
  const [exposureUs, setExposureUs] = useState(3000);
  const [speed, setSpeed] = useState(5000);

  // Run / result state
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [heatmap, setHeatmap] = useState(null); // { samples, brightest, meta }

  // Manual overrides for the accepted offset
  const [manualX, setManualX] = useState("");
  const [manualY, setManualY] = useState("");

  const canvasRef = useRef(null);
  const pollRef = useRef(null);

  const selectedLayout =
    REFERENCE_LAYOUTS.find((l) => l.id === layoutId) || REFERENCE_LAYOUTS[0];

  // ----- helpers ---------------------------------------------------------

  const fetchCurrentPosition = async () => {
    try {
      const x = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/PositionerController/getTruePositionerPositionWithoutOffset?axis=X`
      ).then((r) => r.json());
      const y = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/PositionerController/getTruePositionerPositionWithoutOffset?axis=Y`
      ).then((r) => r.json());
      if (typeof x === "number") setStartX(x);
      if (typeof y === "number") setStartY(y);
    } catch (e) {
      // Non-fatal: user can still type the values manually.
      console.warn("Could not fetch current stage position", e);
    }
  };

  useEffect(() => {
    if (hostIP && hostPort) fetchCurrentPosition();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hostIP, hostPort]);

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
        if (data && data.samples) {
          setHeatmap(data);
          drawHeatmap(data);
        }
        const running = await apiStageCenterCalibrationGetIsRunning();
        if (!running) {
          setIsRunning(false);
          stopPolling();
          setActiveStep(1);
        }
      } catch (e) {
        // Keep polling - one failed tick is not fatal.
        console.warn("Polling heatmap failed", e);
      }
    }, 1000);
  };

  // ----- actions ---------------------------------------------------------

  const handleStart = async () => {
    setError("");
    setInfo("");
    setHeatmap(null);
    setIsRunning(true);
    setActiveStep(0);
    startPolling();
    try {
      const result = await apiStageCenterCalibrationPerformCalibration({
        start_x: Number(startX) || 0,
        start_y: Number(startY) || 0,
        exposure_time_us: Number(exposureUs) || 3000,
        speed: Number(speed) || 5000,
        step_um: Number(stepUm) || 250,
        max_radius_um: Number(maxRadiusUm) || 5000,
      });
      // The backend call blocks until done; the polling loop will also pick
      // this up but we already have the final payload in hand.
      if (result && result.success === false) {
        setError(result.error || "Calibration failed");
      } else if (result && result.samples) {
        setHeatmap(result);
        drawHeatmap(result);
        setActiveStep(1);
      }
    } catch (e) {
      setError(`Calibration failed: ${e.message || e}`);
    } finally {
      setIsRunning(false);
      stopPolling();
    }
  };

  const handleStop = async () => {
    try {
      await apiStageCenterCalibrationStopCalibration();
    } catch (e) {
      console.warn("Stop calibration failed", e);
    }
    setIsRunning(false);
    stopPolling();
  };

  const acceptOffset = async () => {
    setError("");
    setInfo("");
    const xVal = manualX !== "" ? Number(manualX) : heatmap?.brightest?.x;
    const yVal = manualY !== "" ? Number(manualY) : heatmap?.brightest?.y;
    if (xVal == null || yVal == null || Number.isNaN(xVal) || Number.isNaN(yVal)) {
      setError("No valid X/Y value to accept.");
      return;
    }
    try {
      // Use "knownPosition" form so the backend computes the resulting offset
      // from the current stage position. This is the same path the existing
      // StageOffsetCalibration component uses and persists into the JSON
      // setup file via PositionerController.saveStageOffset().
      await fetch(
        `${hostIP}:${hostPort}/imswitch/api${POSITIONER_AXIS_API.replace(
          "getStageOffsetAxis",
          "setStageOffsetAxis"
        )}?knownPosition=${encodeURIComponent(xVal)}&axis=X`
      );
      await fetch(
        `${hostIP}:${hostPort}/imswitch/api${POSITIONER_AXIS_API.replace(
          "getStageOffsetAxis",
          "setStageOffsetAxis"
        )}?knownPosition=${encodeURIComponent(yVal)}&axis=Y`
      );
      setInfo(
        `Stage offset stored: X=${xVal.toFixed(1)} um, Y=${yVal.toFixed(1)} um`
      );
    } catch (e) {
      setError(`Failed to store offset: ${e.message || e}`);
    }
  };

  // ----- heatmap rendering ----------------------------------------------

  const drawHeatmap = (data) => {
    const canvas = canvasRef.current;
    if (!canvas || !data?.samples?.length) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

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

    // Estimate cell size from sample spacing (assume regular grid).
    const cellW = w / Math.max(1, Math.round(xRange / (data.meta?.step_um || 250)) + 1);
    const cellH = h / Math.max(1, Math.round(yRange / (data.meta?.step_um || 250)) + 1);

    data.samples.forEach((s) => {
      const px = ((s.x - xMin) / xRange) * (w - cellW);
      // Flip Y so larger Y is at the top of the canvas.
      const py = h - cellH - ((s.y - yMin) / yRange) * (h - cellH);
      const t = (s.intensity - iMin) / iRange;
      // Simple "viridis-like" gradient: dark blue -> green -> yellow.
      const r = Math.round(255 * Math.max(0, t - 0.5) * 2);
      const g = Math.round(255 * Math.min(1, t * 1.5));
      const b = Math.round(255 * Math.max(0, 1 - t * 1.5));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(px, py, cellW + 1, cellH + 1);
    });

    // Mark brightest spot.
    if (data.brightest) {
      const bx = ((data.brightest.x - xMin) / xRange) * (w - cellW);
      const by =
        h - cellH - ((data.brightest.y - yMin) / yRange) * (h - cellH);
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(bx + cellW / 2, by + cellH / 2, Math.max(cellW, cellH), 0, 2 * Math.PI);
      ctx.stroke();
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
        well-defined hole, then accept the brightest position as the stage
        offset (or override it manually).
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
        {/* Live view + reference layout selector */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Live view
          </Typography>
          <Box sx={{ mb: 2, maxHeight: 380, overflow: "hidden" }}>
            <LiveViewControlWrapper useFastMode={true} />
          </Box>

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="ref-layout-label">Reference Layout</InputLabel>
            <Select
              labelId="ref-layout-label"
              label="Reference Layout"
              value={layoutId}
              onChange={(e) => setLayoutId(e.target.value)}
            >
              {REFERENCE_LAYOUTS.map((l) => (
                <MenuItem key={l.id} value={l.id}>
                  {l.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Alert severity="info" sx={{ mb: 2 }}>
            {selectedLayout.description}
          </Alert>
        </Grid>

        {/* Scan parameters + actions */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Scan parameters
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                size="small"
                label="Start X (um)"
                type="number"
                value={startX}
                onChange={(e) => setStartX(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                size="small"
                label="Start Y (um)"
                type="number"
                value={startY}
                onChange={(e) => setStartY(e.target.value)}
              />
            </Grid>
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
            <Grid item xs={6}>
              <TextField
                fullWidth
                size="small"
                label="Exposure (us)"
                type="number"
                value={exposureUs}
                onChange={(e) => setExposureUs(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                size="small"
                label="Speed (um/s)"
                type="number"
                value={speed}
                onChange={(e) => setSpeed(e.target.value)}
              />
            </Grid>
          </Grid>

          <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
            <Button variant="outlined" onClick={fetchCurrentPosition}>
              Use current position
            </Button>
            {!isRunning ? (
              <Button variant="contained" color="primary" onClick={handleStart}>
                Start scan
              </Button>
            ) : (
              <Button variant="contained" color="warning" onClick={handleStop}>
                Stop
              </Button>
            )}
            {isRunning && <CircularProgress size={24} sx={{ alignSelf: "center" }} />}
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
              <Box
                sx={{
                  border: "1px solid",
                  borderColor: "divider",
                  borderRadius: 1,
                  p: 1,
                  background: "#111",
                }}
              >
                <canvas
                  ref={canvasRef}
                  width={420}
                  height={420}
                  style={{ width: "100%", height: "auto", display: "block" }}
                />
              </Box>
              <Typography variant="caption" color="textSecondary">
                {heatmap?.samples?.length
                  ? `${heatmap.samples.length} samples; brightness max highlighted in red.`
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
                    label="X (um)"
                    value={heatmap?.brightest?.x?.toFixed?.(1) ?? ""}
                    InputProps={{ readOnly: true }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Y (um)"
                    value={heatmap?.brightest?.y?.toFixed?.(1) ?? ""}
                    InputProps={{ readOnly: true }}
                  />
                </Grid>
              </Grid>
              <Typography variant="subtitle2" gutterBottom>
                Manual override (optional)
              </Typography>
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Override X (um)"
                    type="number"
                    value={manualX}
                    onChange={(e) => setManualX(e.target.value)}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Override Y (um)"
                    type="number"
                    value={manualY}
                    onChange={(e) => setManualY(e.target.value)}
                  />
                </Grid>
              </Grid>
              <Button
                variant="contained"
                color="success"
                fullWidth
                disabled={!heatmap?.brightest && manualX === "" && manualY === ""}
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
