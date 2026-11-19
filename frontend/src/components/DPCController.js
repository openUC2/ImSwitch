// src/components/DPCController.js
// DPC / FPM with any LED source (geodesic dome, ring board, LED matrix): live camera,
// reconstructed phase stream, LED map in illumination-NA space with LED -> camera
// calibration, and the FPM per-LED scan.

import React, { useState, useEffect, useCallback, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Paper,
  Typography,
  Slider,
  Button,
  Grid,
  TextField,
  Switch,
  FormControlLabel,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
  MenuItem,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import SaveIcon from "@mui/icons-material/Save";
import CameraAltIcon from "@mui/icons-material/CameraAlt";

// Redux slice
import * as dpcSlice from "../state/slices/dpcSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice";
import * as liveViewSlice from "../state/slices/LiveViewSlice";

// Shared camera live view (same stream as the Live View tab)
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import apiLiveViewControllerStartLiveView from "../backendapi/apiLiveViewControllerStartLiveView";
import apiViewControllerGetLiveViewActive from "../backendapi/apiViewControllerGetLiveViewActive";

// API imports
import {
  getDpcParams,
  setDpcParams,
  getDpcState,
  startDpcProcessing,
  stopDpcProcessing,
  getDpcStreamUrl,
  getDpcWotfUrl,
  getLedLayout,
  setLedCalibration,
  startLedCalibration,
  startFpmAcquisition,
  snapDpc,
  stopDpcTask,
  setLeds,
} from "../backendapi/dpcApi";

// DPC half patterns: LEDs whose illumination NA points along +x / -x / +y / -y of the image
const PATTERN_COLORS = { "+x": "#e53935", "-x": "#1e88e5", "+y": "#43a047", "-y": "#fb8c00" };
const FPM_COLOR = "#8e24aa";

// Streamed image of the last DPC result (backend param "display"); qDPC = absorption + i phase
const DISPLAY_VIEWS = [
  ["phase", "Phase (imaginary part)"],
  ["absorption", "Absorption (real part)"],
  ["dpc_x", "DPC left-right"],
  ["dpc_y", "DPC up-down"],
  ["+x", "Raw image +x"],
  ["-x", "Raw image -x"],
  ["+y", "Raw image +y"],
  ["-y", "Raw image -y"],
];

// LEDs in illumination-NA space, camera frame: x = image column (right), y = image row (down).
const LedMap = ({ leds, na, nai, naiInner, fpmNaMax, view, lit, onLedClick }) => {
  const radius = (l) => Math.hypot(l.na_x, l.na_y);
  const r = Math.max(na, ...leds.map(radius), 0.05) * 1.12;
  const dot = r * 0.035;
  const inView = (l) => (view === "fpm" ? radius(l) <= fpmNaMax : l.dpc.includes(view));
  const fill = (l) => {
    if (lit.includes(l.id)) return "#fdd835";
    if (view === "fpm" && inView(l)) return FPM_COLOR;
    if (view in PATTERN_COLORS && inView(l)) return PATTERN_COLORS[view];
    return "#9e9e9e";
  };
  const thin = r / 250;
  return (
    <Box sx={{ color: "text.primary" }}>
      <svg viewBox={`${-r} ${-r} ${2 * r} ${2 * r}`} style={{ width: "100%", maxWidth: 380, display: "block", margin: "auto" }}>
        <line x1={-r} y1={0} x2={r} y2={0} stroke="currentColor" strokeOpacity={0.25} strokeWidth={thin} />
        <line x1={0} y1={-r} x2={0} y2={r} stroke="currentColor" strokeOpacity={0.25} strokeWidth={thin} />
        <circle r={na} fill="none" stroke="currentColor" strokeWidth={thin * 2} strokeDasharray={`${r / 40} ${r / 40}`} />
        <circle r={nai} fill="none" stroke="#43a047" strokeOpacity={0.6} strokeWidth={thin * 1.5} />
        {naiInner > 0 && <circle r={naiInner} fill="none" stroke="#43a047" strokeOpacity={0.6} strokeWidth={thin * 1.5} />}
        {leds.map((l) => (
          <circle
            key={l.id}
            cx={l.na_x}
            cy={l.na_y}
            r={dot}
            fill={fill(l)}
            stroke={l.brightfield ? "currentColor" : "none"}
            strokeWidth={dot * 0.4}
            style={{ cursor: "pointer" }}
            onClick={() => onLedClick(l.id)}
          >
            <title>{`LED ${l.id} · NA ${radius(l).toFixed(2)} (${l.na_x.toFixed(2)}, ${l.na_y.toFixed(2)})${
              l.brightfield === true ? " · brightfield" : l.brightfield === false ? " · darkfield" : ""}`}</title>
          </circle>
        ))}
        <text x={r * 0.97} y={-r * 0.03} fontSize={r * 0.07} textAnchor="end" fill="currentColor">+x</text>
        <text x={r * 0.03} y={r * 0.95} fontSize={r * 0.07} fill="currentColor">+y</text>
      </svg>
      <Typography variant="caption" color="text.secondary" component="div" align="center">
        Illumination NA in camera frame. Dashed: objective NA, green: DPC illumination NA,
        outlined: brightfield (last calibration). Click an LED to light it.
      </Typography>
    </Box>
  );
};

const DPCController = () => {
  const dispatch = useDispatch();

  // Redux state
  const dpcState = useSelector(dpcSlice.getDpcState);
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const imageFormat = useSelector(liveStreamSlice.getLiveStreamState).imageFormat;

  // Local state
  const [localParams, setLocalParams] = useState({});
  const [layout, setLayout] = useState(null);
  const [view, setView] = useState("+x");
  const [lit, setLit] = useState([]);
  const [cal, setCal] = useState({ rotation_deg: 0, flip: false, offset_x_mm: 0, offset_y_mm: 0, offset_z_mm: 0 });
  const prevTask = useRef("");

  const baseUrl = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api`;
  const processedStreamUrl = getDpcStreamUrl(baseUrl, 85);

  const param = (key) => localParams[key] ?? dpcState[key];
  const rgb = [param("led_intensity_r"), param("led_intensity_g"), param("led_intensity_b")];
  const busy = dpcState.is_processing || Boolean(dpcState.task);
  const leds = layout?.leds ?? [];
  // WOTF of the optics as entered (applied or not); the calibration key refetches after a calibration
  const wotfUrl = getDpcWotfUrl(baseUrl, {
    ...Object.fromEntries(["wavelength", "na", "nai", "nai_inner"]
      .filter((k) => Number.isFinite(param(k))).map((k) => [k, param(k)])),
    cal: JSON.stringify(layout?.calibration ?? {}),
  });

  // Load parameters from backend
  const loadParameters = useCallback(async () => {
    try {
      const params = await getDpcParams(baseUrl);
      dispatch(dpcSlice.updateDpcParams(params));
      setLocalParams(params);
    } catch (error) {
      console.error("Failed to load DPC parameters:", error);
      dispatch(dpcSlice.setError("Failed to load parameters"));
    }
  }, [baseUrl, dispatch]);

  // Load processing state from backend
  const loadState = useCallback(async () => {
    try {
      const state = await getDpcState(baseUrl);
      dispatch(dpcSlice.updateDpcState(state));
    } catch (error) {
      console.error("Failed to load DPC state:", error);
    }
  }, [baseUrl, dispatch]);

  const loadLayout = useCallback(async () => {
    try {
      const result = await getLedLayout(baseUrl);
      setLayout(result);
      const c = result.calibration;
      setCal({ rotation_deg: c.rotation_deg, flip: c.flip, offset_x_mm: c.offset_mm[0],
        offset_y_mm: c.offset_mm[1], offset_z_mm: c.offset_mm[2] });
    } catch (error) {
      console.error("Failed to load LED layout:", error);
    }
  }, [baseUrl]);

  // Load everything on mount
  useEffect(() => {
    loadParameters();
    loadState();
    loadLayout();
  }, [loadParameters, loadState, loadLayout]);

  // The raw view shows the shared live stream: start it unless the Live View tab already has
  useEffect(() => {
    (async () => {
      try {
        const active = await apiViewControllerGetLiveViewActive();
        const res = active ? null : await apiLiveViewControllerStartLiveView(null, imageFormat || "jpeg");
        dispatch(liveViewSlice.setIsStreamRunning(res?.status !== "long_exposure"));
      } catch (error) {
        console.error("Failed to start the live stream:", error);
      }
    })();
  }, [imageFormat, dispatch]);

  // Poll state while live DPC or an LED scan runs; reload the layout when a scan ends
  useEffect(() => {
    if (prevTask.current && !dpcState.task) loadLayout();
    prevTask.current = dpcState.task;
    if (!busy) return undefined;
    const interval = setInterval(loadState, 1000);
    return () => clearInterval(interval);
  }, [busy, dpcState.task, loadState, loadLayout]);

  // Run an action with the current (unapplied) parameters
  const run = useCallback(async (action) => {
    try {
      dispatch(dpcSlice.setLoading(true));
      if (Object.keys(localParams).length) {
        dispatch(dpcSlice.updateDpcParams(await setDpcParams(baseUrl, localParams)));
        loadLayout();
      }
      const response = await action(baseUrl);
      if (response?.state) dispatch(dpcSlice.updateDpcState(response.state));
      loadState();
    } catch (error) {
      console.error("DPC action failed:", error);
      dispatch(dpcSlice.setError("Action failed"));
    } finally {
      dispatch(dpcSlice.setLoading(false));
    }
  }, [baseUrl, localParams, dispatch, loadState, loadLayout]);

  // Update parameter (local state)
  const handleParamChange = (paramName, value) => {
    setLocalParams((prev) => ({ ...prev, [paramName]: value }));
  };

  // Switch the streamed image; the backend re-sends the last result at once
  const showView = async (display) => {
    handleParamChange("display", display);
    try {
      dispatch(dpcSlice.updateDpcParams(await setDpcParams(baseUrl, { display })));
    } catch (error) {
      console.error("Failed to switch the DPC view:", error);
    }
  };

  // Apply parameters to backend (DPC patterns on the map follow the NA settings)
  const handleApplyParams = useCallback(async () => {
    try {
      dispatch(dpcSlice.setLoading(true));
      const updatedParams = await setDpcParams(baseUrl, localParams);
      dispatch(dpcSlice.updateDpcParams(updatedParams));
      loadLayout();
    } catch (error) {
      console.error("Failed to update DPC parameters:", error);
      dispatch(dpcSlice.setError("Failed to update parameters"));
    } finally {
      dispatch(dpcSlice.setLoading(false));
    }
  }, [baseUrl, localParams, dispatch, loadLayout]);

  const light = async (ids) => {
    try {
      await setLeds(baseUrl, ids, rgb);
      setLit(ids);
    } catch (error) {
      console.error("Failed to set LEDs:", error);
    }
  };
  const viewIds = (v) => leds.filter((l) => (v === "fpm"
    ? Math.hypot(l.na_x, l.na_y) <= param("fpm_na_max") : l.dpc.includes(v))).map((l) => l.id);

  const applyCalibration = async (next) => {
    try {
      setCal(next);
      setLayout(await setLedCalibration(baseUrl, next));
    } catch (error) {
      console.error("Failed to set LED calibration:", error);
    }
  };
  const last = layout?.last_calibration;

  const numberField = (label, key, step, helperText) => (
    <TextField
      label={label}
      type="number"
      value={param(key) ?? ""}
      onChange={(e) => handleParamChange(key, parseFloat(e.target.value))}
      inputProps={{ step }}
      size="small"
      fullWidth
      helperText={helperText}
    />
  );

  const calField = (label, key) => (
    <TextField
      label={label}
      type="number"
      value={cal[key]}
      onChange={(e) => setCal((c) => ({ ...c, [key]: parseFloat(e.target.value) || 0 }))}
      size="small"
      sx={{ width: "48%" }}
    />
  );

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" gutterBottom>
        DPC / FPM Controller
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Differential phase contrast and Fourier ptychography with any LED source
        (geodesic dome, ring board, LED matrix): patterns, transfer functions and
        k-vectors come from the LED positions.
      </Typography>

      {/* Status Card */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2">Status</Typography>
              <Chip
                label={dpcState.task ? `Scan: ${dpcState.task}` : dpcState.is_processing ? "Live DPC" : "Idle"}
                color={busy ? "success" : "default"}
                size="small"
              />
            </Grid>
            <Grid item xs={6} md={2}>
              <Typography variant="caption" color="text.secondary">
                Frames Captured
              </Typography>
              <Typography variant="h6">{dpcState.frame_count}</Typography>
            </Grid>
            <Grid item xs={6} md={2}>
              <Typography variant="caption" color="text.secondary">
                Processed
              </Typography>
              <Typography variant="h6">{dpcState.processed_count}</Typography>
            </Grid>
            <Grid item xs={6} md={2}>
              <Typography variant="caption" color="text.secondary">
                Processing FPS
              </Typography>
              <Typography variant="h6">
                {dpcState.processing_fps.toFixed(2)}
              </Typography>
            </Grid>
            <Grid item xs={6} md={2}>
              <Typography variant="caption" color="text.secondary">
                Last Process Time
              </Typography>
              <Typography variant="h6">
                {(dpcState.last_process_time * 1000).toFixed(0)} ms
              </Typography>
            </Grid>
            {(dpcState.task || dpcState.task_message) && (
              <Grid item xs={12}>
                {dpcState.task && (
                  <LinearProgress variant="determinate" value={100 * dpcState.task_progress} sx={{ mb: 1 }} />
                )}
                <Typography variant="body2" color="text.secondary">{dpcState.task_message}</Typography>
              </Grid>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* Control Buttons */}
      <Box sx={{ mb: 2, display: "flex", gap: 2, flexWrap: "wrap" }}>
        <Button
          variant="contained"
          color={dpcState.is_processing ? "error" : "primary"}
          startIcon={dpcState.is_processing ? <StopIcon /> : <PlayArrowIcon />}
          onClick={() => run(dpcState.is_processing ? stopDpcProcessing : startDpcProcessing)}
          disabled={dpcState.isLoading || Boolean(dpcState.task)}
        >
          {dpcState.is_processing ? "Stop Live DPC" : "Start Live DPC"}
        </Button>
        <Button
          variant="outlined"
          startIcon={<CameraAltIcon />}
          onClick={() => run(snapDpc)}
          disabled={dpcState.isLoading || busy}
        >
          Snap DPC (save)
        </Button>
        <Button
          variant="outlined"
          startIcon={<SaveIcon />}
          onClick={handleApplyParams}
          disabled={dpcState.isLoading || dpcState.is_processing}
        >
          Apply Parameters
        </Button>
        {dpcState.task && (
          <Button variant="contained" color="error" startIcon={<StopIcon />} onClick={() => stopDpcTask(baseUrl)}>
            Stop Scan
          </Button>
        )}
      </Box>

      {/* Main Content Grid */}
      <Grid container spacing={2}>
        {/* Left Column - Video Streams */}
        <Grid item xs={12} md={8}>
          {/* Raw Camera Stream */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Raw Camera Stream
            </Typography>
            <Box sx={{ minHeight: 320 }}>
              <LiveViewControlWrapper enableStageMovement={false} />
            </Box>
          </Paper>

          {/* Processed DPC Stream */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Box sx={{ mb: 1, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}>
              <Typography variant="h6">Reconstruction</Typography>
              <TextField
                select
                size="small"
                label="Show"
                value={param("display") ?? "phase"}
                onChange={(e) => showView(e.target.value)}
                sx={{ minWidth: 220 }}
              >
                {DISPLAY_VIEWS.map(([value, label]) => (
                  <MenuItem key={value} value={value}>{label}</MenuItem>
                ))}
              </TextField>
            </Box>
            <Box
              sx={{
                width: "100%",
                aspectRatio: "4/3",
                bgcolor: "black",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {dpcState.is_processing || dpcState.processed_count > 0 ? (
                <img
                  src={processedStreamUrl}
                  alt="Processed DPC stream"
                  style={{
                    maxWidth: "100%",
                    maxHeight: "100%",
                    objectFit: "contain",
                  }}
                  onError={(e) => {
                    console.error("Processed stream failed to load");
                    e.target.style.display = "none";
                  }}
                />
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Start live DPC or snap to view the reconstruction
                </Typography>
              )}
            </Box>
          </Paper>

          {/* Weak-object transfer functions of the optical parameters on the right */}
          <Accordion TransitionProps={{ unmountOnExit: true }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Transfer Functions (WOTF)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <img src={wotfUrl} alt="DPC weak-object transfer functions" style={{ width: "100%", display: "block" }} />
              <Typography variant="caption" color="text.secondary">
                Absorption (real) and phase (imaginary) transfer function of each DPC pattern for the
                optical parameters as entered, in NA units, camera frame (x right, y down). Phase is
                lost where all four phase WOTFs are close to zero.
              </Typography>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Right Column - LEDs and Parameters */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              LED Illumination
            </Typography>
            <ToggleButtonGroup
              size="small"
              exclusive
              value={view}
              onChange={(e, v) => v && setView(v)}
              sx={{ mb: 1, flexWrap: "wrap" }}
            >
              {Object.keys(PATTERN_COLORS).map((p) => (
                <ToggleButton key={p} value={p} sx={{ color: PATTERN_COLORS[p] }}>
                  {p} ({leds.filter((l) => l.dpc.includes(p)).length})
                </ToggleButton>
              ))}
              <ToggleButton value="fpm" sx={{ color: FPM_COLOR }}>
                FPM ({viewIds("fpm").length})
              </ToggleButton>
            </ToggleButtonGroup>
            <LedMap
              leds={leds}
              na={param("na") ?? 0.3}
              nai={param("nai") ?? 0.3}
              naiInner={param("nai_inner") ?? 0}
              fpmNaMax={param("fpm_na_max") ?? 1}
              view={view}
              lit={lit}
              onLedClick={(id) => light([id])}
            />
            <Box sx={{ mt: 1, display: "flex", gap: 1 }}>
              <Button size="small" variant="outlined" disabled={busy} onClick={() => light(viewIds(view))}>
                Light {view}
              </Button>
              <Button size="small" variant="outlined" disabled={busy} onClick={() => light([])}>
                All off
              </Button>
            </Box>
          </Paper>

          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Parameters
            </Typography>

            {/* LED -> camera calibration */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>LED Calibration</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Defocus a textured sample by ~50-100 µm and keep the exposure fixed. The scan lights
                    every LED once, finds the brightfield LEDs and fits the LED rotation / mirror from the
                    image shifts. The defocus direction is unknown: if the phase looks inverted, rotate by 180°.
                  </Typography>
                  <Button variant="contained" disabled={busy} onClick={() => run(startLedCalibration)}>
                    Run Calibration Scan
                  </Button>
                  {last && (
                    <Typography variant="body2">
                      Fit: rotation {last.rotation_deg.toFixed(1)}°, flip {String(last.flip)}, rms {last.rms_px.toFixed(2)} px
                      of {last.shift_spread_px.toFixed(1)} px spread, {last.n_brightfield} brightfield LEDs
                      {last.na_objective_estimate ? `, objective NA ≈ ${last.na_objective_estimate.toFixed(2)}` : ""}
                      {last.applied ? " (applied)" : " (not applied)"}
                    </Typography>
                  )}
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, justifyContent: "space-between" }}>
                    {calField("Rotation (°)", "rotation_deg")}
                    <FormControlLabel
                      sx={{ width: "48%", m: 0 }}
                      control={<Switch checked={Boolean(cal.flip)} onChange={(e) => setCal((c) => ({ ...c, flip: e.target.checked }))} />}
                      label="Mirror x"
                    />
                    {calField("Offset x (mm)", "offset_x_mm")}
                    {calField("Offset y (mm)", "offset_y_mm")}
                    {calField("Offset z (mm)", "offset_z_mm")}
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Offset = position of the LED structure relative to its layout coordinates; for a
                    flat board (ring, matrix) offset z is its height above the sample.
                  </Typography>
                  <Box sx={{ display: "flex", gap: 1 }}>
                    <Button variant="outlined" size="small" disabled={busy} onClick={() => applyCalibration(cal)}>
                      Apply Calibration
                    </Button>
                    <Button
                      variant="outlined"
                      size="small"
                      disabled={busy}
                      onClick={() => applyCalibration({ ...cal, rotation_deg: ((cal.rotation_deg + 360) % 360) - 180 })}
                    >
                      Rotate 180° (invert phase)
                    </Button>
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Fourier ptychography */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>FPM Acquisition</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  {numberField("Max. illumination NA", "fpm_na_max", 0.01,
                    `${viewIds("fpm").length} LEDs, one image each (TIFF stack + JSON with k-vectors)`)}
                  <Button variant="contained" disabled={busy} onClick={() => run(startFpmAcquisition)}>
                    Start FPM Scan
                  </Button>
                  <Typography variant="caption" color="text.secondary">
                    Saved to the data directory below; uses the LED colour and timing of the acquisition parameters.
                  </Typography>
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Optical Parameters */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Optical Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  {numberField("Pixel Size (μm, sample plane)", "pixelsize", 0.01)}
                  {numberField("Wavelength (μm)", "wavelength", 0.01)}
                  {numberField("Objective NA", "na", 0.01)}
                  {numberField("DPC Illumination NA (outer)", "nai", 0.01, "LEDs up to this NA form the DPC halves")}
                  {numberField("DPC Illumination NA (inner)", "nai_inner", 0.01, "> 0 for annular DPC")}
                  {numberField("Refractive Index (n)", "n", 0.01)}
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Acquisition Parameters */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Acquisition Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  {[["Red", "led_intensity_r"], ["Green", "led_intensity_g"], ["Blue", "led_intensity_b"]].map(([name, key]) => (
                    <Box key={key}>
                      <Typography gutterBottom>
                        LED Intensity {name}: {param(key)}
                      </Typography>
                      <Slider
                        value={param(key) ?? 0}
                        onChange={(e, value) => handleParamChange(key, value)}
                        min={0}
                        max={255}
                        step={1}
                      />
                    </Box>
                  ))}
                  {numberField("Wait Time (s)", "wait_time", 0.01, "Settling time after each LED change")}
                  <TextField
                    label="Frame Sync"
                    type="number"
                    value={param("frame_sync") ?? ""}
                    onChange={(e) => handleParamChange("frame_sync", parseInt(e.target.value))}
                    inputProps={{ step: 1, min: 1, max: 10 }}
                    size="small"
                    fullWidth
                    helperText="Frames to skip after an LED change (free-running cameras)"
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Reconstruction Parameters */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Reconstruction Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  {numberField("Regularization (Absorption)", "reg_u", 0.001, "Tikhonov regularization for absorption")}
                  {numberField("Regularization (Phase)", "reg_p", 0.001, "Tikhonov regularization for phase")}
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Data Saving */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Data Saving</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={Boolean(param("save_images"))}
                        onChange={(e) => handleParamChange("save_images", e.target.checked)}
                      />
                    }
                    label="Save every live DPC stack"
                  />
                  <TextField
                    label="Save Directory (DPC snaps, FPM scans)"
                    value={param("save_directory") ?? ""}
                    onChange={(e) => handleParamChange("save_directory", e.target.value)}
                    size="small"
                    fullWidth
                  />
                </Box>
              </AccordionDetails>
            </Accordion>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DPCController;
