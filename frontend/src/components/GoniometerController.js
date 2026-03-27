// src/components/GoniometerController.js
// Contact Angle Measurement Controller
// Provides live stream, snap, manual 3-point measurement, automated analysis,
// measurement history with CSV/JSON export, and result image download.

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useTheme } from "@mui/material/styles";
import {
  Box,
  Button,
  Card,
  CardContent,
  Slider,
  Typography,
  Grid,
  TextField,
  Divider,
  Chip,
  Stack,
  Paper,
  IconButton,
  Tooltip,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
} from "@mui/material";
import {
  CameraAlt as CameraAltIcon,
  PlayArrow as PlayArrowIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  TouchApp as TouchAppIcon,
  AutoFixHigh as AutoFixHighIcon,
} from "@mui/icons-material";

// Redux
import * as goniometerSlice from "../state/slices/GoniometerSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";

// Live stream viewer
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";

// API
import apiGoniometerControllerGetConfig from "../backendapi/apiGoniometerControllerGetConfig";
import apiGoniometerControllerSetConfig from "../backendapi/apiGoniometerControllerSetConfig";
import apiGoniometerControllerResetConfig from "../backendapi/apiGoniometerControllerResetConfig";
import apiGoniometerControllerSnap from "../backendapi/apiGoniometerControllerSnap";
import apiGoniometerControllerMeasureAuto from "../backendapi/apiGoniometerControllerMeasureAuto";
import apiGoniometerControllerMeasureManual from "../backendapi/apiGoniometerControllerMeasureManual";
import apiGoniometerControllerAddMeasurement from "../backendapi/apiGoniometerControllerAddMeasurement";
import apiGoniometerControllerClearMeasurements from "../backendapi/apiGoniometerControllerClearMeasurements";

// ────────────────────────────────────────────
// Helper: download a data URI as a file
// ────────────────────────────────────────────
function downloadDataUri(dataUri, filename) {
  const a = document.createElement("a");
  a.href = dataUri;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ────────────────────────────────────────────
// Helper: export measurements to CSV string
// ────────────────────────────────────────────
function measurementsToCSV(measurements) {
  const header = "id,timestamp,mode,left_angle,right_angle\n";
  const rows = measurements
    .map(
      (m) =>
        `${m.id},${new Date(m.timestamp * 1000).toISOString()},${m.mode},${m.left_angle ?? ""},${m.right_angle ?? ""}`
    )
    .join("\n");
  return header + rows;
}

// ────────────────────────────────────────────
// Component
// ────────────────────────────────────────────
const GoniometerController = () => {
  const dispatch = useDispatch();
  const theme = useTheme();

  const gState = useSelector(goniometerSlice.getGoniometerState);
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );

  // Local reference for the snapped-image canvas
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  // Track natural image size for coordinate mapping
  const [naturalSize, setNaturalSize] = useState({ w: 1, h: 1 });

  // ── Load config from backend on mount ──
  useEffect(() => {
    (async () => {
      try {
        const cfg = await apiGoniometerControllerGetConfig();
        dispatch(goniometerSlice.setConfig(cfg));
      } catch (e) {
        console.error("Failed to load goniometer config", e);
      }
    })();
  }, [dispatch]);

  // ── Snap ──
  const handleSnap = useCallback(async () => {
    dispatch(goniometerSlice.setIsSnapping(true));
    try {
      const data = await apiGoniometerControllerSnap();
      if (data.success) {
        dispatch(goniometerSlice.setSnappedImage(data.image));
        dispatch(goniometerSlice.setSnappedShape(data.shape));
        dispatch(goniometerSlice.setResultImage(null));
        dispatch(goniometerSlice.setLastResult(null));
        dispatch(goniometerSlice.resetManualPoints());
      }
    } catch (e) {
      console.error("Snap failed", e);
    } finally {
      dispatch(goniometerSlice.setIsSnapping(false));
    }
  }, [dispatch]);

  // ── Auto measure ──
  const handleAutoMeasure = useCallback(async () => {
    dispatch(goniometerSlice.setIsMeasuring(true));
    try {
      const data = await apiGoniometerControllerMeasureAuto();
      dispatch(goniometerSlice.setLastResult(data));
      if (data.annotated_image) {
        dispatch(goniometerSlice.setResultImage(data.annotated_image));
      }
    } catch (e) {
      console.error("Auto measure failed", e);
    } finally {
      dispatch(goniometerSlice.setIsMeasuring(false));
    }
  }, [dispatch]);

  // ── Manual measure ──
  const handleManualMeasure = useCallback(async () => {
    const { baselinePt1, baselinePt2, tangentPt } = gState.manualPoints;
    if (!baselinePt1 || !baselinePt2 || !tangentPt) return;
    dispatch(goniometerSlice.setIsMeasuring(true));
    try {
      const data = await apiGoniometerControllerMeasureManual({
        baseline_x1: baselinePt1.x,
        baseline_y1: baselinePt1.y,
        baseline_x2: baselinePt2.x,
        baseline_y2: baselinePt2.y,
        tangent_x: tangentPt.x,
        tangent_y: tangentPt.y,
      });
      dispatch(goniometerSlice.setLastResult(data));
      if (data.annotated_image) {
        dispatch(goniometerSlice.setResultImage(data.annotated_image));
      }
    } catch (e) {
      console.error("Manual measure failed", e);
    } finally {
      dispatch(goniometerSlice.setIsMeasuring(false));
    }
  }, [dispatch, gState.manualPoints]);

  // ── Add measurement to history ──
  const handleAddMeasurement = useCallback(async () => {
    const r = gState.lastResult;
    if (!r || !r.success) return;
    const entry = {
      timestamp: r.timestamp || Date.now() / 1000,
      left_angle: r.left_angle ?? r.angle ?? null,
      right_angle: r.right_angle ?? r.angle ?? null,
      mode: gState.activeTab,
    };
    try {
      const resp = await apiGoniometerControllerAddMeasurement(entry);
      if (resp.success) {
        dispatch(goniometerSlice.addMeasurement(resp.measurement));
      }
    } catch (e) {
      console.error("Add measurement failed", e);
    }
  }, [dispatch, gState.lastResult, gState.activeTab]);

  // ── Clear measurements ──
  const handleClearMeasurements = useCallback(async () => {
    try {
      await apiGoniometerControllerClearMeasurements();
      dispatch(goniometerSlice.clearMeasurements());
    } catch (e) {
      console.error("Clear measurements failed", e);
    }
  }, [dispatch]);

  // ── Export CSV ──
  const handleExportCSV = useCallback(() => {
    const csv = measurementsToCSV(gState.measurements);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    downloadDataUri(url, "contact_angle_measurements.csv");
    URL.revokeObjectURL(url);
  }, [gState.measurements]);

  // ── Export JSON ──
  const handleExportJSON = useCallback(() => {
    const json = JSON.stringify(gState.measurements, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    downloadDataUri(url, "contact_angle_measurements.json");
    URL.revokeObjectURL(url);
  }, [gState.measurements]);

  // ── Download result image ──
  const handleDownloadResult = useCallback(() => {
    if (gState.resultImage) {
      downloadDataUri(gState.resultImage, "contact_angle_result.jpg");
    }
  }, [gState.resultImage]);

  // ── Download snapped image ──
  const handleDownloadSnapped = useCallback(() => {
    if (gState.snappedImage) {
      downloadDataUri(gState.snappedImage, "goniometer_snap.jpg");
    }
  }, [gState.snappedImage]);

  // ── Config update ──
  const handleConfigChange = useCallback(
    async (key, value) => {
      const patch = { [key]: value };
      dispatch(goniometerSlice.setConfig(patch));
      try {
        await apiGoniometerControllerSetConfig(patch);
      } catch (e) {
        console.error("Config update failed", e);
      }
    },
    [dispatch]
  );

  const handleResetConfig = useCallback(async () => {
    try {
      const cfg = await apiGoniometerControllerResetConfig();
      dispatch(goniometerSlice.setConfig(cfg));
    } catch (e) {
      console.error("Reset config failed", e);
    }
  }, [dispatch]);

  // ── Click on snapped image (manual point placement) ──
  const handleImageClick = useCallback(
    (e) => {
      if (!gState.manualPlacingMode) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const scaleX = naturalSize.w / rect.width;
      const scaleY = naturalSize.h / rect.height;
      const x = Math.round((e.clientX - rect.left) * scaleX);
      const y = Math.round((e.clientY - rect.top) * scaleY);

      dispatch(
        goniometerSlice.setManualPoints({ [gState.manualPlacingMode]: { x, y } })
      );

      // Auto-advance placement mode
      const order = ["baselinePt1", "baselinePt2", "tangentPt"];
      const idx = order.indexOf(gState.manualPlacingMode);
      const next = idx < order.length - 1 ? order[idx + 1] : null;
      dispatch(goniometerSlice.setManualPlacingMode(next));
    },
    [dispatch, gState.manualPlacingMode, naturalSize]
  );

  // Track natural image dimensions when snapped image loads
  const handleImgLoad = useCallback((e) => {
    setNaturalSize({
      w: e.currentTarget.naturalWidth,
      h: e.currentTarget.naturalHeight,
    });
  }, []);

  // ── Manual point overlay (SVG circles on the image) ──
  const manualOverlay = useMemo(() => {
    const { baselinePt1, baselinePt2, tangentPt } = gState.manualPoints;
    const pts = [
      { pt: baselinePt1, color: "#00ff00", label: "B1" },
      { pt: baselinePt2, color: "#00ff00", label: "B2" },
      { pt: tangentPt, color: "#ff00ff", label: "T" },
    ];
    return pts
      .filter((p) => p.pt)
      .map((p) => ({ ...p }));
  }, [gState.manualPoints]);

  // ── Computed summary ──
  const meanAngle = useMemo(() => {
    if (gState.measurements.length === 0) return null;
    const lefts = gState.measurements.filter((m) => m.left_angle != null).map((m) => m.left_angle);
    const rights = gState.measurements.filter((m) => m.right_angle != null).map((m) => m.right_angle);
    const all = [...lefts, ...rights];
    if (all.length === 0) return null;
    const mean = all.reduce((a, b) => a + b, 0) / all.length;
    const std = Math.sqrt(all.reduce((a, b) => a + (b - mean) ** 2, 0) / all.length);
    return { mean, std };
  }, [gState.measurements]);

  // ── Config slider items ──
  const configSliders = [
    { key: "canny_low", label: "Canny Low", min: 0, max: 255, step: 1 },
    { key: "canny_high", label: "Canny High", min: 0, max: 255, step: 1 },
    { key: "blur_ksize", label: "Blur Kernel Size", min: 1, max: 15, step: 2 },
    { key: "bright_row_thresh", label: "Bright Row Threshold", min: 0, max: 255, step: 1 },
    { key: "roi_x_margin_frac", label: "ROI X Margin Frac", min: 0, max: 0.5, step: 0.01 },
    { key: "roi_y_above_frac", label: "ROI Y Above Frac", min: 0, max: 1, step: 0.01 },
    { key: "roi_y_below_px", label: "ROI Y Below (px)", min: 0, max: 200, step: 1 },
    { key: "min_contour_length", label: "Min Contour Length", min: 10, max: 500, step: 10 },
    { key: "baseline_tolerance", label: "Baseline Tolerance", min: 1, max: 30, step: 1 },
    { key: "local_fit_frac", label: "Local Fit Fraction", min: 0.01, max: 0.5, step: 0.01 },
    { key: "poly_degree", label: "Polynomial Degree", min: 1, max: 6, step: 1 },
    { key: "tangent_delta", label: "Tangent Delta", min: 0.1, max: 10, step: 0.1 },
  ];

  // ═══════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Goniometer – Contact Angle Measurement
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Snap · Annotate · Measure · Export
      </Typography>

      <Grid container spacing={2}>
        {/* ──────── LEFT: Image area ──────── */}
        <Grid item xs={12} md={8}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Live Camera Stream
              </Typography>
              <Box
                sx={{
                  width: "100%",
                  minHeight: 200,
                  maxHeight: 300,
                  backgroundColor: "#111",
                  borderRadius: 1,
                  overflow: "hidden",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <LiveViewControlWrapper enableStageMovement={false} />
              </Box>
              <Box sx={{ mt: 1, display: "flex", gap: 1 }}>
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<CameraAltIcon />}
                  onClick={handleSnap}
                  disabled={gState.isSnapping}
                >
                  {gState.isSnapping ? "Snapping…" : "Snap"}
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* ──── Snapped / Result image ──── */}
          <Card>
            <CardContent>
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                <Typography variant="subtitle1">
                  {gState.resultImage ? "Result Image" : "Snapped Image"}
                </Typography>
                <Stack direction="row" spacing={1}>
                  {gState.snappedImage && (
                    <Tooltip title="Download snapped image">
                      <IconButton size="small" onClick={handleDownloadSnapped}>
                        <DownloadIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                  {gState.resultImage && (
                    <Tooltip title="Download result image">
                      <IconButton size="small" onClick={handleDownloadResult} color="primary">
                        <DownloadIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                </Stack>
              </Box>

              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  minHeight: 350,
                  backgroundColor: "#222",
                  borderRadius: 1,
                  overflow: "hidden",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: gState.activeTab === "manual" && gState.manualPlacingMode ? "crosshair" : "default",
                }}
                onClick={gState.activeTab === "manual" ? handleImageClick : undefined}
              >
                {gState.resultImage || gState.snappedImage ? (
                  <>
                    <img
                      ref={imgRef}
                      src={gState.resultImage || gState.snappedImage}
                      alt="Goniometer"
                      onLoad={handleImgLoad}
                      style={{
                        maxWidth: "100%",
                        maxHeight: "100%",
                        objectFit: "contain",
                        display: "block",
                      }}
                    />
                    {/* Manual-point overlay (only when showing snapped, not result) */}
                    {gState.activeTab === "manual" && !gState.resultImage && manualOverlay.length > 0 && imgRef.current && (
                      <svg
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          width: "100%",
                          height: "100%",
                          pointerEvents: "none",
                        }}
                        viewBox={`0 0 ${naturalSize.w} ${naturalSize.h}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        {manualOverlay.map((p, i) => (
                          <g key={i}>
                            <circle cx={p.pt.x} cy={p.pt.y} r={8} fill={p.color} opacity={0.8} />
                            <text x={p.pt.x + 12} y={p.pt.y + 4} fill="#fff" fontSize={18} fontWeight="bold">
                              {p.label}
                            </text>
                          </g>
                        ))}
                        {/* Draw baseline line */}
                        {gState.manualPoints.baselinePt1 && gState.manualPoints.baselinePt2 && (
                          <line
                            x1={gState.manualPoints.baselinePt1.x}
                            y1={gState.manualPoints.baselinePt1.y}
                            x2={gState.manualPoints.baselinePt2.x}
                            y2={gState.manualPoints.baselinePt2.y}
                            stroke="#00ff00"
                            strokeWidth={3}
                          />
                        )}
                        {/* Draw tangent line to closest baseline pt */}
                        {gState.manualPoints.tangentPt && gState.manualPoints.baselinePt1 && gState.manualPoints.baselinePt2 && (() => {
                          const tp = gState.manualPoints.tangentPt;
                          const b1 = gState.manualPoints.baselinePt1;
                          const b2 = gState.manualPoints.baselinePt2;
                          const d1 = Math.hypot(tp.x - b1.x, tp.y - b1.y);
                          const d2 = Math.hypot(tp.x - b2.x, tp.y - b2.y);
                          const contact = d1 < d2 ? b1 : b2;
                          return (
                            <line
                              x1={contact.x} y1={contact.y}
                              x2={tp.x} y2={tp.y}
                              stroke="#ff00ff" strokeWidth={3}
                            />
                          );
                        })()}
                      </svg>
                    )}
                  </>
                ) : (
                  <Typography color="text.secondary">
                    Press Snap to capture an image
                  </Typography>
                )}
              </Box>

              {/* Angle readout bar */}
              <Box sx={{ mt: 1, display: "flex", justifyContent: "space-between" }}>
                <Typography variant="body2">
                  {gState.lastResult?.success
                    ? gState.lastResult.left_angle != null
                      ? `L: ${gState.lastResult.left_angle.toFixed(1)}°  R: ${gState.lastResult.right_angle?.toFixed(1) ?? "–"}°`
                      : `Angle: ${gState.lastResult.angle?.toFixed(1) ?? "–"}°`
                    : "Angle: –"}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {gState.snappedImage ? "Image ready" : "Snap an image to start"}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* ──────── RIGHT: Controls ──────── */}
        <Grid item xs={12} md={4}>
          {/* Tab selector */}
          <Card sx={{ mb: 2 }}>
            <Tabs
              value={gState.activeTab}
              onChange={(_, v) => dispatch(goniometerSlice.setActiveTab(v))}
              variant="fullWidth"
            >
              <Tab
                value="auto"
                label="Automated"
                icon={<AutoFixHighIcon />}
                iconPosition="start"
              />
              <Tab
                value="manual"
                label="Manual"
                icon={<TouchAppIcon />}
                iconPosition="start"
              />
            </Tabs>
          </Card>

          {/* ──── AUTO tab ──── */}
          {gState.activeTab === "auto" && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Automated Measurement
                </Typography>
                <Stack spacing={1}>
                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={gState.isMeasuring ? <CircularProgress size={18} /> : <PlayArrowIcon />}
                    onClick={handleAutoMeasure}
                    disabled={!gState.snappedImage || gState.isMeasuring}
                  >
                    Measure
                  </Button>
                  <Button
                    variant="outlined"
                    fullWidth
                    onClick={handleAddMeasurement}
                    disabled={!gState.lastResult?.success}
                  >
                    Add to History
                  </Button>
                </Stack>

                {/* Result display */}
                {gState.lastResult && (
                  <Paper variant="outlined" sx={{ mt: 2, p: 1.5 }}>
                    {gState.lastResult.success ? (
                      <>
                        <Typography variant="body2">
                          Left angle: <strong>{gState.lastResult.left_angle?.toFixed(2)}°</strong>
                        </Typography>
                        <Typography variant="body2">
                          Right angle: <strong>{gState.lastResult.right_angle?.toFixed(2)}°</strong>
                        </Typography>
                        {gState.lastResult.baseline_angle_deg != null && (
                          <Typography variant="caption" color="text.secondary">
                            Baseline tilt: {gState.lastResult.baseline_angle_deg.toFixed(2)}°
                          </Typography>
                        )}
                      </>
                    ) : (
                      <Typography variant="body2" color="error">
                        Detection failed – adjust parameters and retry
                      </Typography>
                    )}
                  </Paper>
                )}
              </CardContent>
            </Card>
          )}

          {/* ──── MANUAL tab ──── */}
          {gState.activeTab === "manual" && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Manual 3-Point Measurement
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                  Click on the image to place points. Two baseline points define the substrate,
                  one tangent point defines the drop edge direction.
                </Typography>

                {/* Point placement buttons */}
                <Stack spacing={1}>
                  {[
                    { key: "baselinePt1", label: "Baseline Point 1", color: "success" },
                    { key: "baselinePt2", label: "Baseline Point 2", color: "success" },
                    { key: "tangentPt", label: "Tangent Point", color: "secondary" },
                  ].map(({ key, label, color }) => (
                    <Button
                      key={key}
                      variant={gState.manualPlacingMode === key ? "contained" : "outlined"}
                      color={color}
                      size="small"
                      onClick={() =>
                        dispatch(
                          goniometerSlice.setManualPlacingMode(
                            gState.manualPlacingMode === key ? null : key
                          )
                        )
                      }
                      disabled={!gState.snappedImage}
                    >
                      {label}{" "}
                      {gState.manualPoints[key]
                        ? `(${gState.manualPoints[key].x}, ${gState.manualPoints[key].y})`
                        : "–"}
                    </Button>
                  ))}
                </Stack>

                <Divider sx={{ my: 1.5 }} />

                <Stack direction="row" spacing={1}>
                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={gState.isMeasuring ? <CircularProgress size={18} /> : <PlayArrowIcon />}
                    onClick={handleManualMeasure}
                    disabled={
                      !gState.manualPoints.baselinePt1 ||
                      !gState.manualPoints.baselinePt2 ||
                      !gState.manualPoints.tangentPt ||
                      gState.isMeasuring
                    }
                  >
                    Measure
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => {
                      dispatch(goniometerSlice.resetManualPoints());
                      dispatch(goniometerSlice.setResultImage(null));
                      dispatch(goniometerSlice.setLastResult(null));
                    }}
                  >
                    Reset
                  </Button>
                </Stack>

                <Button
                  variant="outlined"
                  fullWidth
                  sx={{ mt: 1 }}
                  onClick={handleAddMeasurement}
                  disabled={!gState.lastResult?.success}
                >
                  Add to History
                </Button>

                {gState.lastResult && gState.lastResult.success && (
                  <Paper variant="outlined" sx={{ mt: 2, p: 1.5 }}>
                    <Typography variant="body2">
                      Angle: <strong>{(gState.lastResult.angle ?? gState.lastResult.left_angle)?.toFixed(2)}°</strong>
                    </Typography>
                  </Paper>
                )}
              </CardContent>
            </Card>
          )}

          {/* ──── Advanced Config (auto mode) ──── */}
          <Accordion
            expanded={gState.showAdvancedConfig}
            onChange={() => dispatch(goniometerSlice.setShowAdvancedConfig(!gState.showAdvancedConfig))}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SettingsIcon sx={{ mr: 1 }} />
              <Typography variant="subtitle2">Processing Parameters</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
                {configSliders.map(({ key, label, min, max, step }) => (
                  <Box key={key}>
                    <Typography variant="caption">
                      {label}: {gState.config[key]}
                    </Typography>
                    <Slider
                      size="small"
                      value={gState.config[key]}
                      min={min}
                      max={max}
                      step={step}
                      onChange={(_, v) => dispatch(goniometerSlice.setConfig({ [key]: v }))}
                      onChangeCommitted={(_, v) => handleConfigChange(key, v)}
                    />
                  </Box>
                ))}
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={handleResetConfig}
                >
                  Reset to Defaults
                </Button>
              </Stack>
            </AccordionDetails>
          </Accordion>

          {/* ──── Measurement History ──── */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Measurements ({gState.measurements.length})
              </Typography>

              {gState.measurements.length > 0 ? (
                <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 250 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>#</TableCell>
                        <TableCell>Mode</TableCell>
                        <TableCell align="right">Left (°)</TableCell>
                        <TableCell align="right">Right (°)</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {gState.measurements.map((m) => (
                        <TableRow key={m.id}>
                          <TableCell>{m.id}</TableCell>
                          <TableCell>
                            <Chip label={m.mode} size="small" variant="outlined" />
                          </TableCell>
                          <TableCell align="right">
                            {m.left_angle != null ? m.left_angle.toFixed(1) : "–"}
                          </TableCell>
                          <TableCell align="right">
                            {m.right_angle != null ? m.right_angle.toFixed(1) : "–"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No measurements yet
                </Typography>
              )}

              {meanAngle && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Mean ± StdDev: <strong>{meanAngle.mean.toFixed(1)}° ± {meanAngle.std.toFixed(1)}°</strong>
                </Typography>
              )}

              <Stack direction="row" spacing={1} sx={{ mt: 1.5 }}>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={handleExportCSV}
                  disabled={gState.measurements.length === 0}
                >
                  CSV
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={handleExportJSON}
                  disabled={gState.measurements.length === 0}
                >
                  JSON
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={handleClearMeasurements}
                  disabled={gState.measurements.length === 0}
                >
                  Clear
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default GoniometerController;
