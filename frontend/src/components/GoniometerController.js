// src/components/GoniometerController.js
// Contact Angle Measurement Controller
// Provides live stream with crop ROI, snap, manual 3-point measurement,
// automated mirror-geometry analysis, focus monitor, zoom/pan, history with export.

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useTheme } from "@mui/material/styles";
import {
  Alert,
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  IconButton,
  LinearProgress,
  Paper,
  Slider,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  Tooltip,
  Typography,
} from "@mui/material";
import {
  AutoFixHigh as AutoFixHighIcon,
  CameraAlt as CameraAltIcon,
  Close as CloseIcon,
  Crop as CropIcon,
  CropFree as CropFreeIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayArrowIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  TouchApp as TouchAppIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
} from "@mui/icons-material";

// Redux
import * as goniometerSlice from "../state/slices/GoniometerSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice";

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
import apiGoniometerControllerSetCrop from "../backendapi/apiGoniometerControllerSetCrop";
import apiGoniometerControllerResetCrop from "../backendapi/apiGoniometerControllerResetCrop";
import apiGoniometerControllerGetFocusMetric from "../backendapi/apiGoniometerControllerGetFocusMetric";

// ────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────
function downloadDataUri(dataUri, filename) {
  const a = document.createElement("a");
  a.href = dataUri;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function measurementsToCSV(measurements) {
  const header = "id,timestamp,mode,left_angle,right_angle\n";
  const rows = measurements
    .map(
      (m) =>
        `${m.id},${new Date(m.timestamp * 1000).toISOString()},${m.mode},${
          m.left_angle ?? ""
        },${m.right_angle ?? ""}`
    )
    .join("\n");
  return header + rows;
}

// Config slider definitions — mirror-geometry algorithm (8 params)
const CONFIG_SLIDERS = [
  { key: "canny_low",        label: "Canny Low",             min: 0,    max: 255,  step: 1    },
  { key: "canny_high",       label: "Canny High",            min: 0,    max: 255,  step: 1    },
  { key: "blur_ksize",       label: "Blur Kernel Size",      min: 1,    max: 21,   step: 2    },
  { key: "fit_frac",         label: "Local Fit Fraction",    min: 0.01, max: 0.5,  step: 0.01 },
  { key: "poly_degree",      label: "Polynomial Degree",     min: 1,    max: 6,    step: 1    },
  { key: "tangent_delta",    label: "Tangent Delta (px)",    min: 0.1,  max: 20,   step: 0.1  },
  { key: "angle_range_low",  label: "Angle Min (°)",         min: 0,    max: 90,   step: 1    },
  { key: "angle_range_high", label: "Angle Max (°)",         min: 90,   max: 180,  step: 1    },
];

// ────────────────────────────────────────────
// Component
// ────────────────────────────────────────────
const GoniometerController = () => {
  const dispatch = useDispatch();
  // theme available for potential future use
  // eslint-disable-next-line no-unused-vars
  const theme = useTheme();

  const gState = useSelector(goniometerSlice.getGoniometerState);
  // connectionSettings available for potential future use
  // eslint-disable-next-line no-unused-vars
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  // Stream subsampling factor
  const streamSubsampling = useMemo(
    () =>
      liveStreamState.streamSettings?.jpeg?.subsampling?.factor ||
      liveStreamState.streamSettings?.jpeg?.subsampling_factor ||
      liveStreamState.streamSettings?.binary?.subsampling?.factor ||
      1,
    [liveStreamState.streamSettings]
  );

  // Refs for image display (snapped / result)
  const imgRef = useRef(null);
  const containerRef = useRef(null); // overflow:hidden outer container
  const wrapperRef = useRef(null);   // CSS-transform inner wrapper

  // Refs for zoom / pan interaction
  const longPressTimerRef = useRef(null);
  const isPanningRef = useRef(false);
  const panStartRef = useRef(null); // { mouseX, mouseY, offsetX, offsetY }

  // Rolling max for focus metric progress bar
  const focusMaxRef = useRef(1);

  // Natural image dimensions (in pixels)
  const [naturalSize, setNaturalSize] = useState({ w: 1, h: 1 });

  // Crop drag refs & display state
  const cropDragStartRef = useRef(null); // {x,y} in container display px
  const [cropOverlayRect, setCropOverlayRect] = useState(null); // {x,y,w,h}

  // Ref for the live stream container (crop coordinate measurements)
  const liveContainerRef = useRef(null);

  // ─── Load config from backend on mount ───
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

  // ─── Focus monitor polling (500 ms) ───
  useEffect(() => {
    if (!gState.showFocusMonitor) return;
    const id = setInterval(async () => {
      try {
        const data = await apiGoniometerControllerGetFocusMetric();
        if (data.success) {
          focusMaxRef.current = Math.max(focusMaxRef.current, data.focus_metric);
          dispatch(goniometerSlice.setFocusMetric(data.focus_metric));
        }
      } catch (_) {
        // silently ignore poll errors
      }
    }, 500);
    return () => clearInterval(id);
  }, [dispatch, gState.showFocusMonitor]);

  // ─── Snap ───
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
        dispatch(goniometerSlice.resetZoomPan());
      }
    } catch (e) {
      console.error("Snap failed", e);
    } finally {
      dispatch(goniometerSlice.setIsSnapping(false));
    }
  }, [dispatch]);

  // ─── Auto measure ───
  const handleAutoMeasure = useCallback(async () => {
    dispatch(goniometerSlice.setIsMeasuring(true));
    try {
      const data = await apiGoniometerControllerMeasureAuto();
      dispatch(goniometerSlice.setLastResult(data));
      if (data.annotated_image) {
        dispatch(goniometerSlice.setResultImage(data.annotated_image));
        dispatch(goniometerSlice.resetZoomPan());
      }
    } catch (e) {
      console.error("Auto measure failed", e);
    } finally {
      dispatch(goniometerSlice.setIsMeasuring(false));
    }
  }, [dispatch]);

  // ─── Manual measure ───
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
        dispatch(goniometerSlice.resetZoomPan());
      }
    } catch (e) {
      console.error("Manual measure failed", e);
    } finally {
      dispatch(goniometerSlice.setIsMeasuring(false));
    }
  }, [dispatch, gState.manualPoints]);

  // ─── Add measurement to history ───
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

  // ─── Clear measurements ───
  const handleClearMeasurements = useCallback(async () => {
    try {
      await apiGoniometerControllerClearMeasurements();
      dispatch(goniometerSlice.clearMeasurements());
    } catch (e) {
      console.error("Clear measurements failed", e);
    }
  }, [dispatch]);

  // ─── Export ───
  const handleExportCSV = useCallback(() => {
    const csv = measurementsToCSV(gState.measurements);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    downloadDataUri(url, "contact_angle_measurements.csv");
    URL.revokeObjectURL(url);
  }, [gState.measurements]);

  const handleExportJSON = useCallback(() => {
    const json = JSON.stringify(gState.measurements, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    downloadDataUri(url, "contact_angle_measurements.json");
    URL.revokeObjectURL(url);
  }, [gState.measurements]);

  const handleDownloadResult = useCallback(() => {
    if (gState.resultImage) {
      downloadDataUri(gState.resultImage, "contact_angle_result.jpg");
    }
  }, [gState.resultImage]);

  const handleDownloadSnapped = useCallback(() => {
    if (gState.snappedImage) {
      downloadDataUri(gState.snappedImage, "goniometer_snap.jpg");
    }
  }, [gState.snappedImage]);

  // ─── Config update ───
  const handleConfigChange = useCallback(
    async (key, value) => {
      dispatch(goniometerSlice.setConfig({ [key]: value }));
      try {
        await apiGoniometerControllerSetConfig({ [key]: value });
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

  // ─── Crop drag on live stream ───
  const handleCropMouseDown = useCallback((e) => {
    e.preventDefault();
    const rect = liveContainerRef.current?.getBoundingClientRect();
    if (!rect) return;
    cropDragStartRef.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
    setCropOverlayRect(null);
  }, []);

  const handleCropMouseMove = useCallback((e) => {
    if (!cropDragStartRef.current) return;
    const rect = liveContainerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;
    setCropOverlayRect({
      x: Math.min(cropDragStartRef.current.x, curX),
      y: Math.min(cropDragStartRef.current.y, curY),
      w: Math.abs(curX - cropDragStartRef.current.x),
      h: Math.abs(curY - cropDragStartRef.current.y),
    });
  }, []);

  const handleCropMouseUp = useCallback(
    async (e) => {
      if (!cropDragStartRef.current) return;
      const container = liveContainerRef.current;
      if (!container) return;
      const containerRect = container.getBoundingClientRect();
      const endX = e.clientX - containerRect.left;
      const endY = e.clientY - containerRect.top;
      const startX = cropDragStartRef.current.x;
      const startY = cropDragStartRef.current.y;
      cropDragStartRef.current = null;
      setCropOverlayRect(null);

      if (Math.abs(endX - startX) < 5 || Math.abs(endY - startY) < 5) {
        return; // too small to be intentional
      }

      // Find the actual rendered stream element (canvas for WebGL/JPEG, img or video for others)
      // to correctly account for CSS scaling (browser scales JPEG to fit container) and
      // letter/pillarboxing offsets.
      const streamEl = container.querySelector("canvas, img, video");
      let x1, y1, x2, y2;
      if (streamEl) {
        const elRect = streamEl.getBoundingClientRect();
        const imgOffsetX = elRect.left - containerRect.left;
        const imgOffsetY = elRect.top - containerRect.top;
        const renderedW = elRect.width;
        const renderedH = elRect.height;

        // canvas.width = intrinsic pixel width (set by LiveViewComponent to JPEG dimensions)
        // img.naturalWidth = natural pixel width
        // Both equal camera_width / streamSubsampling when subsampling is active,
        // or camera_width directly when subsampling_factor=1.
        const natW =
          streamEl.naturalWidth ||
          streamEl.videoWidth ||
          streamEl.width ||
          renderedW;
        const natH =
          streamEl.naturalHeight ||
          streamEl.videoHeight ||
          streamEl.height ||
          renderedH;

        // Total scale: camera pixels per display pixel
        // = (natural_px * subsampling) / rendered_px  = camera_px / rendered_px
        const scaleX = renderedW > 0 ? (natW * streamSubsampling) / renderedW : streamSubsampling;
        const scaleY = renderedH > 0 ? (natH * streamSubsampling) / renderedH : streamSubsampling;

        const toCamX = (dpx) => Math.round(Math.max(0, dpx - imgOffsetX) * scaleX);
        const toCamY = (dpy) => Math.round(Math.max(0, dpy - imgOffsetY) * scaleY);
        x1 = toCamX(Math.min(startX, endX));
        y1 = toCamY(Math.min(startY, endY));
        x2 = toCamX(Math.max(startX, endX));
        y2 = toCamY(Math.max(startY, endY));
      } else {
        // Fallback when no stream element is found
        const toCam = (dpx) => Math.round(dpx * streamSubsampling);
        x1 = toCam(Math.min(startX, endX));
        y1 = toCam(Math.min(startY, endY));
        x2 = toCam(Math.max(startX, endX));
        y2 = toCam(Math.max(startY, endY));
      }

      console.log("Crop ROI (camera px):", { x1, y1, x2, y2 });
      try {
        await apiGoniometerControllerSetCrop({ x1, y1, x2, y2 });
        dispatch(goniometerSlice.setCropRoi({ x1, y1, x2, y2 }));
      } catch (err) {
        console.error("Set crop failed", err);
      }
      dispatch(goniometerSlice.setIsCropMode(false));
    },
    [dispatch, streamSubsampling]
  );

  const handleResetCrop = useCallback(async () => {
    try {
      await apiGoniometerControllerResetCrop();
      dispatch(goniometerSlice.setCropRoi(null));
    } catch (e) {
      console.error("Reset crop failed", e);
    }
  }, [dispatch]);

  // ─── Image load ───
  const handleImgLoad = useCallback((e) => {
    setNaturalSize({
      w: e.currentTarget.naturalWidth,
      h: e.currentTarget.naturalHeight,
    });
  }, []);

  // ─── Zoom / pan ───
  const handleZoomIn = useCallback(() => {
    dispatch(goniometerSlice.setZoomLevel(gState.zoomLevel + 1));
  }, [dispatch, gState.zoomLevel]);

  const handleZoomOut = useCallback(() => {
    dispatch(goniometerSlice.setZoomLevel(gState.zoomLevel - 1));
  }, [dispatch, gState.zoomLevel]);

  const handleImageMouseDown = useCallback(
    (e) => {
      longPressTimerRef.current = setTimeout(() => {
        isPanningRef.current = true;
        panStartRef.current = {
          mouseX: e.clientX,
          mouseY: e.clientY,
          offsetX: gState.panOffset.x,
          offsetY: gState.panOffset.y,
        };
      }, 300);
    },
    [gState.panOffset]
  );

  const handleImageMouseMove = useCallback(
    (e) => {
      if (!isPanningRef.current || !panStartRef.current) return;
      dispatch(
        goniometerSlice.setPanOffset({
          x: panStartRef.current.offsetX + (e.clientX - panStartRef.current.mouseX),
          y: panStartRef.current.offsetY + (e.clientY - panStartRef.current.mouseY),
        })
      );
    },
    [dispatch]
  );

  const handleImageMouseUp = useCallback(
    (e) => {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;

      if (isPanningRef.current) {
        isPanningRef.current = false;
        panStartRef.current = null;
        return; // was panning — do not place a point
      }

      // Click → manual point placement
      if (gState.activeTab !== "manual" || !gState.manualPlacingMode) return;
      const container = containerRef.current;
      if (!container || !imgRef.current) return;

      const rect = container.getBoundingClientRect();
      const cw = rect.width;
      const ch = rect.height;

      // Screen coords relative to container center
      const mx = e.clientX - rect.left - cw / 2;
      const my = e.clientY - rect.top - ch / 2;

      // Invert CSS transform: scale(zoom) translate(panX/zoom, panY/zoom)
      // Forward: screen = zoom * element + pan
      // Inverse: element = (screen - pan) / zoom
      const zoom = gState.zoomLevel;
      const lx = (mx - gState.panOffset.x) / zoom + cw / 2;
      const ly = (my - gState.panOffset.y) / zoom + ch / 2;

      // Image is centered inside the wrapper div
      const imgW = imgRef.current.offsetWidth;
      const imgH = imgRef.current.offsetHeight;
      const imgLeft = (cw - imgW) / 2;
      const imgTop = (ch - imgH) / 2;
      const imgX = lx - imgLeft;
      const imgY = ly - imgTop;

      if (imgX < 0 || imgY < 0 || imgX > imgW || imgY > imgH) return;

      const x = Math.round(imgX * (naturalSize.w / imgW));
      const y = Math.round(imgY * (naturalSize.h / imgH));

      dispatch(
        goniometerSlice.setManualPoints({ [gState.manualPlacingMode]: { x, y } })
      );
      const order = ["baselinePt1", "baselinePt2", "tangentPt"];
      const idx = order.indexOf(gState.manualPlacingMode);
      const next = idx < order.length - 1 ? order[idx + 1] : null;
      dispatch(goniometerSlice.setManualPlacingMode(next));
    },
    [dispatch, gState.activeTab, gState.manualPlacingMode, gState.zoomLevel, gState.panOffset, naturalSize]
  );

  const handleImageDoubleClick = useCallback(() => {
    dispatch(goniometerSlice.resetZoomPan());
  }, [dispatch]);

  const handleImageMouseLeave = useCallback(() => {
    clearTimeout(longPressTimerRef.current);
    isPanningRef.current = false;
    panStartRef.current = null;
  }, []);

  // ─── Manual point overlay data ───
  const manualOverlay = useMemo(() => {
    const { baselinePt1, baselinePt2, tangentPt } = gState.manualPoints;
    return [
      { pt: baselinePt1, color: "#00ff00", label: "B1" },
      { pt: baselinePt2, color: "#00ff00", label: "B2" },
      { pt: tangentPt,   color: "#ff00ff", label: "T"  },
    ].filter((p) => p.pt);
  }, [gState.manualPoints]);

  // ─── Statistics ───
  const meanAngle = useMemo(() => {
    if (!gState.measurements.length) return null;
    const all = gState.measurements.flatMap((m) =>
      [m.left_angle, m.right_angle].filter((v) => v != null)
    );
    if (!all.length) return null;
    const mean = all.reduce((a, b) => a + b, 0) / all.length;
    const std = Math.sqrt(all.reduce((a, b) => a + (b - mean) ** 2, 0) / all.length);
    return { mean, std };
  }, [gState.measurements]);

  // Focus progress bar value (0–100)
  const focusProgress = useMemo(() => {
    if (gState.focusMetric == null) return 0;
    return Math.min(100, (gState.focusMetric / focusMaxRef.current) * 100);
  }, [gState.focusMetric]);

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
          {/* Live stream */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  mb: 1,
                }}
              >
                <Typography variant="subtitle1">Live Camera Stream</Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  {gState.cropRoi && (
                    <Chip
                      label="Crop active"
                      size="small"
                      color="warning"
                      onDelete={handleResetCrop}
                      deleteIcon={<CloseIcon />}
                    />
                  )}
                  <Tooltip
                    title={
                      gState.isCropMode
                        ? "Cancel crop selection"
                        : "Draw crop region on live view"
                    }
                  >
                    <IconButton
                      size="small"
                      color={gState.isCropMode ? "warning" : "default"}
                      onClick={() => {
                        dispatch(goniometerSlice.setIsCropMode(!gState.isCropMode));
                        setCropOverlayRect(null);
                        cropDragStartRef.current = null;
                      }}
                    >
                      {gState.isCropMode ? <CropFreeIcon /> : <CropIcon />}
                    </IconButton>
                  </Tooltip>
                </Stack>
              </Box>

              {gState.isCropMode && (
                <Alert severity="info" sx={{ mb: 1 }}>
                  Click and drag on the live view to select the droplet crop region.
                </Alert>
              )}

              {/* Live view with crop overlay */}
              <Box
                ref={liveContainerRef}
                sx={{
                  position: "relative",
                  width: "100%",
                  minHeight: 200,
                  maxHeight: "50vh",
                  backgroundColor: "#111",
                  borderRadius: 1,
                  overflow: "hidden",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <LiveViewControlWrapper enableStageMovement={false} />

                {gState.isCropMode && (
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      cursor: "crosshair",
                      userSelect: "none",
                    }}
                    onMouseDown={handleCropMouseDown}
                    onMouseMove={handleCropMouseMove}
                    onMouseUp={handleCropMouseUp}
                  >
                    {cropOverlayRect && (
                      <svg
                        style={{
                          position: "absolute",
                          inset: 0,
                          width: "100%",
                          height: "100%",
                          pointerEvents: "none",
                        }}
                      >
                        <rect
                          x={cropOverlayRect.x}
                          y={cropOverlayRect.y}
                          width={cropOverlayRect.w}
                          height={cropOverlayRect.h}
                          fill="rgba(255,200,0,0.15)"
                          stroke="#ffd700"
                          strokeWidth={2}
                          strokeDasharray="6 3"
                        />
                      </svg>
                    )}
                  </Box>
                )}
              </Box>

              <Box sx={{ mt: 1 }}>
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

          {/* Snapped / Result image with zoom/pan */}
          <Card>
            <CardContent>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  mb: 1,
                }}
              >
                <Typography variant="subtitle1">
                  {gState.resultImage ? "Result Image" : "Snapped Image"}
                </Typography>
                <Stack direction="row" spacing={0.5} alignItems="center">
                  <Tooltip title="Zoom out (or double-click image to reset)">
                    <span>
                      <IconButton
                        size="small"
                        onClick={handleZoomOut}
                        disabled={gState.zoomLevel <= 1}
                      >
                        <ZoomOutIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                  <Typography variant="caption" sx={{ minWidth: 28, textAlign: "center" }}>
                    {gState.zoomLevel}×
                  </Typography>
                  <Tooltip title="Zoom in">
                    <span>
                      <IconButton
                        size="small"
                        onClick={handleZoomIn}
                        disabled={gState.zoomLevel >= 8}
                      >
                        <ZoomInIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                  {gState.snappedImage && !gState.resultImage && (
                    <Tooltip title="Download snapped image">
                      <IconButton size="small" onClick={handleDownloadSnapped}>
                        <DownloadIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                  {gState.resultImage && (
                    <Tooltip title="Download result image">
                      <IconButton
                        size="small"
                        onClick={handleDownloadResult}
                        color="primary"
                      >
                        <DownloadIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                </Stack>
              </Box>

              {/* Image container */}
              <Box
                ref={containerRef}
                sx={{
                  position: "relative",
                  width: "100%",
                  height: 350,
                  backgroundColor: "#222",
                  borderRadius: 1,
                  overflow: "hidden",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor:
                    gState.activeTab === "manual" && gState.manualPlacingMode
                      ? "crosshair"
                      : "grab",
                }}
                onMouseDown={handleImageMouseDown}
                onMouseMove={handleImageMouseMove}
                onMouseUp={handleImageMouseUp}
                onMouseLeave={handleImageMouseLeave}
                onDoubleClick={handleImageDoubleClick}
              >
                {gState.resultImage || gState.snappedImage ? (
                  <Box
                    ref={wrapperRef}
                    sx={{
                      position: "absolute",
                      inset: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      transform: `scale(${gState.zoomLevel}) translate(${
                        gState.panOffset.x / gState.zoomLevel
                      }px, ${gState.panOffset.y / gState.zoomLevel}px)`,
                      transformOrigin: "center center",
                      willChange: "transform",
                    }}
                  >
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
                        userSelect: "none",
                        pointerEvents: "none",
                      }}
                    />
                    {/* Manual point SVG overlay — only on snapped image before measuring */}
                    {gState.activeTab === "manual" &&
                      !gState.resultImage &&
                      manualOverlay.length > 0 && (
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
                              <circle
                                cx={p.pt.x}
                                cy={p.pt.y}
                                r={8}
                                fill={p.color}
                                opacity={0.85}
                              />
                              <text
                                x={p.pt.x + 12}
                                y={p.pt.y + 4}
                                fill="#fff"
                                fontSize={18}
                                fontWeight="bold"
                              >
                                {p.label}
                              </text>
                            </g>
                          ))}
                          {gState.manualPoints.baselinePt1 &&
                            gState.manualPoints.baselinePt2 && (
                              <line
                                x1={gState.manualPoints.baselinePt1.x}
                                y1={gState.manualPoints.baselinePt1.y}
                                x2={gState.manualPoints.baselinePt2.x}
                                y2={gState.manualPoints.baselinePt2.y}
                                stroke="#00ff00"
                                strokeWidth={3}
                              />
                            )}
                          {gState.manualPoints.tangentPt &&
                            gState.manualPoints.baselinePt1 &&
                            gState.manualPoints.baselinePt2 &&
                            (() => {
                              const tp = gState.manualPoints.tangentPt;
                              const b1 = gState.manualPoints.baselinePt1;
                              const b2 = gState.manualPoints.baselinePt2;
                              const contact =
                                Math.hypot(tp.x - b1.x, tp.y - b1.y) <
                                Math.hypot(tp.x - b2.x, tp.y - b2.y)
                                  ? b1
                                  : b2;
                              return (
                                <line
                                  x1={contact.x}
                                  y1={contact.y}
                                  x2={tp.x}
                                  y2={tp.y}
                                  stroke="#ff00ff"
                                  strokeWidth={3}
                                />
                              );
                            })()}
                        </svg>
                      )}
                  </Box>
                ) : (
                  <Typography color="text.secondary">
                    Press Snap to capture an image
                  </Typography>
                )}
              </Box>

              {/* Angle readout bar */}
              <Box
                sx={{ mt: 1, display: "flex", justifyContent: "space-between" }}
              >
                <Typography variant="body2">
                  {gState.lastResult?.success
                    ? gState.lastResult.left_angle != null
                      ? `L: ${gState.lastResult.left_angle.toFixed(1)}°  R: ${
                          gState.lastResult.right_angle?.toFixed(1) ?? "–"
                        }°`
                      : `Angle: ${gState.lastResult.angle?.toFixed(1) ?? "–"}°`
                    : "Angle: –"}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {gState.zoomLevel > 1
                    ? `${gState.zoomLevel}× — double-click to reset`
                    : gState.snappedImage
                    ? "Image ready"
                    : "Snap an image to start"}
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

                {/* Focus monitor toggle */}
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    mb: 1,
                  }}
                >
                  <Typography variant="caption" color="text.secondary">
                    Focus Monitor
                  </Typography>
                  <Tooltip
                    title={
                      gState.showFocusMonitor
                        ? "Hide focus monitor"
                        : "Show live sharpness value"
                    }
                  >
                    <IconButton
                      size="small"
                      color={gState.showFocusMonitor ? "primary" : "default"}
                      onClick={() =>
                        dispatch(
                          goniometerSlice.setShowFocusMonitor(!gState.showFocusMonitor)
                        )
                      }
                    >
                      {gState.showFocusMonitor ? (
                        <VisibilityIcon />
                      ) : (
                        <VisibilityOffIcon />
                      )}
                    </IconButton>
                  </Tooltip>
                </Box>

                {gState.showFocusMonitor && (
                  <Box sx={{ mb: 1.5 }}>
                    <Alert severity="info" sx={{ mb: 1, py: 0.5, fontSize: "0.75rem" }}>
                      Maximize the focus value by adjusting the focus knob before
                      measuring.
                    </Alert>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={focusProgress}
                        sx={{ flex: 1, height: 8, borderRadius: 4 }}
                        color={
                          focusProgress > 66
                            ? "success"
                            : focusProgress > 33
                            ? "warning"
                            : "error"
                        }
                      />
                      <Typography variant="caption" sx={{ minWidth: 56, textAlign: "right" }}>
                        {gState.focusMetric != null
                          ? gState.focusMetric.toFixed(0)
                          : "–"}
                      </Typography>
                    </Box>
                  </Box>
                )}

                <Stack spacing={1}>
                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={
                      gState.isMeasuring ? (
                        <CircularProgress size={18} />
                      ) : (
                        <PlayArrowIcon />
                      )
                    }
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

                {gState.lastResult && (
                  <Paper variant="outlined" sx={{ mt: 2, p: 1.5 }}>
                    {gState.lastResult.success ? (
                      <>
                        <Typography variant="body2">
                          Left:{" "}
                          <strong>
                            {gState.lastResult.left_angle?.toFixed(2)}°
                          </strong>
                        </Typography>
                        <Typography variant="body2">
                          Right:{" "}
                          <strong>
                            {gState.lastResult.right_angle?.toFixed(2)}°
                          </strong>
                        </Typography>
                        {gState.lastResult.baseline_tilt_deg != null && (
                          <Typography variant="caption" color="text.secondary">
                            Baseline tilt:{" "}
                            {gState.lastResult.baseline_tilt_deg.toFixed(2)}°
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
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: "block", mb: 1 }}
                >
                  Click image to place points. Long-press to pan. Double-click to
                  reset zoom.
                </Typography>

                <Stack spacing={1}>
                  {[
                    {
                      key: "baselinePt1",
                      label: "Baseline Pt 1 (left contact)",
                      color: "success",
                    },
                    {
                      key: "baselinePt2",
                      label: "Baseline Pt 2 (right contact)",
                      color: "success",
                    },
                    {
                      key: "tangentPt",
                      label: "Tangent Point (drop edge)",
                      color: "secondary",
                    },
                  ].map(({ key, label, color }) => (
                    <Button
                      key={key}
                      variant={
                        gState.manualPlacingMode === key ? "contained" : "outlined"
                      }
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
                    startIcon={
                      gState.isMeasuring ? (
                        <CircularProgress size={18} />
                      ) : (
                        <PlayArrowIcon />
                      )
                    }
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

                {gState.lastResult?.success && (
                  <Paper variant="outlined" sx={{ mt: 2, p: 1.5 }}>
                    <Typography variant="body2">
                      Angle:{" "}
                      <strong>
                        {(
                          gState.lastResult.angle ?? gState.lastResult.left_angle
                        )?.toFixed(2)}
                        °
                      </strong>
                    </Typography>
                  </Paper>
                )}
              </CardContent>
            </Card>
          )}

          {/* ──── Processing Parameters ──── */}
          <Accordion
            expanded={gState.showAdvancedConfig}
            onChange={() =>
              dispatch(
                goniometerSlice.setShowAdvancedConfig(!gState.showAdvancedConfig)
              )
            }
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SettingsIcon sx={{ mr: 1 }} />
              <Typography variant="subtitle2">Processing Parameters</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
                {CONFIG_SLIDERS.map(({ key, label, min, max, step }) => (
                  <Box key={key}>
                    <Typography variant="caption">
                      {label}: {gState.config[key]}
                    </Typography>
                    <Slider
                      size="small"
                      value={gState.config[key] ?? min}
                      min={min}
                      max={max}
                      step={step}
                      onChange={(_, v) =>
                        dispatch(goniometerSlice.setConfig({ [key]: v }))
                      }
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
                <TableContainer
                  component={Paper}
                  variant="outlined"
                  sx={{ maxHeight: 250 }}
                >
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
                  Mean ± StdDev:{" "}
                  <strong>
                    {meanAngle.mean.toFixed(1)}° ± {meanAngle.std.toFixed(1)}°
                  </strong>
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
