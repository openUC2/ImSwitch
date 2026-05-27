// src/components/FRAMESettings/StageOffsetCalibrationTab.js
//
// Single source of truth for stage-offset calibration. Flow:
//   1. (one-time) Ask the user whether the stage was homed since boot. Offer
//      "Home now" buttons - the persisted offset is only meaningful relative
//      to a stable device origin.
//   2. Pick the inserted reference layout (openUC2 chart, Heidstar slide …);
//      the layout map shows slot rectangles + current stage position; click
//      moves the stage there.
//   3. Scan: a raster around the current position records (x, y, intensity)
//      tuples. The backend (a) renders into a JSON sidecar so the heatmap
//      survives a page reload and (b) parks the stage at the brightest
//      sample at the end so the user can verify visually.
//   4. Review heatmap. Click anywhere to move the stage there (still in old
//      user coords). The brightest sample is shown next to the layout's
//      known/expected coordinate.
//   5. Accept: a confirmation dialog asks the user to confirm overwriting
//      the persisted offset; we pass the device position observed at the
//      brightest spot so the math is atomic.
//
import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
} from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  ButtonGroup,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stepper,
  Step,
  StepLabel,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";

import LiveViewControlWrapper from "../../axon/LiveViewControlWrapper";
import * as positionSlice from "../../state/slices/PositionSlice";
import * as liveStreamSlice from "../../state/slices/LiveStreamSlice";
import * as liveViewSlice from "../../state/slices/LiveViewSlice";
import apiSettingsControllerGetDetectorNames from "../../backendapi/apiSettingsControllerGetDetectorNames";
import IlluminationController from "../IlluminationController";
import JoystickControl from "../JoystickControl";
import apiStageCenterCalibrationPerformCalibration from "../../backendapi/apiStageCenterCalibrationPerformCalibration";
import apiStageCenterCalibrationStopCalibration from "../../backendapi/apiStageCenterCalibrationStopCalibration";
import apiStageCenterCalibrationGetHeatmap from "../../backendapi/apiStageCenterCalibrationGetHeatmap";
import apiStageCenterCalibrationGetLatestHeatmap from "../../backendapi/apiStageCenterCalibrationGetLatestHeatmap";
import apiStageCenterCalibrationGetIsRunning from "../../backendapi/apiStageCenterCalibrationGetIsRunning";
import apiStageCenterCalibrationGetRecommendedScanParameters from "../../backendapi/apiStageCenterCalibrationGetRecommendedScanParameters";
import apiExperimentControllerGetKnownCalibrationLayouts from "../../backendapi/apiExperimentControllerGetKnownCalibrationLayouts";
import apiPositionerControllerMovePositioner from "../../backendapi/apiPositionerControllerMovePositioner";
import apiPositionerControllerSetStageOffsetAxis from "../../backendapi/apiPositionerControllerSetStageOffsetAxis";
import apiPositionerControllerGetDevicePositionAxis from "../../backendapi/apiPositionerControllerGetDevicePositionAxis";
import apiPositionerControllerHomeAxis from "../../backendapi/apiPositionerControllerHomeAxis";
import apiObjectiveControllerGetStatus from "../../backendapi/apiObjectiveControllerGetStatus";
import apiLiveViewControllerStartLiveView from "../../backendapi/apiLiveViewControllerStartLiveView";
import apiLiveViewControllerStopLiveView from "../../backendapi/apiLiveViewControllerStopLiveView";
import { getConnectionSettingsState } from "../../state/slices/ConnectionSettingsSlice";


const STEPS = ["Insert slide & start scan", "Review heatmap & accept"];

// Local key used to track whether the user has confirmed the stage was homed
// in this session. Cleared on page reload (sessionStorage) so the prompt
// reappears after each restart.
const HOMING_ACK_KEY = "stageOffsetHomingAck.v1";

// Fallback layouts when the backend has no entry yet.
const FALLBACK_LAYOUTS = [
  {
    name: "Heidstar 4x Histosample",
    x: 18400,
    y: 40600,
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
    x: 63500,
    y: 43000,
    description: "openUC2 96-well calibration chart - well A1 (slot 1).",
    bounds: { width: 127000, height: 86000 },
    slots: [{ x: 14380, y: 11240, w: 9000, h: 9000, name: "A1" }],
  },
];


// ----------------------------------------------------------------------
// LayoutMapMini - small canvas rendering layout slots + the current stage
// position. Click moves the stage to that XY (absolute, blocking).
// Optionally overlays the planned scan area (current ± maxRadiusUm) and the
// individual tile grid (one tile = one image at FOV size). Helps the user
// see what they are about to scan before clicking start.
// ----------------------------------------------------------------------
const LayoutMapMini = ({
  layout,
  stageX,
  stageY,
  knownX,
  knownY,
  brightestX,
  brightestY,
  scanCenter = null,
  maxRadiusUm = 0,
  stepUm = 0,
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

    ctx.strokeStyle = "#666";
    ctx.lineWidth = 1;
    ctx.strokeRect(offX, offY, lw * s, lh * s);

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

    // Scan area preview - centred on ``scanCenter`` (defaults to the current
    // stage position when given). Edge dashed = ± maxRadiusUm, faint grid =
    // tile boundaries spaced by stepUm so the user can see where each FOV
    // sample lands.
    const centerForScan = scanCenter || (stageX != null && stageY != null ? { x: stageX, y: stageY } : null);
    if (centerForScan && maxRadiusUm > 0) {
      const cx = offX + centerForScan.x * s;
      const cy = offY + centerForScan.y * s;
      const half = maxRadiusUm * s;
      ctx.save();
      ctx.strokeStyle = "rgba(140, 220, 140, 0.9)";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.2;
      ctx.strokeRect(cx - half, cy - half, 2 * half, 2 * half);
      ctx.setLineDash([]);
      if (stepUm > 0) {
        // Number of tiles per side mirrors the controller: ``n_half`` rows on
        // each side of the centre + the centre row itself.
        const nHalf = Math.max(1, Math.round(maxRadiusUm / Math.max(stepUm, 1)));
        const start = -nHalf * stepUm;
        const end = nHalf * stepUm;
        ctx.strokeStyle = "rgba(140, 220, 140, 0.25)";
        ctx.lineWidth = 0.5;
        for (let i = -nHalf; i <= nHalf + 1; i += 1) {
          const t = start + (i + nHalf) * stepUm;
          const px = cx + t * s;
          if (px < cx - half - 0.5 || px > cx + half + 0.5) continue;
          ctx.beginPath();
          ctx.moveTo(px, cy - half);
          ctx.lineTo(px, cy + half);
          ctx.stroke();
          const py = cy + t * s;
          if (py < cy - half - 0.5 || py > cy + half + 0.5) continue;
          ctx.beginPath();
          ctx.moveTo(cx - half, py);
          ctx.lineTo(cx + half, py);
          ctx.stroke();
        }
        // light centre cross
        ctx.strokeStyle = "rgba(140, 220, 140, 0.65)";
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy);
        ctx.lineTo(cx + 5, cy);
        ctx.moveTo(cx, cy - 5);
        ctx.lineTo(cx, cy + 5);
        ctx.stroke();
      }
      ctx.restore();
    }

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

    if (brightestX != null && brightestY != null) {
      const bx = offX + brightestX * s;
      const by = offY + brightestY * s;
      ctx.fillStyle = "#ff5577";
      ctx.beginPath();
      ctx.arc(bx, by, 4, 0, 2 * Math.PI);
      ctx.fill();
    }

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
  }, [layout, stageX, stageY, knownX, knownY, brightestX, brightestY, scanCenter, maxRadiusUm, stepUm]);

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
// HeatmapCanvas - intensity grid + click-to-pick. We deliberately separate
// "click to move the stage" from "click to pick the calibration target":
// after a scan the user is more likely to want to pick a brighter sample
// than to physically move the stage there. ``onPick`` returns the (x, y)
// the user clicked, ``onMove`` is called if held with shift.
// ----------------------------------------------------------------------
// Shared LUT used for both the heatmap fill and its colorbar legend so that
// the user can read off intensity values consistent with the rendered grid.
const heatmapColor = (t) => {
  const tt = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.max(0, tt - 0.5) * 2);
  const g = Math.round(255 * Math.min(1, tt * 1.5));
  const b = Math.round(255 * Math.max(0, 1 - tt * 1.5));
  return `rgb(${r},${g},${b})`;
};

const HeatmapCanvas = ({ data, onPick, onMove, picked, width = 420, height = 420 }) => {
  const canvasRef = useRef(null);
  const projectionRef = useRef(null);
  // Right-side colorbar; reserved so the click-to-stage projection only spans
  // the heatmap area, not the legend.
  const COLORBAR_W = 24;
  const COLORBAR_PAD_LEFT = 6;

  // Range cached after each draw so the colorbar labels can display min/max
  // without re-iterating the samples in the render path.
  const [range, setRange] = useState({ iMin: null, iMax: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const totalW = canvas.width;
    const h = canvas.height;
    const w = totalW - COLORBAR_W - COLORBAR_PAD_LEFT;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, totalW, h);
    if (!data?.samples?.length) {
      projectionRef.current = null;
      // Still draw an empty colorbar so the layout doesn't jump.
      const cbx = w + COLORBAR_PAD_LEFT;
      for (let i = 0; i < h; i += 1) {
        ctx.fillStyle = heatmapColor(1 - i / h);
        ctx.fillRect(cbx, i, COLORBAR_W, 1);
      }
      setRange({ iMin: null, iMax: null });
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

    const stepUm = data.meta?.step_um || 250;
    const cellW = Math.max(1, Math.round(w / Math.max(1, Math.round(xRange / stepUm) + 1)));
    const cellH = Math.max(1, Math.round(h / Math.max(1, Math.round(yRange / stepUm) + 1)));

    data.samples.forEach((s) => {
      const px = ((s.x - xMin) / xRange) * (w - cellW);
      const py = h - cellH - ((s.y - yMin) / yRange) * (h - cellH);
      const t = (s.intensity - iMin) / iRange;
      ctx.fillStyle = heatmapColor(t);
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

    if (picked) {
      const px = ((picked.x - xMin) / xRange) * (w - cellW);
      const py = h - cellH - ((picked.y - yMin) / yRange) * (h - cellH);
      ctx.strokeStyle = "#33ff99";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px + cellW / 2, py + cellH / 2, Math.max(cellW, cellH) + 2, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Colorbar strip on the right edge.
    const cbx = w + COLORBAR_PAD_LEFT;
    for (let i = 0; i < h; i += 1) {
      const t = 1 - i / (h - 1);
      ctx.fillStyle = heatmapColor(t);
      ctx.fillRect(cbx, i, COLORBAR_W, 1);
    }
    ctx.strokeStyle = "#888";
    ctx.lineWidth = 1;
    ctx.strokeRect(cbx, 0, COLORBAR_W, h - 1);

    projectionRef.current = { xMin, xMax, yMin, yMax, w, h, cellW, cellH };
    setRange({ iMin, iMax });
  }, [data, picked]);

  const eventToStage = (event) => {
    const proj = projectionRef.current;
    if (!proj) return null;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cx = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const cy = ((event.clientY - rect.top) / rect.height) * canvas.height;
    const { xMin, xMax, yMin, yMax, w, h, cellW, cellH } = proj;
    if (cx > w) return null; // Ignore clicks on the colorbar strip.
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const stageX = xMin + (cx / (w - cellW)) * xRange;
    const stageY = yMin + ((h - cellH - cy) / (h - cellH)) * yRange;
    return { x: stageX, y: stageY };
  };

  const handleClick = (event) => {
    const xy = eventToStage(event);
    if (!xy) return;
    if (event.shiftKey && onMove) {
      onMove(xy.x, xy.y);
    } else if (onPick) {
      onPick(xy.x, xy.y);
    }
  };

  const handleContextMenu = (event) => {
    event.preventDefault();
    const xy = eventToStage(event);
    if (xy && onMove) onMove(xy.x, xy.y);
  };

  return (
    <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1, background: "#111" }}>
      <Box sx={{ position: "relative" }}>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{ width: "100%", height: "auto", display: "block", cursor: "crosshair" }}
          onClick={handleClick}
          onContextMenu={handleContextMenu}
        />
        {/* Colorbar min/max labels. Positioned over the rightmost strip. */}
        <Box
          sx={{
            position: "absolute",
            top: 0,
            right: 2,
            height: "100%",
            width: 36,
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            pointerEvents: "none",
            color: "#ddd",
            textShadow: "0 0 2px #000",
            fontSize: 10,
            textAlign: "right",
          }}
        >
          <span>{range.iMax != null ? range.iMax.toFixed(1) : "max"}</span>
          <span>{range.iMin != null ? range.iMin.toFixed(1) : "min"}</span>
        </Box>
      </Box>
      <Typography variant="caption" color="textSecondary" sx={{ display: "block", mt: 0.5 }}>
        Click to pick the calibration target. Shift-click or right-click to move the stage there.
        Colorbar shows mean intensity range.
      </Typography>
    </Box>
  );
};


// ----------------------------------------------------------------------
// Main tab
// ----------------------------------------------------------------------
const StageOffsetCalibrationTab = () => {
  const dispatch = useDispatch();
  const positionState = useSelector(positionSlice.getPositionState);
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  const [activeStep, setActiveStep] = useState(0);
  const [layouts, setLayouts] = useState(FALLBACK_LAYOUTS);
  const [layoutName, setLayoutName] = useState(FALLBACK_LAYOUTS[0].name);

  // Active objective drives step_um (one tile per FOV by default). User can
  // still tweak the step manually.
  const [objectiveStatus, setObjectiveStatus] = useState(null);
  const [stepUm, setStepUm] = useState(250);
  const [maxRadiusUm, setMaxRadiusUm] = useState(5000);
  const [recommended, setRecommended] = useState(null);
  // Has the user manually overridden step_um? Once yes we stop auto-syncing
  // from objective changes so we don't trample on their intent.
  const stepManualRef = useRef(false);

  // Polling-driven scan-running state (controller spawns its own thread, so
  // local optimistic toggles never flip back).
  const [pollingRunning, setPollingRunning] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [heatmap, setHeatmap] = useState(null);

  // The user can override the brightest pick by clicking on the heatmap.
  // null means "use the auto-detected brightest sample".
  const [pickedTarget, setPickedTarget] = useState(null);

  // Manual override applied to the *known* point (gold cross). Empty string
  // means "use the layout default".
  const [overrideKnownX, setOverrideKnownX] = useState("");
  const [overrideKnownY, setOverrideKnownY] = useState("");

  // Homing handshake. Defaults to "not acked" so the prompt appears on first
  // load of the tab and after every page reload.
  const [homingAck, setHomingAck] = useState(false);
  const [homingDialogOpen, setHomingDialogOpen] = useState(false);
  const [homingBusy, setHomingBusy] = useState(false);

  // Accept-confirmation dialog.
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmBusy, setConfirmBusy] = useState(false);

  // Detector-tab switching mirrors the LiveView page: dispatching activeTab
  // alone won't restart the stream when LiveView itself is unmounted, so we
  // run the same stop → wait → start sequence locally to avoid two parallel
  // streams pushing frames.
  const detectorSwitchBusyRef = useRef(false);

  const pollRef = useRef(null);

  const selectedLayout = useMemo(
    () => layouts.find((l) => l.name === layoutName) || layouts[0],
    [layouts, layoutName]
  );

  const knownX =
    overrideKnownX !== "" ? Number(overrideKnownX) : selectedLayout?.x;
  const knownY =
    overrideKnownY !== "" ? Number(overrideKnownY) : selectedLayout?.y;

  // Effective target = manually picked sample (if any) else brightest.
  const targetXY = useMemo(() => {
    if (pickedTarget) return pickedTarget;
    if (heatmap?.brightest) return { x: heatmap.brightest.x, y: heatmap.brightest.y };
    return null;
  }, [pickedTarget, heatmap]);

  // Tile count + ETA. Mirrors the controller's ``n_half = round(max_r/step)``
  // and ``2*n_half + 1`` per side. 1 s per tile is a coarse estimate that
  // matches the empirical performance of the raster worker.
  const tileGeom = useMemo(() => {
    const step = Number(stepUm) || 0;
    const radius = Number(maxRadiusUm) || 0;
    if (step <= 0 || radius <= 0) return { nx: 0, ny: 0, total: 0, etaS: 0 };
    const nHalf = Math.max(1, Math.round(radius / step));
    const nx = 2 * nHalf + 1;
    const ny = nx;
    const total = nx * ny;
    return { nx, ny, total, etaS: total };
  }, [stepUm, maxRadiusUm]);

  // ----- bootstrap -------------------------------------------------------

  useEffect(() => {
    let cancelled = false;
    (async () => {
      // Restore homing handshake from session storage.
      try {
        const ack = sessionStorage.getItem(HOMING_ACK_KEY);
        if (ack === "1") setHomingAck(true);
      } catch (e) {
        /* ignore */
      }
      // Layouts.
      try {
        const res = await apiExperimentControllerGetKnownCalibrationLayouts();
        if (cancelled) return;
        if (res?.layouts?.length) {
          const merged = res.layouts.map((b) => {
            const fb = FALLBACK_LAYOUTS.find((f) => f.name === b.name);
            return { ...(fb || {}), ...b };
          });
          FALLBACK_LAYOUTS.forEach((f) => {
            if (!merged.find((m) => m.name === f.name)) merged.push(f);
          });
          setLayouts(merged);
        }
      } catch (e) {
        /* keep fallback */
      }
      // Recommended scan params.
      try {
        const rec = await apiStageCenterCalibrationGetRecommendedScanParameters();
        if (cancelled) return;
        if (rec?.success) {
          setRecommended(rec);
          setStepUm(Math.round(rec.recommendedStepUm));
          setMaxRadiusUm(Math.round(rec.recommendedMaxRadiusUm));
        }
      } catch (e) {
        /* static defaults */
      }
      // Pre-load the latest heatmap so the tab repopulates after reload.
      // ``getLatestHeatmap`` is new (post-rework) so older backends 404 - in
      // that case fall back to the live ``getCalibrationHeatmap`` which is
      // empty after a fresh boot anyway.
      try {
        let latest = await apiStageCenterCalibrationGetLatestHeatmap();
        if (!latest || latest.samples == null) {
          latest = await apiStageCenterCalibrationGetHeatmap();
        }
        if (cancelled) return;
        if (latest && latest.samples && latest.samples.length) {
          setHeatmap(latest);
          setActiveStep(1);
        }
      } catch (e) {
        /* no-op */
      }
      // Objective FOV → step_um. Step defaults to min(FOV_x, FOV_y) so each
      // raster node corresponds to one full image.
      try {
        const status = await apiObjectiveControllerGetStatus();
        if (cancelled || !status) return;
        setObjectiveStatus(status);
        const fov = status.FOV || [];
        const fovMin = Math.min(...fov.filter((v) => v && Number.isFinite(v)));
        if (Number.isFinite(fovMin) && fovMin > 0 && !stepManualRef.current) {
          setStepUm(Math.round(fovMin));
        }
      } catch (e) {
        /* objective endpoint may not exist - keep static defaults */
      }
      // Populate the detector tab strip. LiveView normally fills this in
      // Redux but only while it is mounted; if the user lands directly on
      // the FRAMESettings tab we still need the list ourselves.
      try {
        if (!liveViewState?.detectors || liveViewState.detectors.length === 0) {
          const names = await apiSettingsControllerGetDetectorNames();
          if (cancelled) return;
          if (Array.isArray(names) && names.length) {
            dispatch(liveViewSlice.setDetectors(names));
          }
        }
      } catch (e) {
        /* detectors will be empty - tab strip just hides itself */
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
          setActiveStep(1);
          // Final fetch to grab the parked-at metadata.
          try {
            const final = await apiStageCenterCalibrationGetHeatmap();
            if (final && final.samples) setHeatmap(final);
          } catch (e) {
            /* ignore */
          }
        }
      } catch (e) {
        /* keep polling */
      }
    }, 1000);
  };

  // ----- actions ---------------------------------------------------------

  const acquirePosition = () => ({
    x: positionState?.x ?? 0,
    y: positionState?.y ?? 0,
  });

  const handleStart = async () => {
    setError("");
    setInfo("");
    if (!homingAck) {
      setHomingDialogOpen(true);
      return;
    }
    setHeatmap(null);
    setPickedTarget(null);
    setActiveStep(0);
    setPollingRunning(true);
    startPolling();
    try {
      const { x, y } = acquirePosition();
      const result = await apiStageCenterCalibrationPerformCalibration({
        start_x: x,
        start_y: y,
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
    // The user often stops the scan early because they already see the bright
    // spot. Persist the partial heatmap (backend does this too) and grab a
    // final snapshot so the brightest sample is immediately available for
    // accept without resetting any UI state.
    try {
      const stopped = await apiStageCenterCalibrationStopCalibration();
      if (stopped && stopped.samples) {
        setHeatmap(stopped);
      } else {
        try {
          const final = await apiStageCenterCalibrationGetHeatmap();
          if (final && final.samples) setHeatmap(final);
        } catch (e) {
          /* ignore */
        }
      }
    } catch (e) {
      /* ignore */
    }
    setPollingRunning(false);
    stopPolling();
    setActiveStep(1);
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

  // ---- homing handshake ------------------------------------------------

  const ackHoming = () => {
    setHomingAck(true);
    try {
      sessionStorage.setItem(HOMING_ACK_KEY, "1");
    } catch (e) {
      /* ignore */
    }
    setHomingDialogOpen(false);
  };

  const runHomeAxis = async (axis) => {
    setHomingBusy(true);
    try {
      await apiPositionerControllerHomeAxis({ axis, isBlocking: true });
    } catch (e) {
      setError(`Homing ${axis} failed: ${e.message || e}`);
    } finally {
      setHomingBusy(false);
    }
  };

  const homeAndAck = async () => {
    setHomingBusy(true);
    try {
      await apiPositionerControllerHomeAxis({ axis: "X", isBlocking: true });
      await apiPositionerControllerHomeAxis({ axis: "Y", isBlocking: true });
      ackHoming();
    } catch (e) {
      setError(`Homing failed: ${e.message || e}`);
    } finally {
      setHomingBusy(false);
    }
  };

  // ---- detector-tab switch (same UX as the LiveView page) ----------------

  const handleDetectorTabChange = async (newIndex) => {
    const detectors = liveViewState?.detectors || [];
    if (newIndex === liveViewState?.activeTab) return;
    // Optimistically update activeTab so the Tabs UI is responsive even
    // while the stream is being restarted.
    dispatch(liveViewSlice.setActiveTab(newIndex));
    if (detectorSwitchBusyRef.current) return;
    detectorSwitchBusyRef.current = true;
    try {
      // Stop whatever is currently streaming, wait briefly so the backend
      // releases the camera, then start the new one. This mirrors LiveView.js
      // and prevents the brief two-stream overlap the user reported.
      await apiLiveViewControllerStopLiveView();
      await new Promise((resolve) => setTimeout(resolve, 200));
      const newDetectorName = detectors[newIndex] || null;
      const protocol = liveStreamState?.imageFormat || "jpeg";
      const savedParams =
        newDetectorName &&
        liveStreamState?.perDetectorSettings?.[newDetectorName];
      const overrideParams =
        savedParams && savedParams.protocol === protocol ? savedParams : null;
      const result = await apiLiveViewControllerStartLiveView(
        newDetectorName,
        protocol,
        overrideParams
      );
      if (result?.params && newDetectorName) {
        dispatch(
          liveStreamSlice.updateDetectorSettings({
            detectorName: newDetectorName,
            settings: result.params,
          })
        );
      }
    } catch (e) {
      setError(`Failed to switch detector: ${e.message || e}`);
    } finally {
      detectorSwitchBusyRef.current = false;
    }
  };

  // ---- accept / store offset -------------------------------------------

  const openConfirm = () => {
    setError("");
    setInfo("");
    if (!targetXY) {
      setError("No calibration target picked. Run a scan first.");
      return;
    }
    if (knownX == null || knownY == null) {
      setError("No known calibration point.");
      return;
    }
    setConfirmOpen(true);
  };

  const acceptOffset = async () => {
    setConfirmBusy(true);
    setError("");
    setInfo("");
    try {
      // Read the raw device position right now so the math is atomic. If the
      // stage is parked at the target (default), this is the device coord we
      // want; if the user moved away after the scan we still capture the
      // *current* device position to keep the contract well-defined.
      const dxRaw = await apiPositionerControllerGetDevicePositionAxis({ axis: "X" });
      const dyRaw = await apiPositionerControllerGetDevicePositionAxis({ axis: "Y" });
      const currentDeviceX = Number(dxRaw);
      const currentDeviceY = Number(dyRaw);
      await apiPositionerControllerSetStageOffsetAxis({
        axis: "X",
        knownPosition: knownX,
        currentDevicePosition: currentDeviceX,
      });
      await apiPositionerControllerSetStageOffsetAxis({
        axis: "Y",
        knownPosition: knownY,
        currentDevicePosition: currentDeviceY,
      });
      setInfo(
        `Stage offset stored. Device (${currentDeviceX.toFixed(1)}, ` +
          `${currentDeviceY.toFixed(1)}) -> known (${knownX.toFixed(1)}, ` +
          `${knownY.toFixed(1)}).`
      );
      // Drop the old heatmap because the coord frame just changed; the user
      // should re-scan to verify. The latest-on-disk one is still in the
      // *old* frame so we do not reload it here.
      setHeatmap(null);
      setPickedTarget(null);
      setActiveStep(0);
    } catch (e) {
      setError(`Failed to store offset: ${e.message || e}`);
    } finally {
      setConfirmBusy(false);
      setConfirmOpen(false);
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
        well-defined hole, then store the offset that maps the current device
        position to the known physical coordinate.
      </Typography>

      {!homingAck && (
        <Alert
          severity="warning"
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={() => setHomingDialogOpen(true)}>
              Confirm / home now
            </Button>
          }
        >
          The persisted offset is only meaningful once the stage has been homed
          this session. Home the stage or confirm you already did before
          starting the scan.
        </Alert>
      )}

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

      {/* 2x2 compact layout. Top: live view + joystick + illumination | heatmap.
          Bottom: layout map + scan controls | brightest/known/accept. */}
      <Grid container spacing={2}>
        {/* ============= TOP-LEFT: live view + detector tabs + joystick + illum ======= */}
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 1, height: "100%" }}>
            <Box sx={{ display: "flex", alignItems: "center", mb: 0.5 }}>
              <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
                Live view
              </Typography>
              {(liveViewState?.detectors?.length || 0) > 0 && (
                <Tabs
                  value={Math.min(
                    liveViewState.activeTab || 0,
                    Math.max(0, (liveViewState.detectors?.length || 1) - 1)
                  )}
                  onChange={(_, v) => handleDetectorTabChange(v)}
                  variant="scrollable"
                  scrollButtons="auto"
                  sx={{
                    minHeight: 32,
                    "& .MuiTab-root": { minHeight: 32, py: 0 },
                  }}
                >
                  {liveViewState.detectors.map((d) => (
                    <Tab key={d} label={d} />
                  ))}
                </Tabs>
              )}
            </Box>
            <Box sx={{ maxHeight: 360, overflow: "hidden" }}>
              <LiveViewControlWrapper useFastMode={true} />
            </Box>
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Joystick — nudge stage over the bright spot
              </Typography>
              <JoystickControl hostIP={hostIP} hostPort={hostPort} />
            </Box>
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Illumination
              </Typography>
              <IlluminationController hostIP={hostIP} hostPort={hostPort} />
            </Box>
          </Paper>
        </Grid>

        {/* ============= TOP-RIGHT: heatmap ============== */}
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 1, height: "100%" }}>
            <Typography variant="subtitle2" gutterBottom>
              Heatmap (click = pick, shift/right-click = move)
            </Typography>
            <HeatmapCanvas
              data={heatmap}
              onPick={(x, y) => setPickedTarget({ x, y })}
              onMove={moveStage}
              picked={pickedTarget}
              width={420}
              height={360}
            />
            <Typography variant="caption" color="textSecondary" sx={{ display: "block", mt: 0.5 }}>
              {heatmap?.samples?.length
                ? `${heatmap.samples.length} samples; latest scan auto-restored on reload.`
                : "No data yet - the last saved scan loads automatically."}
            </Typography>
          </Paper>
        </Grid>

        {/* ============= BOTTOM-LEFT: layout map + scan controls ============ */}
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 1, height: "100%" }}>
            <FormControl fullWidth size="small" sx={{ mb: 1 }}>
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
            <LayoutMapMini
              layout={selectedLayout}
              stageX={positionState?.x}
              stageY={positionState?.y}
              knownX={knownX}
              knownY={knownY}
              brightestX={heatmap?.brightest?.x}
              brightestY={heatmap?.brightest?.y}
              scanCenter={{
                x: positionState?.x ?? 0,
                y: positionState?.y ?? 0,
              }}
              maxRadiusUm={Number(maxRadiusUm) || 0}
              stepUm={Number(stepUm) || 0}
              width={420}
              height={200}
              onClickMove={moveStage}
            />
            <Typography variant="caption" color="textSecondary" sx={{ display: "block", mb: 1 }}>
              Red = stage, gold = pinhole, pink = brightest, green dashed = scan area.
            </Typography>

            <Grid container spacing={1} sx={{ mb: 1 }}>
              <Grid item xs={6}>
                <Tooltip title="One tile per full FOV (= min(FOV_x, FOV_y) by default)">
                  <TextField
                    fullWidth
                    size="small"
                    label="Step (um)"
                    type="number"
                    value={stepUm}
                    onChange={(e) => {
                      stepManualRef.current = true;
                      setStepUm(e.target.value);
                    }}
                  />
                </Tooltip>
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

            <Alert severity="info" variant="outlined" sx={{ py: 0.25, mb: 1 }}>
              <Typography variant="caption">
                {tileGeom.nx} × {tileGeom.ny} tiles ({tileGeom.total} samples,
                ~{tileGeom.etaS}s).
                {objectiveStatus && (() => {
                  const f = Math.min(...(objectiveStatus.FOV || []));
                  if (
                    Number.isFinite(f) &&
                    f > 0 &&
                    Math.abs(Number(stepUm) - f) > 1
                  ) {
                    return (
                      <Button
                        size="small"
                        sx={{ ml: 1, py: 0 }}
                        onClick={() => {
                          stepManualRef.current = false;
                          setStepUm(Math.round(f));
                        }}
                      >
                        Reset to FOV ({Math.round(f)} um)
                      </Button>
                    );
                  }
                  return null;
                })()}
              </Typography>
              {objectiveStatus && (
                <Typography
                  variant="caption"
                  color="textSecondary"
                  sx={{ display: "block" }}
                >
                  {objectiveStatus.objectiveName} ({objectiveStatus.magnification}×,
                  NA {objectiveStatus.NA}, {objectiveStatus.pixelsize?.toFixed?.(3)} um/px,
                  FOV {Math.round(objectiveStatus.FOV?.[0] || 0)} × {Math.round(objectiveStatus.FOV?.[1] || 0)} um)
                </Typography>
              )}
            </Alert>

            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
              {!pollingRunning ? (
                <Button variant="contained" size="small" color="primary" onClick={handleStart}>
                  Start scan
                </Button>
              ) : (
                <Button variant="contained" size="small" color="warning" onClick={handleStop}>
                  Stop
                </Button>
              )}
              {pollingRunning && <CircularProgress size={20} />}
              <ButtonGroup size="small" disabled={homingBusy}>
                <Button onClick={() => runHomeAxis("X")}>Home X</Button>
                <Button onClick={() => runHomeAxis("Y")}>Home Y</Button>
                <Button
                  onClick={async () => {
                    await runHomeAxis("X");
                    await runHomeAxis("Y");
                  }}
                >
                  Home X+Y
                </Button>
              </ButtonGroup>
              {homingBusy && <CircularProgress size={20} />}
              <Box sx={{ flexGrow: 1 }} />
              <Typography variant="caption" color="textSecondary">
                X={positionState?.x?.toFixed?.(1) ?? "?"}, Y={positionState?.y?.toFixed?.(1) ?? "?"} um
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* ============= BOTTOM-RIGHT: brightest / known / accept ============ */}
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 1, height: "100%" }}>
            <Typography variant="subtitle2" gutterBottom>
              Detected brightest position
            </Typography>
            <Grid container spacing={1} sx={{ mb: 1 }}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  size="small"
                  label="Brightest X (um)"
                  value={heatmap?.brightest?.x?.toFixed?.(1) ?? ""}
                  InputProps={{ readOnly: true }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  size="small"
                  label="Brightest Y (um)"
                  value={heatmap?.brightest?.y?.toFixed?.(1) ?? ""}
                  InputProps={{ readOnly: true }}
                />
              </Grid>
            </Grid>
            {/* Adopt-current-stage shortcut: after joystick fine-tuning, the
                stage XY is usually more precise than the raster's brightest
                sample. Stamp that XY as the picked target so the offset is
                computed against the fine-tuned position. The "Brightest"
                fields above intentionally stay readonly so the user can
                always see the raw scan result. */}
            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1, alignItems: "center" }}>
              <Button
                size="small"
                variant="outlined"
                disabled={positionState?.x == null || positionState?.y == null}
                onClick={() =>
                  setPickedTarget({
                    x: Number(positionState?.x ?? 0),
                    y: Number(positionState?.y ?? 0),
                  })
                }
              >
                Use current stage position as target
              </Button>
              <Typography variant="caption" color="textSecondary">
                Now at: X={positionState?.x?.toFixed?.(1) ?? "?"},
                Y={positionState?.y?.toFixed?.(1) ?? "?"} um
              </Typography>
            </Box>
            {pickedTarget && (
              <Alert severity="info" sx={{ mb: 1, py: 0.25 }}>
                Override active: ({pickedTarget.x.toFixed(1)},{" "}
                {pickedTarget.y.toFixed(1)}) um
                <Button
                  size="small"
                  onClick={() => setPickedTarget(null)}
                  sx={{ ml: 1 }}
                >
                  use brightest
                </Button>
              </Alert>
            )}

            <Typography variant="subtitle2" gutterBottom>
              Expected (known) calibration point
            </Typography>
            <Grid container spacing={1} sx={{ mb: 1 }}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  size="small"
                  label="Known X (um)"
                  type="number"
                  value={
                    overrideKnownX !== "" ? overrideKnownX : selectedLayout?.x ?? ""
                  }
                  onChange={(e) => setOverrideKnownX(e.target.value)}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  size="small"
                  label="Known Y (um)"
                  type="number"
                  value={
                    overrideKnownY !== "" ? overrideKnownY : selectedLayout?.y ?? ""
                  }
                  onChange={(e) => setOverrideKnownY(e.target.value)}
                />
              </Grid>
            </Grid>
            {targetXY && knownX != null && knownY != null && (
              <Typography variant="caption" color="textSecondary" sx={{ display: "block", mb: 1 }}>
                Pending shift: dX={(knownX - targetXY.x).toFixed(1)} um,
                dY={(knownY - targetXY.y).toFixed(1)} um.
              </Typography>
            )}
            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1 }}>
              <Button
                size="small"
                variant="outlined"
                disabled={!targetXY}
                onClick={() => targetXY && moveStage(targetXY.x, targetXY.y)}
              >
                Move to sample
              </Button>
              <Button
                size="small"
                variant="outlined"
                disabled={knownX == null || knownY == null}
                onClick={() => moveStage(knownX, knownY)}
              >
                Move to known
              </Button>
              <Button
                size="small"
                variant="contained"
                color="success"
                disabled={!targetXY || !homingAck}
                onClick={openConfirm}
              >
                Accept &amp; store offset
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Homing handshake dialog */}
      <Dialog open={homingDialogOpen} onClose={() => setHomingDialogOpen(false)}>
        <DialogTitle>Has the stage been homed this session?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            The persisted stage offset is the difference between the current
            firmware device position and a known physical coordinate. If the
            firmware was rebooted (or steps were lost) since the last homing
            run, the persisted offset will not land you in the same physical
            place. Home now or confirm you already did.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHomingDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={ackHoming}
            color="primary"
            disabled={homingBusy}
          >
            I already homed
          </Button>
          <Button
            onClick={homeAndAck}
            variant="contained"
            color="primary"
            disabled={homingBusy}
          >
            Home X+Y now
          </Button>
        </DialogActions>
      </Dialog>

      {/* Accept confirmation dialog */}
      <Dialog open={confirmOpen} onClose={() => !confirmBusy && setConfirmOpen(false)}>
        <DialogTitle>Overwrite stored stage offset?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            About to record the current device position as the known
            coordinate ({knownX != null ? knownX.toFixed(1) : "?"},
            {knownY != null ? ` ${knownY.toFixed(1)}` : " ?"}) um. The
            previously persisted offset will be overwritten. Make sure the
            stage is physically over the calibration point - the live view
            should show the hole centred.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)} disabled={confirmBusy}>
            Cancel
          </Button>
          <Button
            onClick={acceptOffset}
            variant="contained"
            color="success"
            disabled={confirmBusy}
          >
            {confirmBusy ? "Storing…" : "Confirm & store"}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};


export default StageOffsetCalibrationTab;
