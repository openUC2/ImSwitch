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
import { useSelector } from "react-redux";
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
import * as positionSlice from "../../state/slices/PositionSlice";
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
// HeatmapCanvas - intensity grid + click-to-pick. We deliberately separate
// "click to move the stage" from "click to pick the calibration target":
// after a scan the user is more likely to want to pick a brighter sample
// than to physically move the stage there. ``onPick`` returns the (x, y)
// the user clicked, ``onMove`` is called if held with shift.
// ----------------------------------------------------------------------
const HeatmapCanvas = ({ data, onPick, onMove, picked, width = 420, height = 420 }) => {
  const canvasRef = useRef(null);
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

    if (picked) {
      const px = ((picked.x - xMin) / xRange) * (w - cellW);
      const py = h - cellH - ((picked.y - yMin) / yRange) * (h - cellH);
      ctx.strokeStyle = "#33ff99";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px + cellW / 2, py + cellH / 2, Math.max(cellW, cellH) + 2, 0, 2 * Math.PI);
      ctx.stroke();
    }

    projectionRef.current = { xMin, xMax, yMin, yMax, w, h, cellW, cellH };
  }, [data, picked]);

  const eventToStage = (event) => {
    const proj = projectionRef.current;
    if (!proj) return null;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cx = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const cy = ((event.clientY - rect.top) / rect.height) * canvas.height;
    const { xMin, xMax, yMin, yMax, w, h, cellW, cellH } = proj;
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
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: "100%", height: "auto", display: "block", cursor: "crosshair" }}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
      />
      <Typography variant="caption" color="textSecondary" sx={{ display: "block", mt: 0.5 }}>
        Click to pick the calibration target. Shift-click or right-click to move the stage there.
      </Typography>
    </Box>
  );
};


// ----------------------------------------------------------------------
// Main tab
// ----------------------------------------------------------------------
const StageOffsetCalibrationTab = () => {
  const positionState = useSelector(positionSlice.getPositionState);

  const [activeStep, setActiveStep] = useState(0);
  const [layouts, setLayouts] = useState(FALLBACK_LAYOUTS);
  const [layoutName, setLayoutName] = useState(FALLBACK_LAYOUTS[0].name);

  // Scan parameters - exposure intentionally removed (managed in detector).
  const [stepUm, setStepUm] = useState(250);
  const [maxRadiusUm, setMaxRadiusUm] = useState(5000);
  const [recommended, setRecommended] = useState(null);

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
      try {
        const latest = await apiStageCenterCalibrationGetLatestHeatmap();
        if (cancelled) return;
        if (latest && latest.samples && latest.samples.length) {
          setHeatmap(latest);
          setActiveStep(1);
        }
      } catch (e) {
        /* no-op */
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
    try {
      await apiStageCenterCalibrationStopCalibration();
    } catch (e) {
      /* ignore */
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
              FOV {Math.round(recommended.fovXUm)} × {Math.round(recommended.fovYUm)} um;
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

          <Box sx={{ mt: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
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
            {homingBusy && <CircularProgress size={20} sx={{ alignSelf: "center" }} />}
            <Box sx={{ flexGrow: 1 }} />
            <Typography variant="caption" color="textSecondary" sx={{ alignSelf: "center" }}>
              Stage at: X={positionState?.x?.toFixed?.(1) ?? "?"} um,
              Y={positionState?.y?.toFixed?.(1) ?? "?"} um
            </Typography>
          </Box>
          <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: "block" }}>
            Requires a valid pixel calibration. The backend will refuse to
            start otherwise. The stage parks at the brightest sample at the
            end of the scan.
          </Typography>
        </Grid>

        {/* Heatmap + accept controls */}
        <Grid item xs={12}>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1" gutterBottom>
            Heatmap &amp; result
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <HeatmapCanvas
                data={heatmap}
                onPick={(x, y) => setPickedTarget({ x, y })}
                onMove={moveStage}
                picked={pickedTarget}
              />
              <Typography variant="caption" color="textSecondary">
                {heatmap?.samples?.length
                  ? `${heatmap.samples.length} samples; the latest scan is auto-restored on reload.`
                  : "No data yet - run a scan to populate (the last saved scan loads automatically)."}
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
              {pickedTarget && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  Manual pick from heatmap will be used:
                  ({pickedTarget.x.toFixed(1)}, {pickedTarget.y.toFixed(1)}) um.
                  <Button
                    size="small"
                    onClick={() => setPickedTarget(null)}
                    sx={{ ml: 1 }}
                  >
                    use brightest instead
                  </Button>
                </Alert>
              )}
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
              {targetXY && knownX != null && knownY != null && (
                <Typography variant="caption" color="textSecondary" sx={{ display: "block", mb: 2 }}>
                  Pending: stage report will shift by
                  dX={(knownX - targetXY.x).toFixed(1)} um,
                  dY={(knownY - targetXY.y).toFixed(1)} um after acceptance.
                </Typography>
              )}
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                <Button
                  variant="outlined"
                  disabled={!targetXY}
                  onClick={() => targetXY && moveStage(targetXY.x, targetXY.y)}
                >
                  Move to picked target
                </Button>
                <Button
                  variant="contained"
                  color="success"
                  disabled={!targetXY || !homingAck}
                  onClick={openConfirm}
                >
                  Accept and store stage offset
                </Button>
              </Box>
            </Grid>
          </Grid>
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
