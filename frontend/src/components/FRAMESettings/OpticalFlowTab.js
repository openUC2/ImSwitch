// src/components/FRAMESettings/OpticalFlowTab.js
//
// Optical-flow alignment tab. Moves the stage in a straight line at constant
// slow velocity and plots the angle between the stage axis and the camera
// axis (computed from optical flow between consecutive frames) live. Used to
// rotate the camera until the mean angle is ~0 so stitching is trivial.
import React, { useEffect, useMemo, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  TextField,
  Typography,
} from "@mui/material";
import Plot from "react-plotly.js";

import * as opticalFlowSlice from "../../state/slices/OpticalFlowSlice.js";
import createAxiosInstance from "../../backendapi/createAxiosInstance.js";
import LiveViewControlWrapper from "../../axon/LiveViewControlWrapper";

const STATE_COLORS = {
  idle: "default",
  starting: "info",
  running: "info",
  finished: "success",
  aborted: "warning",
  error: "error",
};

const OpticalFlowTab = () => {
  const dispatch = useDispatch();
  const {
    distanceUm,
    speedUmS,
    axis,
    warmupFrames,
    minDisplacementPx,
    smoothingWindow,
    state,
    isRunning,
    times,
    angles,
    result,
    errorMessage,
  } = useSelector(opticalFlowSlice.getOpticalFlowState);

  // -------------------- one-shot status restore on mount ----------------
  // We deliberately do NOT poll: live updates flow via the WebSocket signals
  // sigUpdateFlowAngle / sigFlowStateChanged / sigFlowResult (wired in
  // WebSocketHandler.js). This mirrors FocusLockController and avoids the
  // periodic plot blink that came from overwriting the arrays on every poll.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const axiosInstance = createAxiosInstance();
        const response = await axiosInstance.get(
          "/OpticalFlowController/getstatusOpticalFlow",
        );
        if (cancelled) return;
        const status = response.data || {};
        if (status.state) {
          dispatch(opticalFlowSlice.setState(status.state));
        }
        if (Array.isArray(status.times) && Array.isArray(status.angles)) {
          dispatch(
            opticalFlowSlice.setLivePlotData({
              times: status.times,
              angles: status.angles,
            }),
          );
        }
        if (status.result) {
          dispatch(opticalFlowSlice.setResult(status.result));
        }
      } catch (error) {
        // Controller may not be loaded -- ignore.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [dispatch]);

  // -------------------------- actions ------------------------------------
  const handleStart = async () => {
    if (isRunning) return;
    dispatch(opticalFlowSlice.setErrorMessage(null));
    dispatch(opticalFlowSlice.clearLivePlotData());
    dispatch(opticalFlowSlice.clearResult());

    try {
      const axiosInstance = createAxiosInstance();
      const response = await axiosInstance.get(
        "/OpticalFlowController/startMeasurement",
        {
          params: {
            distance_um: distanceUm,
            speed_um_s: speedUmS,
            axis,
            warmup_frames: warmupFrames,
            min_displacement_px: minDisplacementPx,
            smoothing_window: smoothingWindow,
          },
        },
      );
      const payload = response.data || {};
      if (payload.status === "started") {
        dispatch(opticalFlowSlice.setState("running"));
      } else if (payload.status === "error") {
        dispatch(opticalFlowSlice.setErrorMessage(payload.message || "Start failed"));
      }
    } catch (error) {
      dispatch(
        opticalFlowSlice.setErrorMessage(
          `Failed to start measurement: ${error.message}`,
        ),
      );
    }
  };

  const handleAbort = async () => {
    try {
      const axiosInstance = createAxiosInstance();
      const response = await axiosInstance.get(
        "/OpticalFlowController/abortMeasurement",
      );
      const payload = response.data || {};
      if (payload.state) {
        dispatch(opticalFlowSlice.setState(payload.state));
      }
    } catch (error) {
      dispatch(
        opticalFlowSlice.setErrorMessage(
          `Failed to abort measurement: ${error.message}`,
        ),
      );
    }
  };

  // ----------------------------- UI --------------------------------------
  const stateColor = STATE_COLORS[state] || "default";
  const hasPlot = times.length > 1 && angles.length === times.length;

  // The optical-flow loop may exit before the stage reaches its target
  // (slow camera, safety timeout). Keep the Abort button enabled whenever
  // the controller isn't in a terminal/idle state so the user can still
  // stop the stage.
  const canAbort = !(
    state === "idle" ||
    state === "finished" ||
    state === "aborted" ||
    state === "error"
  );

  // Bump revision on every sample so Plotly does an in-place uirevision-stable
  // redraw instead of remounting the trace (which is what produced the blink).
  const plotRevision = times.length;

  // Stable layout reference -- only the dashed mean line moves with result.
  const plotLayout = useMemo(
    () => ({
      margin: { t: 20, r: 20, b: 40, l: 60 },
      xaxis: { title: "Time (s)", autorange: true },
      yaxis: { title: "Flow angle (\u00b0)", autorange: true },
      height: 320,
      uirevision: "opticalflow-plot",
      datarevision: plotRevision,
      shapes: result
        ? [
            {
              type: "line",
              xref: "paper",
              x0: 0,
              x1: 1,
              y0: result.meanAngle,
              y1: result.meanAngle,
              line: { color: "#d62728", width: 1, dash: "dash" },
            },
          ]
        : [],
    }),
    [plotRevision, result],
  );

  // Hold the trace object in a ref so we can mutate x/y in place; Plotly will
  // pick up the new datarevision without redrawing the whole figure.
  const traceRef = useRef({
    x: times,
    y: angles,
    type: "scatter",
    mode: "lines+markers",
    marker: { color: "#1f77b4", size: 4 },
    line: { color: "#1f77b4" },
    name: "angle",
  });
  traceRef.current.x = times;
  traceRef.current.y = angles;

  return (
    <Paper sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom>
        Camera &harr; Stage Rotation Alignment
      </Typography>
      <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
        Moves the stage in a straight line at a constant slow velocity and
        reports the angle between the stage axis and the apparent motion seen
        by the camera. Rotate the camera until the mean angle is ~0&deg; so
        stitching does not need to compensate for rotation. Use a feature-rich
        sample for stable tracking and make sure the camera delivers at least
        5 fps at the chosen speed.
      </Typography>

      <Alert severity="info" sx={{ mb: 2 }}>
        Configure your illumination (laser / LED / brightfield) manually before
        starting &mdash; this controller does not toggle any light source. The
        camera must show a high-contrast, feature-rich image throughout the
        sweep for the optical flow to lock on.
      </Alert>

      <Box sx={{ mb: 2, border: "1px solid", borderColor: "divider", borderRadius: 1, overflow: "hidden" }}>
        <Box sx={{ maxHeight: 360, overflow: "hidden" }}>
          <LiveViewControlWrapper useFastMode={true} />
        </Box>
      </Box>

      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={6} md={3}>
          <TextField
            label="Distance (\u00b5m)"
            type="number"
            value={distanceUm}
            onChange={(e) =>
              dispatch(
                opticalFlowSlice.setDistanceUm(parseFloat(e.target.value)),
              )
            }
            inputProps={{ step: 100, min: 10 }}
            fullWidth
            disabled={isRunning}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <TextField
            label="Speed (\u00b5m/s)"
            type="number"
            value={speedUmS}
            onChange={(e) =>
              dispatch(opticalFlowSlice.setSpeedUmS(parseFloat(e.target.value)))
            }
            inputProps={{ step: 10, min: 1 }}
            fullWidth
            disabled={isRunning}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <FormControl fullWidth disabled={isRunning}>
            <InputLabel>Axis</InputLabel>
            <Select
              value={axis}
              label="Axis"
              onChange={(e) => dispatch(opticalFlowSlice.setAxis(e.target.value))}
            >
              <MenuItem value="X">X</MenuItem>
              <MenuItem value="Y">Y</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <TextField
            label="Warmup frames"
            type="number"
            value={warmupFrames}
            onChange={(e) =>
              dispatch(
                opticalFlowSlice.setWarmupFrames(parseInt(e.target.value, 10)),
              )
            }
            inputProps={{ step: 1, min: 0 }}
            fullWidth
            disabled={isRunning}
            helperText="Frames skipped after move start"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <TextField
            label="Min displ. (px)"
            type="number"
            value={minDisplacementPx}
            onChange={(e) =>
              dispatch(
                opticalFlowSlice.setMinDisplacementPx(
                  parseFloat(e.target.value),
                ),
              )
            }
            inputProps={{ step: 0.1, min: 0 }}
            fullWidth
            disabled={isRunning}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <TextField
            label="Smoothing"
            type="number"
            value={smoothingWindow}
            onChange={(e) =>
              dispatch(
                opticalFlowSlice.setSmoothingWindow(
                  parseInt(e.target.value, 10) || 1,
                ),
              )
            }
            inputProps={{ step: 1, min: 1 }}
            fullWidth
            disabled={isRunning}
            helperText="Rolling-mean window (samples)"
          />
        </Grid>
      </Grid>

      <Box sx={{ display: "flex", gap: 1, mb: 2, alignItems: "center" }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleStart}
          disabled={isRunning}
        >
          Start
        </Button>
        <Button
          variant="outlined"
          color="warning"
          onClick={handleAbort}
          disabled={!canAbort}
        >
          Abort
        </Button>
        <Typography variant="body2" color={`${stateColor}.main`} sx={{ ml: 2 }}>
          State: {state}
        </Typography>
      </Box>

      {errorMessage && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => dispatch(opticalFlowSlice.setErrorMessage(null))}>
          {errorMessage}
        </Alert>
      )}

      {result && (
        <Alert severity={state === "error" ? "error" : "success"} sx={{ mb: 2 }}>
          Mean angle: <strong>{result.meanAngle?.toFixed(2)}&deg;</strong>
          &nbsp;&plusmn; {result.std?.toFixed(2)}&deg; ({result.n}/
          {result.nTotal} samples)
        </Alert>
      )}

      <Box sx={{ minHeight: 320 }}>
        {hasPlot ? (
          <Plot
            data={[traceRef.current]}
            layout={plotLayout}
            revision={plotRevision}
            config={{ displayModeBar: false }}
            style={{ width: "100%" }}
            useResizeHandler
          />
        ) : (
          <Typography variant="body2" color="textSecondary">
            Plot will appear here once the first samples arrive.
          </Typography>
        )}
      </Box>
    </Paper>
  );
};

export default OpticalFlowTab;
