import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  Chip,
  Stack,
  Divider,
  CircularProgress,
  LinearProgress,
} from "@mui/material";
import HomeIcon from "@mui/icons-material/Home";
import StopIcon from "@mui/icons-material/Stop";
import CancelIcon from "@mui/icons-material/Cancel";
import LocalShippingIcon from "@mui/icons-material/LocalShipping";
import SaveIcon from "@mui/icons-material/Save";
import { useSelector } from "react-redux";
import { getHomingState } from "../../state/slices/HomingSlice";

import apiPositionerControllerStartFrameHoming from "../../backendapi/apiPositionerControllerStartFrameHoming";
import apiPositionerControllerCancelFrameHoming from "../../backendapi/apiPositionerControllerCancelFrameHoming";
import apiPositionerControllerGetFrameHomingState from "../../backendapi/apiPositionerControllerGetFrameHomingState";
import apiPositionerControllerMoveToTransportPosition from "../../backendapi/apiPositionerControllerMoveToTransportPosition";
import apiPositionerControllerGetTransportPosition from "../../backendapi/apiPositionerControllerGetTransportPosition";
import apiPositionerControllerSetTransportPosition from "../../backendapi/apiPositionerControllerSetTransportPosition";
import apiPositionerControllerStopAllAxes from "../../backendapi/apiPositionerControllerStopAllAxes";

const AXES = ["Z", "X", "Y"];

// Map a per-axis homing status to a MUI Chip color.
const axisColor = (status) => {
  switch (status) {
    case "done":
      return "success";
    case "homing":
      return "info";
    case "pending":
      return "warning";
    default:
      return "default";
  }
};

/**
 * FrameHomingTab - Collision-safe global homing + transportation position.
 *
 * - Safe Global Homing: homes Z first, lifts Z out of the bottom, then X and Y,
 *   then restores Z. Cancellable; per-axis progress is streamed from the backend
 *   via the sigHomingState websocket signal (Redux `homing` slice).
 * - Transportation Position: move the stage to a stored A/X/Y/Z pose so the
 *   locking blocks can be bolted on; save the current pose as the transport
 *   position (persisted in the setup JSON); stop all axes.
 */
const FrameHomingTab = () => {
  const homingState = useSelector(getHomingState);

  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [transportPosition, setTransportPosition] = useState(null);
  const [busy, setBusy] = useState(false);

  const refreshTransportPosition = useCallback(async () => {
    try {
      const pos = await apiPositionerControllerGetTransportPosition({});
      setTransportPosition(pos || {});
    } catch (err) {
      console.error("Failed to fetch transport position:", err);
    }
  }, []);

  // Initial fetch: transport position + current homing state (in case a run is
  // already in progress when the tab mounts).
  useEffect(() => {
    refreshTransportPosition();
    (async () => {
      try {
        await apiPositionerControllerGetFrameHomingState({});
      } catch (err) {
        // non-fatal; live state arrives via websocket
      }
    })();
  }, [refreshTransportPosition]);

  const handleStartHoming = async () => {
    try {
      setError("");
      setStatus("Starting safe global homing...");
      await apiPositionerControllerStartFrameHoming({ isBlocking: false });
    } catch (err) {
      setError(`Failed to start homing: ${err.message}`);
    }
  };

  const handleCancelHoming = async () => {
    try {
      setStatus("Cancelling homing...");
      await apiPositionerControllerCancelFrameHoming({});
    } catch (err) {
      setError(`Failed to cancel homing: ${err.message}`);
    }
  };

  const handleMoveToTransport = async () => {
    try {
      setBusy(true);
      setError("");
      setStatus("Moving to transport position...");
      await apiPositionerControllerMoveToTransportPosition({ isBlocking: false });
      setStatus("Move to transport position started");
    } catch (err) {
      setError(`Failed to move to transport position: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleSaveTransport = async () => {
    try {
      setBusy(true);
      setError("");
      setStatus("Saving current position as transport position...");
      const res = await apiPositionerControllerSetTransportPosition({
        useCurrent: true,
      });
      if (res && res.transportPosition) {
        setTransportPosition(res.transportPosition);
      } else {
        await refreshTransportPosition();
      }
      setStatus("Transport position saved to config");
    } catch (err) {
      setError(`Failed to save transport position: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleStopAll = async () => {
    try {
      setStatus("Stopping all axes...");
      await apiPositionerControllerStopAllAxes({});
      setStatus("All axes stopped");
    } catch (err) {
      setError(`Failed to stop all axes: ${err.message}`);
    }
  };

  const homingActive = homingState.active;

  const formatPos = (axis) => {
    if (!transportPosition || transportPosition[axis] === undefined) return "—";
    return Number(transportPosition[axis]).toFixed(1);
  };

  return (
    <Box>
      {status && (
        <Alert severity="info" sx={{ mb: 2 }} onClose={() => setStatus("")}>
          {status}
        </Alert>
      )}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError("")}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Safe Global Homing */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Safe Global Homing
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Homes Z first and lifts it clear of the bottom before homing X and
              Y, so the stage cannot drive the objective into the sample.
            </Typography>

            <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={
                  homingActive ? (
                    <CircularProgress size={18} color="inherit" />
                  ) : (
                    <HomeIcon />
                  )
                }
                onClick={handleStartHoming}
                disabled={homingActive}
              >
                Start Homing
              </Button>
              <Button
                variant="outlined"
                color="error"
                startIcon={<CancelIcon />}
                onClick={handleCancelHoming}
                disabled={!homingActive}
              >
                Cancel
              </Button>
            </Stack>

            {homingActive && <LinearProgress sx={{ mb: 2 }} />}

            <Typography variant="subtitle2" gutterBottom>
              Per-axis progress
            </Typography>
            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
              {AXES.map((axis) => (
                <Chip
                  key={axis}
                  label={`${axis}: ${homingState.axes?.[axis] || "idle"}`}
                  color={axisColor(homingState.axes?.[axis])}
                  variant={
                    homingState.axes?.[axis] === "idle" ? "outlined" : "filled"
                  }
                />
              ))}
            </Stack>

            <Typography variant="caption" color="text.secondary">
              Phase: {homingState.phase}
              {homingState.message ? ` — ${homingState.message}` : ""}
            </Typography>
          </Paper>
        </Grid>

        {/* Transportation Position */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Transportation Position
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Move the stage to the stored pose so the locking blocks can be
              bolted on for transport. Save the current pose to store it in the
              config.
            </Typography>

            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
              {["A", "X", "Y", "Z"].map((axis) => (
                <Chip
                  key={axis}
                  label={`${axis}: ${formatPos(axis)}`}
                  variant="outlined"
                />
              ))}
            </Stack>

            <Divider sx={{ my: 2 }} />

            <Stack spacing={1.5}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<LocalShippingIcon />}
                onClick={handleMoveToTransport}
                disabled={busy || homingActive}
                fullWidth
              >
                Move to Transport Position
              </Button>
              <Button
                variant="outlined"
                color="secondary"
                startIcon={<SaveIcon />}
                onClick={handleSaveTransport}
                disabled={busy || homingActive}
                fullWidth
              >
                Save Current as Transport Position
              </Button>
              <Button
                variant="contained"
                color="error"
                startIcon={<StopIcon />}
                onClick={handleStopAll}
                fullWidth
              >
                Stop All Axes
              </Button>
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FrameHomingTab;
