import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  Stack,
  Divider,
  TextField,
  CircularProgress,
  LinearProgress,
  Stepper,
  Step,
  StepLabel,
} from "@mui/material";
import SyncIcon from "@mui/icons-material/Sync";
import StopIcon from "@mui/icons-material/Stop";

import apiPositionerControllerStartZStageSync from "../../backendapi/apiPositionerControllerStartZStageSync";
import apiPositionerControllerCancelZStageSync from "../../backendapi/apiPositionerControllerCancelZStageSync";
import apiPositionerControllerGetZStageSyncState from "../../backendapi/apiPositionerControllerGetZStageSyncState";

// How often to poll the backend for Z-sync progress (no websocket signal).
const POLL_INTERVAL_MS = 2000;

// Ordered list of backend phases mapped to human-readable step labels. Used to
// drive the MUI Stepper so the operator can follow the procedure.
const PHASE_STEPS = [
  { key: "disabling_limit", label: "Disable Z limit switch" },
  { key: "moving_out", label: "Drive Z into the mechanical stop" },
  { key: "moving_back", label: "Move Z back by half" },
  { key: "enabling_limit", label: "Re-enable Z limit switch" },
  { key: "homing", label: "Re-home Z" },
  { key: "restoring", label: "Return to previous Z position" },
];

const phaseToActiveStep = (phase) => {
  if (phase === "starting" || phase === "idle") return 0;
  if (phase === "done") return PHASE_STEPS.length;
  const idx = PHASE_STEPS.findIndex((s) => s.key === phase);
  return idx >= 0 ? idx : 0;
};

/**
 * ZStageSyncTab - Re-synchronise the two Z motors against the mechanical stop.
 *
 * The FRAME's Z stage is pushed up by two stepper motors at once. If one loses
 * steps they drift out of sync. This procedure temporarily disables the Z limit
 * switch, drives Z into the mechanical end (both motors stall together), backs
 * off by half, restores the limit switch, re-homes Z and returns to the previous
 * Z position. Progress is tracked by polling the backend every 2 s.
 */
const ZStageSyncTab = () => {
  const [syncState, setSyncState] = useState({
    active: false,
    phase: "idle",
    message: "",
    steps: 0,
  });
  const [steps, setSteps] = useState(10000);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const stepsInitRef = useRef(false);

  const poll = useCallback(async () => {
    try {
      const s = await apiPositionerControllerGetZStageSyncState({});
      if (s && typeof s === "object") {
        setSyncState((prev) => ({ ...prev, ...s }));
        // Seed the input with the configured default once.
        if (!stepsInitRef.current && s.steps) {
          setSteps(s.steps);
          stepsInitRef.current = true;
        }
      }
    } catch (err) {
      // non-fatal; next tick retries
    }
  }, []);

  // Poll progress periodically while the tab is mounted.
  useEffect(() => {
    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [poll]);

  const active = syncState.active;

  const handleStart = async () => {
    try {
      setError("");
      setStatus("Starting Z-stage synchronisation...");
      await apiPositionerControllerStartZStageSync({
        steps: Number(steps),
      });
      // Reflect the active state immediately, then let polling take over.
      setSyncState((prev) => ({ ...prev, active: true, phase: "starting" }));
      poll();
    } catch (err) {
      setError(`Failed to start Z-stage sync: ${err.message}`);
    }
  };

  const handleStop = async () => {
    try {
      setStatus("Stopping Z-stage sync...");
      await apiPositionerControllerCancelZStageSync({});
      poll();
    } catch (err) {
      setError(`Failed to stop Z-stage sync: ${err.message}`);
    }
  };

  const activeStep = phaseToActiveStep(syncState.phase);

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
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Z-Stage Synchronisation
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              The Z stage is driven by two stepper motors. If one loses steps
              they desync. This drives Z into the mechanical end-stop so both
              motors realign, backs off by half, re-homes Z, then returns to the
              current Z position.
            </Typography>

            <Alert severity="warning" sx={{ mb: 2 }}>
              The Z limit switch is temporarily disabled and Z is driven into the
              mechanical stop. Make sure the stage is clear before starting.
            </Alert>

            <TextField
              label="Travel (µm)"
              type="number"
              value={steps}
              onChange={(e) => setSteps(e.target.value)}
              fullWidth
              size="small"
              disabled={active}
              helperText="How far to drive Z out (e.g. 10000 µm). Z then moves back by half."
              sx={{ mb: 2 }}
            />

            <Stack direction="row" spacing={2}>
              <Button
                variant="contained"
                color="primary"
                startIcon={
                  active ? (
                    <CircularProgress size={18} color="inherit" />
                  ) : (
                    <SyncIcon />
                  )
                }
                onClick={handleStart}
                disabled={active}
              >
                Start Sync
              </Button>
              <Button
                variant="contained"
                color="error"
                startIcon={<StopIcon />}
                onClick={handleStop}
                disabled={!active}
              >
                Stop
              </Button>
            </Stack>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Progress
            </Typography>

            {active && <LinearProgress sx={{ mb: 2 }} />}

            <Stepper activeStep={activeStep} orientation="vertical">
              {PHASE_STEPS.map((step) => (
                <Step key={step.key}>
                  <StepLabel error={syncState.phase === "error"}>
                    {step.label}
                  </StepLabel>
                </Step>
              ))}
            </Stepper>

            <Divider sx={{ my: 2 }} />

            <Typography variant="caption" color="text.secondary">
              Phase: {syncState.phase}
              {syncState.message ? ` — ${syncState.message}` : ""}
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ZStageSyncTab;
