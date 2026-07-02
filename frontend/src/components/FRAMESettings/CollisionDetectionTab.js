import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  Chip,
  TextField,
  Switch,
  FormControlLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Tooltip,
  CircularProgress,
} from "@mui/material";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import { useSelector } from "react-redux";
import { getUc2State } from "../../state/slices/UC2Slice";

import apiUC2ConfigControllerGetGpioStatus from "../../backendapi/apiUC2ConfigControllerGetGpioStatus";
import apiUC2ConfigControllerSetCollisionThreshold from "../../backendapi/apiUC2ConfigControllerSetCollisionThreshold";
import apiUC2ConfigControllerSetCollisionSensitivity from "../../backendapi/apiUC2ConfigControllerSetCollisionSensitivity";
import apiUC2ConfigControllerCalibrateCollisionReference from "../../backendapi/apiUC2ConfigControllerCalibrateCollisionReference";
import apiUC2ConfigControllerArmCollisionProtection from "../../backendapi/apiUC2ConfigControllerArmCollisionProtection";
import apiUC2ConfigControllerResetCollisionAlarm from "../../backendapi/apiUC2ConfigControllerResetCollisionAlarm";

/**
 * CollisionDetectionTab — microscope collision sensor (GPIO CAN slave)
 *
 * The sensor idles at a reference ADC value; a collision deviates from it
 * (rising OR falling). The slave confirms a collision over N consecutive
 * samples (spike rejection) and pushes ONE event over CAN; the backend
 * forwards it here via the sigCollisionStatusUpdate socket signal.
 *
 * This tab:
 *  - polls the rolling mean (~1 s) for live display and calibration
 *  - shows a live OK / COLLISION indicator (socket-driven, async)
 *  - lets the user set the current mean as the reference ("Calibrate")
 *  - exposes threshold (deviation band) and sensitivity (sample count)
 *  - arms automatic stop-all-motors on collision
 *  - pops a crash dialog that must be manually reset after inspection
 */
const POLL_INTERVAL_MS = 1000;

const CollisionDetectionTab = () => {
  const uc2State = useSelector(getUc2State);
  const collisionLatched = uc2State?.collisionLatched ?? false;
  const collisionTrip = uc2State?.collisionTrip ?? false;
  const collisionArmedLive = uc2State?.collisionArmed ?? false;
  const collisionEvent = uc2State?.collisionEvent ?? null;

  // Polled hardware status
  const [status, setStatus] = useState(null); // {mean, filtered, raw, reference, ...}
  const [pollError, setPollError] = useState(null);
  const [pollingPaused, setPollingPaused] = useState(false);

  // Editable settings (local input state; committed on button press)
  const [thresholdInput, setThresholdInput] = useState("");
  const [sensitivityInput, setSensitivityInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState(null);

  // Crash popup: shown when a latched collision arrives, dismissed only via
  // "Reset alarm" (or plain close, which re-opens on the next latched event).
  const [crashDialogOpen, setCrashDialogOpen] = useState(false);
  const lastLatchedRef = useRef(false);

  useEffect(() => {
    if (collisionLatched && !lastLatchedRef.current) {
      setCrashDialogOpen(true);
    }
    lastLatchedRef.current = collisionLatched;
  }, [collisionLatched]);

  // ── Poll the rolling mean ────────────────────────────────────────────
  const pollStatus = useCallback(async () => {
    try {
      const s = await apiUC2ConfigControllerGetGpioStatus();
      if (s && typeof s === "object" && !s.status) {
        setStatus(s);
        setPollError(null);
        // Seed the inputs once so the user edits live values
        setThresholdInput((prev) => (prev === "" ? String(s.threshold ?? "") : prev));
        setSensitivityInput((prev) => (prev === "" ? String(s.sensitivity ?? "") : prev));
      } else if (s && s.message) {
        setPollError(s.message);
      }
    } catch (error) {
      setPollError(String(error));
    }
  }, []);

  useEffect(() => {
    if (pollingPaused) return undefined;
    pollStatus();
    const t = setInterval(pollStatus, POLL_INTERVAL_MS);
    return () => clearInterval(t);
  }, [pollStatus, pollingPaused]);

  // ── Actions ──────────────────────────────────────────────────────────
  const withBusy = async (fn, successMsg) => {
    setBusy(true);
    setFeedback(null);
    try {
      const r = await fn();
      setFeedback({ severity: "success", text: successMsg });
      await pollStatus();
      return r;
    } catch (error) {
      setFeedback({ severity: "error", text: String(error) });
      return null;
    } finally {
      setBusy(false);
    }
  };

  const handleCalibrate = () =>
    withBusy(
      () => apiUC2ConfigControllerCalibrateCollisionReference(),
      "Reference calibrated to current mean",
    );

  const handleApplyThreshold = () =>
    withBusy(
      () => apiUC2ConfigControllerSetCollisionThreshold(parseInt(thresholdInput, 10)),
      `Threshold set to ${thresholdInput} counts`,
    );

  const handleApplySensitivity = () =>
    withBusy(
      () => apiUC2ConfigControllerSetCollisionSensitivity(parseInt(sensitivityInput, 10)),
      `Sensitivity set to ${sensitivityInput} samples`,
    );

  const handleArmToggle = (event) =>
    withBusy(
      () => apiUC2ConfigControllerArmCollisionProtection(event.target.checked),
      event.target.checked
        ? "Auto-stop ARMED — motors stop immediately on collision"
        : "Auto-stop disarmed",
    );

  const handleResetAlarm = () =>
    withBusy(async () => {
      const r = await apiUC2ConfigControllerResetCollisionAlarm();
      setCrashDialogOpen(false);
      return r;
    }, "Collision alarm reset");

  // ── Derived display values ───────────────────────────────────────────
  const mean = status?.mean;
  const reference = status?.reference;
  const threshold = status?.threshold;
  const deviation =
    mean !== undefined && reference !== undefined
      ? Math.abs(mean - reference)
      : null;
  const tripped = collisionTrip || !!status?.trip;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Collision Detection
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Resistive collision sensor on the CAN GPIO node. Calibrate the idle
        reference while the microscope is collision-free, then set how far
        (threshold) and for how many consecutive samples (sensitivity) the
        signal must deviate before a collision event fires.
      </Typography>

      {/* ── Status banner ─────────────────────────────────────────── */}
      {collisionLatched ? (
        <Alert
          severity="error"
          icon={<WarningAmberIcon />}
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={handleResetAlarm} disabled={busy}>
              RESET ALARM
            </Button>
          }
        >
          Collision detected{collisionEvent?.filtered !== undefined
            ? ` (sensor=${collisionEvent.filtered})`
            : ""}
          . Motors {collisionArmedLive ? "were stopped" : "were NOT auto-stopped (not armed)"}.
          Inspect the stage, clear the obstruction, then reset the alarm.
          Consider re-homing — positions may be lost after a crash.
        </Alert>
      ) : (
        <Alert
          severity={tripped ? "warning" : "success"}
          icon={tripped ? <WarningAmberIcon /> : <CheckCircleIcon />}
          sx={{ mb: 2 }}
        >
          {tripped
            ? "Sensor currently out of band (not latched)"
            : "No collision — sensor within idle band"}
        </Alert>
      )}

      <Grid container spacing={2}>
        {/* ── Live values ───────────────────────────────────────────── */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Live sensor values
            </Typography>
            {pollError && (
              <Alert severity="warning" sx={{ mb: 1 }}>
                Poll failed: {pollError}
              </Alert>
            )}
            {!status && !pollError && <CircularProgress size={24} />}
            {status && (
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Tooltip title="Slow rolling average — the calibration source. Freezes while a collision is in progress.">
                    <Typography variant="body2">
                      Rolling mean: <b>{mean}</b>
                    </Typography>
                  </Tooltip>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    Filtered: <b>{status.filtered}</b>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    Reference: <b>{reference}</b>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    Deviation:{" "}
                    <Chip
                      size="small"
                      label={deviation !== null ? `${deviation} / ${threshold}` : "-"}
                      color={
                        deviation !== null && threshold !== undefined
                          ? deviation > threshold
                            ? "error"
                            : "success"
                          : "default"
                      }
                    />
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    E-stop: <b>{status.estop ? "PRESSED" : "released"}</b>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        size="small"
                        checked={pollingPaused}
                        onChange={(e) => setPollingPaused(e.target.checked)}
                      />
                    }
                    label="Pause polling"
                  />
                </Grid>
              </Grid>
            )}
            <Button
              variant="contained"
              sx={{ mt: 2 }}
              onClick={handleCalibrate}
              disabled={busy || collisionLatched}
              fullWidth
            >
              Calibrate — use current mean ({mean ?? "…"}) as reference
            </Button>
          </Paper>
        </Grid>

        {/* ── Settings ──────────────────────────────────────────────── */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Detection settings
            </Typography>
            <Grid container spacing={1} alignItems="center">
              <Grid item xs={8}>
                <TextField
                  label="Threshold (ADC counts)"
                  type="number"
                  size="small"
                  fullWidth
                  value={thresholdInput}
                  onChange={(e) => setThresholdInput(e.target.value)}
                  helperText="Allowed deviation from the reference"
                />
              </Grid>
              <Grid item xs={4}>
                <Button
                  variant="outlined"
                  onClick={handleApplyThreshold}
                  disabled={busy || thresholdInput === ""}
                  fullWidth
                >
                  Apply
                </Button>
              </Grid>
              <Grid item xs={8}>
                <TextField
                  label="Sensitivity (samples)"
                  type="number"
                  size="small"
                  fullWidth
                  value={sensitivityInput}
                  onChange={(e) => setSensitivityInput(e.target.value)}
                  helperText="Consecutive samples @50 Hz to confirm (spike rejection)"
                />
              </Grid>
              <Grid item xs={4}>
                <Button
                  variant="outlined"
                  onClick={handleApplySensitivity}
                  disabled={busy || sensitivityInput === ""}
                  fullWidth
                >
                  Apply
                </Button>
              </Grid>
            </Grid>

            <Box sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={collisionArmedLive}
                    onChange={handleArmToggle}
                    color="error"
                    disabled={busy}
                  />
                }
                label="Arm auto-stop: stop ALL motors immediately on collision"
              />
            </Box>

            {feedback && (
              <Alert severity={feedback.severity} sx={{ mt: 1 }}>
                {feedback.text}
              </Alert>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* ── Crash popup ─────────────────────────────────────────────── */}
      <Dialog open={crashDialogOpen} onClose={() => setCrashDialogOpen(false)}>
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <WarningAmberIcon color="error" /> Crash detected
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            The collision sensor detected a crash
            {collisionEvent?.filtered !== undefined
              ? ` (sensor value ${collisionEvent.filtered})`
              : ""}
            .{" "}
            {collisionArmedLive
              ? "All motors have been stopped."
              : "Auto-stop was not armed — check the stage immediately."}
            <br />
            <br />
            Please inspect the microscope, remove the obstruction and verify
            the stage can move freely. Reset the alarm only once the
            situation is cleared. Re-homing is recommended, since motor
            positions may no longer be trustworthy.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCrashDialogOpen(false)}>Close</Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleResetAlarm}
            disabled={busy}
          >
            Situation cleared — reset alarm
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CollisionDetectionTab;
