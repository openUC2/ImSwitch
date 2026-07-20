import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  AlertTitle,
  Chip,
  TextField,
  Switch,
  FormControlLabel,
  ToggleButton,
  ToggleButtonGroup,
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
import apiUC2ConfigControllerSetCollisionMode from "../../backendapi/apiUC2ConfigControllerSetCollisionMode";
import apiUC2ConfigControllerCalibrateCollisionReference from "../../backendapi/apiUC2ConfigControllerCalibrateCollisionReference";
import apiUC2ConfigControllerArmCollisionProtection from "../../backendapi/apiUC2ConfigControllerArmCollisionProtection";
import apiUC2ConfigControllerResetCollisionAlarm from "../../backendapi/apiUC2ConfigControllerResetCollisionAlarm";
import apiUC2ConfigControllerConfirmSafeHoming from "../../backendapi/apiUC2ConfigControllerConfirmSafeHoming";

/**
 * CollisionDetectionTab — microscope collision sensor (GPIO CAN slave)
 *
 * Two detection modes on the slave:
 *  - AUTO (default, recommended): adaptive baseline + robust-sigma z-score.
 *    Parameter-free — tracks a slowly drifting background and trips on a fast
 *    deflection. No calibration.
 *  - MANUAL: fixed reference +/- threshold, needs calibrating to the idle level.
 *
 * Collision events are pushed async over CAN and reach the frontend via the
 * sigCollisionStatusUpdate socket signal. A crash cuts bus power and (if armed)
 * stops all motors; recovery requires reset (restores power) + a safe frame
 * homing, tracked by requiresHoming.
 */
const POLL_INTERVAL_MS = 1000;
const MODE_AUTO = 0;
const MODE_MANUAL = 1;

const CollisionDetectionTab = () => {
  const uc2State = useSelector(getUc2State);
  const collisionLatched = uc2State?.collisionLatched ?? false;
  const collisionTrip = uc2State?.collisionTrip ?? false;
  const collisionArmedLive = uc2State?.collisionArmed ?? false;
  const collisionRequiresHoming = uc2State?.collisionRequiresHoming ?? false;
  const collisionEvent = uc2State?.collisionEvent ?? null;

  const [status, setStatus] = useState(null);
  const [pollError, setPollError] = useState(null);
  const [pollingPaused, setPollingPaused] = useState(false);

  const [thresholdInput, setThresholdInput] = useState("");
  const [sensitivityInput, setSensitivityInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const [crashDialogOpen, setCrashDialogOpen] = useState(false);
  const lastLatchedRef = useRef(false);

  useEffect(() => {
    if (collisionLatched && !lastLatchedRef.current) setCrashDialogOpen(true);
    lastLatchedRef.current = collisionLatched;
  }, [collisionLatched]);

  const mode = status?.mode ?? MODE_AUTO;
  const isAuto = mode === MODE_AUTO;

  // ── Poll ─────────────────────────────────────────────────────────────
  const pollStatus = useCallback(async () => {
    try {
      const s = await apiUC2ConfigControllerGetGpioStatus();
      if (s && typeof s === "object" && !s.message) {
        setStatus(s);
        setPollError(null);
        setThresholdInput((p) => (p === "" ? String(s.threshold ?? "") : p));
        setSensitivityInput((p) => (p === "" ? String(s.sensitivity ?? "") : p));
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
      await fn();
      setFeedback({ severity: "success", text: successMsg });
      await pollStatus();
    } catch (error) {
      setFeedback({ severity: "error", text: String(error) });
    } finally {
      setBusy(false);
    }
  };

  const handleModeChange = (_e, newMode) => {
    if (newMode === null) return;
    withBusy(
      () => apiUC2ConfigControllerSetCollisionMode(newMode === MODE_MANUAL ? "manual" : "auto"),
      newMode === MODE_MANUAL
        ? "Switched to MANUAL (fixed reference + threshold)"
        : "Switched to AUTO (adaptive, parameter-free)",
    );
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
        ? "Auto-stop ARMED — a collision cuts bus power and stops all motors"
        : "Auto-stop disarmed",
    );

  const handleResetAlarm = () =>
    withBusy(async () => {
      await apiUC2ConfigControllerResetCollisionAlarm();
      setCrashDialogOpen(false);
    }, "Collision alarm reset — bus power restored. Run safe frame homing next.");

  const handleConfirmHoming = () =>
    withBusy(
      () => apiUC2ConfigControllerConfirmSafeHoming(),
      "Safe homing confirmed — collision recovery complete",
    );

  // ── Derived display ──────────────────────────────────────────────────
  const mean = status?.mean;
  const baseline = status?.baseline ?? status?.mean;
  const sigma = status?.sigma;
  const deviation = status?.deviation;
  const reference = status?.reference;
  const threshold = status?.threshold;
  const tripped = collisionTrip || !!status?.trip;
  // AUTO trip band ≈ max(K·sigma, floor); K=10, floor=40 in firmware.
  const autoBand = sigma !== undefined ? Math.max(10 * sigma, 40) : undefined;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Collision Detection
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Resistive collision sensor on the CAN GPIO node. In Automatic mode the
        detector tracks the (slowly drifting) idle level itself and trips on a
        fast deflection — no calibration required.
      </Typography>

      {/* ── Post-crash recovery ────────────────────────────────────── */}
      {collisionRequiresHoming && (
        <Alert
          severity="warning"
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={handleConfirmHoming} disabled={busy}>
              HOMING DONE
            </Button>
          }
        >
          <AlertTitle>Safe homing required</AlertTitle>
          A collision invalidated the stage position. Run{" "}
          <b>Frame Homing &amp; Transport</b>, then confirm here to complete recovery.
        </Alert>
      )}

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
          . {collisionArmedLive ? "Bus power cut, motors stopped." : "Auto-stop was NOT armed."}{" "}
          Inspect the stage, clear the obstruction, then reset (restores power).
        </Alert>
      ) : (
        <Alert
          severity={tripped ? "warning" : "success"}
          icon={tripped ? <WarningAmberIcon /> : <CheckCircleIcon />}
          sx={{ mb: 2 }}
        >
          {tripped ? "Sensor currently out of band" : "No collision — sensor within idle band"}
        </Alert>
      )}

      {/* ── Mode selector ─────────────────────────────────────────── */}
      <Box sx={{ mb: 2 }}>
        <ToggleButtonGroup
          value={mode}
          exclusive
          onChange={handleModeChange}
          size="small"
          disabled={busy}
        >
          <ToggleButton value={MODE_AUTO}>Automatic (recommended)</ToggleButton>
          <ToggleButton value={MODE_MANUAL}>Manual</ToggleButton>
        </ToggleButtonGroup>
      </Box>

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
                  <Typography variant="body2">
                    Filtered: <b>{status.filtered}</b>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Tooltip title={isAuto ? "Adaptive baseline (tracks slow drift)" : "Rolling mean — calibration source"}>
                    <Typography variant="body2">
                      {isAuto ? "Baseline" : "Mean"}: <b>{isAuto ? baseline : mean}</b>
                    </Typography>
                  </Tooltip>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    Deviation:{" "}
                    <Chip
                      size="small"
                      label={
                        deviation !== undefined
                          ? `${deviation} / ${isAuto ? (autoBand ?? "?") : threshold}`
                          : "-"
                      }
                      color={
                        deviation !== undefined
                          ? deviation > (isAuto ? (autoBand ?? Infinity) : threshold)
                            ? "error"
                            : "success"
                          : "default"
                      }
                    />
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  {isAuto ? (
                    <Tooltip title="Robust noise scale (sigma). Trip band = max(10·sigma, 40).">
                      <Typography variant="body2">
                        Noise σ: <b>{sigma}</b>
                      </Typography>
                    </Tooltip>
                  ) : (
                    <Typography variant="body2">
                      Reference: <b>{reference}</b>
                    </Typography>
                  )}
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
            {!isAuto && (
              <Button
                variant="contained"
                sx={{ mt: 2 }}
                onClick={handleCalibrate}
                disabled={busy || collisionLatched}
                fullWidth
              >
                Calibrate — use current mean ({mean ?? "…"}) as reference
              </Button>
            )}
            {isAuto && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: "block" }}>
                Automatic mode needs no calibration — the baseline and noise
                scale adapt continuously.
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* ── Settings ──────────────────────────────────────────────── */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              {isAuto ? "Protection" : "Detection settings"}
            </Typography>

            {!isAuto && (
              <Grid container spacing={1} alignItems="center" sx={{ mb: 1 }}>
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
                  <Button variant="outlined" onClick={handleApplyThreshold} disabled={busy || thresholdInput === ""} fullWidth>
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
                  <Button variant="outlined" onClick={handleApplySensitivity} disabled={busy || sensitivityInput === ""} fullWidth>
                    Apply
                  </Button>
                </Grid>
              </Grid>
            )}

            <FormControlLabel
              control={
                <Switch
                  checked={collisionArmedLive}
                  onChange={handleArmToggle}
                  color="error"
                  disabled={busy}
                />
              }
              label="Arm auto-stop: a collision cuts bus power and stops ALL motors"
            />

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
            {collisionEvent?.filtered !== undefined ? ` (sensor value ${collisionEvent.filtered})` : ""}.{" "}
            {collisionArmedLive
              ? "Bus power was cut and all motors stopped."
              : "Auto-stop was not armed — check the stage immediately."}
            <br />
            <br />
            Recovery: inspect the microscope and remove the obstruction, then
            reset the alarm (this restores bus power). Afterwards run a safe{" "}
            <b>Frame Homing &amp; Transport</b> — motor positions are no longer
            trustworthy after a crash.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCrashDialogOpen(false)}>Close</Button>
          <Button variant="contained" color="error" onClick={handleResetAlarm} disabled={busy}>
            Situation cleared — reset &amp; restore power
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CollisionDetectionTab;
