import React, { useEffect, useRef, useState } from "react";
import {
  Alert, Box, Button, Chip, LinearProgress, Paper, Table, TableBody, TableCell,
  TableHead, TableRow, Typography,
} from "@mui/material";
import { Tune as TuneIcon, Stop as StopIcon } from "@mui/icons-material";
import {
  apiShitScopeStartCalibration,
  apiShitScopeStopCalibration,
  apiShitScopeGetCalibrationStatus,
  apiShitScopeApplyCalibration,
} from "../backendapi/apiShitScopeController.js";

const fmt = (v, digits = 0) => (typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "–");

/**
 * Stage calibration for the ShitScope: the backend moves the stage in small
 * steps and measures every move with the camera (ShitScopeController /
 * model/shitscope_calibration.py). Shows the measured values next to the
 * recommendation; Apply sets them now, Apply & save also writes the setup file.
 */
const ShitScopeCalibrationPanel = ({ disabled }) => {
  const [status, setStatus] = useState({ state: "idle" });
  const [applied, setApplied] = useState(null);
  const [error, setError] = useState(null);
  const running = status.state === "running";
  const runningRef = useRef(false);
  runningRef.current = running;

  useEffect(() => {
    apiShitScopeGetCalibrationStatus().then(setStatus).catch(() => {});
    const id = setInterval(() => {
      if (runningRef.current) apiShitScopeGetCalibrationStatus().then(setStatus).catch(() => {});
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const start = async () => {
    setError(null); setApplied(null);
    try {
      const r = await apiShitScopeStartCalibration();
      if (!r.success) throw new Error(r.error);
      setStatus({ state: "running", phase: "starting", progress: 0, log: [] });
    } catch (err) {
      setError(err.message || "Could not start the calibration");
    }
  };

  const apply = async (persist) => {
    try {
      const r = await apiShitScopeApplyCalibration(persist);
      if (!r.success) throw new Error(r.error);
      setApplied(r);
    } catch (err) {
      setError(err.message || "Could not apply the calibration");
    }
  };

  const res = status.result;
  const axes = res?.axes || {};
  const rec = res?.recommended || {};

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1, flexWrap: "wrap" }}>
        <Typography variant="h6">Stage Calibration</Typography>
        <Button variant="outlined" size="small" startIcon={<TuneIcon />} onClick={start}
                disabled={disabled || running}>
          Calibrate stage (~1 min)
        </Button>
        {running && (
          <Button size="small" color="error" startIcon={<StopIcon />} onClick={() => apiShitScopeStopCalibration()}>
            Stop
          </Button>
        )}
        <Typography variant="caption" color="text.secondary">
          Needs a textured area of the sample. Measures drive threshold, µm per step,
          reversal loss and sled rotation with the camera; nothing changes until you apply.
        </Typography>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
      {status.state === "error" && <Alert severity="error" sx={{ mb: 1 }}>{status.error}</Alert>}
      {running && (
        <Box sx={{ mb: 1 }}>
          <Typography variant="body2" color="text.secondary">{status.phase}</Typography>
          <LinearProgress variant="determinate" value={100 * (status.progress || 0)} sx={{ height: 6, borderRadius: 1 }} />
        </Box>
      )}

      {res && (
        <>
          <Alert severity={res.ok ? "success" : "warning"} sx={{ mb: 1 }}>
            {res.ok ? `Stage OK (${res.moves} moves, ${fmt(res.duration_s)} s).` : "The stage needs attention:"}
            {res.warnings?.map((w) => <div key={w}>• {w}</div>)}
          </Alert>
          <Table size="small" sx={{ mb: 1 }}>
            <TableHead>
              <TableRow>
                <TableCell>Axis</TableCell>
                <TableCell align="right">Moves at</TableCell>
                <TableCell align="right">Threshold drive</TableCell>
                <TableCell align="right">Recommended drive</TableCell>
                <TableCell align="right">µm / µstep</TableCell>
                <TableCell align="right">Step spread</TableCell>
                <TableCell align="right">Reversal loss</TableCell>
                <TableCell align="right">Direction in image</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {["X", "Y"].map((ax) => {
                const a = axes[ax] || {};
                return (
                  <TableRow key={ax}>
                    <TableCell>{ax}</TableCell>
                    {a.moves === false ? (
                      <TableCell colSpan={7}>does not move</TableCell>
                    ) : (
                      <>
                        <TableCell align="right">{fmt(a.probe_power)}</TableCell>
                        <TableCell align="right">{a.threshold_power ?? "above max"}</TableCell>
                        <TableCell align="right"><b>{fmt(a.recommended_power)}</b></TableCell>
                        <TableCell align="right">{fmt(a.um_per_step, 1)}</TableCell>
                        <TableCell align="right">{fmt(a.spread_pct, 1)} %</TableCell>
                        <TableCell align="right">{fmt(a.reversal_loss_um)} µm</TableCell>
                        <TableCell align="right">{fmt(a.image_angle_deg, 1)}°</TableCell>
                      </>
                    )}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
            <Chip size="small" variant="outlined" label={`Sled rotation: ${fmt(res.rotation_deg, 2)}°`}
                  color={Math.abs(res.rotation_deg) > 0.5 ? "warning" : "default"} />
            <Chip size="small" variant="outlined"
                  label={`Return error: ${fmt(Math.hypot(...(res.closure_um || [0, 0])))} µm`} />
            {rec.approachOvershootSteps && (
              <Chip size="small" variant="outlined" label={`Approach overshoot: ${rec.approachOvershootSteps} µsteps`} />
            )}
            <Box sx={{ flex: 1 }} />
            <Button size="small" variant="outlined" disabled={!Object.keys(rec).length || running}
                    onClick={() => apply(false)}>
              Apply
            </Button>
            <Button size="small" variant="contained" disabled={!Object.keys(rec).length || running}
                    onClick={() => apply(true)}>
              Apply &amp; save to setup
            </Button>
          </Box>
          {applied && (
            <Alert severity="success" sx={{ mt: 1 }}>
              Applied{applied.saved ? " and saved" : ""}: {Object.entries(applied.applied).map(([k, v]) => `${k} ${v}`).join(", ")}
            </Alert>
          )}
        </>
      )}
      {!!status.log?.length && (
        <Box component="pre" sx={{ fontSize: 11, maxHeight: 140, overflow: "auto", bgcolor: "action.hover", p: 1, mt: 1, mb: 0 }}>
          {status.log.join("\n")}
        </Box>
      )}
    </Paper>
  );
};

export default ShitScopeCalibrationPanel;
