// src/components/FrameCalibrationApp.jsx
//
// One-button FRAME calibration (backend: FrameCalibrationController). Put the
// FRAME calibration plate in the holder (well-label side up, A1 notch at the
// stage origin), press Start. The map shows the plate in stage coordinates,
// where the microscope looks right now (red cross) and every position the
// routine visited; the step list says what it is doing and why a step failed.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  Dialog,
  FormControlLabel,
  Grid,
  LinearProgress,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import CameraSettingsDialog from "./CameraSettingsDialog";
import * as positionSlice from "../state/slices/PositionSlice";
import {
  apiFrameCalibrationContinue,
  apiFrameCalibrationGetFigure,
  apiFrameCalibrationGetFigures,
  apiFrameCalibrationGetPlate,
  apiFrameCalibrationGetReportPdf,
  apiFrameCalibrationGetReportZip,
  apiFrameCalibrationGetSettings,
  apiFrameCalibrationGetStatus,
  apiFrameCalibrationSetSettings,
  apiFrameCalibrationStart,
  apiFrameCalibrationStop,
} from "../backendapi/apiFrameCalibrationController";

const STEP_COLOR = { ok: "success", warn: "warning", failed: "error", running: "info", aborted: "error" };
const TRACE_COLOR = { tile: "#5b8def", via: "#ffd54f", edge: "#ff7043", overview: "#b39ddb", axes: "#4db6ac" };
const ACTIVE = ["running", "waiting"];

// Settings shown in the UI (all of them are stored in the setup JSON).
const FIELDS = [
  ["searchRadiusUm", "Search radius (µm)"],
  ["searchRadius2Um", "Search radius 2nd objective (µm)"],
  ["focusSearchUm", "Focus search from home (µm)"],
  ["focusStepUm", "Focus step (µm)"],
  ["viaExposureFactor", "Via exposure (× live exposure)"],
  ["illumination", "Light source"],
  ["illuminationValue", "Light value (0 = keep)"],
  ["overviewIlluminationValue", "Light for overview camera (room light)"],
  ["overviewExposureMs", "Overview exposure at full light (ms)"],
];
const FLAGS = [
  ["maxGain", "Maximum camera gain during the run"],
  ["autoExposure", "Set the camera exposure automatically"],
  ["homeTurret", "Home turret"],
  ["secondObjective", "Second objective"],
  ["placementCheck", "Placement check"],
  ["measureBacklash", "Measure backlash"],
  ["applyBacklash", "Apply backlash"],
  ["manualFocusFallback", "Ask me when autofocus fails"],
];

function PlateMap({ plate, trace, stage, placement }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !plate) return;
    const ctx = c.getContext("2d");
    const margin = 8000;
    const s = Math.min((c.width - 20) / (plate.size[0] + 2 * margin), (c.height - 20) / (plate.size[1] + 2 * margin));
    const X = (x) => 10 + (x + margin) * s;
    const Y = (y) => 10 + (y + margin) * s;
    ctx.fillStyle = "#101820";
    ctx.fillRect(0, 0, c.width, c.height);

    const outline = (pts, color, dash) => {
      ctx.strokeStyle = color;
      ctx.setLineDash(dash);
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    };
    outline(plate.outline, "#78909c", placement ? [6, 4] : []);
    if (placement) {
      // measured pose: rotated about the via (the offset anchor)
      const t = (placement.rotationDeg * Math.PI) / 180;
      const [vx, vy] = plate.via;
      const rot = ([x, y]) => [vx + (x - vx) * Math.cos(t) - (y - vy) * Math.sin(t), vy + (x - vx) * Math.sin(t) + (y - vy) * Math.cos(t)];
      outline(plate.outline.map(rot), Math.abs(placement.rotationDeg) > 0.3 ? "#ff7043" : "#66bb6a", []);
    }
    // landmarks
    ctx.strokeStyle = "#cfd8dc";
    ctx.beginPath();
    ctx.arc(X(plate.roundHole[0]), Y(plate.roundHole[1]), (plate.roundHoleDiameter / 2) * s, 0, 2 * Math.PI);
    ctx.stroke();
    const [rw, rh] = plate.rectHoleSize;
    ctx.strokeRect(X(plate.rectHole[0] - rw / 2), Y(plate.rectHole[1] - rh / 2), rw * s, rh * s);
    ctx.fillStyle = "#ffd54f";
    ctx.beginPath();
    ctx.arc(X(plate.via[0]), Y(plate.via[1]), 3, 0, 2 * Math.PI);
    ctx.fill();
    // stage axes and origin
    ctx.fillStyle = "#90a4ae";
    ctx.font = "11px sans-serif";
    ctx.fillText("0,0", X(0) + 3, Y(0) - 3);
    ctx.fillText("+x →", X(plate.size[0]) - 30, Y(0) - 3);
    ctx.fillText("+y ↓", X(0) - 30, Y(plate.size[1]));
    // visited positions
    (trace || []).forEach((p) => {
      ctx.fillStyle = TRACE_COLOR[p.kind] || "#888";
      ctx.fillRect(X(p.x) - 1.5, Y(p.y) - 1.5, 3, 3);
    });
    // where the objective looks now
    if (stage && Number.isFinite(stage.x) && Number.isFinite(stage.y)) {
      ctx.strokeStyle = "#ff1744";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(X(stage.x) - 8, Y(stage.y));
      ctx.lineTo(X(stage.x) + 8, Y(stage.y));
      ctx.moveTo(X(stage.x), Y(stage.y) - 8);
      ctx.lineTo(X(stage.x), Y(stage.y) + 8);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  }, [plate, trace, stage, placement]);
  return <canvas ref={ref} width={640} height={440} style={{ width: "100%", height: "auto", display: "block" }} />;
}

const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");

// Report figures of the running or last run: the image with what was expected (yellow)
// and what was measured (green). Refetched when the backend adds figures or a run starts.
function ReportFigures({ nFigures, runKey, onError }) {
  const [figures, setFigures] = useState([]);
  const [urls, setUrls] = useState({});
  const [open, setOpen] = useState(null);
  const urlsRef = useRef({});

  useEffect(() => {           // new run: drop the old images
    Object.values(urlsRef.current).forEach((u) => URL.revokeObjectURL(u));
    urlsRef.current = {};
    setUrls({});
    setFigures([]);
  }, [runKey]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await apiFrameCalibrationGetFigures();
        if (cancelled) return;
        setFigures(list || []);
        for (const f of list || []) {
          if (urlsRef.current[f.id]) continue;
          const blob = await apiFrameCalibrationGetFigure(f.id);
          if (cancelled) return;
          urlsRef.current[f.id] = URL.createObjectURL(blob);
          setUrls({ ...urlsRef.current });
        }
      } catch (e) {
        onError(`Report figures not available: ${e.message || e}`);
      }
    })();
    return () => { cancelled = true; };
  }, [nFigures, runKey, onError]);

  if (!figures.length) return <Typography variant="body2" color="text.secondary">No figures yet - they appear while the calibration runs.</Typography>;
  const shown = figures.find((f) => f.id === open);
  return (
    <>
      <Grid container spacing={1}>
        {figures.map((f) => (
          <Grid item xs={12} sm={6} lg={4} key={f.id}>
            <Card variant="outlined" sx={{ height: "100%" }}>
              {urls[f.id] && (
                <Box component="img" src={urls[f.id]} alt={f.title} onClick={() => setOpen(f.id)}
                  sx={{ width: "100%", display: "block", cursor: "zoom-in", bgcolor: "black" }} />
              )}
              <CardContent sx={{ p: 1, "&:last-child": { pb: 1 } }}>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                  {f.ok != null && <Chip size="small" label={f.ok ? "ok" : "failed"} color={f.ok ? "success" : "error"} />}
                  <Typography variant="subtitle2">{f.title}</Typography>
                </Stack>
                {f.expected && <Typography variant="caption" display="block" sx={{ color: "#b58900" }}>Expected: {f.expected}</Typography>}
                {f.measured && <Typography variant="caption" display="block" sx={{ color: "success.main" }}>Measured: {f.measured}</Typography>}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      <Dialog open={!!shown} onClose={() => setOpen(null)} maxWidth="lg">
        {shown && urls[shown.id] && <Box component="img" src={urls[shown.id]} alt={shown.title} sx={{ maxWidth: "100%" }} onClick={() => setOpen(null)} />}
      </Dialog>
    </>
  );
}

function Results({ result }) {
  if (!result) return <Typography variant="body2" color="text.secondary">No calibration stored yet.</Typography>;
  const rows = [];
  Object.entries(result.objectives || {}).forEach(([slot, o]) => {
    const m = o.metrics || {};
    rows.push([`Objective ${slot}`, `${fmt(m.scale_x_um_per_pixel, 4)} × ${fmt(m.scale_y_um_per_pixel, 4)} µm/px, rotation ${fmt(m.rotation_deg)}°, shear ${fmt(m.shear_deg)}°, residual ${fmt(o.rmsPx)} px, focus Z ${fmt(o.z, 0)} µm`]);
  });
  if (result.offset) rows.push(["Stage offset", `(${fmt(result.offset.offset[0], 0)}, ${fmt(result.offset.offset[1], 0)}) µm`]);
  if (result.placement) rows.push(["Plate placement", `rotation ${fmt(result.placement.rotationDeg)}°, rect hole ${fmt(result.placement.rectSize?.[0], 0)} × ${fmt(result.placement.rectSize?.[1], 0)} µm`]);
  if (result.backlash) rows.push(["Backlash", `X ${fmt(result.backlash.X, 1)} µm, Y ${fmt(result.backlash.Y, 1)} µm`]);
  if (result.xyOffsets) rows.push(["Parcentricity", result.xyOffsets.map((o, i) => `slot ${i}: (${fmt(o[0], 1)}, ${fmt(o[1], 1)})`).join(", ") + " µm"]);
  if (result.overview) rows.push(["Overview camera", `homography from ${result.overview.nPoints} points, residual ${fmt(result.overview.rmsPx)} px`]);
  if (Number.isFinite(result.verifyUm)) rows.push(["Re-find via", `${fmt(result.verifyUm, 1)} µm from image centre`]);
  return (
    <Table size="small">
      <TableBody>
        <TableRow><TableCell colSpan={2}><Typography variant="caption">{result.timestamp} · {result.detector}</Typography></TableCell></TableRow>
        {rows.map(([k, v]) => (
          <TableRow key={k}><TableCell sx={{ whiteSpace: "nowrap" }}>{k}</TableCell><TableCell>{v}</TableCell></TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function FrameCalibrationApp() {
  const position = useSelector(positionSlice.getPositionState);
  const [plate, setPlate] = useState(null);
  const [status, setStatus] = useState(null);
  const [settings, setSettings] = useState(null);
  const [error, setError] = useState("");
  const [cameraOpen, setCameraOpen] = useState(false);   // exposure/gain: the preflight needs a lit frame

  const refresh = useCallback(async () => {
    try {
      setStatus(await apiFrameCalibrationGetStatus());
    } catch (e) {
      setError(`Backend not reachable: ${e.message || e}`);
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setPlate(await apiFrameCalibrationGetPlate());
        setSettings(await apiFrameCalibrationGetSettings());
      } catch (e) {
        setError(`FrameCalibrationController not available: ${e.message || e}`);
      }
      refresh();
    })();
  }, [refresh]);

  const active = ACTIVE.includes(status?.state);
  useEffect(() => {
    if (!active) return undefined;
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  }, [active, refresh]);

  const saveSetting = async (key, value) => {
    setSettings((s) => ({ ...s, [key]: value }));
    try {
      setSettings(await apiFrameCalibrationSetSettings({ [key]: value }));
    } catch (e) {
      setError(`Could not save setting: ${e.message || e}`);
    }
  };

  const download = async (fetcher, name) => {
    try {
      const blob = await fetcher();
      if (blob.type && blob.type.includes("json")) {      // the backend answered with an error
        setError(JSON.parse(await blob.text()).error || "Download failed");
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (e) {
      setError(`Download failed: ${e.message || e}`);
    }
  };
  const runName = `frame_calibration_${(status?.startedAt || "").replace(/[^0-9]/g, "")}`;
  const onFigureError = useCallback((msg) => setError(msg), []);

  const start = async () => {
    setError("");
    const r = await apiFrameCalibrationStart();
    if (r && r.success === false) setError(r.error);
    refresh();
  };

  const steps = status?.steps || [];
  const done = steps.filter((s) => !["pending", "running"].includes(s.status)).length;
  const preflightFailed = steps.find((s) => s.name === "preflight")?.status === "failed";
  const cameraButton = (
    <Button color="inherit" size="small" onClick={() => setCameraOpen(true)}>Camera settings</Button>
  );

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>FRAME Calibration</Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Put the FRAME calibration plate in the holder — well-label side up, A1 notch in the stage-origin corner — and press
        Start. The microscope homes, finds the plate, calibrates pixel size, rotation and shear of the camera for each
        objective, the stage offset, the offset between the objectives and the overview camera, and stores everything in the
        setup file.
      </Typography>
      {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError("")}>{error}</Alert>}
      {status && ["finished", "failed", "aborted"].includes(status.state) && (
        <Alert severity={status.state === "finished" ? (/warnings|failed/.test(status.message) ? "warning" : "success") : "error"} sx={{ mb: 2 }}
          action={preflightFailed ? cameraButton : undefined}>
          {status.message}{status.reportDir ? ` — report: ${status.reportDir}` : ""}
        </Alert>
      )}

      <Stack direction="row" spacing={1} sx={{ mb: 2 }} alignItems="center">
        <Button variant="contained" onClick={start} disabled={active || !plate}>Start calibration</Button>
        <Button variant="outlined" color="error" onClick={() => apiFrameCalibrationStop().then(refresh)} disabled={!active}>Abort</Button>
        <Button variant="outlined" onClick={() => setCameraOpen(true)} disabled={active}>Camera settings</Button>
        <Typography variant="body2" color="text.secondary">
          Stage X {fmt(position?.x, 0)} · Y {fmt(position?.y, 0)} · Z {fmt(position?.z, 0)} µm
        </Typography>
      </Stack>
      {active && <LinearProgress variant="determinate" value={(100 * done) / Math.max(steps.length, 1)} sx={{ mb: 2 }} />}
      <CameraSettingsDialog open={cameraOpen} onClose={() => setCameraOpen(false)} />

      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 1, mb: 2 }}>
            <Typography variant="subtitle2">Plate map (stage = plate coordinates, µm)</Typography>
            <PlateMap plate={plate} trace={status?.trace} stage={position} placement={status?.result?.placement} />
            <Typography variant="caption" color="text.secondary">
              Red cross: where the objective looks now. Dots: blue search tiles, yellow via found, orange hole edges,
              purple overview-camera positions. Orange/green outline: measured plate pose.
            </Typography>
          </Paper>
          <Paper variant="outlined" sx={{ p: 1 }}>
            <Typography variant="subtitle2">Camera</Typography>
            {status?.camera && (
              <Typography variant="caption" color={active ? "info.main" : "text.secondary"} display="block">
                {active ? "Set by the calibration" : "Last run"}: exposure {fmt(status.camera.exposureMs, 2)} ms, gain{" "}
                {fmt(status.camera.gainDb, 1)} dB - {status.camera.note}
              </Typography>
            )}
            <Box sx={{ maxHeight: 360, overflow: "hidden" }}>
              <LiveViewControlWrapper useFastMode={true} />
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 1, mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Steps</Typography>
            {steps.map((s) => (
              <Box key={s.name} sx={{ mb: 1 }}>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Chip size="small" label={s.status === "running" && status.state === "waiting" ? "waiting for you" : s.status}
                    color={s.status === "running" && status.state === "waiting" ? "warning" : STEP_COLOR[s.status] || "default"}
                    variant={s.status === "pending" ? "outlined" : "filled"} />
                  <Typography variant="body2">{s.title}</Typography>
                  {s.seconds != null && <Typography variant="caption" color="text.secondary">{s.seconds} s</Typography>}
                </Stack>
                {s.detail && <Typography variant="caption" sx={{ display: "block", ml: 1 }} color={s.status === "failed" ? "error" : "text.secondary"}>{s.detail}</Typography>}
                {(s.warnings || []).map((w) => <Typography key={w} variant="caption" sx={{ display: "block", ml: 1 }} color="warning.main">⚠ {w}</Typography>)}
                {s.status === "running" && status.state === "waiting" && (
                  <Alert severity="warning" sx={{ mt: 1 }} action={<Button color="inherit" variant="outlined" onClick={() => apiFrameCalibrationContinue().then(refresh)}>Continue</Button>}>
                    {status.message}
                  </Alert>
                )}
              </Box>
            ))}
          </Paper>
          <Paper variant="outlined" sx={{ p: 1, mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Stored calibration</Typography>
            <Results result={status?.result} />
          </Paper>
          {settings && (
            <Paper variant="outlined" sx={{ p: 1 }}>
              <Typography variant="subtitle2" gutterBottom>Settings (saved in the setup file)</Typography>
              <Grid container spacing={1}>
                {FIELDS.map(([k, label]) => (
                  <Grid item xs={6} key={k}>
                    <TextField
                      size="small"
                      fullWidth
                      label={label}
                      defaultValue={settings[k]}
                      disabled={active}
                      onBlur={(e) => {
                        const v = typeof settings[k] === "number" ? Number(e.target.value) : e.target.value;
                        if (v !== settings[k]) saveSetting(k, v);
                      }}
                    />
                  </Grid>
                ))}
                {FLAGS.map(([k, label]) => (
                  <Grid item xs={6} key={k}>
                    <FormControlLabel
                      control={<Checkbox size="small" checked={!!settings[k]} disabled={active} onChange={(e) => saveSetting(k, e.target.checked)} />}
                      label={<Typography variant="body2">{label}</Typography>}
                    />
                  </Grid>
                ))}
              </Grid>
            </Paper>
          )}
        </Grid>
      </Grid>

      <Paper variant="outlined" sx={{ p: 1, mt: 2 }}>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }} flexWrap="wrap">
          <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>Report: what was expected (yellow), what was measured (green)</Typography>
          <Button size="small" variant="outlined" onClick={() => download(apiFrameCalibrationGetReportPdf, `${runName}.pdf`)}>
            Download PDF
          </Button>
          <Button size="small" variant="outlined" disabled={active} onClick={() => download(apiFrameCalibrationGetReportZip, `${runName}.zip`)}>
            Download raw data (ZIP)
          </Button>
        </Stack>
        <ReportFigures nFigures={status?.nFigures} runKey={status?.startedAt} onError={onFigureError} />
      </Paper>
    </Box>
  );
}
