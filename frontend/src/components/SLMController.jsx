// SLM / DMD app: an external display conjugate to the sample (SLMController backend).
// Tabs: Patterns (built-ins, images), Draw (light where you paint on the camera image),
// Calibrate (camera <-> display affine, automatic or by clicking spots) and Sequence
// (upload, play, acquisition synchronised with the camera).
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSelector } from "react-redux";
import {
  Alert, Box, Button, Chip, Divider, FormControl, FormControlLabel, Grid, IconButton,
  InputLabel, LinearProgress, MenuItem, Paper, Select, Slider, Stack, Switch, Tab, Tabs,
  TextField, ToggleButton, ToggleButtonGroup, Tooltip, Typography,
} from "@mui/material";
import BrushIcon from "@mui/icons-material/Brush";
import CropSquareIcon from "@mui/icons-material/CropSquare";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import AutoFixNormalIcon from "@mui/icons-material/AutoFixNormal";
import UndoIcon from "@mui/icons-material/Undo";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import LightbulbIcon from "@mui/icons-material/Lightbulb";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import DeleteIcon from "@mui/icons-material/Delete";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice";
import apiSettingsControllerGetDetectorNames from "../backendapi/apiSettingsControllerGetDetectorNames";
import * as slm from "../backendapi/apiSLMController";

const errText = (e) => e?.response?.data?.detail || e?.message || String(e);

/** Stream pixels -> sensor pixels factor (the live stream may be subsampled). */
const useSubsampling = () => {
  const liveStream = useSelector(liveStreamSlice.getLiveStreamState);
  return useMemo(() => {
    const fmt = liveStream.streamSettings?.current_compression_algorithm || "jpeg";
    if (fmt === "binary") return liveStream.streamSettings?.binary?.subsampling?.factor || 1;
    if (fmt === "webrtc") return liveStream.streamSettings?.webrtc?.subsampling_factor || 1;
    return 1;
  }, [liveStream.streamSettings]);
};

const useDetectorNames = () => {
  const [names, setNames] = useState([]);
  useEffect(() => {
    apiSettingsControllerGetDetectorNames().then((d) => setNames(Array.isArray(d) ? d : [])).catch(() => {});
  }, []);
  return names;
};

/** Pointer position -> image pixels for a viewer fitted ("contain") into its container. */
const pointerToImage = (container, imageSize, clientX, clientY) => {
  if (!container || !imageSize.width || !imageSize.height) return null;
  const rect = container.getBoundingClientRect();
  const fit = Math.min(rect.width / imageSize.width, rect.height / imageSize.height);
  const ox = rect.left + (rect.width - imageSize.width * fit) / 2;
  const oy = rect.top + (rect.height - imageSize.height * fit) / 2;
  return { x: (clientX - ox) / fit, y: (clientY - oy) / fit };
};

const Overlay = ({ imageSize, children }) => (
  <svg
    style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    viewBox={`0 0 ${imageSize.width || 1} ${imageSize.height || 1}`}
    preserveAspectRatio="xMidYMid meet"
  >
    {children}
  </svg>
);

// ───────────────────────────── display header ─────────────────────────────

const DisplayCard = ({ status, act, previewNonce }) => {
  const [settle, setSettle] = useState("");
  const [displays, setDisplays] = useState(null);
  const previewUrl = useMemo(() => slm.slmPreviewUrl(384), [previewNonce]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (status && settle === "") setSettle(String(status.settleTimeMs));
  }, [status, settle]);
  if (!status) return <LinearProgress />;
  return (
    <Paper sx={{ p: 1.5, mb: 1 }}>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12} md={7}>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
            <Typography variant="h6">{status.name}</Typography>
            <Chip size="small" label={`${status.backend}[${status.displayIndex}]`} />
            <Chip size="small" label={`${status.width} × ${status.height} px`} />
            <Chip size="small" color={status.error ? "warning" : "success"}
              label={status.error ? "virtual fallback" : "output open"} />
            <Chip size="small" color={status.calibrated ? "success" : "default"}
              label={status.calibrated ? `calibrated, ${status.calibration?.residual_px?.toFixed?.(2) ?? "?"} px rms` : "not calibrated"} />
          </Stack>
          {status.error && <Alert severity="warning" sx={{ mt: 1 }}>{status.error}</Alert>}
          <Stack direction="row" spacing={1} sx={{ mt: 1.5 }} alignItems="center" flexWrap="wrap" useFlexGap>
            <TextField size="small" label="Settle time (ms)" type="number" value={settle} sx={{ width: 150 }}
              onChange={(e) => setSettle(e.target.value)}
              onBlur={() => settle !== "" && act(() => slm.apiSLMSetSettleTime(Number(settle)))} />
            <Button size="small" variant="outlined" onClick={() => act(() => slm.apiSLMOpenDisplay())}>Reopen output</Button>
            <Button size="small" variant="outlined" onClick={() => act(() => slm.apiSLMCloseDisplay())}>Close output</Button>
            <Button size="small" onClick={() => act(async () => setDisplays(await slm.apiSLMGetDisplays()))}>List displays</Button>
            {displays && (
              <Typography variant="caption">
                {displays.length ? displays.map((d, i) => `${i}: ${Array.isArray(d) ? d.join("×") : d}`).join(" · ") : "none found"}
              </Typography>
            )}
          </Stack>
        </Grid>
        <Grid item xs={12} md={5}>
          <Typography variant="caption" color="text.secondary">On the display now ({status.source || "-"})</Typography>
          <Box sx={{ border: "1px solid #444", bgcolor: "#000", lineHeight: 0 }}>
            <img src={previewUrl} alt="SLM display" style={{ width: "100%", imageRendering: "pixelated" }} />
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

// ───────────────────────────── patterns ─────────────────────────────

const PatternsTab = ({ status, act }) => {
  const [kind, setKind] = useState("grating");
  const [p, setP] = useState({ value: 255, period: 32, angle: 0, duty: 0.5, phase: 0, sinusoidal: false, cx: -1, cy: -1, radius: 20, nx: 3, ny: 3 });
  const [fit, setFit] = useState("center");
  const set = (k) => (e) => setP({ ...p, [k]: e.target.type === "checkbox" ? e.target.checked : Number(e.target.value) });
  const num = (k, label, step = 1) => (
    <TextField key={k} size="small" type="number" label={label} value={p[k]} onChange={set(k)} inputProps={{ step }} sx={{ width: 120 }} />
  );
  const fields = {
    blank: [num("value", "Value 0-255")],
    checkerboard: [num("period", "Period px"), num("value", "Value")],
    grating: [num("period", "Period px"), num("angle", "Angle °"), num("duty", "Duty", 0.05), num("phase", "Phase (periods)", 0.05), num("value", "Value"),
      <FormControlLabel key="sin" control={<Switch checked={p.sinusoidal} onChange={set("sinusoidal")} />} label="sinusoidal" />],
    disk: [num("cx", "cx (-1 = centre)"), num("cy", "cy (-1 = centre)"), num("radius", "Radius px"), num("value", "Value")],
    grid: [num("nx", "nx"), num("ny", "ny"), num("radius", "Spot radius"), num("value", "Value")],
  };
  const onFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => act(() => slm.apiSLMShowImage(reader.result, fit));
    reader.readAsDataURL(file);
    e.target.value = "";
  };
  return (
    <Stack spacing={2}>
      <Paper sx={{ p: 1.5 }}>
        <Typography variant="subtitle2" gutterBottom>Built-in pattern (display pixels)</Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap alignItems="center">
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Pattern</InputLabel>
            <Select label="Pattern" value={kind} onChange={(e) => setKind(e.target.value)}>
              {(status?.patterns || Object.keys(fields)).map((k) => <MenuItem key={k} value={k}>{k}</MenuItem>)}
            </Select>
          </FormControl>
          {fields[kind]}
          <Button variant="contained" onClick={() => act(() => slm.apiSLMShowPattern(kind, p))}>Show</Button>
        </Stack>
        <Stack direction="row" spacing={1} sx={{ mt: 1.5 }}>
          <Button size="small" variant="outlined" onClick={() => act(() => slm.apiSLMShowPattern("blank", { value: 0 }))}>All off</Button>
          <Button size="small" variant="outlined" onClick={() => act(() => slm.apiSLMShowPattern("blank", { value: 255 }))}>All on</Button>
        </Stack>
      </Paper>
      <Paper sx={{ p: 1.5 }}>
        <Typography variant="subtitle2" gutterBottom>Image file</Typography>
        <Stack direction="row" spacing={1} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Fit</InputLabel>
            <Select label="Fit" value={fit} onChange={(e) => setFit(e.target.value)}>
              <MenuItem value="center">center (1:1 pixels)</MenuItem>
              <MenuItem value="stretch">stretch to display</MenuItem>
            </Select>
          </FormControl>
          <Button variant="outlined" component="label">Show image… <input hidden type="file" accept="image/*" onChange={onFile} /></Button>
          <Typography variant="caption" color="text.secondary">PNG/BMP/TIFF/JPEG, shown as 8-bit grey.</Typography>
        </Stack>
      </Paper>
    </Stack>
  );
};

// ───────────────────────────── draw on the camera ─────────────────────────────

const rasterizeMask = (shapes, w, h) => {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(w));
  canvas.height = Math.max(1, Math.round(h));
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  shapes.forEach((s) => {
    const color = s.erase ? "#000" : "#fff";
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    if (s.type === "brush") {
      ctx.lineWidth = s.size;
      ctx.beginPath();
      s.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (s.points.length === 1) ctx.lineTo(s.points[0][0] + 0.01, s.points[0][1]);
      ctx.stroke();
    } else if (s.type === "rect") {
      ctx.fillRect(Math.min(s.x0, s.x1), Math.min(s.y0, s.y1), Math.abs(s.x1 - s.x0), Math.abs(s.y1 - s.y0));
    } else if (s.type === "ellipse") {
      ctx.beginPath();
      ctx.ellipse((s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2, Math.abs(s.x1 - s.x0) / 2, Math.abs(s.y1 - s.y0) / 2, 0, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
  return canvas.toDataURL("image/png");
};

const ShapeSvg = ({ s, i }) => {
  const color = s.erase ? "rgba(255,80,80,0.6)" : "rgba(120,255,120,0.55)";
  if (s.type === "brush") {
    return (
      <polyline key={i} points={s.points.map((p) => p.join(",")).join(" ")} fill="none" stroke={color}
        strokeWidth={s.size} strokeLinecap="round" strokeLinejoin="round" />
    );
  }
  if (s.type === "rect") {
    return <rect key={i} x={Math.min(s.x0, s.x1)} y={Math.min(s.y0, s.y1)} width={Math.abs(s.x1 - s.x0)} height={Math.abs(s.y1 - s.y0)} fill={color} />;
  }
  return <ellipse key={i} cx={(s.x0 + s.x1) / 2} cy={(s.y0 + s.y1) / 2} rx={Math.abs(s.x1 - s.x0) / 2} ry={Math.abs(s.y1 - s.y0) / 2} fill={color} />;
};

const DrawTab = ({ status, act }) => {
  const sub = useSubsampling();
  const containerRef = useRef(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [tool, setTool] = useState("brush");
  const [brush, setBrush] = useState(30);
  const [shapes, setShapes] = useState([]);
  const [draft, setDraft] = useState(null);
  const [invert, setInvert] = useState(false);
  const [value, setValue] = useState(255);
  const [auto, setAuto] = useState(false);
  const [lastInfo, setLastInfo] = useState("");

  const toImg = (e) => pointerToImage(containerRef.current, imageSize, e.clientX, e.clientY);
  const onDown = (e) => {
    const p = toImg(e);
    if (!p) return;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const erase = tool === "eraser";
    setDraft(tool === "brush" || erase ? { type: "brush", size: brush, erase, points: [[p.x, p.y]] }
      : { type: tool, x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  };
  const onMove = (e) => {
    if (!draft) return;
    const p = toImg(e);
    if (!p) return;
    setDraft(draft.type === "brush" ? { ...draft, points: [...draft.points, [p.x, p.y]] } : { ...draft, x1: p.x, y1: p.y });
  };
  const onUp = () => {
    if (draft) setShapes((s) => [...s, draft]);
    setDraft(null);
  };

  const project = useCallback(() => act(async () => {
    const { width, height } = imageSize;
    if (!width) throw new Error("no camera image yet");
    const mask = rasterizeMask(shapes, width, height);
    const r = await slm.apiSLMShowCameraMask(mask, Math.round(width * sub), Math.round(height * sub), value, invert);
    setLastInfo(`${r.on_pixels} display pixels on`);
  }), [act, imageSize, shapes, sub, value, invert]);

  useEffect(() => {
    if (!auto) return undefined;
    const id = setTimeout(project, 250);
    return () => clearTimeout(id);
  }, [auto, shapes, invert, value, project]);

  const overlay = (
    <Overlay imageSize={imageSize}>
      {shapes.map((s, i) => <ShapeSvg key={i} s={s} i={i} />)}
      {draft && <ShapeSvg s={draft} i="draft" />}
    </Overlay>
  );
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={8}>
        {!status?.calibrated && (
          <Alert severity="info" sx={{ mb: 1 }}>Camera → display is not calibrated: camera pixels map 1:1 onto display pixels. Calibrate first for the light to land where you draw.</Alert>
        )}
        <Box ref={containerRef} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
          sx={{ border: "1px solid #444", borderRadius: 1, overflow: "hidden", cursor: "crosshair", touchAction: "none", position: "relative" }}>
          <LiveViewControlWrapper onImageLoad={(w, h) => setImageSize({ width: w, height: h })} overlayContent={overlay}
            enableStageMovement={false} enableZoomPan={false} />
        </Box>
        <Typography variant="caption" color="text.secondary">
          {imageSize.width ? `stream ${imageSize.width}×${imageSize.height} px, ×${sub} = camera ${Math.round(imageSize.width * sub)}×${Math.round(imageSize.height * sub)} px` : "waiting for the live stream…"}
          {lastInfo && ` · ${lastInfo}`}
        </Typography>
      </Grid>
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 1.5 }}>
          <ToggleButtonGroup size="small" exclusive value={tool} onChange={(e, v) => v && setTool(v)} sx={{ mb: 1.5 }}>
            <ToggleButton value="brush"><Tooltip title="Brush"><BrushIcon /></Tooltip></ToggleButton>
            <ToggleButton value="rect"><Tooltip title="Rectangle"><CropSquareIcon /></Tooltip></ToggleButton>
            <ToggleButton value="ellipse"><Tooltip title="Ellipse"><RadioButtonUncheckedIcon /></Tooltip></ToggleButton>
            <ToggleButton value="eraser"><Tooltip title="Eraser"><AutoFixNormalIcon /></Tooltip></ToggleButton>
          </ToggleButtonGroup>
          <Typography variant="caption">Brush size: {brush} px</Typography>
          <Slider size="small" min={2} max={200} value={brush} onChange={(e, v) => setBrush(v)} />
          <Typography variant="caption">Light value: {value}</Typography>
          <Slider size="small" min={1} max={255} value={value} onChange={(e, v) => setValue(v)} />
          <FormControlLabel control={<Switch checked={invert} onChange={(e) => setInvert(e.target.checked)} />} label="Invert (light outside the drawing)" />
          <FormControlLabel control={<Switch checked={auto} onChange={(e) => setAuto(e.target.checked)} />} label="Project automatically" />
          <Divider sx={{ my: 1 }} />
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            <Button variant="contained" startIcon={<LightbulbIcon />} onClick={project} disabled={!imageSize.width}>Project</Button>
            <IconButton onClick={() => setShapes((s) => s.slice(0, -1))} disabled={!shapes.length}><UndoIcon /></IconButton>
            <IconButton onClick={() => setShapes([])} disabled={!shapes.length}><DeleteSweepIcon /></IconButton>
            <Button variant="outlined" onClick={() => act(() => slm.apiSLMShowPattern("blank", { value: 0 }))}>All off</Button>
          </Stack>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
            {shapes.length} shape(s). The drawing is rasterised at camera resolution and warped onto the display with the calibrated affine.
          </Typography>
        </Paper>
      </Grid>
    </Grid>
  );
};

// ───────────────────────────── calibration ─────────────────────────────

const CalibrateTab = ({ status, act, refresh }) => {
  const sub = useSubsampling();
  const detectors = useDetectorNames();
  const containerRef = useRef(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [nPoints, setNPoints] = useState(5);
  const [radius, setRadius] = useState(15);
  const [detector, setDetector] = useState("");
  const [manual, setManual] = useState(null); // {points, cam, step}
  const run = status?.calibration_run || {};
  const affine = status?.affine || {};

  const startManual = () => act(async () => {
    const points = await slm.apiSLMGetCalibrationPoints(nPoints);
    setManual({ points, cam: points.map(() => null), step: 0 });
    await slm.apiSLMShowPattern("disk", { cx: points[0][0], cy: points[0][1], radius });
  });
  const showStep = (k) => act(() => slm.apiSLMShowPattern("disk", { cx: manual.points[k][0], cy: manual.points[k][1], radius }));
  const onClick = (px, py) => {
    if (!manual) return;
    const cam = [...manual.cam];
    cam[manual.step] = [px * sub, py * sub];
    const step = Math.min(manual.step + 1, manual.points.length - 1);
    setManual({ ...manual, cam, step });
    if (step !== manual.step) showStep(step);
  };
  const compute = () => act(async () => {
    const pairs = manual.points.map((d, i) => [manual.cam[i], d]).filter(([c]) => c);
    await slm.apiSLMComputeAffine(pairs.map(([c]) => c), pairs.map(([, d]) => d));
    await slm.apiSLMShowPattern("blank", { value: 0 });
    setManual(null);
  });
  const overlay = manual && (
    <Overlay imageSize={imageSize}>
      {manual.cam.map((c, i) => c && (
        <g key={i}>
          <circle cx={c[0] / sub} cy={c[1] / sub} r={8} fill="none" stroke="#0f0" strokeWidth={2} />
          <text x={c[0] / sub + 10} y={c[1] / sub + 4} fontSize={12} fill="#0f0">{i + 1}</text>
        </g>
      ))}
      <text x={10} y={22} fontSize={14} fill="#ff0" fontWeight="bold">Click spot {manual.step + 1}/{manual.points.length}</text>
    </Overlay>
  );
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={5}>
        <Paper sx={{ p: 1.5, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Automatic</Typography>
          <Typography variant="caption" color="text.secondary">Shows {nPoints} spots one by one and finds each in the camera image. The DMD light path must reach the camera; pick an exposure where a spot is bright but not saturated.</Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 1.5 }} flexWrap="wrap" useFlexGap alignItems="center">
            <TextField size="small" type="number" label="Points (3-9)" value={nPoints} sx={{ width: 110 }} onChange={(e) => setNPoints(Number(e.target.value))} />
            <TextField size="small" type="number" label="Spot radius px" value={radius} sx={{ width: 130 }} onChange={(e) => setRadius(Number(e.target.value))} />
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Camera</InputLabel>
              <Select label="Camera" value={detector} onChange={(e) => setDetector(e.target.value)}>
                <MenuItem value="">current</MenuItem>
                {detectors.map((d) => <MenuItem key={d} value={d}>{d}</MenuItem>)}
              </Select>
            </FormControl>
            {run.running
              ? <Button color="error" startIcon={<StopIcon />} onClick={() => act(slm.apiSLMStopCalibration)}>Stop</Button>
              : <Button variant="contained" startIcon={<GpsFixedIcon />} onClick={() => act(() => slm.apiSLMRunAutoCalibration(nPoints, radius, detector))}>Run</Button>}
          </Stack>
          {run.running && <LinearProgress variant="determinate" value={(100 * (run.step || 0)) / (run.n_points || 1)} sx={{ mt: 1 }} />}
          {run.finished && (run.error
            ? <Alert severity="error" sx={{ mt: 1 }}>{run.error}</Alert>
            : <Alert severity="success" sx={{ mt: 1 }}>Calibrated: {run.residual_px?.toFixed(2)} px rms over {run.points?.filter((p) => p.camera).length} spots.</Alert>)}
          {run.points && (
            <Box component="table" sx={{ mt: 1, fontSize: 12, "& td": { pr: 1.5 } }}>
              <tbody>
                {run.points.map((p, i) => (
                  <tr key={i}><td>{i + 1}</td><td>display {p.display.join(", ")}</td>
                    <td>{p.camera ? `camera ${p.camera.map((v) => v.toFixed(1)).join(", ")} (snr ${p.snr?.toFixed(0)})` : "not found"}</td></tr>
                ))}
              </tbody>
            </Box>
          )}
        </Paper>
        <Paper sx={{ p: 1.5, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Manual</Typography>
          <Typography variant="caption" color="text.secondary">A spot is shown on the display; click its image on the camera. Repeat for every spot, then compute.</Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 1.5 }} flexWrap="wrap" useFlexGap>
            {!manual
              ? <Button variant="outlined" onClick={startManual}>Start</Button>
              : <>
                <Button size="small" onClick={() => showStep(manual.step)}>Show spot {manual.step + 1} again</Button>
                <Button size="small" onClick={() => { const step = (manual.step + 1) % manual.points.length; setManual({ ...manual, step }); showStep(step); }}>Skip</Button>
                <Button size="small" variant="contained" disabled={manual.cam.filter(Boolean).length < 3} onClick={compute}>Compute ({manual.cam.filter(Boolean).length}/{manual.points.length})</Button>
                <Button size="small" color="inherit" onClick={() => { setManual(null); act(() => slm.apiSLMShowPattern("blank", { value: 0 })); }}>Cancel</Button>
              </>}
          </Stack>
        </Paper>
        <Paper sx={{ p: 1.5 }}>
          <Typography variant="subtitle2" gutterBottom>Affine camera → display</Typography>
          <Typography variant="caption" sx={{ fontFamily: "monospace", display: "block" }}>
            x' = {affine.a11?.toFixed(4)}·x + {affine.a12?.toFixed(4)}·y + {affine.tx?.toFixed(1)}<br />
            y' = {affine.a21?.toFixed(4)}·x + {affine.a22?.toFixed(4)}·y + {affine.ty?.toFixed(1)}
          </Typography>
          <Typography variant="caption" color="text.secondary">{status?.calibration?.time ? `from ${status.calibration.time}` : "identity"}</Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
            <Button size="small" color="warning" onClick={() => act(slm.apiSLMResetAffine)}>Reset to identity</Button>
            <Button size="small" onClick={refresh}>Refresh</Button>
          </Stack>
        </Paper>
      </Grid>
      <Grid item xs={12} md={7}>
        <Box ref={containerRef} sx={{ border: manual ? "2px solid #ff0" : "1px solid #444", borderRadius: 1, overflow: "hidden", position: "relative", cursor: manual ? "crosshair" : "default" }}>
          <LiveViewControlWrapper onClick={onClick} onImageLoad={(w, h) => setImageSize({ width: w, height: h })}
            overlayContent={overlay} enableStageMovement={false} enableZoomPan={false} />
        </Box>
      </Grid>
    </Grid>
  );
};

// ───────────────────────────── sequences ─────────────────────────────

const SequenceTab = ({ status, act }) => {
  const detectors = useDetectorNames();
  const [sequences, setSequences] = useState([]);
  const [selected, setSelected] = useState("");
  const [uploadName, setUploadName] = useState("");
  const [interval, setIntervalMs] = useState(500);
  const [loop, setLoop] = useState(false);
  const [acq, setAcq] = useState({ detectorName: "", settleMs: "", repeats: 1, save: true, fileName: "" });
  const run = status?.acquisition || {};
  const play = status?.play || {};
  const reload = useCallback(() => slm.apiSLMListSequences().then(setSequences).catch(() => {}), []);
  useEffect(() => { reload(); }, [reload]);
  const acquiredUrl = useMemo(() => (run.done ? slm.slmAcquiredFrameUrl(-1, 384) : null), [run.done]);
  const current = sequences.find((s) => s.name === selected);

  const onFiles = (e) => {
    const files = Array.from(e.target.files || []).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    e.target.value = "";
    if (!files.length) return;
    const name = uploadName.trim() || files[0].name.replace(/\.[^.]+$/, "");
    act(async () => {
      const frames = await Promise.all(files.map((f) => new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve({ name: f.name, data_b64: r.result });
        r.onerror = reject;
        r.readAsDataURL(f);
      })));
      const info = await slm.apiSLMUploadSequence(name, frames);
      setSelected(info.name);
      await reload();
    });
  };
  const start = () => act(() => slm.apiSLMStartAcquisition({
    sequence: selected,
    detectorName: acq.detectorName,
    settleMs: acq.settleMs === "" ? null : Number(acq.settleMs),
    repeats: Number(acq.repeats) || 1,
    save: acq.save,
    fileName: acq.fileName,
  }));
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={5}>
        <Paper sx={{ p: 1.5, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Sequences</Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <TextField size="small" label="Name (optional)" value={uploadName} onChange={(e) => setUploadName(e.target.value)} />
            <Button variant="outlined" component="label">Upload images… <input hidden multiple type="file" accept="image/*" onChange={onFiles} /></Button>
          </Stack>
          <Typography variant="caption" color="text.secondary">Files are ordered by name and stored as 8-bit PNG at their own resolution.</Typography>
          <Stack sx={{ mt: 1.5 }} spacing={0.5}>
            {sequences.length === 0 && <Typography variant="body2" color="text.secondary">No sequences yet.</Typography>}
            {sequences.map((s) => (
              <Stack key={s.name} direction="row" spacing={1} alignItems="center"
                sx={{ p: 0.5, borderRadius: 1, cursor: "pointer", bgcolor: s.name === selected ? "action.selected" : undefined }}
                onClick={() => setSelected(s.name)}>
                <img src={slm.slmSequenceFrameUrl(s.name, 0, 64)} alt="" style={{ height: 32, background: "#000" }} />
                <Typography variant="body2" sx={{ flex: 1 }}>{s.name}</Typography>
                <Chip size="small" label={`${s.count} × ${s.size ? s.size.join("×") : "?"}`} />
                <IconButton size="small" onClick={(e) => { e.stopPropagation(); act(async () => { await slm.apiSLMDeleteSequence(s.name); if (selected === s.name) setSelected(""); await reload(); }); }}><DeleteIcon fontSize="small" /></IconButton>
              </Stack>
            ))}
          </Stack>
        </Paper>
        {current && (
          <Paper sx={{ p: 1.5, mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>{current.name}: frames (click to show)</Typography>
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
              {current.frames.slice(0, 48).map((f, i) => (
                <Tooltip key={i} title={f}>
                  <img src={slm.slmSequenceFrameUrl(current.name, i, 96)} alt={f} style={{ height: 48, background: "#000", cursor: "pointer", border: play.running && play.index === i ? "2px solid #0f0" : "1px solid #333" }}
                    onClick={() => act(() => slm.apiSLMShowSequenceFrame(current.name, i))} />
                </Tooltip>
              ))}
            </Box>
            <Stack direction="row" spacing={1} sx={{ mt: 1.5 }} alignItems="center" flexWrap="wrap" useFlexGap>
              <TextField size="small" type="number" label="Interval ms" value={interval} sx={{ width: 120 }} onChange={(e) => setIntervalMs(Number(e.target.value))} />
              <FormControlLabel control={<Switch checked={loop} onChange={(e) => setLoop(e.target.checked)} />} label="loop" />
              {play.running
                ? <Button color="error" startIcon={<StopIcon />} onClick={() => act(slm.apiSLMStopSequence)}>Stop ({play.index + 1}/{play.total})</Button>
                : <Button variant="outlined" startIcon={<PlayArrowIcon />} onClick={() => act(() => slm.apiSLMPlaySequence(current.name, interval, loop))}>Play (display only)</Button>}
            </Stack>
          </Paper>
        )}
      </Grid>
      <Grid item xs={12} md={7}>
        <Paper sx={{ p: 1.5 }}>
          <Typography variant="subtitle2" gutterBottom>Synchronised acquisition</Typography>
          <Typography variant="caption" color="text.secondary">Per frame: show it, wait the settle time, take exactly one camera frame (software trigger when the camera supports it). Saved as OME-TIFF under &lt;data&gt;/SLM, one channel per pattern.</Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 1.5 }} flexWrap="wrap" useFlexGap alignItems="center">
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Camera</InputLabel>
              <Select label="Camera" value={acq.detectorName} onChange={(e) => setAcq({ ...acq, detectorName: e.target.value })}>
                <MenuItem value="">current</MenuItem>
                {detectors.map((d) => <MenuItem key={d} value={d}>{d}</MenuItem>)}
              </Select>
            </FormControl>
            <TextField size="small" type="number" label="Settle ms (blank = default)" value={acq.settleMs} sx={{ width: 190 }} onChange={(e) => setAcq({ ...acq, settleMs: e.target.value })} />
            <TextField size="small" type="number" label="Repeats" value={acq.repeats} sx={{ width: 100 }} onChange={(e) => setAcq({ ...acq, repeats: e.target.value })} />
            <TextField size="small" label="File name" value={acq.fileName} onChange={(e) => setAcq({ ...acq, fileName: e.target.value })} />
            <FormControlLabel control={<Switch checked={acq.save} onChange={(e) => setAcq({ ...acq, save: e.target.checked })} />} label="save" />
          </Stack>
          <Stack direction="row" spacing={1} sx={{ mt: 1.5 }} alignItems="center">
            {run.running
              ? <Button color="error" variant="contained" startIcon={<StopIcon />} onClick={() => act(slm.apiSLMStopAcquisition)}>Stop</Button>
              : <Button variant="contained" startIcon={<PlayArrowIcon />} disabled={!selected} onClick={start}>Start</Button>}
            {run.total > 0 && <Typography variant="body2">{run.done}/{run.total}{run.fps ? ` · ${run.fps.toFixed(1)} fps` : ""}</Typography>}
            {run.triggered != null && <Chip size="small" label={run.triggered ? "software trigger" : "free-run grab"} />}
          </Stack>
          {run.total > 0 && <LinearProgress variant="determinate" value={(100 * run.done) / run.total} sx={{ mt: 1 }} />}
          {run.error && <Alert severity="error" sx={{ mt: 1 }}>{run.error}</Alert>}
          {run.finished && !run.error && run.path && <Alert severity="success" sx={{ mt: 1 }}>Saved {run.path}</Alert>}
          {acquiredUrl && (
            <Box sx={{ mt: 1.5, border: "1px solid #444", bgcolor: "#000", lineHeight: 0 }}>
              <img src={acquiredUrl} alt="last acquired frame" style={{ width: "100%" }} />
            </Box>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
};

// ───────────────────────────── app ─────────────────────────────

const SLMController = () => {
  const [tab, setTab] = useState(0);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [previewNonce, setPreviewNonce] = useState(0);
  const refresh = useCallback(async () => {
    try {
      setStatus(await slm.apiSLMGetStatus());
    } catch (e) {
      setError(`SLM backend unavailable: ${errText(e)}`);
    }
  }, []);
  const running = Boolean(status?.acquisition?.running || status?.calibration_run?.running || status?.play?.running);
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, running ? 400 : 2000);
    return () => clearInterval(id);
  }, [refresh, running]);
  useEffect(() => { setPreviewNonce((n) => n + 1); }, [status?.seq]);
  const act = useCallback(async (fn) => {
    try {
      setError("");
      await fn();
    } catch (e) {
      setError(errText(e));
    } finally {
      refresh();
    }
  }, [refresh]);

  return (
    <Box sx={{ p: 1 }}>
      {error && <Alert severity="error" sx={{ mb: 1 }} onClose={() => setError("")}>{error}</Alert>}
      <DisplayCard status={status} act={act} previewNonce={previewNonce} />
      <Tabs value={tab} onChange={(e, v) => setTab(v)} sx={{ mb: 1 }}>
        <Tab label="Patterns" />
        <Tab label="Draw on camera" />
        <Tab label="Calibrate" />
        <Tab label="Sequence" />
      </Tabs>
      {tab === 0 && <PatternsTab status={status} act={act} />}
      {tab === 1 && <DrawTab status={status} act={act} />}
      {tab === 2 && <CalibrateTab status={status} act={act} refresh={refresh} />}
      {tab === 3 && <SequenceTab status={status} act={act} />}
    </Box>
  );
};

export default SLMController;
