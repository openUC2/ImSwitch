// Spectrometer app: live raw region and spectrum (pixel / nm / cm⁻¹), camera settings,
// region of interest, dark, averaging, calibration sources, the calibration wizard,
// grating alignment with a known line (e.g. the Q-Box 520 nm laser), resolution, saving.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  FormControlLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Slider,
  Stack,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import * as api from "../../backendapi/apiSpectrometerController.js";
import SpectrumChart from "./SpectrumChart.jsx";
import SpectrometerWizard from "./SpectrometerWizard.jsx";

const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? "–" : Number(v).toFixed(d));
const num = (v, f = parseFloat) => {
  const x = f(v);
  return Number.isFinite(x) ? x : undefined;
};

function Section({ title, children, right }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
        <Typography variant="subtitle2">{title}</Typography>
        {right}
      </Stack>
      <Stack spacing={1}>{children}</Stack>
    </Paper>
  );
}

// value shown in the axis units of the chart for a wavelength (nm) using the spectrum arrays
function nmToAxis(nm, spectrum, axis) {
  if (nm == null || !spectrum) return null;
  if (axis === "nm") return nm;
  if (axis === "cm1") {
    const laser = spectrum.metadata?.laserNm;
    return laser ? 1e7 / laser - 1e7 / nm : null;
  }
  // px: interpolate on the (monotonic) nm array
  const xs = spectrum.nm;
  const ps = spectrum.px;
  if (!xs || xs.length < 2) return null;
  for (let i = 0; i < xs.length - 1; i += 1) {
    const a = xs[i];
    const b = xs[i + 1];
    if ((a - nm) * (b - nm) <= 0 && a !== b) return ps[i] + ((nm - a) / (b - a)) * (ps[i + 1] - ps[i]);
  }
  return null;
}

function pxToAxis(px, spectrum, axis) {
  if (px == null || !spectrum) return null;
  if (axis === "px") return px;
  const ps = spectrum.px;
  const xs = axis === "nm" ? spectrum.nm : spectrum.ramanShiftCm1;
  if (!xs || !ps) return null;
  for (let i = 0; i < ps.length - 1; i += 1) {
    if ((ps[i] - px) * (ps[i + 1] - px) <= 0 && ps[i] !== ps[i + 1]) {
      return xs[i] + ((px - ps[i]) / (ps[i + 1] - ps[i])) * (xs[i + 1] - xs[i]);
    }
  }
  return null;
}

export default function SpectrometerController() {
  const [state, setState] = useState(null);
  const [available, setAvailable] = useState(true);
  const [spectrum, setSpectrum] = useState(null);
  const [image, setImage] = useState(null);
  const [sources, setSources] = useState([]);
  const [axis, setAxis] = useState("px");
  const [logY, setLogY] = useState(false);
  const [fullFrame, setFullFrame] = useState(false);
  const [job, setJob] = useState(null);
  const [msg, setMsg] = useState(null);
  const [cam, setCam] = useState({ exposureMs: "", gain: "" });
  const [roi, setRoi] = useState({ rowStart: "", rowStop: "" });
  const [nDark, setNDark] = useState(10);
  const [nAcq, setNAcq] = useState(5);
  const [save, setSave] = useState({ name: "spectrum", fileFormat: "hdf5", includeRoi: true, notes: "" });
  const [wizardOpen, setWizardOpen] = useState(false);
  const [align, setAlign] = useState({ show: false, lineNm: 520, targetNm: "" });
  const [plan, setPlan] = useState(null);
  const [resolution, setResolution] = useState(null);
  const [laserNm, setLaserNm] = useState("");
  const jobTimer = useRef(null);
  const live = !!state?.live?.running;

  const loadState = useCallback(async () => {
    try {
      const s = await api.apiSpectrometerGetState();
      setState(s);
      setAvailable(true);
      return s;
    } catch (e) {
      setAvailable(false);
      return null;
    }
  }, []);

  const loadSources = useCallback(async () => {
    try {
      setSources((await api.apiSpectrometerGetSources()).sources || []);
    } catch {
      /* controller missing: shown by loadState */
    }
  }, []);

  const loadSpectrum = useCallback(async (which = "auto") => {
    try {
      const sp = await api.apiSpectrometerGetSpectrum(which, 1536);
      if (sp.available) setSpectrum(sp);
    } catch {
      /* keep the last one */
    }
  }, []);

  const loadImage = useCallback(async () => {
    try {
      const im = await api.apiSpectrometerGetImage(fullFrame, false, 1536);
      if (im.image) setImage(im);
    } catch {
      /* keep the last one */
    }
  }, [fullFrame]);

  // first load: state, sources, fill the inputs
  useEffect(() => {
    (async () => {
      const s = await loadState();
      if (s) {
        setCam({ exposureMs: s.camera?.exposureMs ?? "", gain: s.camera?.gain ?? "" });
        setRoi({ rowStart: s.processing?.rowStart ?? "", rowStop: s.processing?.rowStop ?? "" });
        setLaserNm(s.laserNm ?? "");
        setAlign((a) => ({ ...a, targetNm: s.alignmentTargetNm ?? "" }));
        setNAcq(s.processing?.nAverage ?? 5);
      }
      loadSources();
      loadSpectrum();
    })();
    const id = setInterval(loadState, 2000);
    return () => clearInterval(id);
  }, [loadState, loadSources, loadSpectrum]);

  // job polling timer: stopped when the page goes away
  useEffect(() => {
    const t = jobTimer;
    return () => clearInterval(t.current);
  }, []);

  // live polling
  useEffect(() => {
    if (!live) return undefined;
    const spectrumId = setInterval(() => loadSpectrum("live"), 300);
    const imageId = setInterval(loadImage, 1000);
    return () => {
      clearInterval(spectrumId);
      clearInterval(imageId);
    };
  }, [live, loadSpectrum, loadImage]);

  useEffect(() => {
    loadImage();
  }, [fullFrame, loadImage]);

  // alignment readout
  useEffect(() => {
    if (!align.show) return undefined;
    const update = async () => {
      try {
        setPlan(await api.apiSpectrometerGetAlignment(num(align.lineNm) || 520, num(align.targetNm) || 0, "auto"));
      } catch {
        /* ignore */
      }
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, [align.show, align.lineNm, align.targetNm]);

  const watchJob = useCallback(() => {
    clearInterval(jobTimer.current);
    jobTimer.current = setInterval(async () => {
      const j = await api.apiSpectrometerGetAcquisition();
      setJob(j);
      if (j.state !== "running") {
        clearInterval(jobTimer.current);
        loadState();
        if (j.kind === "acquisition" && j.state === "done") loadSpectrum("acquired");
      }
    }, 300);
  }, [loadState, loadSpectrum]);

  const run = async (fn, okText) => {
    try {
      const r = await fn();
      if (r && r.success === false) setMsg({ severity: "error", text: r.error || "failed" });
      else if (okText) setMsg({ severity: "success", text: typeof okText === "function" ? okText(r) : okText });
      return r;
    } catch (e) {
      setMsg({ severity: "error", text: String(e) });
      return null;
    }
  };

  // raw region of the shown spectrum as a NumPy file (for notebooks)
  const downloadRegion = async () => {
    try {
      const blob = await api.apiSpectrometerGetRoiArray("auto", false);
      if (blob.type && blob.type.includes("json")) {
        const r = JSON.parse(await blob.text());
        setMsg({ severity: "error", text: r.error || "no region yet" });
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${save.name || "spectrometer"}_region.npy`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setMsg({ severity: "error", text: String(e) });
    }
  };

  const toggleLive = async () => {
    if (live) await run(api.apiSpectrometerStopLive);
    else await run(() => api.apiSpectrometerStartLive(250));
    loadState();
  };

  const applySettings = async (patch) => {
    await run(() => api.apiSpectrometerSetSettings(patch));
    loadState();
  };

  const markers = useMemo(() => {
    const m = [];
    if (!spectrum) return m;
    (spectrum.peaks || []).slice(0, 6).forEach((p, i) =>
      m.push({
        x: axis === "px" ? p.px : axis === "nm" ? p.nm : pxToAxis(p.px, spectrum, axis),
        label: axis === "px" ? `${fmt(p.px, 1)}` : axis === "nm" ? `${fmt(p.nm, 2)} nm` : "",
        color: "#9e9e9e",
        dash: [2, 3],
        width: 1,
        row: i % 3,
      }),
    );
    if (align.show && plan?.success) {
      const target = plan.lineOnSensorAtTarget ? plan.linePxAtTarget : plan.parkPx;
      m.push({ x: pxToAxis(target, spectrum, axis), label: `target ${plan.lineNm} nm`, color: "#e91e63", width: 2 });
      if (plan.measuredPx != null) {
        m.push({ x: pxToAxis(plan.measuredPx, spectrum, axis), label: "now", color: "#2196f3", width: 2, row: 1 });
      }
    }
    if (state?.laserNm && axis !== "cm1") {
      const x = nmToAxis(state.laserNm, spectrum, axis);
      if (x != null) m.push({ x, label: `laser ${state.laserNm} nm`, color: "#4caf50" });
    }
    return m;
  }, [spectrum, axis, align.show, plan, state?.laserNm]);

  if (!available) {
    return (
      <Box p={2}>
        <Alert severity="info">
          No spectrometer in this setup. Add "Spectrometer" to availableWidgets and a
          "spectrometer" block (see docs/SPECTROMETER.md), or start ImSwitch with
          example_virtual_spectrometer.json.
        </Alert>
      </Box>
    );
  }

  const cal = state?.calibration;
  const meta = spectrum?.metadata;
  const sat = meta?.saturation;
  const jobRunning = job?.state === "running";

  return (
    <Box sx={{ p: 2 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1, flexWrap: "wrap", rowGap: 1 }}>
        <Typography variant="h5" sx={{ mr: 1 }}>
          Spectrometer
        </Typography>
        {state?.simulated && <Chip size="small" color="secondary" label="simulated" />}
        <Chip
          size="small"
          color={cal?.calibrated ? "success" : "warning"}
          label={cal?.calibrated ? `${cal.model} · rms ${fmt(cal.rmsNm, 3)} nm` : "not calibrated (design axis)"}
        />
        <Chip size="small" color={state?.dark ? "success" : "default"} label={state?.dark ? `dark ×${state.dark.frames}` : "no dark"} />
        {state?.smile && <Chip size="small" label={`smile ${fmt(state.smile.bowPx, 1)} px`} />}
        {sat && (
          <Chip
            size="small"
            color={sat.saturated ? "error" : sat.peakFraction > 0.8 ? "warning" : "default"}
            label={`peak ${fmt(100 * sat.peakFraction, 0)} % of ${state?.camera?.bitDepth ?? "?"} bit`}
          />
        )}
        <Box sx={{ flex: 1 }} />
        <ToggleButtonGroup size="small" exclusive value={axis} onChange={(_, v) => v && setAxis(v)}>
          <ToggleButton value="px">pixel</ToggleButton>
          <ToggleButton value="nm">nm</ToggleButton>
          <ToggleButton value="cm1" disabled={!state?.laserNm}>
            cm⁻¹
          </ToggleButton>
        </ToggleButtonGroup>
        <FormControlLabel control={<Switch checked={logY} onChange={(e) => setLogY(e.target.checked)} />} label="log" />
        <Button variant={live ? "outlined" : "contained"} color={live ? "error" : "primary"} onClick={toggleLive}>
          {live ? "Stop live" : "Start live"}
        </Button>
      </Stack>

      {(state?.warnings || []).map((w, i) => (
        <Alert key={i} severity="warning" sx={{ mb: 1, py: 0 }}>
          {w}
        </Alert>
      ))}
      {state?.live?.error && (
        <Alert severity="error" sx={{ mb: 1, py: 0 }}>
          live: {state.live.error}
        </Alert>
      )}
      {msg && (
        <Alert severity={msg.severity} onClose={() => setMsg(null)} sx={{ mb: 1, py: 0 }}>
          {msg.text}
        </Alert>
      )}

      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "320px 1fr" }, gap: 2 }}>
        {/* ---------------------------------------------------------------- controls */}
        <Stack spacing={1.5}>
          <Section title="Camera">
            <Stack direction="row" spacing={1}>
              <TextField size="small" label="exposure (ms)" value={cam.exposureMs}
                onChange={(e) => setCam({ ...cam, exposureMs: e.target.value })} />
              <TextField size="small" label="gain" value={cam.gain}
                onChange={(e) => setCam({ ...cam, gain: e.target.value })} />
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Button size="small" variant="outlined"
                onClick={() => run(() => api.apiSpectrometerSetCamera({ exposureMs: num(cam.exposureMs) ?? -1, gain: num(cam.gain) ?? -1 }),
                  (r) => (r.notes?.length ? r.notes.join("; ") : "camera set")).then(loadState)}>
                Apply
              </Button>
              <Typography variant="caption" color="text.secondary">
                {state?.camera?.bitDepth} bit · {state?.camera?.model}
              </Typography>
            </Stack>
          </Section>

          <Section title="Region on the slit (rows)">
            <Stack direction="row" spacing={1}>
              <TextField size="small" label="first row" value={roi.rowStart}
                onChange={(e) => setRoi({ ...roi, rowStart: e.target.value })} />
              <TextField size="small" label="last row (excl.)" value={roi.rowStop}
                onChange={(e) => setRoi({ ...roi, rowStop: e.target.value })} />
            </Stack>
            <Stack direction="row" spacing={1}>
              <Button size="small" variant="outlined"
                onClick={() => applySettings({ rowStart: num(roi.rowStart, parseInt), rowStop: num(roi.rowStop, parseInt) })}>
                Apply
              </Button>
              <Button size="small" variant="outlined"
                onClick={async () => {
                  const r = await run(api.apiSpectrometerAutoRoi, (x) => `slit rows ${x.rowStart}–${x.rowStop}`);
                  if (r?.success) setRoi({ rowStart: r.rowStart, rowStop: r.rowStop });
                  loadState();
                }}>
                Find slit
              </Button>
              <FormControlLabel control={<Switch size="small" checked={fullFrame} onChange={(e) => setFullFrame(e.target.checked)} />} label="full frame" />
            </Stack>
            <TextField select size="small" label="rows combined by" value={state?.processing?.collapse || "mean"}
              onChange={(e) => applySettings({ collapse: e.target.value })}>
              <MenuItem value="mean">mean</MenuItem>
              <MenuItem value="sum">sum</MenuItem>
            </TextField>
          </Section>

          <Section title="Processing">
            <FormControlLabel control={<Switch size="small" checked={!!state?.processing?.subtractDark}
              onChange={(e) => applySettings({ subtractDark: e.target.checked })} />} label="subtract dark" />
            <FormControlLabel control={<Switch size="small" checked={!!state?.processing?.removeSpikes}
              onChange={(e) => applySettings({ removeSpikes: e.target.checked })} />} label="remove spikes" />
            <FormControlLabel control={<Switch size="small" checked={!!state?.processing?.smileCorrection}
              onChange={(e) => applySettings({ smileCorrection: e.target.checked })} />} label="smile correction" />
            <TextField select size="small" label="acquired frames combined by"
              value={state?.processing?.frameCombine || "mean"}
              helperText="sum = frame adding, for more than the longest exposure"
              onChange={(e) => applySettings({ frameCombine: e.target.value })}>
              <MenuItem value="mean">mean</MenuItem>
              <MenuItem value="sum">sum</MenuItem>
            </TextField>
            <Typography variant="caption">live average: {state?.liveAverage ?? 1} spectra</Typography>
            <Slider size="small" min={1} max={20} step={1} value={state?.liveAverage ?? 1}
              onChangeCommitted={(_, v) => applySettings({ liveAverage: v })} />
          </Section>

          <Section title="Dark">
            <Stack direction="row" spacing={1} alignItems="center">
              <TextField size="small" type="number" label="frames" value={nDark} sx={{ width: 90 }}
                onChange={(e) => setNDark(parseInt(e.target.value, 10) || 1)} />
              <Button size="small" variant="outlined" disabled={jobRunning}
                onClick={() => run(() => api.apiSpectrometerTakeDark(nDark)).then(watchJob)}>
                Take dark
              </Button>
              <Button size="small" onClick={() => run(api.apiSpectrometerClearDark).then(loadState)}>
                Clear
              </Button>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              Block the light (lid closed, sources off). Same exposure and gain as the spectra.
            </Typography>
          </Section>

          <Section title="Light sources" right={
            <Button size="small" onClick={() => run(api.apiSpectrometerAllSourcesOff).then(loadSources)}>all off</Button>}>
            {sources.map((s) => (
              <FormControlLabel key={s.id} sx={{ ml: 0 }}
                control={<Switch size="small" checked={(state?.activeSources || []).includes(s.id)}
                  onChange={(e) => run(() => api.apiSpectrometerSetSource(s.id, e.target.checked)).then(() => { loadState(); loadSources(); })} />}
                label={<Typography variant="body2">{s.label}{s.control === "manual" ? " · by hand" : ""}</Typography>} />
            ))}
          </Section>
        </Stack>

        {/* ---------------------------------------------------------------- data */}
        <Stack spacing={1.5} sx={{ minWidth: 0 }}>
          <Paper variant="outlined" sx={{ p: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {fullFrame ? "full frame (region rows marked)" : "raw region"}
              {image?.rows ? ` · rows ${image.rows[0]}–${image.rows[1]}` : ""}
              {image?.frameShape ? ` · frame ${image.frameShape[1]} × ${image.frameShape[0]}` : ""}
            </Typography>
            <Box sx={{ position: "relative", width: "100%", lineHeight: 0 }}>
              {image?.image ? (
                <img src={image.image} alt="spectrometer camera"
                  style={{ width: "100%", height: fullFrame ? 260 : 140, objectFit: "fill", imageRendering: "pixelated" }} />
              ) : (
                <Box sx={{ height: 140, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Typography variant="body2" color="text.secondary">Start live or acquire to see the camera.</Typography>
                </Box>
              )}
              {fullFrame && image?.frameShape && image?.rows && (
                <Box sx={{
                  position: "absolute", left: 0, right: 0, border: "2px solid #e91e63", pointerEvents: "none",
                  top: `${(100 * image.rows[0]) / image.frameShape[0]}%`,
                  height: `${(100 * (image.rows[1] - image.rows[0])) / image.frameShape[0]}%`,
                }} />
              )}
            </Box>
          </Paper>

          <Paper variant="outlined" sx={{ p: 1 }}>
            <SpectrumChart spectrum={spectrum} axis={axis} markers={markers} logY={logY} />
            <Typography variant="caption" color="text.secondary">
              {spectrum
                ? `${spectrum.which === "acquired" ? "acquired" : "live"} · ${meta?.frames ?? 1} frame(s)` +
                  `${meta?.liveAverage ? ` · live average ${meta.liveAverage}` : ""} · ${fmt(meta?.exposureMs, 1)} ms` +
                  ` · dark ${meta?.darkSubtracted ? "on" : "off"} · smile ${meta?.smileCorrected ? "on" : "off"}` +
                  ` · ${meta?.spikesRemoved ?? 0} spikes removed` +
                  `${spectrum.calibrated ? "" : " · nm axis from the design, not calibrated"}`
                : "no spectrum yet"}
            </Typography>
            {spectrum?.peaks?.length > 0 && (
              <Typography variant="body2" sx={{ mt: 0.5 }}>
                Peaks:{" "}
                {spectrum.peaks.slice(0, 6).map((p) =>
                  `${fmt(p.px, 1)} px / ${fmt(p.nm, 2)} nm (FWHM ${fmt(p.fwhmPx, 1)} px = ${fmt(p.fwhmNm, 3)} nm)`).join(" · ")}
              </Typography>
            )}
          </Paper>

          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "1fr 1fr 1fr" }, gap: 1.5 }}>
            <Section title="Acquire and save">
              <Stack direction="row" spacing={1} alignItems="center">
                <TextField size="small" type="number" label="frames" value={nAcq} sx={{ width: 90 }}
                  onChange={(e) => setNAcq(parseInt(e.target.value, 10) || 1)} />
                <Button size="small" variant="contained" disabled={jobRunning}
                  onClick={() => run(() => api.apiSpectrometerAcquire(nAcq)).then(watchJob)}>
                  Acquire
                </Button>
                {jobRunning && <Button size="small" color="error" onClick={api.apiSpectrometerStopAcquisition}>Stop</Button>}
              </Stack>
              {jobRunning && <LinearProgress variant="determinate" value={100 * (job?.progress || 0)} />}
              {job && !jobRunning && job.state !== "idle" && (
                <Typography variant="caption">{job.kind}: {job.state} {job.error || ""}</Typography>
              )}
              <Divider />
              <Stack direction="row" spacing={1}>
                <TextField size="small" label="name" value={save.name} onChange={(e) => setSave({ ...save, name: e.target.value })} />
                <TextField select size="small" label="format" value={save.fileFormat} sx={{ width: 120 }}
                  onChange={(e) => setSave({ ...save, fileFormat: e.target.value })}>
                  <MenuItem value="hdf5">HDF5</MenuItem>
                  <MenuItem value="csv">CSV + JSON</MenuItem>
                </TextField>
              </Stack>
              <TextField size="small" label="notes (sample, filter, …)" value={save.notes}
                onChange={(e) => setSave({ ...save, notes: e.target.value })} />
              <Stack direction="row" spacing={1} alignItems="center">
                <FormControlLabel control={<Switch size="small" checked={save.includeRoi}
                  onChange={(e) => setSave({ ...save, includeRoi: e.target.checked })} />} label="with frame region" />
                <Button size="small" variant="outlined"
                  onClick={() => run(() => api.apiSpectrometerSave({ ...save, which: "auto" }), (r) => `saved ${r.files?.join(", ")}`)}>
                  Save
                </Button>
                <Button size="small" onClick={downloadRegion}>
                  Region .npy
                </Button>
              </Stack>
            </Section>

            <Section title="Calibration" right={
              <Button size="small" variant="contained" onClick={() => { loadSources(); setWizardOpen(true); }}>Wizard</Button>}>
              <Typography variant="body2">
                {cal?.calibrated
                  ? `${cal.model}, ${cal.nPoints} lines, rms ${fmt(cal.rmsNm, 3)} nm (${cal.createdAt})`
                  : "Design axis only (not measured)."}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {fmt(cal?.nmAtFirstPx, 1)} nm at column 0 · {fmt(cal?.nmAtLastPx, 1)} nm at the last ·{" "}
                {fmt(cal?.nmPerPxAtCenter, 4)} nm/px
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center">
                <TextField size="small" label="excitation laser (nm)" value={laserNm} sx={{ width: 170 }}
                  onChange={(e) => setLaserNm(e.target.value)} />
                <Button size="small" variant="outlined"
                  onClick={() => run(() => api.apiSpectrometerSetLaserWavelength(num(laserNm) || 0)).then(loadState)}>
                  Set
                </Button>
              </Stack>
              <Stack direction="row" spacing={1}>
                <Button size="small" variant="outlined"
                  onClick={() => run(() => api.apiSpectrometerMeasureSmile(5), (r) => `smile: bow ${fmt(r.smile?.bowPx, 1)} px`).then(loadState)}>
                  Measure smile
                </Button>
                <Button size="small" color="warning"
                  onClick={() => run(api.apiSpectrometerResetCalibration, "back to the design axis").then(loadState)}>
                  Reset
                </Button>
              </Stack>
            </Section>

            <Section title="Grating alignment" right={
              <Switch size="small" checked={align.show} onChange={(e) => setAlign({ ...align, show: e.target.checked })} />}>
              <Stack direction="row" spacing={1}>
                <TextField size="small" label="known line (nm)" value={align.lineNm} sx={{ width: 130 }}
                  onChange={(e) => setAlign({ ...align, lineNm: e.target.value })} />
                <TextField size="small" label="window centre (nm)" value={align.targetNm} sx={{ width: 150 }}
                  onChange={(e) => setAlign({ ...align, targetNm: e.target.value })} />
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Switch the known line on (e.g. the Q-Box 520 nm laser at low power, onto a card
                in front of the entrance) and turn the grating until the blue marker sits on the
                red one.
              </Typography>
              {align.show && plan?.success && (
                <>
                  <Typography variant="body2">{plan.advice}</Typography>
                  <Typography variant="caption">
                    window at the target {fmt(plan.windowAtTargetNm?.[0], 1)}–{fmt(plan.windowAtTargetNm?.[1], 1)} nm
                    {plan.deltaTurnDeg != null ? ` · turn ${fmt(plan.deltaTurnDeg, 2)}°` : ""}
                    {plan.leverTravelMm != null ? ` (${fmt(plan.leverTravelMm, 2)} mm at the lever)` : ""}
                  </Typography>
                </>
              )}
              {align.show && plan && !plan.success && <Alert severity="error">{plan.error}</Alert>}
              <Divider />
              <Button size="small" variant="outlined"
                onClick={async () => setResolution(await run(() => api.apiSpectrometerCheckResolution("auto")))}>
                Check resolution
              </Button>
              {resolution?.success && (
                <Typography variant="body2">
                  Strongest line {fmt(resolution.line.nm, 2)} nm: FWHM {fmt(resolution.line.fwhmPx, 1)} px ={" "}
                  {fmt(resolution.line.fwhmNm, 3)} nm{resolution.fwhmCm1 ? ` = ${fmt(resolution.fwhmCm1, 1)} cm⁻¹` : ""};
                  slit image {fmt(resolution.slitImagePx, 1)} px (×{fmt(resolution.widthOverSlitImage, 2)})
                  {resolution.calibrated ? "" : " – design axis"}
                </Typography>
              )}
            </Section>
          </Box>
        </Stack>
      </Box>

      <SpectrometerWizard
        open={wizardOpen}
        onClose={() => {
          setWizardOpen(false);
          loadState();
        }}
        sources={sources}
        onChanged={() => {
          loadState();
          loadSources();
        }}
      />
    </Box>
  );
}
