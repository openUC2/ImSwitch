/**
 * Deconvolution panel of the OME-Zarr viewer (docs/DECONVOLUTION.md).
 *
 * Preview: the crop in the dashed box (the middle of the view) is deconvolved on the Pi with the
 * settings here, so the regularisation can be tuned before the whole stack runs. Whole stack: a job
 * on the registered processing worker, or on the Pi for small stacks; the result is a new OME-Zarr
 * next to the source.
 */
import React, { useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Divider,
  IconButton,
  LinearProgress,
  MenuItem,
  Slider,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { Close } from "@mui/icons-material";

import {
  apiProcessingCancelJob,
  apiProcessingGetDefaults,
  apiProcessingGetStatus,
  apiProcessingGetWorker,
  apiProcessingPreview,
  apiProcessingSetWorker,
  apiProcessingStartJob,
} from "../../backendapi/apiProcessingController";
import { autoWindow } from "./omeZarrModel";

const message = (e) => e?.response?.data?.detail || e?.message || String(e);
const decode = (b64) => new Float32Array(Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0)).buffer);
const PLANES = [8, 16, 32, 64];

/** Grey image of a float plane, 0.1 / 99.9 percentile window. */
function drawGray(canvas, data, w, h) {
  if (!canvas || !data) return;
  canvas.width = w;
  canvas.height = h;
  const { start, end } = autoWindow(data);
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const scale = 255 / (end - start);
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] - start) * scale; // the clamped array clips to 0..255
    img.data[4 * i] = v;
    img.data[4 * i + 1] = v;
    img.data[4 * i + 2] = v;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Optics as the form edits them: strings, emission in nm. */
const toForm = (o) => ({
  ...Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)])),
  emission_nm: String(Math.round(o.emission_um * 1000)),
});
const fromForm = (f) => ({
  pixel_um: Number(f.pixel_um),
  z_step_um: Number(f.z_step_um),
  na: Number(f.na),
  emission_um: Number(f.emission_nm) / 1000,
  immersion_n: Number(f.immersion_n),
  modality: f.modality,
  sheet_um: Number(f.sheet_um),
});

const Field = ({ label, value, onChange, unit, ...props }) => (
  <TextField
    size="small"
    label={unit ? `${label} (${unit})` : label}
    value={value}
    onChange={(e) => onChange(e.target.value)}
    inputProps={{ inputMode: "decimal" }}
    sx={{ flex: 1, minWidth: 0 }}
    {...props}
  />
);

const Section = ({ children }) => (
  <Typography variant="overline" color="text.secondary" sx={{ lineHeight: 2, mt: 0.5 }}>
    {children}
  </Typography>
);

const DeconvolutionPanel = ({ path, image, t, z, getCrop, cropSize, setCropSize, onOpen, onClose }) => {
  const [defaults, setDefaults] = useState(null);
  const [error, setError] = useState(null);
  const [channel, setChannel] = useState(0);
  const [optics, setOptics] = useState([]);
  const [params, setParams] = useState({ method: "map", iterations: "30", regularization: 1e-4, background: "" });
  const [planes, setPlanes] = useState(32);
  const [preview, setPreview] = useState(null);
  const [showAfter, setShowAfter] = useState(true);
  const [previewing, setPreviewing] = useState(false);
  const [job, setJob] = useState(null);
  const [runner, setRunner] = useState("auto");
  const [scope, setScope] = useState("all");
  const [workerUrl, setWorkerUrl] = useState("");
  const [worker, setWorker] = useState(null);
  const xyRef = useRef(null);
  const xzRef = useRef(null);

  useEffect(() => {
    let live = true;
    setDefaults(null);
    setPreview(null);
    setError(null);
    apiProcessingGetDefaults(path, image)
      .then((d) => {
        if (!live) return;
        setDefaults(d);
        setOptics(d.optics.map(toForm));
        setParams({ ...d.parameters, iterations: String(d.parameters.iterations), background: "" });
        setPlanes(Math.min(32, d.shape.z));
        setChannel((c) => (c < d.shape.c ? c : 0));
      })
      .catch((e) => {
        if (!live) return;
        const missing = e?.response?.status === 404 && e?.response?.data?.detail === "Not Found";
        setError(missing ? "This ImSwitch has no deconvolution (ProcessingController)." : message(e));
      });
    return () => {
      live = false;
    };
  }, [path, image]);

  useEffect(() => {
    apiProcessingGetWorker()
      .then((w) => {
        setWorker(w);
        setWorkerUrl(w.url);
      })
      .catch(() => {});
    apiProcessingGetStatus()
      .then(setJob)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (job?.state !== "running") return undefined;
    const id = setInterval(() => apiProcessingGetStatus().then(setJob).catch(() => {}), 1000);
    return () => clearInterval(id);
  }, [job?.state]);

  useEffect(() => {
    if (!preview) return;
    const { crop } = preview;
    drawGray(xyRef.current, showAfter ? preview.afterXY : preview.beforeXY, crop.width, crop.height);
    drawGray(xzRef.current, showAfter ? preview.afterXZ : preview.beforeXZ, crop.width, crop.planes);
  }, [preview, showAfter]);

  const header = (
    <Box sx={{ display: "flex", alignItems: "center" }}>
      <Typography variant="subtitle2" sx={{ flex: 1 }}>
        Deconvolve
      </Typography>
      <IconButton size="small" onClick={onClose} aria-label="Close deconvolution">
        <Close fontSize="small" />
      </IconButton>
    </Box>
  );
  if (!defaults) {
    return (
      <Box sx={{ width: 380, flexShrink: 0, borderLeft: 1, borderColor: "divider", p: 1.5 }}>
        {header}
        {error ? <Alert severity="warning">{error}</Alert> : <LinearProgress />}
      </Box>
    );
  }

  const form = optics[channel] || optics[0];
  const setShared = (key, value) => setOptics((list) => list.map((o) => ({ ...o, [key]: value })));
  const setOwn = (key, value) => setOptics((list) => list.map((o, i) => (i === channel ? { ...o, [key]: value } : o)));
  const parameters = () => ({
    ...params,
    iterations: Number(params.iterations),
    background: params.background === "" ? null : Number(params.background),
  });
  const { shape, opticsFrom } = defaults;
  const voxels = shape.z * shape.y * shape.x;
  const maxPlanes = (size) => Math.floor(defaults.previewMaxVoxels / (size * size));
  const planeOptions = [...new Set(PLANES.map((p) => Math.min(p, shape.z)))].filter((p) => p <= maxPlanes(cropSize));
  const onPi = runner === "pi" || (runner === "auto" && !worker?.url);

  const runPreview = async () => {
    const crop = getCrop();
    if (!crop) return;
    setPreviewing(true);
    setError(null);
    try {
      const r = await apiProcessingPreview({
        path,
        image,
        t,
        c: channel,
        z,
        y: crop.cy,
        x: crop.cx,
        size: cropSize,
        planes,
        optics: fromForm(form),
        parameters: parameters(),
      });
      if (!r.cancelled) {
        setPreview({
          ...r,
          aspect: Number(form.z_step_um) / Number(form.pixel_um),
          beforeXY: decode(r.before.xy),
          afterXY: decode(r.after.xy),
          beforeXZ: decode(r.before.xz),
          afterXZ: decode(r.after.xz),
        });
      }
    } catch (e) {
      setError(message(e));
    } finally {
      setPreviewing(false);
    }
  };

  const startJob = async () => {
    setError(null);
    try {
      setJob(
        await apiProcessingStartJob({
          path,
          image,
          channels: scope === "all" ? [] : [channel],
          optics: optics.map(fromForm),
          parameters: parameters(),
          runner,
        }),
      );
    } catch (e) {
      setError(message(e));
    }
  };

  const saveWorker = () =>
    apiProcessingSetWorker(workerUrl.trim())
      .then(setWorker)
      .catch((e) => setError(message(e)));

  return (
    <Box
      sx={{
        width: 380,
        flexShrink: 0,
        borderLeft: 1,
        borderColor: "divider",
        overflow: "auto",
        p: 1.5,
        display: "flex",
        flexDirection: "column",
        gap: 1,
      }}
    >
      {header}
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {shape.c > 1 && (
        <TextField select size="small" label="Channel" value={channel} onChange={(e) => setChannel(e.target.value)}>
          {defaults.channels.map((name, i) => (
            <MenuItem key={i} value={i} dense>
              {name}
            </MenuItem>
          ))}
        </TextField>
      )}

      <Section>Optics</Section>
      <Box sx={{ display: "flex", gap: 1 }}>
        <Field label="NA" value={form.na} onChange={(v) => setShared("na", v)} />
        <Field label="Immersion n" value={form.immersion_n} onChange={(v) => setShared("immersion_n", v)} />
        <Field label="Emission" unit="nm" value={form.emission_nm} onChange={(v) => setOwn("emission_nm", v)} />
      </Box>
      <Box sx={{ display: "flex", gap: 1 }}>
        <Field label="Pixel" unit="µm" value={form.pixel_um} onChange={(v) => setShared("pixel_um", v)} />
        <Field label="Z step" unit="µm" value={form.z_step_um} onChange={(v) => setShared("z_step_um", v)} />
      </Box>
      <Box sx={{ display: "flex", gap: 1 }}>
        <TextField
          select
          size="small"
          label="Illumination"
          value={form.modality}
          onChange={(e) => setShared("modality", e.target.value)}
          sx={{ flex: 1 }}
        >
          <MenuItem value="widefield" dense>
            Wide field
          </MenuItem>
          <MenuItem value="lightsheet" dense>
            Light sheet
          </MenuItem>
        </TextField>
        {form.modality === "lightsheet" && (
          <Field label="Sheet" unit="µm" value={form.sheet_um} onChange={(v) => setShared("sheet_um", v)} />
        )}
      </Box>
      <Typography variant="caption" color={opticsFrom.pixel_um === "unknown" ? "warning.main" : "text.secondary"}>
        NA from the {opticsFrom.na}; pixel size and z step{" "}
        {opticsFrom.pixel_um === "unknown" ? "unknown, enter them" : "from the dataset"}; emission guessed from the
        channel name. The PSF is computed from these: check them.
      </Typography>

      <Section>Settings</Section>
      <Box sx={{ display: "flex", gap: 1 }}>
        <TextField
          select
          size="small"
          label="Method"
          value={params.method}
          onChange={(e) => setParams((p) => ({ ...p, method: e.target.value }))}
          sx={{ flex: 2 }}
        >
          <MenuItem value="map" dense>
            Regularised (MAP)
          </MenuItem>
          <MenuItem value="rl" dense>
            Richardson–Lucy (fast)
          </MenuItem>
        </TextField>
        <Field label="Iterations" value={params.iterations} onChange={(v) => setParams((p) => ({ ...p, iterations: v }))} />
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, opacity: params.method === "map" ? 1 : 0.4 }}>
        <Typography variant="body2" sx={{ width: 100 }}>
          Regularisation
        </Typography>
        <Slider
          size="small"
          min={-6}
          max={-1}
          step={0.25}
          value={Math.log10(params.regularization)}
          disabled={params.method !== "map"}
          onChange={(_, v) => setParams((p) => ({ ...p, regularization: 10 ** v }))}
          aria-label="Regularisation weight"
        />
        <Typography variant="caption" noWrap sx={{ width: 64, flexShrink: 0, fontFamily: "monospace", textAlign: "right" }}>
          {params.regularization.toExponential(1)}
        </Typography>
      </Box>
      <Field
        label="Background"
        unit="counts"
        value={params.background}
        placeholder="auto"
        onChange={(v) => setParams((p) => ({ ...p, background: v }))}
        helperText={preview ? `Auto from the crop: ${preview.background.toFixed(1)}` : "Empty: estimated from the data"}
      />

      <Section>Preview on the Pi</Section>
      <Box sx={{ display: "flex", gap: 1 }}>
        <TextField
          select
          size="small"
          label="Crop (px)"
          value={cropSize}
          onChange={(e) => {
            setCropSize(e.target.value);
            setPlanes((p) => Math.min(p, maxPlanes(e.target.value)));
          }}
          sx={{ flex: 1 }}
        >
          {[128, 256, 512].map((s) => (
            <MenuItem key={s} value={s} dense>
              {s} × {s}
            </MenuItem>
          ))}
        </TextField>
        <TextField
          select
          size="small"
          label="Planes"
          value={planes}
          onChange={(e) => setPlanes(e.target.value)}
          sx={{ flex: 1 }}
        >
          {planeOptions.map((p) => (
            <MenuItem key={p} value={p} dense>
              {p}
            </MenuItem>
          ))}
        </TextField>
        <Button variant="contained" disableElevation onClick={runPreview} disabled={previewing} sx={{ flex: 1 }}>
          Preview
        </Button>
      </Box>
      <Typography variant="caption" color="text.secondary">
        The crop is the dashed box in the middle of the view, centred on plane {z + 1}; drag the image to move it.
      </Typography>
      {previewing && <LinearProgress />}
      {preview && (
        <>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <ToggleButtonGroup
              size="small"
              exclusive
              value={showAfter ? "after" : "before"}
              onChange={(_, v) => v && setShowAfter(v === "after")}
            >
              <ToggleButton value="before">Before</ToggleButton>
              <ToggleButton value="after">After</ToggleButton>
            </ToggleButtonGroup>
            <Typography variant="caption" color="text.secondary" sx={{ fontFamily: "monospace" }}>
              {preview.crop.width} × {preview.crop.height} × {preview.crop.planes} · {preview.seconds} s
            </Typography>
          </Box>
          <canvas ref={xyRef} style={{ width: "100%", imageRendering: "pixelated", background: "#000" }} />
          <Typography variant="caption" color="text.secondary">
            xz section through the middle row
          </Typography>
          <canvas
            ref={xzRef}
            style={{
              width: "100%",
              maxHeight: 220,
              aspectRatio: `${preview.crop.width} / ${preview.crop.planes * preview.aspect}`,
              imageRendering: "pixelated",
              background: "#000",
            }}
          />
          <Typography variant="caption" color="text.secondary">
            Wide-field stacks keep their missing cone: haze goes and detail sharpens, the stack does not become
            confocal. Intensities are no longer linear in the sample; the raw data stay as they are.
          </Typography>
        </>
      )}

      <Divider sx={{ my: 0.5 }} />
      <Section>Whole stack</Section>
      <Box sx={{ display: "flex", gap: 1 }}>
        <TextField select size="small" label="Runs on" value={runner} onChange={(e) => setRunner(e.target.value)} sx={{ flex: 1 }}>
          <MenuItem value="auto" dense>
            Worker if registered
          </MenuItem>
          <MenuItem value="worker" dense>
            Worker
          </MenuItem>
          <MenuItem value="pi" dense>
            This Pi
          </MenuItem>
        </TextField>
        {shape.c > 1 && (
          <TextField select size="small" label="Channels" value={scope} onChange={(e) => setScope(e.target.value)} sx={{ flex: 1 }}>
            <MenuItem value="all" dense>
              All {shape.c}
            </MenuItem>
            <MenuItem value="one" dense>
              {defaults.channels[channel]} only
            </MenuItem>
          </TextField>
        )}
      </Box>
      <Box sx={{ display: "flex", gap: 1, alignItems: "flex-start" }}>
        <Field
          label="Worker URL"
          value={workerUrl}
          placeholder="http://lab-pc:8090"
          onChange={setWorkerUrl}
          helperText={worker?.url ? (worker.ok ? "Reachable" : `Not reachable: ${worker.message}`) : "None registered"}
          FormHelperTextProps={{ sx: { color: worker?.url && !worker.ok ? "warning.main" : undefined } }}
        />
        <Button onClick={saveWorker} sx={{ mt: 0.25 }}>
          Save
        </Button>
      </Box>
      {onPi && voxels > defaults.piMaxVoxels && (
        <Alert severity="info">
          One volume has {(voxels / 1e6).toFixed(0)} M voxels; the Pi takes {(defaults.piMaxVoxels / 1e6).toFixed(0)} M
          at most. Register a worker (docker/processing-worker), or crop the stack.
        </Alert>
      )}
      <Button variant="outlined" onClick={startJob} disabled={job?.state === "running"}>
        Deconvolve {scope === "all" ? shape.c : 1} channel{scope === "all" && shape.c > 1 ? "s" : ""}
        {shape.t > 1 ? ` × ${shape.t} time points` : ""}
      </Button>
      {job && job.state !== "idle" && (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
          <Typography variant="body2">
            {job.state === "running" ? "Running" : job.state[0].toUpperCase() + job.state.slice(1)} on{" "}
            {job.runner === "worker" ? "the worker" : "the Pi"} · volume {job.volume} / {job.volumes}
            {job.seconds != null ? ` · ${Math.round(job.seconds)} s` : ""}
          </Typography>
          <Typography variant="caption" color="text.secondary" noWrap title={job.output}>
            {job.output}
          </Typography>
          {job.state === "running" && (
            <>
              <LinearProgress variant="determinate" value={100 * job.progress} />
              <Button size="small" onClick={() => apiProcessingCancelJob().then(setJob).catch((e) => setError(message(e)))}>
                Cancel
              </Button>
            </>
          )}
          {job.state === "done" && (
            <Button size="small" variant="contained" disableElevation onClick={() => onOpen(job.output)}>
              Open result
            </Button>
          )}
          {job.state === "failed" && <Alert severity="error">{job.error}</Alert>}
        </Box>
      )}
    </Box>
  );
};

export default DeconvolutionPanel;
