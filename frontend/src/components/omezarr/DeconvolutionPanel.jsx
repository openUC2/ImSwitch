/**
 * Deconvolution panel of the OME-Zarr viewer (docs/DECONVOLUTION.md).
 *
 * PSF: modelled from the optics (scalar, vectorial, Gaussian) or measured from a bead stack, shown
 * in xy and xz with its FWHM. Preview: the crop in the dashed box (the middle of the view) is
 * deconvolved on the Pi with the
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
  apiProcessingDistillPSF,
  apiProcessingGetDefaults,
  apiProcessingGetStatus,
  apiProcessingGetWorker,
  apiProcessingPreview,
  apiProcessingPreviewPSF,
  apiProcessingSetWorker,
  apiProcessingStartJob,
} from "../../backendapi/apiProcessingController";
import { autoWindow } from "./omeZarrModel";

const message = (e) => e?.response?.data?.detail || e?.message || String(e);
const decode = (b64) => new Float32Array(Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0)).buffer);
const PLANES = [8, 16, 32, 64];
const PSF_KEY = "imswitch.deconvolution.lastPsf"; // the last distilled PSF, per browser
const MODELS = [
  ["scalar", "Scalar (Fourier optics)"],
  ["vectorial", "Vectorial (Richards–Wolf)"],
  ["gaussian", "Gaussian (set widths)"],
  ["measured", "Measured (beads)"],
];
// the MAP solver's data term and penalty (deconvolution.py LOSSES, PENALTIES)
const LOSSES = [
  ["poisson", "Poisson (photon counts)"],
  ["anscombe", "Anscombe"],
  ["gauss", "Gauss (read noise)"],
];
const PENALTIES = [
  ["good", "Good's roughness"],
  ["hessian", "Hessian (edges, lines; slower)"],
];
const remembered = () => {
  try {
    return localStorage.getItem(PSF_KEY) || "";
  } catch {
    return "";
  }
};

/** A PSF plane scaled to its maximum; log spans four decades, so rings and the cone show. */
function drawPSF(canvas, data, w, h, log, stretch = 1) {
  if (!canvas || !data) return;
  let max = 0;
  for (let i = 0; i < data.length; i++) max = Math.max(max, data[i]);
  const grey = data.map((v) => {
    const r = max > 0 ? v / max : 0;
    return log ? 255 * (1 + Math.log10(Math.max(r, 1e-4)) / 4) : 255 * r;
  });
  paint(canvas, grey, w, h, stretch);
}

/**
 * Put grey values (0..255 per pixel of a w x h plane) on a canvas, rows stretched by `stretch`
 * (z step / pixel size for an xz section), so the canvas has the sample's aspect ratio.
 */
function paint(canvas, grey, w, h, stretch = 1) {
  const src = document.createElement("canvas");
  src.width = w;
  src.height = h;
  const img = src.getContext("2d").createImageData(w, h);
  for (let i = 0; i < grey.length; i++) {
    img.data[4 * i] = grey[i]; // the clamped array clips to 0..255
    img.data[4 * i + 1] = grey[i];
    img.data[4 * i + 2] = grey[i];
    img.data[4 * i + 3] = 255;
  }
  src.getContext("2d").putImageData(img, 0, 0);
  canvas.width = w;
  canvas.height = Math.max(1, Math.round(h * stretch));
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
}

/** Grey image of a float plane, 0.1 / 99.9 percentile window. */
function drawGray(canvas, data, w, h, stretch = 1) {
  if (!canvas || !data) return;
  const { start, end } = autoWindow(data);
  const scale = 255 / (end - start);
  paint(canvas, data.map((v) => (v - start) * scale), w, h, stretch);
}

/** Optics as the form edits them: strings ("" for unset), emission in nm. */
const toForm = (o) => ({
  ...Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v == null ? "" : String(v)])),
  emission_nm: String(Math.round(o.emission_um * 1000)),
});
const fromForm = (f) => ({
  psf_model: f.psf_model || "scalar",
  fwhm_xy_um: f.fwhm_xy_um === "" ? null : Number(f.fwhm_xy_um),
  fwhm_z_um: f.fwhm_z_um === "" ? null : Number(f.fwhm_z_um),
  psf_path: f.psf_path || "",
  pixel_um: Number(f.pixel_um),
  z_step_um: Number(f.z_step_um),
  na: Number(f.na),
  emission_um: Number(f.emission_nm) / 1000,
  immersion_n: Number(f.immersion_n),
  modality: f.modality,
  sheet_um: Number(f.sheet_um),
});

/** Axial FWHM (µm) the optics lead one to expect, as deconvolution.py's expected_fwhm. */
const axialFwhm = (o) => {
  const n = Number(o.immersion_n) || 1;
  const na = Math.min(Number(o.na) || 0, n);
  return (0.88 * (Number(o.emission_nm) / 1000)) / (n - Math.sqrt(n * n - na * na));
};

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
  const [params, setParams] = useState({
    method: "map",
    iterations: "30",
    regularization: 1e-4,
    loss: "poisson",
    penalty: "good",
    background: "",
  });
  const [planes, setPlanes] = useState(32);
  const [preview, setPreview] = useState(null);
  const [showAfter, setShowAfter] = useState(true);
  const [previewing, setPreviewing] = useState(false);
  const [job, setJob] = useState(null);
  const [runner, setRunner] = useState("auto");
  const [scope, setScope] = useState("all");
  const [workerUrl, setWorkerUrl] = useState("");
  const [worker, setWorker] = useState(null);
  const [psfView, setPsfView] = useState(null);
  const [psfError, setPsfError] = useState(null);
  const [psfLog, setPsfLog] = useState(true);
  const [beadWindow, setBeadWindow] = useState("");
  const [distilling, setDistilling] = useState(false);
  const [distilled, setDistilled] = useState(null);
  const [lastPsf, setLastPsf] = useState(remembered);
  const xyRef = useRef(null);
  const xzRef = useRef(null);
  const psfXYRef = useRef(null);
  const psfXZRef = useRef(null);
  const psfRequest = useRef(0);

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

  // the PSF of the current channel's optics, shortly after they stop changing
  const current = optics[channel] || optics[0];
  const psfKey = current ? JSON.stringify(current) : "";
  useEffect(() => {
    if (!current || !defaults) return undefined;
    const id = setTimeout(() => {
      const n = ++psfRequest.current;
      apiProcessingPreviewPSF({ optics: fromForm(current), planes: Math.min(Math.max(defaults.shape.z, 8), 64) })
        .then((r) => {
          if (n !== psfRequest.current) return;
          setPsfView({ ...r, xyData: decode(r.xy), xzData: decode(r.xz) });
          setPsfError(null);
        })
        .catch((e) => {
          if (n !== psfRequest.current) return;
          setPsfView(null);
          setPsfError(message(e));
        });
    }, 400);
    return () => clearTimeout(id);
  }, [psfKey, defaults]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!psfView) return;
    drawPSF(psfXYRef.current, psfView.xyData, psfView.width, psfView.width, psfLog);
    drawPSF(psfXZRef.current, psfView.xzData, psfView.width, psfView.planes, psfLog, psfView.z_step_um / psfView.pixel_um);
  }, [psfView, psfLog]);

  useEffect(() => {
    if (!preview) return;
    const { crop } = preview;
    drawGray(xyRef.current, showAfter ? preview.afterXY : preview.beforeXY, crop.width, crop.height);
    drawGray(xzRef.current, showAfter ? preview.afterXZ : preview.beforeXZ, crop.width, crop.planes, preview.aspect);
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

  const setModel = (model) =>
    setOptics((list) =>
      list.map((o, i) =>
        i === channel ? { ...o, psf_model: model, psf_path: o.psf_path || (model === "measured" ? lastPsf : "") } : o,
      ),
    );

  const distill = async () => {
    setDistilling(true);
    setError(null);
    try {
      const r = await apiProcessingDistillPSF({
        path,
        image,
        t,
        c: channel,
        window_um: beadWindow === "" ? 0 : Number(beadWindow),
      });
      setDistilled(r);
      setLastPsf(r.path);
      try {
        localStorage.setItem(PSF_KEY, r.path);
      } catch {
        // the path stays for this page only
      }
      setOptics((list) => list.map((o, i) => (i === channel ? { ...o, psf_model: "measured", psf_path: r.path } : o)));
    } catch (e) {
      setError(message(e));
    } finally {
      setDistilling(false);
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

      <Section>PSF</Section>
      <TextField
        select
        size="small"
        label="Model"
        value={form.psf_model || "scalar"}
        onChange={(e) => setModel(e.target.value)}
      >
        {MODELS.filter(([m]) => (defaults.psfModels || ["scalar"]).includes(m)).map(([m, label]) => (
          <MenuItem key={m} value={m} dense>
            {label}
          </MenuItem>
        ))}
      </TextField>
      {form.psf_model === "gaussian" && (
        <Box sx={{ display: "flex", gap: 1 }}>
          <Field
            label="FWHM xy"
            unit="µm"
            value={form.fwhm_xy_um}
            placeholder={psfView ? String(psfView.expected_fwhm_um[0]) : "auto"}
            onChange={(v) => setOwn("fwhm_xy_um", v)}
          />
          <Field
            label="FWHM z"
            unit="µm"
            value={form.fwhm_z_um}
            placeholder={psfView ? String(psfView.expected_fwhm_um[1]) : "auto"}
            onChange={(v) => setOwn("fwhm_z_um", v)}
          />
        </Box>
      )}
      {form.psf_model === "measured" && (
        <>
          <Field
            label="PSF file"
            value={form.psf_path}
            placeholder="/beads_psf.ome.zarr"
            onChange={(v) => setOwn("psf_path", v)}
          />
          <Box sx={{ display: "flex", gap: 1, alignItems: "flex-start" }}>
            <Field label="Bead window" unit="µm" value={beadWindow} placeholder="auto" onChange={setBeadWindow} />
            <Button onClick={distill} disabled={distilling} sx={{ flex: 1.4, mt: 0.25 }}>
              Distill from this image
            </Button>
          </Box>
          {distilling && <LinearProgress />}
          {!distilled && (
            <Typography variant="caption" color="text.secondary">
              No bead stack yet? In Live View, "From template" copies the "Bead PSF stack" macro: run it on a sparse
              slide of sub-resolution beads, open the stack here and distill it.
              {Number.isFinite(axialFwhm(form)) &&
                ` For these optics: step ${(axialFwhm(form) / 3).toFixed(2)} µm or less, range about ${Math.ceil(
                  axialFwhm(form) * 6,
                )} µm.`}
            </Typography>
          )}
          {distilled && (
            <Typography variant="caption" color="text.secondary">
              Averaged {distilled.used} of {distilled.found} beads ({distilled.crowded} crowded, {distilled.edge} at
              the edge, {distilled.bright} clumps, {distilled.shallow} at the top or bottom of the stack). Saved as{" "}
              {distilled.path}.
            </Typography>
          )}
        </>
      )}
      {psfError && (
        <Typography variant="caption" color="warning.main">
          {psfError}
        </Typography>
      )}
      {psfView && (
        <>
          <Box sx={{ display: "flex", gap: 1, alignItems: "flex-start" }}>
            <canvas
              ref={psfXYRef}
              title="PSF in the focal plane"
              style={{ width: "50%", aspectRatio: "1 / 1", imageRendering: "pixelated", background: "#000" }}
            />
            <canvas
              ref={psfXZRef}
              title="PSF xz section through its centre"
              style={{ width: "50%", maxHeight: 260, objectFit: "contain", imageRendering: "pixelated", background: "#000" }}
            />
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
              xy at focus and xz, {(psfView.width * psfView.pixel_um).toFixed(1)} µm wide. FWHM {psfView.fwhm_xy_um} µm
              lateral, {psfView.fwhm_z_um} µm axial; these optics lead one to expect {psfView.expected_fwhm_um[0]} /{" "}
              {psfView.expected_fwhm_um[1]} µm.
            </Typography>
            <ToggleButton size="small" value="log" selected={psfLog} onChange={() => setPsfLog((v) => !v)}>
              Log
            </ToggleButton>
          </Box>
        </>
      )}

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
      <Box sx={{ display: "flex", gap: 1, opacity: params.method === "map" ? 1 : 0.4 }}>
        {[
          ["loss", "Noise model", LOSSES],
          ["penalty", "Penalty", PENALTIES],
        ].map(([key, label, options]) => (
          <TextField
            key={key}
            select
            size="small"
            label={label}
            value={params[key]}
            disabled={params.method !== "map"}
            onChange={(e) => setParams((p) => ({ ...p, [key]: e.target.value }))}
            sx={{ flex: 1, minWidth: 0 }}
          >
            {options.map(([value, text]) => (
              <MenuItem key={value} value={value} dense>
                {text}
              </MenuItem>
            ))}
          </TextField>
        ))}
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
            style={{ width: "100%", maxHeight: 220, objectFit: "contain", imageRendering: "pixelated", background: "#000" }}
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
