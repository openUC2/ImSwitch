// src/components/holo/HoloSnapshot.js
// Reconstruct ONE still hologram in the browser: snap it from the camera or
// upload a PNG/JPG, click the region to reconstruct, drag dz. Works for in-line
// (back-propagation) and off-axis (sideband demodulation) holograms and needs
// no running hologram processing on the server.
import React, { useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  FormControlLabel,
  Grid,
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
import {
  PhotoCamera as PhotoCameraIcon,
  UploadFile as UploadFileIcon,
  AutoFixHigh as AutoFixHighIcon,
} from "@mui/icons-material";

import HoloCanvas from "./HoloCanvas";
import FreeNumberField from "../FreeNumberField";
import {
  amplitude,
  demodulateOffAxis,
  findSideband,
  flatField,
  logSpectrum,
  phase,
  reconstructInline,
} from "./holoMath";
import { useT } from "../../i18n";

const um = 1e-6;

// Decode an image Blob into { w, h, rgba }.
async function decode(blob) {
  const bmp = await createImageBitmap(blob);
  const c = document.createElement("canvas");
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext("2d");
  ctx.drawImage(bmp, 0, 0);
  return { w: bmp.width, h: bmp.height, rgba: ctx.getImageData(0, 0, bmp.width, bmp.height).data };
}

const CHANNELS = { red: [1, 0, 0], green: [0, 1, 0], blue: [0, 0, 1], mean: [1 / 3, 1 / 3, 1 / 3] };

function toGray({ w, h, rgba }, channel) {
  const [r, g, b] = CHANNELS[channel];
  const out = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = r * rgba[i * 4] + g * rgba[i * 4 + 1] + b * rgba[i * 4 + 2];
  return out;
}

const HoloSnapshot = ({
  mode = "inline",
  baseUrl,
  detectorName,
  defaultWavelength = 520e-9,
  defaultPixelsize = 1.4e-6,
  defaultDz = 0,
}) => {
  const t = useT();
  const fileRef = useRef(null);
  const [src, setSrc] = useState(null); // { w, h, rgba, name }
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [channel, setChannel] = useState("mean");
  const [roiSize, setRoiSize] = useState(512);
  const [center, setCenter] = useState(null); // full-image pixels
  const [flat, setFlat] = useState(true);
  const [wavelength, setWavelength] = useState(defaultWavelength);
  const [pixelsize, setPixelsize] = useState(defaultPixelsize);
  const [dzUm, setDzUm] = useState(Math.round(defaultDz / um));
  const [dzMaxUm, setDzMaxUm] = useState(5000);
  const [view, setView] = useState("amplitude");
  const [sideband, setSideband] = useState(null); // {cx, cy, r} in ROI spectrum pixels

  const load = async (blob, name) => {
    setBusy(true);
    setError(null);
    try {
      const img = await decode(blob);
      setSrc({ ...img, name });
      setCenter({ x: img.w / 2, y: img.h / 2 });
      setSideband(null);
      const fit = [1024, 512, 256, 128].find((s) => s <= Math.min(img.w, img.h)) || 128;
      setRoiSize((s) => Math.min(s, fit));
    } catch (e) {
      setError(t("Could not read this image ({msg}). Use PNG or JPG.", { msg: e.message }));
    } finally {
      setBusy(false);
    }
  };

  const snap = async () => {
    const q = detectorName ? `?detectorName=${encodeURIComponent(detectorName)}` : "";
    setBusy(true);
    try {
      const r = await fetch(`${baseUrl}/RecordingController/snapNumpyToFastAPI${q}`);
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      await load(await r.blob(), `snap_${new Date().toISOString().replace(/[:.]/g, "-")}`);
    } catch (e) {
      setError(t("Snap failed: {msg}", { msg: e.message }));
      setBusy(false);
    }
  };

  const gray = useMemo(() => src && toGray(src, channel), [src, channel]);
  const maxRoi = src ? Math.min(src.w, src.h) : 1024;

  // Square power-of-two ROI around the clicked centre, clamped to the image.
  const roi = useMemo(() => {
    if (!gray || !center) return null;
    const n = roiSize;
    const x0 = Math.round(Math.min(Math.max(center.x - n / 2, 0), src.w - n));
    const y0 = Math.round(Math.min(Math.max(center.y - n / 2, 0), src.h - n));
    const out = new Float64Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) out[y * n + x] = gray[(y0 + y) * src.w + x0 + x];
    return { n, x0, y0, data: flat ? flatField(out, n, n) : out };
  }, [gray, center, roiSize, flat, src]);

  const spectrum = useMemo(() => mode === "offaxis" && roi && logSpectrum(roi.data, roi.n, roi.n), [mode, roi]);
  const sb = useMemo(() => {
    if (mode !== "offaxis" || !roi) return null;
    return sideband || findSideband(roi.data, roi.n, roi.n);
  }, [mode, roi, sideband]);

  const recon = useMemo(() => {
    if (!roi) return null;
    const n = roi.n;
    const f = mode === "offaxis"
      ? demodulateOffAxis(roi.data, n, n, sb, dzUm * um, wavelength, pixelsize)
      : reconstructInline(roi.data, n, n, dzUm * um, wavelength, pixelsize);
    return { amp: amplitude(f.re, f.im), ph: phase(f.re, f.im) };
  }, [roi, mode, sb, dzUm, wavelength, pixelsize]);

  const base = src ? src.name.replace(/\.[^.]+$/, "") : "hologram";

  return (
    <Box>
      <Paper variant="outlined" sx={{ p: 1.5, mb: 1 }}>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
          <Button variant="contained" startIcon={<PhotoCameraIcon />} onClick={snap} disabled={busy || !baseUrl}>
            {t("Snap from camera")}
          </Button>
          <Button variant="outlined" startIcon={<UploadFileIcon />} onClick={() => fileRef.current?.click()} disabled={busy}>
            {t("Upload image")}
          </Button>
          <input
            ref={fileRef}
            type="file"
            accept="image/png,image/jpeg,image/bmp,image/webp"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) load(f, f.name);
              e.target.value = "";
            }}
          />
          {busy && <CircularProgress size={20} />}
          <Typography variant="body2" color="text.secondary">
            {src
              ? t("{name} · {w} × {h} px · click the image to move the square", { name: src.name, w: src.w, h: src.h })
              : t("Snap a frame or upload a PNG/JPG hologram. The reconstruction runs in this browser.")}
          </Typography>
        </Stack>
        {error && <Alert severity="error" sx={{ mt: 1 }} onClose={() => setError(null)}>{error}</Alert>}
      </Paper>

      {src && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper variant="outlined" sx={{ p: 1.5 }}>
              <Stack spacing={1.5}>
                <TextField select size="small" label={t("Colour channel")} value={channel} onChange={(e) => setChannel(e.target.value)}>
                  <MenuItem value="mean">{t("Mean of R, G, B")}</MenuItem>
                  <MenuItem value="red">{t("Red")}</MenuItem>
                  <MenuItem value="green">{t("Green")}</MenuItem>
                  <MenuItem value="blue">{t("Blue")}</MenuItem>
                </TextField>
                <TextField select size="small" label={t("Square size (px)")} value={roiSize} onChange={(e) => setRoiSize(e.target.value)}>
                  {[128, 256, 512, 1024].filter((s) => s <= maxRoi).map((s) => (
                    <MenuItem key={s} value={s}>{s} × {s}</MenuItem>
                  ))}
                </TextField>
                <FreeNumberField label={t("Wavelength (nm)")} value={wavelength} unitFactor={1e-9} fixedDecimals={0}
                  onCommit={(v) => v > 0 && setWavelength(v)} size="small" />
                <FreeNumberField label={t("Pixel size (µm)")} value={pixelsize} unitFactor={1e-6} fixedDecimals={3}
                  onCommit={(v) => v > 0 && setPixelsize(v)} size="small" />
                <FormControlLabel
                  control={<Switch size="small" checked={flat} onChange={(e) => setFlat(e.target.checked)} />}
                  label={<Typography variant="body2">{t("Divide by smooth background")}</Typography>}
                />
                {mode === "offaxis" && (
                  <>
                    <Button size="small" variant="outlined" startIcon={<AutoFixHighIcon />} onClick={() => setSideband(null)}>
                      {t("Find sideband")}
                    </Button>
                    {sb && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          {t("Crop radius")}: <strong>{sb.r} px</strong>
                        </Typography>
                        <Slider size="small" value={sb.r} min={3} max={Math.round(roi.n / 4)}
                          onChange={(e, r) => setSideband({ ...sb, r })} />
                      </Box>
                    )}
                  </>
                )}
              </Stack>
            </Paper>
          </Grid>

          <Grid item xs={12} md={9}>
            <Grid container spacing={1}>
              <Grid item xs={12} md={mode === "offaxis" ? 4 : 6}>
                <HoloCanvas
                  title={t("Full image - click to choose the region")}
                  values={gray}
                  w={src.w}
                  h={src.h}
                  onPick={(x, y) => setCenter({ x, y })}
                  maxHeight={300}
                  overlay={roi && (
                    <rect x={roi.x0} y={roi.y0} width={roi.n} height={roi.n} fill="none" stroke="red" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
                  )}
                />
              </Grid>
              {mode === "offaxis" && (
                <Grid item xs={12} md={4}>
                  <HoloCanvas
                    title={t("Spectrum - click a sideband")}
                    values={spectrum}
                    w={roi.n}
                    h={roi.n}
                    maxHeight={300}
                    onPick={(x, y) => setSideband({ cx: x, cy: y, r: sb?.r || 20 })}
                    overlay={sb && (
                      <circle cx={sb.cx} cy={sb.cy} r={sb.r} fill="none" stroke="cyan" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
                    )}
                  />
                </Grid>
              )}
              <Grid item xs={12} md={mode === "offaxis" ? 4 : 6}>
                <HoloCanvas
                  title={view === "phase" ? t("Reconstructed phase") : t("Reconstructed amplitude")}
                  values={view === "phase" ? recon?.ph : recon?.amp}
                  w={roi?.n}
                  h={roi?.n}
                  colormap={view === "phase" ? "phase" : "gray"}
                  min={view === "phase" ? -Math.PI : undefined}
                  max={view === "phase" ? Math.PI : undefined}
                  downloadName={`${base}_${view}_dz${dzUm}um`}
                  maxHeight={300}
                />
              </Grid>
            </Grid>

            <Paper variant="outlined" sx={{ p: 1.5, mt: 1 }}>
              <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
                <Box sx={{ flex: 1, minWidth: 260 }}>
                  <Typography variant="caption" color="text.secondary">
                    {t("Distance dz")}: <strong>{dzUm} µm</strong>
                  </Typography>
                  <Slider size="small" value={dzUm} min={mode === "offaxis" ? -dzMaxUm : 0} max={dzMaxUm} step={5}
                    onChange={(e, v) => setDzUm(v)} />
                </Box>
                <FreeNumberField label={t("dz (µm)")} value={dzUm} fixedDecimals={0} onCommit={(v) => setDzUm(v)}
                  size="small" fullWidth={false} sx={{ width: 110 }} />
                <FreeNumberField label={t("Slider max (µm)")} value={dzMaxUm} fixedDecimals={0}
                  onCommit={(v) => v > 0 && setDzMaxUm(v)} size="small" fullWidth={false} sx={{ width: 130 }} />
                <ToggleButtonGroup size="small" exclusive value={view} onChange={(e, v) => v && setView(v)}>
                  <ToggleButton value="amplitude">{t("Amplitude")}</ToggleButton>
                  <ToggleButton value="phase">{t("Phase")}</ToggleButton>
                </ToggleButtonGroup>
              </Stack>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {mode === "offaxis"
                  ? t("The phase is shown wrapped to ±π. A sideband centre that is slightly off adds a tilt; click the peak more precisely or use Find sideband.")
                  : t("Drag dz until the rings collapse into sharp objects. dz is the sample-to-sensor distance (for a point source far away). Use the pixel size of the image as saved - a binned or resized image has bigger pixels.")}
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default HoloSnapshot;
