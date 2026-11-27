// src/components/holo/HoloSimulator.js
// Forward model + reconstruction for students, computed in the browser.
// In-line: point source -> sample -> primary + scattered wave -> hologram ->
// back-propagation (with its twin image). Off-axis: object wave + tilted
// reference -> fringes -> spectrum -> sideband -> amplitude and phase.
import React, { useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Grid,
  MenuItem,
  Paper,
  Slider,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";

import HoloCanvas from "./HoloCanvas";
import {
  amplitude,
  inlineGeometry,
  logSpectrum,
  makeSample,
  phase,
  reconstructInline,
  simulateInline,
  simulateOffAxis,
  demodulateOffAxis,
} from "./holoMath";
import { useT } from "../../i18n";

const N = 256;
const um = 1e-6;
const mm = 1e-3;

// "UC2" letters as a sample, drawn on an offscreen canvas.
function letterSample(n, absorption, phaseShift) {
  const amp = new Float64Array(n * n).fill(1);
  const ph = new Float64Array(n * n);
  const c = document.createElement("canvas");
  c.width = n;
  c.height = n;
  const ctx = c.getContext("2d");
  if (!ctx) return { amp, phase: ph };
  ctx.font = `bold ${Math.round(n / 4)}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("UC2", n / 2, n / 2);
  const a = ctx.getImageData(0, 0, n, n).data;
  for (let i = 0; i < n * n; i++) {
    const m = a[i * 4 + 3] / 255;
    amp[i] = 1 - absorption * m;
    ph[i] = phaseShift * m;
  }
  return { amp, phase: ph };
}

const Labeled = ({ label, value, children }) => (
  <Box>
    <Typography variant="caption" color="text.secondary">
      {label}: <strong>{value}</strong>
    </Typography>
    {children}
  </Box>
);

// Side view of the in-line geometry, drawn to scale along z.
function InlineGeometry({ z1, z2, sourceSize, t }) {
  const W = 560;
  const x0 = 30;
  const span = 480;
  const xs = x0 + (span * z1) / (z1 + z2);
  const xd = x0 + span;
  const cy = 70;
  const arcs = [];
  for (let r = 25; r < xs - x0 + 5; r += 35) {
    arcs.push(<path key={r} d={`M ${x0 + r * Math.cos(0.35)} ${cy - r * Math.sin(0.35)} A ${r} ${r} 0 0 1 ${x0 + r * Math.cos(0.35)} ${cy + r * Math.sin(0.35)}`} fill="none" stroke="#d32f2f" strokeOpacity="0.6" />);
  }
  return (
    <svg viewBox={`0 0 ${W + 70} 150`} width="100%" style={{ maxHeight: 170, color: "inherit" }}>
      <circle cx={x0} cy={cy} r={Math.max(2.5, Math.min(8, sourceSize / um / 20))} fill="#d32f2f" />
      <text x={x0} y={cy + 30} fontSize="11" textAnchor="middle" fill="currentColor">{t("source")}</text>
      {arcs}
      <line x1={x0} y1={cy} x2={xd} y2={cy - 55} stroke="#d32f2f" strokeOpacity="0.35" />
      <line x1={x0} y1={cy} x2={xd} y2={cy + 55} stroke="#d32f2f" strokeOpacity="0.35" />
      <line x1={xs} y1={cy - 45} x2={xs} y2={cy + 45} stroke="currentColor" strokeOpacity="0.5" strokeDasharray="3 3" />
      <circle cx={xs} cy={cy - 8} r="3" fill="currentColor" />
      <circle cx={xs} cy={cy + 12} r="2" fill="currentColor" />
      {[6, 12, 18].map((r) => (
        <path key={r} d={`M ${xs + r * 0.5} ${cy - 8 - r * 0.86} A ${r} ${r} 0 0 1 ${xs + r * 0.5} ${cy - 8 + r * 0.86}`} fill="none" stroke="#1976d2" />
      ))}
      <text x={xs - 6} y={cy - 50} fontSize="11" textAnchor="end" fill="currentColor">{t("sample")}</text>
      <rect x={xd - 3} y={cy - 55} width="6" height="110" fill="currentColor" fillOpacity="0.7" />
      <text x={xd} y={cy + 72} fontSize="11" textAnchor="middle" fill="currentColor">{t("sensor")}</text>
      <text x={(x0 + xs) / 2} y={18} fontSize="11" textAnchor="middle" fill="currentColor">z1 = {(z1 / mm).toFixed(1)} mm</text>
      <text x={xd + 8} y={cy + 4} fontSize="11" fill="currentColor">z2 = {(z2 / mm).toFixed(2)} mm</text>
      <text x={x0} y={135} fontSize="11" fill="#d32f2f">{t("primary wave")}</text>
      <text x={x0 + 100} y={135} fontSize="11" fill="#1976d2">{t("scattered wave")}</text>
    </svg>
  );
}

// Top view of the off-axis overlap: object beam straight, reference tilted.
function OffAxisGeometry({ thetaDeg, t }) {
  const cy = 75;
  const xd = 500;
  const a = (Math.max(thetaDeg, 0.5) * Math.PI) / 180;
  const shown = Math.min(Math.max(a * 4, 0.1), 0.28); // exaggerated for visibility
  return (
    <svg viewBox="0 0 560 150" width="100%" style={{ maxHeight: 170 }}>
      <line x1={20} y1={cy} x2={xd} y2={cy} stroke="#1976d2" strokeWidth="2" />
      <text x={30} y={cy - 8} fontSize="11" fill="#1976d2">{t("object wave (through the sample)")}</text>
      <rect x={180} y={cy - 12} width="8" height="24" fill="currentColor" fillOpacity="0.6" />
      <text x={184} y={cy + 30} fontSize="11" textAnchor="middle" fill="currentColor">{t("sample")}</text>
      <line x1={xd - 280 * Math.cos(shown)} y1={cy - 280 * Math.sin(shown)} x2={xd} y2={cy} stroke="#d32f2f" strokeWidth="2" />
      <text x={xd - 280 * Math.cos(shown)} y={cy - 280 * Math.sin(shown) + 14} fontSize="11" fill="#d32f2f">{t("reference wave")}</text>
      <rect x={xd} y={cy - 55} width="6" height="110" fill="currentColor" fillOpacity="0.7" />
      <text x={xd + 3} y={cy + 70} fontSize="11" textAnchor="middle" fill="currentColor">{t("sensor")}</text>
      <path d={`M ${xd - 70} ${cy} A 70 70 0 0 0 ${xd - 70 * Math.cos(shown)} ${cy - 70 * Math.sin(shown)}`} fill="none" stroke="currentColor" />
      <text x={xd - 80} y={cy + 20} fontSize="11" textAnchor="end" fill="currentColor">θ = {thetaDeg.toFixed(2)}° ({t("drawn larger")})</text>
    </svg>
  );
}

const HoloSimulator = ({ initialMode = "inline", defaultPixelUm = 1.4, defaultWavelengthNm = 520 }) => {
  const t = useT();
  const [mode, setMode] = useState(initialMode);
  const [kind, setKind] = useState(initialMode === "offaxis" ? "cells" : "dust");
  const [absorption, setAbsorption] = useState(0.8);
  const [phaseShift, setPhaseShift] = useState(0.5);
  const [lambdaNm, setLambdaNm] = useState(defaultWavelengthNm);
  const [pixelUm, setPixelUm] = useState(defaultPixelUm);
  // in-line
  const [z1Mm, setZ1Mm] = useState(30);
  const [z2Mm, setZ2Mm] = useState(1.0);
  const [sourceUm, setSourceUm] = useState(5);
  const [dzUm, setDzUm] = useState(0);
  const [reconView, setReconView] = useState("amplitude");
  // off-axis
  // Start at 4 px per fringe for this camera; the slider stops at 1.5 px.
  const degFor = (pxPerFringe, pUm, lNm) =>
    (Math.asin(Math.min(1, (lNm * 1e-3) / (pxPerFringe * pUm))) * 180) / Math.PI;
  const [thetaDeg, setThetaDeg] = useState(() =>
    Math.round(degFor(4, defaultPixelUm, defaultWavelengthNm) * 10) / 10);
  const [orientDeg, setOrientDeg] = useState(30);
  const [defocusUm, setDefocusUm] = useState(200);
  const [cropR, setCropR] = useState(30);
  const [sideband, setSideband] = useState(null); // {cx, cy} chosen by click
  const [oaDzUm, setOaDzUm] = useState(0);

  const lambda = lambdaNm * 1e-9;
  const pixel = pixelUm * um;
  const geo = inlineGeometry({ z1: z1Mm * mm, z2: z2Mm * mm, pixel });
  const dxSample = mode === "inline" ? geo.dxObj : pixel;

  const sample = useMemo(() => {
    const opts = { absorption, phaseShift, seed: 11 };
    return kind === "letters"
      ? letterSample(N, absorption, phaseShift)
      : makeSample(kind, N, dxSample, opts);
  }, [kind, absorption, phaseShift, dxSample]);

  // ---- in-line chain
  const inline = useMemo(() => {
    if (mode !== "inline") return null;
    return simulateInline({
      sample, n: N, lambda, z1: z1Mm * mm, z2: z2Mm * mm, pixel, sourceSize: sourceUm * um,
    });
  }, [mode, sample, lambda, z1Mm, z2Mm, pixel, sourceUm]);

  const inlineRecon = useMemo(() => {
    if (!inline) return null;
    const { re, im } = reconstructInline(inline.hologram, N, N, dzUm * um, lambda, inline.dxObj);
    return reconView === "phase" ? phase(re, im) : amplitude(re, im);
  }, [inline, dzUm, lambda, reconView]);

  // ---- off-axis chain
  const offaxis = useMemo(() => {
    if (mode !== "offaxis") return null;
    const th = (thetaDeg * Math.PI) / 180;
    const o = (orientDeg * Math.PI) / 180;
    return simulateOffAxis({
      sample, n: N, lambda, pixel, defocus: defocusUm * um,
      thetaX: Math.asin(Math.sin(th) * Math.cos(o)), thetaY: Math.asin(Math.sin(th) * Math.sin(o)),
    });
  }, [mode, sample, lambda, pixel, defocusUm, thetaDeg, orientDeg]);

  const spectrum = useMemo(() => offaxis && logSpectrum(offaxis.hologram, N, N), [offaxis]);
  const sb = sideband || (offaxis && offaxis.carrier);
  const oaRecon = useMemo(() => {
    if (!offaxis || !sb) return null;
    const f = demodulateOffAxis(offaxis.hologram, N, N, { cx: sb.cx, cy: sb.cy, r: cropR }, oaDzUm * um, lambda, pixel);
    return { amp: amplitude(f.re, f.im), ph: phase(f.re, f.im) };
  }, [offaxis, sb, cropR, oaDzUm, lambda, pixel]);

  const fovUm = (N * dxSample) / um;
  const sliderRow = (label, value, setter, min, max, step, fmt) => (
    <Labeled label={label} value={fmt ? fmt(value) : value}>
      <Slider size="small" value={value} min={min} max={max} step={step} onChange={(e, v) => setter(v)} />
    </Labeled>
  );

  return (
    <Box>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
        <ToggleButtonGroup size="small" exclusive value={mode} onChange={(e, v) => v && setMode(v)}>
          <ToggleButton value="inline">{t("In-line hologram")}</ToggleButton>
          <ToggleButton value="offaxis">{t("Off-axis hologram")}</ToggleButton>
        </ToggleButtonGroup>
        <Typography variant="body2" color="text.secondary">
          {t("Everything here is computed in your browser - no microscope needed.")}
        </Typography>
      </Stack>

      <Grid container spacing={2}>
        {/* ---------------- controls */}
        <Grid item xs={12} md={3}>
          <Paper variant="outlined" sx={{ p: 1.5 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2">{t("Sample")}</Typography>
              <TextField select size="small" label={t("Sample")} value={kind} onChange={(e) => setKind(e.target.value)}>
                <MenuItem value="point">{t("One point scatterer")}</MenuItem>
                <MenuItem value="dust">{t("Dust particles")}</MenuItem>
                <MenuItem value="cells">{t("Transparent cells")}</MenuItem>
                <MenuItem value="letters">{t("Letters (UC2)")}</MenuItem>
              </TextField>
              {sliderRow(t("Absorption"), absorption, setAbsorption, 0, 1, 0.05, (v) => v.toFixed(2))}
              {sliderRow(t("Phase delay"), phaseShift, setPhaseShift, 0, Math.PI, 0.05, (v) => `${v.toFixed(2)} rad`)}

              <Typography variant="subtitle2" sx={{ pt: 1 }}>{t("Light and camera")}</Typography>
              {sliderRow(t("Wavelength"), lambdaNm, setLambdaNm, 400, 700, 5, (v) => `${v} nm`)}
              {sliderRow(t("Pixel size"), pixelUm, setPixelUm, 0.8, 6, 0.05, (v) => `${v.toFixed(2)} µm`)}

              {mode === "inline" ? (
                <>
                  <Typography variant="subtitle2" sx={{ pt: 1 }}>{t("Geometry")}</Typography>
                  {sliderRow(t("Source to sample z1"), z1Mm, setZ1Mm, 2, 100, 1, (v) => `${v} mm`)}
                  {sliderRow(t("Sample to sensor z2"), z2Mm, setZ2Mm, 0.1, 5, 0.05, (v) => `${v.toFixed(2)} mm`)}
                  {sliderRow(t("Source size"), sourceUm, setSourceUm, 0, 300, 5, (v) => `${v} µm`)}
                  <Typography variant="caption" color="text.secondary">
                    {t("A fibre-coupled laser is a few µm wide, an LED chip 100-300 µm. A bigger source washes out the outer rings - that is spatial coherence.")}
                  </Typography>
                </>
              ) : (
                <>
                  <Typography variant="subtitle2" sx={{ pt: 1 }}>{t("Reference wave")}</Typography>
                  {sliderRow(t("Angle between the waves"), thetaDeg, setThetaDeg, 0,
                    Math.ceil(degFor(1.5, pixelUm, lambdaNm) * 10) / 10, 0.05, (v) => `${v.toFixed(2)}°`)}
                  {sliderRow(t("Fringe direction"), orientDeg, setOrientDeg, 0, 180, 5, (v) => `${v}°`)}
                  {sliderRow(t("Sample out of focus by"), defocusUm, setDefocusUm, -1000, 1000, 10, (v) => `${v} µm`)}
                  {sliderRow(t("Sideband crop radius"), cropR, setCropR, 4, 64, 1, (v) => `${v} px`)}
                </>
              )}
            </Stack>
          </Paper>
        </Grid>

        {/* ---------------- pictures */}
        <Grid item xs={12} md={9}>
          {mode === "inline" && inline && (
            <Stack spacing={1}>
              <Paper variant="outlined" sx={{ p: 1 }}>
                <InlineGeometry z1={z1Mm * mm} z2={z2Mm * mm} sourceSize={sourceUm * um} t={t} />
                <Typography variant="body2" color="text.secondary">
                  {t("Magnification M = {m} · the wave travels like a plane wave over z_eff = {zeff} µm · field of view {fov} µm",
                    { m: geo.M.toFixed(3), zeff: (geo.zEff / um).toFixed(0), fov: fovUm.toFixed(0) })}
                </Typography>
              </Paper>
              <Grid container spacing={1}>
                <Grid item xs={6} md={3}>
                  <HoloCanvas title={t("1 · Sample (transmission)")} values={sample.amp} w={N} h={N} min={0} max={1}
                    caption={t("What we want to see.")} maxHeight={220} />
                </Grid>
                <Grid item xs={6} md={3}>
                  <HoloCanvas title={t("2 · Scattered wave at the sensor")} values={inline.scattered} w={N} h={N} gamma={0.5}
                    caption={t("Only the light the sample scattered.")} maxHeight={220} />
                </Grid>
                <Grid item xs={6} md={3}>
                  <HoloCanvas title={t("3 · Hologram = |primary + scattered|²")} values={inline.hologram} w={N} h={N}
                    caption={t("What the camera records: rings, no phase.")} downloadName="simulated_inline_hologram" maxHeight={220} />
                </Grid>
                <Grid item xs={6} md={3}>
                  <HoloCanvas title={t("4 · Reconstruction at dz")} values={inlineRecon} w={N} h={N}
                    colormap={reconView === "phase" ? "phase" : "gray"}
                    caption={t("Back-propagated by the slider below.")} downloadName="simulated_inline_reconstruction" maxHeight={220} />
                </Grid>
              </Grid>
              <Paper variant="outlined" sx={{ p: 1.5 }}>
                <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
                  <Box sx={{ flex: 1, minWidth: 240 }}>
                    {sliderRow(t("Reconstruction distance dz"), dzUm, setDzUm, -2 * Math.round(geo.zEff / um), 2 * Math.round(geo.zEff / um), 1, (v) => `${v} µm`)}
                  </Box>
                  <Button size="small" variant="outlined" onClick={() => setDzUm(Math.round(geo.zEff / um))}>
                    {t("Jump to focus")}
                  </Button>
                  <ToggleButtonGroup size="small" exclusive value={reconView} onChange={(e, v) => v && setReconView(v)}>
                    <ToggleButton value="amplitude">{t("Amplitude")}</ToggleButton>
                    <ToggleButton value="phase">{t("Phase")}</ToggleButton>
                  </ToggleButtonGroup>
                </Stack>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {t("Drag dz: the rings collapse into the sample at dz = z_eff. The faint halo that stays is the twin image - the camera lost the phase, so the conjugate wave is reconstructed too. At dz = -z_eff you focus the twin instead.")}
                </Typography>
              </Paper>
            </Stack>
          )}

          {mode === "offaxis" && offaxis && (
            <Stack spacing={1}>
              <Paper variant="outlined" sx={{ p: 1 }}>
                <OffAxisGeometry thetaDeg={thetaDeg} t={t} />
                <Typography variant="body2" color="text.secondary">
                  {t("Fringe period {p} px. The camera needs at least 2 px per fringe (better 3-4); bigger angles alias.", { p: offaxis.period.toFixed(1) })}
                </Typography>
                {offaxis.period < 2.2 && (
                  <Alert severity="warning" sx={{ mt: 1 }}>
                    {t("The fringes are finer than the pixels can sample - reduce the angle.")}
                  </Alert>
                )}
              </Paper>
              <Grid container spacing={1}>
                <Grid item xs={6} md={4}>
                  {kind === "dust" ? (
                    <HoloCanvas title={t("1 · Sample (transmission)")} values={sample.amp} w={N} h={N} min={0} max={1}
                      caption={t("Dust only absorbs. Pick cells to see a phase object.")} maxHeight={240} />
                  ) : (
                    <HoloCanvas title={t("1 · Sample phase")} values={sample.phase} w={N} h={N} colormap="phase" min={-Math.PI} max={Math.PI}
                      caption={t("A transparent sample only delays the light.")} maxHeight={240} />
                  )}
                </Grid>
                <Grid item xs={6} md={4}>
                  <HoloCanvas title={t("2 · Hologram (zoom in on the fringes)")} values={offaxis.hologram} w={N} h={N}
                    downloadName="simulated_offaxis_hologram" caption={t("Straight fringes, bent where the sample delays the light.")} maxHeight={240} />
                </Grid>
                <Grid item xs={12} md={4}>
                  <HoloCanvas title={t("3 · Spectrum - click a sideband")} values={spectrum} w={N} h={N}
                    onPick={(x, y) => setSideband({ cx: x, cy: y })}
                    caption={t("Centre: the plain intensities. The two side peaks carry the object wave and its twin.")}
                    maxHeight={240}
                    overlay={sb && (
                      <circle cx={sb.cx} cy={sb.cy} r={cropR} fill="none" stroke="cyan" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
                    )} />
                </Grid>
                <Grid item xs={6} md={6}>
                  <HoloCanvas title={t("4 · Reconstructed amplitude")} values={oaRecon?.amp} w={N} h={N}
                    downloadName="simulated_offaxis_amplitude" maxHeight={260} />
                </Grid>
                <Grid item xs={6} md={6}>
                  <HoloCanvas title={t("5 · Reconstructed phase")} values={oaRecon?.ph} w={N} h={N} colormap="phase"
                    min={-Math.PI} max={Math.PI} downloadName="simulated_offaxis_phase" maxHeight={260} />
                </Grid>
              </Grid>
              <Paper variant="outlined" sx={{ p: 1.5 }}>
                <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
                  <Box sx={{ flex: 1, minWidth: 240 }}>
                    {sliderRow(t("Refocus dz"), oaDzUm, setOaDzUm, -1000, 1000, 10, (v) => `${v} µm`)}
                  </Box>
                  <Button size="small" variant="outlined" onClick={() => setSideband(null)}>
                    {t("Back to the true sideband")}
                  </Button>
                </Stack>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {t("One sideband gives amplitude AND phase from a single image - no twin image. Set dz to the defocus to refocus. The other sideband is the twin: its phase is inverted and it focuses at -dz. A crop that touches the centre lets the plain intensity leak in.")}
                </Typography>
              </Paper>
            </Stack>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default HoloSimulator;
