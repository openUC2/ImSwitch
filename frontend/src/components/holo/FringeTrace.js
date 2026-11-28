// src/components/holo/FringeTrace.js
// Interferometer drift: click a pixel in the live image, record its intensity
// on every camera frame (backend), count the fringes that pass and convert
// them to a path-length change. Export the trace as CSV or the plot as PNG.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  Typography,
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Download as DownloadIcon,
} from "@mui/icons-material";
import Plot from "react-plotly.js";

import LiveViewControlWrapper from "../../axon/LiveViewControlWrapper";
import FreeNumberField from "../FreeNumberField";
import {
  apiOffAxisHoloControllerStartFringeTrace,
  apiOffAxisHoloControllerStopFringeTrace,
  apiOffAxisHoloControllerGetFringeTrace,
} from "../../backendapi/apiOffAxisHoloController";
import { countFringes } from "./holoMath";
import { useT } from "../../i18n";

const POLL_MS = 500;

// Path change per fringe: Michelson light passes the moving mirror twice.
const PER_FRINGE = { michelson: 0.5, machzehnder: 1 };

const FringeTrace = ({ defaultWavelength = 635e-9 }) => {
  const t = useT();
  const [pick, setPick] = useState(null); // {fx, fy} fractions of the frame
  const [radius, setRadius] = useState(1);
  const [running, setRunning] = useState(false);
  const [trace, setTrace] = useState({ t: [], v: [], x: null, y: null });
  const [kind, setKind] = useState("michelson");
  const [wavelength, setWavelength] = useState(defaultWavelength);
  const [hysteresis, setHysteresis] = useState(0.3);
  const [error, setError] = useState(null);
  const nextRef = useRef(0);
  const busyRef = useRef(false); // one poll at a time, or samples arrive twice
  const genRef = useRef(0); // bumped on start: drop replies for the old trace

  const poll = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    const gen = genRef.current;
    try {
      const d = await apiOffAxisHoloControllerGetFringeTrace(nextRef.current);
      if (gen !== genRef.current) return;
      nextRef.current = d.next;
      setRunning(d.running);
      if (d.t.length) {
        setTrace((p) => ({ t: p.t.concat(d.t), v: p.v.concat(d.v), x: d.x, y: d.y }));
      }
    } catch (e) {
      setError(t("Reading the trace failed: {msg}", { msg: e.message }));
      setRunning(false);
    } finally {
      busyRef.current = false;
    }
  }, [t]);

  // A trace keeps running on the server while this tab is closed: pick it up.
  useEffect(() => {
    poll();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [running, poll]);

  const start = async () => {
    if (!pick) return;
    setError(null);
    try {
      await apiOffAxisHoloControllerStartFringeTrace(pick.fx, pick.fy, radius);
      genRef.current += 1;
      nextRef.current = 0;
      setTrace({ t: [], v: [], x: null, y: null });
      setRunning(true);
    } catch (e) {
      setError(t("Could not start the trace: {msg}", { msg: e.message }));
    }
  };

  const stop = async () => {
    await apiOffAxisHoloControllerStopFringeTrace().catch(() => {});
    while (busyRef.current) await new Promise((r) => setTimeout(r, 50));
    await poll(); // the samples recorded since the last poll
    setRunning(false);
  };

  const stats = useMemo(() => countFringes(trace.v, hysteresis), [trace.v, hysteresis]);
  const v0 = trace.v.length ? trace.v[0] : 0;
  const delta = useMemo(() => trace.v.map((v) => v - v0), [trace.v, v0]);
  const duration = trace.t.length ? trace.t[trace.t.length - 1] : 0;
  const rate = duration > 0 ? (trace.t.length - 1) / duration : 0;
  const perFringe = PER_FRINGE[kind];
  const displacementNm = stats.fringes * perFringe * wavelength * 1e9;

  const exportCsv = () => {
    const head = [
      `# pixel x=${trace.x} y=${trace.y} box=${2 * radius + 1}x${2 * radius + 1}`,
      `# wavelength_nm=${(wavelength * 1e9).toFixed(1)} interferometer=${kind} fringes=${stats.fringes} displacement_nm=${displacementNm.toFixed(1)}`,
      "time_s,intensity,delta_intensity",
    ];
    const rows = trace.t.map((s, i) => `${s},${trace.v[i]},${(trace.v[i] - v0).toFixed(3)}`);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([[...head, ...rows].join("\n")], { type: "text/csv" }));
    a.download = `fringe_trace_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const overlay = pick && (
    <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none"
      style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}>
      <line x1={pick.fx * 100 - 3} y1={pick.fy * 100} x2={pick.fx * 100 + 3} y2={pick.fy * 100} stroke="#ff1744" strokeWidth="0.4" />
      <line x1={pick.fx * 100} y1={pick.fy * 100 - 3} x2={pick.fx * 100} y2={pick.fy * 100 + 3} stroke="#ff1744" strokeWidth="0.4" />
    </svg>
  );

  return (
    <Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {t("Put the camera where the fringes are (instead of the screen). Click a pixel on a fringe flank, start, then warm one arm with your hand or push the table lightly: every fringe that passes the pixel is one bright-dark-bright cycle.")}
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={5}>
          <Box sx={{ position: "relative", bgcolor: "#000", borderRadius: 1, overflow: "hidden", minHeight: 200, lineHeight: 0 }}>
            <LiveViewControlWrapper
              onClick={(px, py, w, h) => setPick({ fx: px / w, fy: py / h })}
              overlayContent={overlay}
              enableStageMovement={false}
            />
          </Box>
          <Paper variant="outlined" sx={{ p: 1.5, mt: 1 }}>
            <Stack spacing={1.5}>
              <Typography variant="body2">
                {pick
                  ? t("Pixel at {x}% / {y}% of the frame", { x: (pick.fx * 100).toFixed(1), y: (pick.fy * 100).toFixed(1) })
                  : t("Click the live image to choose the pixel.")}
                {trace.x !== null && ` · ${t("sensor pixel")} (${trace.x}, ${trace.y})`}
              </Typography>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  {t("Average over")}: <strong>{2 * radius + 1} × {2 * radius + 1} px</strong>
                </Typography>
                <Slider size="small" value={radius} min={0} max={5} step={1} disabled={running}
                  onChange={(e, v) => setRadius(v)} />
              </Box>
              <Stack direction="row" spacing={1}>
                <Button variant="contained" startIcon={<PlayArrowIcon />} onClick={start} disabled={!pick || running}>
                  {t("Start trace")}
                </Button>
                <Button variant="contained" color="error" startIcon={<StopIcon />} onClick={stop} disabled={!running}>
                  {t("Stop")}
                </Button>
                <Button variant="outlined" startIcon={<DownloadIcon />} onClick={exportCsv} disabled={!trace.t.length}>
                  CSV
                </Button>
              </Stack>
              {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}
            </Stack>
          </Paper>
        </Grid>

        <Grid item xs={12} md={7}>
          <Paper variant="outlined" sx={{ p: 1 }}>
            <Plot
              data={[
                { x: trace.t, y: delta, type: "scattergl", mode: "lines", name: "ΔI", line: { width: 1, color: "#1976d2" } },
              ]}
              layout={{
                autosize: true,
                height: 320,
                margin: { l: 50, r: 10, t: 30, b: 40 },
                title: { text: t("Intensity change at the pixel"), font: { size: 13 } },
                xaxis: { title: { text: t("time (s)") } },
                yaxis: { title: { text: t("I − I(0) (counts)") } },
                shapes: trace.v.length ? [stats.mean + stats.threshold, stats.mean - stats.threshold].map((y) => ({
                  type: "line", xref: "paper", x0: 0, x1: 1, y0: y - v0, y1: y - v0,
                  line: { dash: "dot", width: 1, color: "#888" },
                })) : [],
                showlegend: false,
              }}
              config={{ displaylogo: false, responsive: true, toImageButtonOptions: { filename: "fringe_trace", scale: 2 } }}
              useResizeHandler
              style={{ width: "100%" }}
            />
            <Typography variant="caption" color="text.secondary">
              {t("Dotted lines: the counting thresholds. Export the plot with the camera icon above it.")}
            </Typography>
          </Paper>

          <Paper variant="outlined" sx={{ p: 1.5, mt: 1 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6}>
                <TextField select size="small" fullWidth label={t("Interferometer")} value={kind} onChange={(e) => setKind(e.target.value)}>
                  <MenuItem value="michelson">{t("Michelson (mirror moves λ/2 per fringe)")}</MenuItem>
                  <MenuItem value="machzehnder">{t("Mach-Zehnder (path changes λ per fringe)")}</MenuItem>
                </TextField>
              </Grid>
              <Grid item xs={6} sm={3}>
                <FreeNumberField label={t("Wavelength (nm)")} value={wavelength} unitFactor={1e-9} fixedDecimals={0}
                  onCommit={(v) => v > 0 && setWavelength(v)} size="small" />
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  {t("Hysteresis")}: <strong>{Math.round(hysteresis * 100)}%</strong>
                </Typography>
                <Slider size="small" value={hysteresis} min={0.1} max={0.9} step={0.05} onChange={(e, v) => setHysteresis(v)} />
              </Grid>
            </Grid>
            <Stack direction="row" spacing={4} sx={{ mt: 1 }} flexWrap="wrap" useFlexGap>
              <Box>
                <Typography variant="caption" color="text.secondary">{t("Fringes counted")}</Typography>
                <Typography variant="h5">{stats.fringes}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  {kind === "michelson" ? t("Mirror moved") : t("Path length changed")}
                </Typography>
                <Typography variant="h5">
                  {displacementNm >= 1000 ? `${(displacementNm / 1000).toFixed(2)} µm` : `${displacementNm.toFixed(0)} nm`}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">{t("Samples")}</Typography>
                <Typography variant="h6">{trace.t.length} · {rate.toFixed(1)} {t("per s")} · {duration.toFixed(1)} s</Typography>
              </Box>
            </Stack>
            <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 1 }}>
              {t("N fringes = N × {f} × λ. One pixel cannot tell the direction: a mirror that moves forth and back counts both ways. Fringes faster than ~1 per 20 frames are missed - slow down or shorten the exposure.",
                { f: perFringe === 0.5 ? "½" : "1" })}
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FringeTrace;
