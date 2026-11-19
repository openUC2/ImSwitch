// src/components/OPMController.jsx
// dvOPM app: digital twin of the rig (follows the live stage axes), launch optics trace,
// and a stage-scan volume acquisition like the LightSheet app. Drawings are ported from
// openuc2-opmsimulator/opm_ui.html; the geometry comes from the backend (OPMController).
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  FormControl,
  FormControlLabel,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Switch,
  Tab,
  Tabs,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";
import * as api from "../backendapi/apiOPMController.js";

const MEDIA = [
  ["Water", 1.333], ["OptiMuS", 1.47], ["Ce3D", 1.5], ["EasyIndex", 1.52], ["ECi", 1.558], ["DBE", 1.56],
];
// [group, key, label, unit, decimals]
const AXES = [
  ["sample", "X", "Sample X", "mm", 4], ["sample", "Y", "Sample Y", "mm", 4], ["sample", "Z", "Sample Z", "mm", 4],
  ["launch", "x", "Launch X", "mm", 4], ["launch", "y", "Launch Y", "mm", 4], ["launch", "z", "Launch Z", "mm", 4],
  ["launch", "phi_deg", "Launch tilt φ", "°", 3], ["launch", "focus", "Launch focus", "mm", 4],
];
const PART = {
  objective: "objective", cyl_mount: "cylinder-lens mount", collimator: "collimator", arm_tube: "arm tube",
  arm_tube2: "arm tube", rail_carriage: "rail carriage", holder_l: "holder plate (left)",
  holder_r: "holder plate (right)", dish_bottom: "dish bottom", dish_wall_l: "dish wall", dish_wall_r: "dish wall",
  det_lens: "telecentric lens", camera: "camera", sample_carriage: "sample stage", sample_riser: "sample riser",
};
const pn = (id) => PART[id] || id.replace(/_/g, " ");
const FONT = "500 12px system-ui, sans-serif";

const fmt = (v, d = 3) => {
  if (v == null || !Number.isFinite(+v)) return "–";
  const t = (+v).toFixed(d);
  return /^-0(\.0*)?$/.test(t) ? t.slice(1) : t;
};
const errMsg = (e) => e?.response?.data?.detail || e?.message || String(e);

function usePoll(fn, ms, enabled = true) {
  const busy = useRef(false);
  useEffect(() => {
    if (!enabled) return undefined;
    const tick = async () => {
      if (busy.current || document.hidden) return;
      busy.current = true;
      try { await fn(); } finally { busy.current = false; }
    };
    tick();
    const id = setInterval(tick, ms);
    return () => clearInterval(id);
  }, [fn, ms, enabled]);
}

function colours(theme) {
  const dark = theme.palette.mode === "dark";
  return {
    ink: theme.palette.text.primary, muted: theme.palette.text.secondary, panel: theme.palette.background.paper,
    line: theme.palette.divider, steel: dark ? "#58A9D1" : "#17587A", steelSoft: dark ? "#1C313C" : "#E1EDF3",
    glass: dark ? "#35566A" : "#CFE6F2", warn: dark ? "#FF7A6E" : "#B8322A", sheet: dark ? "#B5DB4E" : "#9BC437",
    sheetInk: dark ? "#C9E77A" : "#4F6E14", body: dark ? "#3A464D" : "#1E2429",
  };
}

function prep(cv) {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  const c = cv.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, r.width, r.height);
  return [c, r.width, r.height];
}

function waiting(c, k, text) {
  c.fillStyle = k.muted;
  c.font = FONT;
  c.fillText(text, 14, 22);
}

/* ------------------------------------------------------------------ rig drawing */
function drawRig(cv, d, view, pz, k) {
  const [c, W, H] = prep(cv);
  if (!d) return waiting(c, k, "Waiting for the rig geometry…");
  const ST = {
    detection: { fill: k.body, text: "#fff" }, detection_top: { stroke: k.muted, dash: [5, 4] },
    dish: { fill: k.glass, stroke: k.steel }, medium: { fill: "rgba(118,188,226,.35)" },
    sample: { fill: k.steelSoft, stroke: k.steel }, sample_stage: { fill: k.steel, text: "#fff" },
    stage_base: { stroke: k.muted, dash: [3, 3] }, launch: { fill: "#3E7163", text: "#fff" },
    launch_rail: { fill: "#8FA8A0" }, launch_stage: { fill: "#2F5A4E", text: "#fff" },
    rail: { fill: "#7FA83A", stroke: "#5E7F22" }, hole: { fill: k.panel, stroke: k.steel },
    fov: { fill: "rgba(23,88,122,.18)", stroke: k.steel }, beam_top: { fill: "rgba(216,50,42,.35)" },
    sheet_top: { fill: "rgba(155,196,55,.55)", stroke: "#5E7F22" },
  };
  const LS = {
    beam: [k.warn, 2], sheet: ["#6E9A1E", 2.6], det_plane: [k.steel, 1.2, [7, 5]],
    fov: ["rgba(23,88,122,.45)", 7], chief: [k.steel, 0.9, [2, 3]], fibre: [k.ink, 1.4],
    annot: [k.muted, 1, [4, 5]], detection_sensor: ["#E0B64A", 3.5],
  };
  const v = view === "top" ? d.top : d.side;
  const bb = view === "dish" ? d.side.dish_bbox : v.bbox;
  const pad = view === "dish" ? 0.04 : 0.06;
  const sc = Math.min(W / ((bb[2] - bb[0]) * (1 + 2 * pad)), H / ((bb[3] - bb[1]) * (1 + 2 * pad))) * pz.zoom;
  const cx = (bb[0] + bb[2]) / 2, cy = (bb[1] + bb[3]) / 2;
  const X = (x) => W / 2 + pz.x + (x - cx) * sc, Y = (y) => H / 2 + pz.y - (y - cy) * sc;
  const items = v.items;
  const polys = items.filter((i) => i.kind === "poly");
  const circles = items.filter((i) => i.kind === "circle");
  const order = ["stage_base", "medium", "dish", "sample", "sample_stage", "hole", "detection", "detection_top",
    "fov", "rail", "launch_rail", "launch_stage", "launch", "beam_top", "sheet_top"];
  const rank = (g) => (order.indexOf(g) < 0 ? 99 : order.indexOf(g));
  const path = (pts) => { c.beginPath(); pts.forEach((p, i) => (i ? c.lineTo(X(p[0]), Y(p[1])) : c.moveTo(X(p[0]), Y(p[1])))); };
  const drawPoly = (it) => {
    const st = ST[it.group] || { fill: k.line };
    path(it.pts); c.closePath();
    if (st.fill) { c.fillStyle = st.fill; c.fill(); }
    if (st.stroke) { c.strokeStyle = st.stroke; c.lineWidth = 1.2; c.setLineDash(st.dash || []); c.stroke(); c.setLineDash([]); }
    if (it.hit) { c.fillStyle = "rgba(216,50,42,.35)"; c.fill(); c.strokeStyle = k.warn; c.lineWidth = 2.4; c.stroke(); }
  };
  const drawCircle = (it) => {
    const st = ST[it.group] || { stroke: k.muted, dash: it.group === "annot" ? [4, 5] : [] };
    c.beginPath(); c.arc(X(it.pts[0][0]), Y(it.pts[0][1]), it.r * sc, 0, 2 * Math.PI);
    if (st.fill) { c.fillStyle = st.fill; c.fill(); }
    c.strokeStyle = st.stroke || k.muted; c.lineWidth = 1.2; c.setLineDash(st.dash || []); c.stroke(); c.setLineDash([]);
  };
  const holes = () => circles.filter((i) => i.group === "hole" || i.group === "dish").forEach(drawCircle);
  circles.filter((i) => i.group === "annot" || i.group === "detection_top").forEach(drawCircle);
  polys.slice().sort((a, b) => rank(a.group) - rank(b.group)).forEach((it) => { drawPoly(it); if (it.group === "sample") holes(); });
  if (!polys.some((p) => p.group === "sample")) holes();
  for (const it of items.filter((i) => i.kind === "line")) {
    const [s, w, dash] = LS[it.group] || [k.ink, 1];
    path(it.pts);
    c.strokeStyle = it.hit ? k.warn : s; c.lineWidth = it.hit ? Math.max(w, 2.4) : w;
    c.setLineDash(dash || []); c.stroke(); c.setLineDash([]);
  }
  for (const it of items.filter((i) => i.kind === "point")) {
    const [px, py] = [X(it.pts[0][0]), Y(it.pts[0][1])];
    c.beginPath(); c.arc(px, py, 4, 0, 2 * Math.PI); c.fillStyle = k.panel; c.fill();
    c.strokeStyle = it.group === "sheet" ? "#6E9A1E" : it.group === "pivot" ? k.ink : k.steel; c.lineWidth = 1.6; c.stroke();
    if (it.group === "pivot") { c.beginPath(); c.moveTo(px - 7, py); c.lineTo(px + 7, py); c.moveTo(px, py - 7); c.lineTo(px, py + 7); c.stroke(); }
  }
  rigLabels(c, items, ST, X, Y, W, H, k);
  scaleBar(c, sc, W, H, k, view === "top" ? "y" : "z");
}

function rigLabels(c, items, ST, X, Y, W, H, k) {
  c.font = FONT; c.textBaseline = "middle";
  const placed = [];
  const halo = (t, x, y, colr, al = "left") => {
    c.textAlign = al; c.lineWidth = 3; c.strokeStyle = k.panel; c.strokeText(t, x, y); c.fillStyle = colr; c.fillText(t, x, y);
  };
  for (const it of items.filter((i) => i.label)) {
    let px, py, wpx = 0, hpx = 0;
    if (it.kind === "circle") { px = X(it.pts[0][0]); py = Y(it.pts[0][1] - it.r) + 12; }
    else if (it.kind === "poly") {
      const xs = it.pts.map((p) => X(p[0])), ys = it.pts.map((p) => Y(p[1]));
      px = (Math.min(...xs) + Math.max(...xs)) / 2; py = (Math.min(...ys) + Math.max(...ys)) / 2;
      wpx = Math.max(...xs) - Math.min(...xs); hpx = Math.max(...ys) - Math.min(...ys);
    } else { const p = it.kind === "line" ? it.pts[it.pts.length - 1] : it.pts[0]; px = X(p[0]) + 8; py = Y(p[1]) - 9; }
    if (px < -50 || px > W + 50 || py < -20 || py > H + 20) continue;
    const st = ST[it.group] || {}, tw = c.measureText(it.label).width;
    if (it.kind === "poly" && st.text && tw + 8 < wpx && hpx > 14) { c.textAlign = "center"; c.fillStyle = st.text; c.fillText(it.label, px, py); continue; }
    if (placed.some((q) => Math.abs(q[0] - px) < tw * 0.6 && Math.abs(q[1] - py) < 13)) py += 14;
    placed.push([px, py]);
    if (it.kind === "poly") halo(it.label, px + wpx / 2 + 6, py, k.ink);
    else halo(it.label, px, py, it.group === "sheet" ? "#5E7F22" : it.group === "pivot" ? k.ink : k.steel, it.kind === "circle" ? "center" : "left");
  }
  c.textBaseline = "alphabetic";
}

function scaleBar(c, sc, W, H, k, vert) {
  const target = (W * 0.12) / sc, mag = Math.pow(10, Math.floor(Math.log10(target)));
  const nice = [10, 5, 2, 1].map((m) => m * mag).find((v) => v <= target) || mag;
  c.fillStyle = k.ink; c.fillRect(14, H - 16, nice * sc, 3); c.font = FONT; c.textAlign = "left";
  c.fillText(nice >= 1 ? `${nice} mm` : `${+(nice * 1000).toPrecision(3)} µm`, 14, H - 22);
  if (!vert) return;
  const ox = W - 58, oy = H - 18;
  c.strokeStyle = k.muted; c.lineWidth = 1.3; c.beginPath(); c.moveTo(ox, oy); c.lineTo(ox + 28, oy); c.moveTo(ox, oy); c.lineTo(ox, oy - 28); c.stroke();
  c.fillStyle = k.muted; c.fillText("x", ox + 32, oy + 4); c.fillText(vert, ox - 3, oy - 33);
}

/* ------------------------------------------------------------------ optics drawing */
function drawOptics(cv, d, view, k) {
  const [c, W, H] = prep(cv);
  if (!d) return waiting(c, k, "Waiting for the optics trace…");
  const L = d.lab, s = d.summary, F = L.F, fb = d.focus_parax || F;
  let x0, x1, z0, z1;
  if (view === "train") {
    const pts = [L.fiber, L.collimator, L.objective, F, [-8, L.zb], [8, L.top]];
    x0 = Math.min(...pts.map((p) => p[0])) - 6; x1 = Math.max(...pts.map((p) => p[0])) + 6;
    z0 = Math.min(...pts.map((p) => p[1])) - 6; z1 = Math.max(...pts.map((p) => p[1])) + 6;
    const padX = (x1 - x0) * 0.16; x0 -= padX * 0.3; x1 += padX;
  } else if (view === "dish") {
    const E = d.chief ? d.chief.E : [3, L.hi];
    const mx = (E[0] + fb[0]) / 2, mz = (L.hi + fb[1]) / 2, sp = Math.max(3.2, Math.hypot(E[0] - fb[0], E[1] - fb[1]) * 0.95);
    x0 = mx - sp; x1 = mx + sp; z0 = mz - sp * 0.62; z1 = mz + sp * 0.62;
  } else { x0 = fb[0] - 0.35; x1 = fb[0] + 0.35; z0 = fb[1] - 0.22; z1 = fb[1] + 0.22; }
  const sc = Math.min(W / (x1 - x0), H / (z1 - z0)), cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
  const X = (x) => W / 2 + (x - cx) * sc, Y = (z) => H / 2 - (z - cz) * sc;
  const ln = (pts, st, w = 1, dash = []) => {
    c.beginPath(); pts.forEach((p, i) => (i ? c.lineTo(X(p[0]), Y(p[1])) : c.moveTo(X(p[0]), Y(p[1]))));
    c.strokeStyle = st; c.lineWidth = w; c.setLineDash(dash); c.stroke(); c.setLineDash([]);
  };
  const txt = (t, p, st, dx = 6, dy = -6) => { c.font = FONT; c.fillStyle = st; c.textAlign = "left"; c.fillText(t, X(p[0]) + dx, Y(p[1]) + dy); };
  // dish glass and medium, 35 mm wide
  c.fillStyle = k.steelSoft; c.fillRect(X(-17.5), Y(L.hi), 35 * sc, (L.hi - L.zb) * sc);
  c.fillStyle = "rgba(118,188,226,.25)"; c.fillRect(X(-17.5), Y(L.top), 35 * sc, (L.top - L.hi) * sc);
  ln([[-17.5, L.zb], [17.5, L.zb]], k.steel); ln([[-17.5, L.hi], [17.5, L.hi]], k.steel);
  ln(L.det_plane, k.steel, 1.4, [7, 5]);
  if (view !== "train") txt("detection plane", L.det_plane[1], k.steel, 6, 4);
  if (d.envelope) {
    c.beginPath(); d.envelope.forEach((e, i) => (i ? c.lineTo(X(e[1][0]), Y(e[1][1])) : c.moveTo(X(e[1][0]), Y(e[1][1]))));
    for (let i = d.envelope.length - 1; i >= 0; i--) c.lineTo(X(d.envelope[i][2][0]), Y(d.envelope[i][2][1]));
    c.closePath(); c.globalAlpha = 0.4; c.fillStyle = k.sheet; c.fill(); c.globalAlpha = 1;
  }
  c.globalAlpha = 0.75; for (const r of d.rays) ln(r, k.warn, 0.9); c.globalAlpha = 1;
  if (view === "train") {
    const w = L.w, D = s.beam_D_mm;
    const perp = (p, h) => [[p[0] + w[0] * h, p[1] + w[1] * h], [p[0] - w[0] * h, p[1] - w[1] * h]];
    const lens = (p, h, name) => { const [a, b] = perp(p, h); ln([a, b], k.ink, 2); txt(name, a, k.ink, 4, -4); };
    lens(L.collimator, D * 0.9, "collimator");
    const h = D * 1.1, f = perp(L.cyl_front, h), b = perp(L.cyl_back, h);
    ln([f[0], f[1], b[1], b[0], f[0]], k.steel, 1.4); txt("cylinder (power ⊥ page)", f[0], k.steel, 4, -4);
    lens(L.objective, D * 1.3, "objective");
    c.beginPath(); c.arc(X(L.fiber[0]), Y(L.fiber[1]), 3.5, 0, 2 * Math.PI); c.fillStyle = k.ink; c.fill(); txt("fibre", L.fiber, k.ink, 6, 12);
  }
  const mk = (p, fill, label, dy) => {
    c.beginPath(); c.arc(X(p[0]), Y(p[1]), 4, 0, 2 * Math.PI); c.fillStyle = fill; c.fill();
    c.strokeStyle = k.panel; c.lineWidth = 1.2; c.stroke(); if (label) txt(label, p, fill, 8, dy);
  };
  mk(F, k.steel, "detection focus", -8);
  if (d.focus_best) mk(d.focus_best, k.sheetInk, view === "train" ? null : "best focus of the fan", 16);
  scaleBar(c, sc, W, H, k, null);
}

/* ------------------------------------------------------------------ small parts */
const Card = ({ title, sub, children }) => (
  <Paper variant="outlined" sx={{ p: 1.5, mb: 1.5 }}>
    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", mb: 1 }}>
      <Typography variant="subtitle2" sx={{ flexShrink: 0 }}>{title}</Typography>
      {sub && <Typography variant="caption" color="text.secondary" sx={{ ml: 2, textAlign: "right", wordBreak: "break-all" }}>{sub}</Typography>}
    </Box>
    {children}
  </Paper>
);

const Readout = ({ rows }) => (
  <Box sx={{ display: "grid", gridTemplateColumns: "1fr auto auto", columnGap: 1.5, rowGap: 0.25 }}>
    {rows.map(([k, v, sub, bad], i) =>
      k === "-" ? <Box key={i} sx={{ gridColumn: "1/-1", borderTop: 1, borderColor: "divider", my: 0.5 }} /> : (
        <Box key={i} sx={{ display: "contents" }}>
          <Typography variant="body2" color="text.secondary">{k}</Typography>
          <Typography variant="body2" sx={{ textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 500,
            color: bad === undefined ? "text.primary" : bad ? "error.main" : "success.main" }}>{v}</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ textAlign: "right" }}>{sub || ""}</Typography>
        </Box>
      ))}
  </Box>
);

function AxisRow({ label, unit, dec, value, link, raw, step, disabled, onMove }) {
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(false);
  const shown = editing ? draft : fmt(value, dec);
  return (
    <Box sx={{ display: "grid", gridTemplateColumns: "7.5em 2em 7.5em 2em 2.2em 1fr", alignItems: "center", gap: 0.5, py: 0.25 }}>
      <Typography variant="body2">{label}</Typography>
      <Button size="small" sx={{ minWidth: 0 }} disabled={disabled} onClick={() => onMove(-step, true)}>−</Button>
      <TextField size="small" value={shown} disabled={disabled}
        inputProps={{ style: { textAlign: "right", padding: "3px 6px", fontVariantNumeric: "tabular-nums" } }}
        onFocus={() => { setDraft(fmt(value, dec)); setEditing(true); }}
        onBlur={() => setEditing(false)}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") { onMove(parseFloat(draft), false); e.target.blur(); } }} />
      <Button size="small" sx={{ minWidth: 0 }} disabled={disabled} onClick={() => onMove(step, true)}>+</Button>
      <Typography variant="caption" color="text.secondary">{unit}</Typography>
      <Typography variant="caption" color="text.secondary" noWrap>
        {link ? `stage ${link}  ${fmt(raw, 1)} µm` : "set by hand"}
      </Typography>
    </Box>
  );
}

function RigChecks({ c }) {
  if (!c) return null;
  const rows = [];
  const add = (t, text, bad) => rows.push([t, text, bad]);
  add("Collisions", c.collisions.length ? c.collisions.map((x) => `${pn(x.a)} hits ${pn(x.b)}`).join("; ") : "none", c.collisions.length > 0);
  if (c.clearances.length) { const x = c.clearances[0]; add("Closest approach", `${pn(x.a)} to ${pn(x.b)}: ${fmt(x.mm, 1)} mm`, x.mm < 2); }
  add("Launch beam", c.beam_blocked_by.length ? `blocked by ${c.beam_blocked_by.map(pn).join(", ")}` : "passes the holder and dish wall", c.beam_blocked_by.length > 0);
  add("Detection field", c.field_blocked_by.length ? `clipped by ${c.field_blocked_by.map(pn).join(", ")}` : "clear of the holder", c.field_blocked_by.length > 0);
  const sr = c.scan_range_mm;
  add("Usable sample-X scan", sr ? `${fmt(sr[0], 2)} to ${fmt(sr[1], 2)} mm` : "none at this position", !sr);
  add("Sheet width vs field", `${fmt(c.sheet_height_mm, 2)} mm sheet covers ${fmt(c.sheet_covers_fov_frac * 100, 0)} % of ${fmt(c.fov_width_mm, 1)} mm`, c.sheet_covers_fov_frac < 0.9);
  add("Holder above detection lens", `${fmt(c.holder_to_detection_mm, 1)} mm`, c.holder_to_detection_mm < 2);
  const lim = c.limits.filter((x) => !x.ok);
  if (lim.length) add("Outside travel", lim.map((x) => `${x.axis} ${fmt(x.value, 3)}`).join(", "), true);
  (c.notes || []).forEach((n) => add("Note", n, true));
  return (
    <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: 1, mt: 1 }}>
      {rows.map(([t, text, bad]) => (
        <Box key={t} sx={{ borderLeft: 3, borderColor: bad ? "error.main" : "success.main", pl: 1 }}>
          <Typography variant="caption" color="text.secondary" display="block">{t}</Typography>
          <Typography variant="body2">{text}</Typography>
        </Box>
      ))}
    </Box>
  );
}

/* ------------------------------------------------------------------ app */
export default function OPMController() {
  const theme = useTheme();
  const k = colours(theme);
  const [tab, setTab] = useState(0);
  const [S, setS] = useState(null);
  const [error, setError] = useState("");
  const [rig, setRig] = useState(null);
  const [trace, setTrace] = useState(null);
  const [rigView, setRigView] = useState("side");
  const [opticsView, setOpticsView] = useState("dish");
  const [linStep, setLinStep] = useState(0.1);
  const [angStep, setAngStep] = useState(0.05);
  const [track, setTrack] = useState(false);
  const [nIn, setNIn] = useState("1.450");
  const [handHint, setHandHint] = useState(null);
  const [virtual, setVirtual] = useState(false);
  const [notice, setNotice] = useState(null);   // {text, retry?}: move refusals, shown in place, fade out
  const rigCanvas = useRef(null), opticsCanvas = useRef(null), pz = useRef({ x: 0, y: 0, zoom: 1 }), drag = useRef(null);

  const run = async (fn) => {
    try { setError(""); return await fn(); } catch (e) { setError(errMsg(e)); return null; }
  };
  const pollState = useCallback(async () => {
    try { setS(await api.apiOPMGetState()); } catch (e) { setError(`OPM controller not reachable: ${errMsg(e)}`); }
  }, []);
  const pollRig = useCallback(async () => { try { setRig(await api.apiOPMGetRigLayout()); } catch (e) { setRig(null); } }, []);
  const pollTrace = useCallback(async () => { try { setTrace(await api.apiOPMTraceOptics()); } catch (e) { setTrace(null); } }, []);
  usePoll(pollState, 500);
  usePoll(pollRig, 600, tab === 0);
  usePoll(pollTrace, 1500, tab === 1);

  useEffect(() => { if (S && !document.activeElement?.dataset?.nin) setNIn(fmt(S.n, 3)); }, [S?.n]); // eslint-disable-line react-hooks/exhaustive-deps
  const redrawRig = useCallback(() => rigCanvas.current && drawRig(rigCanvas.current, rig, rigView, pz.current, k), [rig, rigView, k]);
  useEffect(() => { redrawRig(); }, [redrawRig]);
  useEffect(() => { if (opticsCanvas.current) drawOptics(opticsCanvas.current, trace, opticsView, k); }, [trace, opticsView, k]);
  useEffect(() => {
    const onResize = () => { redrawRig(); if (opticsCanvas.current) drawOptics(opticsCanvas.current, trace, opticsView, k); };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [redrawRig, trace, opticsView, k]);

  useEffect(() => {
    if (!notice) return undefined;
    const id = setTimeout(() => setNotice(null), 8000);
    return () => clearTimeout(id);
  }, [notice]);

  const scanning = !!S?.scan?.isRunning;
  const send = async (body) => {
    try {
      setNotice(null);
      const r = await api.apiOPMMoveAxes(body);
      if (r && Object.keys(r.manual).length) setHandHint(r.manual);
      if (r?.note) setNotice({ text: r.note });
    } catch (e) {
      const text = errMsg(e);
      setNotice({ text, retry: text.startsWith("Outside travel") ? () => send({ ...body, force: true }) : null });
    }
    pollState(); pollRig();
  };
  const move = (grp, key, val, relative) => {
    if (Number.isFinite(val)) send({ [grp]: { [key]: val }, relative, track: track && !virtual, virtual });
  };
  const setMedium = (n) => run(async () => {
    const r = await api.apiOPMSetMedium(n, true);
    setHandHint(r?.manual || null);
    pollState(); pollRig();
  });
  const resetTwin = () => run(async () => { await api.apiOPMSetReference({}); pollState(); pollRig(); });

  const m = S?.model;
  const bad = (v, tol) => Math.abs(v) > tol;
  const alignRows = m ? [
    ["Detection plane tilt θ", `${fmt(m.theta_deg, 2)}°`], ["Launch tilt needed", `${fmt(m.phi_deg, 3)}°`],
    ["Index implied by the tilt", fmt(m.phi_now_implies_n, 4)], ["-"],
    ["Tilt error", `${fmt(m.err.tilt_deg, 3)}°`, `tol ${fmt(m.tol.tilt_deg, 3)}°`, bad(m.err.tilt_deg, m.tol.tilt_deg)],
    ["Offset error", `${fmt(m.err.offset_um, 1)} µm`, `tol ${fmt(m.tol.offset_um, 1)} µm`, bad(m.err.offset_um, m.tol.offset_um)],
    ["Waist error", `${fmt(m.err.waist_mm, 3)} mm`, `tol ${fmt(m.tol.waist_mm, 2)} mm`, bad(m.err.waist_mm, m.tol.waist_mm)],
    ["Sheet FWHM / 2·zR", `${fmt(m.sheet.fwhm_um, 1)} µm / ${fmt(m.sheet.confocal_mm, 2)} mm`, `field ${fmt(m.sheet.fov_mm, 2)} mm`],
  ] : [["Model", S?.model_error || "–"]];

  const onWheel = (e) => {
    const cv = rigCanvas.current, r = cv.getBoundingClientRect(), p = pz.current;
    const mx = e.clientX - r.left - r.width / 2 - p.x, my = e.clientY - r.top - r.height / 2 - p.y;
    const z = Math.min(40, Math.max(0.3, p.zoom * Math.exp(-e.deltaY * 0.0015))), f = z / p.zoom;
    pz.current = { zoom: z, x: p.x - mx * (f - 1), y: p.y - my * (f - 1) };
    redrawRig();
  };
  useEffect(() => {   // non-passive wheel, so the page does not scroll while zooming
    const cv = rigCanvas.current;
    if (!cv) return undefined;
    const h = (e) => { e.preventDefault(); onWheel(e); };
    cv.addEventListener("wheel", h, { passive: false });
    return () => cv.removeEventListener("wheel", h);
  });

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1, flexWrap: "wrap" }}>
        <Typography variant="h6">dvOPM</Typography>
        <Typography variant="body2" color="text.secondary">
          n {fmt(S?.n, 4)} · sheet {fmt(m?.theta_deg, 2)}° · {S?.scan?.message || "…"}
        </Typography>
        <Box sx={{ flex: 1 }} />
        {scanning && <Button color="error" variant="contained" onClick={() => run(api.apiOPMStopScan)}>Stop scan</Button>}
      </Box>
      {error && <Alert severity="error" sx={{ mb: 1 }} onClose={() => setError("")}>{error}</Alert>}
      <Tabs value={tab} onChange={(e, v) => setTab(v)} sx={{ mb: 1.5 }}>
        <Tab label="Digital twin" />
        <Tab label="Launch optics" />
        <Tab label="Volume scan" />
      </Tabs>

      {tab === 0 && (
        <Grid container spacing={2}>
          <Grid item xs={12} lg={7}>
            <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1, flexWrap: "wrap" }}>
              <ToggleButtonGroup size="small" exclusive value={rigView}
                onChange={(e, v) => { if (v) { pz.current = { x: 0, y: 0, zoom: 1 }; setRigView(v); } }}>
                <ToggleButton value="side">Side (x–z)</ToggleButton>
                <ToggleButton value="top">Top (x–y)</ToggleButton>
                <ToggleButton value="dish">Dish</ToggleButton>
              </ToggleButtonGroup>
              <Typography variant="caption" color="text.secondary">Drag to pan, scroll to zoom, double-click to reset</Typography>
            </Box>
            <canvas ref={rigCanvas} aria-label="Rig drawing from the live axis positions"
              style={{ width: "100%", height: 480, display: "block", border: `${virtual ? 2 : 1}px solid ${virtual ? "#C9822A" : k.line}`, borderRadius: 6, touchAction: "none", cursor: "grab" }}
              onPointerDown={(e) => { drag.current = [e.clientX, e.clientY]; e.currentTarget.setPointerCapture(e.pointerId); }}
              onPointerMove={(e) => {
                if (!drag.current) return;
                const p = pz.current;
                pz.current = { ...p, x: p.x + e.clientX - drag.current[0], y: p.y + e.clientY - drag.current[1] };
                drag.current = [e.clientX, e.clientY]; redrawRig();
              }}
              onPointerUp={() => { drag.current = null; }}
              onDoubleClick={() => { pz.current = { x: 0, y: 0, zoom: 1 }; redrawRig(); }} />
            <RigChecks c={rig?.checks} />
          </Grid>

          <Grid item xs={12} lg={5}>
            <Card title="Axes" sub="model frame, mm and degrees">
              <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1, flexWrap: "wrap" }}>
                <ToggleButtonGroup size="small" exclusive value={virtual ? "virtual" : "physical"}
                  onChange={(e, v) => v && setVirtual(v === "virtual")}>
                  <ToggleButton value="physical">Move stages</ToggleButton>
                  <ToggleButton value="virtual">Align drawing</ToggleButton>
                </ToggleButtonGroup>
                <Button size="small" disabled={!S || scanning} onClick={resetTwin}>Reset to nominal</Button>
                <FormControlLabel sx={{ ml: "auto" }} label={<Typography variant="body2">Travel limits</Typography>}
                  control={<Switch size="small" checked={S?.limits ?? true} disabled={!S}
                    onChange={(e) => run(async () => { await api.apiOPMSetLimits(e.target.checked); pollState(); })} />} />
              </Box>
              {virtual && (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Nothing moves. Step the axes until the sheet in the drawing sits where you see it on
                  the instrument; the offset is kept when you switch back to moving stages.
                </Typography>
              )}
              {AXES.map(([grp, key, label, unit, dec]) => {
                const t = `${grp}.${key}`;
                return (
                  <AxisRow key={t} label={label} unit={unit} dec={dec} value={S?.values?.[t]} link={S?.links?.[t]}
                    raw={S?.raw?.[t]} step={key === "phi_deg" ? angStep : linStep} disabled={!S || scanning}
                    onMove={(v, rel) => move(grp, key, v, rel)} />
                );
              })}
              <Box sx={{ display: "flex", gap: 1.5, alignItems: "center", mt: 1, flexWrap: "wrap" }}>
                <FormControl size="small">
                  <InputLabel>Step</InputLabel>
                  <Select label="Step" value={linStep} onChange={(e) => setLinStep(e.target.value)}>
                    {[0.001, 0.01, 0.1, 1].map((v) => <MenuItem key={v} value={v}>{v < 1 ? `${v * 1000} µm` : `${v} mm`}</MenuItem>)}
                  </Select>
                </FormControl>
                <FormControl size="small">
                  <InputLabel>Tilt step</InputLabel>
                  <Select label="Tilt step" value={angStep} onChange={(e) => setAngStep(e.target.value)}>
                    {[0.01, 0.05, 0.5, 2].map((v) => <MenuItem key={v} value={v}>{v}°</MenuItem>)}
                  </Select>
                </FormControl>
                <FormControlLabel control={<Switch size="small" checked={track} onChange={(e) => setTrack(e.target.checked)} />}
                  label={<Typography variant="body2">Linked launch axes follow sample Z</Typography>} />
              </Box>
              {notice && (
                <Alert severity="warning" sx={{ mt: 1 }} onClose={() => setNotice(null)}
                  action={notice.retry && <Button color="inherit" size="small" onClick={notice.retry}>Move anyway</Button>}>
                  {notice.text}
                </Alert>
              )}
              {handHint && Object.keys(handHint).length > 0 && (
                <Alert severity="info" sx={{ mt: 1 }} onClose={() => setHandHint(null)}>
                  Set by hand on the instrument: {Object.entries(handHint).map(([t, v]) =>
                    `${t.replace("launch.", "launch ").replace("_deg", "")} ${fmt(v, t.endsWith("_deg") ? 3 : 4)}`).join(", ")}
                </Alert>
              )}
            </Card>

            <Card title="Medium" sub={`current n ${fmt(S?.n, 4)}`}>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 1 }}>
                {MEDIA.map(([name, n]) => (
                  <Button key={name} size="small" variant="outlined" disabled={scanning}
                    onClick={() => { setNIn(n.toFixed(3)); setMedium(n); }}>{name} {n}</Button>
                ))}
              </Box>
              <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                <TextField size="small" label="Index n" value={nIn} sx={{ width: "8em" }} inputProps={{ "data-nin": 1 }}
                  onChange={(e) => setNIn(e.target.value)} />
                <Button variant="contained" disabled={scanning} onClick={() => setMedium(parseFloat(nIn))}>Set medium</Button>
                <Button disabled={scanning || !S} onClick={() => setMedium(S.n)}>Launch to model pose</Button>
              </Box>
            </Card>

            <Card title="Alignment" sub="model prediction"><Readout rows={alignRows} /></Card>
          </Grid>
        </Grid>
      )}

      {tab === 1 && <OpticsTab trace={trace} view={opticsView} setView={setOpticsView} canvasRef={opticsCanvas} run={run} />}
      {tab === 2 && <ScanTab S={S} run={run} />}
    </Box>
  );
}

/* ------------------------------------------------------------------ optics tab */
function OpticsTab({ trace, view, setView, canvasRef, run }) {
  const s = trace?.summary;
  const exportScript = () => run(async () => {
    const blob = await api.apiOPMExportOptiland();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `dvopm_launch_n${(trace?.inputs?.n ?? 1.45).toFixed(3)}_optiland.py`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
  const rows = !s ? [] : trace.tir ? [["Status", "Total internal reflection", "", true]] : [
    ["Collimated beam", `${fmt(s.beam_D_mm, 2)} mm`], ["Sheet NA in air / medium", `${fmt(s.NA_air, 4)} / ${fmt(s.NA_medium, 4)}`],
    ["Sheet width (cylinder axis)", `${fmt(s.sheet_width_mm, 2)} mm`], ["Sheet FWHM at waist", `${fmt(s.fwhm_um, 1)} µm`],
    ["Confocal length 2·zR", `${fmt(2 * s.rayleigh_mm, 2)} mm`, `field ${fmt(s.fov_along_sheet_mm, 2)} mm`, 2 * s.rayleigh_mm < s.fov_along_sheet_mm],
    ["-"], ["Launch tilt φ", `${fmt(s.phi_deg, 3)}°`], ["Sheet angle in medium", `${fmt(s.theta_sheet_deg, 3)}°`],
    ["Waist, Coddington", `${fmt(s.coddington_s_mm, 4)} mm`, "from inner surface"],
    ["Waist, real rays (paraxial)", `${fmt(s.paraxial_s_mm, 4)} mm`], ["Best focus of the fan", `${fmt(s.best_s_mm, 4)} mm`],
    ["Zone focus spread", `${fmt(s.focus_spread_um, 0)} µm`, "along the sheet"], ["Geometric line width (rms)", `${fmt(s.rms_spot_um, 2)} µm`],
    ["-"], ["Transmission, s-pol", `${fmt(s.T_s * 100, 1)} %`, "", s.T_s < 0.85], ["Transmission, p-pol", `${fmt(s.T_p * 100, 1)} %`, "", s.T_p < 0.85],
    ["Sheet tilt / offset error", `${fmt(s.angle_err_deg, 3)}° / ${fmt(s.offset_err_um, 1)} µm`],
    ["Cylinder → objective", `${fmt(s.cyl_to_objective_mm, 1)} mm`, "line focus in BFP"],
  ];
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} lg={8}>
        <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1, flexWrap: "wrap" }}>
          <ToggleButtonGroup size="small" exclusive value={view} onChange={(e, v) => v && setView(v)}>
            <ToggleButton value="train">Whole launch</ToggleButton>
            <ToggleButton value="dish">Dish</ToggleButton>
            <ToggleButton value="focus">Focus</ToggleButton>
          </ToggleButtonGroup>
          <Typography variant="caption" color="text.secondary">
            Red: tangential ray fan (sheet thickness). Green: Gaussian sheet to scale. Follows the twin's medium, sample Z and launch pose.
          </Typography>
        </Box>
        <canvas ref={canvasRef} aria-label="Side view of the launch rays refracting into the dish"
          style={{ width: "100%", height: 460, display: "block", borderRadius: 6 }} />
      </Grid>
      <Grid item xs={12} lg={4}>
        <Card title="What the rays say"><Readout rows={rows} /></Card>
        <Button variant="outlined" disabled={!s || trace?.tir} onClick={exportScript}>Export Optiland script</Button>
      </Grid>
    </Grid>
  );
}

/* ------------------------------------------------------------------ scan tab */
function ScanTab({ S, run }) {
  const [p, setP] = useState({ length_mm: 1, fps: 20, step_um: "", ramp_mm: 0.05, illuSource: "", illuValue: 0,
    storageFormat: "ome_zarr", deskew: true, deskewFlip: false, experimentName: "opm_scan" });
  const [plan, setPlan] = useState(null);
  const set = (key) => (e) => setP({ ...p, [key]: e.target.type === "checkbox" ? e.target.checked : e.target.value });
  const scan = S?.scan || {};
  const light = (S?.illumination || []).find((l) => l.name === p.illuSource);

  useEffect(() => {
    const id = setTimeout(async () => {
      try { setPlan(await api.apiOPMGetScanPlan(+p.length_mm, +p.fps, +p.step_um || null)); } catch (e) { setPlan(null); }
    }, 250);
    return () => clearTimeout(id);
  }, [p.length_mm, p.fps, p.step_um, S?.n]);

  const start = () => run(() => api.apiOPMStartScan({
    ...p, length_mm: +p.length_mm, fps: +p.fps, step_um: +p.step_um || null, ramp_mm: +p.ramp_mm, illuValue: +p.illuValue,
  }));
  const last = scan.last;
  const num = (key, label, extra = {}) => (
    <TextField size="small" fullWidth label={label} value={p[key]} onChange={set(key)} type="number" {...extra} />
  );
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Typography variant="subtitle2" gutterBottom>Live view ({S?.detector || "camera"})</Typography>
        <LiveViewControlWrapper />
      </Grid>
      <Grid item xs={12} md={6}>
        <Card title="Stage scan along sample X" sub="the sample moves through the static sheet">
          <Grid container spacing={1.5}>
            <Grid item xs={6}>{num("length_mm", "Length (mm)")}</Grid>
            <Grid item xs={6}>{num("fps", "Camera frame rate (fps)")}</Grid>
            <Grid item xs={6}>{num("step_um", "Step per frame (µm)", { placeholder: "auto", InputLabelProps: { shrink: true } })}</Grid>
            <Grid item xs={6}>{num("ramp_mm", "Run-up each end (mm)")}</Grid>
            <Grid item xs={6}>
              <FormControl size="small" fullWidth>
                <InputLabel>Illumination</InputLabel>
                <Select label="Illumination" value={p.illuSource} onChange={set("illuSource")}>
                  <MenuItem value="">none</MenuItem>
                  {(S?.illumination || []).map((l) => <MenuItem key={l.name} value={l.name}>{l.name}</MenuItem>)}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>{num("illuValue", light ? `Intensity (${light.min}–${light.max})` : "Intensity", { disabled: !light })}</Grid>
            <Grid item xs={6}>
              <FormControl size="small" fullWidth>
                <InputLabel>Storage</InputLabel>
                <Select label="Storage" value={p.storageFormat} onChange={set("storageFormat")}>
                  <MenuItem value="ome_zarr">OME-Zarr</MenuItem>
                  <MenuItem value="tiff">OME-TIFF</MenuItem>
                  <MenuItem value="both">Both</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}><TextField size="small" fullWidth label="Name" value={p.experimentName} onChange={set("experimentName")} /></Grid>
            <Grid item xs={12}>
              <FormControlLabel control={<Switch size="small" checked={p.deskew} onChange={set("deskew")} />} label="Deskew into a z, y, x volume" />
              <FormControlLabel control={<Switch size="small" checked={p.deskewFlip} disabled={!p.deskew} onChange={set("deskewFlip")} />} label="Tilted axis reversed" />
            </Grid>
          </Grid>
          {plan && (
            <Box sx={{ mt: 1 }}>
              <Readout rows={[
                ["Step per frame", `${fmt(plan.step_um, 2)} µm`], ["Slice spacing", `${fmt(plan.slice_spacing_um, 2)} µm`],
                ["Stage speed", `${fmt(plan.velocity_mm_s, 4)} mm/s`], ["Frames / duration", `${plan.frames} / ${fmt(plan.duration_s, 1)} s`],
              ]} />
            </Box>
          )}
          <Box sx={{ display: "flex", gap: 1, mt: 1.5 }}>
            <Button variant="contained" disabled={!S || scan.isRunning} onClick={start}>Start scan</Button>
            <Button color="error" variant="outlined" disabled={!scan.isRunning} onClick={() => run(api.apiOPMStopScan)}>Stop</Button>
          </Box>
          {(scan.isRunning || scan.progress > 0) && (
            <Box sx={{ mt: 1.5 }}>
              <Typography variant="body2" color="text.secondary">{scan.message}</Typography>
              <LinearProgress variant="determinate" value={Math.round((scan.progress || 0) * 100)} />
            </Box>
          )}
          {scan.error && <Alert severity="warning" sx={{ mt: 1 }}>{scan.error}</Alert>}
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            The camera free-runs while the stage moves. Each frame is placed by the speed and frame rate
            measured during the move, not the planned ones.
          </Typography>
        </Card>
        {last && (
          <Card title="Last scan" sub={last.folder}>
            <Readout rows={[
              ["Frames kept", last.frames], ["Camera frame rate", `${fmt(last.camera_fps ?? last.fps, 2)} fps`, "from its frame counter"],
              ["Kept frames per second", `${fmt(last.fps, 2)} fps`],
              ["Measured stage speed", `${fmt(last.velocity_mm_s, 4)} mm/s`], ["Step per frame", `${fmt(last.step_um, 2)} µm`],
              ["Dropped frames", last.dropped_frames ?? "–", "", last.dropped_frames > 0],
            ]} />
            {[...last.raw, ...last.deskewed].map((f) => (
              <Typography key={f} variant="caption" display="block" sx={{ wordBreak: "break-all" }}>{f}</Typography>
            ))}
            {last.deskewError && <Alert severity="warning" sx={{ mt: 1 }}>{last.deskewError}</Alert>}
          </Card>
        )}
      </Grid>
    </Grid>
  );
}
