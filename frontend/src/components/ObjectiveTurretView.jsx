import React, { useId } from "react";
import { useSelector } from "react-redux";
import { Box, Chip } from "@mui/material";
import { useTheme } from "@mui/material/styles";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";

/*
 * Live side view of the linear two-slot objective turret (inverted
 * microscope). The carriage slides along A -- slot i is in the beam when A
 * equals its stored position x_i -- and the turret rides on Z, drawn relative
 * to the in-focus Z stored for the objective in the beam. Each objective shows
 * magnification (with its ISO 8578 colour ring), NA and pixel size; the one in
 * the beam throws its NA cone at the sample. Read-only: a pure function of the
 * backend state (ObjectiveController.getstatus + the live stage position).
 */

const W = 640;
const H = 322;
const AXIS_X = W / 2;
const FOCAL_Y = 74; // specimen plane: top of the coverslip
const CARRIAGE_Y = 214; // carriage top while Z sits at the stored focus
const MAX_LIFT = 24; // px the turret may travel either way on Z
const CAPTION_Y = CARRIAGE_Y + 32 + MAX_LIFT + 16; // below the lowest turret
const SPACING = 170; // on-screen distance between the two slots
const LENS_R = 12; // front-lens radius
const MOUNT_H = 12;

// ISO 8578 magnification colour ring: upper bound of the range -> colour
const MAG_RING = [
  [1.5, "#212121"],
  [2.5, "#8d5a2b"],
  [5, "#d32f2f"],
  [10, "#f4c20d"],
  [20, "#2e9b57"],
  [32, "#26a69a"],
  [50, "#64b5f6"],
  [63, "#1e3fae"],
  [Infinity, "#fafafa"],
];
const ringColor = (mag) =>
  mag > 0 ? MAG_RING.find(([max]) => mag <= max)[1] : null;

// Front lens -> focus from the cone half-angle asin(NA / n): high NA means a
// short working distance and, the objectives being parfocal, a longer barrel.
const workingDistance = (na) => {
  const n = na > 1 ? 1.515 : 1; // immersion oil above NA 1
  const sinTheta = Math.min((na > 0 ? na : 0.25) / n, 0.97);
  return Math.min(80, Math.max(8, LENS_R / Math.tan(Math.asin(sinTheta))));
};

// µm of defocus -> px, log-compressed so that 1 µm and 1 mm both show
const defocusPx = (dz) =>
  Math.sign(dz) * Math.min(MAX_LIFT, 9 * Math.log10(1 + Math.abs(dz)));

const num = (v) => (v === null || v === undefined || v === "" ? NaN : Number(v));
const fmt = (v, digits = 1) =>
  Number.isFinite(v)
    ? v.toLocaleString(undefined, { maximumFractionDigits: digits })
    : "—";
const clip = (s, n = 22) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

// specimen: [x, rx, ry, colour], resting on the coverslip
const CELLS = [
  [74, 15, 7, "#e57373"],
  [128, 10, 5, "#81c784"],
  [196, 18, 8, "#ba68c8"],
  [258, 11, 6, "#4fc3f7"],
  [316, 16, 8, "#81c784"],
  [372, 10, 5, "#e57373"],
  [430, 17, 7, "#4fc3f7"],
  [494, 12, 6, "#ba68c8"],
  [558, 15, 7, "#81c784"],
];

export default function ObjectiveTurretView() {
  const theme = useTheme();
  const id = useId().replace(/:/g, "");
  const obj = useSelector(objectiveSlice.getObjectiveState);
  const pos = useSelector(positionSlice.getPositionState);

  const slots = [0, 1].map((i) => ({
    i,
    x: num(i === 0 ? obj.posX0 : obj.posX1),
    z: num(i === 0 ? obj.posZ0 : obj.posZ1),
    name: obj.availableObjectivesNames?.[i] || `Obj ${i + 1}`,
    mag: num(obj.availableObjectiveMagnifications?.[i]),
    na: num(obj.availableObjectiveNAs?.[i]),
    pixelSize: num(obj.availableObjectivePixelSizes?.[i]),
    configured: obj.slotConfigured?.[i] !== false,
  }));
  const current =
    obj.currentObjective === 0 || obj.currentObjective === 1
      ? obj.currentObjective
      : null;
  const ref = slots[current ?? 0];

  // --- A: which slot is in the beam --------------------------------------
  const span = Math.abs(slots[1].x - slots[0].x);
  const scale = span > 0 ? SPACING / span : 0;
  // Without an A motor the switch is software-only, so the objective the
  // backend reports is by definition the one in the beam.
  const a =
    obj.hasMotor === false || !Number.isFinite(num(pos.a)) ? ref.x : num(pos.a);
  const tolerance = Math.max(5, 0.02 * span);
  const inBeam =
    scale > 0
      ? slots.find((s) => s.configured && Math.abs(a - s.x) <= tolerance) ??
        null
      : ref;
  const dir = slots[1].x >= slots[0].x ? 1 : -1;
  const localX = (s) => s.i * dir * SPACING;
  let x0 =
    scale > 0
      ? AXIS_X + (slots[0].x - a) * scale
      : AXIS_X - (current ?? 0) * SPACING;
  // keep the carriage in view when A is far from both slots (e.g. just homed)
  const lo = Math.min(x0, x0 + dir * SPACING);
  const hi = Math.max(x0, x0 + dir * SPACING);
  if (lo < 90) x0 += 90 - lo;
  if (hi > W - 90) x0 -= hi - (W - 90);

  // --- Z: distance from the stored in-focus Z of the objective in the beam
  const z = num(pos.z);
  const focusStored =
    current !== null &&
    Number.isFinite(ref.z) &&
    !(slots[0].z === 0 && slots[1].z === 0); // both 0 = never set
  const dz = focusStored && Number.isFinite(z) ? z - ref.z : 0;
  // + = turret up, towards the sample: the "↑" of the Z buttons
  const lift = defocusPx(dz);

  // --- status text -------------------------------------------------------
  const beamOk = inBeam !== null && inBeam.i === current;
  let beamLabel = inBeam
    ? `In the beam: ${inBeam.name}`
    : `Turret between slots (A ${fmt(a)} µm)`;
  if (inBeam && current === null) beamLabel += " (not confirmed by the backend)";
  else if (inBeam && !beamOk) beamLabel += ` (backend: ${slots[current].name})`;
  const atFocus = focusStored && Math.abs(dz) < 0.05;
  const zLabel = `Z ${fmt(z)} µm · ${
    !focusStored
      ? "no stored focus"
      : atFocus
        ? "at stored focus"
        : `${dz > 0 ? "+" : ""}${fmt(dz)} µm from stored focus`
  }`;

  // --- drawing -----------------------------------------------------------
  const dark = theme.palette.mode === "dark";
  const accent = theme.palette.primary.main;
  const ink = theme.palette.text.primary;
  const muted = theme.palette.text.secondary;
  const glass = dark ? "#7fb3d5" : "#8cc4e6";
  const metal = dark ? ["#9aa2ad", "#4b525c"] : ["#f4f6f8", "#98a1ad"];
  const engraving = dark ? "#f3f5f7" : "#1d232b";
  const transition = "transform 350ms ease-out";

  const objective = (s) => {
    const x = localX(s);
    const base = CARRIAGE_Y - MOUNT_H;
    if (!s.configured) {
      return (
        <g key={s.i}>
          <rect x={x - 25} y={base} width={50} height={MOUNT_H} rx={2}
            fill="none" stroke={muted} strokeDasharray="3 3" />
          <text x={x} y={base - 8} fontSize={11} fill={muted} textAnchor="middle">
            empty
          </text>
        </g>
      );
    }
    const beam = inBeam?.i === s.i;
    const top = FOCAL_Y + workingDistance(s.na); // front lens
    const ring = ringColor(s.mag);
    return (
      <g key={s.i} opacity={beam || !inBeam ? 1 : 0.55}>
        {beam && (
          <>
            <polygon
              points={`${x - LENS_R},${top} ${x + LENS_R},${top} ${x},${FOCAL_Y}`}
              fill={`url(#${id}-cone)`}
            />
            <circle cx={x} cy={FOCAL_Y} r={3.5} fill={accent}
              filter={`url(#${id}-glow)`} />
          </>
        )}
        <rect x={x - 25} y={base} width={50} height={MOUNT_H} rx={2}
          fill={`url(#${id}-thread)`} />
        <path
          d={`M${x - 21} ${base} L${x + 21} ${base} L${x + 16} ${top + 12} L${x - 16} ${top + 12} Z`}
          fill={`url(#${id}-metal)`}
          stroke={beam ? accent : "none"}
          strokeWidth={1.5}
        />
        {ring && (
          <rect x={x - 20.5} y={base - 9} width={41} height={5} fill={ring} />
        )}
        <rect x={x - LENS_R - 1} y={top + 1} width={2 * LENS_R + 2} height={11}
          rx={3} fill={`url(#${id}-metal)`} />
        <ellipse cx={x} cy={top + 1} rx={LENS_R} ry={2.5} fill={glass}
          stroke={glass} />
        <text x={x} y={base - 13} fontSize={13} fontWeight={700}
          fill={engraving} textAnchor="middle">
          {s.mag > 0 ? `${s.mag}×` : "?×"}
        </text>
        {s.na > 0 && (
          <text x={x} y={base - 26} fontSize={9.5} fill={engraving}
            textAnchor="middle">
            {fmt(s.na, 2)}
          </text>
        )}
      </g>
    );
  };

  const caption = (s) => {
    const x = localX(s);
    const beam = inBeam?.i === s.i;
    return (
      <g key={s.i} textAnchor="middle" opacity={s.configured ? 1 : 0.55}>
        <text x={x} y={CAPTION_Y} fontSize={12.5}
          fontWeight={beam ? 700 : 500} fill={beam ? accent : ink}>
          {clip(s.name)}
        </text>
        <text x={x} y={CAPTION_Y + 15} fontSize={11} fill={muted}>
          {s.configured
            ? `${fmt(s.mag, 0)}× · NA ${fmt(s.na, 2)} · ${fmt(s.pixelSize, 3)} µm/px`
            : "not configured"}
        </text>
        <text x={x} y={CAPTION_Y + 29} fontSize={10} fill={muted}>
          {`slot at A ${fmt(s.x)} · focus Z ${fmt(s.z)} µm`}
        </text>
      </g>
    );
  };

  const carriageLeft = Math.min(0, dir * SPACING) - 70;
  return (
    <Box>
      <Box
        component="svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Objective turret. ${beamLabel}. A ${fmt(a)} µm, ${zLabel}.`}
        sx={{ width: "100%", maxWidth: 760, display: "block", mx: "auto" }}
      >
        <defs>
          <linearGradient id={`${id}-metal`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor={metal[1]} />
            <stop offset="0.42" stopColor={metal[0]} />
            <stop offset="1" stopColor={metal[1]} />
          </linearGradient>
          <linearGradient id={`${id}-thread`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={metal[1]} />
            <stop offset="0.5" stopColor={metal[0]} />
            <stop offset="1" stopColor={metal[1]} />
          </linearGradient>
          <linearGradient id={`${id}-cone`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={accent} stopOpacity={0.6} />
            <stop offset="1" stopColor={accent} stopOpacity={0.1} />
          </linearGradient>
          <filter id={`${id}-glow`} x="-3" y="-3" width="7" height="7">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* optical axis */}
        <line x1={AXIS_X} y1={6} x2={AXIS_X} y2={CAPTION_Y - 14}
          stroke={accent} strokeOpacity={0.4} strokeDasharray="4 5" />
        <text x={AXIS_X + 6} y={14} fontSize={10} fill={muted}>
          optical axis
        </text>

        {/* sample: medium, cells resting on the coverslip */}
        <rect x={30} y={20} width={W - 60} height={FOCAL_Y - 20} rx={6}
          fill={glass} fillOpacity={0.1} />
        {CELLS.map(([cx, rx, ry, color]) => (
          <g key={cx}>
            <ellipse cx={cx} cy={FOCAL_Y - ry} rx={rx} ry={ry} fill={color}
              fillOpacity={0.35} stroke={color} strokeOpacity={0.8} />
            <circle cx={cx + rx * 0.2} cy={FOCAL_Y - ry}
              r={Math.min(rx, ry) * 0.4} fill={color} fillOpacity={0.85} />
          </g>
        ))}
        <rect x={30} y={FOCAL_Y} width={W - 60} height={7} fill={glass}
          fillOpacity={0.45} stroke={glass} strokeOpacity={0.8} />
        <text x={38} y={35} fontSize={12} fontWeight={600} fill={muted}>
          Sample
        </text>
        <text x={W - 38} y={FOCAL_Y + 20} fontSize={10} fill={muted}
          textAnchor="end">
          focal plane
        </text>

        {/* the turret rides on Z: rail with the beam index, and on it ... */}
        <g style={{ transform: `translate(0px, ${-lift}px)`, transition }}>
          <rect x={24} y={CARRIAGE_Y + 16} width={W - 48} height={7} rx={3.5}
            fill={`url(#${id}-thread)`} />
          <path
            d={`M${AXIS_X - 6} ${CARRIAGE_Y + 32} L${AXIS_X + 6} ${CARRIAGE_Y + 32} L${AXIS_X} ${CARRIAGE_Y + 24} Z`}
            fill={accent}
          />
          {/* ... the carriage, sliding along A */}
          <g style={{ transform: `translate(${x0}px, 0px)`, transition }}>
            <rect x={carriageLeft} y={CARRIAGE_Y} width={SPACING + 140}
              height={14} rx={4} fill={`url(#${id}-thread)`} />
            {slots.map((s) => (
              <path key={s.i}
                d={`M${localX(s) - 5} ${CARRIAGE_Y + 14} L${localX(s) + 5} ${CARRIAGE_Y + 14} L${localX(s)} ${CARRIAGE_Y + 20} Z`}
                fill={inBeam?.i === s.i ? accent : muted} />
            ))}
            {slots.map(objective)}
          </g>
        </g>

        {/* slot captions slide with the carriage, not with Z */}
        <g style={{ transform: `translate(${x0}px, 0px)`, transition }}>
          {slots.map(caption)}
        </g>
      </Box>
      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", justifyContent: "center", mt: 1 }}>
        <Chip size="small" variant="outlined"
          color={beamOk ? "success" : "warning"} label={beamLabel} />
        <Chip size="small" variant="outlined" label={`A ${fmt(a)} µm`} />
        <Chip size="small" variant="outlined"
          color={atFocus ? "success" : "default"} label={zLabel} />
      </Box>
    </Box>
  );
}
