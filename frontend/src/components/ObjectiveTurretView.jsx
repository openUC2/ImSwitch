import React, { useId } from "react";
import { useSelector } from "react-redux";
import { Box, Typography } from "@mui/material";
import { useTheme } from "@mui/material/styles";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import { TURRET_OUTLINE } from "./objectiveTurretOutline.js";

/*
 * Side view of the linear two-slot objective turret, traced from its CAD
 * (scripts/turret_side_view.py) and drawn to scale. The plate slides along A
 * -- slot i is under the optical axis when A equals its stored position x_i --
 * and the turret rides on Z, drawn relative to the stored focus of the
 * objective in the beam (log-compressed, so microns show). Read-only: a pure
 * function of the backend state (ObjectiveController.getstatus + the live
 * stage position).
 */

const W = 600;
const H = 318;
const K = 3.4; // px per mm
const AXIS_X = W / 2;
const SAMPLE_Y = 34; // specimen plane: top face of the slide
const MAX_FOCUS_Z = 82; // mm above the plate origin, keeps the turret inside H
const LIFT = 12; // px the turret may travel either way for defocus
const LENS_R = 3.3; // mm, front-lens radius the NA cone starts from

// Colours of the parts in the CAD render; they do not follow the theme.
const COLOR = {
  plate: "#b07d31",
  pins: "#aeb3ba",
  adapter: "#26292c",
  barrel: "#d9d6cf",
  ring: "#e4e1da",
  nose: "#80848f",
};
const ENGRAVING = "#35383c";

// Front lens -> focus from the cone half-angle asin(NA / n): a schematic
// working distance, but ordered like real ones (4x far off, oil close).
const workingDistance = (na) => {
  const n = na > 1 ? 1.515 : 1; // immersion oil above NA 1
  const sinTheta = Math.min((na > 0 ? na : 0.25) / n, 0.97);
  return Math.max(0.6, LENS_R / Math.tan(Math.asin(sinTheta)));
};

// µm of defocus -> px, log-compressed so that 1 µm and 1 mm both show
const defocusPx = (dz) =>
  Math.sign(dz) * Math.min(LIFT, 9 * Math.log10(1 + Math.abs(dz)));

const num = (v) => (v === null || v === undefined || v === "" ? NaN : Number(v));
const fmt = (v, digits = 1) =>
  Number.isFinite(v)
    ? v.toLocaleString(undefined, { maximumFractionDigits: digits })
    : "—";
const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

const Readout = ({ label, value, warn }) => (
  <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.75 }}>
    <Typography
      variant="caption"
      sx={{ color: "text.secondary", textTransform: "uppercase", letterSpacing: "0.06em" }}
    >
      {label}
    </Typography>
    <Typography
      variant="body2"
      sx={{ fontVariantNumeric: "tabular-nums", color: warn ? "warning.main" : "text.primary" }}
    >
      {value}
    </Typography>
  </Box>
);

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

  // The turret holds a short 4x-style and a tall 20x-style objective; the
  // tall one carries whichever slot has the higher magnification.
  const highSlot = (slots[1].mag || 0) >= (slots[0].mag || 0) ? 1 : 0;
  const bodyOf = (s) =>
    s.i === highSlot ? TURRET_OUTLINE.high : TURRET_OUTLINE.low;

  // --- A: plate position and the slot in the beam ------------------------
  const span = Math.abs(slots[1].x - slots[0].x);
  // Without an A motor the switch is software-only: the objective the backend
  // reports is by definition the one in the beam.
  const noMotor = obj.hasMotor === false || !Number.isFinite(num(pos.a));
  const a = noMotor ? slots[current ?? 0].x : num(pos.a);
  const tolerance = Math.max(5, 0.02 * span);
  const inBeam =
    slots.find(
      (s) =>
        s.configured &&
        (span > 0 ? Math.abs(a - s.x) <= tolerance : s.i === (current ?? 0)),
    ) ?? null;
  // plate centre relative to the optical axis (mm): slot i's axis is on the
  // optical axis at A = x_i, linear in between, clamped when far outside
  const c0 = bodyOf(slots[0]).axis;
  const c1 = bodyOf(slots[1]).axis;
  const t =
    span > 0 ? (a - slots[0].x) / (slots[1].x - slots[0].x) : current === 1 ? 1 : 0;
  const plateX = Math.max(-45, Math.min(45, -(c0 + t * (c1 - c0))));

  // --- Z: where the objective in the beam focuses, and how far off we are
  const ref = inBeam ?? slots[current ?? 0];
  const focusZ = Math.min(MAX_FOCUS_Z, bodyOf(ref).top + workingDistance(ref.na));
  const z = num(pos.z);
  const focusStored =
    Number.isFinite(ref.z) && !(slots[0].z === 0 && slots[1].z === 0); // both 0 = never set
  const dz = focusStored && Number.isFinite(z) ? z - ref.z : 0;
  const atFocus = focusStored && Math.abs(dz) < 0.05;
  // + = turret up, towards the sample: the "↑" of the Z buttons
  const originY = SAMPLE_Y + focusZ * K - defocusPx(dz);

  // --- readouts ----------------------------------------------------------
  const beamOk = inBeam !== null && inBeam.i === current;
  let beamText = inBeam ? inBeam.name : "between slots";
  if (inBeam && current === null) beamText += " (not confirmed by the backend)";
  else if (inBeam && !beamOk) beamText += ` (backend: ${slots[current].name})`;
  const aText = obj.hasMotor === false ? "no motor" : `${fmt(a)} µm`;
  const zText = `${fmt(z)} µm · ${
    !focusStored
      ? "no stored focus"
      : atFocus
        ? "at stored focus"
        : `${dz > 0 ? "+" : ""}${fmt(dz)} µm from stored focus`
  }`;

  // --- drawing -----------------------------------------------------------
  const dark = theme.palette.mode === "dark";
  const accent = theme.palette.primary.main;
  const line = theme.palette.text.secondary;
  const edge = { stroke: "rgba(0,0,0,0.4)", strokeWidth: 0.75, vectorEffect: "non-scaling-stroke" };
  const transition = "transform 350ms ease-out";

  const part = (key, d, color) => (
    <g key={key}>
      <path d={d} fill={color} />
      <path d={d} fill={`url(#${id}-shade)`} {...edge} />
    </g>
  );

  const objective = (s) => {
    const body = bodyOf(s);
    if (!s.configured) {
      return (
        <g key={s.i} fill="none">
          {Object.entries(body.parts).map(([key, d]) => (
            <path key={key} d={d} stroke={line} strokeOpacity={0.6}
              strokeWidth={0.75} strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
          ))}
        </g>
      );
    }
    return (
      <g key={s.i}>
        {Object.entries(body.parts).map(([key, d]) => part(key, d, COLOR[key]))}
      </g>
    );
  };

  // magnification/NA, name and pixel size, "engraved" on the barrel
  const engraving = (s) => {
    const body = bodyOf(s);
    const magLine = s.mag > 0 ? `${fmt(s.mag)}x${s.na > 0 ? `/${fmt(s.na, 2)}` : ""}` : "—";
    const lines = !s.configured
      ? ["empty"]
      : [
          magLine,
          ...(s.name !== `${fmt(s.mag)}x` ? [clip(s.name, Math.floor((body.width * K) / 5.8))] : []),
          ...(s.pixelSize > 0 ? [`${fmt(s.pixelSize, 3)} µm/px`] : []),
        ];
    return lines.map((text, k) => (
      <text
        key={`${s.i}-${k}`}
        x={body.axis * K}
        y={-body.labelZ * K + (k - (lines.length - 1) / 2) * 12}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={k === 0 && s.configured ? 11 : 9.5}
        fontWeight={k === 0 && s.configured ? 600 : 400}
        fill={s.configured ? ENGRAVING : line}
        style={{ fontVariantNumeric: "tabular-nums" }}
      >
        {text}
      </text>
    ));
  };

  const beamBody = inBeam && bodyOf(inBeam);
  return (
    <Box>
      <Box
        component="svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Objective turret. In beam: ${beamText}. A ${aText}. Z ${zText}.`}
        sx={{ width: "100%", maxWidth: 720, display: "block", mx: "auto" }}
      >
        <defs>
          <linearGradient id={`${id}-shade`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#000" stopOpacity={0.16} />
            <stop offset="0.35" stopColor="#fff" stopOpacity={0.14} />
            <stop offset="0.7" stopColor="#000" stopOpacity={0} />
            <stop offset="1" stopColor="#000" stopOpacity={0.24} />
          </linearGradient>
        </defs>

        {/* optical axis */}
        <line x1={AXIS_X} y1={6} x2={AXIS_X} y2={H - 6} stroke={line}
          strokeOpacity={0.5} strokeWidth={0.75} strokeDasharray="3 4" />

        {/* the turret rides on Z ... */}
        <g style={{ transform: `translate(0px, ${originY}px)`, transition }}>
          <rect x={12} y={-4.75 * K} width={W - 24} height={1.8 * K}
            fill={dark ? "#4d535a" : "#a3a9b0"} />
          {/* ... and its plate slides along A */}
          <g style={{ transform: `translate(${AXIS_X + plateX * K}px, 0px)`, transition }}>
            <g transform={`scale(${K} ${-K})`}>
              {TURRET_OUTLINE.pins.split(/(?=M)/).map((d, k) => part(`pin${k}`, d, COLOR.pins))}
              <path d={TURRET_OUTLINE.plate} fill={COLOR.plate} {...edge} />
              {slots.map(objective)}
              {beamBody && (
                <path
                  d={`M${beamBody.axis - LENS_R} ${beamBody.top} L${beamBody.axis} ${focusZ} L${beamBody.axis + LENS_R} ${beamBody.top} Z`}
                  fill={accent} fillOpacity={0.2} stroke={accent} strokeOpacity={0.6}
                  strokeWidth={0.75} vectorEffect="non-scaling-stroke"
                />
              )}
            </g>
            {slots.map(engraving)}
          </g>
        </g>

        {/* slide; the specimen sits on its top face */}
        <rect x={12} y={SAMPLE_Y} width={W - 24} height={1 * K}
          fill={dark ? "#33434d" : "#d4e5ee"} fillOpacity={0.85}
          stroke={dark ? "#62808f" : "#94b5c6"} strokeWidth={0.75} />
        <text x={W - 14} y={SAMPLE_Y - 5} textAnchor="end" fontSize={10} fill={line}>
          sample
        </text>
      </Box>
      <Box sx={{ display: "flex", gap: 3, rowGap: 0.5, flexWrap: "wrap", justifyContent: "center", mt: 1 }}>
        <Readout label="In beam" value={beamText} warn={!beamOk} />
        <Readout label="A" value={aText} />
        <Readout label="Z" value={zText} />
      </Box>
    </Box>
  );
}
