import React, { useCallback, useRef } from "react";
import { Box, useTheme } from "@mui/material";

// Oblique/perspective animation of the Z-stack: a tapered sample chamber
// (frustum) sitting above the objective (drawn from below, inverted-scope
// convention), a Z ruler with First/Last ticks, a highlighted focal plane
// that tracks the live stage Z, and optional tick markers for individual
// slice positions. Clicking/dragging anywhere in the chamber commands the
// stage to that Z — the same "click to move" pattern the XY wellplate
// canvas uses (see WellSelectorCanvas.js's MOVE_CAMERA click handler),
// just projected onto a single vertical axis.
//
// All layout constants below are in SVG viewBox units (240 x 320).
const VIEW_W = 240;
// Tall enough that the objective + its label (drawn below CHAMBER_BOTTOM_Y)
// fit inside the viewBox instead of being clipped.
const VIEW_H = 340;
const CHAMBER_TOP_Y = 24; // y of the highest Z (top of drawing)
const CHAMBER_BOTTOM_Y = 250; // y of the lowest Z, just above the objective
const CHAMBER_TOP_HALF_W = 60; // chamber is wider at the top (tapered frustum)
const CHAMBER_BOTTOM_HALF_W = 50;
const CHAMBER_CX = 100;
const RULER_X = 190;

const ZStackVisualization = ({
  firstAbs,
  lastAbs,
  currentAbs,
  slicesAbs = [],
  onSeek,
  height = 340,
}) => {
  const theme = useTheme();
  const svgRef = useRef(null);

  const zMin = Math.min(firstAbs, lastAbs);
  const zMax = Math.max(firstAbs, lastAbs);
  // Guard against a zero-range stack (First === Last): fall back to a small
  // symmetric window purely for the drawing so the ruler/chamber don't
  // collapse to a line.
  const zSpan = zMax - zMin > 1e-6 ? zMax - zMin : 1;
  const drawMin = zMax - zMin > 1e-6 ? zMin : zMin - zSpan / 2;
  const drawMax = zMax - zMin > 1e-6 ? zMax : zMax + zSpan / 2;

  // z -> y (top of drawing = drawMax, bottom = drawMin)
  const zToY = useCallback(
    (z) => {
      const t = (z - drawMin) / (drawMax - drawMin);
      return CHAMBER_BOTTOM_Y - t * (CHAMBER_BOTTOM_Y - CHAMBER_TOP_Y);
    },
    [drawMin, drawMax],
  );

  // y -> z, clamped to the chamber's drawn extent plus a small overshoot
  // margin so a stray click can't send the stage far outside the stack.
  const yToZ = useCallback(
    (y) => {
      const t = (CHAMBER_BOTTOM_Y - y) / (CHAMBER_BOTTOM_Y - CHAMBER_TOP_Y);
      return drawMin + t * (drawMax - drawMin);
    },
    [drawMin, drawMax],
  );

  const clampSeek = useCallback(
    (z) => {
      const margin = Math.max(zSpan * 0.1, 5);
      return Math.min(zMax + margin, Math.max(zMin - margin, z));
    },
    [zMin, zMax, zSpan],
  );

  const handlePointer = useCallback(
    (e) => {
      if (!onSeek || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      const fracY = (clientY - rect.top) / rect.height;
      const y = fracY * VIEW_H;
      onSeek(clampSeek(yToZ(y)));
    },
    [onSeek, yToZ, clampSeek],
  );

  // Half-width of the chamber at a given y (linear taper top -> bottom).
  const halfWidthAtY = (y) => {
    const t = (y - CHAMBER_TOP_Y) / (CHAMBER_BOTTOM_Y - CHAMBER_TOP_Y);
    return CHAMBER_TOP_HALF_W + t * (CHAMBER_BOTTOM_HALF_W - CHAMBER_TOP_HALF_W);
  };

  const currentY = zToY(Math.min(drawMax, Math.max(drawMin, currentAbs ?? drawMin)));
  const firstY = zToY(firstAbs);
  const lastY = zToY(lastAbs);

  const accentColor = theme.palette.primary.main;

  return (
    <Box sx={{ width: "100%", display: "flex", justifyContent: "center" }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        width="100%"
        height={height}
        style={{ maxWidth: 260, cursor: onSeek ? "pointer" : "default", touchAction: "none" }}
        role="img"
        aria-label="Z-stack focus position within the sample"
        onPointerDown={(e) => {
          // Handles both a plain click and drag-to-seek. Deliberately not
          // paired with onClick — click always fires after pointerdown/up on
          // the same target, which would otherwise send the same move twice.
          e.currentTarget.setPointerCapture?.(e.pointerId);
          handlePointer(e);
        }}
        onPointerMove={(e) => {
          if (e.buttons === 1) handlePointer(e);
        }}
      >
        {/* Sample chamber (tapered frustum, oblique/perspective look) */}
        <path
          d={`M ${CHAMBER_CX - CHAMBER_TOP_HALF_W} ${CHAMBER_TOP_Y}
              L ${CHAMBER_CX + CHAMBER_TOP_HALF_W} ${CHAMBER_TOP_Y}
              L ${CHAMBER_CX + CHAMBER_BOTTOM_HALF_W} ${CHAMBER_BOTTOM_Y}
              L ${CHAMBER_CX - CHAMBER_BOTTOM_HALF_W} ${CHAMBER_BOTTOM_Y} Z`}
          fill={theme.palette.mode === "dark" ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"}
          stroke="currentColor"
          strokeWidth="1.5"
          opacity="0.7"
        />
        {/* Top/bottom rim ellipses for the 3D-glass feel */}
        <ellipse
          cx={CHAMBER_CX}
          cy={CHAMBER_TOP_Y}
          rx={CHAMBER_TOP_HALF_W}
          ry="7"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.2"
          opacity="0.5"
        />
        <ellipse
          cx={CHAMBER_CX}
          cy={CHAMBER_BOTTOM_Y}
          rx={CHAMBER_BOTTOM_HALF_W}
          ry="5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.2"
          opacity="0.5"
        />

        {/* First / Last bound lines */}
        {[
          { y: firstY, label: "First" },
          { y: lastY, label: "Last" },
        ].map(({ y, label }) => (
          <g key={label} opacity="0.55">
            <line
              x1={CHAMBER_CX - halfWidthAtY(y)}
              y1={y}
              x2={CHAMBER_CX + halfWidthAtY(y)}
              y2={y}
              stroke="currentColor"
              strokeWidth="1"
              strokeDasharray="4,3"
            />
          </g>
        ))}

        {/* Individual slice markers */}
        {slicesAbs.map((z, i) => {
          const y = zToY(z);
          return (
            <line
              key={i}
              x1={CHAMBER_CX - halfWidthAtY(y)}
              y1={y}
              x2={CHAMBER_CX + halfWidthAtY(y)}
              y2={y}
              stroke="#e0a020"
              strokeWidth="2"
              opacity="0.85"
            />
          );
        })}

        {/* Live focal plane — animates smoothly as the stage moves */}
        <g style={{ transform: `translateY(${currentY}px)`, transition: "transform 0.25s ease-out" }}>
          <ellipse
            cx={CHAMBER_CX}
            cy={0}
            rx={halfWidthAtY(currentY)}
            ry="6"
            fill={accentColor}
            opacity="0.35"
          />
          <line
            x1={CHAMBER_CX - halfWidthAtY(currentY)}
            y1={0}
            x2={CHAMBER_CX + halfWidthAtY(currentY)}
            y2={0}
            stroke={accentColor}
            strokeWidth="2.5"
          />
        </g>

        {/* Objective, viewed from below (inverted microscope) */}
        <path
          d={`M ${CHAMBER_CX - 10} ${CHAMBER_BOTTOM_Y + 18}
              L ${CHAMBER_CX + 10} ${CHAMBER_BOTTOM_Y + 18}
              L ${CHAMBER_CX + 22} ${CHAMBER_BOTTOM_Y + 35}
              L ${CHAMBER_CX - 22} ${CHAMBER_BOTTOM_Y + 35} Z`}
          fill="currentColor"
          opacity="0.65"
        />
        <path
          d={`M ${CHAMBER_CX - 22} ${CHAMBER_BOTTOM_Y + 35}
              L ${CHAMBER_CX + 22} ${CHAMBER_BOTTOM_Y + 35}
              L ${CHAMBER_CX + 22} ${CHAMBER_BOTTOM_Y + 70}
              L ${CHAMBER_CX - 22} ${CHAMBER_BOTTOM_Y + 70} Z`}
          fill="currentColor"
          opacity="0.65"
        />
        <circle
          cx={CHAMBER_CX}
          cy={CHAMBER_BOTTOM_Y + 18}
          r="7"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
        <text
          x={CHAMBER_CX}
          y={CHAMBER_BOTTOM_Y + 78}
          fill="currentColor"
          fontSize="10"
          textAnchor="middle"
          opacity="0.7"
        >
          Objective
        </text>

        {/* Z ruler */}
        <line
          x1={RULER_X}
          y1={CHAMBER_TOP_Y}
          x2={RULER_X}
          y2={CHAMBER_BOTTOM_Y}
          stroke="currentColor"
          strokeWidth="1"
          opacity="0.4"
        />
        <line x1={RULER_X - 4} y1={firstY} x2={RULER_X + 4} y2={firstY} stroke="currentColor" strokeWidth="1.5" />
        <text x={RULER_X + 8} y={firstY + 3} fill="currentColor" fontSize="9" opacity="0.8">
          {firstAbs.toFixed(1)}
        </text>
        <line x1={RULER_X - 4} y1={lastY} x2={RULER_X + 4} y2={lastY} stroke="currentColor" strokeWidth="1.5" />
        <text x={RULER_X + 8} y={lastY + 3} fill="currentColor" fontSize="9" opacity="0.8">
          {lastAbs.toFixed(1)}
        </text>
      </svg>
    </Box>
  );
};

export default ZStackVisualization;
