import React, { useRef, useState } from "react";
import { useSelector } from "react-redux";
import * as positionSlice from "../state/slices/PositionSlice.js";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";

// Served from public/ (CRA's SVG loader rejects the Inkscape namespaces)
const pcbUrl = `${process.env.PUBLIC_URL}/assets/shitscope_pcb.svg`;
// Geometry in the PCB drawing's mm (the SVG is A4, 1 user unit = 1 mm).
const VIEW = { x: -4, y: 100, w: 156, h: 62 };   // board 0..148 × 108.8..156.8 plus margin
const AXIS = { x: 74.04, y: 132.78 };             // centre of the optical-axis hole
const SLIDE = { w: 75, h: 25 };
// Slide top-left (board mm) at stage (0, 0), i.e. right after homing into the
// -X/-Y stop. Default is a guess; right-click "we are here" on the map
// overrides it (kept per browser). DIR: which way +X/+Y stage moves the slide.
const DEFAULT_SLIDE_AT_HOME = { x: AXIS.x - SLIDE.w + 5, y: AXIS.y - SLIDE.h + 5 };
const DIR = { x: 1, y: 1 };
const STORE_KEY = "shitscope.slideAtHome";

const loadSlideAtHome = () => {
  try {
    const v = JSON.parse(localStorage.getItem(STORE_KEY));
    if (v && isFinite(v.x) && isFinite(v.y)) return v;
  } catch (e) { /* storage blocked: default */ }
  return DEFAULT_SLIDE_AT_HOME;
};

/**
 * PCB stage overview: the board and optical axis stay put, the slide moves.
 * Tiles are drawn on the slide (done ones filled); click the slide to bring
 * that point under the optical axis.
 */
const ShitScopeStageMap = ({ tiles = [], doneTiles = 0, fovUm = { x: 0, y: 0 }, disabled = false }) => {
  const svgRef = useRef(null);
  const [home, setHome] = useState(loadSlideAtHome);
  const pos = useSelector(positionSlice.getPositionState);
  // Slide coordinate (mm, from its top-left) under the optical axis at stage p (µm)
  const underAxis = (p) => ({
    x: AXIS.x - home.x - (DIR.x * p.x) / 1000,
    y: AXIS.y - home.y - (DIR.y * p.y) / 1000,
  });
  const here = underAxis({ x: pos.x || 0, y: pos.y || 0 });
  const slideTL = { x: AXIS.x - here.x, y: AXIS.y - here.y };
  const fw = fovUm.x / 1000, fh = fovUm.y / 1000;

  const toBoard = (e) => {
    const pt = svgRef.current.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    return pt.matrixTransform(svgRef.current.getScreenCTM().inverse());
  };

  // Right-click "we are here": the clicked board point is what the camera
  // sees now, so shift the slide until that point sits on the optical axis.
  const handleWeAreHere = (e) => {
    e.preventDefault();
    if (!svgRef.current) return;
    const b = toBoard(e);
    const next = { x: home.x + AXIS.x - b.x, y: home.y + AXIS.y - b.y };
    setHome(next);
    try { localStorage.setItem(STORE_KEY, JSON.stringify(next)); } catch (err) { /* not persisted */ }
  };

  const handleClick = async (e) => {
    if (disabled || !svgRef.current) return;
    const b = toBoard(e);
    const s = { x: b.x - slideTL.x, y: b.y - slideTL.y };          // clicked slide point
    if (s.x < 0 || s.y < 0 || s.x > SLIDE.w || s.y > SLIDE.h) return;
    // stage position that puts s under the axis (inverse of underAxis)
    const x = ((AXIS.x - home.x - s.x) * 1000) / DIR.x;
    const y = ((AXIS.y - home.y - s.y) * 1000) / DIR.y;
    await apiPositionerControllerMovePositioner({ axis: "X", dist: Math.round(x), isAbsolute: true, isBlocking: true });
    await apiPositionerControllerMovePositioner({ axis: "Y", dist: Math.round(y), isAbsolute: true, isBlocking: true });
  };

  return (
    <svg ref={svgRef} viewBox={`${VIEW.x} ${VIEW.y} ${VIEW.w} ${VIEW.h}`} onClick={handleClick}
         onContextMenu={handleWeAreHere}
         style={{ width: "100%", background: "#fff", cursor: disabled ? "default" : "crosshair", borderRadius: 4 }}>
      <image href={pcbUrl} x={0} y={0} width={210} height={297} />
      <g transform={`translate(${slideTL.x} ${slideTL.y})`}>
        <rect width={SLIDE.w} height={SLIDE.h} fill="#9ecbff" fillOpacity={0.35} stroke="#1976d2" strokeWidth={0.3} />
        {tiles.map((t, k) => {
          const c = underAxis(t);
          return (
            <rect key={k} x={c.x - fw / 2} y={c.y - fh / 2} width={fw} height={fh}
                  fill={k < doneTiles ? "#4caf50" : "none"} fillOpacity={0.4}
                  stroke={k < doneTiles ? "#2e7d32" : "#ff9800"} strokeWidth={0.15} />
          );
        })}
      </g>
      <rect x={AXIS.x - fw / 2} y={AXIS.y - fh / 2} width={fw} height={fh} fill="none" stroke="#d32f2f" strokeWidth={0.3} />
      <circle cx={AXIS.x} cy={AXIS.y} r={0.4} fill="#d32f2f" />
      <text x={VIEW.x + 1} y={VIEW.y + 3} fontSize={2.5} fill="#555">
        X {Math.round(pos.x || 0)} µm · Y {Math.round(pos.y || 0)} µm · right-click: we are here
      </text>
    </svg>
  );
};

export default ShitScopeStageMap;
