import React, { useCallback, useMemo, useRef, useState } from 'react';

/**
 * GalvoScanPreview - the raster tab's scan-pattern canvas.
 *
 * Draws the full 4096x4096 DAC range, the scan rectangle with its serpentine
 * path, and - once snapped - the 2D camera frame behind it, placed through the
 * camera->galvo affine transform so the rectangle shows what the beam will
 * actually cover. The rectangle can be dragged (move) and its corners pulled
 * (resize); both report back via `onAreaChange` in DAC counts.
 *
 * Props:
 *   config            { x_min, x_max, y_min, y_max, nx, ny, bidirectional }
 *   affine            { a11, a12, tx, a21, a22, ty } camera px -> DAC
 *   background        { image, width, height, frameWidth, frameHeight, subsampling } | null
 *   backgroundOpacity 0..1
 *   physical          getScanPhysical() result or null (for µm labels)
 *   onAreaChange      ({ x_min, x_max, y_min, y_max }) => void
 *   size              canvas edge in px (default 320)
 */

const DAC_RANGE = 4096;
const DAC_MAX = 4095;

const isIdentity = (a) =>
  !a || (a.a11 === 1 && a.a12 === 0 && a.tx === 0 && a.a21 === 0 && a.a22 === 1 && a.ty === 0);

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const fmtUm = (um) => {
  if (um == null || !isFinite(um)) return '';
  if (um >= 1000) return `${(um / 1000).toFixed(2)} mm`;
  if (um >= 10) return `${um.toFixed(0)} µm`;
  return `${um.toFixed(2)} µm`;
};

const GalvoScanPreview = ({
  config,
  affine,
  background = null,
  backgroundOpacity = 0.7,
  physical = null,
  onAreaChange,
  size = 320,
}) => {
  const svgRef = useRef(null);
  // Active drag: { mode: 'move' | 'nw' | 'ne' | 'sw' | 'se', start: {x, y} (DAC), area: {...} }
  const dragRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const padding = 25;
  const inner = size - 2 * padding;
  const k = inner / DAC_RANGE; // canvas px per DAC count
  const toCanvas = useCallback((dac) => padding + dac * k, [k]);

  // ---- pointer <-> DAC -------------------------------------------------
  const clientToDac = useCallback((clientX, clientY) => {
    const svg = svgRef.current;
    if (!svg) return null;
    let x, y;
    if (typeof svg.getScreenCTM === 'function' && svg.getScreenCTM()) {
      const pt = svg.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      x = p.x;
      y = p.y;
    } else {
      // jsdom / old browsers: assume the SVG is not CSS-scaled
      const rect = svg.getBoundingClientRect();
      x = (clientX - rect.left) * (size / (rect.width || size));
      y = (clientY - rect.top) * (size / (rect.height || size));
    }
    return { x: (x - padding) / k, y: (y - padding) / k };
  }, [k, size]);

  const beginDrag = useCallback((mode) => (e) => {
    if (!onAreaChange) return;
    const start = clientToDac(e.clientX, e.clientY);
    if (!start) return;
    e.preventDefault();
    e.stopPropagation();
    dragRef.current = {
      mode,
      start,
      area: { x_min: config.x_min, x_max: config.x_max, y_min: config.y_min, y_max: config.y_max },
    };
    setDragging(true);
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch (_) { /* jsdom */ }
  }, [clientToDac, config.x_min, config.x_max, config.y_min, config.y_max, onAreaChange]);

  const onPointerMove = useCallback((e) => {
    const drag = dragRef.current;
    if (!drag) return;
    const cur = clientToDac(e.clientX, e.clientY);
    if (!cur) return;
    const dx = cur.x - drag.start.x;
    const dy = cur.y - drag.start.y;
    const a = drag.area;
    let next;
    if (drag.mode === 'move') {
      // Slide, keep the size, stop at the edges
      const shift = (lo, hi, d) => {
        const w = hi - lo;
        const nlo = clamp(Math.round(lo + d), 0, DAC_MAX - w);
        return [nlo, nlo + w];
      };
      const [x_min, x_max] = shift(a.x_min, a.x_max, dx);
      const [y_min, y_max] = shift(a.y_min, a.y_max, dy);
      next = { x_min, x_max, y_min, y_max };
    } else {
      next = { ...a };
      if (drag.mode.includes('w')) next.x_min = clamp(Math.round(a.x_min + dx), 0, a.x_max - 1);
      if (drag.mode.includes('e')) next.x_max = clamp(Math.round(a.x_max + dx), a.x_min + 1, DAC_MAX);
      if (drag.mode.includes('n')) next.y_min = clamp(Math.round(a.y_min + dy), 0, a.y_max - 1);
      if (drag.mode.includes('s')) next.y_max = clamp(Math.round(a.y_max + dy), a.y_min + 1, DAC_MAX);
    }
    if (next.x_min !== config.x_min || next.x_max !== config.x_max ||
        next.y_min !== config.y_min || next.y_max !== config.y_max) {
      onAreaChange(next);
    }
  }, [clientToDac, config.x_min, config.x_max, config.y_min, config.y_max, onAreaChange]);

  const endDrag = useCallback((e) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    setDragging(false);
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch (_) { /* jsdom */ }
  }, []);

  // ---- geometry ----------------------------------------------------------
  const scanLeft = toCanvas(config.x_min);
  const scanRight = toCanvas(config.x_max);
  const scanTop = toCanvas(config.y_min);
  const scanBottom = toCanvas(config.y_max);
  const scanWidth = scanRight - scanLeft;
  const scanHeight = scanBottom - scanTop;

  // Serpentine path (first 64x64 samples max, to keep the SVG light)
  const { pathPoints, stepY, previewNy } = useMemo(() => {
    const maxPreviewPoints = 64;
    const previewNx = Math.min(config.nx, maxPreviewPoints);
    const previewNyL = Math.min(config.ny, maxPreviewPoints);
    const stepX = scanWidth / Math.max(previewNx - 1, 1);
    const stepYL = scanHeight / Math.max(previewNyL - 1, 1);
    const pts = [];
    for (let y = 0; y < previewNyL; y++) {
      const yPos = scanTop + y * stepYL;
      const isReverse = config.bidirectional && y % 2 === 1;
      for (let x = 0; x < previewNx; x++) {
        const xIdx = isReverse ? (previewNx - 1 - x) : x;
        pts.push({ x: scanLeft + xIdx * stepX, y: yPos });
      }
    }
    return { pathPoints: pts, stepY: stepYL, previewNy: previewNyL };
  }, [config.nx, config.ny, config.bidirectional, scanLeft, scanTop, scanWidth, scanHeight]);

  const gridLines = useMemo(() => {
    const lines = [];
    for (let i = 0; i <= 4; i++) {
      lines.push({ pos: padding + (i / 4) * inner, dacVal: Math.round((i / 4) * DAC_RANGE) });
    }
    return lines;
  }, [inner]);

  // Camera frame -> canvas as one SVG matrix. Image px * subsampling = camera
  // frame px; the affine takes frame px to DAC; k takes DAC to canvas.
  // Without a calibration the frame is stretched over the full DAC range,
  // the same assumption the Arbitrary Points tab makes.
  const bgTransform = useMemo(() => {
    if (!background?.image) return null;
    const sub = background.subsampling || (background.frameWidth / background.width) || 1;
    let a11, a12, a21, a22, tx, ty;
    if (isIdentity(affine)) {
      a11 = DAC_MAX / Math.max(1, background.frameWidth);
      a22 = DAC_MAX / Math.max(1, background.frameHeight);
      a12 = a21 = tx = ty = 0;
    } else {
      ({ a11, a12, a21, a22, tx, ty } = affine);
    }
    const m = [
      k * sub * a11, k * sub * a21,
      k * sub * a12, k * sub * a22,
      padding + k * tx, padding + k * ty,
    ];
    return `matrix(${m.map((v) => v.toFixed(5)).join(' ')})`;
  }, [background, affine, k]);

  const calibrated = !isIdentity(affine);
  const spanX = config.x_max - config.x_min;
  const spanY = config.y_max - config.y_min;
  const sizeLabel = physical
    ? `${spanX}×${spanY} ≈ ${fmtUm(physical.fovUmX)} × ${fmtUm(physical.fovUmY)}`
    : `${spanX}×${spanY} DAC`;

  const handleR = 5;
  const handles = [
    { mode: 'nw', x: scanLeft, y: scanTop, cursor: 'nwse-resize' },
    { mode: 'ne', x: scanRight, y: scanTop, cursor: 'nesw-resize' },
    { mode: 'sw', x: scanLeft, y: scanBottom, cursor: 'nesw-resize' },
    { mode: 'se', x: scanRight, y: scanBottom, cursor: 'nwse-resize' },
  ];

  return (
    <svg
      ref={svgRef}
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      data-testid="galvo-scan-preview"
      style={{
        border: '1px solid #444',
        borderRadius: 4,
        backgroundColor: '#0a0a15',
        touchAction: 'none',
        userSelect: 'none',
        maxWidth: '100%',
        height: 'auto',
      }}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onPointerLeave={endDrag}
    >
      <defs>
        <clipPath id="galvo-dac-clip">
          <rect x={padding} y={padding} width={inner} height={inner} />
        </clipPath>
      </defs>

      {/* Full DAC range */}
      <rect x={padding} y={padding} width={inner} height={inner} fill="#12121f" stroke="#333" strokeWidth={1} />

      {/* Camera backdrop through the affine */}
      {bgTransform && (
        <g clipPath="url(#galvo-dac-clip)">
          <image
            href={background.image}
            x={0}
            y={0}
            width={background.width}
            height={background.height}
            preserveAspectRatio="none"
            transform={bgTransform}
            opacity={backgroundOpacity}
            style={{ imageRendering: 'auto' }}
            data-testid="galvo-preview-background"
          />
          {/* Camera frame outline so the FOV is visible even where the image is dark */}
          <rect
            x={0}
            y={0}
            width={background.width}
            height={background.height}
            transform={bgTransform}
            fill="none"
            stroke={calibrated ? '#ffcc00' : '#888'}
            strokeWidth={1}
            strokeDasharray={calibrated ? '' : '3,3'}
            vectorEffect="non-scaling-stroke"
            opacity={0.8}
          />
        </g>
      )}

      {/* Grid */}
      {gridLines.map((line, i) => (
        <React.Fragment key={i}>
          <line x1={line.pos} y1={padding} x2={line.pos} y2={size - padding} stroke="#2a2a4a" strokeWidth={0.5} />
          <line x1={padding} y1={line.pos} x2={size - padding} y2={line.pos} stroke="#2a2a4a" strokeWidth={0.5} />
          <text x={line.pos} y={size - 5} fontSize={8} fill="#666" textAnchor="middle">{line.dacVal}</text>
          <text x={5} y={line.pos + 3} fontSize={8} fill="#666" textAnchor="start">{line.dacVal}</text>
        </React.Fragment>
      ))}

      {/* Scan path */}
      {pathPoints.length > 1 && (
        <polyline
          points={pathPoints.map((p) => `${p.x},${p.y}`).join(' ')}
          fill="none"
          stroke={config.bidirectional ? '#ff9900' : '#00ff88'}
          strokeWidth={1}
          opacity={0.8}
          pointerEvents="none"
        />
      )}
      {pathPoints.slice(0, 200).map((point, i) => (
        <circle
          key={i}
          cx={point.x}
          cy={point.y}
          r={Math.max(1, 3 - pathPoints.length / 50)}
          fill={i === 0 ? '#ff0000' : '#00aaff'}
          pointerEvents="none"
        />
      ))}
      {pathPoints.length > 0 && (
        <circle cx={pathPoints[0].x} cy={pathPoints[0].y} r={5} fill="none" stroke="#ff0000" strokeWidth={2} pointerEvents="none" />
      )}

      {/* Direction arrows for bidirectional */}
      {config.bidirectional && previewNy >= 2 && (
        <>
          <polygon
            points={`${scanRight - 8},${scanTop - 2} ${scanRight},${scanTop + 4} ${scanRight - 8},${scanTop + 10}`}
            fill="#00ff88"
            pointerEvents="none"
          />
          <polygon
            points={`${scanLeft + 8},${scanTop + stepY - 2} ${scanLeft},${scanTop + stepY + 4} ${scanLeft + 8},${scanTop + stepY + 10}`}
            fill="#ff9900"
            pointerEvents="none"
          />
        </>
      )}

      {/* Scan area - draggable */}
      <rect
        x={scanLeft}
        y={scanTop}
        width={Math.max(scanWidth, 1)}
        height={Math.max(scanHeight, 1)}
        fill="rgba(0, 150, 255, 0.15)"
        stroke="#0096ff"
        strokeWidth={2}
        strokeDasharray="4,2"
        style={{ cursor: onAreaChange ? (dragging ? 'grabbing' : 'grab') : 'default' }}
        onPointerDown={beginDrag('move')}
        data-testid="galvo-scan-area"
      />
      {/* Corner handles - resize */}
      {onAreaChange && handles.map((h) => (
        <rect
          key={h.mode}
          x={h.x - handleR}
          y={h.y - handleR}
          width={2 * handleR}
          height={2 * handleR}
          fill="#0096ff"
          stroke="#fff"
          strokeWidth={1}
          style={{ cursor: h.cursor }}
          onPointerDown={beginDrag(h.mode)}
          data-testid={`galvo-scan-handle-${h.mode}`}
        />
      ))}

      {/* Size label under the rectangle (or above, near the bottom edge) */}
      <text
        x={(scanLeft + scanRight) / 2}
        y={scanBottom + 12 < size - padding ? scanBottom + 12 : scanTop - 6}
        fontSize={9}
        fill="#8fd0ff"
        textAnchor="middle"
        pointerEvents="none"
      >
        {sizeLabel}
      </text>

      {/* Title */}
      <text x={size / 2} y={12} fontSize={10} fill="#888" textAnchor="middle" fontWeight="bold">
        {physical
          ? `DAC 0–4095 ≈ ${fmtUm(physical.fullScaleUmX)} × ${fmtUm(physical.fullScaleUmY)}`
          : 'DAC Range: 0-4095'}
      </text>

      {/* Scan mode chip */}
      <rect x={size - 85} y={2} width={80} height={16} rx={3} fill={config.bidirectional ? '#ff9900' : '#00ff88'} opacity={0.3} />
      <text
        x={size - 45}
        y={13}
        fontSize={9}
        fill={config.bidirectional ? '#ff9900' : '#00ff88'}
        textAnchor="middle"
        fontWeight="bold"
      >
        {config.bidirectional ? 'BIDI' : 'UNI'}
      </text>
    </svg>
  );
};

export default GalvoScanPreview;
