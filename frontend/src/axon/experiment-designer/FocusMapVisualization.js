import React, { useRef, useEffect, useMemo } from "react";
import { Box, Typography } from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";

/**
 * FocusMapVisualization – Canvas-based visualization of measured focus points
 * and the fitted Z-surface as a colour-coded heatmap.
 *
 * Props:
 *   data.measured_points: [{ x, y, z }]   – autofocus measurements
 *   data.preview_grid: { x: [], y: [], z: [[]] }  – fitted surface on regular grid
 *   data.fit_stats: { method, z_range, mean_abs_error, ... }
 */

const CANVAS_SIZE = 320;
const PADDING = 40;

// Colour ramp: blue (low Z) → green → yellow → red (high Z)
const colorForValue = (t) => {
  // t in [0,1]
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 * t - 0.25)));
  const g = Math.round(255 * Math.min(1, Math.max(0, t < 0.5 ? 2 * t : 2 - 2 * t)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 1.25 - 1.5 * t)));
  return `rgb(${r},${g},${b})`;
};

const FocusMapVisualization = ({ data }) => {
  const theme = useTheme();
  const canvasRef = useRef(null);

  // Extract bounds from measured points or preview grid
  const { measured_points = [], preview_grid, fit_stats } = data || {};

  // Compute data bounds
  const bounds = useMemo(() => {
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    let zMin = Infinity, zMax = -Infinity;

    measured_points.forEach((p) => {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
      if (p.z < zMin) zMin = p.z;
      if (p.z > zMax) zMax = p.z;
    });

    if (preview_grid && preview_grid.z) {
      const flatZ = preview_grid.z.flat();
      flatZ.forEach((z) => {
        if (z < zMin) zMin = z;
        if (z > zMax) zMax = z;
      });
      if (preview_grid.x && preview_grid.x.length) {
        const gxMin = Math.min(...preview_grid.x);
        const gxMax = Math.max(...preview_grid.x);
        const gyMin = Math.min(...preview_grid.y);
        const gyMax = Math.max(...preview_grid.y);
        if (gxMin < xMin) xMin = gxMin;
        if (gxMax > xMax) xMax = gxMax;
        if (gyMin < yMin) yMin = gyMin;
        if (gyMax > yMax) yMax = gyMax;
      }
    }

    // Prevent zero-range
    if (xMax === xMin) { xMin -= 1; xMax += 1; }
    if (yMax === yMin) { yMin -= 1; yMax += 1; }
    if (zMax === zMin) { zMin -= 0.5; zMax += 0.5; }

    return { xMin, xMax, yMin, yMax, zMin, zMax };
  }, [measured_points, preview_grid]);

  // Map world → canvas coordinates
  const toCanvas = (x, y) => {
    const cx = PADDING + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * (CANVAS_SIZE - 2 * PADDING);
    const cy = PADDING + ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * (CANVAS_SIZE - 2 * PADDING);
    return [cx, cy];
  };

  const normalizeZ = (z) => {
    return (z - bounds.zMin) / (bounds.zMax - bounds.zMin);
  };

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_SIZE * dpr;
    canvas.height = CANVAS_SIZE * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = theme.palette.background.paper;
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Draw heatmap from preview_grid
    if (preview_grid && preview_grid.x && preview_grid.y && preview_grid.z) {
      const nx = preview_grid.x.length;
      const ny = preview_grid.y.length;
      const drawW = CANVAS_SIZE - 2 * PADDING;
      const drawH = CANVAS_SIZE - 2 * PADDING;
      const cellW = drawW / (nx - 1 || 1);
      const cellH = drawH / (ny - 1 || 1);

      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const z = preview_grid.z[iy]?.[ix];
          if (z == null) continue;
          const t = normalizeZ(z);
          ctx.fillStyle = colorForValue(t);
          const [cx, cy] = toCanvas(preview_grid.x[ix], preview_grid.y[iy]);
          ctx.fillRect(cx - cellW / 2, cy - cellH / 2, cellW, cellH);
        }
      }
    }

    // Draw measured points as circles with Z-coloured fill
    measured_points.forEach((p) => {
      const [cx, cy] = toCanvas(p.x, p.y);
      const t = normalizeZ(p.z);
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = colorForValue(t);
      ctx.fill();
      ctx.strokeStyle = theme.palette.text.primary;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Axes labels
    ctx.fillStyle = theme.palette.text.secondary;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`X: ${bounds.xMin.toFixed(0)} – ${bounds.xMax.toFixed(0)}`, CANVAS_SIZE / 2, CANVAS_SIZE - 4);
    ctx.save();
    ctx.translate(12, CANVAS_SIZE / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`Y: ${bounds.yMin.toFixed(0)} – ${bounds.yMax.toFixed(0)}`, 0, 0);
    ctx.restore();

    // Colour bar legend
    const barX = CANVAS_SIZE - 18;
    const barY = PADDING;
    const barH = CANVAS_SIZE - 2 * PADDING;
    for (let i = 0; i < barH; i++) {
      const t = 1 - i / barH;
      ctx.fillStyle = colorForValue(t);
      ctx.fillRect(barX, barY + i, 10, 1);
    }
    ctx.fillStyle = theme.palette.text.secondary;
    ctx.font = "9px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${bounds.zMax.toFixed(1)}`, barX - 4, barY - 4);
    ctx.fillText(`${bounds.zMin.toFixed(1)}`, barX - 4, barY + barH + 10);

  }, [data, bounds, theme]);

  if (!data || (measured_points.length === 0 && !preview_grid)) {
    return (
      <Box sx={{ p: 2, textAlign: "center" }}>
        <Typography variant="body2" color="text.secondary">
          No focus map data to display.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <canvas
        ref={canvasRef}
        style={{
          width: CANVAS_SIZE,
          height: CANVAS_SIZE,
          border: `1px solid ${theme.palette.divider}`,
          borderRadius: 4,
        }}
      />
      {/* Stats below canvas */}
      {fit_stats && (
        <Box sx={{ mt: 1, display: "flex", gap: 2, flexWrap: "wrap" }}>
          <Typography variant="caption" color="text.secondary">
            Method: <strong>{fit_stats.method}</strong>
          </Typography>
          {fit_stats.z_range != null && (
            <Typography variant="caption" color="text.secondary">
              Z range: <strong>{fit_stats.z_range.toFixed(2)} µm</strong>
            </Typography>
          )}
          {fit_stats.mean_abs_error != null && (
            <Typography variant="caption" color="text.secondary">
              MAE: <strong>{fit_stats.mean_abs_error.toFixed(3)} µm</strong>
            </Typography>
          )}
          {fit_stats.n_points != null && (
            <Typography variant="caption" color="text.secondary">
              Points: <strong>{fit_stats.n_points}</strong>
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
};

export default FocusMapVisualization;
