/**
 * plannedFocusGrid.js
 *
 * Client-side mirror of the backend's automatic focus-map grid
 * (FocusMap.generate_grid in imswitch/imcontrol/model/focus_map.py), so the
 * planned autofocus positions can be previewed — and pruned — BEFORE the
 * automatic measurement runs. The kept points are sent to the backend as
 * `grid_points` in the FocusMapConfig; the backend then measures exactly
 * these positions instead of regenerating its own grid.
 */

import { calculateScanCoordinates } from "../CoordinateCalculator";

/** Stable identity of a planned point, used to remember user deletions. */
export const plannedPointKey = (pt) =>
  `${pt.groupId}|${pt.x.toFixed(1)}|${pt.y.toFixed(1)}`;

/**
 * Mirror of backend FocusMap.generate_grid: uniform rows×cols grid within
 * bounds, optional 5% inward margin, row-major (Y outer), deduplicated for
 * degenerate (single-FOV) bounds.
 *
 * @returns Array of [x, y] pairs in µm.
 */
export function generateGridForBounds(
  bounds,
  rows,
  cols,
  addMargin,
  marginFraction = 0.05,
) {
  let minX = bounds?.minX;
  let maxX = bounds?.maxX;
  let minY = bounds?.minY;
  let maxY = bounds?.maxY;
  if (
    [minX, maxX, minY, maxY].some(
      (v) => typeof v !== "number" || !isFinite(v),
    )
  ) {
    return [];
  }

  if (addMargin) {
    const dx = (maxX - minX) * marginFraction;
    const dy = (maxY - minY) * marginFraction;
    minX += dx;
    maxX -= dx;
    minY += dy;
    maxY -= dy;
  }

  const r = Math.max(1, Math.floor(rows) || 1);
  const c = Math.max(1, Math.floor(cols) || 1);

  const xs =
    c > 1
      ? Array.from({ length: c }, (_, i) => minX + (i * (maxX - minX)) / (c - 1))
      : [(minX + maxX) / 2];
  const ys =
    r > 1
      ? Array.from({ length: r }, (_, i) => minY + (i * (maxY - minY)) / (r - 1))
      : [(minY + maxY) / 2];

  const seen = new Set();
  const grid = [];
  for (const y of ys) {
    for (const x of xs) {
      const key = `${x.toFixed(3)}|${y.toFixed(3)}`;
      if (!seen.has(key)) {
        seen.add(key);
        grid.push([x, y]);
      }
    }
  }
  return grid;
}

/**
 * Compute the planned automatic focus-grid points for the current experiment
 * selection: one rows×cols grid per scan area (same per-area semantics as the
 * backend's computeFocusMap endpoint).
 *
 * @returns Array of { x, y, groupId, groupName }.
 */
export function computePlannedFocusPoints(
  experimentState,
  objectiveState,
  wellSelectorState,
  focusMapConfig,
) {
  if (!focusMapConfig?.enabled || focusMapConfig?.use_manual_map) return [];

  let scanAreas = [];
  try {
    const scanConfig = calculateScanCoordinates(
      experimentState,
      objectiveState,
      wellSelectorState,
    );
    scanAreas = scanConfig?.scanAreas || [];
  } catch (err) {
    console.warn("Could not compute planned focus grid:", err);
    return [];
  }

  const points = [];
  scanAreas.forEach((area) => {
    const grid = generateGridForBounds(
      area.bounds,
      focusMapConfig.rows,
      focusMapConfig.cols,
      focusMapConfig.add_margin,
    );
    grid.forEach(([x, y]) => {
      points.push({
        x,
        y,
        groupId: area.areaId,
        groupName: area.areaName || area.areaId,
      });
    });
  });
  return points;
}

/** Drop the points whose key is in removedKeys (user-deleted planned points). */
export function filterRemovedPlannedPoints(points, removedKeys) {
  if (!removedKeys || removedKeys.length === 0) return points;
  const removed = new Set(removedKeys);
  return points.filter((pt) => !removed.has(plannedPointKey(pt)));
}
