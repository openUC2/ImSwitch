/**
 * CoordinateCalculator.js
 * 
 * Centralized coordinate calculation for microscope scanning experiments.
 * Handles all tile generation, scan pattern ordering, and well association.
 * The backend receives pre-calculated, ordered coordinates and simply executes them.
 */

import * as wsUtils from './WellSelectorUtils';

/**
 * Main function to calculate all scan coordinates from experiment configuration
 * 
 * @param {Object} experimentState - Redux experiment state
 * @param {Object} objectiveState - Redux objective state
 * @param {Object} wellSelectorState - Redux well selector state
 * @returns {Object} Complete scan configuration with ordered coordinates and metadata
 */
export function calculateScanCoordinates(experimentState, objectiveState, wellSelectorState) {
  // Use experiment state is_snakescan as primary, with wellSelectorState as fallback
  // This ensures TilingDimension snake toggle takes precedence
  const isSnakeScan = experimentState.parameterValue.is_snakescan ?? wellSelectorState.areaSelectSnakescan ?? false;
  
  const scanConfig = {
    version: "1.0",
    scanAreas: [],
    metadata: {
      totalPositions: 0,
      fovX: objectiveState.fovX,
      fovY: objectiveState.fovY,
      overlapWidth: experimentState.parameterValue.overlapWidth || 0,
      overlapHeight: experimentState.parameterValue.overlapHeight || 0,
      scanPattern: isSnakeScan ? "snake" : "raster"
    }
  };

  // Calculate effective step sizes with overlap
  const effectiveStepX = objectiveState.fovX * (1 - scanConfig.metadata.overlapWidth);
  const effectiveStepY = objectiveState.fovY * (1 - scanConfig.metadata.overlapHeight);

  // Process each point in the point list
  experimentState.pointList.forEach((point, pointIndex) => {
    const scanArea = processScanPoint(
      point, 
      pointIndex, 
      experimentState, 
      objectiveState, 
      wellSelectorState,
      effectiveStepX,
      effectiveStepY
    );
    
    if (scanArea && scanArea.positions.length > 0) {
      scanConfig.scanAreas.push(scanArea);
      scanConfig.metadata.totalPositions += scanArea.positions.length;
    }
  });

  // Apply snake/raster ordering to the AREAS themselves (not just tiles within each area).
  // This reorders scanAreas so the stage travels efficiently across multi-area experiments.
  if (scanConfig.scanAreas.length > 1) {
    scanConfig.scanAreas = applyAreaLevelScanPattern(scanConfig.scanAreas, isSnakeScan, effectiveStepY);
  }

  return scanConfig;
}

/**
 * Process a single scan point and generate all positions within it
 * 
 * @param {Object} point - Point from experiment point list
 * @param {number} pointIndex - Index of the point
 * @param {Object} experimentState - Redux experiment state
 * @param {Object} objectiveState - Redux objective state
 * @param {Object} wellSelectorState - Redux well selector state
 * @param {number} effectiveStepX - Effective step size in X (with overlap)
 * @param {number} effectiveStepY - Effective step size in Y (with overlap)
 * @returns {Object} Scan area object with ordered positions
 */
function processScanPoint(point, pointIndex, experimentState, objectiveState, wellSelectorState, effectiveStepX, effectiveStepY) {
  // Find associated wells if any
  const associatedWells = point.wellMode 
    ? wsUtils.findWellsAtPosition(point, experimentState.wellLayout)
    : [];

  // Determine area name and type
  let areaName = point.name || `Area_${pointIndex + 1}`;
  let areaType = "free_scan";
  let wellId = null;

  if (associatedWells.length > 0) {
    const well = associatedWells[0];
    wellId = well.id || well.name;
    areaName = `Well_${wellId}_${pointIndex}`;
    areaType = "well";
  }

  // Calculate positions based on point shape and mode
  let rawPositions = [];
  
  if (point.shape === "off" || !point.shape) {
    // Single position
    rawPositions = [{
      x: point.x,
      y: point.y,
      z: point.z ?? 0,
      iX: 0,
      iY: 0
    }];
  } else if (point.wellMode === "center_only") {
    // Well center positions
    const centerPositions = wsUtils.generateWellCenterPositions(associatedWells, experimentState.wellLayout);
    rawPositions = centerPositions;
  } else if (point.wellMode === "pattern" && associatedWells.length > 0) {
    // Pattern within wells
    rawPositions = generateWellPattern(
      point, 
      associatedWells, 
      experimentState, 
      objectiveState, 
      effectiveStepX, 
      effectiveStepY
    );
  } else if (point.shape === "circle") {
    // Circular scan area
    rawPositions = wsUtils.calculateRasterOval(
      point,
      effectiveStepX,
      effectiveStepY,
      point.circleRadiusX,
      point.circleRadiusY
    );
  } else if (point.shape === "rectangle") {
    // Rectangular scan area
    rawPositions = wsUtils.calculateRasterRect(
      point,
      effectiveStepX,
      effectiveStepY,
      point.rectPlusX,
      point.rectMinusX,
      point.rectPlusY,
      point.rectMinusY
    );
  }

  // Apply scan pattern ordering (snake vs raster)
  // Use experiment state is_snakescan as primary, with wellSelectorState as fallback
  const isSnakeScan = experimentState.parameterValue.is_snakescan ?? wellSelectorState.areaSelectSnakescan ?? false;
  const orderedPositions = applyScanPattern(
    rawPositions,
    isSnakeScan
  );

  // Calculate bounding box for metadata
  const bounds = calculateBounds(orderedPositions);

  return {
    areaId: `area_${pointIndex}`,
    areaName: areaName,
    areaType: areaType,
    wellId: wellId,
    centerPosition: {
      x: point.x,
      y: point.y,
      z: point.z ?? 0
    },
    bounds: bounds,
    scanPattern: isSnakeScan ? "snake" : "raster",
    positions: orderedPositions.map((pos, idx) => ({
      index: idx,
      x: pos.x,
      y: pos.y,
      z: pos.z ?? (point.z ?? 0),
      iX: pos.iX,
      iY: pos.iY
    }))
  };
}

/**
 * Generate pattern positions within wells
 */
function generateWellPattern(point, wells, experimentState, objectiveState, effectiveStepX, effectiveStepY) {
  let positions = [];
  
  if (point.patternType === "circle") {
    positions = wsUtils.generateWellCirclePattern(
      wells,
      point.patternRadius || 50,
      point.patternOverlap || 0.1,
      effectiveStepX
    );
  } else if (point.patternType === "rectangle") {
    positions = wsUtils.generateWellRectanglePattern(
      wells,
      point.patternWidth || 100,
      point.patternHeight || 100,
      point.patternOverlap || 0.1,
      effectiveStepX
    );
  }
  
  return positions;
}

/**
 * Apply scan pattern ordering to positions
 * 
 * @param {Array} positions - Raw positions with iX, iY indices
 * @param {boolean} isSnakeScan - True for snake pattern, false for raster
 * @returns {Array} Ordered positions
 */
function applyScanPattern(positions, isSnakeScan) {
  if (!positions || positions.length === 0) {
    return [];
  }

  // Group by rows (iY coordinate)
  const rows = {};
  positions.forEach(pos => {
    const rowKey = pos.iY;
    if (!rows[rowKey]) {
      rows[rowKey] = [];
    }
    rows[rowKey].push(pos);
  });

  // Sort each row by iX
  Object.keys(rows).forEach(rowKey => {
    rows[rowKey].sort((a, b) => a.iX - b.iX);
  });

  // Sort row keys numerically
  const sortedRowKeys = Object.keys(rows).map(Number).sort((a, b) => a - b);

  // Build ordered list
  const orderedPositions = [];
  sortedRowKeys.forEach((rowKey, rowIndex) => {
    const row = rows[rowKey];
    
    if (isSnakeScan && rowIndex % 2 === 1) {
      // Reverse odd rows for snake pattern
      orderedPositions.push(...row.reverse());
    } else {
      orderedPositions.push(...row);
    }
  });

  return orderedPositions;
}

/**
 * Calculate bounding box for a set of positions
 */
function calculateBounds(positions) {
  if (!positions || positions.length === 0) {
    return { minX: 0, maxX: 0, minY: 0, maxY: 0, width: 0, height: 0 };
  }

  const xs = positions.map(p => p.x);
  const ys = positions.map(p => p.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return {
    minX,
    maxX,
    minY,
    maxY,
    width: maxX - minX,
    height: maxY - minY
  };
}

/**
 * Apply scan-pattern ordering at the *area* level.
 *
 * Areas are grouped into rows by quantising their center-Y coordinate
 * (bucket size = effectiveStepY, falling back to 10 % of the total Y span).
 * Within each row the areas are sorted left-to-right by center-X.
 * For snake mode the direction alternates per row (odd rows are reversed).
 *
 * @param {Array}   areas          - Array of scan-area objects (each has centerPosition {x,y})
 * @param {boolean} isSnakeScan    - true  → snake (bidirectional), false → raster (monodirectional)
 * @param {number}  effectiveStepY - Y step size used for row bucketing
 * @returns {Array} Re-ordered copy of the areas array
 */
function applyAreaLevelScanPattern(areas, isSnakeScan, effectiveStepY) {
  if (!areas || areas.length <= 1) return areas;

  // Determine bucket size for grouping areas into rows.
  // Use effectiveStepY if positive, otherwise fall back to 10 % of the Y span.
  const ys = areas.map(a => a.centerPosition.y);
  const ySpan = Math.max(...ys) - Math.min(...ys);
  const bucket = effectiveStepY > 0 ? effectiveStepY : (ySpan > 0 ? ySpan * 0.1 : 1);

  // Assign each area a row key (quantised Y)
  const rowKey = (a) => Math.round(a.centerPosition.y / bucket);

  // Group into rows
  const rowMap = {};
  areas.forEach(area => {
    const key = rowKey(area);
    if (!rowMap[key]) rowMap[key] = [];
    rowMap[key].push(area);
  });

  // Sort rows top→bottom, each row left→right by center X
  const sortedRowKeys = Object.keys(rowMap).map(Number).sort((a, b) => a - b);
  sortedRowKeys.forEach(key => {
    rowMap[key].sort((a, b) => a.centerPosition.x - b.centerPosition.x);
  });

  // Flatten – reverse odd rows for snake pattern
  const ordered = [];
  sortedRowKeys.forEach((key, rowIndex) => {
    const row = rowMap[key];
    if (isSnakeScan && rowIndex % 2 === 1) {
      ordered.push(...[...row].reverse());
    } else {
      ordered.push(...row);
    }
  });

  return ordered;
}

/**
 * Convert scan configuration to backend API format
 * Maintains compatibility with existing Experiment model
 */
export function convertToBackendFormat(scanConfig, experiment) {
  // Create a new experiment object with pre-calculated coordinates
  const backendExperiment = {
    ...experiment,
    // Add the pre-calculated scan areas
    scanAreas: scanConfig.scanAreas,
    // Signal to backend that coordinates are pre-calculated
    coordinatesPreCalculated: true,
    // Remove the old pointList or mark it as processed
    _originalPointList: experiment.pointList,
    pointList: convertScanAreasToPointList(scanConfig.scanAreas)
  };

  return backendExperiment;
}

/**
 * Convert scan areas back to point list format for backward compatibility
 */
function convertScanAreasToPointList(scanAreas) {
  return scanAreas.map(area => ({
    id: area.areaId,
    name: area.areaName,
    x: area.centerPosition.x,
    y: area.centerPosition.y,
    z: area.centerPosition.z ?? 0,
    iX: 0,
    iY: 0,
    wellId: area.wellId,
    areaType: area.areaType,
    // Positions are already ordered - backend should NOT resort
    neighborPointList: area.positions.map(pos => ({
      x: pos.x,
      y: pos.y,
      z: pos.z ?? 0,
      iX: pos.iX,
      iY: pos.iY
    }))
  }));
}

export default {
  calculateScanCoordinates,
  convertToBackendFormat
};
