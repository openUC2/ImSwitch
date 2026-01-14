//##################################################################################
/**
 * Use:
 * import * as WellSelectorUtils from './WellSelectorUtils.js';
 * WellSelectorUtils.isCircleInsideRect()
 */
//##################################################################################
export function calculateRasterRect(
  center,
  rasterWidth,
  rasterHeight,
  rectPlusX,
  rectMinusX,
  rectPlusY,
  rectMinusY
) {
  if (rasterWidth <= 0 || rasterHeight <= 0) {
    throw new Error("Raster width and height must be greater than 0.");
  }

  const rasters = [];

  const minX = center.x - rectMinusX;
  const maxX = center.x + rectPlusX;
  const minY = center.y - rectMinusY;
  const maxY = center.y + rectPlusY;

  // Raster sind an ihrem Zentrum ausgerichtet:
  const centerX = center.x;
  const centerY = center.y;

  const visited = new Set();

  const toKey = (x, y) => `${x},${y}`;

  function rectsOverlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) {
    return !(bx2 <= ax1 || bx1 >= ax2 || by2 <= ay1 || by1 >= ay2);
  }

  const queue = [{ x: centerX, y: centerY }];
  const directions = [
    { dx: rasterWidth, dy: 0 },
    { dx: -rasterWidth, dy: 0 },
    { dx: 0, dy: rasterHeight },
    { dx: 0, dy: -rasterHeight },
  ];

  while (queue.length > 0) {
    const { x, y } = queue.shift();
    const key = toKey(x, y);
    if (visited.has(key)) continue;
    visited.add(key);

    // Raster-Ecken berechnen
    const rasterLeft = x - rasterWidth / 2;
    const rasterRight = x + rasterWidth / 2;
    const rasterTop = y - rasterHeight / 2;
    const rasterBottom = y + rasterHeight / 2;

    if (
      rectsOverlap(
        minX,
        minY,
        maxX,
        maxY,
        rasterLeft,
        rasterTop,
        rasterRight,
        rasterBottom
      )
    ) {
        const ix = Math.round((x - centerX) / rasterWidth);
        const iy = Math.round((y - centerY) / rasterHeight);
      rasters.push({ x, y, iX: ix, iY: iy });

      for (const { dx, dy } of directions) {
        const nx = x + dx;
        const ny = y + dy;
        const nkey = toKey(nx, ny);
        if (!visited.has(nkey)) {
          queue.push({ x: nx, y: ny });
        }
      }
    }
  }

  return rasters;
}

//##################################################################################
export function calculateRasterOval(
  center,
  rasterWidth,
  rasterHeight,
  radiusX,
  radiusY
) {
  const rasters = [];

  // Horizontal und vertikal Halbachsen (für das Oval)
  const a = radiusX; // horizontale Halbachse (nach rechts)
  const b = radiusY; // vertikale Halbachse (nach oben)

  // Visited Set, um Duplikate zu vermeiden
  const visited = new Set();
  const toKey = (x, y) => `${x},${y}`;

  // Funktion, um zu prüfen, ob ein Punkt (x, y) innerhalb des Ovals liegt
  const isInsideOval = (x, y) => {
    const dx = x - center.x;
    const dy = y - center.y;
    return (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1;
  };

  // Queue für die Expansion, Startpunkt ist das Zentrum
  const queue = [{ x: center.x, y: center.y }];
  const directions = [
    { dx: rasterWidth, dy: 0 }, // Rechts expandieren
    { dx: -rasterWidth, dy: 0 }, // Links expandieren
    { dx: 0, dy: rasterHeight }, // Oben expandieren
    { dx: 0, dy: -rasterHeight }, // Unten expandieren
  ];

  while (queue.length > 0) {
    const { x, y } = queue.shift();
    const key = toKey(x, y);
    if (visited.has(key)) continue;
    visited.add(key);

    // Wenn das Raster innerhalb des Ovals liegt, füge es hinzu
    if (isInsideOval(x, y)) {
        const ix = Math.round((x - center.x) / rasterWidth);
        const iy = Math.round((y - center.y) / rasterHeight);
      rasters.push({ x: x, y: y, iX: ix, iY: iy });

      // Expansion in alle vier Richtungen
      for (const { dx, dy } of directions) {
        const nextX = x + dx;
        const nextY = y + dy;
        const nextKey = toKey(nextX, nextY);
        if (!visited.has(nextKey)) {
          queue.push({ x: nextX, y: nextY });
        }
      }
    }
  }

  return rasters;
}

//##################################################################################
export function calculateNeighborPointsSquare(
  squareX,
  squareY,
  squareWidth,
  squareHeight,
  numNeighborsX,
  numNeighborsY
) {
  const points = [];

  for (let i = -numNeighborsX; i <= numNeighborsX; i++) {
    for (let j = -numNeighborsY; j <= numNeighborsY; j++) {
      // Skip the original rectangle at the center
      if (i === 0 && j === 0) continue;

      // Calculate the position for the neighboring rectangle
      const neighborX = squareX + i * squareWidth;
      const neighborY = squareY + j * squareHeight;

      points.push({ x: neighborX, y: neighborY, iX: i, iY: j });
    }
  }

  return points;
}

//##################################################################################
export function calculateNeighborPointsCircle(
  squareX,
  squareY,
  squareWidth,
  squareHeight,
  numNeighbors
) {
  const points = [];

  // Determine the maximum dimension and circle radius
  const maxDim = Math.max(squareWidth, squareHeight);
  const circleRadius = numNeighbors * maxDim;

  // Calculate neighbor limits based on aspect ratio
  const adjustedNeighborsX = Math.ceil(circleRadius / squareWidth);
  const adjustedNeighborsY = Math.ceil(circleRadius / squareHeight);

  // Helper function to check if a point lies within the circle
  function isInCircle(i, j) {
    const centerX = i * squareWidth;
    const centerY = j * squareHeight;
    const distance = Math.sqrt(centerX * centerX + centerY * centerY);
    return distance <= circleRadius;
  }

  // Loop to compute valid neighbor points
  for (let i = -adjustedNeighborsX; i <= adjustedNeighborsX; i++) {
    for (let j = -adjustedNeighborsY; j <= adjustedNeighborsY; j++) {
      // Skip the original rectangle at the center
      if (i === 0 && j === 0) continue;

      // Ensure the point lies within the circle
      if (!isInCircle(i, j)) continue;

      // Calculate the position for the neighboring point
      const neighborX = squareX + i * squareWidth;
      const neighborY = squareY + j * squareHeight;
      points.push({ x: neighborX, y: neighborY, iX: i, iY: j });
    }
  }

  return points;
}

//##################################################################################
export function generateCenterPointsInRect(
  rectStartPoint,
  rectEndPoint,
  wellWidth,
  wellHeight
) {
  // Ensure starting and ending positions are correctly ordered
  const startX = Math.min(rectStartPoint.x, rectEndPoint.x);
  const endX = Math.max(rectStartPoint.x, rectEndPoint.x);
  const startY = Math.min(rectStartPoint.y, rectEndPoint.y);
  const endY = Math.max(rectStartPoint.y, rectEndPoint.y);

  const points = [];

  // Generate points inside the bounding box
  for (let x = startX; x <= endX - wellWidth; x += wellWidth) {
    for (let y = startY; y <= endY - wellHeight; y += wellHeight) {
      points.push({ x, y });
    }
  }

  return points;
}

//##################################################################################
export function isRectInsideRect(x1, y1, width, height, rx1, ry1, rx2, ry2) {
  const left = Math.min(rx1, rx2);
  const right = Math.max(rx1, rx2);
  const top = Math.min(ry1, ry2);
  const bottom = Math.max(ry1, ry2);

  // Calculate the four corners based on the center (x1, y1)
  const corners = [
    { x: x1 - width / 2, y: y1 - height / 2 }, // Top-left
    { x: x1 + width / 2, y: y1 - height / 2 }, // Top-right
    { x: x1 - width / 2, y: y1 + height / 2 }, // Bottom-left
    { x: x1 + width / 2, y: y1 + height / 2 }, // Bottom-right
  ];

  // Check if any of the corners are inside the second rectangle
  return corners.some(
    (corner) =>
      corner.x >= left &&
      corner.x <= right &&
      corner.y >= top &&
      corner.y <= bottom
  );
}

//##################################################################################
export function isCircleInsideRect(cx, cy, r, x1, y1, x2, y2) {
  //console.log("isCircleInsideRect2", cx, cy, r, x1, y1, x2, y2);
  // Ensure x1, y1 is the top-left and x2, y2 is the bottom-right
  const rectLeft = Math.min(x1, x2);
  const rectRight = Math.max(x1, x2);
  const rectTop = Math.min(y1, y2);
  const rectBottom = Math.max(y1, y2);

  // Find the closest point on the rectangle to the circle's center
  const closestX = Math.max(rectLeft, Math.min(cx, rectRight));
  const closestY = Math.max(rectTop, Math.min(cy, rectBottom));

  // Calculate the distance between the circle's center and this closest point
  const dx = cx - closestX;
  const dy = cy - closestY;

  // If the distance from the circle's center to the closest point is less than or equal to the circle's radius, it's inside or overlapping
  const distanceSquared = dx * dx + dy * dy;

  return distanceSquared <= r * r; // If true, part of the circle is inside the rectangle
}

//##################################################################################
export function isPointInsideSquare(point, squareCenter, squareSize) {
  const squareX = squareCenter.x - squareSize / 2;
  const squareY = squareCenter.y - squareSize / 2;
  return (
    point.x >= squareX &&
    point.x <= squareX + squareSize &&
    point.y >= squareY &&
    point.y <= squareY + squareSize
  );
}

//##################################################################################
export function isPointInsideRect(
  point,
  squareCenter,
  squareWidth,
  squareHeight
) {
  const squareX = squareCenter.x - squareWidth / 2;
  const squareY = squareCenter.y - squareHeight / 2;
  return (
    point.x >= squareX &&
    point.x <= squareX + squareWidth &&
    point.y >= squareY &&
    point.y <= squareY + squareHeight
  );
}

//##################################################################################
export function isSquareInsideRect(x, y, rasterSize, x1, y1, x2, y2) {
  // Check if the square's top-left corner is inside the global rect
  // AND the square's bottom-right corner is inside the global rect
  return x >= x1 && y >= y1 && x + rasterSize <= x2 && y + rasterSize <= y2;
}

//##################################################################################
export function isSquareInsideRect2(x, y, rasterSize, x1, y1, x2, y2) {
  // Normalize the rectangle to ensure x1, y1 is the top-left and x2, y2 is the bottom-right
  const rectX1 = Math.min(x1, x2);
  const rectY1 = Math.min(y1, y2);
  const rectX2 = Math.max(x1, x2);
  const rectY2 = Math.max(y1, y2);

  // Check if the square is overlapping or inside the rectangle
  return !(
    x + rasterSize <= rectX1 || // Square is completely to the left of the rect
    y + rasterSize <= rectY1 || // Square is completely above the rect
    x >= rectX2 || // Square is completely to the right of the rect
    y >= rectY2
  ); // Square is completely below the rect
}

//##################################################################################
export const calcCenterPoint = (pointA, pointB) => {
  const centerX = (pointA.x + pointB.x) / 2;
  const centerY = (pointA.y + pointB.y) / 2;
  return { x: centerX, y: centerY };
};

//##################################################################################
export function isLineIntersectingSquare(
  x1,
  y1,
  x2,
  y2,
  squareX,
  squareY,
  size
) {
  // Define the square's corners
  const topLeft = { x: squareX, y: squareY };
  const topRight = { x: squareX + size, y: squareY };
  const bottomLeft = { x: squareX, y: squareY + size };
  const bottomRight = { x: squareX + size, y: squareY + size };

  // Helper function to check if a point is inside the square
  function isPointInsideSquare(px, py) {
    return (
      px >= squareX &&
      px <= squareX + size &&
      py >= squareY &&
      py <= squareY + size
    );
  }

  // Helper function to check if two line segments intersect
  function doLinesIntersect(p1, q1, p2, q2) {
    function orientation(p, q, r) {
      const val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
      if (val === 0) return 0; // Collinear
      return val > 0 ? 1 : 2; // Clockwise or Counterclockwise
    }

    function onSegment(p, q, r) {
      return (
        q.x <= Math.max(p.x, r.x) &&
        q.x >= Math.min(p.x, r.x) &&
        q.y <= Math.max(p.y, r.y) &&
        q.y >= Math.min(p.y, r.y)
      );
    }

    const o1 = orientation(p1, q1, p2);
    const o2 = orientation(p1, q1, q2);
    const o3 = orientation(p2, q2, p1);
    const o4 = orientation(p2, q2, q1);

    // General case
    if (o1 !== o2 && o3 !== o4) return true;

    // Special cases
    if (o1 === 0 && onSegment(p1, p2, q1)) return true;
    if (o2 === 0 && onSegment(p1, q2, q1)) return true;
    if (o3 === 0 && onSegment(p2, p1, q2)) return true;
    if (o4 === 0 && onSegment(p2, q1, q2)) return true;

    return false;
  }

  // Check if either endpoint of the line is inside the square
  if (isPointInsideSquare(x1, y1) || isPointInsideSquare(x2, y2)) {
    return true;
  }

  // Check if the line intersects any of the square's edges
  if (
    doLinesIntersect({ x: x1, y: y1 }, { x: x2, y: y2 }, topLeft, topRight) ||
    doLinesIntersect(
      { x: x1, y: y1 },
      { x: x2, y: y2 },
      topRight,
      bottomRight
    ) ||
    doLinesIntersect(
      { x: x1, y: y1 },
      { x: x2, y: y2 },
      bottomRight,
      bottomLeft
    ) ||
    doLinesIntersect({ x: x1, y: y1 }, { x: x2, y: y2 }, bottomLeft, topLeft)
  ) {
    return true;
  }

  return false; // No intersection
}

//##################################################################################
export const wellLayoutDefault = {
  name: "Default",
  unit: "um",
  width: 120000, // Standard width for 32-well plate
  height: 80000, // Standard height for 32-well plate
  wells: [],
};

export const wellLayoutDevelopment = {
  name: "Heidstar 4x Histosample",
  unit: "um",
  width: 170000,
  height: 130000,
  wells: [
    {
      x: 40000,
      y: 67250,
      shape: "rectangle",
      width: 27000,
      height: 74000,
      name: "Slide1",
    },
    {
      x: 40000 + 30000,
      y: 67250,
      shape: "rectangle",
      width: 27000,
      height: 74000,
      name: "Slide2",
    },
    {
      x: 40000 + 2 * 30000,
      y: 67250,
      shape: "rectangle",
      width: 27000,
      height: 74000,
      name: "Slide3",
    },
    {
      x: 40000 + 3 * 30000,
      y: 67250,
      shape: "rectangle",
      width: 27000,
      height: 74000,
      name: "Slide4",
    },    
    {
      x: 40750,
      y: 68000,
      shape: "circle",
      radius: 1000,
      name: "calibration",
    },
  ],
};

export const ropodLayout = {
  name: "Ropod Layout",
  unit: "um",
  width: 170000,
  height: 130000,
  wells: [
    {
      x: 85000,
      y: 46000,
      shape: "rectangle",
      width: 66600,
      height: 24100,
      name: "Slide1",
    },
    {
      x: 85000,
      y: 84000,
      shape: "rectangle",
      width: 66600,
      height: 24100,
      name: "Slide2",
    },
    {
      x: 85000,
      y: 65000,
      shape: "circle",
      radius: 1000,
      name: "calibration",
    },
  ],
};

export const wellLayout32 = {
  name: "Wellplate 32",
  unit: "um",
  width: 128000,
  height: 86000,
  wells: [
    { x: 15000, y: 15000, shape: "circle", radius: 4500 },
    { x: 45000, y: 15000, shape: "circle", radius: 4500 },
    { x: 75000, y: 15000, shape: "circle", radius: 4500 },
    { x: 105000, y: 15000, shape: "circle", radius: 4500 },

    { x: 15000, y: 30000, shape: "circle", radius: 4500 },
    { x: 45000, y: 30000, shape: "circle", radius: 4500 },
    { x: 75000, y: 30000, shape: "circle", radius: 4500 },
    { x: 105000, y: 30000, shape: "circle", radius: 4500 },

    { x: 15000, y: 45000, shape: "circle", radius: 4500 },
    { x: 45000, y: 45000, shape: "circle", radius: 4500 },
    { x: 75000, y: 45000, shape: "circle", radius: 4500 },
    { x: 105000, y: 45000, shape: "circle", radius: 4500 },

    { x: 15000, y: 60000, shape: "circle", radius: 4500 },
    { x: 45000, y: 60000, shape: "circle", radius: 4500 },
    { x: 75000, y: 60000, shape: "circle", radius: 4500 },
    { x: 105000, y: 60000, shape: "circle", radius: 4500 },

    { x: 15000, y: 75000, shape: "circle", radius: 4500 },
    { x: 45000, y: 75000, shape: "circle", radius: 4500 },
    { x: 75000, y: 75000, shape: "circle", radius: 4500 },
    { x: 105000, y: 75000, shape: "circle", radius: 4500 },
  ],
};

export const histolayout = {
  name: "histolayout",
  unit: "um",
  width: 128000,
  height: 86000,
  wells: [
    {
      x: 15000,
      y: 0,
      shape: "rectangle",
      width: 24000,
      height: 100000,
      name: "A1",
    },
  ],
};

export const wellLayout96 = {
  name: "Wellplate 96",
  unit: "um",
  width: 128000,
  height: 86000,
  wells: [
    { x: 10667, y: 10667, shape: "circle", radius: 4500 },
    { x: 21334, y: 10667, shape: "circle", radius: 4500 },
    { x: 32001, y: 10667, shape: "circle", radius: 4500 },
    { x: 42668, y: 10667, shape: "circle", radius: 4500 },
    { x: 53335, y: 10667, shape: "circle", radius: 4500 },
    { x: 64002, y: 10667, shape: "circle", radius: 4500 },
    { x: 74669, y: 10667, shape: "circle", radius: 4500 },
    { x: 85336, y: 10667, shape: "circle", radius: 4500 },

    { x: 10667, y: 21334, shape: "circle", radius: 4500 },
    { x: 21334, y: 21334, shape: "circle", radius: 4500 },
    { x: 32001, y: 21334, shape: "circle", radius: 4500 },
    { x: 42668, y: 21334, shape: "circle", radius: 4500 },
    { x: 53335, y: 21334, shape: "circle", radius: 4500 },
    { x: 64002, y: 21334, shape: "circle", radius: 4500 },
    { x: 74669, y: 21334, shape: "circle", radius: 4500 },
    { x: 85336, y: 21334, shape: "circle", radius: 4500 },

    { x: 10667, y: 32001, shape: "circle", radius: 4500 },
    { x: 21334, y: 32001, shape: "circle", radius: 4500 },
    { x: 32001, y: 32001, shape: "circle", radius: 4500 },
    { x: 42668, y: 32001, shape: "circle", radius: 4500 },
    { x: 53335, y: 32001, shape: "circle", radius: 4500 },
    { x: 64002, y: 32001, shape: "circle", radius: 4500 },
    { x: 74669, y: 32001, shape: "circle", radius: 4500 },
    { x: 85336, y: 32001, shape: "circle", radius: 4500 },

    { x: 10667, y: 42668, shape: "circle", radius: 4500 },
    { x: 21334, y: 42668, shape: "circle", radius: 4500 },
    { x: 32001, y: 42668, shape: "circle", radius: 4500 },
    { x: 42668, y: 42668, shape: "circle", radius: 4500 },
    { x: 53335, y: 42668, shape: "circle", radius: 4500 },
    { x: 64002, y: 42668, shape: "circle", radius: 4500 },
    { x: 74669, y: 42668, shape: "circle", radius: 4500 },
    { x: 85336, y: 42668, shape: "circle", radius: 4500 },

    { x: 10667, y: 53335, shape: "circle", radius: 4500 },
    { x: 21334, y: 53335, shape: "circle", radius: 4500 },
    { x: 32001, y: 53335, shape: "circle", radius: 4500 },
    { x: 42668, y: 53335, shape: "circle", radius: 4500 },
    { x: 53335, y: 53335, shape: "circle", radius: 4500 },
    { x: 64002, y: 53335, shape: "circle", radius: 4500 },
    { x: 74669, y: 53335, shape: "circle", radius: 4500 },
    { x: 85336, y: 53335, shape: "circle", radius: 4500 },

    { x: 10667, y: 64002, shape: "circle", radius: 4500 },
    { x: 21334, y: 64002, shape: "circle", radius: 4500 },
    { x: 32001, y: 64002, shape: "circle", radius: 4500 },
    { x: 42668, y: 64002, shape: "circle", radius: 4500 },
    { x: 53335, y: 64002, shape: "circle", radius: 4500 },
    { x: 64002, y: 64002, shape: "circle", radius: 4500 },
    { x: 74669, y: 64002, shape: "circle", radius: 4500 },
    { x: 85336, y: 64002, shape: "circle", radius: 4500 },
  ],
};



// src/utils/generateWellLayout384.js

export function generateWellLayout384({
  cols = 24,
  rows = 16,
  pitch = 4500, // ~9 mm spacing (adjust if needed)
  radius = 2000,       // 3 mm radius
  startX = 12000,
  startY = 9000,
  width = 127000,
  height = 84000,
  offsetX = 0,  // Global offset in X direction (micrometers)
  offsetY = 0,  // Global offset in Y direction (micrometers)
} = {}) {
  const wells = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = startX + c * pitch + offsetX;
      const y = startY + r * pitch + offsetY;
      wells.push({
        x: Number(x.toFixed(3)),
        y: Number(y.toFixed(3)),
        shape: "circle",
        radius,
      });
    }
  }

  return {
    name: "Wellplate 384",
    unit: "um",
    width,
    height,
    wells,
  };
}

// Optional: export a ready-to-use layout
export const wellLayout384 = generateWellLayout384();

//##################################################################################
/**
 * Apply global offset to all wells in a layout
 * @param {Object} layout - Well layout object
 * @param {number} offsetX - X offset in micrometers
 * @param {number} offsetY - Y offset in micrometers
 * @returns {Object} New layout with offset applied
 */
export function applyLayoutOffset(layout, offsetX = 0, offsetY = 0) {
  if (!layout || !layout.wells) {
    return layout;
  }

  return {
    ...layout,
    wells: layout.wells.map(well => ({
      ...well,
      x: well.x + offsetX,
      y: well.y + offsetY
    }))
  };
}

//##################################################################################
// New functions for well pattern generation
//##################################################################################

/**
 * Generate center-only positions for selected wells
 * @param {Array} selectedWells - Array of well objects with x, y, shape properties
 * @param {Object} wellLayout - Layout object containing well spacing information
 * @returns {Array} Array of position objects with x, y coordinates
 */
export function generateWellCenterPositions(selectedWells, wellLayout) {
  const positions = [];
  
  selectedWells.forEach((well, index) => {
    positions.push({
      x: well.x,
      y: well.y,
      name: `Well_${index + 1}_Center`,
      wellId: well.id || index,
      iX: 0,
      iY: 0
    });
  });
  
  return positions;
}

/**
 * Generate circular pattern positions within wells
 * @param {Array} selectedWells - Array of well objects
 * @param {number} patternRadius - Radius of the circular pattern in micrometers
 * @param {number} overlap - Overlap factor (0.0 = no overlap, 0.9 = 90% overlap)
 * @param {number} stepSize - Base step size between positions in micrometers
 * @returns {Array} Array of position objects
 */
export function generateWellCirclePattern(selectedWells, patternRadius, overlap, stepSize = 10) {
  const positions = [];
  
  // Calculate actual step size based on overlap
  const actualStepSize = stepSize * (1 - overlap);
  
  selectedWells.forEach((well, wellIndex) => {
    const wellCenterX = well.x;
    const wellCenterY = well.y;
    
    // Generate circular pattern using the existing calculateRasterOval function
    const circlePositions = calculateRasterOval(
      { x: wellCenterX, y: wellCenterY },
      actualStepSize,
      actualStepSize,
      patternRadius,
      patternRadius
    );
    
    // Add each position with well-specific naming
    circlePositions.forEach((pos, posIndex) => {
      positions.push({
        x: pos.x,
        y: pos.y,
        name: `Well_${wellIndex + 1}_Pos_${posIndex + 1}`,
        wellId: well.id || wellIndex,
        iX: pos.iX,
        iY: pos.iY,
        localIndex: posIndex
      });
    });
  });
  
  return positions;
}

/**
 * Generate rectangular pattern positions within wells
 * @param {Array} selectedWells - Array of well objects
 * @param {number} patternWidth - Width of the rectangular pattern in micrometers
 * @param {number} patternHeight - Height of the rectangular pattern in micrometers
 * @param {number} overlap - Overlap factor (0.0 = no overlap, 0.9 = 90% overlap)
 * @param {number} stepSize - Base step size between positions in micrometers
 * @returns {Array} Array of position objects
 */
export function generateWellRectanglePattern(selectedWells, patternWidth, patternHeight, overlap, stepSize = 10) {
  const positions = [];
  
  // Calculate actual step size based on overlap
  const actualStepSize = stepSize * (1 - overlap);
  
  selectedWells.forEach((well, wellIndex) => {
    const wellCenterX = well.x;
    const wellCenterY = well.y;
    
    // Generate rectangular pattern using the existing calculateRasterRect function
    const rectPositions = calculateRasterRect(
      { x: wellCenterX, y: wellCenterY },
      actualStepSize,
      actualStepSize,
      patternWidth / 2,  // rectPlusX
      patternWidth / 2,  // rectMinusX
      patternHeight / 2, // rectPlusY
      patternHeight / 2  // rectMinusY
    );
    
    // Add each position with well-specific naming
    rectPositions.forEach((pos, posIndex) => {
      positions.push({
        x: pos.x,
        y: pos.y,
        name: `Well_${wellIndex + 1}_Pos_${posIndex + 1}`,
        wellId: well.id || wellIndex,
        iX: pos.iX,
        iY: pos.iY,
        localIndex: posIndex
      });
    });
  });
  
  return positions;
}

/**
 * Generate positions based on point list with well patterns
 * @param {Array} pointList - Point list from Redux state with shape configurations
 * @param {Object} wellLayout - Current well layout
 * @returns {Array} Array of final positions for experiment
 */
export function generatePositionsFromPointList(pointList, wellLayout) {
  const allPositions = [];
  
  pointList.forEach((point, pointIndex) => {
    if (!point.shape || point.shape === "off") {
      // No pattern, just add the original point
      allPositions.push({
        x: point.x,
        y: point.y,
        name: point.name || `Point_${pointIndex + 1}`,
        iX: 0,
        iY: 0
      });
      return;
    }
    
    if (point.wellMode === "center_only") {
      // Find wells that intersect with this point and return their centers
      const intersectingWells = findWellsAtPosition(point, wellLayout);
      const centerPositions = generateWellCenterPositions(intersectingWells, wellLayout);
      allPositions.push(...centerPositions);
      
    } else if (point.wellMode === "pattern") {
      // Generate pattern within wells
      const intersectingWells = findWellsAtPosition(point, wellLayout);
      
      if (point.patternType === "circle") {
        const patternPositions = generateWellCirclePattern(
          intersectingWells,
          point.patternRadius || 50,
          point.patternOverlap || 0.1
        );
        allPositions.push(...patternPositions);
        
      } else if (point.patternType === "rectangle") {
        const patternPositions = generateWellRectanglePattern(
          intersectingWells,
          point.patternWidth || 100,
          point.patternHeight || 100,
          point.patternOverlap || 0.1
        );
        allPositions.push(...patternPositions);
      }
      
    } else {
      // Traditional rectangle or circle patterns (existing functionality)
      if (point.shape === "rectangle") {
        const rectPositions = calculateRasterRect(
          { x: point.x, y: point.y },
          10, // rasterWidth - this should come from experiment settings
          10, // rasterHeight - this should come from experiment settings
          point.rectPlusX || 0,
          point.rectMinusX || 0,
          point.rectPlusY || 0,
          point.rectMinusY || 0
        );
        allPositions.push(...rectPositions.map((pos, i) => ({
          ...pos,
          name: `${point.name || `Point_${pointIndex + 1}`}_${i + 1}`
        })));
        
      } else if (point.shape === "circle") {
        const circlePositions = calculateRasterOval(
          { x: point.x, y: point.y },
          10, // rasterWidth - this should come from experiment settings
          10, // rasterHeight - this should come from experiment settings
          point.circleRadiusX || 0,
          point.circleRadiusY || 0
        );
        allPositions.push(...circlePositions.map((pos, i) => ({
          ...pos,
          name: `${point.name || `Point_${pointIndex + 1}`}_${i + 1}`
        })));
      }
    }
  });
  
  return allPositions;
}

/**
 * Find wells that intersect with a given point/position
 * @param {Object} point - Point with x, y coordinates
 * @param {Object} wellLayout - Well layout with wells array
 * @returns {Array} Array of wells that intersect with the point
 */
export function findWellsAtPosition(point, wellLayout) {
  if (!wellLayout || !wellLayout.wells) {
    return [];
  }
  
  const intersectingWells = [];
  
  wellLayout.wells.forEach((well, index) => {
    let isIntersecting = false;
    
    if (well.shape === "circle") {
      // Check if point is within circle
      const dx = point.x - well.x;
      const dy = point.y - well.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      isIntersecting = distance <= (well.radius || 0);
      
    } else if (well.shape === "rectangle") {
      // Check if point is within rectangle
      const halfWidth = (well.width || 0) / 2;
      const halfHeight = (well.height || 0) / 2;
      isIntersecting = isPointInsideRect(
        point,
        { x: well.x, y: well.y },
        well.width || 0,
        well.height || 0
      );
    }
    
    if (isIntersecting) {
      intersectingWells.push({
        ...well,
        id: well.id || index
      });
    }
  });
  
  return intersectingWells;
}
