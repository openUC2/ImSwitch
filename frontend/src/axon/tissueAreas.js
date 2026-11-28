// src/axon/tissueAreas.js
// From StageMapController.detectTissue to the plate map's point list: pure
// helpers behind the WellPlate "Detect tissue" button (unit-tested).
import * as wsUtils from "./WellSelectorUtils.js";

const overlapArea = (a, b) =>
  Math.max(0, Math.min(a.maxX, b.maxX) - Math.max(a.minX, b.minX)) *
  Math.max(0, Math.min(a.maxY, b.maxY) - Math.max(a.minY, b.minY));

// The slide slots (rectangular wells of the layout, centred x/y) that the
// prescan tiles cover for at least a quarter of their area. Empty when the
// layout has none: detectTissue then treats the whole prescan as one slide.
export const slideSlots = (layout, tiles) => {
  const covered = (tiles || []).map((t) => ({
    minX: t.x - t.widthUm / 2,
    maxX: t.x + t.widthUm / 2,
    minY: t.y - t.heightUm / 2,
    maxY: t.y + t.heightUm / 2,
  }));
  return (layout?.wells || [])
    .filter((w) => w.shape === "rectangle" && w.width > 0 && w.height > 0)
    .map((w, i) => ({
      id: w.name || `Slot${i + 1}`,
      name: w.name || `Slot${i + 1}`,
      minX: w.x - w.width / 2,
      maxX: w.x + w.width / 2,
      minY: w.y - w.height / 2,
      maxY: w.y + w.height / 2,
    }))
    .filter((slot) => {
      const area = (slot.maxX - slot.minX) * (slot.maxY - slot.minY);
      const seen = covered.reduce((sum, c) => sum + overlapArea(slot, c), 0);
      return seen >= 0.25 * area;
    });
};

// One freehand point (createPoint payload) per tissue region, plus what the
// detection says about ink, clear glass and problems.
export const tissueAreaPoints = (res, { fovX, fovY, overlap = 0.1, z = 0, stamp = Date.now() }) => {
  const points = [];
  let ink = 0;
  let blank = null;
  const warnings = [];
  (res?.slides || []).forEach((slide) => {
    ink += (slide.ink || []).length;
    if (!blank && slide.blank) blank = { x: slide.blank.x, y: slide.blank.y };
    (slide.warnings || [])
      .filter((w) => !/clear glass/i.test(w))
      .forEach((w) => warnings.push(`${slide.name}: ${w}`));
    (slide.regions || []).forEach((region, k) => {
      const outline = region.scanPolygon || [];
      if (outline.length < 3) return;
      const polygon = outline.map(([x, y]) => ({ x, y }));
      const areaId = `tissue_${slide.id}_${k + 1}_${stamp}`;
      const tiles =
        fovX > 0 && fovY > 0 ? wsUtils.generatePolygonScanPositions(polygon, fovX, fovY, overlap) : [];
      points.push({
        x: (region.bounds.minX + region.bounds.maxX) / 2,
        y: (region.bounds.minY + region.bounds.maxY) / 2,
        z,
        name: `${slide.name}_tissue${k + 1}`,
        shape: "",
        areaType: "free_scan",
        areaId,
        groupId: areaId,
        polygon,
        focusPoints: region.focusPoints || [],
        neighborPointList: tiles.map((p) => ({ x: p.x, y: p.y, z, iX: p.iX, iY: p.iY })),
      });
    });
  });
  return { points, ink, blank, warnings };
};

// Indices of points that the tissue regions replace: earlier tissue areas, and
// selections covering at least half of a searched slot (the slide wells or the
// area drawn over the slides that the prescan was swept from). Anything else
// the user placed stays.
export const replacedByTissue = (pointList, slots) =>
  (pointList || []).reduce((drop, p, index) => {
    if (String(p.areaId || "").startsWith("tissue_")) return drop.concat(index);
    const extent = {
      minX: p.x - (p.rectMinusX || p.circleRadiusX || 0),
      maxX: p.x + (p.rectPlusX || p.circleRadiusX || 0),
      minY: p.y - (p.rectMinusY || p.circleRadiusY || 0),
      maxY: p.y + (p.rectPlusY || p.circleRadiusY || 0),
    };
    const coversASlot = (slots || []).some((s) => {
      const slotArea = (s.maxX - s.minX) * (s.maxY - s.minY);
      const inside = p.x >= s.minX && p.x <= s.maxX && p.y >= s.minY && p.y <= s.maxY;
      return overlapArea(extent, s) >= 0.5 * slotArea || (p.wellMode && inside);
    });
    return coversASlot ? drop.concat(index) : drop;
  }, []);
