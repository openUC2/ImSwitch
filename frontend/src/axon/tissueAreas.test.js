import { replacedByTissue, slideSlots, tissueAreaPoints } from "./tissueAreas";
import { wellLayoutDevelopment, isRectInsideRect, isCircleInsideRect } from "./WellSelectorUtils";
import { plannedPointsForAreas } from "./experiment-designer/plannedFocusGrid";

// Prescan strips over slots 2 and 3 of the Heidstar layout (x 34900..91900).
const strips = Array.from({ length: 10 }, (_, i) => ({
  x: 63400,
  y: 3600 + 7400 * i + 3700,
  widthUm: 57000,
  heightUm: 7400,
}));

describe("tissue areas from detectTissue", () => {
  test("only the slide slots the prescan covers are searched", () => {
    const slots = slideSlots(wellLayoutDevelopment, strips);
    expect(slots.map((s) => s.id)).toEqual(["Slide2", "Slide3"]);
    expect(slots[0]).toMatchObject({ minX: 34900, maxX: 61900, minY: 3600, maxY: 77600 });
    expect(slideSlots(wellLayoutDevelopment, [])).toEqual([]);
  });

  test("every region becomes one freehand area with its outline, tiles and focus points", () => {
    const res = {
      slides: [
        {
          id: "Slide2",
          name: "Slide2",
          ink: [{ polygon: [], areaMm2: 2 }],
          blank: { x: 50000, y: 60000, clearanceUm: 3000 },
          warnings: ["No clear glass on this slide for the flat-field reference."],
          regions: [
            {
              id: "tissue1",
              bounds: { minX: 40000, maxX: 44000, minY: 20000, maxY: 23000 },
              scanPolygon: [[39000, 19000], [45000, 19000], [45000, 24000], [39000, 24000]],
              focusPoints: [{ x: 42000, y: 21500 }],
            },
          ],
        },
      ],
    };
    const { points, ink, blank, warnings } = tissueAreaPoints(res, {
      fovX: 1830, fovY: 1220, overlap: 0.1, z: 7300, stamp: 1,
    });
    expect(ink).toBe(1);
    expect(blank).toEqual({ x: 50000, y: 60000 });
    expect(warnings).toEqual([]);                       // the glass warning is not news here
    expect(points).toHaveLength(1);
    const p = points[0];
    expect(p).toMatchObject({ name: "Slide2_tissue1", areaType: "free_scan", z: 7300, x: 42000, y: 21500 });
    expect(p.groupId).toBe(p.areaId);                   // one scan area, id = groupId
    expect(p.areaId.startsWith("tissue_")).toBe(true);
    expect(p.polygon[0]).toEqual({ x: 39000, y: 19000 });
    expect(p.neighborPointList.length).toBeGreaterThan(4);
    expect(p.focusPoints).toEqual([{ x: 42000, y: 21500 }]);
  });

  test("tissue replaces earlier tissue areas and the slide-wide selection, nothing else", () => {
    const slot = { minX: 34900, maxX: 61900, minY: 3600, maxY: 77600 };
    const list = [
      { areaId: "tissue_Slide2_1_5", x: 1, y: 1 },                                  // earlier tissue
      { x: 48400, y: 40600, rectPlusX: 13000, rectMinusX: 13000, rectPlusY: 36000, rectMinusY: 36000 },
      { x: 48400, y: 40600, wellMode: "center_only" },                             // the slide as a well
      { x: 45000, y: 30000 },                                                        // a point the user set
      { x: 10000, y: 30000, rectPlusX: 13000, rectMinusX: 13000, rectPlusY: 36000, rectMinusY: 36000 },
    ];
    expect(replacedByTissue(list, [slot])).toEqual([0, 1, 2]);
    // one area drawn over two slides (its centre between them) goes as well
    const slot3 = { minX: 64900, maxX: 91900, minY: 3600, maxY: 77600 };
    const overBoth = { x: 63400, y: 40600, rectPlusX: 28500, rectMinusX: 28500, rectPlusY: 37000, rectMinusY: 37000 };
    expect(replacedByTissue([overBoth], [slot, slot3])).toEqual([0]);
  });

  test("the focus plan uses an area's own points instead of a grid over its bounds", () => {
    const areas = [
      { areaId: "tissue_A", areaName: "A", bounds: { minX: 0, maxX: 1000, minY: 0, maxY: 1000 } },
      { areaId: "area_1", areaName: "B", bounds: { minX: 0, maxX: 1000, minY: 0, maxY: 1000 } },
    ];
    const pointList = [{ groupId: "tissue_A", focusPoints: [{ x: 10, y: 20 }, { x: 30, y: 40 }] }];
    const planned = plannedPointsForAreas(areas, pointList, { rows: 2, cols: 2, add_margin: false });
    expect(planned.filter((p) => p.groupId === "tissue_A")).toEqual([
      { x: 10, y: 20, groupId: "tissue_A", groupName: "A" },
      { x: 30, y: 40, groupId: "tissue_A", groupName: "A" },
    ]);
    expect(planned.filter((p) => p.groupId === "area_1")).toHaveLength(4);
  });
});

describe("scan areas keep the group id of a single freehand point", () => {
  test("a tissue area's scan area is called by its groupId", () => {
    // eslint-disable-next-line global-require
    const { calculateScanCoordinates } = require("./CoordinateCalculator");
    const polygon = [{ x: 0, y: 0 }, { x: 4000, y: 0 }, { x: 4000, y: 3000 }, { x: 0, y: 3000 }];
    const experimentState = {
      pointList: [{ id: "p1", x: 2000, y: 1500, z: 0, name: "Slide2_tissue1", shape: "",
                    areaId: "tissue_Slide2_1_9", groupId: "tissue_Slide2_1_9", polygon,
                    neighborPointList: [] }],
      parameterValue: { is_snakescan: true },
      wellLayout: { wells: [] },
    };
    const scan = calculateScanCoordinates(experimentState, { fovX: 1000, fovY: 1000 },
                                          { areaSelectOverlap: 0 });
    expect(scan.scanAreas.map((a) => a.areaId)).toEqual(["tissue_Slide2_1_9"]);
  });
});

describe("the WELL tool selects a slide by a click inside it", () => {
  test("a click (no drag) inside a rectangular well selects it, like a round one", () => {
    // slide 2 of the Heidstar layout: centre 48400 / 40600, 27 x 74 mm
    expect(isRectInsideRect(48400, 40600, 27000, 74000, 45000, 30000, 45000, 30000)).toBe(true);
    expect(isCircleInsideRect(0, 0, 3000, 100, 100, 100, 100)).toBe(true);
    // a drag across part of it, without one of its corners, selects it too
    expect(isRectInsideRect(48400, 40600, 27000, 74000, 40000, 30000, 56000, 50000)).toBe(true);
    // a click beside it does not
    expect(isRectInsideRect(48400, 40600, 27000, 74000, 63000, 30000, 63000, 30000)).toBe(false);
  });
});
