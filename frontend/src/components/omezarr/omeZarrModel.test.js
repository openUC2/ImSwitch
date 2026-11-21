import {
  autoWindow,
  compositeRGBA,
  dataUrl,
  describeImage,
  groupByFolder,
  pickLevel,
  readNgff,
  tilesInView,
  zoomAt,
} from "./omeZarrModel";

// .zattrs as ImSwitch's storage.py writes it (NGFF 0.4, omero window = full uint16 range)
const attrs = {
  multiscales: [
    {
      axes: [
        { name: "t", type: "time" },
        { name: "c", type: "channel" },
        { name: "z", type: "space", unit: "micrometer" },
        { name: "y", type: "space", unit: "micrometer" },
        { name: "x", type: "space", unit: "micrometer" },
      ],
      datasets: [0, 1, 2].map((i) => ({
        path: String(i),
        coordinateTransformations: [{ type: "scale", scale: [1, 1, 2, 0.5 * 2 ** i, 0.5 * 2 ** i] }],
      })),
    },
  ],
  omero: {
    channels: [
      { label: "LED", color: "FFFFFF", window: { start: 0, end: 65535 } },
      { label: "LASER", color: "FFFFFF", window: { start: 0, end: 65535 } },
    ],
  },
};
const shapes = [
  [2, 2, 3, 900, 1200],
  [2, 2, 3, 450, 600],
  [2, 2, 3, 225, 300],
];

test("describes an ImSwitch image: sizes, pyramid factors, pixel size, channels", () => {
  const img = describeImage(readNgff(attrs), shapes, "uint16");
  expect(img.sizes).toEqual({ t: 2, c: 2, z: 3, y: 900, x: 1200 });
  expect(img.levels.map((l) => l.factor)).toEqual([1, 2, 4]);
  expect(img.pixelSize).toBe(0.5);
  expect(img.unit).toBe("micrometer");
  // all-white omero colours on two channels -> distinct colours; full-range window -> auto
  expect(img.channels[0].rgb).not.toEqual(img.channels[1].rgb);
  expect(img.channels.map((c) => [c.label, c.auto])).toEqual([["LED", true], ["LASER", true]]);
});

test("a stored contrast window that is not the full range is kept", () => {
  const ms = { ...attrs.multiscales[0], datasets: attrs.multiscales[0].datasets.slice(0, 1) };
  const a = { multiscales: [ms], omero: { channels: [{ label: "x", color: "00FF00", window: { start: 10, end: 300 } }] } };
  const img = describeImage(readNgff(a), [[1, 1, 1, 10, 10]], "uint16");
  expect(img.channels[0]).toMatchObject({ start: 10, end: 300, auto: false, rgb: [0, 255, 0] });
});

test("recognises plates, wells and NGFF 0.5 nesting", () => {
  expect(readNgff({ plate: { wells: [{ path: "A/1" }, { path: "B/2" }] } })).toEqual({
    kind: "plate",
    wells: ["A/1", "B/2"],
  });
  expect(readNgff({ well: { images: [{ path: "0" }] } })).toEqual({ kind: "well", images: ["0"] });
  expect(readNgff({ ome: attrs }).kind).toBe("image");
  expect(readNgff({}).kind).toBe("unknown");
});

test("picks the coarsest level that is not magnified on screen", () => {
  const levels = [{ factor: 1 }, { factor: 2 }, { factor: 4 }, { factor: 8 }];
  expect(pickLevel(levels, 2)).toBe(0);
  expect(pickLevel(levels, 1)).toBe(0);
  expect(pickLevel(levels, 0.5)).toBe(1);
  expect(pickLevel(levels, 0.3)).toBe(1);
  expect(pickLevel(levels, 0.01)).toBe(3);
});

test("only the chunks under the canvas are requested, edge chunks clipped", () => {
  const level = { shape: [1, 1, 1, 900, 1200], factor: 1 };
  // whole image visible
  expect(tilesInView(level, [300, 400], { x: 0, y: 0, zoom: 0.5 }, 600, 450)).toHaveLength(9);
  // top-left 300x400 window at zoom 1 touches exactly one chunk
  expect(tilesInView(level, [300, 400], { x: 0, y: 0, zoom: 1 }, 399, 299)).toEqual([
    { ty: 0, tx: 0, y0: 0, x0: 0, h: 300, w: 400 },
  ]);
  // level 1 of the same image, last chunk is narrower than the chunk size
  const l1 = { shape: [1, 1, 1, 450, 600], factor: 2 };
  const tiles = tilesInView(l1, [256, 256], { x: 0, y: 0, zoom: 0.5 }, 600, 450);
  expect(tiles.at(-1)).toEqual({ ty: 1, tx: 2, y0: 256, x0: 512, h: 194, w: 88 });
});

test("zooming keeps the point under the cursor fixed", () => {
  const v = zoomAt({ x: 100, y: 50, zoom: 1 }, 200, 100, 2);
  expect(v.zoom).toBe(2);
  expect(v.x + 200 / v.zoom).toBeCloseTo(300);
  expect(v.y + 100 / v.zoom).toBeCloseTo(150);
});

test("auto contrast ignores a few hot pixels", () => {
  const data = new Uint16Array(10000).map((_, i) => 10 + (i % 30));
  data[5] = 60000;
  const w = autoWindow(data);
  expect(w.start).toBe(10);
  expect(w.end).toBe(39);
  expect(w.max).toBe(60000);
});

test("channels are windowed and added in their colours", () => {
  const out = new Uint8ClampedArray(2 * 4);
  compositeRGBA(
    [new Uint16Array([10, 20]), new Uint16Array([0, 100])],
    [
      { visible: true, rgb: [0, 255, 0], start: 10, end: 20 },
      { visible: true, rgb: [255, 0, 255], start: 0, end: 100 },
    ],
    out,
  );
  expect([...out]).toEqual([0, 0, 0, 255, 255, 255, 255, 255]);
});

test("datasets are grouped per run folder, newest first, zarr only", () => {
  const groups = groupByFolder([
    { name: "b.ome.zarr", folder: "/A/old", format: "zarr", modifiedTime: 1 },
    { name: "x_10.ome.zarr", folder: "/A/new", format: "zarr", modifiedTime: 5 },
    { name: "x_2.ome.zarr", folder: "/A/new", format: "zarr", modifiedTime: 4 },
    { name: "x.ome.tif", folder: "/A/new", format: "tiff", modifiedTime: 9 },
  ]);
  expect(groups.map((g) => g.folder)).toEqual(["/A/new", "/A/old"]);
  expect(groups[0].items.map((i) => i.name)).toEqual(["x_2.ome.zarr", "x_10.ome.zarr"]);
});

test("data URLs are built below /imswitch/data exactly once", () => {
  expect(dataUrl("http://pi", 80, "/Aufgaben/run/a.ome.zarr/")).toBe("http://pi:80/imswitch/data/Aufgaben/run/a.ome.zarr");
  expect(dataUrl("http://pi", 80, "/imswitch/data/x.zarr")).toBe("http://pi:80/imswitch/data/x.zarr");
  expect(dataUrl("http://pi", 80, "http://other/x.zarr")).toBe("http://other/x.zarr");
});
