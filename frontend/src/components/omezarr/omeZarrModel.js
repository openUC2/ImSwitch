// Pure helpers for the OME-Zarr viewer: NGFF metadata, pyramid level and tile choice,
// contrast and channel compositing. No I/O and no zarrita here, so Jest can test it.

const PALETTE = [
  [0, 255, 0],
  [255, 0, 255],
  [0, 255, 255],
  [255, 255, 0],
  [255, 0, 0],
  [0, 0, 255],
];

/** A TIFF file: the server reads it as a zarr store (/FileManager/tiffzarr). */
export const isTiff = (name) => /\.tiff?$/i.test(name || "");

/**
 * Absolute URL of the zarr store for a path below the server's data folder: the folder
 * itself (/imswitch/data), or for a TIFF the server's zarr view of the file.
 */
export function dataUrl(ip, port, path) {
  if (/^https?:\/\//.test(path)) return path.replace(/\/+$/, "");
  const rel = path.replace(/^\/imswitch\/data/, "").replace(/\/+$/, "");
  const root = isTiff(rel) ? "/imswitch/api/FileManager/tiffzarr" : "/imswitch/data";
  return `${ip}:${port}${root}${rel.startsWith("/") ? rel : `/${rel}`}`;
}

/** What an NGFF group is: an image, a plate, a well, or something else. */
export function readNgff(attrs) {
  const a = attrs?.ome || attrs || {}; // NGFF 0.5 nests everything under "ome"
  if (a.multiscales?.length) return { kind: "image", multiscales: a.multiscales[0], omero: a.omero };
  if (a.plate) return { kind: "plate", wells: (a.plate.wells || []).map((w) => w.path) };
  if (a.well) return { kind: "well", images: (a.well.images || []).map((i) => i.path) };
  return { kind: "unknown" };
}

function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function dtypeMax(dtype) {
  const m = /^u?int(8|16|32)$/.exec(dtype);
  if (!m) return null;
  return dtype.startsWith("u") ? 2 ** Number(m[1]) - 1 : 2 ** (Number(m[1]) - 1) - 1;
}

/**
 * Describe an NGFF image from its multiscales/omero metadata and the shape of every level.
 * Missing axes (2D, 3D images) get size 1; levels carry their downsampling factor.
 */
export function describeImage(ngff, levelShapes, dtype) {
  const ms = ngff.multiscales;
  const shape0 = levelShapes[0];
  const axes = (ms.axes || "tczyx".slice(5 - shape0.length).split("")).map((a) =>
    (typeof a === "string" ? a : a.name).toLowerCase(),
  );
  const size = (name) => (axes.includes(name) ? shape0[axes.indexOf(name)] : 1);
  const xi = axes.indexOf("x");
  const levels = ms.datasets.map((d, i) => ({
    path: d.path,
    shape: levelShapes[i],
    factor: Math.max(1, Math.round(shape0[xi] / levelShapes[i][xi])),
  }));
  const scale = (ms.datasets[0].coordinateTransformations || []).find((t) => t.type === "scale");
  const xAxis = ms.axes?.[xi];
  const nC = size("c");
  const omeroChannels = ngff.omero?.channels || [];
  const allWhite = omeroChannels.every((ch) => (ch.color || "").toUpperCase() === "FFFFFF");
  const max = dtypeMax(dtype);
  const channels = Array.from({ length: nC }, (_, i) => {
    const o = omeroChannels[i] || {};
    const win = o.window || {};
    const full = win.start == null || win.end == null || (win.start <= 0 && max != null && win.end >= max);
    return {
      label: o.label || `Channel ${i + 1}`,
      rgb: nC > 1 && (allWhite || !hexToRgb(o.color)) ? PALETTE[i % PALETTE.length] : hexToRgb(o.color) || [255, 255, 255],
      visible: o.active !== false && (nC <= 4 || i < 4),
      start: full ? 0 : win.start,
      end: full ? max ?? 1 : win.end,
      auto: full, // the stored window says nothing; replace it from the data once loaded
    };
  });
  return {
    axes,
    sizes: { t: size("t"), c: nC, z: size("z"), y: size("y"), x: size("x") },
    levels,
    pixelSize: scale ? scale.scale[xi] : null,
    unit: (typeof xAxis === "object" && xAxis.unit) || "",
    channels,
  };
}

/** Coarsest level whose pixels are still no larger than a screen pixel at this zoom. */
export function pickLevel(levels, zoom) {
  let best = 0;
  levels.forEach((l, i) => {
    if (l.factor * zoom <= 1 && l.factor > levels[best].factor) best = i;
  });
  return best;
}

/**
 * Chunk-aligned tiles of a level that intersect the canvas.
 * view = {x, y, zoom}: level-0 pixel at the canvas top-left and canvas pixels per level-0 pixel.
 */
export function tilesInView(level, chunkYX, view, width, height) {
  const [H, W] = level.shape.slice(-2);
  const [ch, cw] = chunkYX;
  const f = level.factor;
  const range = (start, extent, chunk, total) => {
    const a = Math.max(0, Math.floor(start / f / chunk));
    const b = Math.min(Math.ceil(total / chunk) - 1, Math.floor((start + extent) / f / chunk));
    return [a, b];
  };
  const [ty0, ty1] = range(view.y, height / view.zoom, ch, H);
  const [tx0, tx1] = range(view.x, width / view.zoom, cw, W);
  const tiles = [];
  for (let ty = ty0; ty <= ty1; ty++) {
    for (let tx = tx0; tx <= tx1; tx++) {
      const y0 = ty * ch;
      const x0 = tx * cw;
      tiles.push({ ty, tx, y0, x0, h: Math.min(ch, H - y0), w: Math.min(cw, W - x0) });
    }
  }
  return tiles;
}

/** View that fits a w x h image into the canvas, centred. */
export function fitView(w, h, width, height) {
  const zoom = Math.min(width / w, height / h) || 1;
  return { zoom, x: w / 2 - width / zoom / 2, y: h / 2 - height / zoom / 2 };
}

/**
 * The crop a deconvolution preview uses: size x size full-resolution pixels centred on the view and
 * kept inside the image, the way the server places it. cx, cy is the centre to send.
 */
export function cropAt(view, width, height, size, imgW, imgH) {
  const w = Math.min(size, imgW);
  const h = Math.min(size, imgH);
  const x0 = Math.round(Math.min(Math.max(view.x + width / view.zoom / 2 - w / 2, 0), imgW - w));
  const y0 = Math.round(Math.min(Math.max(view.y + height / view.zoom / 2 - h / 2, 0), imgH - h));
  return { x0, y0, w, h, cx: x0 + Math.floor(w / 2), cy: y0 + Math.floor(h / 2) };
}

/** Zoom by `factor` keeping the image point under canvas pixel (sx, sy) fixed. */
export function zoomAt(view, sx, sy, factor) {
  const zoom = Math.min(64, Math.max(1e-4, view.zoom * factor));
  return { zoom, x: view.x + sx / view.zoom - sx / zoom, y: view.y + sy / view.zoom - sy / zoom };
}

/** 0.1 / 99.9 percentile window of a plane (sampled), and the plane's maximum. */
export function autoWindow(data) {
  const step = Math.max(1, Math.floor(data.length / 65536));
  const sample = [];
  let max = -Infinity;
  for (let i = 0; i < data.length; i += step) sample.push(data[i]);
  for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.001)] ?? 0;
  let hi = sample[Math.min(sample.length - 1, Math.floor(sample.length * 0.999))] ?? 1;
  if (hi <= lo) hi = lo + 1;
  return { start: lo, end: hi, max: Number.isFinite(max) ? max : 1 };
}

/** Additive blend of the visible channels into an RGBA buffer (Uint8ClampedArray clamps). */
export function compositeRGBA(planes, channels, out) {
  out.fill(0);
  for (let c = 0; c < channels.length; c++) {
    const ch = channels[c];
    const p = planes[c];
    if (!ch.visible || !p) continue;
    const s = 1 / Math.max(1e-9, ch.end - ch.start);
    const [r, g, b] = ch.rgb;
    for (let i = 0, j = 0; i < p.length; i++, j += 4) {
      const v = Math.min(1, Math.max(0, (p[i] - ch.start) * s));
      out[j] += v * r;
      out[j + 1] += v * g;
      out[j + 2] += v * b;
    }
  }
  for (let j = 3; j < out.length; j += 4) out[j] = 255;
  return out;
}

/** Datasets from /FileManager/datasets grouped by run folder, newest folder first. */
export function groupByFolder(datasets) {
  const groups = new Map();
  for (const d of datasets) {
    const g = groups.get(d.folder) || { folder: d.folder, modifiedTime: 0, items: [] };
    g.items.push(d);
    g.modifiedTime = Math.max(g.modifiedTime, d.modifiedTime);
    groups.set(d.folder, g);
  }
  const list = [...groups.values()];
  list.forEach((g) => g.items.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })));
  return list.sort((a, b) => b.modifiedTime - a.modifiedTime);
}
