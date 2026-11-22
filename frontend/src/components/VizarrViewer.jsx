/**
 * OME-Zarr viewer for what ImSwitch writes (ExperimentController scans, Aufgaben runs).
 *
 * Reads NGFF 0.4 / 0.5 straight from the server's /imswitch/data mount with zarrita and
 * draws only the chunks under the canvas, from the pyramid level that matches the zoom,
 * on top of the coarsest level. Works offline: no external viewer, no CDN.
 *
 * Left: datasets below the data folder, grouped by run folder, and the fields of a plate.
 * Pure logic (levels, tiles, contrast, compositing) lives in omezarr/omeZarrModel.js.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSelector } from "react-redux";
import * as zarr from "zarrita";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Divider,
  IconButton,
  LinearProgress,
  List,
  ListItemButton,
  ListItemText,
  MenuItem,
  Slider,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { Close, FitScreen, Refresh } from "@mui/icons-material";

import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import apiFileManagerGetDatasets from "../backendapi/apiFileManagerGetDatasets";
import DeconvolutionPanel from "./omezarr/DeconvolutionPanel";
import {
  autoWindow,
  compositeRGBA,
  cropAt,
  dataUrl,
  describeImage,
  fitView,
  groupByFolder,
  pickLevel,
  readNgff,
  tilesInView,
  zoomAt,
} from "./omezarr/omeZarrModel";

const CACHE_TILES = 400; // raw chunk planes kept in memory (LRU)

const relPath = (p) => (p || "").replace(/^\/imswitch\/data/, "").replace(/\/+$/, "");
const parentOf = (p) => p.slice(0, p.lastIndexOf("/")) || "/";
const toHex = (rgb) => `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
const fromHex = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

async function fetchJson(url) {
  const r = await fetch(url);
  return r.ok ? r.json() : null;
}

/** Group attributes of a zarr v2 (.zattrs) or v3 (zarr.json) node. */
async function readAttrs(url) {
  const v2 = await fetchJson(`${url}/.zattrs`);
  if (v2) return { attrs: v2, version: 2 };
  const v3 = await fetchJson(`${url}/zarr.json`);
  return v3 ? { attrs: v3.attributes || {}, version: 3 } : null;
}

/** The images of a dataset: the store itself, or every field of a plate or well. */
async function listImages(url) {
  const meta = await readAttrs(url);
  if (!meta) throw new Error("No image here: not an OME-Zarr store, or a TIFF layout the viewer cannot read.");
  const ngff = readNgff(meta.attrs);
  if (ngff.kind === "image") return [{ label: "", url }];
  if (ngff.kind !== "plate" && ngff.kind !== "well") throw new Error("Not an OME-NGFF image or plate.");
  const wells = ngff.kind === "plate" ? ngff.wells : [""];
  const fields = await Promise.all(
    wells.map(async (w) => {
      const wellUrl = w ? `${url}/${w}` : url;
      const well = w ? readNgff((await readAttrs(wellUrl))?.attrs) : ngff;
      return (well.images || []).map((i) => ({ label: [w, i].filter(Boolean).join(" / "), url: `${wellUrl}/${i}` }));
    }),
  );
  return fields.flat();
}

async function openImage(url) {
  const meta = await readAttrs(url);
  const ngff = readNgff(meta?.attrs);
  if (ngff.kind !== "image") throw new Error("Not an OME-NGFF image.");
  const root = zarr.root(new zarr.FetchStore(url));
  const arrays = await Promise.all(
    ngff.multiscales.datasets.map((d) =>
      meta.version === 2
        ? zarr.open.v2(root.resolve(d.path), { kind: "array", attrs: false })
        : zarr.open.v3(root.resolve(d.path), { kind: "array" }),
    ),
  );
  return { url, arrays, info: describeImage(ngff, arrays.map((a) => a.shape), arrays[0].dtype) };
}

/** One y/x block of one channel at (t, z). */
async function readBlock(arr, axes, t, c, z, y0, y1, x0, x1) {
  const sel = axes.map((a) =>
    a === "t" ? t : a === "c" ? c : a === "z" ? z : a === "y" ? zarr.slice(y0, y1) : a === "x" ? zarr.slice(x0, x1) : 0,
  );
  return (await zarr.get(arr, sel)).data;
}

const VizarrViewer = ({ zarrUrl = null, onClose = null, embedded = false, height = "100%", width = "100%" }) => {
  const { ip, apiPort } = useSelector(connectionSettingsSlice.getConnectionSettingsState);

  const [groups, setGroups] = useState([]);
  const [folder, setFolder] = useState("");
  const [path, setPath] = useState(relPath(zarrUrl));
  const [images, setImages] = useState([]);
  const [imageIndex, setImageIndex] = useState(0);
  const [image, setImage] = useState(null);
  const [channels, setChannels] = useState([]);
  const [t, setT] = useState(0);
  const [z, setZ] = useState(0);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reload, setReload] = useState(0);
  const [status, setStatus] = useState({ zoom: 1, level: 0, pending: 0 });
  const [deconv, setDeconv] = useState(false);
  const [cropSize, setCropSize] = useState(256);

  const canvasRef = useRef(null);
  const boxRef = useRef(null);
  const viewRef = useRef(null);
  const rawRef = useRef(new Map()); // key -> {data} | {pending: true} | {error}
  const tileRef = useRef(new Map()); // key -> {sig, canvas}
  const frameRef = useRef(0);
  const drawRef = useRef(() => {});

  // A new dataset from the File Manager / experiment button replaces the current one.
  useEffect(() => {
    if (zarrUrl) setPath(relPath(zarrUrl));
  }, [zarrUrl]);

  // Datasets below the data folder (skipped for absolute URLs on another host).
  const loadDatasets = useCallback(() => {
    apiFileManagerGetDatasets()
      .then((list) => setGroups(groupByFolder(list)))
      .catch((e) => setError(`Could not list datasets: ${e.message}`));
  }, []);
  useEffect(() => {
    if (!embedded) loadDatasets();
  }, [embedded, loadDatasets, ip, apiPort, reload]);
  useEffect(() => {
    if (path && !/^https?:/.test(path)) setFolder(parentOf(path));
    else if (!path && groups.length) setFolder((f) => f || groups[0].folder);
  }, [path, groups]);

  // Dataset -> list of images (fields of a plate, or the image itself).
  useEffect(() => {
    if (!path) return undefined;
    let live = true;
    setError(null);
    setImages([]);
    setImage(null);
    listImages(dataUrl(ip, apiPort, path))
      .then((list) => {
        if (!live) return;
        setImages(list);
        setImageIndex(0);
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [path, ip, apiPort, reload]);

  // Image -> arrays, metadata, initial contrast from the coarsest level.
  useEffect(() => {
    const entry = images[imageIndex];
    if (!entry) return undefined;
    let live = true;
    setLoading(true);
    rawRef.current.clear();
    tileRef.current.clear();
    (async () => {
      const img = await openImage(entry.url);
      const { info, arrays } = img;
      const z0 = Math.floor(info.sizes.z / 2);
      const coarse = arrays[arrays.length - 1];
      const [h, w] = info.levels[info.levels.length - 1].shape.slice(-2);
      const chans = await Promise.all(
        info.channels.map(async (ch, c) => {
          const stats = autoWindow(await readBlock(coarse, info.axes, 0, c, z0, 0, h, 0, w));
          const range = Math.max(stats.max, ch.auto ? 0 : ch.end, 1);
          return ch.auto ? { ...ch, start: stats.start, end: stats.end, range } : { ...ch, range };
        }),
      );
      if (!live) return;
      const box = boxRef.current;
      viewRef.current = fitView(info.sizes.x, info.sizes.y, box.clientWidth, box.clientHeight);
      setT(0);
      setZ(z0);
      setChannels(chans);
      setImage(img);
    })()
      .catch((e) => live && setError(`Could not open ${entry.url}: ${e.message}`))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [images, imageIndex]);

  const requestDraw = useCallback(() => {
    if (!frameRef.current) {
      frameRef.current = requestAnimationFrame(() => {
        frameRef.current = 0;
        drawRef.current();
      });
    }
  }, []);

  // Raw plane of one chunk; starts the fetch and redraws when it arrives.
  const raw = useCallback(
    (key, load) => {
      const cache = rawRef.current;
      const hit = cache.get(key);
      if (hit) {
        if (hit.data) {
          cache.delete(key); // LRU: move to the end
          cache.set(key, hit);
        }
        return hit.data || null;
      }
      cache.set(key, { pending: true });
      load()
        .then((data) => cache.set(key, { data }))
        .catch((e) => {
          cache.set(key, { error: e });
          setError(`Could not read a chunk: ${e.message}`);
        })
        .finally(() => {
          while (cache.size > CACHE_TILES) cache.delete(cache.keys().next().value);
          requestDraw();
        });
      return null;
    },
    [requestDraw],
  );

  drawRef.current = () => {
    const canvas = canvasRef.current;
    const box = boxRef.current;
    const view = viewRef.current;
    if (!canvas || !box || !image || !view) return;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(box.clientWidth * dpr);
    const H = Math.round(box.clientHeight * dpr);
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    // view is in CSS pixels; draw in device pixels
    const v = { x: view.x, y: view.y, zoom: view.zoom * dpr };
    const { info, arrays } = image;
    const sig = JSON.stringify(channels.map((c) => [c.visible, c.rgb, c.start, c.end]));
    const coarsest = info.levels.length - 1;
    const best = pickLevel(info.levels, v.zoom);
    let pending = 0;
    const drawLevel = (li) => {
      const level = info.levels[li];
      const chunk = arrays[li].chunks.slice(-2);
      for (const tile of tilesInView(level, chunk, v, W, H)) {
        const base = `${li}|${t}|${z}|${tile.ty}|${tile.tx}`;
        let entry = tileRef.current.get(base);
        if (!entry || entry.sig !== sig) {
          const planes = channels.map((ch, c) =>
            ch.visible
              ? raw(`${base}|${c}`, () =>
                  readBlock(arrays[li], info.axes, t, c, z, tile.y0, tile.y0 + tile.h, tile.x0, tile.x0 + tile.w),
                )
              : null,
          );
          if (planes.some((p, c) => channels[c].visible && !p)) {
            pending += 1;
            continue;
          }
          const tc = entry?.canvas || document.createElement("canvas");
          tc.width = tile.w;
          tc.height = tile.h;
          const tctx = tc.getContext("2d");
          const img = tctx.createImageData(tile.w, tile.h);
          compositeRGBA(planes, channels, img.data);
          tctx.putImageData(img, 0, 0);
          entry = { sig, canvas: tc };
          tileRef.current.set(base, entry);
          if (tileRef.current.size > CACHE_TILES) tileRef.current.delete(tileRef.current.keys().next().value);
        }
        const f = level.factor;
        ctx.imageSmoothingEnabled = f * v.zoom < 1;
        ctx.drawImage(
          entry.canvas,
          (tile.x0 * f - v.x) * v.zoom,
          (tile.y0 * f - v.y) * v.zoom,
          tile.w * f * v.zoom,
          tile.h * f * v.zoom,
        );
      }
    };
    drawLevel(coarsest); // whole image at once, so panning never shows holes
    if (best !== coarsest) drawLevel(best);
    if (deconv) {
      // the crop a deconvolution preview takes: the middle of the view
      const crop = cropAt(view, box.clientWidth, box.clientHeight, cropSize, info.sizes.x, info.sizes.y);
      ctx.setLineDash([6 * dpr, 4 * dpr]);
      ctx.strokeStyle = "#ffd54f";
      ctx.lineWidth = dpr;
      ctx.strokeRect((crop.x0 - v.x) * v.zoom, (crop.y0 - v.y) * v.zoom, crop.w * v.zoom, crop.h * v.zoom);
      ctx.setLineDash([]);
    }
    setStatus((s) =>
      s.zoom === view.zoom && s.level === best && s.pending === pending ? s : { zoom: view.zoom, level: best, pending },
    );
  };

  useEffect(requestDraw, [image, channels, t, z, requestDraw, deconv, cropSize]);

  // Resize, wheel zoom (non-passive so the page does not scroll) and drag to pan.
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return undefined;
    const ro = new ResizeObserver(requestDraw);
    ro.observe(box);
    const onWheel = (e) => {
      if (!viewRef.current) return;
      e.preventDefault();
      const r = box.getBoundingClientRect();
      viewRef.current = zoomAt(viewRef.current, e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.2 : 1 / 1.2);
      requestDraw();
    };
    box.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      ro.disconnect();
      box.removeEventListener("wheel", onWheel);
    };
  }, [requestDraw]);

  const onPointerDown = (e) => {
    if (!viewRef.current) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    let last = [e.clientX, e.clientY];
    const move = (ev) => {
      const v = viewRef.current;
      viewRef.current = { ...v, x: v.x - (ev.clientX - last[0]) / v.zoom, y: v.y - (ev.clientY - last[1]) / v.zoom };
      last = [ev.clientX, ev.clientY];
      requestDraw();
    };
    const up = () => {
      e.target.removeEventListener("pointermove", move);
      e.target.removeEventListener("pointerup", up);
    };
    e.target.addEventListener("pointermove", move);
    e.target.addEventListener("pointerup", up);
  };

  const fit = () => {
    if (!image) return;
    const box = boxRef.current;
    viewRef.current = fitView(image.info.sizes.x, image.info.sizes.y, box.clientWidth, box.clientHeight);
    requestDraw();
  };
  const oneToOne = () => {
    const box = boxRef.current;
    const v = viewRef.current;
    if (!v) return;
    viewRef.current = zoomAt(v, box.clientWidth / 2, box.clientHeight / 2, 1 / (v.zoom * (window.devicePixelRatio || 1)));
    requestDraw();
  };

  const setChannel = (c, patch) => setChannels((list) => list.map((ch, i) => (i === c ? { ...ch, ...patch } : ch)));
  const autoChannel = (c) => {
    const { info, arrays } = image;
    const li = info.levels.length - 1;
    const [h, w] = info.levels[li].shape.slice(-2);
    readBlock(arrays[li], info.axes, t, c, z, 0, h, 0, w).then((data) => {
      const s = autoWindow(data);
      setChannel(c, { start: s.start, end: s.end, range: Math.max(channels[c].range, s.max) });
    });
  };

  const group = useMemo(() => groups.find((g) => g.folder === folder), [groups, folder]);
  const info = image?.info;
  const remote = /^https?:/.test(path);
  // the image inside the store (a plate field such as A/1/0); "" for the store itself
  const imageRel = image && !remote ? image.url.slice(dataUrl(ip, apiPort, path).length).replace(/^\//, "") : "";
  const getCrop = () => {
    const box = boxRef.current;
    if (!viewRef.current || !box || !info) return null;
    return cropAt(viewRef.current, box.clientWidth, box.clientHeight, cropSize, info.sizes.x, info.sizes.y);
  };
  const name = path.split("/").pop();
  const level = info?.levels[status.level];

  return (
    <Box sx={{ height: embedded ? height : "100%", width, display: "flex", bgcolor: "background.default", minHeight: 0 }}>
      {!embedded && (
        <Box sx={{ width: 300, flexShrink: 0, borderRight: 1, borderColor: "divider", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <Box sx={{ p: 1, display: "flex", alignItems: "center", gap: 1 }}>
            <TextField
              select
              size="small"
              fullWidth
              label="Run folder"
              value={group ? folder : ""}
              onChange={(e) => setFolder(e.target.value)}
            >
              {groups.map((g) => (
                <MenuItem key={g.folder} value={g.folder} dense>
                  {g.folder}
                </MenuItem>
              ))}
            </TextField>
            <Tooltip title="Reload list">
              <IconButton size="small" onClick={() => setReload((n) => n + 1)}>
                <Refresh fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          {!groups.length && (
            <Typography variant="body2" color="text.secondary" sx={{ px: 2 }}>
              No OME-Zarr datasets in the data folder yet.
            </Typography>
          )}
          <List dense sx={{ overflow: "auto", flex: 1, py: 0 }}>
            {(group?.items || []).map((d) => (
              <ListItemButton key={d.path} selected={d.path === path} onClick={() => setPath(d.path)}>
                <ListItemText primary={d.name.replace(/\.ome\.zarr$|\.zarr$/, "")} primaryTypographyProps={{ noWrap: true, title: d.name }} />
              </ListItemButton>
            ))}
          </List>
          {images.length > 1 && (
            <>
              <Divider />
              <Typography variant="caption" color="text.secondary" sx={{ px: 2, pt: 1 }}>
                Fields ({images.length})
              </Typography>
              <List dense sx={{ overflow: "auto", maxHeight: "35%", py: 0 }}>
                {images.map((im, i) => (
                  <ListItemButton key={im.url} selected={i === imageIndex} onClick={() => setImageIndex(i)}>
                    <ListItemText primary={im.label} />
                  </ListItemButton>
                ))}
              </List>
            </>
          )}
        </Box>
      )}

      <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0 }}>
        <Box sx={{ px: 1, py: 0.5, display: "flex", alignItems: "center", gap: 1, borderBottom: 1, borderColor: "divider" }}>
          <Typography variant="subtitle2" noWrap title={path} sx={{ minWidth: 0 }}>
            {name || "OME-Zarr Viewer"}
          </Typography>
          {info && (
            <Typography variant="caption" color="text.secondary" noWrap>
              {info.sizes.x} × {info.sizes.y} px · {info.sizes.c} c · {info.sizes.z} z · {info.sizes.t} t ·{" "}
              {image.arrays[0].dtype} · {info.levels.length} levels
            </Typography>
          )}
          <Box sx={{ flex: 1 }} />
          <Tooltip title="Deconvolve: preview on a crop, then the whole stack">
            <span>
              <Button
                size="small"
                variant={deconv ? "contained" : "text"}
                disableElevation
                onClick={() => setDeconv((d) => !d)}
                disabled={!info || remote}
              >
                Deconvolve
              </Button>
            </span>
          </Tooltip>
          <Button size="small" onClick={oneToOne} disabled={!info}>
            1:1
          </Button>
          <Tooltip title="Fit">
            <span>
              <IconButton size="small" onClick={fit} disabled={!info}>
                <FitScreen fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Reload">
            <IconButton size="small" onClick={() => setReload((n) => n + 1)}>
              <Refresh fontSize="small" />
            </IconButton>
          </Tooltip>
          {onClose && (
            <Tooltip title="Close">
              <IconButton size="small" onClick={onClose}>
                <Close fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
        {(loading || status.pending > 0) && <LinearProgress sx={{ height: 2 }} />}
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ borderRadius: 0 }}>
            {error}
          </Alert>
        )}

        <Box
          ref={boxRef}
          onPointerDown={onPointerDown}
          sx={{ flex: 1, position: "relative", overflow: "hidden", bgcolor: "#000", cursor: "grab", touchAction: "none", minHeight: 200 }}
        >
          <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
          {!path && (
            <Typography color="text.secondary" sx={{ position: "absolute", top: "45%", width: "100%", textAlign: "center" }}>
              Pick a dataset on the left, or use "Open" on a .zarr in the File Manager.
            </Typography>
          )}
        </Box>

        {info && (
          <Box sx={{ px: 2, py: 1, borderTop: 1, borderColor: "divider", maxHeight: "40%", overflow: "auto" }}>
            {[
              ["T", info.sizes.t, t, setT],
              ["Z", info.sizes.z, z, setZ],
            ]
              .filter(([, n]) => n > 1)
              .map(([label, n, value, set]) => (
                <Box key={label} sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <Typography variant="body2" sx={{ width: 90 }}>
                    {label} {value + 1} / {n}
                  </Typography>
                  <Slider size="small" min={0} max={n - 1} step={1} value={value} onChange={(_, val) => set(val)} />
                </Box>
              ))}
            {channels.map((ch, c) => (
              <Box key={c} sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Checkbox size="small" checked={ch.visible} onChange={(e) => setChannel(c, { visible: e.target.checked })} sx={{ p: 0.5 }} />
                <input
                  type="color"
                  value={toHex(ch.rgb)}
                  onChange={(e) => setChannel(c, { rgb: fromHex(e.target.value) })}
                  style={{ width: 22, height: 22, padding: 0, border: "none", background: "none" }}
                  aria-label={`${ch.label} colour`}
                />
                <Typography variant="body2" noWrap sx={{ width: 110 }} title={ch.label}>
                  {ch.label}
                </Typography>
                <Slider
                  size="small"
                  min={0}
                  max={ch.range}
                  step={ch.range > 10 ? 1 : ch.range / 1000}
                  value={[ch.start, ch.end]}
                  onChange={(_, [start, end]) => setChannel(c, { start, end: Math.max(end, start + 1e-6) })}
                />
                <Typography variant="caption" sx={{ width: 110, fontFamily: "monospace", textAlign: "right" }}>
                  {Math.round(ch.start)}–{Math.round(ch.end)}
                </Typography>
                <Button size="small" onClick={() => autoChannel(c)}>
                  Auto
                </Button>
              </Box>
            ))}
            <Typography variant="caption" color="text.secondary" component="div" sx={{ fontFamily: "monospace", mt: 0.5 }}>
              level {status.level} (1:{level?.factor}) · zoom {(status.zoom * 100).toFixed(status.zoom < 0.1 ? 1 : 0)} %
              {info.pixelSize ? ` · ${info.pixelSize} ${info.unit === "micrometer" ? "µm" : info.unit}/px` : ""}
              {images.length > 1 ? ` · field ${images[imageIndex]?.label}` : ""}
            </Typography>
          </Box>
        )}
      </Box>
      {deconv && info && !remote && (
        <DeconvolutionPanel
          path={path}
          image={imageRel}
          t={t}
          z={z}
          getCrop={getCrop}
          cropSize={cropSize}
          setCropSize={setCropSize}
          onClose={() => setDeconv(false)}
          onOpen={(result) => {
            setPath(result);
            setReload((n) => n + 1);
          }}
        />
      )}
    </Box>
  );
};

export default VizarrViewer;
