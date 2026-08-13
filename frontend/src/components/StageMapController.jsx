// src/components/StageMapController.jsx
// Stage Map: MicroMagellan / Google-Maps-style live stitched map of the
// sample. While mapping runs, the backend snaps a camera frame wherever the
// stage settles and pushes a downscaled preview tile (sigStageMapTileAdded).
// The tiles are rendered on this pan/zoom canvas at their stage coordinates,
// so a stitched overview grows as the user moves around. Double-click moves
// the stage; channels can be toggled/tinted as color overlays; the raw
// full-resolution tiles can be fused into a stitched OME-TIFF on the backend.

import {
  AddAPhoto as SnapIcon,
  CenterFocusStrong as FitIcon,
  Delete as ClearIcon,
  GpsFixed as FollowIcon,
  PlayArrow as StartIcon,
  Save as SaveIcon,
  Settings as SettingsIcon,
  Stop as StopIcon,
  SwapVert as FlipIcon,
} from "@mui/icons-material";
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  FormControlLabel,
  IconButton,
  MenuItem,
  Paper,
  Popover,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useDispatch, useSelector } from "react-redux";

import {
  apiStageMapClear,
  apiStageMapGetParams,
  apiStageMapGetStatus,
  apiStageMapGetTiles,
  apiStageMapGotoPosition,
  apiStageMapSaveOmeTiff,
  apiStageMapSetChannel,
  apiStageMapSetParams,
  apiStageMapSnapTile,
  apiStageMapStart,
  apiStageMapStop,
} from "../backendapi/apiStageMapController";
import * as notificationSlice from "../state/slices/NotificationSlice";
import * as positionSlice from "../state/slices/PositionSlice";
import * as stageMapSlice from "../state/slices/StageMapSlice";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Tint a grayscale tile image with a channel color on an offscreen canvas. */
const makeTintedCanvas = (img, color) => {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  if (color && color.toLowerCase() !== "#ffffff") {
    ctx.globalCompositeOperation = "multiply";
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, c.width, c.height);
  }
  return c;
};

const niceScaleBarLength = (targetUm) => {
  const pow = Math.pow(10, Math.floor(Math.log10(targetUm)));
  for (const m of [5, 2, 1]) {
    if (m * pow <= targetUm) return m * pow;
  }
  return pow;
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const StageMapController = () => {
  const dispatch = useDispatch();
  const theme = useTheme();

  const stageMapState = useSelector(stageMapSlice.getStageMapState);
  const positionState = useSelector(positionSlice.getPositionState);
  const { tiles, channels, isMapping, status } = stageMapState;

  // View transform lives in a ref so pan/zoom does not re-render React.
  const viewRef = useRef({ cx: 0, cy: 0, scale: 0.2, flipY: false, initialized: false });
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const imageCacheRef = useRef(new Map()); // tileId -> {img, ready, tintKey, tinted}
  const dragRef = useRef(null);
  const rafRef = useRef(null);

  const [followStage, setFollowStage] = useState(true);
  const [flipY, setFlipY] = useState(false);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [settingsAnchor, setSettingsAnchor] = useState(null);
  const [params, setParams] = useState(null);
  const [channelInput, setChannelInput] = useState("");

  const notify = useCallback(
    (message, type = "info") =>
      dispatch(notificationSlice.setNotification({ message, type })),
    [dispatch],
  );

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  // draw() closes over the current tiles/channels/theme; requestDraw is a
  // stable identity that always calls the latest draw via this ref.
  const drawRef = useRef(() => {});
  const requestDraw = useCallback(() => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      drawRef.current();
    });
  }, []);

  const getTileBitmap = useCallback(
    (tile, color) => {
      if (!tile.image) return null;
      const cache = imageCacheRef.current;
      let entry = cache.get(tile.id);
      if (!entry) {
        const img = new Image();
        entry = { img, ready: false, tintKey: null, tinted: null };
        img.onload = () => {
          entry.ready = true;
          requestDraw();
        };
        img.src = `data:image/jpeg;base64,${tile.image}`;
        cache.set(tile.id, entry);
      }
      if (!entry.ready) return null;
      if (!color || color.toLowerCase() === "#ffffff") return entry.img;
      if (entry.tintKey !== color) {
        entry.tinted = makeTintedCanvas(entry.img, color);
        entry.tintKey = color;
      }
      return entry.tinted;
    },
    [requestDraw],
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    }

    const view = viewRef.current;
    const ctx = canvas.getContext("2d");
    const isDark = theme.palette.mode === "dark";

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = isDark ? "#101418" : "#e8eaed";
    ctx.fillRect(0, 0, w, h);

    // stage-coordinate transform: x right, y down (or up when flipY)
    const ySign = view.flipY ? -1 : 1;
    const toScreenX = (x) => w / 2 + (x - view.cx) * view.scale;
    const toScreenY = (y) => h / 2 + ySign * (y - view.cy) * view.scale;

    // --- grid (adaptive spacing, in stage µm) ---
    const gridTarget = 80 / view.scale; // ~80 px spacing
    const gridStep = niceScaleBarLength(gridTarget) || 100;
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)";
    ctx.lineWidth = 1;
    const x0 = view.cx - w / 2 / view.scale;
    const x1 = view.cx + w / 2 / view.scale;
    const yHalf = h / 2 / view.scale;
    const y0 = view.cy - yHalf;
    const y1 = view.cy + yHalf;
    ctx.beginPath();
    for (let gx = Math.floor(x0 / gridStep) * gridStep; gx <= x1; gx += gridStep) {
      const sx = toScreenX(gx);
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, h);
    }
    for (let gy = Math.floor(y0 / gridStep) * gridStep; gy <= y1; gy += gridStep) {
      const sy = toScreenY(gy);
      ctx.moveTo(0, sy);
      ctx.lineTo(w, sy);
    }
    ctx.stroke();

    // --- tiles, grouped by channel; additive blending for overlays ---
    const visibleChannels = Object.entries(channels)
      .filter(([, c]) => c.visible)
      .map(([name]) => name);
    const additive = visibleChannels.length > 1;

    visibleChannels.forEach((channelName, channelIndex) => {
      const color = channels[channelName]?.color || "#ffffff";
      ctx.globalCompositeOperation =
        additive && channelIndex > 0 ? "lighter" : "source-over";
      for (const tile of tiles) {
        if ((tile.channel || "default") !== channelName) continue;
        const bmp = getTileBitmap(tile, color);
        if (!bmp) continue;
        const sw = tile.widthUm * view.scale;
        const sh = tile.heightUm * view.scale;
        const sx = toScreenX(tile.x - tile.widthUm / 2);
        // top edge depends on y direction
        const syTop = view.flipY
          ? toScreenY(tile.y + tile.heightUm / 2)
          : toScreenY(tile.y - tile.heightUm / 2);
        if (sx > w || syTop > h || sx + sw < 0 || syTop + sh < 0) continue;
        if (view.flipY) {
          ctx.save();
          ctx.translate(sx, syTop + sh);
          ctx.scale(1, -1);
          ctx.drawImage(bmp, 0, 0, sw, sh);
          ctx.restore();
        } else {
          ctx.drawImage(bmp, sx, syTop, sw, sh);
        }
      }
    });
    ctx.globalCompositeOperation = "source-over";

    // --- current FOV rectangle + position marker ---
    const px = positionState.x ?? 0;
    const py = positionState.y ?? 0;
    if (status.fovX > 0 && status.fovY > 0) {
      const sw = status.fovX * view.scale;
      const sh = status.fovY * view.scale;
      ctx.strokeStyle = theme.palette.primary.main;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(
        toScreenX(px - status.fovX / 2),
        view.flipY ? toScreenY(py + status.fovY / 2) : toScreenY(py - status.fovY / 2),
        sw,
        sh,
      );
      ctx.setLineDash([]);
    }
    ctx.fillStyle = theme.palette.primary.main;
    ctx.beginPath();
    ctx.arc(toScreenX(px), toScreenY(py), 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = isDark ? "#000" : "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // --- scale bar ---
    const barUm = niceScaleBarLength(120 / view.scale);
    const barPx = barUm * view.scale;
    const bx = 16;
    const by = h - 20;
    ctx.strokeStyle = isDark ? "#fff" : "#111";
    ctx.fillStyle = isDark ? "#fff" : "#111";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx + barPx, by);
    ctx.stroke();
    ctx.font = "12px sans-serif";
    ctx.fillText(
      barUm >= 1000 ? `${barUm / 1000} mm` : `${barUm} µm`,
      bx,
      by - 6,
    );
  }, [channels, tiles, positionState, status, theme, getTileBitmap]);

  // Keep the ref pointing at the latest draw closure
  useEffect(() => {
    drawRef.current = draw;
  }, [draw]);

  // Redraw when data changes
  useEffect(() => {
    requestDraw();
  }, [tiles, channels, positionState, status, flipY, requestDraw, draw]);

  // Keep viewRef.flipY in sync with UI state
  useEffect(() => {
    viewRef.current.flipY = flipY;
    requestDraw();
  }, [flipY, requestDraw]);

  // Resize handling
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    const observer = new ResizeObserver(() => requestDraw());
    observer.observe(container);
    return () => observer.disconnect();
  }, [requestDraw]);

  // Follow the stage position
  useEffect(() => {
    if (!followStage) return;
    const view = viewRef.current;
    view.cx = positionState.x ?? 0;
    view.cy = positionState.y ?? 0;
    requestDraw();
  }, [positionState.x, positionState.y, followStage, requestDraw]);

  // ------------------------------------------------------------------
  // Data loading / polling
  // ------------------------------------------------------------------

  const refreshStatus = useCallback(async () => {
    try {
      const s = await apiStageMapGetStatus();
      dispatch(stageMapSlice.setStatus(s));
      // center once on the current stage position on first load
      if (!viewRef.current.initialized) {
        viewRef.current.cx = s.positionX ?? 0;
        viewRef.current.cy = s.positionY ?? 0;
        viewRef.current.initialized = true;
        requestDraw();
      }
    } catch (e) {
      // backend not reachable / controller missing - leave state as-is
    }
  }, [dispatch, requestDraw]);

  useEffect(() => {
    // initial load: status, params and any tiles from a previous session
    refreshStatus();
    apiStageMapGetParams()
      .then((p) => {
        setParams(p);
        setChannelInput(p.activeChannel || "");
      })
      .catch(() => {});
    apiStageMapGetTiles(0, true)
      .then((data) => {
        if (Array.isArray(data?.tiles) && data.tiles.length > 0) {
          dispatch(stageMapSlice.setTiles(data.tiles));
        }
      })
      .catch(() => {});

    const interval = setInterval(refreshStatus, 3000);
    return () => clearInterval(interval);
  }, [dispatch, refreshStatus]);

  // Free the tile image cache when tiles are cleared
  useEffect(() => {
    if (tiles.length === 0) imageCacheRef.current.clear();
  }, [tiles.length]);

  // ------------------------------------------------------------------
  // Interactions: pan / zoom / double-click-to-move
  // ------------------------------------------------------------------

  const screenToStage = useCallback((clientX, clientY) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const view = viewRef.current;
    const ySign = view.flipY ? -1 : 1;
    const x = view.cx + (clientX - rect.left - rect.width / 2) / view.scale;
    const y =
      view.cy + (ySign * (clientY - rect.top - rect.height / 2)) / view.scale;
    return { x, y };
  }, []);

  const handlePointerDown = useCallback((e) => {
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      cx: viewRef.current.cx,
      cy: viewRef.current.cy,
    };
    e.currentTarget.setPointerCapture(e.pointerId);
  }, []);

  const handlePointerMove = useCallback(
    (e) => {
      if (!dragRef.current) return;
      const view = viewRef.current;
      const ySign = view.flipY ? -1 : 1;
      view.cx = dragRef.current.cx - (e.clientX - dragRef.current.startX) / view.scale;
      view.cy =
        dragRef.current.cy - (ySign * (e.clientY - dragRef.current.startY)) / view.scale;
      if (
        followStage &&
        (Math.abs(e.clientX - dragRef.current.startX) > 3 ||
          Math.abs(e.clientY - dragRef.current.startY) > 3)
      ) {
        setFollowStage(false); // manual pan disables follow mode
      }
      requestDraw();
    },
    [followStage, requestDraw],
  );

  const handlePointerUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  const handleWheel = useCallback(
    (e) => {
      e.preventDefault();
      const view = viewRef.current;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const newScale = Math.min(50, Math.max(0.0005, view.scale * factor));
      // zoom around the cursor: keep the stage point under it fixed
      const before = screenToStage(e.clientX, e.clientY);
      view.scale = newScale;
      const after = screenToStage(e.clientX, e.clientY);
      view.cx += before.x - after.x;
      view.cy += before.y - after.y;
      requestDraw();
    },
    [screenToStage, requestDraw],
  );

  // Non-passive wheel listener (React onWheel is passive and can't preventDefault)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  const handleDoubleClick = useCallback(
    async (e) => {
      const { x, y } = screenToStage(e.clientX, e.clientY);
      try {
        await apiStageMapGotoPosition(x, y, true);
        notify(`Moving stage to X=${x.toFixed(1)} µm, Y=${y.toFixed(1)} µm`);
      } catch (err) {
        notify("Failed to move stage", "error");
      }
    },
    [screenToStage, notify],
  );

  // ------------------------------------------------------------------
  // Keyboard: arrow keys step the stage by exactly one FOV and snap a
  // tile at the new position, so the map can be grown tile-by-tile.
  // ------------------------------------------------------------------

  // The key handler reads live values through a ref so the window listener
  // is registered once and never sees stale FOV / mapping state.
  const arrowMoveBusyRef = useRef(false);
  const arrowStateRef = useRef({});
  arrowStateRef.current = {
    fovX: status.fovX,
    fovY: status.fovY,
    flipY,
    isMapping,
    autoSnap: params?.autoSnapEnabled,
    settleS: params?.settleTimeS,
  };

  useEffect(() => {
    const isTypingTarget = (el) =>
      el &&
      (el.tagName === "INPUT" ||
        el.tagName === "TEXTAREA" ||
        el.tagName === "SELECT" ||
        el.isContentEditable);

    const onKeyDown = (e) => {
      const dirs = {
        ArrowLeft: [-1, 0],
        ArrowRight: [1, 0],
        ArrowUp: [0, -1],
        ArrowDown: [0, 1],
      };
      const dir = dirs[e.key];
      if (!dir) return;
      if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault(); // keep the page from scrolling
      if (arrowMoveBusyRef.current) return; // swallow key-repeat while moving

      const {
        fovX,
        fovY,
        flipY: flip,
        isMapping: mapping,
        autoSnap,
        settleS,
      } = arrowStateRef.current;
      if (!fovX || !fovY) {
        notify("Field of view unknown — cannot step by one FOV", "warning");
        return;
      }
      // Screen-up is -Y in the default orientation and +Y when Y is flipped,
      // so the stage always steps the way the map visually moves.
      const dx = dir[0] * fovX;
      const dy = dir[1] * (flip ? -fovY : fovY);

      arrowMoveBusyRef.current = true;
      (async () => {
        try {
          await apiStageMapGotoPosition(dx, dy, false, true); // relative + blocking
          // While mapping with auto-snap the backend captures on settle by
          // itself; otherwise snap explicitly so the tile shows up right away.
          if (!(mapping && autoSnap)) {
            await new Promise((r) =>
              setTimeout(r, Math.max(0, (settleS ?? 0.25) * 1000)),
            );
            await apiStageMapSnapTile();
          }
        } catch (err) {
          notify("Arrow-key stage step failed", "error");
        } finally {
          arrowMoveBusyRef.current = false;
        }
      })();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [notify]);

  const fitToTiles = useCallback(() => {
    if (tiles.length === 0) return;
    const view = viewRef.current;
    const minX = Math.min(...tiles.map((t) => t.x - t.widthUm / 2));
    const maxX = Math.max(...tiles.map((t) => t.x + t.widthUm / 2));
    const minY = Math.min(...tiles.map((t) => t.y - t.heightUm / 2));
    const maxY = Math.max(...tiles.map((t) => t.y + t.heightUm / 2));
    const container = containerRef.current;
    view.cx = (minX + maxX) / 2;
    view.cy = (minY + maxY) / 2;
    const pad = 1.15;
    view.scale = Math.min(
      50,
      Math.max(
        0.0005,
        Math.min(
          container.clientWidth / ((maxX - minX) * pad || 1),
          container.clientHeight / ((maxY - minY) * pad || 1),
        ),
      ),
    );
    setFollowStage(false);
    requestDraw();
  }, [tiles, requestDraw]);

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------

  const handleStartStop = useCallback(async () => {
    setBusy(true);
    try {
      if (isMapping) {
        await apiStageMapStop();
        dispatch(stageMapSlice.setIsMapping(false));
      } else {
        const ok = await apiStageMapStart();
        if (ok === false) {
          notify("Could not start mapping (detector/stage missing?)", "error");
        } else {
          dispatch(stageMapSlice.setIsMapping(true));
        }
      }
      refreshStatus();
    } catch (e) {
      notify("Start/stop request failed", "error");
    } finally {
      setBusy(false);
    }
  }, [isMapping, dispatch, refreshStatus, notify]);

  const handleSnap = useCallback(async () => {
    try {
      await apiStageMapSnapTile();
    } catch (e) {
      notify("Snap failed", "error");
    }
  }, [notify]);

  const handleClear = useCallback(async () => {
    if (!window.confirm("Discard the current map? Raw tiles already saved on disk are kept.")) {
      return;
    }
    try {
      await apiStageMapClear();
      dispatch(stageMapSlice.clearTiles());
      imageCacheRef.current.clear();
      refreshStatus();
    } catch (e) {
      notify("Clear failed", "error");
    }
  }, [dispatch, refreshStatus, notify]);

  const handleSaveOmeTiff = useCallback(async () => {
    setSaving(true);
    try {
      const result = await apiStageMapSaveOmeTiff("");
      if (result?.success) {
        notify(`Stitched OME-TIFF saved: ${result.path}`, "success");
      } else {
        notify(`OME-TIFF export failed: ${result?.error || "unknown"}`, "error");
      }
    } catch (e) {
      notify("OME-TIFF export failed", "error");
    } finally {
      setSaving(false);
    }
  }, [notify]);

  const handleChannelApply = useCallback(async () => {
    try {
      await apiStageMapSetChannel(channelInput.trim());
      refreshStatus();
    } catch (e) {
      notify("Could not set channel", "error");
    }
  }, [channelInput, refreshStatus, notify]);

  const handleParamChange = useCallback(
    async (patch) => {
      if (!params) return;
      const next = { ...params, ...patch };
      setParams(next);
      try {
        await apiStageMapSetParams(next);
      } catch (e) {
        notify("Could not update parameters", "error");
      }
    },
    [params, notify],
  );

  const channelEntries = useMemo(() => Object.entries(channels), [channels]);

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: 1,
        minHeight: 0,
      }}
    >
      {/* Toolbar */}
      <Paper sx={{ p: 1 }}>
        <Stack
          direction="row"
          spacing={1}
          alignItems="center"
          flexWrap="wrap"
          useFlexGap
        >
          <Button
            variant="contained"
            color={isMapping ? "error" : "success"}
            startIcon={
              busy ? <CircularProgress size={16} /> : isMapping ? <StopIcon /> : <StartIcon />
            }
            onClick={handleStartStop}
            disabled={busy}
          >
            {isMapping ? "Stop Mapping" : "Start Mapping"}
          </Button>

          <Tooltip title="Snap a tile at the current position now">
            <span>
              <IconButton onClick={handleSnap} color="primary">
                <SnapIcon />
              </IconButton>
            </span>
          </Tooltip>

          <Tooltip title="Fit view to the collected map">
            <span>
              <IconButton onClick={fitToTiles} disabled={tiles.length === 0}>
                <FitIcon />
              </IconButton>
            </span>
          </Tooltip>

          <Tooltip title={followStage ? "Following stage (click to unlock)" : "Follow stage position"}>
            <IconButton
              color={followStage ? "primary" : "default"}
              onClick={() => setFollowStage((v) => !v)}
            >
              <FollowIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Flip Y axis (stage vs. screen orientation)">
            <IconButton
              color={flipY ? "primary" : "default"}
              onClick={() => setFlipY((v) => !v)}
            >
              <FlipIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Discard the current map">
            <span>
              <IconButton onClick={handleClear} disabled={tiles.length === 0} color="warning">
                <ClearIcon />
              </IconButton>
            </span>
          </Tooltip>

          <Button
            variant="outlined"
            startIcon={saving ? <CircularProgress size={16} /> : <SaveIcon />}
            onClick={handleSaveOmeTiff}
            disabled={saving || status.tileCount === 0}
          >
            Save OME-TIFF
          </Button>

          <IconButton onClick={(e) => setSettingsAnchor(e.currentTarget)}>
            <SettingsIcon />
          </IconButton>

          <Divider orientation="vertical" flexItem />

          <TextField
            size="small"
            label="Channel label"
            value={channelInput}
            placeholder="auto"
            onChange={(e) => setChannelInput(e.target.value)}
            onBlur={handleChannelApply}
            onKeyDown={(e) => e.key === "Enter" && handleChannelApply()}
            sx={{ width: 140 }}
          />

          <Typography variant="body2" color="text.secondary" sx={{ ml: "auto" }}>
            {status.tileCount} tiles · px {Number(status.pixelSizeUm).toFixed(3)} µm ·{" "}
            X {Number(positionState.x ?? 0).toFixed(1)} / Y{" "}
            {Number(positionState.y ?? 0).toFixed(1)} µm
          </Typography>
        </Stack>

        {/* Channel overlay chips */}
        {channelEntries.length > 0 && (
          <Stack direction="row" spacing={1} sx={{ mt: 1 }} flexWrap="wrap" useFlexGap>
            {channelEntries.map(([name, c]) => (
              <Chip
                key={name}
                label={name}
                size="small"
                variant={c.visible ? "filled" : "outlined"}
                onClick={() =>
                  dispatch(
                    stageMapSlice.setChannelVisible({
                      channel: name,
                      visible: !c.visible,
                    }),
                  )
                }
                sx={{
                  borderColor: c.color,
                  bgcolor: c.visible ? `${c.color}33` : "transparent",
                  "& .MuiChip-label": { fontWeight: 500 },
                }}
                icon={
                  <Box
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: "50%",
                      bgcolor: c.color,
                      ml: "6px !important",
                    }}
                  />
                }
              />
            ))}
          </Stack>
        )}
      </Paper>

      {/* Map canvas */}
      <Box
        ref={containerRef}
        sx={{
          flexGrow: 1,
          position: "relative",
          minHeight: 300,
          borderRadius: 1,
          overflow: "hidden",
          cursor: "grab",
          "&:active": { cursor: "grabbing" },
        }}
      >
        <canvas
          ref={canvasRef}
          style={{ display: "block", width: "100%", height: "100%" }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
          onDoubleClick={handleDoubleClick}
        />
        <Typography
          variant="caption"
          sx={{
            position: "absolute",
            top: 8,
            right: 12,
            color: "text.secondary",
            pointerEvents: "none",
          }}
        >
          drag: pan · wheel: zoom · double-click: move stage here · arrow keys:
          step 1 FOV + snap
        </Typography>
      </Box>

      {/* Settings popover */}
      <Popover
        open={Boolean(settingsAnchor)}
        anchorEl={settingsAnchor}
        onClose={() => setSettingsAnchor(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      >
        <Box sx={{ p: 2, width: 320 }}>
          <Typography variant="subtitle2" gutterBottom>
            Stage Map Settings
          </Typography>
          {params ? (
            <Stack spacing={2} sx={{ mt: 1 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={params.autoSnapEnabled}
                    onChange={(e) =>
                      handleParamChange({ autoSnapEnabled: e.target.checked })
                    }
                  />
                }
                label="Auto-snap when stage settles"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={params.saveRawTiles}
                    onChange={(e) =>
                      handleParamChange({ saveRawTiles: e.target.checked })
                    }
                  />
                }
                label="Store raw full-res tiles on disk"
              />
              <TextField
                select
                size="small"
                label="Preview tile size (px, longest side)"
                value={params.previewMaxSize}
                onChange={(e) =>
                  handleParamChange({ previewMaxSize: Number(e.target.value) })
                }
              >
                {[128, 256, 384, 512, 768].map((v) => (
                  <MenuItem key={v} value={v}>
                    {v}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                size="small"
                type="number"
                label="Min move before new tile (fraction of FOV)"
                inputProps={{ step: 0.05, min: 0.05, max: 1 }}
                value={params.minMoveFraction}
                onChange={(e) =>
                  handleParamChange({
                    minMoveFraction: Math.max(0.05, Number(e.target.value) || 0.35),
                  })
                }
              />
              <TextField
                size="small"
                type="number"
                label="Settle time (s)"
                inputProps={{ step: 0.05, min: 0 }}
                value={params.settleTimeS}
                onChange={(e) =>
                  handleParamChange({ settleTimeS: Math.max(0, Number(e.target.value) || 0.25) })
                }
              />
              <TextField
                size="small"
                type="number"
                label="Pixel size override (µm, 0 = from detector)"
                inputProps={{ step: 0.01, min: 0 }}
                value={params.pixelSizeUm}
                onChange={(e) =>
                  handleParamChange({ pixelSizeUm: Math.max(0, Number(e.target.value) || 0) })
                }
              />
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Parameters not loaded (backend unreachable?)
            </Typography>
          )}
        </Box>
      </Popover>
    </Box>
  );
};

export default StageMapController;
