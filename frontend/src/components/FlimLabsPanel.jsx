/**
 * FlimLabsPanel.jsx - Dev-mode bridge between the galvo scanner and a remote
 * FLIM LABS flim-imager server (Docker, port 5249).
 *
 * The panel talks to the FLIM server directly from the browser (the server
 * allows any CORS origin): REST for control, binary WebSocket (/data) for the
 * live line stream, rendered into a canvas next to the galvo settings.
 *
 * Scan geometry / dwell can be derived from the current galvo raster config so
 * that both devices always agree (galvo nx/ny -> FLIM image size, galvo
 * sample_period_us -> FLIM dwell time). Optionally the galvo scan is started
 * automatically right after the FLIM acquisition is armed.
 */
import React, { useRef, useState, useCallback, useEffect } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  TextField,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Tooltip,
  Collapse,
  IconButton,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import BiotechIcon from '@mui/icons-material/Biotech';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import CableIcon from '@mui/icons-material/Cable';
import { useSelector, useDispatch } from 'react-redux';
import { getConnectionSettingsState } from '../state/slices/ConnectionSettingsSlice';
import { getGalvoScannerState } from '../state/slices/GalvoScannerSlice';
import {
  getFlimLabsState,
  setFlimHost,
  setFlimPort,
  setFlimConnected,
  setFlimParam,
  toggleFlimChannel,
  setFlimRunning,
  setFlimProgress,
  setFlimDataFile,
  setFlimError,
  clearFlimError,
} from '../state/slices/FlimLabsSlice';
import {
  apiFlimCheckCard,
  apiFlimResolveFirmware,
  apiFlimStart,
  apiFlimStop,
  buildFlimImagingPayload,
  flimWsUrl,
} from '../backendapi/apiFlimLabs';
import { apiStartGalvoScan, apiStopGalvoScan } from '../backendapi/apiGalvoScannerController';
import { parseFlimChunk, FLIM_MSG } from '../utils/flimBinaryParser';

// Simple "hot"-style colormap: black -> red -> yellow -> white
const applyHotColormap = (t) => {
  const r = Math.min(255, Math.round(t * 3 * 255));
  const g = Math.min(255, Math.max(0, Math.round((t * 3 - 1) * 255)));
  const b = Math.min(255, Math.max(0, Math.round((t * 3 - 2) * 255)));
  return [r, g, b];
};

const FlimLabsPanel = () => {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const galvoState = useSelector(getGalvoScannerState);
  const flim = useSelector(getFlimLabsState);

  const [expanded, setExpanded] = useState(true);
  const [busy, setBusy] = useState(false);

  const galvoConfig = galvoState.config;
  const selectedScanner = galvoState.selectedScanner;

  // Effective acquisition geometry (synced with galvo or manual)
  const imageWidth = flim.syncWithGalvo ? galvoConfig.nx : flim.manualImageWidth;
  const imageHeight = flim.syncWithGalvo ? galvoConfig.ny : flim.manualImageHeight;
  const dwellTime = flim.syncWithGalvo
    ? Math.max(1, galvoConfig.sample_period_us)
    : flim.manualDwellTime;

  // Live image buffers (refs: high-rate updates must not go through Redux)
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const imageBufRef = useRef(null); // Uint32Array width*height, summed over channels
  const imageDimsRef = useRef({ w: 0, h: 0 });
  const maxValRef = useRef(1);
  const drawPendingRef = useRef(false);

  const resetImageBuffer = useCallback((w, h) => {
    imageBufRef.current = new Uint32Array(w * h);
    imageDimsRef.current = { w, h };
    maxValRef.current = 1;
  }, []);

  const drawImage = useCallback(() => {
    drawPendingRef.current = false;
    const canvas = canvasRef.current;
    const buf = imageBufRef.current;
    if (!canvas || !buf) return;
    const { w, h } = imageDimsRef.current;
    if (w === 0 || h === 0) return;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);
    const maxVal = Math.max(1, maxValRef.current);
    for (let i = 0; i < w * h; i++) {
      const t = buf[i] / maxVal;
      const [r, g, b] = applyHotColormap(t);
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }
    // Draw at native resolution; CSS scales the canvas element
    canvas.width = w;
    canvas.height = h;
    ctx.putImageData(imgData, 0, 0);
  }, []);

  const scheduleDraw = useCallback(() => {
    if (!drawPendingRef.current) {
      drawPendingRef.current = true;
      requestAnimationFrame(drawImage);
    }
  }, [drawImage]);

  const handleWsMessage = useCallback(
    (event) => {
      const messages = parseFlimChunk(event.data);
      const buf = imageBufRef.current;
      const { w, h } = imageDimsRef.current;
      let touched = false;
      for (const msg of messages) {
        switch (msg.type) {
          case FLIM_MSG.LINE: {
            if (!buf || msg.line >= h) break;
            const row = msg.line * w;
            const n = Math.min(msg.pixels.length, w);
            for (let x = 0; x < n; x++) {
              // Cumulative intensity across frames and enabled channels
              const v = buf[row + x] + msg.pixels[x];
              buf[row + x] = v;
              if (v > maxValRef.current) maxValRef.current = v;
            }
            touched = true;
            if (msg.frame !== undefined) {
              dispatch(setFlimProgress({ currentFrame: msg.frame }));
            }
            break;
          }
          case FLIM_MSG.CPS: {
            dispatch(setFlimProgress({ cps: msg.cps }));
            break;
          }
          case FLIM_MSG.IMAGING_END: {
            // v2 imaging runs terminate with IMAGING_END (END_EXPERIMENT is
            // only emitted by non-imaging experiments).
            if (msg.dataFile) dispatch(setFlimDataFile(msg.dataFile));
            dispatch(setFlimRunning({ running: false }));
            break;
          }
          case FLIM_MSG.END_EXPERIMENT: {
            dispatch(setFlimRunning({ running: false }));
            break;
          }
          default:
            break;
        }
      }
      if (touched) scheduleDraw();
    },
    [dispatch, scheduleDraw]
  );

  const closeWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const openWs = useCallback(() => {
    closeWs();
    const ws = new WebSocket(flimWsUrl(flim.host, flim.port));
    ws.binaryType = 'arraybuffer';
    ws.onmessage = handleWsMessage;
    ws.onerror = () => dispatch(setFlimError('WebSocket error (is the FLIM server reachable?)'));
    ws.onclose = () => {
      // Server closes /data when the experiment ends
      wsRef.current = null;
    };
    wsRef.current = ws;
  }, [closeWs, flim.host, flim.port, handleWsMessage, dispatch]);

  // Close socket on unmount
  useEffect(() => closeWs, [closeWs]);

  const handleCheckCard = async () => {
    setBusy(true);
    dispatch(clearFlimError());
    try {
      const res = await apiFlimCheckCard(flim.host, flim.port);
      dispatch(setFlimConnected({ connected: true, cardSerial: res.data }));
    } catch (e) {
      dispatch(setFlimConnected({ connected: false }));
      dispatch(setFlimError(`Card check failed: ${e.message}`));
    } finally {
      setBusy(false);
    }
  };

  const handleStart = async (step) => {
    setBusy(true);
    dispatch(clearFlimError());
    try {
      // 1. Resolve firmware for current laser/channel settings
      const enabledIdx = flim.channels
        .map((on, i) => (on ? i + 1 : -1))
        .filter((i) => i > 0);
      const fwRes = await apiFlimResolveFirmware(flim.host, flim.port, {
        sync: flim.laserSync,
        frequencyMhz: flim.frequencyMhz,
        channels: enabledIdx,
        channel: 'sma',
        reconstruction: flim.reconstruction,
        enable100ps: flim.enable100ps,
      });
      const firmware = fwRes.data;

      // 2. Open data stream BEFORE starting so no lines are missed
      resetImageBuffer(imageWidth, imageHeight);
      openWs();

      // 3. Arm the FLIM acquisition.
      // In LF/F reconstruction the card slices lines by dwell time starting at
      // the line marker, which the firmware fires at the START of each line
      // (sample 0). One galvo sample = one FLIM pixel (dwell is synced), so:
      //   offset_left  = pre + overscan            (marker -> first pixel)
      //   offset_right = overscan + fly + settle    (last pixel -> line end;
      //                  no flyback in bidirectional mode)
      // With per-pixel markers (PLF) the pixel clock defines the geometry and
      // no overscan modelling is needed.
      const modelOverscan = flim.syncWithGalvo && flim.reconstruction !== 'PLF';
      const ov = galvoConfig.overscan_samples || 0;
      const offsetLeft = modelOverscan ? (galvoConfig.pre_samples || 0) + ov : 0;
      const offsetRight = modelOverscan
        ? ov +
          (galvoConfig.bidirectional ? 0 : galvoConfig.fly_samples || 0) +
          (galvoConfig.line_settle_samples || 0)
        : 0;
      const payload = buildFlimImagingPayload({
        firmware,
        step,
        frequencyMhz: flim.frequencyMhz,
        enable100ps: flim.enable100ps,
        reconstruction: flim.reconstruction,
        imageWidth,
        imageHeight,
        offsets: { top: 0, right: offsetRight, bottom: 0, left: offsetLeft },
        channels: flim.channels,
        dwellTime,
        maxFrames: flim.maxFrames,
      });
      await apiFlimStart(flim.host, flim.port, payload);
      dispatch(setFlimRunning({ running: true, step, firmware }));

      // 4. Optionally start the galvo scan (scanner is the master clock).
      //    Small settle delay so the card is armed before triggers arrive.
      if (flim.autoStartGalvo && selectedScanner) {
        await new Promise((r) => setTimeout(r, 500));
        const galvoScanConfig = {
          ...galvoConfig,
          frame_count: flim.maxFrames > 0 ? flim.maxFrames : 0,
          enable_trigger: 1,
        };
        await apiStartGalvoScan(
          connectionSettings.ip,
          connectionSettings.apiPort,
          selectedScanner,
          galvoScanConfig
        );
      }
    } catch (e) {
      dispatch(setFlimError(`Start failed: ${e.message}`));
      closeWs();
      dispatch(setFlimRunning({ running: false }));
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      // Stop the trigger source first, then the acquisition
      if (flim.autoStartGalvo && selectedScanner) {
        try {
          await apiStopGalvoScan(connectionSettings.ip, connectionSettings.apiPort, selectedScanner);
        } catch (e) {
          console.warn('Galvo stop failed:', e);
        }
      }
      await apiFlimStop(flim.host, flim.port);
    } catch (e) {
      dispatch(setFlimError(`Stop failed: ${e.message}`));
    } finally {
      closeWs();
      dispatch(setFlimRunning({ running: false }));
      setBusy(false);
    }
  };

  const lineRateHz =
    dwellTime > 0 ? 1e6 / (dwellTime * (galvoConfig.pre_samples + galvoConfig.nx + galvoConfig.fly_samples + galvoConfig.line_settle_samples || 1)) : 0;

  return (
    <Paper sx={{ p: 2, mt: 2, border: '1px dashed', borderColor: 'warning.main' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <BiotechIcon color="warning" />
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          FLIM LABS Bridge
          <Chip label="dev" size="small" color="warning" sx={{ ml: 1 }} />
        </Typography>
        {flim.connected && (
          <Chip icon={<CableIcon />} label={`Card ${flim.cardSerial}`} size="small" color="success" />
        )}
        {flim.running && <Chip label={`${flim.step} running`} size="small" color="info" />}
        <IconButton size="small" onClick={() => setExpanded(!expanded)}>
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>

      <Collapse in={expanded}>
        {flim.error && (
          <Alert severity="error" onClose={() => dispatch(clearFlimError())} sx={{ mt: 1 }}>
            {flim.error}
          </Alert>
        )}

        <Grid container spacing={2} sx={{ mt: 0 }}>
          {/* Left column: connection + parameters */}
          <Grid item xs={12} md={5}>
            {/* Remote server connection */}
            <Typography variant="subtitle2" gutterBottom>
              Remote FLIM Server (Docker)
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
              <TextField
                label="Host / IP"
                size="small"
                fullWidth
                value={flim.host}
                onChange={(e) => dispatch(setFlimHost(e.target.value))}
                placeholder="e.g. 192.168.2.100"
              />
              <TextField
                label="Port"
                size="small"
                type="number"
                sx={{ width: 100 }}
                value={flim.port}
                onChange={(e) => dispatch(setFlimPort(Number(e.target.value)))}
              />
              <Button variant="outlined" size="small" onClick={handleCheckCard} disabled={busy}>
                Check
              </Button>
            </Box>

            {/* Laser / reconstruction */}
            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
              <FormControl size="small" fullWidth>
                <InputLabel>Laser Freq (MHz)</InputLabel>
                <Select
                  value={flim.frequencyMhz}
                  label="Laser Freq (MHz)"
                  onChange={(e) => dispatch(setFlimParam({ param: 'frequencyMhz', value: e.target.value }))}
                >
                  {[20, 40, 80, 100].map((f) => (
                    <MenuItem key={f} value={f}>
                      {f}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl size="small" fullWidth>
                <InputLabel>Laser Sync</InputLabel>
                <Select
                  value={flim.laserSync}
                  label="Laser Sync"
                  onChange={(e) => dispatch(setFlimParam({ param: 'laserSync', value: e.target.value }))}
                >
                  <MenuItem value="in">Sync In</MenuItem>
                  <MenuItem value="out">Sync Out</MenuItem>
                </Select>
              </FormControl>
              <Tooltip title="Which scanner markers are wired to the card: Pixel+Line+Frame, Line+Frame, or Frame only">
                <FormControl size="small" fullWidth>
                  <InputLabel>Reconstruction</InputLabel>
                  <Select
                    value={flim.reconstruction}
                    label="Reconstruction"
                    onChange={(e) =>
                      dispatch(setFlimParam({ param: 'reconstruction', value: e.target.value }))
                    }
                  >
                    <MenuItem value="PLF">PLF (pixel+line+frame)</MenuItem>
                    <MenuItem value="LF">LF (line+frame)</MenuItem>
                    <MenuItem value="F">F (frame only)</MenuItem>
                  </Select>
                </FormControl>
              </Tooltip>
            </Box>

            {/* Channels */}
            <Typography variant="subtitle2">Channels</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', mb: 1 }}>
              {flim.channels.map((on, i) => (
                <FormControlLabel
                  key={i}
                  control={
                    <Checkbox
                      size="small"
                      checked={on}
                      onChange={() => dispatch(toggleFlimChannel(i))}
                    />
                  }
                  label={`${i + 1}`}
                  sx={{ mr: 0.5 }}
                />
              ))}
            </Box>

            {/* Sync with galvo */}
            <FormControlLabel
              control={
                <Checkbox
                  checked={flim.syncWithGalvo}
                  onChange={(e) => dispatch(setFlimParam({ param: 'syncWithGalvo', value: e.target.checked }))}
                />
              }
              label="Sync geometry with galvo scan (nx/ny → image size, dwell)"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={flim.autoStartGalvo}
                  onChange={(e) => dispatch(setFlimParam({ param: 'autoStartGalvo', value: e.target.checked }))}
                />
              }
              label="Auto start/stop galvo scan with acquisition"
            />

            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              <TextField
                label="Image W"
                size="small"
                type="number"
                value={imageWidth}
                disabled={flim.syncWithGalvo}
                onChange={(e) => dispatch(setFlimParam({ param: 'manualImageWidth', value: Number(e.target.value) }))}
              />
              <TextField
                label="Image H"
                size="small"
                type="number"
                value={imageHeight}
                disabled={flim.syncWithGalvo}
                onChange={(e) => dispatch(setFlimParam({ param: 'manualImageHeight', value: Number(e.target.value) }))}
              />
              <TextField
                label="Dwell (µs)"
                size="small"
                type="number"
                value={dwellTime}
                disabled={flim.syncWithGalvo}
                onChange={(e) => dispatch(setFlimParam({ param: 'manualDwellTime', value: Number(e.target.value) }))}
              />
              <TextField
                label="Max Frames (0=∞)"
                size="small"
                type="number"
                value={flim.maxFrames}
                onChange={(e) => dispatch(setFlimParam({ param: 'maxFrames', value: Number(e.target.value) }))}
              />
            </Box>

            {flim.syncWithGalvo && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                From galvo: {galvoConfig.nx}×{galvoConfig.ny} px, {galvoConfig.sample_period_us} µs/px,
                ≈{lineRateHz.toFixed(1)} lines/s
              </Typography>
            )}

            {/* Start / stop */}
            <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={() => handleStart('scouting')}
                disabled={busy || flim.running}
              >
                Scouting
              </Button>
              <Button
                variant="contained"
                color="success"
                startIcon={<PlayArrowIcon />}
                onClick={() => handleStart('imaging')}
                disabled={busy || flim.running}
              >
                Imaging
              </Button>
              <Button
                variant="outlined"
                color="error"
                startIcon={<StopIcon />}
                onClick={handleStop}
                disabled={busy || !flim.running}
              >
                Stop
              </Button>
            </Box>
          </Grid>

          {/* Right column: live image */}
          <Grid item xs={12} md={7}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
                Live Intensity ({imageWidth}×{imageHeight})
              </Typography>
              {flim.cps > 0 && (
                <Chip label={`${(flim.cps / 1000).toFixed(1)} kCPS`} size="small" />
              )}
              <Chip label={`Frame ${flim.currentFrame}`} size="small" variant="outlined" />
            </Box>
            <Box
              sx={{
                bgcolor: 'black',
                borderRadius: 1,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: 260,
              }}
            >
              <canvas
                ref={canvasRef}
                style={{
                  width: '100%',
                  maxWidth: 512,
                  imageRendering: 'pixelated',
                  aspectRatio: `${imageWidth} / ${imageHeight}`,
                }}
              />
            </Box>
            {flim.lastDataFile && (
              <Typography variant="caption" color="text.secondary">
                Saved: {flim.lastDataFile}
              </Typography>
            )}
          </Grid>
        </Grid>
      </Collapse>
    </Paper>
  );
};

export default FlimLabsPanel;
