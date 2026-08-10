/**
 * FlimLabsPanel.jsx - FLIM LABS control tab inside the galvo scanner UI.
 *
 * The ImSwitch backend owns the FLIM connection (FLIMLabsController +
 * FLIMLabsDetectorManager): the flim-imager server's /data WebSocket is a
 * single-consumer stream, so the browser must not open its own. This panel is
 * therefore a thin client - it polls the ImSwitch API for status, a rendered
 * intensity PNG and the accumulated phasor histogram. The upside is that the
 * FLIM tab keeps working while an ExperimentController acquisition uses the
 * same card as a detector, and closing the browser no longer kills the stream.
 *
 * Three modes mirroring the FLIM LABS app:
 *   - Scouting:    live intensity image
 *   - Calibration: solid-calibrator run (known tau) -> per-channel/harmonic
 *                  phase & modulation; the server writes a reference JSON
 *   - Phasors:     G/S cloud against the universal semicircle, corrected by
 *                  that calibration reference
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
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import BiotechIcon from '@mui/icons-material/Biotech';
import CableIcon from '@mui/icons-material/Cable';
import SaveIcon from '@mui/icons-material/Save';
import DownloadIcon from '@mui/icons-material/Download';
import SpeedIcon from '@mui/icons-material/Speed';
import TuneIcon from '@mui/icons-material/Tune';
import ScatterPlotIcon from '@mui/icons-material/ScatterPlot';
import GridOnIcon from '@mui/icons-material/GridOn';
import { useSelector, useDispatch } from 'react-redux';
import { getConnectionSettingsState } from '../state/slices/ConnectionSettingsSlice';
import {
  getFlimLabsState,
  setFlimParam,
  setFlimRunning,
  setFlimProgress,
  setFlimError,
  clearFlimError,
  setFlimHealth,
  setFlimConnected,
  setFlimCalibrationReference,
  hydrateFlimConfig,
} from '../state/slices/FlimLabsSlice';
import {
  apiFlimGetStatus,
  apiFlimGetImage,
  apiFlimGetPhasor,
  apiFlimStart,
  apiFlimStop,
  apiFlimReset,
  apiFlimDetectLaserFrequency,
  apiFlimSetParameter,
} from '../backendapi/apiFlimLabs';
import {
  apiGetFlimLabsConfig,
  apiSetFlimLabsConfig,
} from '../backendapi/apiGalvoScannerController';

// "hot"-style colormap for the phasor density (the intensity image is
// colormapped server-side and arrives as a ready PNG)
const applyHotColormap = (t) => {
  const r = Math.min(255, Math.round(t * 3 * 255));
  const g = Math.min(255, Math.max(0, Math.round((t * 3 - 1) * 255)));
  const b = Math.min(255, Math.max(0, Math.round((t * 3 - 2) * 255)));
  return [r, g, b];
};

const MODE_TABS = ['scouting', 'calibration', 'phasors'];
// A FLIM frame takes seconds — fast polling only adds server load and log
// noise. Status: slow heartbeat when idle, quicker while running. Images are
// only fetched while an acquisition runs (plus once on stop for the final).
const STATUS_POLL_IDLE_MS = 5000;
const STATUS_POLL_RUNNING_MS = 2000;
const IMAGE_POLL_MS = 2000;

const FlimLabsPanel = () => {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const flim = useSelector(getFlimLabsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  const [busy, setBusy] = useState(false);
  const [modeTab, setModeTab] = useState(0);
  const [configStatus, setConfigStatus] = useState('');
  const [status, setStatus] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);

  const mode = MODE_TABS[modeTab];
  const phasorCanvasRef = useRef(null);
  const phasorDataRef = useRef(null);

  // ------------------------------------------------------------------
  // Phasor rendering (sparse density from the backend + universal semicircle)
  // ------------------------------------------------------------------
  const drawPhasor = useCallback(() => {
    const canvas = phasorCanvasRef.current;
    if (!canvas) return;
    const data = phasorDataRef.current;
    const W = data?.width || 420;
    const H = data?.height || 280;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);

    if (data && data.points && data.points.length) {
      const imgData = ctx.getImageData(0, 0, W, H);
      const maxVal = Math.max(1, Math.log1p(data.max || 1));
      for (const [x, y, count] of data.points) {
        if (x < 0 || x >= W || y < 0 || y >= H) continue;
        const t = Math.log1p(count) / maxVal;
        const [r, g, b] = applyHotColormap(t);
        const i = (y * W + x) * 4;
        imgData.data[i] = r;
        imgData.data[i + 1] = g;
        imgData.data[i + 2] = b;
        imgData.data[i + 3] = 255;
      }
      ctx.putImageData(imgData, 0, 0);
    }

    const gMin = data?.gMin ?? -0.05;
    const gMax = data?.gMax ?? 1.05;
    const sMin = data?.sMin ?? -0.02;
    const sMax = data?.sMax ?? 0.65;
    const gx = (g) => ((g - gMin) / (gMax - gMin)) * W;
    const sy = (s) => H - ((s - sMin) / (sMax - sMin)) * H;

    // Universal semicircle: s = sqrt(0.25 - (g - 0.5)^2)
    ctx.strokeStyle = 'rgba(120, 220, 200, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const g = i / 200;
      const s = Math.sqrt(Math.max(0, 0.25 - (g - 0.5) ** 2));
      if (i === 0) ctx.moveTo(gx(g), sy(s));
      else ctx.lineTo(gx(g), sy(s));
    }
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(gx(0), sy(0));
    ctx.lineTo(gx(1), sy(0));
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '10px sans-serif';
    ctx.fillText('0', gx(0) - 3, sy(0) + 12);
    ctx.fillText('0.5', gx(0.5) - 7, sy(0) + 12);
    ctx.fillText('1', gx(1) - 3, sy(0) + 12);
    ctx.fillText('G', W - 14, sy(0) - 6);
    ctx.fillText('S', gx(0) + 6, 12);
  }, []);

  useEffect(() => {
    if (mode === 'phasors') drawPhasor();
  }, [mode, drawPhasor]);

  // ------------------------------------------------------------------
  // Polling: status + image (+ phasor while in phasor mode)
  // ------------------------------------------------------------------
  const pollStatus = useCallback(async () => {
    try {
      const st = await apiFlimGetStatus(hostIP, hostPort);
      setStatus(st);
      if (st.available === false) {
        dispatch(setFlimHealth(false));
        return;
      }
      dispatch(setFlimHealth(!!st.serverHealthy));
      dispatch(setFlimConnected({ connected: !!st.cardSerial, cardSerial: st.cardSerial }));
      dispatch(setFlimRunning({ running: !!st.running, step: st.step }));
      dispatch(setFlimProgress({ currentFrame: st.frameNumber, cps: st.cps }));
      if (st.calibrationReference) {
        dispatch(setFlimCalibrationReference({ referenceFile: st.calibrationReference }));
      }
    } catch (e) {
      dispatch(setFlimHealth(false));
    }
  }, [hostIP, hostPort, dispatch]);

  const pollImage = useCallback(async () => {
    try {
      const res = await apiFlimGetImage(hostIP, hostPort, 512);
      if (res && res.image) setImageUrl(res.image);
    } catch (e) {
      /* transient - the status poll surfaces connection problems */
    }
  }, [hostIP, hostPort]);

  const pollPhasor = useCallback(async () => {
    try {
      const res = await apiFlimGetPhasor(hostIP, hostPort);
      phasorDataRef.current = res;
      drawPhasor();
    } catch (e) {
      /* ignore */
    }
  }, [hostIP, hostPort, drawPhasor]);

  const running = !!status?.running;

  useEffect(() => {
    pollStatus();
    const id = setInterval(
      pollStatus, running ? STATUS_POLL_RUNNING_MS : STATUS_POLL_IDLE_MS
    );
    return () => clearInterval(id);
  }, [pollStatus, running]);

  // Image/phasor data: poll only while an acquisition is running; fetch once
  // on mount and once when the run ends so the final result stays visible.
  useEffect(() => {
    const fetchOnce = mode === 'phasors' ? pollPhasor : pollImage;
    fetchOnce();
    if (!running) return undefined;
    const id = setInterval(fetchOnce, IMAGE_POLL_MS);
    return () => clearInterval(id);
  }, [mode, running, pollImage, pollPhasor]);

  // ------------------------------------------------------------------
  // Config persistence (ImSwitch setup JSON)
  // ------------------------------------------------------------------
  useEffect(() => {
    (async () => {
      try {
        const cfg = await apiGetFlimLabsConfig(hostIP, hostPort);
        if (cfg && !cfg.error && Object.keys(cfg).length > 0) {
          dispatch(hydrateFlimConfig(cfg));
        }
      } catch (e) {
        console.warn('FLIM config load failed:', e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSaveConfig = async () => {
    setBusy(true);
    try {
      const cfg = {
        maxFrames: flim.maxFrames,
        calibrationTauNs: flim.calibrationTauNs,
        calibrationHarmonics: flim.calibrationHarmonics,
        calibrationReferenceFile: flim.calibrationReferenceFile,
        exportEnabled: flim.exportEnabled,
        exportFilename: flim.exportFilename,
      };
      const res = await apiSetFlimLabsConfig(hostIP, hostPort, cfg);
      setConfigStatus(res && !res.error
        ? 'Settings saved to ImSwitch setup file'
        : `Save failed: ${res?.error}`);
    } catch (e) {
      setConfigStatus(`Save failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  // ------------------------------------------------------------------
  // Detector parameters (owned by the backend detector)
  // ------------------------------------------------------------------
  const setDetectorParam = async (name, value) => {
    try {
      await apiFlimSetParameter(hostIP, hostPort, name, value);
      pollStatus();
    } catch (e) {
      dispatch(setFlimError(`Could not set ${name}: ${e.message}`));
    }
  };

  const handleDetectFrequency = async () => {
    setBusy(true);
    dispatch(clearFlimError());
    try {
      const res = await apiFlimDetectLaserFrequency(hostIP, hostPort);
      setConfigStatus(
        `Laser detected at ${Number(res.frequency).toFixed(3)} MHz → using ${res.nominal} MHz firmware`
      );
      pollStatus();
    } catch (e) {
      dispatch(setFlimError(`Frequency detection failed: ${e.message} (laser sync connected?)`));
    } finally {
      setBusy(false);
    }
  };

  // ------------------------------------------------------------------
  // Acquisition control
  // ------------------------------------------------------------------
  const handleStart = async () => {
    setBusy(true);
    dispatch(clearFlimError());
    setConfigStatus('');
    try {
      await apiFlimStart(hostIP, hostPort, {
        step: mode,
        maxFrames: flim.maxFrames,
        tauNs: mode === 'scouting' ? null : Number(flim.calibrationTauNs),
        harmonics: mode === 'scouting' ? 1 : Number(flim.calibrationHarmonics) || 1,
        exportData: flim.exportEnabled && mode !== 'calibration',
        exportFilename: flim.exportFilename,
      });
      pollStatus();
    } catch (e) {
      dispatch(setFlimError(`Start failed: ${e.message}`));
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      await apiFlimStop(hostIP, hostPort);
      pollStatus();
    } catch (e) {
      dispatch(setFlimError(`Stop failed: ${e.message}`));
    } finally {
      setBusy(false);
    }
  };

  const handleReset = async () => {
    setBusy(true);
    try {
      await apiFlimReset(hostIP, hostPort);
      phasorDataRef.current = null;
      drawPhasor();
      setImageUrl(null);
      pollStatus();
    } catch (e) {
      dispatch(setFlimError(`Reset failed: ${e.message}`));
    } finally {
      setBusy(false);
    }
  };

  const handleDownloadPng = () => {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const a = document.createElement('a');
    a.download = `flim_${mode}_${stamp}.png`;
    if (mode === 'phasors') {
      const canvas = phasorCanvasRef.current;
      if (!canvas) return;
      a.href = canvas.toDataURL('image/png');
    } else {
      if (!imageUrl) return;
      a.href = imageUrl;
    }
    a.click();
  };

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  const available = status?.available !== false;
  const params = status?.parameters || {};
  const calibrationResults = status?.calibrationResults || [];
  const calibrationReference = status?.calibrationReference || null;

  const healthChip =
    flim.serverHealthy === null ? (
      <Chip label="server: ?" size="small" />
    ) : flim.serverHealthy ? (
      <Chip label="server: up" size="small" color="success" variant="outlined" />
    ) : (
      <Chip label="server: down" size="small" color="error" />
    );

  if (!available) {
    return (
      <Paper sx={{ p: 2, mt: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <BiotechIcon color="warning" />
          <Typography variant="h6">FLIM LABS</Typography>
        </Box>
        <Alert severity="info">
          No FLIM detector configured. Add a <code>FLIMLabsDetectorManager</code> entry
          under <code>detectors</code> in the setup file (and <code>"FLIMLabs"</code> to
          <code> availableWidgets</code>), then restart ImSwitch.
          {status?.error ? ` — ${status.error}` : ''}
        </Alert>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
        <BiotechIcon color="warning" />
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          FLIM LABS
        </Typography>
        {healthChip}
        {status?.cardSerial ? (
          <Chip icon={<CableIcon />} label={`Card ${status.cardSerial}`} size="small" color="success" />
        ) : (
          <Chip icon={<CableIcon />} label="no card" size="small" color="warning" variant="outlined" />
        )}
        {running && <Chip label={`${status.step} running`} size="small" color="info" />}
        <Tooltip title="Save FLIM settings into the ImSwitch setup file">
          <Button size="small" startIcon={<SaveIcon />} onClick={handleSaveConfig} disabled={busy}>
            Save
          </Button>
        </Tooltip>
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        Acquisition is owned by the ImSwitch backend (detector{' '}
        <b>{status?.detectorName}</b> → {status?.serverUrl}), so it keeps running
        independently of this browser tab.
      </Typography>

      {flim.error && (
        <Alert severity="error" onClose={() => dispatch(clearFlimError())} sx={{ mt: 1 }}>
          {flim.error}
        </Alert>
      )}
      {configStatus && (
        <Alert severity="info" onClose={() => setConfigStatus('')} sx={{ mt: 1 }}>
          {configStatus}
        </Alert>
      )}
      {status?.hint && (
        <Alert severity="warning" sx={{ mt: 1 }}>
          {status.hint}
        </Alert>
      )}

      {/* Mode tabs */}
      <Tabs value={modeTab} onChange={(e, v) => setModeTab(v)} sx={{ mb: 1 }}>
        <Tab icon={<GridOnIcon />} iconPosition="start" label="Scouting" />
        <Tab icon={<TuneIcon />} iconPosition="start" label="Calibration" />
        <Tab icon={<ScatterPlotIcon />} iconPosition="start" label="Phasors" />
      </Tabs>

      <Grid container spacing={2}>
        {/* Left: parameters + controls */}
        <Grid item xs={12} md={5}>
          <Typography variant="subtitle2" gutterBottom>
            Acquisition (detector parameters)
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
            <FormControl size="small" fullWidth>
              <InputLabel>Laser Freq (MHz)</InputLabel>
              <Select
                value={params.frequency_mhz ?? 40}
                label="Laser Freq (MHz)"
                disabled={running}
                onChange={(e) => setDetectorParam('frequency_mhz', e.target.value)}
              >
                {[20, 40, 80, 100].map((f) => (
                  <MenuItem key={f} value={f}>{f}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <Tooltip title="Measure the laser sync frequency with the card's frequency meter">
              <span>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<SpeedIcon />}
                  onClick={handleDetectFrequency}
                  disabled={busy || running}
                  sx={{ whiteSpace: 'nowrap', minWidth: 110 }}
                >
                  Detect
                </Button>
              </span>
            </Tooltip>
          </Box>

          <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
            <Tooltip title="Which scanner markers are wired to the card: Pixel+Line+Frame, Line+Frame, or Frame only">
              <FormControl size="small" fullWidth>
                <InputLabel>Reconstruction</InputLabel>
                <Select
                  value={params.reconstruction ?? 'PLF'}
                  label="Reconstruction"
                  disabled={running}
                  onChange={(e) => setDetectorParam('reconstruction', e.target.value)}
                >
                  <MenuItem value="PLF">PLF (pixel+line+frame)</MenuItem>
                  <MenuItem value="LF">LF (line+frame)</MenuItem>
                  <MenuItem value="F">F (frame only)</MenuItem>
                </Select>
              </FormControl>
            </Tooltip>
            <TextField
              label="Dwell (µs)"
              size="small"
              type="number"
              value={params.dwell_time ?? 25}
              disabled={running}
              onChange={(e) => setDetectorParam('dwell_time', Number(e.target.value))}
            />
          </Box>

          <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
            <TextField
              label="Max Frames (0=∞)"
              size="small"
              type="number"
              value={flim.maxFrames}
              onChange={(e) => dispatch(setFlimParam({ param: 'maxFrames', value: Number(e.target.value) }))}
            />
            <Tooltip title="Frames summed per snapSync() grab when the experiment engine uses FLIM as a detector">
              <TextField
                label="Frames / grab"
                size="small"
                type="number"
                value={params.frames_to_integrate ?? 1}
                disabled={running}
                onChange={(e) => setDetectorParam('frames_to_integrate', Number(e.target.value))}
              />
            </Tooltip>
          </Box>

          {/* Scan-region crop: image = scan (galvo nx/ny) minus offsets */}
          <Typography variant="subtitle2" sx={{ mt: 1 }}>Scan Area (crop)</Typography>
          <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
            {[['offset_top', 'Top'], ['offset_right', 'Right'],
              ['offset_bottom', 'Bottom'], ['offset_left', 'Left']].map(([key, label]) => (
              <TextField
                key={key}
                label={label}
                size="small"
                type="number"
                inputProps={{ min: 0 }}
                value={params[key] ?? 0}
                disabled={running}
                onChange={(e) => setDetectorParam(key, Number(e.target.value))}
                sx={{ width: 90 }}
              />
            ))}
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Scan {status?.scanWidth}×{status?.scanHeight} px (galvo nx/ny) → image{' '}
            {status?.imageWidth}×{status?.imageHeight} px after crop. Use Left to drop the
            first column(s) where flyback/settle photons pile up. Galvo:{' '}
            {status?.galvoScanner || 'not bound'}.
          </Typography>
          {status?.debug && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              Debug: {status.debug.linesInLastFrame ?? 0}/{status?.imageHeight} lines/frame,{' '}
              {status.debug.lateLines ?? 0} late, {status.debug.frameResets ?? 0} resets,{' '}
              {status.debug.unknownTagDrops ?? 0} unknown-tag drops
            </Typography>
          )}

          {/* Calibration-specific parameters */}
          {mode !== 'scouting' && (
            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              <Tooltip title="Known fluorescence lifetime of the solid calibrator (see its datasheet)">
                <TextField
                  label="Calibrator τ (ns)"
                  size="small"
                  type="number"
                  inputProps={{ step: 0.1, min: 0 }}
                  value={flim.calibrationTauNs}
                  disabled={mode === 'phasors'}
                  onChange={(e) => dispatch(setFlimParam({ param: 'calibrationTauNs', value: Number(e.target.value) }))}
                />
              </Tooltip>
              <TextField
                label="Harmonics"
                size="small"
                type="number"
                inputProps={{ min: 1, max: 4 }}
                value={flim.calibrationHarmonics}
                disabled={mode === 'phasors'}
                onChange={(e) => dispatch(setFlimParam({ param: 'calibrationHarmonics', value: Number(e.target.value) }))}
              />
            </Box>
          )}

          {/* Export */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={flim.exportEnabled}
                  onChange={(e) => dispatch(setFlimParam({ param: 'exportEnabled', value: e.target.checked }))}
                />
              }
              label="Export on server"
            />
            <TextField
              label="Filename"
              size="small"
              value={flim.exportFilename}
              disabled={!flim.exportEnabled}
              onChange={(e) => dispatch(setFlimParam({ param: 'exportFilename', value: e.target.value }))}
            />
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Exports land in the FLIM container volume ~/.flim-labs (docker volume flim-home).
          </Typography>

          {/* Start / stop / reset */}
          <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
            <Button
              variant="contained"
              color={mode === 'calibration' ? 'warning' : 'primary'}
              startIcon={<PlayArrowIcon />}
              onClick={handleStart}
              disabled={busy || running || (mode === 'phasors' && !calibrationReference)}
            >
              Start {mode}
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={handleStop}
              disabled={busy || !running}
            >
              Stop
            </Button>
            <Button variant="outlined" startIcon={<RestartAltIcon />} onClick={handleReset} disabled={busy}>
              Reset
            </Button>
            <Button variant="outlined" startIcon={<DownloadIcon />} onClick={handleDownloadPng}>
              PNG
            </Button>
          </Box>
        </Grid>

        {/* Right: visualization */}
        <Grid item xs={12} md={7}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
              {mode === 'phasors'
                ? `Phasor Plot (harmonic ${flim.calibrationHarmonics})`
                : `Live Intensity (${status?.imageWidth}×${status?.imageHeight})`}
            </Typography>
            {status?.cps > 0 && <Chip label={`${(status.cps / 1000).toFixed(1)} kCPS`} size="small" />}
            <Chip label={`Frame ${status?.frameNumber ?? 0}`} size="small" variant="outlined" />
          </Box>

          {/* Intensity (PNG rendered by the backend) */}
          <Box
            sx={{
              bgcolor: 'black',
              borderRadius: 1,
              display: mode === 'phasors' ? 'none' : 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              minHeight: 260,
            }}
          >
            {imageUrl ? (
              <img
                src={imageUrl}
                alt="FLIM intensity"
                style={{ width: '100%', maxWidth: 512, imageRendering: 'pixelated' }}
              />
            ) : (
              <Typography variant="caption" color="text.secondary" sx={{ p: 4 }}>
                No frames yet — start an acquisition (and make sure the galvo is scanning).
              </Typography>
            )}
          </Box>

          {/* Phasor */}
          <Box
            sx={{
              bgcolor: 'black',
              borderRadius: 1,
              display: mode === 'phasors' ? 'flex' : 'none',
              justifyContent: 'center',
              alignItems: 'center',
              minHeight: 260,
            }}
          >
            <canvas ref={phasorCanvasRef} style={{ width: '100%', maxWidth: 560 }} />
          </Box>

          {/* Calibration results */}
          {mode === 'calibration' && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2">Calibration results</Typography>
              {calibrationResults.length === 0 ? (
                <Typography variant="caption" color="text.secondary">
                  Point the scanner at the solid calibrator (τ = {flim.calibrationTauNs} ns), then
                  Start calibration. Phase/modulation per channel appear here.
                </Typography>
              ) : (
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Channel</TableCell>
                      <TableCell>Harmonic</TableCell>
                      <TableCell>Phase (°)</TableCell>
                      <TableCell>Modulation</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {calibrationResults.map((r) => (
                      <TableRow key={`${r.channel}-${r.harmonic}`}>
                        <TableCell>{r.channel + 1}</TableCell>
                        <TableCell>{r.harmonic}</TableCell>
                        <TableCell>{((r.phase * 180) / Math.PI).toFixed(2)}</TableCell>
                        <TableCell>{r.modulation.toFixed(4)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
              {calibrationReference && (
                <Alert severity="success" sx={{ mt: 1 }}>
                  Calibration reference (server): {calibrationReference}
                </Alert>
              )}
            </Box>
          )}

          {mode === 'phasors' && !calibrationReference && (
            <Alert severity="warning" sx={{ mt: 1 }}>
              No calibration reference — run a Calibration first (solid calibrator). The phasor
              run needs it to correct phase/modulation.
            </Alert>
          )}
          {mode === 'phasors' && calibrationReference && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
              Reference: {calibrationReference}
            </Typography>
          )}

          {status?.lastDataFile && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
              Saved: {status.lastDataFile}
            </Typography>
          )}
        </Grid>
      </Grid>
    </Paper>
  );
};

export default FlimLabsPanel;
