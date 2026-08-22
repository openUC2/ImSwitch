import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  TextField,
  Slider,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  IconButton,
  Chip,
  Tab,
  Tabs,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';
import SettingsIcon from '@mui/icons-material/Settings';
import SendIcon from '@mui/icons-material/Send';
import GridOnIcon from '@mui/icons-material/GridOn';
import ScatterPlotIcon from '@mui/icons-material/ScatterPlot';
import BiotechIcon from '@mui/icons-material/Biotech';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';
import { useSelector, useDispatch } from 'react-redux';
import { getConnectionSettingsState } from '../state/slices/ConnectionSettingsSlice';
import {
  getGalvoScannerState,
  getGalvoConfig,
  getGalvoStatus,
  getScanInfo,
  getActiveTab,
  setScannerNames,
  setSelectedScanner,
  setConfig,
  setConfigParam,
  setXRange,
  setYRange,
  toggleBidirectional,
  setStatus,
  setRunning,
  setError,
  clearError,
  setStatusMessage,
  clearStatusMessage,
  setAutoRefresh,
  applyPreset,
  setActiveTab,
} from '../state/slices/GalvoScannerSlice';
import {
  apiGetGalvoScannerNames,
  apiGetGalvoScannerConfig,
  apiGetGalvoScannerStatus,
  apiStartGalvoScan,
  apiStopGalvoScan,
  apiGetGalvoParkConfig,
  apiSetGalvoParkConfig,
  apiParkGalvo,
} from '../backendapi/apiGalvoScannerController';
import GalvoArbitraryPointsTab from './GalvoArbitraryPointsTab';
import FlimLabsPanel from './FlimLabsPanel';
import { apiFlimGetStatus } from '../backendapi/apiFlimLabs';

/**
 * Rich, human-readable explanations for each scan parameter, shown as hover
 * tooltips next to the corresponding field.
 */
const PARAM_TOOLTIPS = {
  nx: 'Number of samples (pixels) acquired per horizontal line. Higher = more X resolution but slower lines.',
  ny: 'Number of lines per frame (Y resolution). Higher = more Y resolution but slower frames.',
  x_min: 'Left edge of the scan in DAC counts (0–4095). Maps to the galvo X mirror voltage.',
  x_max: 'Right edge of the scan in DAC counts (0–4095).',
  y_min: 'Top edge of the scan in DAC counts (0–4095).',
  y_max: 'Bottom edge of the scan in DAC counts (0–4095).',
  sample_period_us:
    'Dwell time per sample in microseconds. 0 = go as fast as the DAC/loop allows. Larger = slower scan, brighter/less noisy pixels.',
  frame_count: 'Number of frames to acquire, then stop. 0 = scan continuously until you press Stop.',
  bidirectional:
    'Scan both sweep directions: even lines left→right, odd lines right→left. Roughly doubles frame rate but needs correct phase/settle to avoid a zig-zag offset.',
  pre_samples:
    'Blanking samples emitted at the start of each line before imaging begins — lets the mirror reach constant velocity. Increase if the left edge is smeared.',
  fly_samples:
    'Fly-back samples between lines (cosine-eased) while the mirror returns for the next line. Increase if lines tear or overshoot.',
  line_settle_samples:
    'Extra settle samples held after the fly-back before the next line starts. Increase if the start of each line is distorted.',
  trig_delay_us:
    'Gap between the frame marker and the line marker at the start of frame (µs). Both markers fire during pre-blanking, before the first pixel.',
  trig_width_us:
    'Trigger pulse width in µs for the frame/line markers and the pixel clock (pixel width is capped at half the dwell time; 0 = fastest possible pulse).',
  enable_trigger: 'Emit the pixel/line trigger output during the scan. 0 = off, 1 = on.',
  apply_x_lut:
    'Apply a per-column X lookup table to linearize the mirror. 0 = off, 1 = on (requires an uploaded LUT).',
  park_x: 'X position (DAC counts, 0–4095) the beam moves to when a scan stops.',
  park_y: 'Y position (DAC counts, 0–4095) the beam moves to when a scan stops.',
  overscan_samples:
    'Extends the X ramp at the same per-pixel slope on both sides of the imaging window, so the mirror is already at constant velocity when triggers/laser start. Compensates the galvo lagging behind the DAC. Costs 2×overscan samples per line.',
  laser_blanking:
    'Gate the laser (galvo laser pin) HIGH only during the imaging window — off during pre-blanking, overscan, fly-back and settle. Prevents fly-back photons from smearing the image (e.g. into a FLIM acquisition).',
  hw_pixel_clock:
    'Generate the pixel clock with the ESP32-S3 RMT peripheral: exactly nx hardware-timed, equidistant pulses per line, decoupled from the DAC/SPI software loop. Falls back to software pulses on unsupported chips.',
};

/**
 * An info icon with a hover tooltip, intended as a TextField endAdornment.
 */
const InfoTip = ({ text }) => (
  <Tooltip title={text} placement="top" arrow enterTouchDelay={0}>
    <InfoOutlinedIcon
      fontSize="small"
      sx={{ color: 'text.disabled', cursor: 'help', ml: 0.5 }}
    />
  </Tooltip>
);

/**
 * GalvoScannerController - Control panel for galvo mirror scanners
 * 
 * Features:
 * - Configure scan parameters (nx, ny, x/y ranges, timing)
 * - Start/stop galvo scans
 * - Real-time status polling
 * - Visual preview of scan pattern on full 4096x4096 canvas
 * - Multiple scanner device support
 * - Redux state management
 */
const GalvoScannerController = () => {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  // The FLIM tab only exists when the backend has a FLIMLabsController with a
  // configured FLIMLabsDetectorManager (probed once on mount).
  const [flimAvailable, setFlimAvailable] = useState(false);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const st = await apiFlimGetStatus(hostIP, hostPort);
        if (!cancelled) setFlimAvailable(st?.available !== false);
      } catch (e) {
        if (!cancelled) setFlimAvailable(false);
      }
    })();
    return () => { cancelled = true; };
  }, [hostIP, hostPort]);

  // Redux state
  const galvoState = useSelector(getGalvoScannerState);
  const config = useSelector(getGalvoConfig);
  const status = useSelector(getGalvoStatus);
  const scanInfo = useSelector(getScanInfo);
  const activeTab = useSelector(getActiveTab);

  // If the stored tab points at the (absent) FLIM tab, fall back to Raster
  useEffect(() => {
    if (!flimAvailable && activeTab === 2) dispatch(setActiveTab(0));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flimAvailable, activeTab]);
  
  // Destructure with defaults for safety
  const scannerNames = galvoState?.scannerNames || [];
  const selectedScanner = galvoState?.selectedScanner || '';
  const error = galvoState?.error || null;
  const statusMessage = galvoState?.statusMessage || '';
  const autoRefresh = galvoState?.autoRefresh || false;

  // Parking config (local component state; persisted server-side via the manager)
  const [parkConfig, setParkConfigState] = useState({
    park_x: 2048,
    park_y: 2048,
    park_on_stop: true,
  });

  // ========================
  // API Functions
  // ========================

  const fetchScannerNames = useCallback(async () => {
    try {
      const data = await apiGetGalvoScannerNames(hostIP, hostPort);
      if (Array.isArray(data)) {
        dispatch(setScannerNames(data));
      }
    } catch (err) {
      console.error('Failed to fetch scanner names:', err);
    }
  }, [hostIP, hostPort, dispatch]);

  const fetchConfig = useCallback(async () => {
    if (!selectedScanner) return;
    try {
      const data = await apiGetGalvoScannerConfig(hostIP, hostPort, selectedScanner);
      if (data.config) {
        dispatch(setConfig(data.config));
      }
    } catch (err) {
      console.error('Failed to fetch config:', err);
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  const fetchParkConfig = useCallback(async () => {
    if (!selectedScanner) return;
    try {
      const data = await apiGetGalvoParkConfig(hostIP, hostPort, selectedScanner);
      if (!data.error) {
        setParkConfigState({
          park_x: data.park_x ?? 2048,
          park_y: data.park_y ?? 2048,
          park_on_stop: data.park_on_stop ?? true,
        });
      }
    } catch (err) {
      console.error('Failed to fetch park config:', err);
    }
  }, [hostIP, hostPort, selectedScanner]);

  const saveParkConfig = useCallback(async (partial) => {
    const next = { ...parkConfig, ...partial };
    setParkConfigState(next);
    if (!selectedScanner) return;
    try {
      await apiSetGalvoParkConfig(hostIP, hostPort, selectedScanner, partial);
    } catch (err) {
      console.error('Failed to save park config:', err);
    }
  }, [hostIP, hostPort, selectedScanner, parkConfig]);

  const parkNow = useCallback(async () => {
    if (!selectedScanner) return;
    try {
      await apiParkGalvo(hostIP, hostPort, selectedScanner);
      dispatch(setStatusMessage('Beam parked'));
      setTimeout(() => dispatch(clearStatusMessage()), 2000);
    } catch (err) {
      dispatch(setError(`Failed to park: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  const fetchStatus = useCallback(async () => {
    if (!selectedScanner) return;
    try {
      const data = await apiGetGalvoScannerStatus(hostIP, hostPort, selectedScanner);
      if (!data.error) {
        dispatch(setStatus({
          running: data.running || false,
          current_frame: data.current_frame || 0,
          current_line: data.current_line || 0
        }));
      }
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  const startScan = useCallback(async () => {
    if (!selectedScanner) return;
    dispatch(setRunning(true)); // Optimistic update - button stays enabled
    dispatch(clearError());
    
    try {
      const data = await apiStartGalvoScan(hostIP, hostPort, selectedScanner, config);
      if (data.error) {
        dispatch(setError(data.error));
        dispatch(setRunning(false));
      } else {
        dispatch(setStatusMessage('Scan started'));
        setTimeout(() => dispatch(clearStatusMessage()), 2000);
      }
    } catch (err) {
      dispatch(setError(`Failed to start scan: ${err.message}`));
      dispatch(setRunning(false));
    }
  }, [hostIP, hostPort, selectedScanner, config, dispatch]);

  const stopScan = useCallback(async () => {
    if (!selectedScanner) return;
    dispatch(setRunning(false)); // Optimistic update - button stays enabled
    
    try {
      const data = await apiStopGalvoScan(hostIP, hostPort, selectedScanner);
      if (data.error) {
        dispatch(setError(data.error));
      } else {
        dispatch(setStatusMessage('Scan stopped'));
        setTimeout(() => dispatch(clearStatusMessage()), 2000);
      }
    } catch (err) {
      dispatch(setError(`Failed to stop scan: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  // Apply config and immediately start scan
  const applyConfigAndStartScan = useCallback(async () => {
    if (!selectedScanner) return;
    dispatch(setRunning(true)); // Optimistic update
    dispatch(clearError());
    dispatch(setStatusMessage('Applying configuration and starting scan...'));
    
    try {
      const data = await apiStartGalvoScan(hostIP, hostPort, selectedScanner, config);
      if (data.error) {
        dispatch(setError(data.error));
        dispatch(setRunning(false));
      } else {
        dispatch(setStatusMessage('Configuration applied, scan started'));
        setTimeout(() => dispatch(clearStatusMessage()), 2000);
      }
    } catch (err) {
      dispatch(setError(`Failed to apply config and start: ${err.message}`));
      dispatch(setRunning(false));
    }
  }, [hostIP, hostPort, selectedScanner, config, dispatch]);

  // ========================
  // Effects
  // ========================

  useEffect(() => {
    fetchScannerNames();
  }, [fetchScannerNames]);

  useEffect(() => {
    if (selectedScanner) {
      fetchConfig();
      fetchStatus();
      fetchParkConfig();
    }
  }, [selectedScanner, fetchConfig, fetchStatus, fetchParkConfig]);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchStatus, 500);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, fetchStatus]);

  // ========================
  // Handlers
  // ========================

  const handleConfigChange = (field) => (event) => {
    const value = event.target.type === 'checkbox' 
      ? event.target.checked 
      : Number(event.target.value);
    dispatch(setConfigParam({ param: field, value }));
  };

  const handleXRangeChange = (event, newValue) => {
    dispatch(setXRange(newValue));
  };

  const handleYRangeChange = (event, newValue) => {
    dispatch(setYRange(newValue));
  };

  const handlePresetClick = (preset) => () => {
    dispatch(applyPreset(preset));
  };

  // ========================
  // Enhanced Scan Pattern Visualization
  // Shows full 4096x4096 DAC range with scan area highlighted
  // ========================

  const ScanPatternPreview = useMemo(() => {
    const canvasSize = 280;
    const dacMax = 4096;
    const padding = 25;
    const innerSize = canvasSize - 2 * padding;
    
    // Map DAC values (0-4095) to canvas coordinates
    const mapToCanvas = (dacVal) => padding + (dacVal / dacMax) * innerSize;
    
    // Scan area bounds on canvas
    const scanLeft = mapToCanvas(config.x_min);
    const scanRight = mapToCanvas(config.x_max);
    const scanTop = mapToCanvas(config.y_min);
    const scanBottom = mapToCanvas(config.y_max);
    const scanWidth = scanRight - scanLeft;
    const scanHeight = scanBottom - scanTop;

    // Generate scan points for visualization (first 64 points max)
    const maxPreviewPoints = 64;
    const previewNx = Math.min(config.nx, maxPreviewPoints);
    const previewNy = Math.min(config.ny, maxPreviewPoints);
    
    const stepX = scanWidth / Math.max(previewNx - 1, 1);
    const stepY = scanHeight / Math.max(previewNy - 1, 1);

    // Generate scan path
    const pathPoints = [];
    for (let y = 0; y < previewNy; y++) {
      const yPos = scanTop + y * stepY;
      const isReverse = config.bidirectional && y % 2 === 1;
      
      for (let x = 0; x < previewNx; x++) {
        const xIdx = isReverse ? (previewNx - 1 - x) : x;
        const xPos = scanLeft + xIdx * stepX;
        pathPoints.push({ x: xPos, y: yPos });
      }
    }

    // Grid lines for full DAC range
    const gridLines = [];
    for (let i = 0; i <= 4; i++) {
      const pos = padding + (i / 4) * innerSize;
      const dacVal = (i / 4) * dacMax;
      gridLines.push({ pos, dacVal: Math.round(dacVal) });
    }

    return (
      <svg 
        width={canvasSize} 
        height={canvasSize} 
        style={{ 
          border: '1px solid #444', 
          borderRadius: 4, 
          backgroundColor: '#0a0a15' 
        }}
      >
        {/* Background - Full 4096x4096 DAC range */}
        <rect
          x={padding}
          y={padding}
          width={innerSize}
          height={innerSize}
          fill="#12121f"
          stroke="#333"
          strokeWidth={1}
        />

        {/* Grid lines */}
        {gridLines.map((line, i) => (
          <React.Fragment key={i}>
            {/* Vertical grid line */}
            <line
              x1={line.pos}
              y1={padding}
              x2={line.pos}
              y2={canvasSize - padding}
              stroke="#2a2a4a"
              strokeWidth={0.5}
            />
            {/* Horizontal grid line */}
            <line
              x1={padding}
              y1={line.pos}
              x2={canvasSize - padding}
              y2={line.pos}
              stroke="#2a2a4a"
              strokeWidth={0.5}
            />
            {/* X axis labels */}
            {i < gridLines.length && (
              <text
                x={line.pos}
                y={canvasSize - 5}
                fontSize={8}
                fill="#666"
                textAnchor="middle"
              >
                {line.dacVal}
              </text>
            )}
            {/* Y axis labels */}
            {i < gridLines.length && (
              <text
                x={5}
                y={line.pos + 3}
                fontSize={8}
                fill="#666"
                textAnchor="start"
              >
                {line.dacVal}
              </text>
            )}
          </React.Fragment>
        ))}

        {/* Scan area highlight (the actual scan region) */}
        <rect
          x={scanLeft}
          y={scanTop}
          width={scanWidth}
          height={scanHeight}
          fill="rgba(0, 150, 255, 0.15)"
          stroke="#0096ff"
          strokeWidth={2}
          strokeDasharray="4,2"
        />

        {/* Scan path lines */}
        {pathPoints.length > 1 && (
          <polyline
            points={pathPoints.map(p => `${p.x},${p.y}`).join(' ')}
            fill="none"
            stroke={config.bidirectional ? '#ff9900' : '#00ff88'}
            strokeWidth={1}
            opacity={0.8}
          />
        )}

        {/* Sample points */}
        {pathPoints.slice(0, 200).map((point, i) => (
          <circle
            key={i}
            cx={point.x}
            cy={point.y}
            r={Math.max(1, 3 - pathPoints.length / 50)}
            fill={i === 0 ? '#ff0000' : '#00aaff'}
          />
        ))}

        {/* Start point marker */}
        {pathPoints.length > 0 && (
          <circle
            cx={pathPoints[0].x}
            cy={pathPoints[0].y}
            r={5}
            fill="none"
            stroke="#ff0000"
            strokeWidth={2}
          />
        )}

        {/* Scan direction arrows for bidirectional */}
        {config.bidirectional && previewNy >= 2 && (
          <>
            {/* Forward arrow (line 0) */}
            <polygon
              points={`${scanRight - 8},${scanTop - 2} ${scanRight},${scanTop + 4} ${scanRight - 8},${scanTop + 10}`}
              fill="#00ff88"
            />
            {/* Reverse arrow (line 1) */}
            <polygon
              points={`${scanLeft + 8},${scanTop + stepY - 2} ${scanLeft},${scanTop + stepY + 4} ${scanLeft + 8},${scanTop + stepY + 10}`}
              fill="#ff9900"
            />
          </>
        )}

        {/* Labels */}
        <text x={canvasSize / 2} y={12} fontSize={10} fill="#888" textAnchor="middle" fontWeight="bold">
          DAC Range: 0-4095
        </text>
        
        {/* Scan mode indicator */}
        <rect
          x={canvasSize - 85}
          y={2}
          width={80}
          height={16}
          rx={3}
          fill={config.bidirectional ? '#ff9900' : '#00ff88'}
          opacity={0.3}
        />
        <text 
          x={canvasSize - 45} 
          y={13} 
          fontSize={9} 
          fill={config.bidirectional ? '#ff9900' : '#00ff88'} 
          textAnchor="middle"
          fontWeight="bold"
        >
          {config.bidirectional ? 'BIDI' : 'UNI'}
        </text>
      </svg>
    );
  }, [config]);

  // ========================
  // Render
  // ========================

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Galvo Scanner Controller
      </Typography>

      {/* Scanner Selection — shared across all tabs */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <FormControl fullWidth size="small">
          <InputLabel>Scanner Device</InputLabel>
          <Select
            value={selectedScanner}
            label="Scanner Device"
            onChange={(e) => dispatch(setSelectedScanner(e.target.value))}
          >
            {scannerNames.map(name => (
              <MenuItem key={name} value={name}>{name}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Paper>

      {/* Tab selector */}
      <Tabs
        value={activeTab}
        onChange={(e, v) => dispatch(setActiveTab(v))}
        sx={{ mb: 2 }}
        variant="fullWidth"
      >
        <Tab icon={<GridOnIcon />} label="Raster Scan" />
        <Tab icon={<ScatterPlotIcon />} label="Arbitrary Points" />
        {flimAvailable && <Tab icon={<BiotechIcon />} label="FLIM" />}
      </Tabs>

      {/* Status and Alerts — shared */}
      {statusMessage && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => dispatch(clearStatusMessage())}>
          {statusMessage}
        </Alert>
      )}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => dispatch(clearError())}>
          {error}
        </Alert>
      )}

      {/* ========== TAB 0: Raster Scan ========== */}
      {activeTab === 0 && (
      <Grid container spacing={3}>
        {/* Left Column: Configuration */}
        <Grid item xs={12} md={6}>

          {/* Scan Resolution */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle', fontSize: 20 }} />
              Scan Resolution
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="NX (pixels/line)"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.nx}
                  onChange={handleConfigChange('nx')}
                  inputProps={{ min: 1, max: 4096 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.nx} /> }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="NY (lines)"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.ny}
                  onChange={handleConfigChange('ny')}
                  inputProps={{ min: 1, max: 4096 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.ny} /> }}
                />
              </Grid>
            </Grid>

            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Total pixels: {scanInfo.totalPixels.toLocaleString()}
              </Typography>
            </Box>
          </Paper>

          {/* Position Range */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              X Position Range (DAC: 0-4095)
            </Typography>
            <Slider
              value={[config.x_min, config.x_max]}
              onChange={handleXRangeChange}
              valueLabelDisplay="auto"
              min={0}
              max={4095}
              sx={{ mb: 1 }}
            />
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="X Min"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.x_min}
                  onChange={handleConfigChange('x_min')}
                  inputProps={{ min: 0, max: 4095 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="X Max"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.x_max}
                  onChange={handleConfigChange('x_max')}
                  inputProps={{ min: 0, max: 4095 }}
                />
              </Grid>
            </Grid>

            <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
              Y Position Range (DAC: 0-4095)
            </Typography>
            <Slider
              value={[config.y_min, config.y_max]}
              onChange={handleYRangeChange}
              valueLabelDisplay="auto"
              min={0}
              max={4095}
              sx={{ mb: 1 }}
            />
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="Y Min"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.y_min}
                  onChange={handleConfigChange('y_min')}
                  inputProps={{ min: 0, max: 4095 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Y Max"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.y_max}
                  onChange={handleConfigChange('y_max')}
                  inputProps={{ min: 0, max: 4095 }}
                />
              </Grid>
            </Grid>
          </Paper>

          {/* Timing Parameters */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Timing & Frames
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="Sample Period (µs)"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.sample_period_us}
                  onChange={handleConfigChange('sample_period_us')}
                  inputProps={{ min: 0 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.sample_period_us} /> }}
                  helperText="0 = max speed"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Frame Count"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.frame_count}
                  onChange={handleConfigChange('frame_count')}
                  inputProps={{ min: 0 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.frame_count} /> }}
                  helperText="0 = infinite (raster over CAN is always continuous)"
                />
              </Grid>
            </Grid>

            <FormControlLabel
              control={
                <Checkbox
                  checked={config.bidirectional}
                  onChange={() => dispatch(toggleBidirectional())}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  Bidirectional Scanning
                  <InfoTip text={PARAM_TOOLTIPS.bidirectional} />
                </Box>
              }
              sx={{ mt: 1 }}
            />

            <Box sx={{ mt: 2, p: 1, backgroundColor: 'rgba(0,150,255,0.1)', borderRadius: 1 }}>
              <Typography variant="body2">
                Frame time: ~{scanInfo.frameTimeMs} ms | 
                Rate: ~{scanInfo.frameRate} Hz
              </Typography>
            </Box>
          </Paper>

          {/* Advanced Parameters */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Advanced Parameters
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="Pre-samples"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.pre_samples}
                  onChange={handleConfigChange('pre_samples')}
                  inputProps={{ min: 0 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.pre_samples} /> }}
                  helperText="Pre-scan blanking samples"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Fly-samples"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.fly_samples}
                  onChange={handleConfigChange('fly_samples')}
                  inputProps={{ min: 0 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.fly_samples} /> }}
                  helperText="Fly-back samples"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Trig Delay (µs)"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.trig_delay_us}
                  onChange={handleConfigChange('trig_delay_us')}
                  inputProps={{ min: 0 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.trig_delay_us} /> }}
                  helperText="Frame→line marker gap"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Trig Width (µs)"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.trig_width_us}
                  onChange={handleConfigChange('trig_width_us')}
                  inputProps={{ min: 0 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.trig_width_us} /> }}
                  helperText="Marker/pixel pulse width"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Line Settle Samples"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.line_settle_samples}
                  onChange={handleConfigChange('line_settle_samples')}
                  inputProps={{ min: 0 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.line_settle_samples} /> }}
                  helperText="Line settling samples"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Enable Trigger"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.enable_trigger}
                  onChange={handleConfigChange('enable_trigger')}
                  inputProps={{ min: 0, max: 1 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.enable_trigger} /> }}
                  helperText="0=off, 1=on"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Apply X LUT"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.apply_x_lut}
                  onChange={handleConfigChange('apply_x_lut')}
                  inputProps={{ min: 0, max: 1 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.apply_x_lut} /> }}
                  helperText="0=off, 1=on"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Overscan Samples"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.overscan_samples ?? 0}
                  onChange={handleConfigChange('overscan_samples')}
                  inputProps={{ min: 0 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.overscan_samples} /> }}
                  helperText="Constant-velocity margin"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Laser Blanking"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.laser_blanking ?? 0}
                  onChange={handleConfigChange('laser_blanking')}
                  inputProps={{ min: 0, max: 1 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.laser_blanking} /> }}
                  helperText="0=off, 1=gate laser"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="HW Pixel Clock"
                  type="number"
                  size="small"
                  fullWidth
                  value={config.hw_pixel_clock ?? 0}
                  onChange={handleConfigChange('hw_pixel_clock')}
                  inputProps={{ min: 0, max: 1 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.hw_pixel_clock} /> }}
                  helperText="0=software, 1=RMT"
                />
              </Grid>
            </Grid>
          </Paper>

          {/* Parking Position */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              <CenterFocusStrongIcon sx={{ mr: 1, verticalAlign: 'middle', fontSize: 20 }} />
              Parking Position
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Where the beam is sent when a scan stops. Default is the center (2048, 2048).
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="Park X"
                  type="number"
                  size="small"
                  fullWidth
                  value={parkConfig.park_x}
                  onChange={(e) => saveParkConfig({ park_x: Number(e.target.value) })}
                  inputProps={{ min: 0, max: 4095 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.park_x} /> }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Park Y"
                  type="number"
                  size="small"
                  fullWidth
                  value={parkConfig.park_y}
                  onChange={(e) => saveParkConfig({ park_y: Number(e.target.value) })}
                  inputProps={{ min: 0, max: 4095 }}
                  InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS.park_y} /> }}
                />
              </Grid>
            </Grid>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 1 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={parkConfig.park_on_stop}
                    onChange={(e) => saveParkConfig({ park_on_stop: e.target.checked })}
                  />
                }
                label="Park on stop"
              />
              <Button
                variant="outlined"
                size="small"
                startIcon={<CenterFocusStrongIcon />}
                onClick={parkNow}
              >
                Park now
              </Button>
            </Box>
          </Paper>

          {/* Apply & Start Button */}
          <Button
            variant="contained"
            color="primary"
            startIcon={<SendIcon />}
            onClick={applyConfigAndStartScan}
            fullWidth
            sx={{ mb: 2 }}
          >
            Apply Configuration & Start Scan
          </Button>
        </Grid>

        {/* Right Column: Preview and Controls */}
        <Grid item xs={12} md={6}>
          {/* Scan Pattern Preview */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Scan Pattern Preview
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
              {ScanPatternPreview}
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, textAlign: 'center' }}>
              Full DAC range (4096×4096) • Scan area highlighted in blue
              <br />
              {config.bidirectional 
                ? '🟠 Bidirectional: alternating scan direction' 
                : '🟢 Unidirectional: same direction each line'}
            </Typography>
          </Paper>

          {/* Scanner Status */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="subtitle1">
                Scanner Status
              </Typography>
              <Box>
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={autoRefresh}
                      onChange={(e) => dispatch(setAutoRefresh(e.target.checked))}
                    />
                  }
                  label="Auto-refresh"
                />
                <Tooltip title="Refresh Status">
                  <IconButton size="small" onClick={fetchStatus}>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>

            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Chip
                label={status.running ? 'Running' : 'Stopped'}
                color={status.running ? 'success' : 'default'}
                variant={status.running ? 'filled' : 'outlined'}
              />
              <Chip
                label={`Frame: ${status.current_frame}`}
                variant="outlined"
              />
              <Chip
                label={`Line: ${status.current_line}`}
                variant="outlined"
              />
            </Box>
          </Paper>

          {/* Control Buttons - Always enabled */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Scan Control
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<PlayArrowIcon />}
                  onClick={startScan}
                  fullWidth
                  size="large"
                >
                  Start Scan
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  variant="contained"
                  color="error"
                  startIcon={<StopIcon />}
                  onClick={stopScan}
                  fullWidth
                  size="large"
                >
                  Stop Scan
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {/* Quick Presets */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Quick Presets
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={4}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={handlePresetClick('64x64')}
                >
                  64×64
                </Button>
              </Grid>
              <Grid item xs={4}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={handlePresetClick('256x256')}
                >
                  256×256
                </Button>
              </Grid>
              <Grid item xs={4}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={handlePresetClick('512x512')}
                >
                  512×512
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={handlePresetClick('fullRange')}
                >
                  Full Range
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={handlePresetClick('center50')}
                >
                  Center 50%
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
      )}

      {/* ========== TAB 1: Arbitrary Points ========== */}
      {activeTab === 1 && (
        <GalvoArbitraryPointsTab />
      )}

      {/* ========== TAB 2: FLIM LABS bridge (only when backend-enabled) ========== */}
      {flimAvailable && activeTab === 2 && <FlimLabsPanel />}
    </Box>
  );
};

export default GalvoScannerController;
