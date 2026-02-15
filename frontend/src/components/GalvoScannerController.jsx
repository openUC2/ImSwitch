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
  Chip
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';
import SettingsIcon from '@mui/icons-material/Settings';
import SendIcon from '@mui/icons-material/Send';
import { useSelector, useDispatch } from 'react-redux';
import { getConnectionSettingsState } from '../state/slices/ConnectionSettingsSlice';
import {
  getGalvoScannerState,
  getGalvoConfig,
  getGalvoStatus,
  getScanInfo,
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
} from '../state/slices/GalvoScannerSlice';
import {
  apiGetGalvoScannerNames,
  apiGetGalvoScannerConfig,
  apiGetGalvoScannerStatus,
  apiStartGalvoScan,
  apiStopGalvoScan,
} from '../backendapi/apiGalvoScannerController';

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

  // Redux state
  const galvoState = useSelector(getGalvoScannerState);
  const config = useSelector(getGalvoConfig);
  const status = useSelector(getGalvoStatus);
  const scanInfo = useSelector(getScanInfo);
  
  // Destructure with defaults for safety
  const scannerNames = galvoState?.scannerNames || [];
  const selectedScanner = galvoState?.selectedScanner || '';
  const error = galvoState?.error || null;
  const statusMessage = galvoState?.statusMessage || '';
  const autoRefresh = galvoState?.autoRefresh || false;

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
    }
  }, [selectedScanner, fetchConfig, fetchStatus]);

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

      <Grid container spacing={3}>
        {/* Left Column: Configuration */}
        <Grid item xs={12} md={6}>
          {/* Scanner Selection */}
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

          {/* Status and Alerts */}
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
                  helperText="0 = infinite"
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
              label="Bidirectional Scanning"
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
                  helperText="Pre-scan samples"
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
                  helperText="Trigger delay"
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
                  helperText="Trigger width"
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
                  helperText="Line settling"
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
                  helperText="0=off, 1=on"
                />
              </Grid>
            </Grid>
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
    </Box>
  );
};

export default GalvoScannerController;
