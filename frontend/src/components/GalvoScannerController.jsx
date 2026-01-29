import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  CircularProgress,
  TextField,
  Slider,
  FormControlLabel,
  Checkbox,
  Divider,
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
import SaveIcon from '@mui/icons-material/Save';
import { useSelector } from 'react-redux';
import { getConnectionSettingsState } from '../state/slices/ConnectionSettingsSlice';

/**
 * GalvoScannerController - Control panel for galvo mirror scanners
 * 
 * Features:
 * - Configure scan parameters (nx, ny, x/y ranges, timing)
 * - Start/stop galvo scans
 * - Real-time status polling
 * - Visual preview of scan pattern
 * - Multiple scanner device support
 */
const GalvoScannerController = () => {
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  // Scanner selection
  const [scannerNames, setScannerNames] = useState([]);
  const [selectedScanner, setSelectedScanner] = useState('');

  // Scan parameters
  const [config, setConfig] = useState({
    nx: 256,
    ny: 256,
    x_min: 500,
    x_max: 3500,
    y_min: 500,
    y_max: 3500,
    sample_period_us: 1,
    frame_count: 0,
    bidirectional: false
  });

  // Status
  const [status, setStatus] = useState({
    running: false,
    current_frame: 0,
    current_line: 0
  });

  // UI state
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(false);

  // API base URL
  const apiBase = useMemo(() => 
    `${hostIP}:${hostPort}/imswitch/api/GalvoScannerController`,
    [hostIP, hostPort]
  );

  // ========================
  // API Functions
  // ========================

  const fetchScannerNames = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/getGalvoScannerNames`);
      const data = await response.json();
      if (Array.isArray(data)) {
        setScannerNames(data);
        if (data.length > 0 && !selectedScanner) {
          setSelectedScanner(data[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch scanner names:', err);
    }
  }, [apiBase, selectedScanner]);

  const fetchConfig = useCallback(async () => {
    if (!selectedScanner) return;
    try {
      const response = await fetch(
        `${apiBase}/getGalvoScannerConfig?scannerName=${selectedScanner}`
      );
      const data = await response.json();
      if (data.config) {
        setConfig(data.config);
      }
    } catch (err) {
      console.error('Failed to fetch config:', err);
    }
  }, [apiBase, selectedScanner]);

  const fetchStatus = useCallback(async () => {
    if (!selectedScanner) return;
    try {
      const response = await fetch(
        `${apiBase}/getGalvoScannerStatus?scannerName=${selectedScanner}`
      );
      const data = await response.json();
      if (!data.error) {
        setStatus({
          running: data.running || false,
          current_frame: data.current_frame || 0,
          current_line: data.current_line || 0
        });
      }
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  }, [apiBase, selectedScanner]);

  const updateConfig = useCallback(async (newConfig) => {
    if (!selectedScanner) return;
    try {
      const params = new URLSearchParams({
        scannerName: selectedScanner,
        ...Object.fromEntries(
          Object.entries(newConfig).map(([k, v]) => [k, String(v)])
        )
      });
      const response = await fetch(`${apiBase}/setGalvoScanConfig?${params}`);
      const data = await response.json();
      if (data.config) {
        setConfig(data.config);
        setStatusMessage('Configuration updated');
        setTimeout(() => setStatusMessage(''), 2000);
      }
    } catch (err) {
      setError(`Failed to update config: ${err.message}`);
    }
  }, [apiBase, selectedScanner]);

  const startScan = useCallback(async () => {
    if (!selectedScanner) return;
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({
        scannerName: selectedScanner,
        nx: String(config.nx),
        ny: String(config.ny),
        x_min: String(config.x_min),
        x_max: String(config.x_max),
        y_min: String(config.y_min),
        y_max: String(config.y_max),
        sample_period_us: String(config.sample_period_us),
        frame_count: String(config.frame_count),
        bidirectional: String(config.bidirectional)
      });
      const response = await fetch(`${apiBase}/startGalvoScan?${params}`);
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setStatusMessage('Scan started');
        setStatus(prev => ({ ...prev, running: true }));
      }
    } catch (err) {
      setError(`Failed to start scan: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [apiBase, selectedScanner, config]);

  const stopScan = useCallback(async () => {
    if (!selectedScanner) return;
    setLoading(true);
    try {
      const response = await fetch(
        `${apiBase}/stopGalvoScan?scannerName=${selectedScanner}`
      );
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setStatusMessage('Scan stopped');
        setStatus(prev => ({ ...prev, running: false }));
      }
    } catch (err) {
      setError(`Failed to stop scan: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [apiBase, selectedScanner]);

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
    if (autoRefresh && status.running) {
      const interval = setInterval(fetchStatus, 500);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, status.running, fetchStatus]);

  // ========================
  // Handlers
  // ========================

  const handleConfigChange = (field) => (event) => {
    const value = event.target.type === 'checkbox' 
      ? event.target.checked 
      : Number(event.target.value);
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleSliderChange = (field) => (event, newValue) => {
    setConfig(prev => ({ ...prev, [field]: newValue }));
  };

  const handleApplyConfig = () => {
    updateConfig(config);
  };

  // ========================
  // Scan Pattern Visualization
  // ========================

  const ScanPatternPreview = useMemo(() => {
    const width = 200;
    const height = 200;
    const padding = 10;
    
    // Map DAC values to canvas coordinates
    const mapX = (dac) => padding + ((dac - config.x_min) / (config.x_max - config.x_min)) * (width - 2 * padding);
    const mapY = (dac) => padding + ((dac - config.y_min) / (config.y_max - config.y_min)) * (height - 2 * padding);

    // Generate sample points for preview
    const points = [];
    const stepX = (config.x_max - config.x_min) / Math.min(config.nx, 32);
    const stepY = (config.y_max - config.y_min) / Math.min(config.ny, 32);

    for (let y = 0; y < Math.min(config.ny, 32); y++) {
      const yVal = config.y_min + y * stepY;
      const isReverse = config.bidirectional && y % 2 === 1;
      
      for (let x = 0; x < Math.min(config.nx, 32); x++) {
        const xIdx = isReverse ? (Math.min(config.nx, 32) - 1 - x) : x;
        const xVal = config.x_min + xIdx * stepX;
        points.push({ x: mapX(xVal), y: mapY(yVal) });
      }
    }

    return (
      <svg width={width} height={height} style={{ border: '1px solid #ccc', borderRadius: 4, backgroundColor: '#1a1a2e' }}>
        {/* Scan area rectangle */}
        <rect
          x={padding}
          y={padding}
          width={width - 2 * padding}
          height={height - 2 * padding}
          fill="none"
          stroke="#4a4a6a"
          strokeWidth={1}
        />
        
        {/* Grid lines */}
        {[...Array(5)].map((_, i) => (
          <React.Fragment key={i}>
            <line
              x1={padding + (i / 4) * (width - 2 * padding)}
              y1={padding}
              x2={padding + (i / 4) * (width - 2 * padding)}
              y2={height - padding}
              stroke="#3a3a5a"
              strokeWidth={0.5}
            />
            <line
              x1={padding}
              y1={padding + (i / 4) * (height - 2 * padding)}
              x2={width - padding}
              y2={padding + (i / 4) * (height - 2 * padding)}
              stroke="#3a3a5a"
              strokeWidth={0.5}
            />
          </React.Fragment>
        ))}

        {/* Scan path */}
        {points.length > 1 && (
          <polyline
            points={points.map(p => `${p.x},${p.y}`).join(' ')}
            fill="none"
            stroke="#00ff88"
            strokeWidth={1}
            opacity={0.7}
          />
        )}

        {/* Sample points */}
        {points.slice(0, 100).map((point, i) => (
          <circle
            key={i}
            cx={point.x}
            cy={point.y}
            r={2}
            fill="#00aaff"
          />
        ))}

        {/* Labels */}
        <text x={width / 2} y={height - 2} fontSize={10} fill="#888" textAnchor="middle">
          X: {config.x_min}-{config.x_max}
        </text>
        <text x={5} y={height / 2} fontSize={10} fill="#888" transform={`rotate(-90, 5, ${height / 2})`} textAnchor="middle">
          Y: {config.y_min}-{config.y_max}
        </text>
      </svg>
    );
  }, [config]);

  // ========================
  // Computed values
  // ========================

  const scanInfo = useMemo(() => {
    const totalPixels = config.nx * config.ny;
    const frameTimeMs = (totalPixels * config.sample_period_us) / 1000;
    const frameRate = config.sample_period_us > 0 ? 1000 / frameTimeMs : '∞';
    return {
      totalPixels,
      frameTimeMs: frameTimeMs.toFixed(2),
      frameRate: typeof frameRate === 'number' ? frameRate.toFixed(1) : frameRate
    };
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
                onChange={(e) => setSelectedScanner(e.target.value)}
              >
                {scannerNames.map(name => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Paper>

          {/* Status and Alerts */}
          {statusMessage && (
            <Alert severity="success" sx={{ mb: 2 }} onClose={() => setStatusMessage('')}>
              {statusMessage}
            </Alert>
          )}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
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
              onChange={(e, val) => setConfig(prev => ({ ...prev, x_min: val[0], x_max: val[1] }))}
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
              onChange={(e, val) => setConfig(prev => ({ ...prev, y_min: val[0], y_max: val[1] }))}
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
                  onChange={handleConfigChange('bidirectional')}
                />
              }
              label="Bidirectional Scanning"
              sx={{ mt: 1 }}
            />

            <Box sx={{ mt: 2, p: 1, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
              <Typography variant="body2">
                Frame time: ~{scanInfo.frameTimeMs} ms | 
                Rate: ~{scanInfo.frameRate} Hz
              </Typography>
            </Box>
          </Paper>

          {/* Apply Button */}
          <Button
            variant="outlined"
            startIcon={<SaveIcon />}
            onClick={handleApplyConfig}
            fullWidth
            sx={{ mb: 2 }}
          >
            Apply Configuration
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
              Preview shows first 32×32 points of scan pattern
              {config.bidirectional && ' (bidirectional)'}
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
                      onChange={(e) => setAutoRefresh(e.target.checked)}
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

          {/* Control Buttons */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Scan Control
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Button
                  variant="contained"
                  color="success"
                  startIcon={loading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
                  onClick={startScan}
                  disabled={loading || status.running}
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
                  startIcon={loading ? <CircularProgress size={20} /> : <StopIcon />}
                  onClick={stopScan}
                  disabled={loading || !status.running}
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
                  onClick={() => setConfig(prev => ({ ...prev, nx: 64, ny: 64 }))}
                >
                  64×64
                </Button>
              </Grid>
              <Grid item xs={4}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={() => setConfig(prev => ({ ...prev, nx: 256, ny: 256 }))}
                >
                  256×256
                </Button>
              </Grid>
              <Grid item xs={4}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={() => setConfig(prev => ({ ...prev, nx: 512, ny: 512 }))}
                >
                  512×512
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={() => setConfig(prev => ({ 
                    ...prev, 
                    x_min: 0, x_max: 4095, 
                    y_min: 0, y_max: 4095 
                  }))}
                >
                  Full Range
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  variant="outlined"
                  size="small"
                  fullWidth
                  onClick={() => setConfig(prev => ({ 
                    ...prev, 
                    x_min: 1024, x_max: 3072, 
                    y_min: 1024, y_max: 3072 
                  }))}
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
