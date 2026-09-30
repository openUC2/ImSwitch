import React, { useState, useEffect, useCallback } from 'react';
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
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import GpsFixedIcon from '@mui/icons-material/GpsFixed';
import StraightenIcon from '@mui/icons-material/Straighten';
import ClearIcon from '@mui/icons-material/Clear';
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
  setAxisWindow,
  setScanArea,
  setCameraCalibration,
  setAffineTransform,
  startCalibration,
  getAffineTransformState,
  getCalibrationState,
  getCameraCalibration,
  getScanPhysical,
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
  apiGetAffineTransform,
  apiGetGalvoCameraCalibration,
  apiSnapGalvoCameraBackground,
} from '../backendapi/apiGalvoScannerController';
import GalvoArbitraryPointsTab from './GalvoArbitraryPointsTab';
import GalvoScanPreview from './GalvoScanPreview';
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
  x_width: 'Width of the X scan window in DAC counts. Changing it keeps the centre; the window slides to stay inside 0–4095.',
  y_width: 'Height of the Y scan window in DAC counts. Changing it keeps the centre; the window slides to stay inside 0–4095.',
  x_center: 'Centre of the X scan window. Drag the Position slider (or the blue rectangle in the preview) to move the whole window.',
  y_center: 'Centre of the Y scan window. Drag the Position slider (or the blue rectangle in the preview) to move the whole window.',
  park_x: 'X position (DAC counts, 0–4095) the beam moves to when a scan stops.',
  park_y: 'Y position (DAC counts, 0–4095) the beam moves to when a scan stops.',
  overscan_samples:
    'Extends the X ramp at the same per-pixel slope on both sides of the imaging window, so the mirror is already at constant velocity when triggers/laser start. Compensates the galvo lagging behind the DAC. Costs 2×overscan samples per line.',
  laser_blanking:
    'Gate the laser (galvo laser pin) HIGH only during the imaging window — off during pre-blanking, overscan, fly-back and settle. Prevents fly-back photons from smearing the image (e.g. into a FLIM acquisition).',
  hw_pixel_clock:
    'Generate the pixel clock with the ESP32-S3 RMT peripheral: exactly nx hardware-timed, equidistant pulses per line, decoupled from the DAC/SPI software loop. Falls back to software pulses on unsupported chips.',
};

const fmtUm = (um, digits) => {
  if (um == null || !isFinite(um)) return '–';
  if (um >= 1000) return `${(um / 1000).toFixed(2)} mm`;
  if (um >= 10) return `${um.toFixed(digits ?? 0)} µm`;
  return `${um.toFixed(digits ?? 2)} µm`;
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
  const affine = useSelector(getAffineTransformState);
  const calibration = useSelector(getCalibrationState);
  const cameraCal = useSelector(getCameraCalibration);
  const physical = useSelector(getScanPhysical);

  // Camera snapshot drawn behind the scan preview (component-local: it is a
  // base64 PNG and only this panel needs it; tabs are conditional renders of
  // this same component, so it survives tab switches).
  const [background, setBackground] = useState(null);
  const [bgOpacity, setBgOpacity] = useState(0.7);
  const [snapping, setSnapping] = useState(false);
  // Set when the wizard was launched from here, so we come back afterwards
  const [returnAfterWizard, setReturnAfterWizard] = useState(false);

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

  // The camera->galvo affine is shared with the Arbitrary Points tab; load it
  // here too so the raster preview can place the camera frame without the
  // user having visited that tab first.
  const fetchAffine = useCallback(async () => {
    if (!selectedScanner) return;
    try {
      const data = await apiGetAffineTransform(hostIP, hostPort, selectedScanner);
      if (data.affine_transform) dispatch(setAffineTransform(data.affine_transform));
    } catch (err) {
      console.error('Failed to fetch affine transform:', err);
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  const fetchCameraCalibration = useCallback(async (detectorName = null) => {
    if (!selectedScanner) return;
    try {
      const data = await apiGetGalvoCameraCalibration(hostIP, hostPort, selectedScanner, detectorName);
      if (!data.error) {
        dispatch(setCameraCalibration({
          calibrated: !!data.calibrated,
          detectorName: data.detectorName ?? null,
          cameraDetectors: data.cameraDetectors || [],
          frameWidth: data.frameWidth ?? null,
          frameHeight: data.frameHeight ?? null,
          pixelSizeUmX: data.pixelSizeUmX ?? null,
          pixelSizeUmY: data.pixelSizeUmY ?? null,
          umPerDacX: data.umPerDacX ?? null,
          umPerDacY: data.umPerDacY ?? null,
          rotationDeg: data.rotationDeg ?? null,
          hint: data.hint ?? null,
          flim: data.flim || {},
        }));
      }
    } catch (err) {
      console.error('Failed to fetch camera calibration:', err);
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  const snapBackground = useCallback(async () => {
    setSnapping(true);
    try {
      const data = await apiSnapGalvoCameraBackground(
        hostIP, hostPort, selectedScanner || null, cameraCal.detectorName || null, 1024
      );
      if (data.error) {
        dispatch(setError(`Snap failed: ${data.error}`));
      } else {
        setBackground(data);
      }
    } catch (err) {
      dispatch(setError(`Snap failed: ${err.message}`));
    } finally {
      setSnapping(false);
    }
  }, [hostIP, hostPort, selectedScanner, cameraCal.detectorName, dispatch]);

  // Hand over to the 3-point wizard on the Arbitrary Points tab (it needs the
  // live camera view to click the laser spot).
  const goToCalibrationWizard = useCallback(() => {
    setReturnAfterWizard(true);
    dispatch(setActiveTab(1));
    dispatch(startCalibration());
  }, [dispatch]);

  useEffect(() => {
    if (!returnAfterWizard || calibration.active) return;
    // Wizard finished (completed) or was cancelled: either way it is over.
    setReturnAfterWizard(false);
    if (calibration.completed) dispatch(setActiveTab(0));
  }, [returnAfterWizard, calibration.active, calibration.completed, dispatch]);

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
      fetchAffine();
      fetchCameraCalibration();
    }
  }, [selectedScanner, fetchConfig, fetchStatus, fetchParkConfig, fetchAffine, fetchCameraCalibration]);

  // The wizard (Arbitrary Points tab) writes a new affine on completion; the
  // µm/DAC summary depends on it, so refresh whenever the transform changes.
  useEffect(() => {
    if (selectedScanner) fetchCameraCalibration();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [affine, calibration.completed]);

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

  const handleWindowField = (axis, key) => (event) => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) return;
    dispatch(setAxisWindow({ axis, [key]: value }));
  };

  const handleAreaChange = useCallback((area) => {
    dispatch(setScanArea(area));
  }, [dispatch]);

  const xWidth = config.x_max - config.x_min;
  const yWidth = config.y_max - config.y_min;
  const xCenter = Math.round((config.x_min + config.x_max) / 2);
  const yCenter = Math.round((config.y_min + config.y_max) / 2);
  const umPerDacX = physical?.umPerDacX;
  const umPerDacY = physical?.umPerDacY;


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
            {[
              { axis: 'x', label: 'X', min: config.x_min, max: config.x_max, width: xWidth, center: xCenter, umPerDac: umPerDacX, onRange: handleXRangeChange },
              { axis: 'y', label: 'Y', min: config.y_min, max: config.y_max, width: yWidth, center: yCenter, umPerDac: umPerDacY, onRange: handleYRangeChange },
            ].map((ax) => (
              <Box key={ax.axis} sx={{ mb: ax.axis === 'x' ? 2 : 0 }}>
                <Typography variant="subtitle1" gutterBottom>
                  {ax.label} Position Range (DAC: 0-4095)
                  {ax.umPerDac ? (
                    <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                      1 count ≈ {fmtUm(ax.umPerDac, 3)}
                    </Typography>
                  ) : null}
                </Typography>
                {/* Min/max thumbs */}
                <Slider
                  value={[ax.min, ax.max]}
                  onChange={ax.onRange}
                  valueLabelDisplay="auto"
                  min={0}
                  max={4095}
                  sx={{ mb: 0 }}
                  data-testid={`galvo-${ax.axis}-range-slider`}
                />
                {/* Whole-window position: single thumb limited so the window stays in range */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ minWidth: 56 }}>
                    Position
                  </Typography>
                  <Slider
                    size="small"
                    value={ax.center}
                    onChange={(e, v) => dispatch(setAxisWindow({ axis: ax.axis, center: Number(v) }))}
                    valueLabelDisplay="auto"
                    min={Math.ceil(ax.width / 2)}
                    max={Math.max(Math.ceil(ax.width / 2), 4095 - Math.floor(ax.width / 2))}
                    disabled={ax.width >= 4095}
                    data-testid={`galvo-${ax.axis}-center-slider`}
                  />
                </Box>
                <Grid container spacing={1} sx={{ mt: 0.5 }}>
                  <Grid item xs={6} sm={3}>
                    <TextField
                      label={`${ax.label} Min`}
                      type="number"
                      size="small"
                      fullWidth
                      value={ax.min}
                      onChange={handleConfigChange(`${ax.axis}_min`)}
                      inputProps={{ min: 0, max: 4095 }}
                    />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <TextField
                      label={`${ax.label} Max`}
                      type="number"
                      size="small"
                      fullWidth
                      value={ax.max}
                      onChange={handleConfigChange(`${ax.axis}_max`)}
                      inputProps={{ min: 0, max: 4095 }}
                    />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <TextField
                      label={ax.axis === 'x' ? 'Width' : 'Height'}
                      type="number"
                      size="small"
                      fullWidth
                      value={ax.width}
                      onChange={handleWindowField(ax.axis, 'width')}
                      inputProps={{ min: 1, max: 4095, 'data-testid': `galvo-${ax.axis}-width` }}
                      InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS[`${ax.axis}_width`]} /> }}
                      helperText={ax.umPerDac ? `≈ ${fmtUm(ax.width * ax.umPerDac)}` : undefined}
                    />
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <TextField
                      label="Center"
                      type="number"
                      size="small"
                      fullWidth
                      value={ax.center}
                      onChange={handleWindowField(ax.axis, 'center')}
                      inputProps={{ min: 0, max: 4095, 'data-testid': `galvo-${ax.axis}-center` }}
                      InputProps={{ endAdornment: <InfoTip text={PARAM_TOOLTIPS[`${ax.axis}_center`]} /> }}
                    />
                  </Grid>
                </Grid>
              </Box>
            ))}
            {physical && (
              <Typography variant="body2" sx={{ mt: 1.5, p: 1, backgroundColor: 'rgba(0,150,255,0.1)', borderRadius: 1 }}>
                Scan field: {fmtUm(physical.fovUmX)} × {fmtUm(physical.fovUmY)} —
                pixel {fmtUm(physical.pixelUmX, 3)} × {fmtUm(physical.pixelUmY, 3)} at {config.nx}×{config.ny}
              </Typography>
            )}
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
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mb: 1 }}>
              <Typography variant="subtitle1">
                Scan Pattern Preview
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Tooltip title={cameraCal.detectorName
                  ? `Grab one frame from "${cameraCal.detectorName}" and show it behind the scan area`
                  : 'No 2D camera configured'}>
                  <span>
                    <Button
                      size="small"
                      variant="outlined"
                      startIcon={<CameraAltIcon />}
                      onClick={snapBackground}
                      disabled={snapping || !cameraCal.detectorName}
                      data-testid="galvo-snap-background"
                    >
                      {snapping ? 'Snapping…' : 'Snap camera'}
                    </Button>
                  </span>
                </Tooltip>
                {background && (
                  <Tooltip title="Remove the camera background">
                    <IconButton size="small" onClick={() => setBackground(null)} data-testid="galvo-clear-background">
                      <ClearIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
              <GalvoScanPreview
                config={config}
                affine={affine}
                background={background}
                backgroundOpacity={bgOpacity}
                physical={physical}
                onAreaChange={handleAreaChange}
                size={320}
              />
            </Box>
            {background && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1, px: 1 }}>
                <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                  Camera opacity
                </Typography>
                <Slider
                  size="small"
                  value={bgOpacity}
                  onChange={(e, v) => setBgOpacity(Number(v))}
                  min={0}
                  max={1}
                  step={0.05}
                  sx={{ maxWidth: 160 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                  {background.detectorName} {background.frameWidth}×{background.frameHeight}px
                </Typography>
              </Box>
            )}
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, textAlign: 'center' }}>
              Drag the blue rectangle to move the scan, pull its corners to resize
              {background && !cameraCal.calibrated && (
                <>
                  <br />
                  ⚠️ Uncalibrated: camera frame is assumed to span the full DAC range
                </>
              )}
              <br />
              {config.bidirectional 
                ? '🟠 Bidirectional: alternating scan direction' 
                : '🟢 Unidirectional: same direction each line'}
            </Typography>
          </Paper>

          {/* Scanner <-> camera calibration (physical units) */}
          <Paper sx={{ p: 2, mb: 2 }} data-testid="galvo-calibration-panel">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <StraightenIcon sx={{ fontSize: 20 }} />
              <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                Scanner ↔ Camera Calibration
              </Typography>
              <Chip
                size="small"
                label={cameraCal.calibrated ? 'Calibrated' : 'Not calibrated'}
                color={cameraCal.calibrated ? 'success' : 'warning'}
                variant={cameraCal.calibrated ? 'filled' : 'outlined'}
              />
              <Tooltip title="Re-read the calibration from the backend">
                <IconButton size="small" onClick={() => fetchCameraCalibration()}>
                  <RefreshIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            {cameraCal.cameraDetectors.length > 1 && (
              <FormControl size="small" fullWidth sx={{ mb: 1 }}>
                <InputLabel>Camera</InputLabel>
                <Select
                  value={cameraCal.detectorName || ''}
                  label="Camera"
                  onChange={(e) => fetchCameraCalibration(e.target.value)}
                >
                  {cameraCal.cameraDetectors.map((name) => (
                    <MenuItem key={name} value={name}>{name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}
            {cameraCal.calibrated && physical ? (
              <>
                <Typography variant="body2">
                  1 DAC count ≈ {fmtUm(physical.umPerDacX, 4)} (X) / {fmtUm(physical.umPerDacY, 4)} (Y)
                  {cameraCal.rotationDeg != null && Math.abs(cameraCal.rotationDeg) > 0.5 && (
                    <> · scanner X rotated {cameraCal.rotationDeg.toFixed(1)}° vs. camera</>
                  )}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Full DAC range ≈ {fmtUm(physical.fullScaleUmX)} × {fmtUm(physical.fullScaleUmY)} ·
                  camera pixel {fmtUm(cameraCal.pixelSizeUmX, 3)}
                  {cameraCal.detectorName ? ` (${cameraCal.detectorName})` : ''}
                </Typography>
                {cameraCal.pixelSizeUmX === 1 && cameraCal.pixelSizeUmY === 1 && (
                  <Alert severity="warning" sx={{ mt: 1 }} variant="outlined">
                    The camera reports a pixel size of exactly 1 µm — the default. The µm values above are
                    only as good as the camera's pixel-size calibration.
                  </Alert>
                )}
                {Object.keys(cameraCal.flim || {}).length > 0 && (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                    FLIM detector{Object.keys(cameraCal.flim).length > 1 ? 's' : ''} {Object.keys(cameraCal.flim).join(', ')} use
                    this calibration for pixel size / field of view
                    {Object.values(cameraCal.flim).some((f) => f?.source && f.source !== 'affine') ? ' (setup-file values)' : ''}.
                  </Typography>
                )}
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                {cameraCal.hint || 'Without a calibration the scan range is only known in DAC counts. '}
                {cameraCal.detectorName
                  ? ' The 3-point wizard moves the beam to three positions and asks you to click the laser spot in the camera view; the result relates DAC counts to camera pixels and, via the camera pixel size, to micrometres.'
                  : ''}
              </Typography>
            )}
            <Button
              variant={cameraCal.calibrated ? 'outlined' : 'contained'}
              color="secondary"
              size="small"
              startIcon={<GpsFixedIcon />}
              onClick={goToCalibrationWizard}
              sx={{ mt: 1.5 }}
              fullWidth
              data-testid="galvo-open-wizard"
            >
              {cameraCal.calibrated ? 'Re-run calibration wizard' : 'Calibrate scanner to camera (3-point wizard)'}
            </Button>
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
