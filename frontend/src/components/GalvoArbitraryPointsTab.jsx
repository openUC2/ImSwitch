import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  TextField,
  Slider,
  IconButton,
  Tooltip,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import PauseIcon from '@mui/icons-material/Pause';
import DeleteIcon from '@mui/icons-material/Delete';
import ClearAllIcon from '@mui/icons-material/ClearAll';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import TuneIcon from '@mui/icons-material/Tune';
import GpsFixedIcon from '@mui/icons-material/GpsFixed';
import SaveIcon from '@mui/icons-material/Save';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import DownloadIcon from '@mui/icons-material/Download';
import UploadIcon from '@mui/icons-material/Upload';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import { useSelector, useDispatch } from 'react-redux';
import { getConnectionSettingsState } from '../state/slices/ConnectionSettingsSlice';
import {
  getArbitraryPointsState,
  getArbitraryPointsList,
  getAffineTransformState,
  getCalibrationState,
  getGalvoScannerState,
  addArbitraryPoint,
  removeArbitraryPoint,
  updateArbitraryPoint,
  clearArbitraryPoints,
  setArbitraryPoints,
  setDefaultDwellUs,
  setDefaultIntensity,
  applyDefaultDwellToAll,
  applyDefaultIntensityToAll,
  setLaserTrigger,
  setArbScanRunning,
  setArbScanPaused,
  setAffineTransform,
  setAffineParam,
  resetAffineTransform,
  startCalibration,
  cancelCalibration,
  setCalibrationCamPoint,
  advanceCalibrationStep,
  setCalibrationComplete,
  setError,
  clearError,
  setStatusMessage,
  clearStatusMessage,
} from '../state/slices/GalvoScannerSlice';
import {
  apiStartArbitraryScan,
  apiStopArbitraryScan,
  apiPauseArbitraryScan,
  apiResumeArbitraryScan,
  apiGetAffineTransform,
  apiSetAffineTransform,
  apiResetAffineTransform,
  apiRunAffineCalibration,
  apiGetCalibrationPoints,
  apiSetArbitraryPoints,
} from '../backendapi/apiGalvoScannerController';
import GalvoAffineCalibrationWizard from './GalvoAffineCalibrationWizard';
import * as liveStreamSlice from '../state/slices/LiveStreamSlice';
import LiveViewControlWrapper from '../axon/LiveViewControlWrapper';

/**
 * GalvoArbitraryPointsTab - Tab for interactive point-based galvo scanning
 * 
 * Features:
 * - Camera overlay with interactive point drawing
 * - Points table with per-point dwell time and laser intensity
 * - Global defaults with "apply to all"
 * - Affine transform editor (advanced panel)
 * - Calibration wizard trigger
 * - Start/Stop/Pause/Resume controls
 */
const GalvoArbitraryPointsTab = () => {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  const galvoState = useSelector(getGalvoScannerState);
  const selectedScanner = galvoState?.selectedScanner || '';
  const arbState = useSelector(getArbitraryPointsState);
  const pointsList = useSelector(getArbitraryPointsList);
  const affine = useSelector(getAffineTransformState);
  const calibration = useSelector(getCalibrationState);
  const error = galvoState?.error || null;
  const statusMessage = galvoState?.statusMessage || '';
  // Live stream state — used for subsampling factor to convert display pixels → sensor coords
  const liveStream = useSelector(liveStreamSlice.getLiveStreamState);

  // Displayed image dimensions reported by LiveViewControlWrapper
  const [imageSize, setImageSize] = useState({ width: 640, height: 480 });
  const [isDrawing, setIsDrawing] = useState(true);

  // Manual point creation form state
  const [newPointX, setNewPointX] = useState(2048);
  const [newPointY, setNewPointY] = useState(2048);
  const [newPointDwell, setNewPointDwell] = useState(arbState.defaultDwellUs);
  const [newPointIntensity, setNewPointIntensity] = useState(arbState.defaultIntensity);

  // ========================
  // Coordinate helpers (DAC space = 0-4095 = full scanner range)
  // ========================

  // Returns subsampling factor of the current stream format
  const getSubsamplingFactor = useCallback(() => {
    const fmt = liveStream.streamSettings?.current_compression_algorithm || 'jpeg';
    if (fmt === 'binary') return liveStream.streamSettings?.binary?.subsampling?.factor || 1;
    if (fmt === 'webrtc') return liveStream.streamSettings?.webrtc?.subsampling_factor || 1;
    return 1; // jpeg: full resolution, no extra subsampling
  }, [liveStream.streamSettings]);

  // Convert streamed image pixel (pixelX/Y within imgWidth/imgHeight) → galvo DAC (0-4095)
  const pixelToDac = useCallback((pixelX, pixelY, imgWidth, imgHeight) => {
    const sub = getSubsamplingFactor();
    // Scale up to sensor resolution
    const sensorX = pixelX * sub;
    const sensorY = pixelY * sub;
    const sensorW = imgWidth * sub;
    const sensorH = imgHeight * sub;
    const { a11, a12, tx, a21, a22, ty } = affine;
    const isIdentity = a11 === 1 && a12 === 0 && tx === 0 && a21 === 0 && a22 === 1 && ty === 0;
    if (isIdentity) {
      // Linear mapping: sensor full range → 0-4095
      return {
        x: Math.max(0, Math.min(4095, Math.round((sensorX / sensorW) * 4095))),
        y: Math.max(0, Math.min(4095, Math.round((sensorY / sensorH) * 4095))),
      };
    }
    // Affine: galvo = A * sensor + b
    return {
      x: Math.max(0, Math.min(4095, Math.round(a11 * sensorX + a12 * sensorY + tx))),
      y: Math.max(0, Math.min(4095, Math.round(a21 * sensorX + a22 * sensorY + ty))),
    };
  }, [affine, getSubsamplingFactor]);

  // Convert galvo DAC (0-4095) → display pixel coords in the streamed image space
  const dacToDisplayPixel = useCallback((dacX, dacY) => {
    const sub = getSubsamplingFactor();
    const { a11, a12, tx, a21, a22, ty } = affine;
    const isIdentity = a11 === 1 && a12 === 0 && tx === 0 && a21 === 0 && a22 === 1 && ty === 0;
    const { width: imgW, height: imgH } = imageSize;
    if (isIdentity) {
      return { x: (dacX / 4095) * imgW, y: (dacY / 4095) * imgH };
    }
    // Inverse affine: sensor = A^-1 * (dac - b)
    const det = a11 * a22 - a12 * a21;
    if (Math.abs(det) < 1e-9) {
      return { x: (dacX / 4095) * imgW, y: (dacY / 4095) * imgH };
    }
    const sensorX = (a22 * (dacX - tx) - a12 * (dacY - ty)) / det;
    const sensorY = (-a21 * (dacX - tx) + a11 * (dacY - ty)) / det;
    return { x: sensorX / sub, y: sensorY / sub };
  }, [affine, getSubsamplingFactor, imageSize]);

  // ========================
  // Click Handler (from LiveViewControlWrapper)
  // ========================

  // onClick signature: (pixelX, pixelY, imgWidth, imgHeight, displayInfo)
  const handleLiveViewClick = useCallback((pixelX, pixelY, imgWidth, imgHeight) => {
    if (!isDrawing) return;
    if (pointsList.length >= 265) return;

    // In calibration mode, record sensor coords (not DAC)
    if (calibration.active) {
      const sub = getSubsamplingFactor();
      dispatch(setCalibrationCamPoint({
        step: calibration.currentStep,
        x: Math.round(pixelX * sub),
        y: Math.round(pixelY * sub),
      }));
      return;
    }

    const dac = pixelToDac(pixelX, pixelY, imgWidth, imgHeight);
    dispatch(addArbitraryPoint({ x: dac.x, y: dac.y }));
  }, [isDrawing, pointsList.length, calibration, pixelToDac, getSubsamplingFactor, dispatch]);

  // ========================
  // Scan Controls
  // ========================

  const handleStart = useCallback(async () => {
    if (!selectedScanner || pointsList.length === 0) return;
    dispatch(setArbScanRunning(true));
    dispatch(clearError());
    try {
      // Points in Redux are already in DAC space (0-4095), converted by pixelToDac()
      // or entered manually in DAC coords.  Do NOT apply affine again on the backend.
      const data = await apiStartArbitraryScan(
        hostIP, hostPort, selectedScanner, pointsList,
        arbState.laserTrigger, false
      );
      if (data.error) {
        dispatch(setError(data.error));
        dispatch(setArbScanRunning(false));
      } else {
        dispatch(setStatusMessage('Arbitrary scan started'));
        setTimeout(() => dispatch(clearStatusMessage()), 2000);
      }
    } catch (err) {
      dispatch(setError(`Failed to start: ${err.message}`));
      dispatch(setArbScanRunning(false));
    }
  }, [hostIP, hostPort, selectedScanner, pointsList, arbState.laserTrigger, dispatch]);

  const handleStop = useCallback(async () => {
    dispatch(setArbScanRunning(false));
    try {
      await apiStopArbitraryScan(hostIP, hostPort, selectedScanner);
      dispatch(setStatusMessage('Arbitrary scan stopped'));
      setTimeout(() => dispatch(clearStatusMessage()), 2000);
    } catch (err) {
      dispatch(setError(`Failed to stop: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  const handlePause = useCallback(async () => {
    dispatch(setArbScanPaused(true));
    try {
      await apiPauseArbitraryScan(hostIP, hostPort, selectedScanner);
      dispatch(setStatusMessage('Arbitrary scan paused'));
      setTimeout(() => dispatch(clearStatusMessage()), 2000);
    } catch (err) {
      dispatch(setError(`Failed to pause: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  const handleResume = useCallback(async () => {
    dispatch(setArbScanPaused(false));
    try {
      await apiResumeArbitraryScan(hostIP, hostPort, selectedScanner);
      dispatch(setStatusMessage('Arbitrary scan resumed'));
      setTimeout(() => dispatch(clearStatusMessage()), 2000);
    } catch (err) {
      dispatch(setError(`Failed to resume: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  // ========================
  // Affine Transform Actions
  // ========================

  const handleLoadAffine = useCallback(async () => {
    try {
      const data = await apiGetAffineTransform(hostIP, hostPort, selectedScanner);
      if (data.affine_transform) {
        dispatch(setAffineTransform(data.affine_transform));
        dispatch(setStatusMessage('Affine transform loaded'));
        setTimeout(() => dispatch(clearStatusMessage()), 2000);
      }
    } catch (err) {
      dispatch(setError(`Failed to load affine: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  const handleSaveAffine = useCallback(async () => {
    try {
      await apiSetAffineTransform(hostIP, hostPort, selectedScanner, affine, true);
      dispatch(setStatusMessage('Affine transform saved'));
      setTimeout(() => dispatch(clearStatusMessage()), 2000);
    } catch (err) {
      dispatch(setError(`Failed to save affine: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, affine, dispatch]);

  const handleResetAffine = useCallback(async () => {
    dispatch(resetAffineTransform());
    try {
      await apiResetAffineTransform(hostIP, hostPort, selectedScanner, true);
      dispatch(setStatusMessage('Affine transform reset to identity'));
      setTimeout(() => dispatch(clearStatusMessage()), 2000);
    } catch (err) {
      dispatch(setError(`Failed to reset affine: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  // Load affine from backend on mount
  useEffect(() => {
    if (selectedScanner) {
      handleLoadAffine();
    }
  }, [selectedScanner]); // eslint-disable-line react-hooks/exhaustive-deps

  // ========================
  // Import / Export Points
  // ========================

  const handleExportPoints = useCallback(() => {
    const data = JSON.stringify(pointsList, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'galvo_points.json';
    a.click();
    URL.revokeObjectURL(url);
  }, [pointsList]);

  const handleImportPoints = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        try {
          const imported = JSON.parse(evt.target.result);
          if (Array.isArray(imported)) {
            dispatch(setArbitraryPoints(imported.slice(0, 265)));
            dispatch(setStatusMessage(`Imported ${imported.length} points`));
            setTimeout(() => dispatch(clearStatusMessage()), 2000);
          }
        } catch {
          dispatch(setError('Invalid JSON file'));
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }, [dispatch]);

  // ========================
  // SVG overlay rendered on top of the live stream
  // Points are stored in galvo DAC space (0-4095 × 0-4095).
  // They are mapped back to the displayed image coordinate space for rendering.
  // ========================

  const overlayContent = useMemo(() => {
    const { width: imgW, height: imgH } = imageSize;

    // Point markers
    const pointMarkers = pointsList.map((pt, i) => {
      const { x: cx, y: cy } = dacToDisplayPixel(pt.x, pt.y);
      const intensity = pt.laser_intensity ?? 128;
      const r = Math.round((intensity / 255) * 255);
      const b = Math.round(((255 - intensity) / 255) * 255);
      const color = `rgb(${r},50,${b})`;
      return (
        <React.Fragment key={i}>
          <circle cx={cx} cy={cy} r={6} fill={color} stroke="#fff" strokeWidth={1.5} opacity={0.85} />
          <text x={cx + 8} y={cy - 4} fontSize={10} fill="#fff" fontWeight="bold">{i + 1}</text>
        </React.Fragment>
      );
    });

    // Calibration markers (sensor coords stored, divide by subsampling for display)
    let calMarkers = null;
    if (calibration.active) {
      const sub = getSubsamplingFactor();
      const step = calibration.currentStep;
      calMarkers = (
        <>
          {calibration.camPoints.map((pt, i) => {
            if (!pt || i > step) return null;
            const cx = pt.x / sub;
            const cy = pt.y / sub;
            return (
              <React.Fragment key={`cal-${i}`}>
                <circle cx={cx} cy={cy} r={8} fill="none" stroke="#00ff00" strokeWidth={2} />
                <circle cx={cx} cy={cy} r={3} fill="#00ff00" />
                <text x={cx + 10} y={cy + 4} fontSize={11} fill="#00ff00" fontWeight="bold">Cal {i + 1}</text>
              </React.Fragment>
            );
          })}
          <text x={10} y={24} fontSize={13} fill="#ffff00" fontWeight="bold">
            CALIBRATION: Click point {step + 1}/3 ({calibration.galvoPoints[step]?.label})
          </text>
        </>
      );
    }

    return (
      <svg
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
        viewBox={`0 0 ${imgW} ${imgH}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {pointMarkers}
        {calMarkers}
      </svg>
    );
  }, [pointsList, calibration, dacToDisplayPixel, getSubsamplingFactor, imageSize]);

  // ========================
  // Render
  // ========================
  return (
    <Box>
      {/* Alerts */}
      {statusMessage && (
        <Alert severity="success" sx={{ mb: 1 }} onClose={() => dispatch(clearStatusMessage())}>
          {statusMessage}
        </Alert>
      )}
      {error && (
        <Alert severity="error" sx={{ mb: 1 }} onClose={() => dispatch(clearError())}>
          {error}
        </Alert>
      )}

      <Grid container spacing={2}>
        {/* ===== LEFT: Camera + Overlay ===== */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ p: 1, mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Camera View — Click to add points ({pointsList.length}/265)
            </Typography>

            {/* Live stream with SVG point overlay.
                 Points are stored as galvo DAC coords (0-4095).
                 overlayContent maps them back to display-pixel space. */}
            <Box sx={{
              border: calibration.active ? '2px solid #ffff00' : '1px solid #444',
              borderRadius: 1,
              overflow: 'hidden',
              cursor: isDrawing ? 'crosshair' : 'default',
            }}>
              <LiveViewControlWrapper
                onClick={handleLiveViewClick}
                onImageLoad={(w, h) => setImageSize({ width: w, height: h })}
                overlayContent={overlayContent}
                enableStageMovement={false}
              />
            </Box>

            {/* Camera overlay toolbar */}
            <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap', alignItems: 'center' }}>
              <Button
                size="small"
                variant={isDrawing ? 'contained' : 'outlined'}
                onClick={() => setIsDrawing(!isDrawing)}
              >
                {isDrawing ? '✏️ Drawing ON' : '✏️ Drawing OFF'}
              </Button>
              <Button
                size="small"
                variant="outlined"
                color="error"
                startIcon={<ClearAllIcon />}
                onClick={() => dispatch(clearArbitraryPoints())}
              >
                Clear All
              </Button>
              <Tooltip title="Export points to JSON">
                <IconButton size="small" onClick={handleExportPoints}>
                  <DownloadIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Import points from JSON">
                <IconButton size="small" onClick={handleImportPoints}>
                  <UploadIcon />
                </IconButton>
              </Tooltip>
              <Chip
                label={arbState.running ? (arbState.paused ? 'Paused' : 'Scanning') : 'Idle'}
                color={arbState.running ? (arbState.paused ? 'warning' : 'success') : 'default'}
                size="small"
              />
            </Box>
          </Paper>

          {/* Scan Controls */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Arbitrary Scan Control
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={3}>
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<PlayArrowIcon />}
                  onClick={handleStart}
                  fullWidth
                  disabled={pointsList.length === 0}
                >
                  Start
                </Button>
              </Grid>
              <Grid item xs={3}>
                <Button
                  variant="contained"
                  color="warning"
                  startIcon={<PauseIcon />}
                  onClick={handlePause}
                  fullWidth
                  disabled={!arbState.running || arbState.paused}
                >
                  Pause
                </Button>
              </Grid>
              <Grid item xs={3}>
                <Button
                  variant="contained"
                  color="info"
                  startIcon={<PlayArrowIcon />}
                  onClick={handleResume}
                  fullWidth
                  disabled={!arbState.paused}
                >
                  Resume
                </Button>
              </Grid>
              <Grid item xs={3}>
                <Button
                  variant="contained"
                  color="error"
                  startIcon={<StopIcon />}
                  onClick={handleStop}
                  fullWidth
                >
                  Stop
                </Button>
              </Grid>
            </Grid>

            {/* Laser Trigger Mode */}
            <FormControl size="small" sx={{ mt: 2, minWidth: 160 }}>
              <InputLabel>Laser Trigger</InputLabel>
              <Select
                value={arbState.laserTrigger}
                label="Laser Trigger"
                onChange={(e) => dispatch(setLaserTrigger(e.target.value))}
              >
                <MenuItem value="AUTO">AUTO</MenuItem>
                <MenuItem value="HIGH">HIGH (always on)</MenuItem>
                <MenuItem value="LOW">LOW (always off)</MenuItem>
                <MenuItem value="CONTINUOUS">CONTINUOUS</MenuItem>
              </Select>
            </FormControl>
          </Paper>

          {/* Calibration Button */}
          <Paper sx={{ p: 2 }}>
            <Button
              variant="outlined"
              color="secondary"
              startIcon={<GpsFixedIcon />}
              onClick={() => dispatch(startCalibration())}
              disabled={calibration.active}
              fullWidth
            >
              Start Affine Calibration (3 Points)
            </Button>
          </Paper>
        </Grid>

        {/* ===== RIGHT: Points Table + Settings ===== */}
        <Grid item xs={12} md={5}>
          {/* Global Defaults */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Global Defaults
            </Typography>
            <Grid container spacing={1} alignItems="center">
              <Grid item xs={5}>
                <TextField
                  label="Default Dwell (µs)"
                  type="number"
                  size="small"
                  fullWidth
                  value={arbState.defaultDwellUs}
                  onChange={(e) => dispatch(setDefaultDwellUs(Number(e.target.value)))}
                  inputProps={{ min: 1 }}
                />
              </Grid>
              <Grid item xs={3}>
                <Button size="small" variant="outlined" onClick={() => dispatch(applyDefaultDwellToAll())} fullWidth>
                  Apply All
                </Button>
              </Grid>
            </Grid>
            <Grid container spacing={1} alignItems="center" sx={{ mt: 1 }}>
              <Grid item xs={5}>
                <TextField
                  label="Default Intensity"
                  type="number"
                  size="small"
                  fullWidth
                  value={arbState.defaultIntensity}
                  onChange={(e) => dispatch(setDefaultIntensity(Number(e.target.value)))}
                  inputProps={{ min: 0, max: 255 }}
                />
              </Grid>
              <Grid item xs={3}>
                <Button size="small" variant="outlined" onClick={() => dispatch(applyDefaultIntensityToAll())} fullWidth>
                  Apply All
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {/* Points Table */}
          <Paper sx={{ p: 1, mb: 2, maxHeight: 350, overflow: 'auto' }}>
            <Typography variant="subtitle2" gutterBottom sx={{ px: 1 }}>
              Points ({pointsList.length})
            </Typography>
            <TableContainer>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>#</TableCell>
                    <TableCell>X</TableCell>
                    <TableCell>Y</TableCell>
                    <TableCell>Dwell (µs)</TableCell>
                    <TableCell>Intensity</TableCell>
                    <TableCell></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {pointsList.map((pt, i) => (
                    <TableRow key={i} hover>
                      <TableCell>{i + 1}</TableCell>
                      <TableCell>
                        <TextField
                          type="number"
                          size="small"
                          value={pt.x}
                          onChange={(e) => {
                            const val = Math.max(0, Math.min(4095, Number(e.target.value) || 0));
                            dispatch(updateArbitraryPoint({ index: i, x: val }));
                          }}
                          inputProps={{ min: 0, max: 4095, style: { width: 70 } }}
                          variant="standard"
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          type="number"
                          size="small"
                          value={pt.y}
                          onChange={(e) => {
                            const val = Math.max(0, Math.min(4095, Number(e.target.value) || 0));
                            dispatch(updateArbitraryPoint({ index: i, y: val }));
                          }}
                          inputProps={{ min: 0, max: 4095, style: { width: 70 } }}
                          variant="standard"
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          type="number"
                          size="small"
                          value={pt.dwell_us}
                          onChange={(e) => dispatch(updateArbitraryPoint({ index: i, dwell_us: Number(e.target.value) }))}
                          inputProps={{ min: 1, style: { width: 70 } }}
                          variant="standard"
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          type="number"
                          size="small"
                          value={pt.laser_intensity ?? 128}
                          onChange={(e) => dispatch(updateArbitraryPoint({ index: i, laser_intensity: Number(e.target.value) }))}
                          inputProps={{ min: 0, max: 255, style: { width: 60 } }}
                          variant="standard"
                        />
                      </TableCell>
                      <TableCell>
                        <IconButton size="small" color="error" onClick={() => dispatch(removeArbitraryPoint(i))}>
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                  {/* Manual add-point row */}
                  <TableRow sx={{ backgroundColor: 'action.hover' }}>
                    <TableCell>
                      <AddCircleIcon fontSize="small" color="primary" />
                    </TableCell>
                    <TableCell>
                      <TextField
                        type="number"
                        size="small"
                        value={newPointX}
                        onChange={(e) => setNewPointX(Math.max(0, Math.min(4095, Number(e.target.value) || 0)))}
                        inputProps={{ min: 0, max: 4095, style: { width: 70 } }}
                        variant="standard"
                        placeholder="X"
                      />
                    </TableCell>
                    <TableCell>
                      <TextField
                        type="number"
                        size="small"
                        value={newPointY}
                        onChange={(e) => setNewPointY(Math.max(0, Math.min(4095, Number(e.target.value) || 0)))}
                        inputProps={{ min: 0, max: 4095, style: { width: 70 } }}
                        variant="standard"
                        placeholder="Y"
                      />
                    </TableCell>
                    <TableCell>
                      <TextField
                        type="number"
                        size="small"
                        value={newPointDwell}
                        onChange={(e) => setNewPointDwell(Number(e.target.value) || 1)}
                        inputProps={{ min: 1, style: { width: 70 } }}
                        variant="standard"
                        placeholder="µs"
                      />
                    </TableCell>
                    <TableCell>
                      <TextField
                        type="number"
                        size="small"
                        value={newPointIntensity}
                        onChange={(e) => setNewPointIntensity(Math.max(0, Math.min(255, Number(e.target.value) || 0)))}
                        inputProps={{ min: 0, max: 255, style: { width: 60 } }}
                        variant="standard"
                        placeholder="Int"
                      />
                    </TableCell>
                    <TableCell>
                      <Tooltip title="Add point manually">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => {
                            dispatch(addArbitraryPoint({
                              x: newPointX,
                              y: newPointY,
                              dwell_us: newPointDwell,
                              laser_intensity: newPointIntensity,
                            }));
                          }}
                        >
                          <AddCircleIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                  {pointsList.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Typography variant="body2" color="text.secondary">
                          Click on the camera view or use the form above to add points
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>

          {/* Advanced: Affine Transform Editor */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <TuneIcon sx={{ mr: 1 }} />
              <Typography variant="subtitle2">Affine Transform (Advanced)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                Camera → Galvo mapping: Galvo = A · Camera + b
              </Typography>
              <Grid container spacing={1}>
                {['a11', 'a12', 'tx', 'a21', 'a22', 'ty'].map((key) => (
                  <Grid item xs={4} key={key}>
                    <TextField
                      label={key}
                      type="number"
                      size="small"
                      fullWidth
                      value={affine[key]}
                      onChange={(e) => dispatch(setAffineParam({ param: key, value: parseFloat(e.target.value) || 0 }))}
                      inputProps={{ step: 0.1 }}
                    />
                  </Grid>
                ))}
              </Grid>
              <Box sx={{ display: 'flex', gap: 1, mt: 2, flexWrap: 'wrap' }}>
                <Button size="small" variant="outlined" startIcon={<SaveIcon />} onClick={handleSaveAffine}>
                  Save to Backend
                </Button>
                <Button size="small" variant="outlined" onClick={handleLoadAffine}>
                  Load from Backend
                </Button>
                <Button size="small" variant="outlined" color="warning" startIcon={<RestartAltIcon />} onClick={handleResetAffine}>
                  Reset to Identity
                </Button>
              </Box>
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>

      {/* Calibration Wizard Dialog */}
      {calibration.active && (
        <GalvoAffineCalibrationWizard />
      )}
    </Box>
  );
};

export default GalvoArbitraryPointsTab;
