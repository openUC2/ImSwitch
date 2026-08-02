import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useSelector } from 'react-redux';
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  Chip,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ArrowUpward,
  ArrowDownward,
  ArrowBack,
  ArrowForward,
  Replay as ResetIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';

import LiveViewControlWrapper from '../../axon/LiveViewControlWrapper';
import * as liveStreamSlice from '../../state/slices/LiveStreamSlice';
import apiPositionerControllerMovePositioner from '../../backendapi/apiPositionerControllerMovePositioner';
import apiPixelCalibrationControllerGetAvailableDetectors from '../../backendapi/apiPixelCalibrationControllerGetAvailableDetectors';
import apiPixelCalibrationControllerGetCalibrationData from '../../backendapi/apiPixelCalibrationControllerGetCalibrationData';
import apiLiveViewControllerGetStreamParameters from '../../backendapi/apiLiveViewControllerGetStreamParameters';

/**
 * VerifyCalibrationTab – visually sanity-check the *current* pixel-size calibration.
 *
 * Mark a recognisable feature in the live image, then nudge the stage by a known
 * distance (default 50 µm) up / down / left / right. The overlay draws:
 *   • the fixed crosshair where you marked the feature,
 *   • an "expected distance" ring (radius = move distance / pixel size), and
 *   • a predicted crosshair (direction from the stored affine matrix).
 * If the calibration is correct the feature lands on the predicted marker / ring.
 */
const VerifyCalibrationTab = () => {
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  const [detectorName, setDetectorName] = useState('');
  const [availableDetectors, setAvailableDetectors] = useState([]);
  const [objectiveId, setObjectiveId] = useState('current');
  const [moveDistanceUm, setMoveDistanceUm] = useState(50);

  const [calib, setCalib] = useState(null); // { pixelSizeUm, affine2x2 }
  const [markedPoint, setMarkedPoint] = useState(null); // {x,y} in preview px
  const [imageDims, setImageDims] = useState(null);
  const [netMove, setNetMove] = useState({ dx: 0, dy: 0 }); // µm since the mark
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const [backendSubsampling, setBackendSubsampling] = useState(null);
  const subsamplingFromRedux = useMemo(() => {
    const s = liveStreamState.streamSettings;
    return s?.jpeg?.subsampling?.factor || s?.jpeg?.subsampling_factor ||
           s?.binary?.subsampling?.factor || s?.webrtc?.subsampling_factor || null;
  }, [liveStreamState.streamSettings]);
  const subsamplingFactor = backendSubsampling ?? subsamplingFromRedux ?? 1;

  // --- load detectors once ---
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await apiPixelCalibrationControllerGetAvailableDetectors();
        if (cancelled) return;
        const names = resp?.detectorNames || [];
        setAvailableDetectors(names);
        if (names.length > 0) setDetectorName((prev) => prev || names[0]);
      } catch (_e) { /* ignore */ }
    })();
    return () => { cancelled = true; };
  }, []);

  // --- pull the real subsampling factor from the backend ---
  useEffect(() => {
    (async () => {
      try {
        const resp = await apiLiveViewControllerGetStreamParameters();
        const active = resp?.current_protocol;
        const protos = resp?.protocols || {};
        let factor = null;
        if (active && protos[active]?.subsampling_factor != null) {
          factor = Number(protos[active].subsampling_factor);
        } else {
          for (const p of Object.keys(protos)) {
            if (protos[p]?.subsampling_factor != null) { factor = Number(protos[p].subsampling_factor); break; }
          }
        }
        if (factor && factor > 0) setBackendSubsampling(factor);
      } catch (_e) { /* keep redux fallback */ }
    })();
  }, []);

  // --- load the calibration to verify whenever detector / objective changes ---
  useEffect(() => {
    let cancelled = false;
    if (!detectorName) { setCalib(null); return undefined; }
    (async () => {
      try {
        const resp = await apiPixelCalibrationControllerGetCalibrationData(detectorName, objectiveId);
        if (cancelled) return;
        if (resp?.success) {
          const m = resp.metrics || {};
          const sx = Number(m.scale_x_um_per_pixel ?? 0);
          const sy = Number(m.scale_y_um_per_pixel ?? 0);
          const pixelSizeUm = (Math.abs(sx) + Math.abs(sy)) / 2 || null;
          const A = resp.affineMatrix; // 2x3 [[a,b,tx],[c,d,ty]]
          const affine2x2 = (Array.isArray(A) && A.length >= 2)
            ? [[Number(A[0][0]), Number(A[0][1])], [Number(A[1][0]), Number(A[1][1])]]
            : null;
          setCalib({ pixelSizeUm, affine2x2, scaleX: sx, scaleY: sy });
        } else {
          setCalib(null);
        }
      } catch (_e) { setCalib(null); }
    })();
    return () => { cancelled = true; };
  }, [detectorName, objectiveId]);

  const handleImageClick = useCallback((pixelX, pixelY, w, h) => {
    if (!imageDims) setImageDims({ width: w, height: h });
    setMarkedPoint({ x: pixelX, y: pixelY });
    setNetMove({ dx: 0, dy: 0 });
    setStatus(`Marked (${Math.round(pixelX)}, ${Math.round(pixelY)}). Nudge the stage to check the calibration.`);
  }, [imageDims]);

  const nudge = async (dx, dy) => {
    try {
      setLoading(true); setError('');
      if (dx !== 0) {
        await apiPositionerControllerMovePositioner({ axis: 'X', dist: dx, isAbsolute: false, isBlocking: true });
      }
      if (dy !== 0) {
        await apiPositionerControllerMovePositioner({ axis: 'Y', dist: dy, isAbsolute: false, isBlocking: true });
      }
      setNetMove((p) => ({ dx: p.dx + dx, dy: p.dy + dy }));
      setStatus('Stage moved. Compare the feature to the predicted marker / ring.');
    } catch (err) {
      setError(`Stage move failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const returnToStart = async () => {
    if (netMove.dx === 0 && netMove.dy === 0) return;
    await nudge(-netMove.dx, -netMove.dy);
    setNetMove({ dx: 0, dy: 0 });
    setStatus('Returned to the marked position.');
  };

  const clearMark = () => {
    setMarkedPoint(null);
    setNetMove({ dx: 0, dy: 0 });
    setStatus('');
  };

  // --- predicted pixel shift (preview px) for the accumulated stage move ---
  // affine A maps sensor-pixel -> stage(µm): stage = A @ pixel. So a stage move
  // of (dx,dy) µm shifts the image content by A⁻¹ @ (dx,dy) sensor px, which is
  // (A⁻¹ @ move) / subsampling preview px.
  const predicted = useMemo(() => {
    if (!markedPoint || (netMove.dx === 0 && netMove.dy === 0)) return null;
    if (!calib) return null;
    const { affine2x2, pixelSizeUm } = calib;
    let shiftSensor = null;
    if (affine2x2) {
      const [[a, b], [c, d]] = affine2x2;
      const det = a * d - b * c;
      if (Math.abs(det) > 1e-9) {
        // A⁻¹ @ [dx,dy]
        shiftSensor = {
          x: (d * netMove.dx - b * netMove.dy) / det,
          y: (-c * netMove.dx + a * netMove.dy) / det,
        };
      }
    }
    if (!shiftSensor && pixelSizeUm) {
      // Fallback: magnitude only along the commanded axes (no rotation/flip info).
      shiftSensor = { x: netMove.dx / pixelSizeUm, y: netMove.dy / pixelSizeUm };
    }
    if (!shiftSensor) return null;
    const shiftPreview = { x: shiftSensor.x / subsamplingFactor, y: shiftSensor.y / subsamplingFactor };
    const radius = Math.sqrt(shiftPreview.x ** 2 + shiftPreview.y ** 2);
    return {
      point: { x: markedPoint.x + shiftPreview.x, y: markedPoint.y + shiftPreview.y },
      radius,
      expectedUm: Math.sqrt(netMove.dx ** 2 + netMove.dy ** 2),
    };
  }, [markedPoint, netMove, calib, subsamplingFactor]);

  const renderOverlay = () => {
    if (!markedPoint) return null;
    const c = '#00e5ff';
    const p = '#ffea00';
    return (
      <svg
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 }}
        viewBox={imageDims ? `0 0 ${imageDims.width} ${imageDims.height}` : '0 0 100 100'}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* marked feature crosshair */}
        <line x1={markedPoint.x - 14} y1={markedPoint.y} x2={markedPoint.x + 14} y2={markedPoint.y} stroke={c} strokeWidth="2" />
        <line x1={markedPoint.x} y1={markedPoint.y - 14} x2={markedPoint.x} y2={markedPoint.y + 14} stroke={c} strokeWidth="2" />
        <circle cx={markedPoint.x} cy={markedPoint.y} r="6" fill="none" stroke={c} strokeWidth="2" />
        {predicted && (
          <>
            {/* expected-distance ring */}
            <circle cx={markedPoint.x} cy={markedPoint.y} r={predicted.radius} fill="none" stroke={p} strokeWidth="1.5" strokeDasharray="5,4" opacity="0.8" />
            {/* predicted feature position */}
            <line x1={predicted.point.x - 12} y1={predicted.point.y} x2={predicted.point.x + 12} y2={predicted.point.y} stroke={p} strokeWidth="2" />
            <line x1={predicted.point.x} y1={predicted.point.y - 12} x2={predicted.point.x} y2={predicted.point.y + 12} stroke={p} strokeWidth="2" />
            <line x1={markedPoint.x} y1={markedPoint.y} x2={predicted.point.x} y2={predicted.point.y} stroke={p} strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
          </>
        )}
      </svg>
    );
  };

  const dirBtn = (icon, title, dx, dy) => (
    <Tooltip title={title}>
      <span>
        <IconButton color="primary" onClick={() => nudge(dx, dy)} disabled={loading || !markedPoint}>
          {icon}
        </IconButton>
      </span>
    </Tooltip>
  );

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Left: sticky live view */}
        <Grid item xs={12} md={7}>
          <Box sx={{ position: { xs: 'static', md: 'sticky' }, top: { md: 16 } }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Detector Camera</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                {markedPoint ? 'Nudge the stage and watch the feature vs. the yellow ring/marker.' : 'Click a recognisable feature to mark it.'}
              </Typography>
              <Box sx={{ border: '1px solid #ddd', borderRadius: 2, overflow: 'hidden', minHeight: 400, maxHeight: 520, backgroundColor: '#000', position: 'relative' }}>
                <LiveViewControlWrapper
                  useFastMode={true}
                  onClick={handleImageClick}
                  onImageLoad={(w, h) => setImageDims({ width: w, height: h })}
                  overlayContent={renderOverlay()}
                  enableStageMovement={false}
                  enableZoomPan={false}
                />
              </Box>
              <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {markedPoint && <Chip size="small" color="info" label={`Mark: (${Math.round(markedPoint.x)}, ${Math.round(markedPoint.y)})`} />}
                {(netMove.dx !== 0 || netMove.dy !== 0) && <Chip size="small" color="warning" label={`Net move: X ${netMove.dx} µm, Y ${netMove.dy} µm`} />}
                {predicted && <Chip size="small" variant="outlined" label={`Expected shift: ${predicted.radius.toFixed(1)} px`} />}
                {subsamplingFactor > 1 && <Chip size="small" variant="outlined" label={`Subsampling ×${subsamplingFactor}`} />}
              </Box>
            </Paper>
          </Box>
        </Grid>

        {/* Right: controls */}
        <Grid item xs={12} md={5}>
          {status && <Alert severity="info" sx={{ mb: 2 }} onClose={() => setStatus('')}>{status}</Alert>}
          {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>{error}</Alert>}

          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>Verify Calibration</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              A visual sanity check of the stored pixel size: mark a feature, move
              the stage a known distance, and confirm the feature lands on the
              predicted marker (yellow) and the expected-distance ring.
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Detector</InputLabel>
              <Select value={detectorName} label="Detector" onChange={(e) => setDetectorName(e.target.value)}>
                {availableDetectors.length === 0 && <MenuItem value="" disabled><em>(no detectors)</em></MenuItem>}
                {availableDetectors.map((n) => <MenuItem key={n} value={n}>{n}</MenuItem>)}
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Objective</InputLabel>
              <Select value={objectiveId} label="Objective" onChange={(e) => setObjectiveId(e.target.value)}>
                <MenuItem value="current">Current</MenuItem>
                <MenuItem value="0">Objective 0</MenuItem>
                <MenuItem value="1">Objective 1</MenuItem>
              </Select>
            </FormControl>

            {calib ? (
              <Alert severity={calib.pixelSizeUm ? 'success' : 'warning'} sx={{ mb: 2 }}>
                {calib.pixelSizeUm
                  ? <>Pixel size to verify: <strong>{calib.pixelSizeUm.toFixed(4)} µm/px</strong>
                      {(calib.scaleX < 0 || calib.scaleY < 0) && ' (flipped)'}</>
                  : 'Calibration found but has no pixel size.'}
              </Alert>
            ) : (
              <Alert severity="warning" sx={{ mb: 2 }}>
                No calibration stored for this detector / objective — run a calibration first.
              </Alert>
            )}

            <TextField
              label="Move distance (µm)"
              type="number"
              value={moveDistanceUm}
              onChange={(e) => setMoveDistanceUm(parseFloat(e.target.value) || 0)}
              fullWidth size="small" sx={{ mb: 2 }}
              inputProps={{ step: 10, min: 1 }}
            />

            {/* D-pad */}
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 1 }}>
              {dirBtn(<ArrowUpward />, `Move -${moveDistanceUm} µm in Y`, 0, -moveDistanceUm)}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                {dirBtn(<ArrowBack />, `Move +${moveDistanceUm} µm in X`, moveDistanceUm, 0)}
                {dirBtn(<ArrowForward />, `Move ${moveDistanceUm} µm in X`, -moveDistanceUm, 0)}
              </Box>
              {dirBtn(<ArrowDownward />, `Move +${moveDistanceUm} µm in Y`, 0, moveDistanceUm)}
            </Box>

            <Divider sx={{ my: 1.5 }} />
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button variant="outlined" startIcon={<ResetIcon />} onClick={returnToStart}
                disabled={loading || (netMove.dx === 0 && netMove.dy === 0)} fullWidth>
                Return to start
              </Button>
              <Button variant="outlined" color="secondary" startIcon={<ClearIcon />} onClick={clearMark}
                disabled={!markedPoint} fullWidth>
                Clear mark
              </Button>
            </Box>
          </Paper>

          <Paper sx={{ p: 2, backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(144,202,249,0.08)' : '#f0f7ff' }}>
            <Typography variant="subtitle2" gutterBottom>How to read it</Typography>
            <Typography variant="body2">
              <span style={{ color: '#00b8d4' }}>●</span> Cyan crosshair = where you marked the feature.<br />
              <span style={{ color: '#c8b900' }}>●</span> Yellow ring = expected distance for the move (= move ÷ pixel size).<br />
              <span style={{ color: '#c8b900' }}>✛</span> Yellow crosshair = predicted feature position (direction from the affine).<br /><br />
              If the calibration is correct, after the move the feature sits on the
              yellow marker. If it falls short / overshoots the ring, the pixel size
              is off; if it lands off-axis, the rotation/flip is off.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default VerifyCalibrationTab;
