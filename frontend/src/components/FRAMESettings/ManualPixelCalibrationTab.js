import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import { useSelector } from 'react-redux';
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Chip,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  MyLocation as CrosshairIcon,
  OpenWith as MoveIcon,
  Calculate as CalcIcon,
  CheckCircle as DoneIcon,
  Replay as ResetIcon,
  Save as SaveIcon,
} from '@mui/icons-material';

import LiveViewControlWrapper from '../../axon/LiveViewControlWrapper';
import * as liveStreamSlice from '../../state/slices/LiveStreamSlice';
import apiPositionerControllerMovePositioner from '../../backendapi/apiPositionerControllerMovePositioner';
import apiPixelCalibrationControllerManualFourPointCalibration from '../../backendapi/apiPixelCalibrationControllerManualFourPointCalibration';
import apiPixelCalibrationControllerGetAvailableDetectors from '../../backendapi/apiPixelCalibrationControllerGetAvailableDetectors';
import apiPixelCalibrationControllerGetCalibrationData from '../../backendapi/apiPixelCalibrationControllerGetCalibrationData';
import apiPixelCalibrationControllerSetCalibrationData from '../../backendapi/apiPixelCalibrationControllerSetCalibrationData';
import apiObjectiveControllerGetStatus from '../../backendapi/apiObjectiveControllerGetStatus';
import apiLiveViewControllerGetStreamParameters from '../../backendapi/apiLiveViewControllerGetStreamParameters';

/**
 * ManualPixelCalibrationTab – Interactive four-point affine calibration
 *
 * Computes a full affine matrix (scale, rotation, flip) from two pairs of
 * feature clicks – one pair per stage axis – exactly like the automated
 * calibration but with user-selected key points.
 *
 * Workflow:
 *   Phase 1 (X-axis):
 *     0. (Optional) Backlash pre-move along X
 *     1. Click feature → P_A1
 *     2. Move stage in X
 *     3. Click same feature → P_A2
 *   Phase 2 (Y-axis):
 *     4. Click feature → P_B1
 *     5. Move stage in Y
 *     6. Click same feature → P_B2
 *   Final:
 *     7. Calculate & save
 */
const ManualPixelCalibrationTab = () => {
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  // Authoritative subsampling factor fetched from the backend on tab open.
  // Falls back to Redux/streamSettings only if the backend call hasn't
  // returned yet (avoids the "factor = 1 by default" bug when the calibration
  // tab is opened before the user has touched any stream setting).
  const [backendSubsamplingFactor, setBackendSubsamplingFactor] = useState(null);

  const subsamplingFromRedux = useMemo(() => {
    const settings = liveStreamState.streamSettings;
    return settings?.jpeg?.subsampling?.factor ||
           settings?.jpeg?.subsampling_factor ||
           settings?.binary?.subsampling?.factor ||
           settings?.webrtc?.subsampling_factor || null;
  }, [liveStreamState.streamSettings]);

  const subsamplingFactor = backendSubsamplingFactor ?? subsamplingFromRedux ?? 1;

  // ---- calibration state ----
  const [activeStep, setActiveStep] = useState(0);
  const [pointA1, setPointA1] = useState(null);
  const [pointA2, setPointA2] = useState(null);
  const [pointB1, setPointB1] = useState(null);
  const [pointB2, setPointB2] = useState(null);
  const [imageDims, setImageDims] = useState(null);
  const [movementDistanceXUm, setMovementDistanceXUm] = useState(100);
  const [movementDistanceYUm, setMovementDistanceYUm] = useState(100);
  const [backlashDistanceUm, setBacklashDistanceUm] = useState(50);
  const [detectorName, setDetectorName] = useState('');
  const [availableDetectors, setAvailableDetectors] = useState([]);
  const [objectiveId, setObjectiveId] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const [editPixelSizeUm, setEditPixelSizeUm] = useState('');
  const [existingCalib, setExistingCalib] = useState(null);
  const [editLoading, setEditLoading] = useState(false);

  // --- objective info ---
  const [objectiveInfo, setObjectiveInfo] = useState(null);

  const overlayRef = useRef(null);

  // Fetch objective status on mount and when objectiveId changes
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await apiObjectiveControllerGetStatus();
        if (cancelled) return;
        setObjectiveInfo(resp);
      } catch (_e) { /* ignore */ }
    })();
    return () => { cancelled = true; };
  }, [objectiveId]);

  // Pull the *actual* subsampling factor from the backend, so we don't rely
  // on (potentially undefined) Redux state. Uses the currently-active
  // protocol if any, otherwise just reports the first non-null one found.
  const refreshBackendSubsampling = async () => {
    try {
      const resp = await apiLiveViewControllerGetStreamParameters();
      const activeProtocol = resp?.current_protocol;
      const protos = resp?.protocols || {};
      let factor = null;
      if (activeProtocol && protos[activeProtocol]?.subsampling_factor != null) {
        factor = Number(protos[activeProtocol].subsampling_factor);
      } else {
        for (const p of Object.keys(protos)) {
          if (protos[p]?.subsampling_factor != null) {
            factor = Number(protos[p].subsampling_factor);
            break;
          }
        }
      }
      if (factor && factor > 0) setBackendSubsamplingFactor(factor);
    } catch (_e) {
      /* keep Redux fallback */
    }
  };

  useEffect(() => {
    refreshBackendSubsampling();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  useEffect(() => {
    let cancelled = false;
    if (!detectorName) { setExistingCalib(null); setEditPixelSizeUm(''); return undefined; }
    (async () => {
      try {
        const resp = await apiPixelCalibrationControllerGetCalibrationData(detectorName, objectiveId);
        if (cancelled) return;
        if (resp?.success) {
          setExistingCalib(resp);
          const sx = Number(resp?.metrics?.scale_x_um_per_pixel ?? 0);
          const sy = Number(resp?.metrics?.scale_y_um_per_pixel ?? 0);
          const avg = (Math.abs(sx) + Math.abs(sy)) / 2;
          setEditPixelSizeUm(avg ? avg.toFixed(4) : '');
        } else {
          setExistingCalib(null);
          setEditPixelSizeUm('');
        }
      } catch (_e) {
        setExistingCalib(null);
        setEditPixelSizeUm('');
      }
    })();
    return () => { cancelled = true; };
  }, [detectorName, objectiveId]);

  const handleReset = () => {
    setActiveStep(0);
    setPointA1(null);
    setPointA2(null);
    setPointB1(null);
    setPointB2(null);
    setResult(null);
    setStatus('');
    setError('');
  };

  // ---- step 0: backlash compensation ----
  const handleBacklashMove = async () => {
    try {
      setLoading(true);
      setError('');
      setStatus(`Backlash pre-move: ${backlashDistanceUm} µm along X…`);
      await apiPositionerControllerMovePositioner({
        axis: 'X',
        dist: backlashDistanceUm,
        isAbsolute: false,
        isBlocking: true,
      });
      setStatus('Backlash done. Click a recognisable feature in the image.');
      setActiveStep(1);
    } catch (err) {
      setError(`Backlash move failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBacklashMoveY = async () => {
    try {
      setLoading(true);
      setError('');
      setStatus(`Backlash pre-move: ${backlashDistanceUm} µm along Y…`);
      await apiPositionerControllerMovePositioner({
        axis: 'Y',
        dist: backlashDistanceUm,
        isAbsolute: false,
        isBlocking: true,
      });
      setStatus('Backlash done. Click a recognisable feature in the image.');
      setActiveStep(5);
    } catch (err) {
      setError(`Backlash move failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }
  
  // ---- image click handler (dispatches to the right step) ----
  const handleImageClick = useCallback(
    (pixelX, pixelY, imageWidth, imageHeight) => {
      const dims = { width: imageWidth, height: imageHeight };
      if (!imageDims) setImageDims(dims);

      if (activeStep === 1) {
        setPointA1({ x: pixelX, y: pixelY });
        setStatus(`P1 marked at (${Math.round(pixelX)}, ${Math.round(pixelY)}). Now move stage in X.`);
        setActiveStep(2);
      } else if (activeStep === 3) {
        setPointA2({ x: pixelX, y: pixelY });
        setStatus(`P2 marked. X-axis done. Now click a feature for the Y-axis measurement.`);
        setActiveStep(4);
      } else if (activeStep === 5) {
        setPointB1({ x: pixelX, y: pixelY });
        setStatus(`P3 marked at (${Math.round(pixelX)}, ${Math.round(pixelY)}). Now move stage in Y.`);
        setActiveStep(6);
      } else if (activeStep === 7) {
        setPointB2({ x: pixelX, y: pixelY });
        setStatus('P4 marked. Ready to calculate the full affine calibration.');
        setActiveStep(8);
      }
    },
    [activeStep, imageDims],
  );

  // ---- step 2: move stage X ----
  const handleMoveStageX = async () => {
    try {
      setLoading(true);
      setError('');
      setStatus(`Moving stage ${movementDistanceXUm} µm along X…`);
      await apiPositionerControllerMovePositioner({
        axis: 'X',
        dist: movementDistanceXUm,
        isAbsolute: false,
        isBlocking: true,
      });
      setStatus('Stage moved in X. Click the SAME feature again.');
      setActiveStep(3);
    } catch (err) {
      setError(`X stage move failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ---- step 5: move stage Y ----
  const handleMoveStageY = async () => {
    try {
      setLoading(true);
      setError('');
      setStatus(`Moving stage ${movementDistanceYUm} µm along Y…`);
      await apiPositionerControllerMovePositioner({
        axis: 'Y',
        dist: movementDistanceYUm,
        isAbsolute: false,
        isBlocking: true,
      });
      setStatus('Stage moved in Y. Click the SAME feature again.');
      setActiveStep(7);
    } catch (err) {
      setError(`Y stage move failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ---- step 8: compute calibration (does NOT save yet) ----
  const handleCalculate = async () => {
    if (!pointA1 || !pointA2 || !pointB1 || !pointB2) return;
    try {
      setLoading(true);
      setError('');
      setStatus('Computing full affine calibration…');

      // Always re-confirm the live preview subsampling factor right before
      // computing — otherwise a stale Redux value would silently undercount
      // the displayed pixel size when the user changed it elsewhere.
      let effectiveSubsampling = subsamplingFactor;
      try {
        const resp = await apiLiveViewControllerGetStreamParameters();
        const activeProtocol = resp?.current_protocol;
        const protos = resp?.protocols || {};
        let factor = null;
        if (activeProtocol && protos[activeProtocol]?.subsampling_factor != null) {
          factor = Number(protos[activeProtocol].subsampling_factor);
        } else {
          for (const p of Object.keys(protos)) {
            if (protos[p]?.subsampling_factor != null) {
              factor = Number(protos[p].subsampling_factor);
              break;
            }
          }
        }
        if (factor && factor > 0) {
          effectiveSubsampling = factor;
          setBackendSubsamplingFactor(factor);
        }
      } catch (_e) {
        /* fall back to the previously known factor */
      }

      const res = await apiPixelCalibrationControllerManualFourPointCalibration({
        pointA1X: pointA1.x,
        pointA1Y: pointA1.y,
        pointA2X: pointA2.x,
        pointA2Y: pointA2.y,
        movementDistanceXUm,
        pointB1X: pointB1.x,
        pointB1Y: pointB1.y,
        pointB2X: pointB2.x,
        pointB2Y: pointB2.y,
        movementDistanceYUm,
        detectorName: detectorName || undefined,
        objectiveId: objectiveId || undefined,
        previewSubsamplingFactor: effectiveSubsampling,
      });

      if (res.success) {
        setResult(res);
        setStatus('Calibration computed. Review the results below, then Accept or Discard.');
        setActiveStep(9);
      } else {
        setError(res.error || 'Calibration failed');
      }
    } catch (err) {
      setError(`Calibration failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ---- step 9: accept (persist) computed calibration ----
  const handleAcceptCalibration = async () => {
    if (!result) return;
    try {
      setLoading(true);
      setError('');

      // Build affine matrix and metrics from the computed result
      const affineMatrix = result.affineMatrix || result.affine_matrix;
      const metrics = result.metrics || {
        scale_x_um_per_pixel: result.scaleXUmPerPixel,
        scale_y_um_per_pixel: result.scaleYUmPerPixel,
      };

      const resp = await apiPixelCalibrationControllerSetCalibrationData({
        detectorName: detectorName || undefined,
        objectiveId: objectiveId || undefined,
        affineMatrix,
        metrics,
      });

      if (resp?.success) {
        setStatus(resp.message || 'Calibration applied and saved to detector.');
        setActiveStep(10);
      } else {
        setError(resp?.error || 'Failed to apply calibration');
      }
    } catch (err) {
      setError(`Failed to apply calibration: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ---- discard computed calibration ----
  const handleDiscardCalibration = () => {
    setResult(null);
    setStatus('Calibration discarded. You can re-run the workflow.');
    setActiveStep(8);
  };

  // ---- overlay ----
  const pointColor = (label) => {
    if (label === 'A1' || label === 'A2') return '#00ff00';
    return '#4488ff';
  };
  const renderPoint = (pt, label) => {
    if (!pt) return null;
    const c = pointColor(label);
    return (
      <g key={label}>
        <line x1={pt.x - 12} y1={pt.y} x2={pt.x + 12} y2={pt.y} stroke={c} strokeWidth="2" />
        <line x1={pt.x} y1={pt.y - 12} x2={pt.x} y2={pt.y + 12} stroke={c} strokeWidth="2" />
        <circle cx={pt.x} cy={pt.y} r="7" fill="none" stroke={c} strokeWidth="2" />
        <text x={pt.x + 12} y={pt.y - 8} fill={c} fontSize="13" fontWeight="bold">{label}</text>
      </g>
    );
  };

  const renderOverlay = () => {
    if (!pointA1 && !pointA2 && !pointB1 && !pointB2) return null;
    return (
      <svg
        ref={overlayRef}
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 }}
        viewBox={imageDims ? `0 0 ${imageDims.width} ${imageDims.height}` : '0 0 100 100'}
        preserveAspectRatio="xMidYMid meet"
      >
        {renderPoint(pointA1, 'A1')}
        {renderPoint(pointA2, 'A2')}
        {renderPoint(pointB1, 'B1')}
        {renderPoint(pointB2, 'B2')}
        {pointA1 && pointA2 && (
          <line x1={pointA1.x} y1={pointA1.y} x2={pointA2.x} y2={pointA2.y}
            stroke="#00ff00" strokeWidth="1.5" strokeDasharray="6,4" opacity="0.7" />
        )}
        {pointB1 && pointB2 && (
          <line x1={pointB1.x} y1={pointB1.y} x2={pointB2.x} y2={pointB2.y}
            stroke="#4488ff" strokeWidth="1.5" strokeDasharray="6,4" opacity="0.7" />
        )}
      </svg>
    );
  };

  const steps = [
    'Backlash compensation in X',
    'Mark feature (P1 before X move)',
    'Move stage in X',
    'Mark same feature (P2 after X move)',
    'Backlash compensation in Y (optional)',
    'Mark feature (P3 before Y move)',
    'Move stage in Y',
    'Mark same feature (P4 after Y move)',
    'Compute affine calibration',
    'Review & accept',
  ];

  // Clickable steps — highlight which steps expect a click
  const clickSteps = new Set([1, 3, 5, 7]);

  return (
    <Box>
      <Grid container spacing={3}>
        {/* ---- Left: Live View ---- */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Detector Camera
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {clickSteps.has(activeStep)
                ? activeStep <= 3
                  ? 'Click on a feature in the image (X-axis measurement)'
                  : 'Click on a feature in the image (Y-axis measurement)'
                : 'Live preview'}
            </Typography>

            <Box sx={{ border: '1px solid #ddd', borderRadius: 2, overflow: 'hidden', minHeight: 400, maxHeight: 500, backgroundColor: '#000', position: 'relative' }}>
              <LiveViewControlWrapper
                useFastMode={true}
                onClick={handleImageClick}
                onImageLoad={(w, h) => setImageDims({ width: w, height: h })}
                overlayContent={renderOverlay()}
                enableStageMovement={false}
              />
            </Box>

            {/* Coordinate chips */}
            <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {pointA1 && <Chip icon={<CrosshairIcon />} label={`A1: (${Math.round(pointA1.x)}, ${Math.round(pointA1.y)})`} color="success" size="small" />}
              {pointA2 && <Chip icon={<CrosshairIcon />} label={`A2: (${Math.round(pointA2.x)}, ${Math.round(pointA2.y)})`} color="success" size="small" variant="outlined" />}
              {pointB1 && <Chip icon={<CrosshairIcon />} label={`B1: (${Math.round(pointB1.x)}, ${Math.round(pointB1.y)})`} color="primary" size="small" />}
              {pointB2 && <Chip icon={<CrosshairIcon />} label={`B2: (${Math.round(pointB2.x)}, ${Math.round(pointB2.y)})`} color="primary" size="small" variant="outlined" />}
              {pointA1 && pointA2 && (() => {
                const d = Math.sqrt((pointA2.x - pointA1.x) ** 2 + (pointA2.y - pointA1.y) ** 2);
                return <Chip label={`ΔX = ${Math.round(d * subsamplingFactor)} sensor px`} color="warning" size="small" />;
              })()}
              {pointB1 && pointB2 && (() => {
                const d = Math.sqrt((pointB2.x - pointB1.x) ** 2 + (pointB2.y - pointB1.y) ** 2);
                return <Chip label={`ΔY = ${Math.round(d * subsamplingFactor)} sensor px`} color="warning" size="small" />;
              })()}
              {subsamplingFactor > 1 && <Chip label={`Subsampling ×${subsamplingFactor}`} variant="outlined" size="small" />}
            </Box>
          </Paper>
        </Grid>

        {/* ---- Right: Controls ---- */}
        <Grid item xs={12} md={5}>
          {status && <Alert severity="info" sx={{ mb: 2 }} onClose={() => setStatus('')}>{status}</Alert>}
          {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>{error}</Alert>}

          {/* Parameters */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>Calibration Parameters</Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Detector</InputLabel>
              <Select value={detectorName} label="Detector" onChange={(e) => setDetectorName(e.target.value)}>
                {availableDetectors.length === 0 && <MenuItem value="" disabled><em>(no detectors)</em></MenuItem>}
                {availableDetectors.map((name) => <MenuItem key={name} value={name}>{name}</MenuItem>)}
              </Select>
            </FormControl>

            <TextField label="X Movement Distance (µm)" type="number" value={movementDistanceXUm}
              onChange={(e) => setMovementDistanceXUm(parseFloat(e.target.value) || 0)}
              fullWidth sx={{ mb: 2 }} helperText="Known stage travel along X" />

            <TextField label="Y Movement Distance (µm)" type="number" value={movementDistanceYUm}
              onChange={(e) => setMovementDistanceYUm(parseFloat(e.target.value) || 0)}
              fullWidth sx={{ mb: 2 }} helperText="Known stage travel along Y" />

            <FormControl fullWidth sx={{ mb: 1 }}>
              <InputLabel>Objective</InputLabel>
              <Select value={objectiveId} label="Objective" onChange={(e) => setObjectiveId(e.target.value)}>
                <MenuItem value="">Current</MenuItem>
                <MenuItem value="0">Objective 0</MenuItem>
                <MenuItem value="1">Objective 1</MenuItem>
              </Select>
            </FormControl>

            {/* Active objective info */}
            {objectiveInfo && (
              <Alert severity="info" sx={{ mt: 1 }}>
                <Typography variant="body2">
                  <strong>Active objective:</strong>{' '}
                  {objectiveInfo.currentObjective != null
                    ? (objectiveInfo.availableObjectivesNames?.[objectiveInfo.currentObjective] || `#${objectiveInfo.currentObjective}`)
                    : 'None'}
                  {objectiveInfo.availableObjectivePixelSizes?.[objectiveInfo.currentObjective] > 0 && (
                    <> &mdash; Pixel size: {objectiveInfo.availableObjectivePixelSizes[objectiveInfo.currentObjective].toFixed(4)} µm/px</>
                  )}
                  {objectiveInfo.availableObjectiveMagnifications?.[objectiveInfo.currentObjective] > 0 && (
                    <> &mdash; {objectiveInfo.availableObjectiveMagnifications[objectiveInfo.currentObjective]}×</>
                  )}
                  {objectiveInfo.availableObjectiveNAs?.[objectiveInfo.currentObjective] > 0 && (
                    <> (NA {objectiveInfo.availableObjectiveNAs[objectiveInfo.currentObjective]})</>
                  )}
                </Typography>
              </Alert>
            )}
          </Paper>

          {/* Editable calibration override */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>Edit Calibration</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {existingCalib
                ? `Current: ${detectorName} @ ${existingCalib.objectiveId}`
                : `No calibration stored for ${detectorName || '(none)'}`}
            </Typography>
            <TextField label="Pixel Size (µm/px)" type="number" value={editPixelSizeUm}
              onChange={(e) => setEditPixelSizeUm(e.target.value)}
              fullWidth size="small" sx={{ mb: 1 }} inputProps={{ step: 0.001, min: 0 }} />
            <Button variant="outlined" size="small"
              startIcon={editLoading ? <CircularProgress size={16} /> : <SaveIcon />}
              disabled={editLoading || !detectorName || !editPixelSizeUm}
              onClick={async () => {
                const px = parseFloat(editPixelSizeUm);
                if (!isFinite(px) || px <= 0) { setError('Pixel size must be positive'); return; }
                setEditLoading(true); setError('');
                try {
                  const resp = await apiPixelCalibrationControllerSetCalibrationData({
                    detectorName, objectiveId: objectiveId || undefined,
                    affineMatrix: [[px, 0.0, 0.0], [0.0, px, 0.0]],
                    metrics: { scale_x_um_per_pixel: px, scale_y_um_per_pixel: px, rotation_deg: 0.0, method: 'manual_edit', quality: 'manual' },
                  });
                  if (resp?.success) {
                    setStatus(resp.message || 'Calibration saved');
                    setExistingCalib({ success: true, detectorName, objectiveId: resp.objectiveId, metrics: resp.metrics, affineMatrix: resp.affineMatrix });
                  } else { setError(resp?.error || 'Save failed'); }
                } catch (e) { setError(`Save failed: ${e.message}`); }
                finally { setEditLoading(false); }
              }}>Save calibration</Button>
          </Paper>

          {/* Stepper workflow */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>Four-Point Calibration Workflow</Typography>

            <Stepper activeStep={activeStep} orientation="vertical">
              {/* Step 0 – backlash */}
              <Step>
                <StepLabel StepIconProps={{ icon: <MoveIcon color={activeStep === 0 ? 'primary' : 'inherit'} /> }}>
                  {steps[0]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Move stage <strong>{backlashDistanceUm} µm</strong> along X to compensate backlash.
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
                    <TextField label="Backlash (µm)" type="number" size="small" value={backlashDistanceUm}
                      onChange={(e) => setBacklashDistanceUm(parseFloat(e.target.value) || 0)} sx={{ width: 120 }} />
                    <Button variant="contained" size="small"
                      startIcon={loading ? <CircularProgress size={18} /> : <MoveIcon />}
                      onClick={handleBacklashMove} disabled={loading}>
                      {loading ? 'Moving…' : 'Move'}
                    </Button>
                    <Button variant="outlined" size="small" onClick={() => setActiveStep(1)}>Skip</Button>
                  </Box>
                </StepContent>
              </Step>

              {/* Step 1 – mark P_A1 */}
              <Step>
                <StepLabel StepIconProps={{ icon: <CrosshairIcon color={activeStep === 1 ? 'primary' : 'inherit'} /> }}>
                  {steps[1]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2">
                    Click on a clearly identifiable feature in the live image.
                  </Typography>
                </StepContent>
              </Step>

              {/* Step 2 – move X */}
              <Step>
                <StepLabel StepIconProps={{ icon: <MoveIcon color={activeStep === 2 ? 'primary' : 'inherit'} /> }}>
                  {steps[2]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Move stage <strong>{movementDistanceXUm} µm</strong> along <strong>X</strong>.
                  </Typography>
                  <Button variant="contained"
                    startIcon={loading ? <CircularProgress size={18} /> : <MoveIcon />}
                    onClick={handleMoveStageX} disabled={loading}>
                    {loading ? 'Moving…' : `Move ${movementDistanceXUm} µm X`}
                  </Button>
                </StepContent>
              </Step>

              {/* Step 3 – mark P_A2 */}
              <Step>
                <StepLabel StepIconProps={{ icon: <CrosshairIcon color={activeStep === 3 ? 'primary' : 'inherit'} /> }}>
                  {steps[3]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2">
                    Click the <strong>same</strong> feature you marked in step 1.
                  </Typography>
                </StepContent>
              </Step>

              {/* Optional Step – backlash for Y */}
              <Step>
                  <StepLabel StepIconProps={{ icon: <MoveIcon color={activeStep === 4 ? 'primary' : 'inherit'} /> }}>
                  {steps[4]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Move stage <strong>{backlashDistanceUm} µm</strong> along Y to compensate backlash.
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
                    <TextField label="Backlash (µm)" type="number" size="small" value={backlashDistanceUm}
                      onChange={(e) => setBacklashDistanceUm(parseFloat(e.target.value) || 0)} sx={{ width: 120 }} />
                    <Button variant="contained" size="small"
                      startIcon={loading ? <CircularProgress size={18} /> : <MoveIcon />}
                      onClick={handleBacklashMoveY} disabled={loading}>
                      {loading ? 'Moving…' : 'Move'}
                    </Button>
                    <Button variant="outlined" size="small" onClick={() => setActiveStep(5)}>Skip</Button>
                  </Box>
                </StepContent>
              </Step>

              {/* Step 4 – mark P_B1 */}
              <Step>
                <StepLabel StepIconProps={{ icon: <CrosshairIcon color={activeStep === 5 ? 'primary' : 'inherit'} /> }}>
                  {steps[5]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Click a feature for the Y-axis measurement (can be the same or a different one).
                  </Typography>
                </StepContent>
              </Step>

              {/* Step 6 – move Y */}
              <Step>
                <StepLabel StepIconProps={{ icon: <MoveIcon color={activeStep === 6 ? 'primary' : 'inherit'} /> }}>
                  {steps[6]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Move stage <strong>{movementDistanceYUm} µm</strong> along <strong>Y</strong>.
                  </Typography>
                  <Button variant="contained"
                    startIcon={loading ? <CircularProgress size={18} /> : <MoveIcon />}
                    onClick={handleMoveStageY} disabled={loading}>
                    {loading ? 'Moving…' : `Move ${movementDistanceYUm} µm Y`}
                  </Button>
                </StepContent>
              </Step>

              {/* Step 7 – mark P_B2 */}
              <Step>
                <StepLabel StepIconProps={{ icon: <CrosshairIcon color={activeStep === 7 ? 'primary' : 'inherit'} /> }}>
                  {steps[7]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2">
                    Click the <strong>same</strong> feature you marked in step 4.
                  </Typography>
                </StepContent>
              </Step>

              {/* Step 8 – compute */}
              <Step>
                <StepLabel StepIconProps={{ icon: <CalcIcon color={activeStep === 8 ? 'primary' : 'inherit'} /> }}>
                  {steps[8]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Review the four points and press <em>Compute</em> to calculate the affine calibration.
                    Nothing is saved until you accept in the next step.
                  </Typography>
                  <Button variant="contained" color="primary"
                    startIcon={loading ? <CircularProgress size={18} /> : <CalcIcon />}
                    onClick={handleCalculate} disabled={loading}>
                    {loading ? 'Computing…' : 'Compute calibration'}
                  </Button>
                </StepContent>
              </Step>

              {/* Step 9 – review & accept */}
              <Step>
                <StepLabel StepIconProps={{ icon: <DoneIcon color={activeStep === 9 ? 'success' : 'inherit'} /> }}>
                  {steps[9]}
                </StepLabel>
                <StepContent>
                  {result && (
                    <Box>
                      <Alert severity="warning" sx={{ mb: 2 }}>
                        Verify the computed values below. Nothing is applied to the detector
                        until you press <strong>Accept</strong>.
                      </Alert>
                      <Typography variant="body2" sx={{ mb: 0.5 }}>
                        <strong>Pixel size:</strong> {result.pixelSizeUm?.toFixed(4)} µm/px
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 0.5 }}>
                        Scale X: {result.scaleXUmPerPixel?.toFixed(4)} µm/px
                        {result.scaleXUmPerPixel < 0 && ' (flipped)'}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 0.5 }}>
                        Scale Y: {result.scaleYUmPerPixel?.toFixed(4)} µm/px
                        {result.scaleYUmPerPixel < 0 && ' (flipped)'}
                      </Typography>
                      {result.metrics?.rotation_deg != null && (
                        <Typography variant="body2" sx={{ mb: 0.5 }}>
                          Rotation: {result.metrics.rotation_deg.toFixed(1)}°
                        </Typography>
                      )}
                      {result.metrics?.condition_number != null && (
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          Condition number: {result.metrics.condition_number.toFixed(2)}
                        </Typography>
                      )}
                      <Grid container spacing={1} sx={{ mt: 1 }}>
                        <Grid item xs={6}>
                          <Button variant="contained" color="success" fullWidth
                            startIcon={loading ? <CircularProgress size={18} /> : <DoneIcon />}
                            onClick={handleAcceptCalibration} disabled={loading}>
                            {loading ? 'Saving…' : 'Accept & Save'}
                          </Button>
                        </Grid>
                        <Grid item xs={6}>
                          <Button variant="outlined" color="error" fullWidth
                            onClick={handleDiscardCalibration} disabled={loading}>
                            Discard
                          </Button>
                        </Grid>
                      </Grid>
                    </Box>
                  )}
                </StepContent>
              </Step>
            </Stepper>

            {activeStep === 10 && (
              <Box sx={{ mt: 2, textAlign: 'center' }}>
                <DoneIcon color="success" sx={{ fontSize: 48 }} />
                <Typography variant="subtitle1" color="success.main">
                  Affine calibration applied and saved!
                </Typography>
              </Box>
            )}

            <Divider sx={{ my: 2 }} />
            <Button variant="outlined" startIcon={<ResetIcon />} onClick={handleReset} fullWidth>
              Reset / Start Over
            </Button>
          </Paper>

          {/* Info box */}
          <Paper sx={{ p: 2, mb: 2, backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(144,202,249,0.08)' : '#f0f7ff' }}>
            <Typography variant="subtitle2" gutterBottom>How it works</Typography>
            <Typography variant="body2">
              This calibration measures <strong>both</strong> stage axes to compute a full
              affine matrix — encoding pixel size, rotation <em>and</em> flip, just like the
              automatic calibration.<br /><br />
              <strong>X-axis:</strong> Mark a feature (P1), move stage in X, mark it again (P2).<br />
              <strong>Y-axis:</strong> Mark a feature (P3), move stage in Y, mark it again (P4).<br />
              The two pixel-shift vectors fully determine the 2×2 transform.
            </Typography>
            {subsamplingFactor > 1 && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                <em>
                  The live preview is subsampled by &times;{subsamplingFactor}.
                  Coordinates are automatically scaled to full sensor resolution.
                </em>
              </Typography>
            )}
          </Paper>

          {/* Result display */}
          {result && (
            <Paper sx={{ p: 2, backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : '#f5f5f5' }}>
              <Typography variant="h6" gutterBottom>Result</Typography>
              <Alert severity="success" sx={{ mb: 2 }}>{result.message}</Alert>

              <Box sx={{ mb: 1 }}>
                <Typography variant="subtitle2">
                  Pixel Size: <strong>{result.pixelSizeUm?.toFixed(4)} µm/px</strong>
                </Typography>
              </Box>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                  Scale X: {result.scaleXUmPerPixel?.toFixed(4)} µm/px
                  {result.scaleXUmPerPixel < 0 && ' (flipped)'}
                </Typography>
              </Box>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                  Scale Y: {result.scaleYUmPerPixel?.toFixed(4)} µm/px
                  {result.scaleYUmPerPixel < 0 && ' (flipped)'}
                </Typography>
              </Box>
              {result.metrics?.rotation_deg != null && (
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2">
                    Rotation: {result.metrics.rotation_deg.toFixed(1)}°
                  </Typography>
                </Box>
              )}
              {result.metrics?.condition_number != null && (
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2">
                    Condition number: {result.metrics.condition_number.toFixed(2)}
                  </Typography>
                </Box>
              )}
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                  Movement: {movementDistanceXUm} µm X, {movementDistanceYUm} µm Y
                </Typography>
              </Box>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2">Objective: {result.objectiveId}</Typography>
              </Box>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default ManualPixelCalibrationTab;
