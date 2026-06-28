import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  Paper,
  Alert,
  CircularProgress,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  LinearProgress,
} from '@mui/material';

import LiveViewControlWrapper from '../../axon/LiveViewControlWrapper';
import apiPixelCalibrationControllerCalibrateStageAffine from '../../backendapi/apiPixelCalibrationControllerCalibrateStageAffine';
import apiPixelCalibrationControllerGetPendingCalibration from '../../backendapi/apiPixelCalibrationControllerGetPendingCalibration';
import apiPixelCalibrationControllerApplyPendingCalibration from '../../backendapi/apiPixelCalibrationControllerApplyPendingCalibration';
import apiPixelCalibrationControllerDiscardPendingCalibration from '../../backendapi/apiPixelCalibrationControllerDiscardPendingCalibration';
import apiPixelCalibrationControllerGetAvailableDetectors from '../../backendapi/apiPixelCalibrationControllerGetAvailableDetectors';
import apiPixelCalibrationControllerGetCalibrationProgress from '../../backendapi/apiPixelCalibrationControllerGetCalibrationProgress';
import apiPixelCalibrationControllerStopCalibration from '../../backendapi/apiPixelCalibrationControllerStopCalibration';
import apiPixelCalibrationControllerMeasureBacklash from '../../backendapi/apiPixelCalibrationControllerMeasureBacklash';
import apiPixelCalibrationControllerApplyBacklash from '../../backendapi/apiPixelCalibrationControllerApplyBacklash';
import apiObjectiveControllerGetStatus from '../../backendapi/apiObjectiveControllerGetStatus';

/**
 * PixelCalibrationTab - Per-detector stage <-> camera affine calibration.
 *
 * Workflow:
 *  1. User picks a detector (by name) and configures the stage move pattern.
 *  2. Backend runs the calibration in a background thread.
 *  3. Frontend polls `getPendingCalibration` until a result appears.
 *  4. User reviews / edits pixel size, flips and the affine matrix, then
 *     either Applies (persists + pushes to detector) or Discards.
 */
const PixelCalibrationTab = () => {
  // --- selection / parameters ---
  const [detectorName, setDetectorName] = useState('');
  const [availableDetectors, setAvailableDetectors] = useState([]);
  // Objective: 'current' = active slot, '0', '1' (string so it survives the round-trip).
  const [objectiveId, setObjectiveId] = useState('current');
  const [stepSizeUm, setStepSizeUm] = useState(100.0);
  const [pattern, setPattern] = useState('cross');
  const [nSteps, setNSteps] = useState(4);
  const [backlashUm, setBacklashUm] = useState(50.0);

  // --- backlash measurement ---
  const [blAxis, setBlAxis] = useState('X');
  const [blStepUm, setBlStepUm] = useState(20.0);
  const [blNSteps, setBlNSteps] = useState(8);
  const [blRunning, setBlRunning] = useState(false);
  const [blResult, setBlResult] = useState(null);
  const [blManualUm, setBlManualUm] = useState('');
  const [blApplying, setBlApplying] = useState(false);

  // --- objective info ---
  const [objectiveInfo, setObjectiveInfo] = useState(null);

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

  // Load the list of available detectors once on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await apiPixelCalibrationControllerGetAvailableDetectors();
        if (cancelled) return;
        const names = resp?.detectorNames || [];
        setAvailableDetectors(names);
        if (names.length > 0) {
          setDetectorName((prev) => prev || names[0]);
        }
      } catch (_e) {
        // leave list empty; user can still type by selecting from an empty dropdown.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // --- run state ---
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [progress, setProgress] = useState(null); // {step,total,message} while running
  const [stopping, setStopping] = useState(false);

  // --- pending calibration review state ---
  const [pending, setPending] = useState(null); // raw pending payload from backend
  const [editPixelSizeX, setEditPixelSizeX] = useState('');
  const [editPixelSizeY, setEditPixelSizeY] = useState('');
  const [editAffineMatrix, setEditAffineMatrix] = useState('');

  const pollRef = useRef(null);

  // --- helpers ---------------------------------------------------------------

  const populateReviewFromPending = (data) => {
    if (!data) return;
    const sx = Number(data?.metrics?.scale_x_um_per_pixel ?? 0);
    const sy = Number(data?.metrics?.scale_y_um_per_pixel ?? 0);
    setEditPixelSizeX(Math.abs(sx).toString());
    setEditPixelSizeY(Math.abs(sy).toString());
    setEditAffineMatrix(JSON.stringify(data?.affine_matrix ?? [], null, 2));
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  // Poll progress + pending while a run is in flight.
  useEffect(() => {
    if (!loading || !detectorName) return undefined;
    pollRef.current = setInterval(async () => {
      // Live progress for the bar / status line.
      try {
        const prog = await apiPixelCalibrationControllerGetCalibrationProgress();
        if (prog?.success) setProgress(prog);
      } catch (_e) { /* ignore */ }

      try {
        const resp = await apiPixelCalibrationControllerGetPendingCalibration(
          detectorName,
          objectiveId,
        );
        const data = resp?.pending ?? resp;
        if (data && data.error) {
          // The background thread failed – surface it instead of polling forever.
          setError(`Calibration failed: ${data.error}`);
          setStatus('');
          setLoading(false);
          setStopping(false);
          setProgress(null);
          stopPolling();
          return;
        }
        if (data && (data.affine_matrix || data.metrics)) {
          setPending(data);
          populateReviewFromPending(data);
          setLoading(false);
          setStopping(false);
          setProgress(null);
          setStatus('Calibration complete – please review and apply.');
          stopPolling();
        }
      } catch (err) {
        // keep polling unless backend explicitly errors
      }
    }, 1000);
    return stopPolling;
  }, [loading, detectorName, objectiveId]);

  // If the user stopped the run (no pending result will appear), detect the
  // "not running anymore" transition via progress and exit the loading state.
  useEffect(() => {
    if (!loading || !stopping) return;
    if (progress && progress.running === false) {
      setLoading(false);
      setStopping(false);
      setStatus(progress.message || 'Calibration stopped.');
      setProgress(null);
      stopPolling();
    }
  }, [loading, stopping, progress]);

  // --- actions ---------------------------------------------------------------

  const handleCalibrate = async () => {
    if (!detectorName) {
      setError('Please enter a detector name.');
      return;
    }
    try {
      setError('');
      setStatus('Calibration started in background – polling for result…');
      setPending(null);
      setProgress(null);
      setStopping(false);
      setLoading(true);

      const resp = await apiPixelCalibrationControllerCalibrateStageAffine({
        detectorName,
        objectiveId,
        stepSizeUm,
        pattern,
        nSteps,
        backlashUm,
      });
      // The backend rejects synchronously on bad camera intensity / busy state.
      if (resp && resp.success === false) {
        setLoading(false);
        setStatus('');
        setError(resp.error || 'Calibration could not be started.');
      }
    } catch (err) {
      setLoading(false);
      setError(`Failed to start calibration: ${err.message}`);
      setStatus('');
    }
  };

  const handleStop = async () => {
    try {
      setStopping(true);
      setStatus('Stopping calibration…');
      await apiPixelCalibrationControllerStopCalibration();
    } catch (err) {
      setError(`Failed to stop: ${err.message}`);
    }
  };

  const handleMeasureBacklash = async () => {
    if (!detectorName) {
      setError('Please select a detector.');
      return;
    }
    try {
      setError('');
      setBlResult(null);
      setBlRunning(true);
      setStatus(`Measuring ${blAxis} backlash — the stage will scan back and forth…`);
      const resp = await apiPixelCalibrationControllerMeasureBacklash({
        axis: blAxis,
        stepSizeUm: blStepUm,
        nSteps: blNSteps,
        detectorName,
        applyToStage: false,
      });
      if (resp && resp.success === false) {
        setError(resp.error || resp.message || 'Backlash measurement failed.');
        setStatus('');
      } else {
        setBlResult(resp);
        setBlManualUm(Number(resp.backlash_um).toFixed(1));
        setStatus(`Measured ${resp.axis} backlash: ${Number(resp.backlash_um).toFixed(1)} µm — review, then Apply.`);
      }
    } catch (err) {
      setError(`Failed to measure backlash: ${err.message}`);
      setStatus('');
    } finally {
      setBlRunning(false);
    }
  };

  const handleApplyBacklash = async (axis, um) => {
    const value = Number(um);
    if (!Number.isFinite(value)) {
      setError('Enter a numeric backlash value.');
      return;
    }
    try {
      setError('');
      setBlApplying(true);
      const resp = await apiPixelCalibrationControllerApplyBacklash({ axis, backlashUm: value });
      if (resp && resp.success === false) {
        setError(resp.message || resp.error || 'Could not apply backlash.');
      } else {
        setStatus(resp.message || `Backlash ${value.toFixed(1)} µm applied to ${axis}.`);
      }
    } catch (err) {
      setError(`Failed to apply backlash: ${err.message}`);
    } finally {
      setBlApplying(false);
    }
  };

  const handleApply = async () => {
    try {
      setError('');
      // Parse edited matrix
      let affineMatrix = null;
      try {
        affineMatrix = JSON.parse(editAffineMatrix);
      } catch (e) {
        setError(`Affine matrix is not valid JSON: ${e.message}`);
        return;
      }

      // Flip is purely calibration-derived: keep the sign the calibration
      // measured and only let the user adjust the pixel-size magnitude.
      const measuredSx = Number(pending?.metrics?.scale_x_um_per_pixel ?? 0);
      const measuredSy = Number(pending?.metrics?.scale_y_um_per_pixel ?? 0);
      const sx = Math.abs(parseFloat(editPixelSizeX)) * (measuredSx < 0 ? -1 : 1);
      const sy = Math.abs(parseFloat(editPixelSizeY)) * (measuredSy < 0 ? -1 : 1);

      const metrics = {
        ...(pending?.metrics ?? {}),
        scale_x_um_per_pixel: sx,
        scale_y_um_per_pixel: sy,
      };

      const resp = await apiPixelCalibrationControllerApplyPendingCalibration({
        detectorName,
        objectiveId,
        affineMatrix,
        metrics,
      });

      setStatus(resp?.message || 'Calibration applied to detector.');
      setPending(null);
    } catch (err) {
      setError(`Failed to apply calibration: ${err.message}`);
    }
  };

  const handleDiscard = async () => {
    try {
      setError('');
      await apiPixelCalibrationControllerDiscardPendingCalibration(detectorName, objectiveId);
      setPending(null);
      setStatus('Pending calibration discarded.');
    } catch (err) {
      setError(`Failed to discard calibration: ${err.message}`);
    }
  };

  // --- render ----------------------------------------------------------------

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Left: live detector view (sticky so it stays visible while the right column scrolls) */}
        <Grid item xs={12} md={7}>
          <Box sx={{ position: { xs: 'static', md: 'sticky' }, top: { md: 16 } }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Detector Live View
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Live view of the active detector. Flip / pixel-size are applied
              automatically once a calibration is approved.
            </Typography>
            <Box
              sx={{
                border: '1px solid #ddd',
                borderRadius: 2,
                overflow: 'hidden',
                minHeight: 400,
                maxHeight: 600,
                backgroundColor: '#000',
              }}
            >
              <LiveViewControlWrapper useFastMode={true} />
            </Box>
          </Paper>
          </Box>
        </Grid>

        {/* Right: parameters + review */}
        <Grid item xs={12} md={5}>
          {status && (
            <Alert severity="info" sx={{ mb: 2 }} onClose={() => setStatus('')}>
              {status}
            </Alert>
          )}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          {/* Calibration parameters */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Stage / Camera Affine Calibration
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Calibrates the camera ↔ stage transform for one detector.
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Detector</InputLabel>
              <Select
                value={detectorName}
                label="Detector"
                onChange={(e) => setDetectorName(e.target.value)}
              >
                {availableDetectors.length === 0 && (
                  <MenuItem value="" disabled>
                    <em>(no detectors available)</em>
                  </MenuItem>
                )}
                {availableDetectors.map((name) => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Objective</InputLabel>
              <Select
                value={objectiveId}
                label="Objective"
                onChange={(e) => setObjectiveId(e.target.value)}
              >
                <MenuItem value="current">Current</MenuItem>
                <MenuItem value="0">Objective 0</MenuItem>
                <MenuItem value="1">Objective 1</MenuItem>
              </Select>
            </FormControl>

            {/* Active objective info */}
            {objectiveInfo && (
              <Alert severity="info" sx={{ mb: 2 }}>
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

            <TextField
              label="Step size (µm)"
              type="number"
              value={stepSizeUm}
              onChange={(e) => setStepSizeUm(parseFloat(e.target.value))}
              fullWidth
              sx={{ mb: 2 }}
            />

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Pattern</InputLabel>
              <Select
                value={pattern}
                label="Pattern"
                onChange={(e) => setPattern(e.target.value)}
              >
                <MenuItem value="cross">Cross (4 points)</MenuItem>
                <MenuItem value="grid">Grid (NxN points)</MenuItem>
                <MenuItem value="star">Star</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Number of steps"
              type="number"
              value={nSteps}
              onChange={(e) => setNSteps(parseInt(e.target.value, 10))}
              fullWidth
              sx={{ mb: 2 }}
            />

            <TextField
              label="Backlash compensation (µm)"
              type="number"
              value={backlashUm}
              onChange={(e) => setBacklashUm(parseFloat(e.target.value) || 0)}
              fullWidth
              sx={{ mb: 2 }}
              helperText="Pre-move distance in X and Y to eliminate backlash (0 = disabled)"
              inputProps={{ step: 10, min: 0 }}
            />

            {!loading ? (
              <Button
                variant="contained"
                color="primary"
                onClick={handleCalibrate}
                disabled={!detectorName}
                fullWidth
                size="large"
              >
                Start calibration
              </Button>
            ) : (
              <Button
                variant="contained"
                color="error"
                onClick={handleStop}
                disabled={stopping}
                fullWidth
                size="large"
              >
                {stopping ? (
                  <>
                    <CircularProgress size={22} sx={{ mr: 1, color: 'inherit' }} />
                    Stopping…
                  </>
                ) : (
                  'Stop calibration'
                )}
              </Button>
            )}

            {/* Progress bar while a run is in flight */}
            {loading && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress
                  variant={progress && progress.total > 0 ? 'determinate' : 'indeterminate'}
                  value={
                    progress && progress.total > 0
                      ? Math.min(100, Math.round((progress.step / progress.total) * 100))
                      : undefined
                  }
                />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                  {progress && progress.total > 0
                    ? `Step ${progress.step}/${progress.total}${progress.message ? ` – ${progress.message}` : ''}`
                    : (progress?.message || 'Working…')}
                </Typography>
              </Box>
            )}
          </Paper>

          {/* Backlash measurement + apply */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Backlash
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Camera-tracked reversing scan to measure one axis&apos; backlash, then
              apply it to the stage (UC2 motor.py compensates it on every reversal).
              Measure with the stage&apos;s own backlash compensation disabled
              (config backlash = 0), otherwise this reports the residual.
            </Typography>

            <Grid container spacing={2}>
              <Grid item xs={4}>
                <FormControl fullWidth>
                  <InputLabel>Axis</InputLabel>
                  <Select
                    value={blAxis}
                    label="Axis"
                    onChange={(e) => setBlAxis(e.target.value)}
                  >
                    <MenuItem value="X">X</MenuItem>
                    <MenuItem value="Y">Y</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={4}>
                <TextField
                  label="Step (µm)"
                  type="number"
                  value={blStepUm}
                  onChange={(e) => setBlStepUm(parseFloat(e.target.value) || 0)}
                  fullWidth
                  inputProps={{ step: 5, min: 1 }}
                />
              </Grid>
              <Grid item xs={4}>
                <TextField
                  label="Steps / dir"
                  type="number"
                  value={blNSteps}
                  onChange={(e) => setBlNSteps(parseInt(e.target.value, 10) || 1)}
                  fullWidth
                  inputProps={{ step: 1, min: 2 }}
                />
              </Grid>
            </Grid>

            <Button
              variant="outlined"
              color="primary"
              onClick={handleMeasureBacklash}
              disabled={blRunning || blApplying || loading || !detectorName}
              fullWidth
              sx={{ mt: 2 }}
            >
              {blRunning ? (
                <>
                  <CircularProgress size={20} sx={{ mr: 1, color: 'inherit' }} />
                  Measuring…
                </>
              ) : (
                'Measure backlash'
              )}
            </Button>

            {blResult && (
              <Alert
                severity={Number(blResult.quality_min) >= 0.2 ? 'success' : 'warning'}
                sx={{ mt: 2 }}
                onClose={() => setBlResult(null)}
                action={(
                  <Button
                    color="inherit"
                    size="small"
                    disabled={blApplying}
                    onClick={() => handleApplyBacklash(blResult.axis, blResult.backlash_um)}
                  >
                    Apply to stage
                  </Button>
                )}
              >
                <Typography variant="body2">
                  <strong>
                    {blResult.axis} backlash: {Number(blResult.backlash_um).toFixed(1)} µm
                  </strong>
                  <br />
                  Scale {Number(blResult.scale_px_per_um).toFixed(3)} px/µm · residual{' '}
                  {Number(blResult.residual_px_zero_backlash).toFixed(2)}→
                  {Number(blResult.residual_px).toFixed(2)} px · min corr{' '}
                  {Number(blResult.quality_min).toFixed(2)}
                </Typography>
              </Alert>
            )}

            <Divider sx={{ my: 2 }}>or set manually</Divider>

            <Grid container spacing={1} alignItems="center">
              <Grid item xs={7}>
                <TextField
                  label={`Backlash ${blAxis} (µm)`}
                  type="number"
                  value={blManualUm}
                  onChange={(e) => setBlManualUm(e.target.value)}
                  fullWidth
                  size="small"
                  inputProps={{ step: 5, min: 0 }}
                />
              </Grid>
              <Grid item xs={5}>
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  disabled={blApplying || blManualUm === ''}
                  onClick={() => handleApplyBacklash(blAxis, blManualUm)}
                >
                  {blApplying ? (
                    <CircularProgress size={20} sx={{ color: 'inherit' }} />
                  ) : (
                    'Apply'
                  )}
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {/* Pending calibration review */}
          {pending && (
            <Paper sx={{ p: 2, mb: 2, border: '2px solid', borderColor: 'warning.main' }}>
              <Typography variant="h6" gutterBottom>
                Review pending calibration
              </Typography>
              <Alert severity="warning" sx={{ mb: 2 }}>
                Verify the values below. Nothing is applied to the detector
                until you press <b>Apply</b>.
              </Alert>

              {pending?.metrics?.validation_ok === false && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  Quality check failed: {pending.metrics.validation_message || 'low-quality calibration'}.
                  The pixel size may be unreliable — consider re-running with more
                  contrast/exposure or a larger step size.
                </Alert>
              )}
              {pending?.metrics?.validation_ok === true && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Quality check passed
                  {pending?.metrics?.mean_correlation != null
                    ? ` (mean correlation ${Number(pending.metrics.mean_correlation).toFixed(2)})`
                    : ''}.
                </Alert>
              )}

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Pixel size X (µm)"
                    type="number"
                    value={editPixelSizeX}
                    onChange={(e) => setEditPixelSizeX(e.target.value)}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Pixel size Y (µm)"
                    type="number"
                    value={editPixelSizeY}
                    onChange={(e) => setEditPixelSizeY(e.target.value)}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12}>
                  <Alert severity="info" icon={false} sx={{ py: 0.5 }}>
                    <Typography variant="body2">
                      <strong>Image flip (measured by calibration):</strong>{' '}
                      X&nbsp;{Number(pending?.metrics?.scale_x_um_per_pixel ?? 0) < 0 ? 'flipped' : 'normal'},{' '}
                      Y&nbsp;{Number(pending?.metrics?.scale_y_um_per_pixel ?? 0) < 0 ? 'flipped' : 'normal'}.
                      Flip is calibration-owned and applied automatically on Apply.
                    </Typography>
                  </Alert>
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                Affine matrix (editable JSON)
              </Typography>
              <TextField
                value={editAffineMatrix}
                onChange={(e) => setEditAffineMatrix(e.target.value)}
                fullWidth
                multiline
                minRows={5}
                maxRows={12}
                sx={{
                  mb: 2,
                  fontFamily: 'monospace',
                  '& textarea': { fontFamily: 'monospace', fontSize: '0.85em' },
                }}
              />

              {pending.metrics && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Raw metrics
                  </Typography>
                  <pre
                    style={{
                      fontSize: '0.8em',
                      overflow: 'auto',
                      maxHeight: 160,
                      backgroundColor: 'secondary',
                      padding: 8,
                      borderRadius: 4,
                      margin: 0,
                    }}
                  >
                    {JSON.stringify(pending.metrics, null, 2)}
                  </pre>
                </Box>
              )}

              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Button
                    variant="contained"
                    color="success"
                    fullWidth
                    onClick={handleApply}
                  >
                    Apply
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    variant="outlined"
                    color="error"
                    fullWidth
                    onClick={handleDiscard}
                  >
                    Discard
                  </Button>
                </Grid>
              </Grid>
            </Paper>
          )}

          {/* Help box */}
          <Paper sx={{ p: 2, backgroundColor: 'secondary.main' }}>
            <Typography variant="subtitle2" gutterBottom>
              How it works
            </Typography>
            <Typography variant="body2">
              1. Stage moves through the chosen pattern.<br />
              2. Pixel shifts are measured per move.<br />
              3. An affine matrix and pixel size are computed.<br />
              4. Result is staged as <i>pending</i> for your review.<br />
              5. Apply pushes flip + pixel size to the detector and saves to disk.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PixelCalibrationTab;
