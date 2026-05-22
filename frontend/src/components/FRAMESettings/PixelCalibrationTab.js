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
  FormControlLabel,
  Checkbox,
  Divider,
} from '@mui/material';

import LiveViewControlWrapper from '../../axon/LiveViewControlWrapper';
import apiPixelCalibrationControllerCalibrateStageAffine from '../../backendapi/apiPixelCalibrationControllerCalibrateStageAffine';
import apiPixelCalibrationControllerGetPendingCalibration from '../../backendapi/apiPixelCalibrationControllerGetPendingCalibration';
import apiPixelCalibrationControllerApplyPendingCalibration from '../../backendapi/apiPixelCalibrationControllerApplyPendingCalibration';
import apiPixelCalibrationControllerDiscardPendingCalibration from '../../backendapi/apiPixelCalibrationControllerDiscardPendingCalibration';
import apiPixelCalibrationControllerGetAvailableDetectors from '../../backendapi/apiPixelCalibrationControllerGetAvailableDetectors';
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
  // Objective: '' = current, '0', '1' (string so it survives the round-trip).
  const [objectiveId, setObjectiveId] = useState('');
  const [stepSizeUm, setStepSizeUm] = useState(100.0);
  const [pattern, setPattern] = useState('cross');
  const [nSteps, setNSteps] = useState(4);
  const [backlashUm, setBacklashUm] = useState(50.0);

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

  // --- pending calibration review state ---
  const [pending, setPending] = useState(null); // raw pending payload from backend
  const [editPixelSizeX, setEditPixelSizeX] = useState('');
  const [editPixelSizeY, setEditPixelSizeY] = useState('');
  const [editFlipX, setEditFlipX] = useState(false);
  const [editFlipY, setEditFlipY] = useState(false);
  const [editAffineMatrix, setEditAffineMatrix] = useState('');

  const pollRef = useRef(null);

  // --- helpers ---------------------------------------------------------------

  const populateReviewFromPending = (data) => {
    if (!data) return;
    const sx = Number(data?.metrics?.scale_x_um_per_pixel ?? 0);
    const sy = Number(data?.metrics?.scale_y_um_per_pixel ?? 0);
    setEditPixelSizeX(Math.abs(sx).toString());
    setEditPixelSizeY(Math.abs(sy).toString());
    setEditFlipX(sx < 0);
    setEditFlipY(sy < 0);
    setEditAffineMatrix(JSON.stringify(data?.affine_matrix ?? [], null, 2));
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  // Poll while loading and no pending yet
  useEffect(() => {
    if (!loading || !detectorName) return undefined;
    pollRef.current = setInterval(async () => {
      try {
        const resp = await apiPixelCalibrationControllerGetPendingCalibration(
          detectorName,
          objectiveId,
        );
        const data = resp?.pending ?? resp;
        if (data && (data.affine_matrix || data.metrics)) {
          setPending(data);
          populateReviewFromPending(data);
          setLoading(false);
          setStatus('Calibration complete – please review and apply.');
          stopPolling();
        }
      } catch (err) {
        // keep polling unless backend explicitly errors
      }
    }, 2000);
    return stopPolling;
  }, [loading, detectorName, objectiveId]);

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
      setLoading(true);

      await apiPixelCalibrationControllerCalibrateStageAffine({
        detectorName,
        objectiveId,
        stepSizeUm,
        pattern,
        nSteps,
        backlashUm,
      });
    } catch (err) {
      setLoading(false);
      setError(`Failed to start calibration: ${err.message}`);
      setStatus('');
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

      // Re-encode flips into signed scale values (the backend convention)
      const sx = Math.abs(parseFloat(editPixelSizeX)) * (editFlipX ? -1 : 1);
      const sy = Math.abs(parseFloat(editPixelSizeY)) * (editFlipY ? -1 : 1);

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
        {/* Left: live detector view */}
        <Grid item xs={12} md={7}>
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
                <MenuItem value="">Current</MenuItem>
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

            <Button
              variant="contained"
              color="primary"
              onClick={handleCalibrate}
              disabled={loading || !detectorName}
              fullWidth
              size="large"
            >
              {loading ? (
                <>
                  <CircularProgress size={22} sx={{ mr: 1 }} />
                  Running…
                </>
              ) : (
                'Start calibration'
              )}
            </Button>
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
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={editFlipX}
                        onChange={(e) => setEditFlipX(e.target.checked)}
                      />
                    }
                    label="Flip X"
                  />
                </Grid>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={editFlipY}
                        onChange={(e) => setEditFlipY(e.target.checked)}
                      />
                    }
                    label="Flip Y"
                  />
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
