import React, { useState, useCallback, useRef } from 'react';
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
} from '@mui/icons-material';

import LiveViewControlWrapper from '../../axon/LiveViewControlWrapper';
import apiPositionerControllerMovePositioner from '../../backendapi/apiPositionerControllerMovePositioner';
import apiPixelCalibrationControllerManualPixelSizeCalibration from '../../backendapi/apiPixelCalibrationControllerManualPixelSizeCalibration';

/**
 * ManualPixelCalibrationTab – Interactive two-point pixel-size calibration
 *
 * Workflow:
 *  1. Click a recognisable feature in the live image   → point1
 *  2. Move the stage by a KNOWN distance in X or Y
 *  3. Click the SAME feature again                      → point2
 *  4. Calculate and store: pixelSize = movementDistance / pixelDisplacement
 */
const ManualPixelCalibrationTab = () => {
  // ---- calibration state ----
  const [activeStep, setActiveStep] = useState(0);
  const [point1, setPoint1] = useState(null);            // { x, y }
  const [point2, setPoint2] = useState(null);             // { x, y }
  const [imageDims, setImageDims] = useState(null);       // { width, height }
  const [movementDistanceUm, setMovementDistanceUm] = useState(100);
  const [movementAxis, setMovementAxis] = useState('X');
  const [objectiveId, setObjectiveId] = useState('');     // empty = auto-detect
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  // Reference for overlay SVG dimensions
  const overlayRef = useRef(null);

  // ---- helper: reset everything ----
  const handleReset = () => {
    setActiveStep(0);
    setPoint1(null);
    setPoint2(null);
    setResult(null);
    setStatus('');
    setError('');
  };

  // ---- step 1: mark first point ----
  const handleImageClick = useCallback(
    (pixelX, pixelY, imageWidth, imageHeight) => {
      if (activeStep === 0) {
        // Mark point 1
        setPoint1({ x: pixelX, y: pixelY });
        setImageDims({ width: imageWidth, height: imageHeight });
        setStatus(`Point 1 marked at (${Math.round(pixelX)}, ${Math.round(pixelY)})`);
        setActiveStep(1);
      } else if (activeStep === 2) {
        // Mark point 2
        setPoint2({ x: pixelX, y: pixelY });
        setStatus(`Point 2 marked at (${Math.round(pixelX)}, ${Math.round(pixelY)})`);
        setActiveStep(3);
      }
    },
    [activeStep],
  );

  // ---- step 2: move stage ----
  const handleMoveStage = async () => {
    try {
      setLoading(true);
      setError('');
      setStatus(`Moving stage ${movementDistanceUm} µm along ${movementAxis}…`);
      await apiPositionerControllerMovePositioner({
        axis: movementAxis,
        dist: movementDistanceUm,
        isAbsolute: false,
        isBlocking: true,
      });
      setStatus(
        `Stage moved ${movementDistanceUm} µm along ${movementAxis}. ` +
        'Now click the SAME feature again in the image.'
      );
      setActiveStep(2);
    } catch (err) {
      setError(`Stage movement failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ---- step 4: calculate & save ----
  const handleCalculate = async () => {
    if (!point1 || !point2) return;

    try {
      setLoading(true);
      setError('');
      setStatus('Calculating pixel size…');

      const res = await apiPixelCalibrationControllerManualPixelSizeCalibration({
        point1X: point1.x,
        point1Y: point1.y,
        point2X: point2.x,
        point2Y: point2.y,
        movementDistanceUm,
        movementAxis,
        objectiveId: objectiveId || undefined,
      });

      if (res.success) {
        setResult(res);
        setStatus('');
        setActiveStep(4);
      } else {
        setError(res.error || 'Calibration failed');
      }
    } catch (err) {
      setError(`Calibration failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ---- overlay: render marked points on top of the live view ----
  const renderOverlay = () => {
    // Only show overlay when we have at least one point
    if (!point1 && !point2) return null;

    return (
      <svg
        ref={overlayRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          zIndex: 10,
        }}
        viewBox={
          imageDims
            ? `0 0 ${imageDims.width} ${imageDims.height}`
            : '0 0 100 100'
        }
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Point 1 marker */}
        {point1 && (
          <g>
            {/* crosshair */}
            <line x1={point1.x - 15} y1={point1.y} x2={point1.x + 15} y2={point1.y}
              stroke="#00ff00" strokeWidth="2" />
            <line x1={point1.x} y1={point1.y - 15} x2={point1.x} y2={point1.y + 15}
              stroke="#00ff00" strokeWidth="2" />
            <circle cx={point1.x} cy={point1.y} r="8"
              fill="none" stroke="#00ff00" strokeWidth="2" />
            <text x={point1.x + 14} y={point1.y - 10}
              fill="#00ff00" fontSize="14" fontWeight="bold">P1</text>
          </g>
        )}

        {/* Point 2 marker */}
        {point2 && (
          <g>
            <line x1={point2.x - 15} y1={point2.y} x2={point2.x + 15} y2={point2.y}
              stroke="#ff4444" strokeWidth="2" />
            <line x1={point2.x} y1={point2.y - 15} x2={point2.x} y2={point2.y + 15}
              stroke="#ff4444" strokeWidth="2" />
            <circle cx={point2.x} cy={point2.y} r="8"
              fill="none" stroke="#ff4444" strokeWidth="2" />
            <text x={point2.x + 14} y={point2.y - 10}
              fill="#ff4444" fontSize="14" fontWeight="bold">P2</text>
          </g>
        )}

        {/* Line between points */}
        {point1 && point2 && (
          <line x1={point1.x} y1={point1.y} x2={point2.x} y2={point2.y}
            stroke="#ffaa00" strokeWidth="1.5" strokeDasharray="6,4" />
        )}
      </svg>
    );
  };

  // Stepper labels
  const steps = [
    'Mark feature (point 1)',
    'Move stage',
    'Mark same feature (point 2)',
    'Calculate pixel size',
  ];

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
              {activeStep === 0
                ? 'Click on a recognisable feature in the image to set Point 1'
                : activeStep === 2
                  ? 'Click on the SAME feature again to set Point 2'
                  : 'Live preview'}
            </Typography>

            <Box
              sx={{
                border: '1px solid #ddd',
                borderRadius: 2,
                overflow: 'hidden',
                minHeight: 400,
                maxHeight: 500,
                backgroundColor: '#000',
                position: 'relative',
              }}
            >
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
              {point1 && (
                <Chip
                  icon={<CrosshairIcon />}
                  label={`P1: (${Math.round(point1.x)}, ${Math.round(point1.y)})`}
                  color="success"
                  size="small"
                />
              )}
              {point2 && (
                <Chip
                  icon={<CrosshairIcon />}
                  label={`P2: (${Math.round(point2.x)}, ${Math.round(point2.y)})`}
                  color="error"
                  size="small"
                />
              )}
              {point1 && point2 && (
                <Chip
                  label={`Δ = ${Math.round(
                    Math.sqrt(
                      (point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2,
                    ),
                  )} px`}
                  color="warning"
                  size="small"
                />
              )}
            </Box>
          </Paper>
        </Grid>

        {/* ---- Right: Controls ---- */}
        <Grid item xs={12} md={5}>
          {/* Messages */}
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

          {/* Parameters */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Calibration Parameters
            </Typography>

            <TextField
              label="Movement Distance (µm)"
              type="number"
              value={movementDistanceUm}
              onChange={(e) => setMovementDistanceUm(parseFloat(e.target.value) || 0)}
              fullWidth
              sx={{ mb: 2 }}
              helperText="Known stage travel distance for the calibration move"
            />

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Movement Axis</InputLabel>
              <Select
                value={movementAxis}
                label="Movement Axis"
                onChange={(e) => setMovementAxis(e.target.value)}
              >
                <MenuItem value="X">X</MenuItem>
                <MenuItem value="Y">Y</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Objective ID (optional, take current objective if empty)"
              value={objectiveId}
              onChange={(e) => setObjectiveId(e.target.value)}
              fullWidth
              sx={{ mb: 1 }}
              helperText="Leave empty to use the current objective"
            />
          </Paper>

          {/* Stepper workflow */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Calibration Workflow
            </Typography>

            <Stepper activeStep={activeStep} orientation="vertical">
              {/* Step 0 – mark point 1 */}
              <Step>
                <StepLabel
                  StepIconProps={{
                    icon: <CrosshairIcon color={activeStep === 0 ? 'primary' : 'inherit'} />,
                  }}
                >
                  {steps[0]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2">
                    Click on a clearly identifiable feature (e.g. an edge, particle
                    or structure) in the live image.
                  </Typography>
                </StepContent>
              </Step>

              {/* Step 1 – move stage */}
              <Step>
                <StepLabel
                  StepIconProps={{
                    icon: <MoveIcon color={activeStep === 1 ? 'primary' : 'inherit'} />,
                  }}
                >
                  {steps[1]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Move the stage by <strong>{movementDistanceUm} µm</strong> along{' '}
                    <strong>{movementAxis}</strong>.
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={loading ? <CircularProgress size={18} /> : <MoveIcon />}
                    onClick={handleMoveStage}
                    disabled={loading}
                  >
                    {loading ? 'Moving…' : `Move ${movementDistanceUm} µm ${movementAxis}`}
                  </Button>
                </StepContent>
              </Step>

              {/* Step 2 – mark point 2 */}
              <Step>
                <StepLabel
                  StepIconProps={{
                    icon: <CrosshairIcon color={activeStep === 2 ? 'primary' : 'inherit'} />,
                  }}
                >
                  {steps[2]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2">
                    Click on the <strong>same</strong> feature you marked before. The
                    shift in pixels will determine the pixel size.
                  </Typography>
                </StepContent>
              </Step>

              {/* Step 3 – calculate */}
              <Step>
                <StepLabel
                  StepIconProps={{
                    icon: <CalcIcon color={activeStep === 3 ? 'primary' : 'inherit'} />,
                  }}
                >
                  {steps[3]}
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Review the marked points and press <em>Calculate</em> to compute and
                    save the pixel size.
                  </Typography>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={loading ? <CircularProgress size={18} /> : <CalcIcon />}
                    onClick={handleCalculate}
                    disabled={loading}
                  >
                    {loading ? 'Calculating…' : 'Calculate & Save'}
                  </Button>
                </StepContent>
              </Step>
            </Stepper>

            {/* Done state */}
            {activeStep === 4 && (
              <Box sx={{ mt: 2, textAlign: 'center' }}>
                <DoneIcon color="success" sx={{ fontSize: 48 }} />
                <Typography variant="subtitle1" color="success.main">
                  Calibration complete!
                </Typography>
              </Box>
            )}

            <Divider sx={{ my: 2 }} />

            <Button
              variant="outlined"
              startIcon={<ResetIcon />}
              onClick={handleReset}
              fullWidth
            >
              Reset / Start Over
            </Button>
          </Paper>

          {/* Info box */}
          <Paper sx={{ p: 2, mb: 2, backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(144,202,249,0.08)' : '#f0f7ff' }}>
            <Typography variant="subtitle2" gutterBottom>
              How it works
            </Typography>
            <Typography variant="body2">
              1. Mark a feature in the live image (Point 1).<br />
              2. Move the stage by a <em>known</em> distance.<br />
              3. Mark the <em>same</em> feature again (Point 2).<br />
              4. The pixel size is: <strong>distance / displacement</strong>.<br />
              5. The value is saved for the current objective.
            </Typography>
          </Paper>

          {/* Result display */}
          {result && (
            <Paper sx={{ p: 2, backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : '#f5f5f5' }}>
              <Typography variant="h6" gutterBottom>
                Result
              </Typography>
              <Alert severity="success" sx={{ mb: 2 }}>
                {result.message}
              </Alert>

              <Box sx={{ mb: 1 }}>
                <Typography variant="subtitle2">
                  Pixel Size:{' '}
                  <strong>{result.pixelSizeUm?.toFixed(4)} µm / pixel</strong>
                </Typography>
              </Box>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                  Displacement: {result.displacementPx?.toFixed(1)} px
                </Typography>
              </Box>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                  Movement: {movementDistanceUm} µm along {movementAxis}
                </Typography>
              </Box>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                  Objective: {result.objectiveId}
                </Typography>
              </Box>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default ManualPixelCalibrationTab;
