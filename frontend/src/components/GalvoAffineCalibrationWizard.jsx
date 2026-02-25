import React, { useCallback, useMemo } from 'react';
import {
  Box,
  Button,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  Chip,
} from '@mui/material';
import GpsFixedIcon from '@mui/icons-material/GpsFixed';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import { useSelector, useDispatch } from 'react-redux';
import { getConnectionSettingsState } from '../state/slices/ConnectionSettingsSlice';
import {
  getCalibrationState,
  getAffineTransformState,
  getGalvoScannerState,
  cancelCalibration,
  advanceCalibrationStep,
  setCalibrationComplete,
  setAffineTransform,
  setError,
  setStatusMessage,
  clearStatusMessage,
} from '../state/slices/GalvoScannerSlice';
import {
  apiRunAffineCalibration,
  apiStartGalvoScan,
  apiStopGalvoScan,
} from '../backendapi/apiGalvoScannerController';

/**
 * GalvoAffineCalibrationWizard - 3-point guided calibration workflow
 * 
 * Steps:
 * 1. Galvo moves to position 1 → user clicks bright spot on camera
 * 2. Galvo moves to position 2 → user clicks bright spot on camera
 * 3. Galvo moves to position 3 → user clicks bright spot on camera
 * 4. Compute affine transform and save
 * 
 * This component renders as an overlay panel at the bottom of the ArbitraryPointsTab.
 * The actual click handling on the camera is in GalvoArbitraryPointsTab (checks calibration.active).
 */
const GalvoAffineCalibrationWizard = () => {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  const galvoState = useSelector(getGalvoScannerState);
  const selectedScanner = galvoState?.selectedScanner || '';
  const calibration = useSelector(getCalibrationState);
  const affine = useSelector(getAffineTransformState);

  const { currentStep, galvoPoints, camPoints, completed } = calibration;

  // Move galvo to calibration position (use single-point raster as workaround)
  const moveGalvoToPoint = useCallback(async (galvoPt) => {
    if (!selectedScanner) return;
    try {
      // Use a small 1x1 raster centered on the target to position the galvo
      // and keep the laser on
      const config = {
        nx: 1,
        ny: 1,
        x_min: galvoPt.x,
        x_max: galvoPt.x,
        y_min: galvoPt.y,
        y_max: galvoPt.y,
        sample_period_us: 1000,
        frame_count: 0, // infinite
        enable_trigger: 1,
      };
      await apiStartGalvoScan(hostIP, hostPort, selectedScanner, config);
    } catch (err) {
      dispatch(setError(`Failed to move galvo: ${err.message}`));
    }
  }, [hostIP, hostPort, selectedScanner, dispatch]);

  // Stop galvo after calibration step
  const stopGalvo = useCallback(async () => {
    try {
      await apiStopGalvoScan(hostIP, hostPort, selectedScanner);
    } catch (err) {
      // Silently ignore
    }
  }, [hostIP, hostPort, selectedScanner]);

  // Move galvo to current step's position when step changes
  React.useEffect(() => {
    if (calibration.active && currentStep < 3) {
      moveGalvoToPoint(galvoPoints[currentStep]);
    }
    return () => {
      // Stop galvo when leaving calibration
    };
  }, [currentStep, calibration.active]); // eslint-disable-line react-hooks/exhaustive-deps

  // Confirm current step and advance
  const handleConfirmStep = useCallback(async () => {
    if (!camPoints[currentStep]) {
      dispatch(setError('Please click on the bright spot in the camera view first'));
      return;
    }

    if (currentStep < 2) {
      // Advance to next step
      dispatch(advanceCalibrationStep());
    } else {
      // All 3 points collected → compute affine
      await stopGalvo();

      const camPtsArray = camPoints.map(pt => [pt.x, pt.y]);
      const galvoPtsArray = galvoPoints.map(pt => [pt.x, pt.y]);

      try {
        const result = await apiRunAffineCalibration(
          hostIP, hostPort, selectedScanner, camPtsArray, galvoPtsArray, true
        );
        if (result.error) {
          dispatch(setError(`Calibration failed: ${result.error}`));
        } else {
          dispatch(setAffineTransform(result.affine_transform));
          dispatch(setCalibrationComplete());
          dispatch(setStatusMessage('Calibration complete! Affine transform saved.'));
          setTimeout(() => dispatch(clearStatusMessage()), 3000);
        }
      } catch (err) {
        dispatch(setError(`Calibration error: ${err.message}`));
      }
    }
  }, [currentStep, camPoints, galvoPoints, hostIP, hostPort, selectedScanner, dispatch, stopGalvo]);

  // Cancel calibration
  const handleCancel = useCallback(async () => {
    await stopGalvo();
    dispatch(cancelCalibration());
  }, [dispatch, stopGalvo]);

  // Step status
  const stepStatuses = useMemo(() => {
    return galvoPoints.map((_, i) => ({
      confirmed: camPoints[i] !== null && i < currentStep,
      active: i === currentStep,
      pending: i > currentStep,
    }));
  }, [galvoPoints, camPoints, currentStep]);

  if (completed) return null;

  return (
    <Paper
      sx={{
        p: 2, mt: 2,
        border: '2px solid #ffff00',
        borderRadius: 2,
        backgroundColor: 'rgba(255, 255, 0, 0.05)',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <GpsFixedIcon sx={{ mr: 1, color: '#ffff00' }} />
        <Typography variant="subtitle1" fontWeight="bold">
          Affine Calibration Wizard
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        <Button
          size="small"
          color="error"
          startIcon={<CancelIcon />}
          onClick={handleCancel}
        >
          Cancel
        </Button>
      </Box>

      <Alert severity="info" sx={{ mb: 2 }}>
        The galvo will move to each calibration position with laser ON.
        Click the bright spot in the camera view, then press "Confirm".
      </Alert>

      <Stepper activeStep={currentStep} orientation="vertical">
        {galvoPoints.map((gPt, index) => {
          const camPt = camPoints[index];
          return (
            <Step key={index}>
              <StepLabel
                optional={
                  <Typography variant="caption">
                    Galvo: ({gPt.x}, {gPt.y}) — {gPt.label}
                  </Typography>
                }
              >
                Point {index + 1}: {gPt.label}
              </StepLabel>
              <StepContent>
                <Box sx={{ mb: 1 }}>
                  {camPt ? (
                    <Chip
                      icon={<CheckCircleIcon />}
                      label={`Camera click: (${camPt.x}, ${camPt.y})`}
                      color="success"
                      size="small"
                    />
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      Click the bright laser spot on the camera view…
                    </Typography>
                  )}
                </Box>
                <Button
                  variant="contained"
                  size="small"
                  disabled={!camPt}
                  onClick={handleConfirmStep}
                  endIcon={index < 2 ? <NavigateNextIcon /> : <CheckCircleIcon />}
                >
                  {index < 2 ? 'Confirm & Next' : 'Confirm & Compute'}
                </Button>
              </StepContent>
            </Step>
          );
        })}
      </Stepper>
    </Paper>
  );
};

export default GalvoAffineCalibrationWizard;
