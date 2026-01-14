import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  Box,
  Button,
  Typography,
  Paper,
  Alert,
  Grid,
  TextField,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import {
  Search as SearchIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  Brightness7 as BrightnessIcon,
  Speed as SpeedIcon,
} from "@mui/icons-material";
import * as stageCenterCalibrationSlice from "../../state/slices/StageCenterCalibrationSlice";
import apiStageCenterCalibrationPerformCalibration from "../../backendapi/apiStageCenterCalibrationPerformCalibration";
import apiStageCenterCalibrationGetStatus from "../../backendapi/apiStageCenterCalibrationGetStatus";
import apiStageCenterCalibrationStopCalibration from "../../backendapi/apiStageCenterCalibrationStopCalibration";
import LiveStreamTile from "../LiveStreamTile";
import { useTheme } from '@mui/material/styles';

const StageCenterStep4 = ({ onNext, onBack, activeStep, totalSteps }) => {
  const dispatch = useDispatch();
  const theme = useTheme();
  const stageCenterState = useSelector(stageCenterCalibrationSlice.getStageCenterCalibrationState);
  
  const {
    startX,
    startY,
    exposureTimeUs,
    speed,
    stepUm,
    maxRadiusUm,
    brightnessFactor,
    isCalibrationRunning,
    calibrationResults,
    currentX,
    currentY,
    isLoading,
    error,
    successMessage
  } = stageCenterState;

  const [statusCheckInterval, setStatusCheckInterval] = useState(null);

  // Check calibration status periodically when running
  useEffect(() => {
    if (isCalibrationRunning) {
      const interval = setInterval(checkCalibrationStatus, 1000);
      setStatusCheckInterval(interval);
      return () => clearInterval(interval);
    } else if (statusCheckInterval) {
      clearInterval(statusCheckInterval);
      setStatusCheckInterval(null);
    }
  }, [isCalibrationRunning]);

  const checkCalibrationStatus = async () => {
    try {
      const isRunning = await apiStageCenterCalibrationGetStatus();
      dispatch(stageCenterCalibrationSlice.setIsCalibrationRunning(isRunning));
      
      if (!isRunning && calibrationResults.length === 0) {
        // Calibration finished, might need to fetch results
        dispatch(stageCenterCalibrationSlice.setSuccessMessage("Calibration completed"));
      }
    } catch (error) {
      console.error("Error checking calibration status:", error);
    }
  };

  const startCalibration = async () => {
    dispatch(stageCenterCalibrationSlice.setIsLoading(true));
    dispatch(stageCenterCalibrationSlice.clearMessages());
    
    try {
      const results = await apiStageCenterCalibrationPerformCalibration({
        start_x: startX,
        start_y: startY,
        exposure_time_us: exposureTimeUs,
        speed: speed,
        step_um: stepUm,
        max_radius_um: maxRadiusUm,
        brightness_factor: brightnessFactor,
      });
      
      dispatch(stageCenterCalibrationSlice.setIsCalibrationRunning(true));
      dispatch(stageCenterCalibrationSlice.setCalibrationResults(results));
      
      if (results && results.length > 0) {
        // Use the last position as the found center (this is where the bright spot was found)
        const lastPos = results[results.length - 1];
        dispatch(stageCenterCalibrationSlice.setFoundCenter({
          x: lastPos[0],
          y: lastPos[1]
        }));
        dispatch(stageCenterCalibrationSlice.setSuccessMessage(
          `Bright spot found at (${lastPos[0].toFixed(1)}, ${lastPos[1].toFixed(1)})`
        ));
      }
    } catch (error) {
      console.error("Error starting calibration:", error);
      dispatch(stageCenterCalibrationSlice.setError("Failed to start automatic calibration"));
    } finally {
      dispatch(stageCenterCalibrationSlice.setIsLoading(false));
    }
  };

  const stopCalibration = async () => {
    try {
      await apiStageCenterCalibrationStopCalibration();
      dispatch(stageCenterCalibrationSlice.setIsCalibrationRunning(false));
      dispatch(stageCenterCalibrationSlice.setSuccessMessage("Calibration stopped"));
    } catch (error) {
      console.error("Error stopping calibration:", error);
      dispatch(stageCenterCalibrationSlice.setError("Failed to stop calibration"));
    }
  };

  const useCurrentAsStart = () => {
    dispatch(stageCenterCalibrationSlice.setStartX(currentX));
    dispatch(stageCenterCalibrationSlice.setStartY(currentY));
    dispatch(stageCenterCalibrationSlice.setSuccessMessage("Current position set as start point"));
  };

  const clearMessages = () => {
    dispatch(stageCenterCalibrationSlice.clearMessages());
  };

  return (
    <Box sx={{ maxWidth: 1000, mx: "auto", p: 2 }}>
      {/* Live Stream Tile - positioned in top right */}
      <Box sx={{ position: 'absolute', top: 20, right: 20, zIndex: 10 }}>
        <LiveStreamTile width={200} height={150} />
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Step 4: Automatic Bright Spot Detection
        </Typography>
        Configure and run an automated spiral scan to find bright spots on your stage. The system will 
        scan in an expanding spiral pattern until it finds an area with increased brightness. Monitor the live stream.
      </Alert>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={clearMessages}>
          {error}
        </Alert>
      )}

      {successMessage && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={clearMessages}>
          {successMessage}
        </Alert>
      )}

      {isCalibrationRunning && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="subtitle1">Calibration in Progress</Typography>
              <Typography variant="body2">
                Scanning stage for bright spots... This may take several minutes.
              </Typography>
            </Box>
            <Button
              variant="contained"
              color="error"
              onClick={stopCalibration}
              startIcon={<StopIcon />}
            >
              Stop
            </Button>
          </Box>
          <LinearProgress sx={{ mt: 2 }} />
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Control Panel */}
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <SearchIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Calibration Control
              </Typography>
              
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6}>
                  <TextField
                    label="Start X (μm)"
                    value={startX}
                    onChange={(e) => dispatch(stageCenterCalibrationSlice.setStartX(parseFloat(e.target.value) || 0))}
                    fullWidth
                    type="number"
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Start Y (μm)"
                    value={startY}
                    onChange={(e) => dispatch(stageCenterCalibrationSlice.setStartY(parseFloat(e.target.value) || 0))}
                    fullWidth
                    type="number"
                    size="small"
                  />
                </Grid>
              </Grid>

              <Button
                variant="outlined"
                onClick={useCurrentAsStart}
                fullWidth
                sx={{ mb: 3 }}
                size="small"
              >
                Use Current Position as Start
              </Button>

              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  onClick={startCalibration}
                  disabled={isCalibrationRunning || isLoading}
                  startIcon={<PlayIcon />}
                  fullWidth
                >
                  Start Auto Detection
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Results */}
          {calibrationResults.length > 0 && (
            <Card elevation={2} sx={{ mt: 2, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Detection Results
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Positions scanned:</strong> {calibrationResults.length}
                </Typography>
                {calibrationResults.length > 0 && (
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      label={`Found at: (${calibrationResults[calibrationResults.length - 1][0].toFixed(1)}, ${calibrationResults[calibrationResults.length - 1][1].toFixed(1)})`}
                      color="success"
                      variant="outlined"
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Advanced Settings */}
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Current Position
              </Typography>
              
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <TextField
                    label="Current X (μm)"
                    value={currentX.toFixed(1)}
                    InputProps={{ readOnly: true }}
                    fullWidth
                    size="small"
                    sx={{ backgroundColor: theme.palette.action.disabledBackground }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Current Y (μm)"
                    value={currentY.toFixed(1)}
                    InputProps={{ readOnly: true }}
                    fullWidth
                    size="small"
                    sx={{ backgroundColor: theme.palette.action.disabledBackground }}
                  />
                </Grid>
              </Grid>

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle2">Advanced Parameters</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        label="Exposure (μs)"
                        value={exposureTimeUs}
                        onChange={(e) => dispatch(stageCenterCalibrationSlice.setExposureTimeUs(parseInt(e.target.value) || 3000))}
                        fullWidth
                        type="number"
                        size="small"
                        InputProps={{
                          startAdornment: <BrightnessIcon sx={{ mr: 1, fontSize: 16 }} />,
                        }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="Speed"
                        value={speed}
                        onChange={(e) => dispatch(stageCenterCalibrationSlice.setSpeed(parseInt(e.target.value) || 5000))}
                        fullWidth
                        type="number"
                        size="small"
                        InputProps={{
                          startAdornment: <SpeedIcon sx={{ mr: 1, fontSize: 16 }} />,
                        }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="Step Size (μm)"
                        value={stepUm}
                        onChange={(e) => dispatch(stageCenterCalibrationSlice.setStepUm(parseFloat(e.target.value) || 50))}
                        fullWidth
                        type="number"
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="Max Radius (μm)"
                        value={maxRadiusUm}
                        onChange={(e) => dispatch(stageCenterCalibrationSlice.setMaxRadiusUm(parseFloat(e.target.value) || 2000))}
                        fullWidth
                        type="number"
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        label="Brightness Factor"
                        value={brightnessFactor}
                        onChange={(e) => dispatch(stageCenterCalibrationSlice.setBrightnessFactor(parseFloat(e.target.value) || 1.4))}
                        fullWidth
                        type="number"
                        size="small"
                        helperText="Detection threshold (e.g., 1.4 = 40% brighter than baseline)"
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper elevation={1} sx={{ p: 3, mt: 3, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
        <Typography variant="h6" gutterBottom>
          How Automatic Detection Works
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Spiral Scan:</strong> The system moves the stage in an expanding spiral pattern, 
          starting from your specified start position.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Brightness Detection:</strong> At each position, it captures an image and measures 
          the mean brightness. When the brightness increases by the specified factor, it stops.
        </Typography>
        <Typography variant="body2">
          <strong>Parameters:</strong> Adjust the step size for scan resolution, max radius for scan area, 
          and brightness factor for detection sensitivity.
        </Typography>
      </Paper>

      <Box sx={{ display: "flex", justifyContent: "space-between", mt: 3 }}>
        <Button 
          variant="outlined" 
          onClick={onBack}
          size="large"
          disabled={isCalibrationRunning}
        >
          Back
        </Button>
        <Button 
          variant="contained" 
          onClick={onNext}
          size="large"
          disabled={isCalibrationRunning}
          sx={{
            background: theme.palette.primary.main,
            color: theme.palette.primary.contrastText,
            '&:hover': {
              background: theme.palette.primary.dark,
            }
          }}
        >
          Next: Review Results
        </Button>
      </Box>
    </Box>
  );
};

export default StageCenterStep4;