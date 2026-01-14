import React from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  Box,
  Button,
  Typography,
  Paper,
  Alert,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Divider,
} from "@mui/material";
import {
  CheckCircle as CheckIcon,
  Assessment as AssessmentIcon,
  CenterFocusStrong as CenterIcon,
  Save as SaveIcon,
} from "@mui/icons-material";
import { useTheme } from '@mui/material/styles';
import * as stageCenterCalibrationSlice from "../../state/slices/StageCenterCalibrationSlice";
import apiPositionerControllerSetStageOffsetAxis from "../../backendapi/apiPositionerControllerSetStageOffsetAxis";
import StageMapCanvas from "../StageMapCanvas";
import LiveStreamTile from "../LiveStreamTile";

const StageCenterStep5 = ({ onNext, onBack, activeStep, totalSteps }) => {
  const dispatch = useDispatch();
  const theme = useTheme();
  const stageCenterState = useSelector(stageCenterCalibrationSlice.getStageCenterCalibrationState);
  
  const {
    currentX,
    currentY,
    manualCenterX,
    manualCenterY,
    foundCenterX,
    foundCenterY,
    calibrationResults,
    startX,
    startY,
    stepUm,
    maxRadiusUm,
    brightnessFactor,
    error,
    successMessage
  } = stageCenterState;

  const hasManualCenter = manualCenterX && manualCenterY;
  const hasFoundCenter = foundCenterX !== null && foundCenterY !== null;
  const hasCalibrationData = calibrationResults.length > 0;

  const calculateDistance = (x1, y1, x2, y2) => {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  };

  const applyManualCenter = async () => {
    if (!hasManualCenter) return;
    
    try {
      await apiPositionerControllerSetStageOffsetAxis({
        axis: "X",
        knownPosition: parseFloat(manualCenterX),
        currentPosition: 0
      });
      
      await apiPositionerControllerSetStageOffsetAxis({
        axis: "Y",
        knownPosition: parseFloat(manualCenterY),
        currentPosition: 0
      });
      
      dispatch(stageCenterCalibrationSlice.setSuccessMessage("Manual center position applied successfully"));
    } catch (error) {
      console.error("Error applying manual center:", error);
      dispatch(stageCenterCalibrationSlice.setError("Failed to apply manual center position"));
    }
  };

  const applyFoundCenter = async () => {
    if (!hasFoundCenter) return;
    
    try {
      await apiPositionerControllerSetStageOffsetAxis({
        axis: "X",
        knownPosition: parseFloat(foundCenterX),
        currentPosition: 0
      });
      
      await apiPositionerControllerSetStageOffsetAxis({
        axis: "Y",
        knownPosition: parseFloat(foundCenterY),
        currentPosition: 0
      });
      
      dispatch(stageCenterCalibrationSlice.setSuccessMessage("Found center position applied successfully"));
    } catch (error) {
      console.error("Error applying found center:", error);
      dispatch(stageCenterCalibrationSlice.setError("Failed to apply found center position"));
    }
  };

  const clearMessages = () => {
    dispatch(stageCenterCalibrationSlice.clearMessages());
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: "auto", p: 2 }}>
      {/* Live Stream Tile - positioned in top right */}
      <Box sx={{ position: 'absolute', top: 20, right: 20, zIndex: 10 }}>
        <LiveStreamTile width={200} height={150} />
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Step 5: Review Calibration Results
        </Typography>
        Review the calibration results from manual entry and automatic detection. Choose which 
        center position to apply or proceed to the final step. Check the live stream for verification.
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

      <Grid container spacing={3}>
        {/* Results Summary */}
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ mb: 2, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <AssessmentIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Calibration Summary
              </Typography>
              
              <Grid container spacing={2}>
                {/* Current Position */}
                <Grid item xs={12}>
                  <Paper elevation={1} sx={{ p: 2, background: theme.palette.action.disabledBackground }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Current Stage Position
                    </Typography>
                    <Typography variant="body2">
                      X: {currentX.toFixed(2)} μm, Y: {currentY.toFixed(2)} μm
                    </Typography>
                  </Paper>
                </Grid>

                {/* Manual Center */}
                {hasManualCenter && (
                  <Grid item xs={12}>
                    <Paper elevation={1} sx={{ p: 2, background: theme.palette.warning.light }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="subtitle2">
                          Manual Center Position
                        </Typography>
                        <Chip label="Manual" color="warning" size="small" />
                      </Box>
                      <Typography variant="body2" gutterBottom>
                        X: {parseFloat(manualCenterX).toFixed(2)} μm, Y: {parseFloat(manualCenterY).toFixed(2)} μm
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        Distance from current: {calculateDistance(
                          currentX, currentY, 
                          parseFloat(manualCenterX), parseFloat(manualCenterY)
                        ).toFixed(1)} μm
                      </Typography>
                      <Box sx={{ mt: 1 }}>
                        <Button
                          variant="contained"
                          size="small"
                          onClick={applyManualCenter}
                          startIcon={<SaveIcon />}
                          color="warning"
                        >
                          Apply Manual Center
                        </Button>
                      </Box>
                    </Paper>
                  </Grid>
                )}

                {/* Found Center */}
                {hasFoundCenter && (
                  <Grid item xs={12}>
                    <Paper elevation={1} sx={{ p: 2, background: theme.palette.success.light }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="subtitle2">
                          Auto-Detected Center
                        </Typography>
                        <Chip label="Auto" color="success" size="small" />
                      </Box>
                      <Typography variant="body2" gutterBottom>
                        X: {foundCenterX.toFixed(2)} μm, Y: {foundCenterY.toFixed(2)} μm
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        Distance from current: {calculateDistance(
                          currentX, currentY, foundCenterX, foundCenterY
                        ).toFixed(1)} μm
                      </Typography>
                      <Box sx={{ mt: 1 }}>
                        <Button
                          variant="contained"
                          size="small"
                          onClick={applyFoundCenter}
                          startIcon={<SaveIcon />}
                          color="success"
                        >
                          Apply Found Center
                        </Button>
                      </Box>
                    </Paper>
                  </Grid>
                )}

                {/* Comparison */}
                {hasManualCenter && hasFoundCenter && (
                  <Grid item xs={12}>
                    <Paper elevation={1} sx={{ p: 2, background: theme.palette.info.light }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Center Comparison
                      </Typography>
                      <Typography variant="body2">
                        Distance between centers: {calculateDistance(
                          parseFloat(manualCenterX), parseFloat(manualCenterY),
                          foundCenterX, foundCenterY
                        ).toFixed(1)} μm
                      </Typography>
                    </Paper>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>

          {/* Calibration Parameters */}
          {hasCalibrationData && (
            <Card elevation={2}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Detection Parameters
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>Start Position</TableCell>
                        <TableCell>({startX.toFixed(1)}, {startY.toFixed(1)}) μm</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Step Size</TableCell>
                        <TableCell>{stepUm} μm</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Max Radius</TableCell>
                        <TableCell>{maxRadiusUm} μm</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Brightness Factor</TableCell>
                        <TableCell>{brightnessFactor}x</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Positions Scanned</TableCell>
                        <TableCell>{calibrationResults.length}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Visual Results */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <CenterIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Visual Results
              </Typography>
              
              <StageMapCanvas 
                width={450} 
                height={350} 
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Status Messages */}
      {!hasManualCenter && !hasFoundCenter && (
        <Alert severity="warning" sx={{ mt: 3 }}>
          No calibration results available. Please go back and perform manual entry or automatic detection.
        </Alert>
      )}

      {(hasManualCenter || hasFoundCenter) && (
        <Alert severity="success" sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            <CheckIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Calibration Results Available
          </Typography>
          <Typography variant="body2">
            You can apply one of the found center positions or proceed to the final step.
            {hasManualCenter && hasFoundCenter && 
              " Compare the manual and automatic results to choose the most accurate one."
            }
          </Typography>
        </Alert>
      )}

      <Divider sx={{ my: 3 }} />

      <Paper elevation={1} sx={{ p: 3, backgroundColor: '#f8f9fa' }}>
        <Typography variant="h6" gutterBottom>
          Recommendation
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>If both methods found centers:</strong> Compare the positions and consider the 
          accuracy of your manual positioning versus the reliability of the automatic detection.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>For automatic results:</strong> The system found the brightest spot within the 
          search area. This is typically a reliable indicator of a sample or reference point.
        </Typography>
        <Typography variant="body2">
          <strong>For manual results:</strong> Use this if you have precise knowledge of where 
          the center should be or if automatic detection failed to find the correct spot.
        </Typography>
      </Paper>

      <Box sx={{ display: "flex", justifyContent: "space-between", mt: 3 }}>
        <Button 
          variant="outlined" 
          onClick={onBack}
          size="large"
        >
          Back
        </Button>
        <Button 
          variant="contained" 
          onClick={onNext}
          size="large"
          sx={{
            background: theme.palette.primary.main,
            color: theme.palette.primary.contrastText,
            '&:hover': {
              background: theme.palette.primary.dark,
            }
          }}
        >
          Complete Calibration
        </Button>
      </Box>
    </Box>
  );
};

export default StageCenterStep5;