import React, { useEffect } from "react";
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
  Divider,
} from "@mui/material";
import {
  Refresh as RefreshIcon,
  MyLocation as LocationIcon,
  Save as SaveIcon,
  Input as InputIcon,
} from "@mui/icons-material";
import * as stageCenterCalibrationSlice from "../../state/slices/StageCenterCalibrationSlice";
import apiPositionerControllerMovePositioner from "../../backendapi/apiPositionerControllerMovePositioner";
import apiPositionerControllerGetPositions from "../../backendapi/apiPositionerControllerGetPositions";
import LiveStreamTile from "../LiveStreamTile";
import { useTheme } from '@mui/material/styles';

const StageCenterStep2 = ({ onNext, onBack, activeStep, totalSteps }) => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const stageCenterState = useSelector(stageCenterCalibrationSlice.getStageCenterCalibrationState);
  
  const {
    currentX,
    currentY,
    manualCenterX,
    manualCenterY,
    isLoading,
    error,
    successMessage
  } = stageCenterState;

  // Fetch current position on component mount
  useEffect(() => {
    fetchCurrentPosition();
  }, []);

  const fetchCurrentPosition = async () => {
    dispatch(stageCenterCalibrationSlice.setIsLoading(true));
    try {
      const data = await apiPositionerControllerGetPositions();
      
      if (data.ESP32Stage) {
        dispatch(stageCenterCalibrationSlice.setCurrentPosition({
          x: data.ESP32Stage.X,
          y: data.ESP32Stage.Y
        }));
      } else if (data.VirtualStage) {
        dispatch(stageCenterCalibrationSlice.setCurrentPosition({
          x: data.VirtualStage.X,
          y: data.VirtualStage.Y
        }));
      }
      dispatch(stageCenterCalibrationSlice.setSuccessMessage("Position updated successfully"));
    } catch (error) {
      console.error("Error fetching position:", error);
      dispatch(stageCenterCalibrationSlice.setError("Failed to fetch current position"));
    } finally {
      dispatch(stageCenterCalibrationSlice.setIsLoading(false));
    }
  };

  const moveToPosition = async (x, y) => {
    if (!x || !y) {
      dispatch(stageCenterCalibrationSlice.setError("Please enter valid X and Y coordinates"));
      return;
    }

    dispatch(stageCenterCalibrationSlice.setIsLoading(true));
    try {
      // Move to X position
      await apiPositionerControllerMovePositioner({
        positionerName: "ESP32Stage",
        axis: "X",
        dist: parseFloat(x),
        isAbsolute: true,
        isBlocking: false,
        speed: 1000,
      });

      // Move to Y position
      await apiPositionerControllerMovePositioner({
        positionerName: "ESP32Stage",
        axis: "Y",
        dist: parseFloat(y),
        isAbsolute: true,
        isBlocking: false,
        speed: 1000,
      });

      dispatch(stageCenterCalibrationSlice.setSuccessMessage(`Moving to position (${x}, ${y})`));
      
      // Update current position after a short delay
      setTimeout(() => {
        fetchCurrentPosition();
      }, 2000);
    } catch (error) {
      console.error("Error moving to position:", error);
      dispatch(stageCenterCalibrationSlice.setError("Failed to move to position"));
    } finally {
      dispatch(stageCenterCalibrationSlice.setIsLoading(false));
    }
  };

  const useCurrentAsCenter = () => {
    dispatch(stageCenterCalibrationSlice.setManualCenter({
      x: currentX.toString(),
      y: currentY.toString()
    }));
    dispatch(stageCenterCalibrationSlice.setSuccessMessage("Current position set as center"));
  };

  const moveToManualCenter = () => {
    moveToPosition(manualCenterX, manualCenterY);
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
          Step 2: Manual Position Entry
        </Typography>
        Enter known center coordinates or use the current stage position as your reference point.
        Watch the live stream to verify your position.
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
        {/* Current Position Card */}
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ height: '100%', background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <LocationIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Current Stage Position
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Current X (μm)"
                    value={currentX}
                    InputProps={{ readOnly: true }}
                    fullWidth
                    variant="outlined"
                    sx={{ backgroundColor: theme.palette.action.disabledBackground }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Current Y (μm)"
                    value={currentY}
                    InputProps={{ readOnly: true }}
                    fullWidth
                    variant="outlined"
                    sx={{ backgroundColor: theme.palette.action.disabledBackground }}
                  />
                </Grid>
              </Grid>

              <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Button
                  variant="outlined"
                  onClick={fetchCurrentPosition}
                  disabled={isLoading}
                  startIcon={<RefreshIcon />}
                  size="small"
                >
                  Refresh
                </Button>
                <Button
                  variant="contained"
                  onClick={useCurrentAsCenter}
                  disabled={isLoading}
                  startIcon={<SaveIcon />}
                  size="small"
                  color="success"
                >
                  Use as Center
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Manual Entry Card */}
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ height: '100%', background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <InputIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Manual Center Entry
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Center X (μm)"
                    value={manualCenterX}
                    onChange={(e) => dispatch(stageCenterCalibrationSlice.setManualCenterX(e.target.value))}
                    fullWidth
                    variant="outlined"
                    type="number"
                    placeholder="0"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Center Y (μm)"
                    value={manualCenterY}
                    onChange={(e) => dispatch(stageCenterCalibrationSlice.setManualCenterY(e.target.value))}
                    fullWidth
                    variant="outlined"
                    type="number"
                    placeholder="0"
                  />
                </Grid>
              </Grid>

              <Box sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  onClick={moveToManualCenter}
                  disabled={isLoading || !manualCenterX || !manualCenterY}
                  fullWidth
                  sx={{
                    background: theme.palette.primary.main,
                    color: theme.palette.primary.contrastText,
                    '&:hover': {
                      background: theme.palette.primary.dark,
                    }
                  }}
                >
                  Move to Position
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Divider sx={{ my: 3 }} />

      <Paper elevation={1} sx={{ p: 3, mb: 3, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
        <Typography variant="h6" gutterBottom>
          Manual Calibration Instructions
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Method 1:</strong> If you know the exact center coordinates of your stage, enter them 
          in the "Manual Center Entry" fields above and click "Move to Position" to test.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Method 2:</strong> Manually move your stage to what you believe is the center position 
          using the joystick controls or other methods, then click "Use as Center" to save the current position.
        </Typography>
        <Typography variant="body2">
          <strong>Tip:</strong> You can combine both methods - move roughly to the center area, then fine-tune 
          with manual coordinate entry.
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
          Next: Stage Map
        </Button>
      </Box>
    </Box>
  );
};

export default StageCenterStep2;