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
} from "@mui/material";
import {
  Map as MapIcon,
  Refresh as RefreshIcon,
} from "@mui/icons-material";
import * as stageCenterCalibrationSlice from "../../state/slices/StageCenterCalibrationSlice";
import apiPositionerControllerGetPositions from "../../backendapi/apiPositionerControllerGetPositions";
import StageMapCanvas from "../StageMapCanvas";
import LiveStreamTile from "../LiveStreamTile";
import { useTheme } from '@mui/material/styles';

const StageCenterStep3 = ({ onNext, onBack, activeStep, totalSteps }) => {
  const dispatch = useDispatch();
  const theme = useTheme();
  const stageCenterState = useSelector(stageCenterCalibrationSlice.getStageCenterCalibrationState);
  
  const {
    currentX,
    currentY,
    isLoading,
    error,
    successMessage
  } = stageCenterState;

  // Fetch current position on component mount
  useEffect(() => {
    fetchCurrentPosition();
    const interval = setInterval(fetchCurrentPosition, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchCurrentPosition = async () => {
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
    } catch (error) {
      console.error("Error fetching position:", error);
    }
  };

  const handleMapWidthChange = (event, newValue) => {
    // Map dimension changes are handled by the WellSelectorCanvas internally
  };

  const handleMapHeightChange = (event, newValue) => {
    // Map dimension changes are handled by the WellSelectorCanvas internally
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
          Step 3: Stage Map Visualization
        </Typography>
        Use the interactive stage map to visualize your current position and navigate to different areas. 
        Click anywhere on the map to move the stage to that location. The live stream shows your current view.
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
        {/* Stage Map - taking full width */}
        <Grid item xs={12} md={8}>
          <Card elevation={2} sx={{ background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                <MapIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Interactive Stage Map
              </Typography>
              
              <StageMapCanvas 
                width={500} 
                height={400} 
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Current Position and Controls */}
        <Grid item xs={12} md={4}>
          <Card elevation={2} sx={{ mb: 2, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                Current Position
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="X (μm)"
                    value={currentX.toFixed(1)}
                    InputProps={{ readOnly: true }}
                    fullWidth
                    variant="outlined"
                    size="small"
                    sx={{ backgroundColor: theme.palette.action.disabledBackground }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Y (μm)"
                    value={currentY.toFixed(1)}
                    InputProps={{ readOnly: true }}
                    fullWidth
                    variant="outlined"
                    size="small"
                    sx={{ backgroundColor: theme.palette.action.disabledBackground }}
                  />
                </Grid>
              </Grid>

              <Button
                variant="outlined"
                onClick={fetchCurrentPosition}
                disabled={isLoading}
                startIcon={<RefreshIcon />}
                fullWidth
                sx={{ mt: 2 }}
              >
                Refresh Position
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper elevation={1} sx={{ p: 3, mt: 3, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
        <Typography variant="h6" gutterBottom>
          How to Use the Stage Map
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Navigation:</strong> Click anywhere on the grid to move the stage to that position. 
          The current position is shown in real-time and updates automatically.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Map Controls:</strong> Use mouse wheel to zoom in/out and drag to pan the view.
          The map shows the coordinate system and allows precise positioning.
        </Typography>
        <Typography variant="body2">
          <strong>Live Stream:</strong> The camera feed on the top right shows what you're currently viewing.
          This helps you navigate to find bright spots or specific features.
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
          Next: Auto Detection
        </Button>
      </Box>
    </Box>
  );
};

export default StageCenterStep3;