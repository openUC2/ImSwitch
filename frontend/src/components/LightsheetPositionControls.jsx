import React, { useState } from "react";
import { Box, Button, Grid, TextField, Typography } from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import * as lightsheetSlice from "../state/slices/LightsheetSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner";

/**
 * LightsheetPositionControls - Simple joystick-style position controls for lightsheet axes
 * Controls X, Y, Z, and A axes with visual feedback in 3D viewer
 */
const LightsheetPositionControls = () => {
  const dispatch = useDispatch();
  const lightsheetState = useSelector(lightsheetSlice.getLightsheetState);
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );

  const [stepSize, setStepSize] = useState(100);

  // Move a specific axis by a certain distance
  const moveAxis = async (axis, distance) => {
    try {
      await apiPositionerControllerMovePositioner({
        axis: axis.toUpperCase(),
        dist: distance,
        isAbsolute: false,
      });

      // Update Redux state with new position
      const currentPos = lightsheetState.stagePositions[axis.toLowerCase()] || 0;
      dispatch(lightsheetSlice.setStagePosition({
        axis: axis.toLowerCase(),
        value: currentPos + distance
      }));

      console.log(`Moved ${axis} by ${distance}`);
    } catch (error) {
      console.error(`Error moving ${axis}:`, error);
    }
  };

  // Button style for movement controls
  const buttonStyle = {
    minWidth: "60px",
    minHeight: "60px",
    fontSize: "20px"
  };

  const smallButtonStyle = {
    minWidth: "40px",
    minHeight: "40px",
    fontSize: "16px"
  };

  return (
    <Box sx={{ p: 2, border: "1px solid #333", borderRadius: 1 }}>
      <Typography variant="h6" gutterBottom>
        Stage Position Controls
      </Typography>

      <Grid container spacing={3}>
        {/* Step size control */}
        <Grid item xs={12}>
          <TextField
            label="Step Size (µm)"
            type="number"
            value={stepSize}
            onChange={(e) => setStepSize(parseFloat(e.target.value) || 100)}
            size="small"
            inputProps={{ step: 10, min: 1 }}
          />
        </Grid>

        {/* X-Y Control (Sample X and Sample Z) */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>
            X-Y Control (Sample)
          </Typography>
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
            {/* Y+ */}
            <Button
              variant="contained"
              onClick={() => moveAxis("Y", stepSize)}
              sx={buttonStyle}
            >
              ↑
            </Button>
            
            <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
              {/* X- */}
              <Button
                variant="contained"
                onClick={() => moveAxis("X", -stepSize)}
                sx={buttonStyle}
              >
                ←
              </Button>
              
              {/* Center label */}
              <Typography sx={{ width: "60px", textAlign: "center" }}>
                X-Y
              </Typography>
              
              {/* X+ */}
              <Button
                variant="contained"
                onClick={() => moveAxis("X", stepSize)}
                sx={buttonStyle}
              >
                →
              </Button>
            </Box>
            
            {/* Y- */}
            <Button
              variant="contained"
              onClick={() => moveAxis("Y", -stepSize)}
              sx={buttonStyle}
            >
              ↓
            </Button>
          </Box>
        </Grid>

        {/* Z and A Controls */}
        <Grid item xs={12} md={6}>
          <Grid container spacing={2}>
            {/* Z Control (Objective) */}
            <Grid item xs={6}>
              <Typography variant="subtitle2" gutterBottom>
                Z (Objective)
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <Button
                  variant="contained"
                  onClick={() => moveAxis("Z", stepSize)}
                  sx={smallButtonStyle}
                  fullWidth
                >
                  Z+
                </Button>
                <Button
                  variant="contained"
                  onClick={() => moveAxis("Z", -stepSize)}
                  sx={smallButtonStyle}
                  fullWidth
                >
                  Z-
                </Button>
              </Box>
            </Grid>

            {/* A Control (Sample) */}
            <Grid item xs={6}>
              <Typography variant="subtitle2" gutterBottom>
                A (Sample)
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={() => moveAxis("A", stepSize)}
                  sx={smallButtonStyle}
                  fullWidth
                >
                  A+
                </Button>
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={() => moveAxis("A", -stepSize)}
                  sx={smallButtonStyle}
                  fullWidth
                >
                  A-
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Grid>

        {/* Current positions display */}
        <Grid item xs={12}>
          <Typography variant="caption" color="text.secondary">
            Current Positions (µm): 
            X: {lightsheetState.stagePositions?.x?.toFixed(1) || 0}, 
            Y: {lightsheetState.stagePositions?.y?.toFixed(1) || 0}, 
            Z: {lightsheetState.stagePositions?.z?.toFixed(1) || 0}, 
            A: {lightsheetState.stagePositions?.a?.toFixed(1) || 0}
          </Typography>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LightsheetPositionControls;
