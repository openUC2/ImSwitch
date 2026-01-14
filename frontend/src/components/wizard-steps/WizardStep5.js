import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Typography,
  Box,
  Paper,
  Button,
  Grid,
  Alert,
} from "@mui/material";
import {
  Save as SaveIcon,
  Visibility as FocusIcon,
} from "@mui/icons-material";
import LiveViewControlWrapper from "../../axon/LiveViewControlWrapper";
import PositionControllerComponent from "../../axon/PositionControllerComponent";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice.js";

import apiObjectiveControllerMoveToObjective from "../../backendapi/apiObjectiveControllerMoveToObjective.js";
import apiPositionerControllerGetPositions from "../../backendapi/apiPositionerControllerGetPositions.js";
import apiObjectiveControllerSetPositions from "../../backendapi/apiObjectiveControllerSetPositions.js";

const WizardStep5 = ({ hostIP, hostPort, onNext, onBack, activeStep, totalSteps }) => {
  const dispatch = useDispatch();
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  
  const [movedToObjective2, setMovedToObjective2] = useState(false);
  const [currentPositions, setCurrentPositions] = useState({
    X: null,
    Y: null,
    Z: null,
    A: null,
  });

  useEffect(() => {
    // Fetch current positions when component mounts
    const fetchPos = () => {
      apiPositionerControllerGetPositions()
        .then((data) => {
          if (data.ESP32Stage) {
            setCurrentPositions({
              X: data.ESP32Stage.X,
              Y: data.ESP32Stage.Y,
              Z: data.ESP32Stage.Z,
              A: data.ESP32Stage.A,
            });
            dispatch(objectiveSlice.setCurrentZ(data.ESP32Stage.Z));
            dispatch(objectiveSlice.setCurrentA(data.ESP32Stage.A));
          } else if (data.VirtualStage) {
            setCurrentPositions({
              X: data.VirtualStage.X,
              Y: data.VirtualStage.Y,
              Z: data.VirtualStage.Z,
              A: data.VirtualStage.A,
            });
            dispatch(objectiveSlice.setCurrentZ(data.VirtualStage.Z));
            dispatch(objectiveSlice.setCurrentA(data.VirtualStage.A));
          }
        })
        .catch((err) => {
          console.error("Error fetching current positions:", err);
        });
    };
    fetchPos();
  }, [dispatch]);

  const fetchCurrentPositions = () => {
    apiPositionerControllerGetPositions()
      .then((data) => {
        if (data.ESP32Stage) {
          setCurrentPositions({
            X: data.ESP32Stage.X,
            Y: data.ESP32Stage.Y,
            Z: data.ESP32Stage.Z,
            A: data.ESP32Stage.A,
          });
          dispatch(objectiveSlice.setCurrentZ(data.ESP32Stage.Z));
          dispatch(objectiveSlice.setCurrentA(data.ESP32Stage.A));
        } else if (data.VirtualStage) {
          setCurrentPositions({
            X: data.VirtualStage.X,
            Y: data.VirtualStage.Y,
            Z: data.VirtualStage.Z,
            A: data.VirtualStage.A,
          });
          dispatch(objectiveSlice.setCurrentZ(data.VirtualStage.Z));
          dispatch(objectiveSlice.setCurrentA(data.VirtualStage.A));
        }
      })
      .catch((err) => {
        console.error("Error fetching current positions:", err);
      });
  };

  const handleSwitchToObjective2 = () => {
    apiObjectiveControllerMoveToObjective(2)
      .then((data) => {
        dispatch(objectiveSlice.setCurrentObjective(2));
        setMovedToObjective2(true);
        // Small delay to ensure position is updated
        setTimeout(fetchCurrentPositions, 1000);
      })
      .catch((err) => {
        console.error("Error switching to objective 2:", err);
      });
  };

  const handleSaveZ1Position = () => {
    if (currentPositions.Z !== null) {
      // Save current Z position as Z1 using the proper API
      apiObjectiveControllerSetPositions({
        z1: currentPositions.Z,
        isBlocking: false,
      })
        .then((data) => {
          dispatch(objectiveSlice.setPosZ1(currentPositions.Z));
          alert(`Z1 position set to: ${currentPositions.Z}`);
        })
        .catch((err) => {
          console.error("Error setting Z1:", err);
        });
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Step 5: Calibrate Focus Position Z1
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Switch to objective 2, find the same feature (or a similar one), 
        achieve sharp focus, and save the Z1 focus position.
      </Alert>

      <Grid container spacing={3}>
        {/* Live View */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Live View with XYZ Controls
            </Typography>
            <Box sx={{ height: '400px', border: '1px solid #ccc' }}>
              <LiveViewControlWrapper />
            </Box>
            {/* XYZ Joystick Controls */}
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
              <PositionControllerComponent />
            </Box>
          </Paper>
        </Grid>

        {/* Controls */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              1. Switch to Objective 2
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Switch to the second objective to calibrate its focus position.
            </Typography>
            <Button
              variant="contained"
              color="secondary"
              onClick={handleSwitchToObjective2}
              disabled={movedToObjective2}
            >
              {movedToObjective2 ? "On Objective 2 ✓" : "Switch to Objective 2"}
            </Button>
          </Paper>

          <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              <FocusIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              2. Find Focus
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Use the XYZ joystick controls to:
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              • Navigate to the same feature you used for Z0 calibration (or a similar feature)
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              • Adjust the Z position to achieve sharp focus with objective 2
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              • The focus position may be different due to objective variations
            </Typography>
          </Paper>

          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              <SaveIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              3. Save Z1 Focus Position
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Current Positions:
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              X: <strong>{currentPositions.X || "Unknown"}</strong>, 
              Y: <strong>{currentPositions.Y || "Unknown"}</strong>
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Z: <strong>{currentPositions.Z || "Unknown"}</strong>, 
              A: <strong>{currentPositions.A || "Unknown"}</strong>
            </Typography>
            <Button
              variant="contained"
              color="success"
              onClick={handleSaveZ1Position}
              disabled={!movedToObjective2 || currentPositions.Z === null}
              startIcon={<SaveIcon />}
            >
              Save Current Z as Z1
            </Button>
          </Paper>

          {/* Summary of calibrated positions */}
          <Paper elevation={1} sx={{ p: 2, mt: 2, backgroundColor: '#f8f9fa' }}>
            <Typography variant="h6" gutterBottom>
              Calibration Summary
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              X0 (Slot 1): <strong>{objectiveState.posX0 || "Not set"}</strong>
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              X1 (Slot 2): <strong>{objectiveState.posX1 || "Not set"}</strong>
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Z0 (Focus 1): <strong>{objectiveState.posZ0 || "Not set"}</strong>
            </Typography>
            <Typography variant="body2">
              Z1 (Focus 2): <strong>{objectiveState.posZ1 || "Not set"}</strong>
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      <Alert severity="success" sx={{ mt: 3 }}>
        <strong>Almost Done!</strong> Once you save the Z1 position, 
        you'll have completed the full objective calibration process.
      </Alert>
    </Box>
  );
};

export default WizardStep5;