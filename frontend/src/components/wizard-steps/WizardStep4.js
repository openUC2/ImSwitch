import { useState, useEffect } from "react";
import { useDispatch } from "react-redux";
import {
  Typography,
  Box,
  Paper,
  Button,
  Grid,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import {
  CheckCircle as CheckIcon,
  Save as SaveIcon,
  Visibility as FocusIcon,
} from "@mui/icons-material";
import LiveViewControlWrapper from "../../axon/LiveViewControlWrapper";
import PositionControllerComponent from "../../axon/PositionControllerComponent";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice.js";

import apiObjectiveControllerMoveToObjective from "../../backendapi/apiObjectiveControllerMoveToObjective.js";
import apiPositionerControllerGetPositions from "../../backendapi/apiPositionerControllerGetPositions.js";
import apiObjectiveControllerSetPositions from "../../backendapi/apiObjectiveControllerSetPositions.js";

const WizardStep4 = ({
  hostIP,
  hostPort,
  onNext,
  onBack,
  activeStep,
  totalSteps,
}) => {
  const dispatch = useDispatch();

  const [setupComplete, setSetupComplete] = useState(false);
  const [movedToObjective1, setMovedToObjective1] = useState(false);
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

  const handleSwitchToObjective1 = () => {
    apiObjectiveControllerMoveToObjective(1)
      .then((data) => {
        dispatch(objectiveSlice.setCurrentObjective(1));
        setMovedToObjective1(true);
        // Small delay to ensure position is updated
        setTimeout(fetchCurrentPositions, 1000);
      })
      .catch((err) => {
        console.error("Error switching to objective 1:", err);
      });
  };

  const handleSaveZ0Position = () => {
    if (currentPositions.Z !== null) {
      // Save current Z position as Z0 using the proper API
      apiObjectiveControllerSetPositions({
        z0: currentPositions.Z,
        isBlocking: false,
      })
        .then((data) => {
          dispatch(objectiveSlice.setPosZ0(currentPositions.Z));
          alert(`Z0 position set to: ${currentPositions.Z}`);
        })
        .catch((err) => {
          console.error("Error setting Z0:", err);
        });
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Step 4: Calibrate Focus Position Z0
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Insert objectives into slots 1 and 2, add a sample, switch to objective
        1, find a feature to focus on, and save the Z0 focus position.
      </Alert>

      <Grid container spacing={3}>
        {/* Live View */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Live View with XYZ Controls
            </Typography>
            <Box sx={{ height: "400px", border: "1px solid #ccc" }}>
              <LiveViewControlWrapper />
            </Box>
            {/* XYZ Joystick Controls */}
            <Box sx={{ mt: 2, display: "flex", justifyContent: "center" }}>
              <PositionControllerComponent />
            </Box>
          </Paper>
        </Grid>

        {/* Setup and Controls */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              1. Hardware Setup
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon color={setupComplete ? "success" : "disabled"} />
                </ListItemIcon>
                <ListItemText
                  primary="Insert objective lenses into slots 1 and 2"
                  secondary="Screw in both objectives securely"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon color={setupComplete ? "success" : "disabled"} />
                </ListItemIcon>
                <ListItemText
                  primary="Insert a sample"
                  secondary="Place a sample with visible features under the objectives"
                />
              </ListItem>
            </List>
            <Button
              variant="contained"
              color="primary"
              onClick={() => setSetupComplete(true)}
              disabled={setupComplete}
            >
              {setupComplete ? "Setup Complete ✓" : "Mark Setup Complete"}
            </Button>
          </Paper>

          <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              2. Switch to Objective 1
            </Typography>
            <Button
              variant="contained"
              color="secondary"
              onClick={handleSwitchToObjective1}
              disabled={!setupComplete || movedToObjective1}
            >
              {movedToObjective1 ? "On Objective 1 ✓" : "Switch to Objective 1"}
            </Button>
          </Paper>

          <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              <FocusIcon sx={{ mr: 1, verticalAlign: "middle" }} />
              3. Find Focus
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Use the XYZ joystick controls below the live view to:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="• Move X/Y to find a feature on your sample"
                  secondary="Use the arrow buttons for X/Y movement"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="• Adjust Z to achieve sharp focus"
                  secondary="Use the +/- buttons for Z movement (focus)"
                />
              </ListItem>
            </List>
          </Paper>

          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              <SaveIcon sx={{ mr: 1, verticalAlign: "middle" }} />
              4. Save Z0 Focus Position
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Current Positions:
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              X: <strong>{currentPositions.X || "Unknown"}</strong>, Y:{" "}
              <strong>{currentPositions.Y || "Unknown"}</strong>
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Z: <strong>{currentPositions.Z || "Unknown"}</strong>, A:{" "}
              <strong>{currentPositions.A || "Unknown"}</strong>
            </Typography>
            <Button
              variant="contained"
              color="success"
              onClick={handleSaveZ0Position}
              disabled={!movedToObjective1 || currentPositions.Z === null}
              startIcon={<SaveIcon />}
            >
              Save Current Z as Z0
            </Button>
          </Paper>
        </Grid>
      </Grid>

      <Alert severity="success" sx={{ mt: 3 }}>
        <strong>Tip:</strong> Once you have a sharp focus on a feature, click
        "Save Current Z as Z0" to store this focus position for objective 1.
      </Alert>
    </Box>
  );
};

export default WizardStep4;
