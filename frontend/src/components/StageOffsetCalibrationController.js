import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Paper,
  TextField,
  Button,
  Grid,
  Typography,
  Card,
  CardContent,
  Box,
} from "@mui/material";
import { useWebSocket } from "../context/WebSocketContext";
import * as stageOffsetCalibrationSlice from "../state/slices/StageOffsetCalibrationSlice.js";
import * as stageCenterCalibrationSlice from "../state/slices/StageCenterCalibrationSlice.js";
import StageCenterCalibrationWizard from "./StageCenterCalibrationWizard";
import JoystickController from "./JoystickController";


const StageOffsetCalibration = ({ hostIP, hostPort }) => {
  const socket = useWebSocket(); 
  const dispatch = useDispatch();
  
  // Access global Redux state
  const stageOffsetState = useSelector(stageOffsetCalibrationSlice.getStageOffsetCalibrationState);
  
  // Use Redux state instead of local useState
  const currentX = stageOffsetState.currentX;
  const currentY = stageOffsetState.currentY;
  const targetX = stageOffsetState.targetX;
  const targetY = stageOffsetState.targetY;
  const calculatedOffsetX = stageOffsetState.calculatedOffsetX;
  const calculatedOffsetY = stageOffsetState.calculatedOffsetY;
  const loadedOffsetX = stageOffsetState.loadedOffsetX;
  const loadedOffsetY = stageOffsetState.loadedOffsetY;
  const manualOffsetX = stageOffsetState.manualOffsetX;
  const manualOffsetY = stageOffsetState.manualOffsetY;
  const imageUrls = stageOffsetState.imageUrls;
  const detectors = stageOffsetState.detectors;
  const reloadTrigger = stageOffsetState.reloadTrigger;

  // Fetch the list of detectors from the server
  useEffect(() => {
    const fetchDetectorNames = async () => {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/SettingsController/getDetectorNames`
        );
        const data = await response.json();
        dispatch(stageOffsetCalibrationSlice.setDetectors(data || [])); // Set detectors or default to an empty array
      } catch (error) {
        console.error("Error fetching detector names:", error);
      }
    };

    fetchDetectorNames();
  }, [hostIP, hostPort, dispatch]);

  // Retrieve offsets + current position on first load
  useEffect(() => {
    const fetchAll = async () => {
      try {
        // Offsets
        const offsetXData = await fetch(
          `${hostIP}:${hostPort}/PositionerController/getStageOffsetAxis?axis=X`
        ).then((res) => res.json());
        dispatch(stageOffsetCalibrationSlice.setLoadedOffsetX(offsetXData ?? ""));

        const offsetYData = await fetch(
          `${hostIP}:${hostPort}/PositionerController/getStageOffsetAxis?axis=Y`
        ).then((res) => res.json());
        dispatch(stageOffsetCalibrationSlice.setLoadedOffsetY(offsetYData ?? ""));

        // Current Positions
        const posData = await fetch(
          `${hostIP}:${hostPort}/PositionerController/getPositionerPositions`
        ).then((res) => res.json());
        if (posData.ESP32Stage) {
          dispatch(stageOffsetCalibrationSlice.setCurrentX(posData.ESP32Stage.X));
          dispatch(stageOffsetCalibrationSlice.setCurrentY(posData.ESP32Stage.Y));
        }
        else if (posData.VirtualStage) {
          dispatch(stageOffsetCalibrationSlice.setCurrentX(posData.VirtualStage.X));
          dispatch(stageOffsetCalibrationSlice.setCurrentY(posData.VirtualStage.Y));
        }
      } catch (error) {
        console.error(error);
      }
    };
    fetchAll();
  }, [hostIP, hostPort, reloadTrigger]);

  // Auto-calc offset from current vs target
  useEffect(() => {
    const cX = parseFloat(currentX);
    const tX = parseFloat(targetX);
    if (!isNaN(cX) && !isNaN(tX)) {
      dispatch(stageOffsetCalibrationSlice.setCalculatedOffsetX(tX - cX));
    } else {
      dispatch(stageOffsetCalibrationSlice.setCalculatedOffsetX(""));
    }
  }, [currentX, targetX]);

  useEffect(() => {
    const cY = parseFloat(currentY);
    const tY = parseFloat(targetY);
    if (!isNaN(cY) && !isNaN(tY)) {
      dispatch(stageOffsetCalibrationSlice.setCalculatedOffsetY(tY - cY));
    } else {
      dispatch(stageOffsetCalibrationSlice.setCalculatedOffsetY(""));
    }
  }, [currentY, targetY]);

  // Handle socket signals
  useEffect(() => {
    if (!socket) return;
    const handleSignal = (rawData) => {
      try {
        const jdata = JSON.parse(rawData);
        if (jdata.name === "sigUpdateImage") {
          const detectorName = jdata.detectorname;
          const imgSrc = `data:image/jpeg;base64,${jdata.image}`;
          dispatch(stageOffsetCalibrationSlice.updateImageUrl({
            detector: detectorName,
            url: imgSrc
          }));
          if (jdata.pixelsize) {
            // If needed: dispatch(objectiveSlice.setPixelSize(jdata.pixelsize));
          }
        }
      } catch (error) {
        console.error("Error parsing signal data:", error);
      }
    };
    socket.on("signal", handleSignal);
    return () => {
      socket.off("signal", handleSignal);
    };
  }, [socket, dispatch]);

  // Move stage
  const moveStage = (axis, distance) => {
    fetch(
      `${hostIP}:${hostPort}/PositionerController/movePositioner?axis=${axis}&dist=${distance}&isAbsolute=false&isBlocking=false`
    )
      .then((res) => res.json())
      .catch(console.error);
  };

  // Fetch current position manually
  const fetchCurrentPosition = () => {
    fetch(`${hostIP}:${hostPort}/PositionerController/getPositionerPositions`)
      .then((res) => res.json())
      .then((data) => {
        if (data.ESP32Stage) {
          dispatch(stageOffsetCalibrationSlice.setCurrentX(data.ESP32Stage.X));
          dispatch(stageOffsetCalibrationSlice.setCurrentY(data.ESP32Stage.Y));
        }
        else if (data.VirtualStage) {
          dispatch(stageOffsetCalibrationSlice.setCurrentX(data.VirtualStage.X));
          dispatch(stageOffsetCalibrationSlice.setCurrentY(data.VirtualStage.Y));
        }
      })
      .catch(console.error);
  };

  // Submit offsets
  const submitKnownOffset = (axis, offset) => {
    if (offset === "") return;
    fetch(
      `${hostIP}:${hostPort}/PositionerController/setStageOffsetAxis?knownOffset=${offset}&axis=${axis}`
    )
      .then((res) => res.json())
      .then(() => dispatch(stageOffsetCalibrationSlice.incrementReloadTrigger())) // Trigger reload
      .catch(console.error);
  };

  const submitPositions = (axis, knownPos, currPos) => {
    if (knownPos === "") return;
    fetch(
      `${hostIP}:${hostPort}/PositionerController/setStageOffsetAxis?knownPosition=${knownPos}&currentPosition=${currPos}&axis=${axis}`
    )
      .then((res) => res.json())
      .then(() => dispatch(stageOffsetCalibrationSlice.incrementReloadTrigger())) // Trigger reload
      .catch(console.error);
  };

  const submitKnownPosition = (axis, knownPos) => {
    if (knownPos === "") return;
    fetch(
      `${hostIP}:${hostPort}/PositionerController/setStageOffsetAxis?knownPosition=${knownPos}&axis=${axis}`
    )
      .then((res) => res.json())
      .then(() => dispatch(stageOffsetCalibrationSlice.incrementReloadTrigger())) // Trigger reload
      .catch(console.error);
  };

  const resetStageOffsets = (axis) => {
    fetch(
      `${hostIP}:${hostPort}/PositionerController/resetStageOffsetAxis?axis=${axis}`
    )
      .then((res) => res.json())
      .then(() => {
        dispatch(stageOffsetCalibrationSlice.setCalculatedOffsetX(0));
        dispatch(stageOffsetCalibrationSlice.setCalculatedOffsetY(0));
        dispatch(stageOffsetCalibrationSlice.setLoadedOffsetX(0));
        dispatch(stageOffsetCalibrationSlice.setLoadedOffsetY(0));
        dispatch(stageOffsetCalibrationSlice.incrementReloadTrigger()); // Trigger reload
      })
      .catch(console.error);
  }
  return (
    <Paper style={{ padding: "20px", margin: "20px" }}>
      <Grid container spacing={3}>
        {/* Title and Prominent Wizard Button */}
        <Grid item xs={12}>
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            mb: 3, 
            p: 3,
            background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
            borderRadius: 2,
            boxShadow: 2
          }}>
            <Typography variant="h4" gutterBottom sx={{ 
              fontWeight: 'bold', 
              background: 'linear-gradient(45deg,rgb(0, 0, 0) 30%,rgb(53, 53, 53) 90%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              mb: 2 
            }}>
              Stage Offset Calibration
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              size="large"
              onClick={() => dispatch(stageCenterCalibrationSlice.setWizardOpen(true))}
              sx={{
                fontSize: '1.2rem',
                py: 2,
                px: 4,
                background: 'linear-gradient(45deg,rgb(193, 193, 193) 30%,rgb(137, 137, 137) 90%)',
                boxShadow: '0 4px 20px 0 rgba(33, 150, 243, 0.4)',
                '&:hover': {
                  background: 'linear-gradient(45deg, rgb(134, 134, 134),rgb(181, 181, 181) 90%)',
                  boxShadow: '0 6px 25px 0 rgba(33, 150, 243, 0.6)',
                  transform: 'translateY(-2px)',
                },
                transition: 'all 0.3s ease-in-out'
              }}
            >
              🔍 Launch Stage Center Wizard
            </Button>
            <Typography variant="body2" sx={{ mt: 1, textAlign: 'center', color: 'text.secondary' }}>
              Use the guided wizard to find your stage center with automatic detection
            </Typography>
          </Box>
        </Grid>


        {/* Joystick Controller - Full Width */}
        <Grid item xs={12}>
          <JoystickController hostIP={hostIP} hostPort={hostPort} />
        </Grid>

        {/* Stage Offset Calibration */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Manual Stage Offset Calibration
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Current X"
                    value={currentX}
                    onChange={(e) => dispatch(stageOffsetCalibrationSlice.setCurrentX(e.target.value))}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Current Y"
                    value={currentY}
                    onChange={(e) => dispatch(stageOffsetCalibrationSlice.setCurrentY(e.target.value))}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Target X"
                    value={targetX}
                    onChange={(e) => dispatch(stageOffsetCalibrationSlice.setTargetX(e.target.value))}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Target Y"
                    value={targetY}
                    onChange={(e) => dispatch(stageOffsetCalibrationSlice.setTargetY(e.target.value))}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Calc Offset X"
                    value={calculatedOffsetX}
                    InputProps={{ readOnly: true }}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Calc Offset Y"
                    value={calculatedOffsetY}
                    InputProps={{ readOnly: true }}
                    fullWidth
                  />
                </Grid>

                <Grid item xs={12}>
                  <Button variant="contained" onClick={fetchCurrentPosition}>
                    Fetch Current Position
                  </Button>
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    label="Loaded Offset X"
                    value={loadedOffsetX}
                    InputProps={{ readOnly: true }}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Loaded Offset Y"
                    value={loadedOffsetY}
                    InputProps={{ readOnly: true }}
                    fullWidth
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    label="Manual Offset X"
                    value={manualOffsetX}
                    onChange={(e) => dispatch(stageOffsetCalibrationSlice.setManualOffsetX(e.target.value))}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Manual Offset Y"
                    value={manualOffsetY}
                    onChange={(e) => dispatch(stageOffsetCalibrationSlice.setManualOffsetY(e.target.value))}
                    fullWidth
                  />
                </Grid>

                {/* X axis submission */}
                <Grid item xs={12}>
                  <Typography variant="subtitle2">X-Axis Submissions</Typography>
                  <Button
                    variant="contained"
                    style={{ marginRight: 8 }}
                    onClick={() => submitKnownOffset("X", manualOffsetX || calculatedOffsetX)}
                  >
                    Submit Known Offset
                  </Button>
                  <Button
                    variant="contained"
                    style={{ marginRight: 8 }}
                    onClick={() =>
                      submitPositions(
                        "X",
                        targetX,
                        currentX
                      )
                    }
                  >
                    Submit Both Positions
                  </Button>
                  <Button
                    variant="contained"
                    onClick={() => submitKnownPosition("X", targetX)}
                  >
                    Submit Known Position
                  </Button>

                                    <Button
                    variant="outlined"
                    color="error"
                    onClick={() => resetStageOffsets("X")}
                    style={{ marginLeft: 8 }}
                  >
                    Reset X Offsets
                  </Button>
                </Grid>

                {/* Y axis submission */}
                <Grid item xs={12}>
                  <Typography variant="subtitle2">Y-Axis Submissions</Typography>
                  <Button
                    variant="contained"
                    style={{ marginRight: 8 }}
                    onClick={() => submitKnownOffset("Y", manualOffsetY || calculatedOffsetY)}
                  >
                    Submit Known Offset
                  </Button>
                  <Button
                    variant="contained"
                    style={{ marginRight: 8 }}
                    onClick={() =>
                      submitPositions(
                        "Y",
                        targetY,
                        currentY
                      )
                    }
                  >
                    Submit Both Positions
                  </Button>
                  <Button
                    variant="contained"
                    onClick={() => submitKnownPosition("Y", targetY)}
                  >
                    Submit Known Position
                  </Button>

                  <Button
                    variant="outlined"
                    color="error"
                    onClick={() => resetStageOffsets("Y")}
                    style={{ marginLeft: 8 }}
                  >
                    Reset Y Offsets
                  </Button>
                  
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Stage Center Calibration Wizard */}
      <StageCenterCalibrationWizard />
    </Paper>
  );
};

export default StageOffsetCalibration;
