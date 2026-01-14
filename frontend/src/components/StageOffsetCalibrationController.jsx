import {
  Button,
  Card,
  CardContent,
  Grid,
  Paper,
  TextField,
  Typography,
  Box,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useWebSocket } from "../context/WebSocketContext.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as stageOffsetCalibrationSlice from "../state/slices/StageOffsetCalibrationSlice.js";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";

const StageOffsetCalibration = () => {
  // Access ImSwitch backend connection settings from Redux - following Copilot Instructions
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const hostIP = connectionSettingsState.ip;
  const hostPort = connectionSettingsState.apiPort;
  const socket = useWebSocket();
  const dispatch = useDispatch();

  // Local state for joystick
  const [stepSizeXY, setStepSizeXY] = useState(100);
  const [positionerName, setPositionerName] = useState("");

  // Access global Redux state
  const stageOffsetState = useSelector(
    stageOffsetCalibrationSlice.getStageOffsetCalibrationState
  );

  // Use Redux state
  const currentOffsetX = stageOffsetState.loadedOffsetX;
  const currentOffsetY = stageOffsetState.loadedOffsetY;
  const manualOffsetX = stageOffsetState.manualOffsetX;
  const manualOffsetY = stageOffsetState.manualOffsetY;
  const knownPositionX = stageOffsetState.targetX; // Reuse targetX for known position
  const knownPositionY = stageOffsetState.targetY; // Reuse targetY for known position
  const currentDisplayX = stageOffsetState.currentX; // Display current position
  const currentDisplayY = stageOffsetState.currentY; // Display current position
  const reloadTrigger = stageOffsetState.reloadTrigger;

  // Fetch positioner name for joystick control
  useEffect(() => {
    const fetchPositionerName = async () => {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/PositionerController/getPositionerNames`
        );
        const data = await response.json();
        if (data && data.length > 0) {
          setPositionerName(data[0]);
        }
      } catch (error) {
        console.error("Error fetching positioner names:", error);
      }
    };
    fetchPositionerName();
  }, [hostIP, hostPort]);

  // Retrieve offsets and current positions on load and reload
  useEffect(() => {
    const fetchAll = async () => {
      try {
        // Fetch current offsets
        const offsetXData = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/PositionerController/getStageOffsetAxis?axis=X`
        ).then((res) => res.json());
        dispatch(
          stageOffsetCalibrationSlice.setLoadedOffsetX(
            offsetXData !== null && offsetXData !== undefined ? offsetXData : 0
          )
        );

        const offsetYData = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/PositionerController/getStageOffsetAxis?axis=Y`
        ).then((res) => res.json());
        dispatch(
          stageOffsetCalibrationSlice.setLoadedOffsetY(
            offsetYData !== null && offsetYData !== undefined ? offsetYData : 0
          )
        );

        // Fetch true positions (without offset)
        const truePosXData = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/PositionerController/getTruePositionerPositionWithoutOffset?axis=X`
        ).then((res) => res.json());
        dispatch(
          stageOffsetCalibrationSlice.setCurrentX(
            truePosXData !== null && truePosXData !== undefined ? truePosXData : 0
          )
        );

        const truePosYData = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/PositionerController/getTruePositionerPositionWithoutOffset?axis=Y`
        ).then((res) => res.json());
        dispatch(
          stageOffsetCalibrationSlice.setCurrentY(
            truePosYData !== null && truePosYData !== undefined ? truePosYData : 0
          )
        );
      } catch (error) {
        console.error("Error fetching offset calibration data:", error);
      }
    };
    fetchAll();
  }, [hostIP, hostPort, reloadTrigger, dispatch]);

  // Poll for position updates every 1 second when component is active
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const truePosXData = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/PositionerController/getTruePositionerPositionWithoutOffset?axis=X`
        ).then((res) => res.json());
        dispatch(
          stageOffsetCalibrationSlice.setCurrentX(
            truePosXData !== null && truePosXData !== undefined ? truePosXData : 0
          )
        );

        const truePosYData = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/PositionerController/getTruePositionerPositionWithoutOffset?axis=Y`
        ).then((res) => res.json());
        dispatch(
          stageOffsetCalibrationSlice.setCurrentY(
            truePosYData !== null && truePosYData !== undefined ? truePosYData : 0
          )
        );
      } catch (error) {
        console.error("Error polling position:", error);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [hostIP, hostPort, dispatch]);

  // Move stage
  const moveStage = (axis, distance) => {
    fetch(
      `${hostIP}:${hostPort}/imswitch/api/PositionerController/movePositioner?axis=${axis}&dist=${distance}&isAbsolute=false&isBlocking=false`
    )
      .then((res) => res.json())
      .catch(console.error);
  };

  // Joystick movement handler
  const handleJoystickClick = (direction, stepSize) => {
    switch (direction) {
      case "Y+":
        moveStage("Y", stepSize);
        break;
      case "Y-":
        moveStage("Y", -stepSize);
        break;
      case "X+":
        moveStage("X", stepSize);
        break;
      case "X-":
        moveStage("X", -stepSize);
        break;
      default:
        console.log("Unknown direction:", direction);
    }
  };

  // Use Case 1: Submit known offset directly
  const submitKnownOffset = (axis) => {
    const offset = axis === "X" ? manualOffsetX : manualOffsetY;
    if (offset === "" || offset === null || offset === undefined) {
      alert(`Please enter an offset value for ${axis} axis`);
      return;
    }
    fetch(
      `${hostIP}:${hostPort}/imswitch/api/PositionerController/setStageOffsetAxis?knownOffset=${offset}&axis=${axis}`
    )
      .then((res) => res.json())
      .then(() => {
        console.log(`Offset for ${axis} set to ${offset}`);
        dispatch(stageOffsetCalibrationSlice.incrementReloadTrigger());
      })
      .catch(console.error);
  };

  // Use Case 2: Submit known position (backend calculates offset)
  const submitKnownPosition = (axis) => {
    const knownPos = axis === "X" ? knownPositionX : knownPositionY;
    if (knownPos === "" || knownPos === null || knownPos === undefined) {
      alert(`Please enter a known position value for ${axis} axis`);
      return;
    }
    fetch(
      `${hostIP}:${hostPort}/imswitch/api/PositionerController/setStageOffsetAxis?knownPosition=${knownPos}&axis=${axis}`
    )
      .then((res) => res.json())
      .then(() => {
        console.log(`Known position for ${axis} set to ${knownPos}`);
        dispatch(stageOffsetCalibrationSlice.incrementReloadTrigger());
      })
      .catch(console.error);
  };

  // Reset offset for a specific axis
  const resetStageOffset = (axis) => {
    fetch(
      `${hostIP}:${hostPort}/imswitch/api/PositionerController/resetStageOffsetAxis?axis=${axis}`
    )
      .then((res) => res.json())
      .then(() => {
        console.log(`Offset for ${axis} reset to 0`);
        dispatch(stageOffsetCalibrationSlice.incrementReloadTrigger());
      })
      .catch(console.error);
  };
  return (
    <Paper style={{ padding: "20px", margin: "20px" }}>
      <Typography variant="h4" gutterBottom>
        Stage Offset Calibration
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Calibrate the stage offset to align physical coordinates with the microscope's coordinate system.
      </Typography>

      <Grid container spacing={3}>
        {/* Live Stream - Full Width */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Live Stream
              </Typography>
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  maxHeight: "500px",
                  overflow: "hidden",
                }}
              >
                <LiveViewControlWrapper useFastMode={true} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Joystick Controls - Left Side */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Stage Control
              </Typography>
              
              {/* Current Position Display */}
              <Box sx={{ mb: 3, p: 2, bgcolor: "background.paper", borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Current True Position (without offset):
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      label="True Position X (µm)"
                      value={currentDisplayX}
                      InputProps={{ readOnly: true }}
                      fullWidth
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="True Position Y (µm)"
                      value={currentDisplayY}
                      InputProps={{ readOnly: true }}
                      fullWidth
                      size="small"
                    />
                  </Grid>
                </Grid>
              </Box>

              {/* Step Size Control */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>XY Step Size</InputLabel>
                <Select
                  value={stepSizeXY}
                  onChange={(e) => setStepSizeXY(e.target.value)}
                  label="XY Step Size"
                >
                  <MenuItem value={1000}>1000 µm</MenuItem>
                  <MenuItem value={100}>100 µm</MenuItem>
                  <MenuItem value={10}>10 µm</MenuItem>
                  <MenuItem value={1}>1 µm</MenuItem>
                </Select>
              </FormControl>

              {/* SVG Joystick */}
              <Box display="flex" justifyContent="center" marginTop={2}>
                <svg
                  width="280"
                  height="220"
                  viewBox="0 0 280 220"
                  xmlns="http://www.w3.org/2000/svg"
                  style={{ border: "1px solid #ccc", borderRadius: "8px" }}
                >
                  <defs>
                    <style>
                      {`
                        .joyStd { stroke: black; stroke-width: 1; filter: url(#joyF1); cursor: pointer; }
                        .joyStd:hover { fill: orange; }
                        .joyScl { font-family: helvetica; stroke: white; stroke-width: 1; fill: white; pointer-events: none; }
                      `}
                    </style>
                    <filter id="joyF1" x="-1" y="-1" width="300%" height="300%">
                      <feOffset result="offOut" in="SourceAlpha" dx="3" dy="3" />
                      <feGaussianBlur result="blurOut" in="offOut" stdDeviation="4" />
                      <feBlend in="SourceGraphic" in2="blurOut" mode="normal" />
                    </filter>
                  </defs>

                  {/* XY Movement Rings - Outermost */}
                  <g fill="#c0c0c0" transform="translate(10, 10)">
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("Y-", stepSizeXY)}
                    >
                      <path
                        className="joyStd"
                        d="M-50 -56 L-63,-69 A94,94 0 0,1 63,-69 L50,-56 A75,75 0 0,0 -50,-56 z"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("X+", stepSizeXY)}
                    >
                      <path
                        className="joyStd"
                        d="M56,-50 L69,-63 A94,94 0 0,1 69,63 L56,50 A75,75 0 0,0 56,-50"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("Y+", stepSizeXY)}
                    >
                      <path
                        className="joyStd"
                        d="M-50,56 L-63,69 A94,94 0 0,0 63,69 L50,56 A75,75 0 0,1 -50,56 z"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("X-", stepSizeXY)}
                    >
                      <path
                        className="joyStd"
                        d="M-56,-50 L-69,-63 A94,94 0 0,0 -69,63 L-56,50 A75,75 0 0,1 -56,-50 z"
                      />
                    </g>
                  </g>

                  {/* XY Movement Rings - Middle */}
                  <g fill="#d0d0d0" transform="translate(10, 10)">
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("Y-", stepSizeXY / 10)}
                    >
                      <path
                        className="joyStd"
                        d="M-37 -43 L-50,-56 A75,75 0 0,1 50,-56 L37,-43 A56,56 0 0,0 -37,-43 z"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("X+", stepSizeXY / 10)}
                    >
                      <path
                        className="joyStd"
                        d="M43 37 L56,50 A75,75 0 0,0 56,-50 L43,-37 A56,56 0 0,1 43,37 z"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("Y+", stepSizeXY / 10)}
                    >
                      <path
                        className="joyStd"
                        d="M-37 43 L-50,56 A75,75 0 0,0 50,56 L37,43 A56,56 0 0,1 -37,43 z"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("X-", stepSizeXY / 10)}
                    >
                      <path
                        className="joyStd"
                        d="M-43 37 L-56,50 A75,75 0 0,1 -56,-50 L-43,-37 A56,56 0 0,0 -43,37 z"
                      />
                    </g>
                  </g>

                  {/* XY Movement Rings - Inner */}
                  <g fill="#e0e0e0" transform="translate(10, 10)">
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("Y-", stepSizeXY / 100)}
                    >
                      <path
                        className="joyStd"
                        d="M-23 -29 L-37,-43 A56,56 0 0,1 37,-43 L23,-29 A37,37 0 0,0 -23,-29 z"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("X+", stepSizeXY / 100)}
                    >
                      <path
                        className="joyStd"
                        d="M29 -23 L43,-37 A56,56 0 0,1 43,37 L29,23 A37,37 0 0,0 29,-23 z"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("Y+", stepSizeXY / 100)}
                    >
                      <path
                        className="joyStd"
                        d="M-23 29 L-37,43 A56,56 0 0,0 37,43 L23,29 A37,37 0 0,1 -23,29 z"
                      />
                    </g>
                    <g
                      transform="translate(100 100)"
                      onClick={() => handleJoystickClick("X-", stepSizeXY / 100)}
                    >
                      <path
                        className="joyStd"
                        d="M-29 -23 L-43,-37 A56,56 0 0,0 -43,37 L-29,23 A37,37 0 0,1 -29,-23 z"
                      />
                    </g>
                  </g>

                  {/* Direction indicators */}
                  <g
                    pointerEvents="none"
                    fontWeight="900"
                    fontSize="11"
                    fillOpacity=".6"
                  >
                    <path
                      d="M100,15 l14,14 h-8 v9 h-12 v-9 h-8 z"
                      fill="SteelBlue"
                      transform="translate(10, 10)"
                    />
                    <path
                      d="M100,185 l14,-14 h-8 v-9 h-12 v9 h-8 z"
                      fill="SteelBlue"
                      transform="translate(10, 10)"
                    />
                    <path
                      d="M15,100 l14,14 v-8 h9 v-12 h-9 v-8 z"
                      fill="Khaki"
                      transform="translate(10, 10)"
                    />
                    <path
                      d="M185,100 l-14,-14 v8 h-9 v12 h9 v8 z"
                      fill="Khaki"
                      transform="translate(10, 10)"
                    />
                    <text x="105" y="40" fill="black">
                      {" "}
                      -Y
                    </text>
                    <text x="105" y="198" fill="black">
                      {" "}
                      +Y
                    </text>
                    <text x="32" y="114" fill="black">
                      {" "}
                      -X
                    </text>
                    <text x="175" y="114" fill="black">
                      {" "}
                      +X
                    </text>
                  </g>
                </svg>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Calibration Controls - Right Side */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Offset Calibration
              </Typography>

              {/* Current Offset Display */}
              <Box sx={{ mb: 3, p: 2, bgcolor: "background.paper", borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Current Offset Values:
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      label="Offset X (µm)"
                      value={currentOffsetX}
                      InputProps={{ readOnly: true }}
                      fullWidth
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Offset Y (µm)"
                      value={currentOffsetY}
                      InputProps={{ readOnly: true }}
                      fullWidth
                      size="small"
                    />
                  </Grid>
                </Grid>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Use Case 1: Direct Offset Entry */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                  Method 1: Enter Known Offset
                </Typography>
                <Typography variant="body2" color="textSecondary" paragraph>
                  If you know the offset value directly, enter it here and submit.
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" color="primary">
                      X-Axis Offset:
                    </Typography>
                  </Grid>
                  <Grid item xs={7}>
                    <TextField
                      label="Offset X (µm)"
                      type="number"
                      value={manualOffsetX}
                      onChange={(e) =>
                        dispatch(
                          stageOffsetCalibrationSlice.setManualOffsetX(
                            e.target.value
                          )
                        )
                      }
                      fullWidth
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={5}>
                    <Button
                      variant="contained"
                      color="primary"
                      fullWidth
                      onClick={() => submitKnownOffset("X")}
                    >
                      Submit X
                    </Button>
                  </Grid>

                  <Grid item xs={12}>
                    <Typography variant="subtitle2" color="primary">
                      Y-Axis Offset:
                    </Typography>
                  </Grid>
                  <Grid item xs={7}>
                    <TextField
                      label="Offset Y (µm)"
                      type="number"
                      value={manualOffsetY}
                      onChange={(e) =>
                        dispatch(
                          stageOffsetCalibrationSlice.setManualOffsetY(
                            e.target.value
                          )
                        )
                      }
                      fullWidth
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={5}>
                    <Button
                      variant="contained"
                      color="primary"
                      fullWidth
                      onClick={() => submitKnownOffset("Y")}
                    >
                      Submit Y
                    </Button>
                  </Grid>
                </Grid>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Use Case 2: Known Position Entry */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                  Method 2: Enter Known Position
                </Typography>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Move to a known physical position, enter that position, and the backend will calculate the offset.
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" color="secondary">
                      X-Axis Known Position:
                    </Typography>
                  </Grid>
                  <Grid item xs={7}>
                    <TextField
                      label="Known Position X (µm)"
                      type="number"
                      value={knownPositionX}
                      onChange={(e) =>
                        dispatch(
                          stageOffsetCalibrationSlice.setTargetX(e.target.value)
                        )
                      }
                      fullWidth
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={5}>
                    <Button
                      variant="contained"
                      color="secondary"
                      fullWidth
                      onClick={() => submitKnownPosition("X")}
                    >
                      Submit X
                    </Button>
                  </Grid>

                  <Grid item xs={12}>
                    <Typography variant="subtitle2" color="secondary">
                      Y-Axis Known Position:
                    </Typography>
                  </Grid>
                  <Grid item xs={7}>
                    <TextField
                      label="Known Position Y (µm)"
                      type="number"
                      value={knownPositionY}
                      onChange={(e) =>
                        dispatch(
                          stageOffsetCalibrationSlice.setTargetY(e.target.value)
                        )
                      }
                      fullWidth
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={5}>
                    <Button
                      variant="contained"
                      color="secondary"
                      fullWidth
                      onClick={() => submitKnownPosition("Y")}
                    >
                      Submit Y
                    </Button>
                  </Grid>
                </Grid>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Reset Controls */}
              <Box>
                <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                  Reset Offsets
                </Typography>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Reset the offset to zero for each axis independently.
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Button
                      variant="outlined"
                      color="error"
                      fullWidth
                      onClick={() => resetStageOffset("X")}
                    >
                      Reset X Offset
                    </Button>
                  </Grid>
                  <Grid item xs={6}>
                    <Button
                      variant="outlined"
                      color="error"
                      fullWidth
                      onClick={() => resetStageOffset("Y")}
                    >
                      Reset Y Offset
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default StageOffsetCalibration;
