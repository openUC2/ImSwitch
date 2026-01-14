// src/components/DemoController.js
import CancelIcon from "@mui/icons-material/Cancel";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import {
  Box,
  Button,
  Card,
  CardContent,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Tab,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import { green, red } from "@mui/material/colors";
import { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import apiDemoControllerGetDemoParams from "../backendapi/apiDemoControllerGetDemoParams.js";
import apiDemoControllerGetDemoResults from "../backendapi/apiDemoControllerGetDemoResults.js";
import apiDemoControllerSetDemoParams from "../backendapi/apiDemoControllerSetDemoParams.js";
import apiDemoControllerStartDemo from "../backendapi/apiDemoControllerStartDemo.js";
import apiDemoControllerStopDemo from "../backendapi/apiDemoControllerStopDemo.js";
import * as demoSlice from "../state/slices/DemoSlice.js";

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`demo-tabpanel-${index}`}
      aria-labelledby={`demo-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );
};

const DemoController = ({ title = "Demo Controller" }) => {
  const dispatch = useDispatch();
  const demoState = useSelector(demoSlice.getDemoState);

  // Destructure state for easier access
  const {
    maxRangeX,
    maxRangeY,
    scanningScheme,
    illuminationMode,
    gridRows,
    gridColumns,
    spiralShells,
    numRandomPositions,
    dwellTime,
    totalRunTime,
    isRunning,
    currentPosition,
    completedPositions,
    totalPositions,
    elapsedTime,
    remainingTime,
    tabIndex,
  } = demoState;

  // Fetch initial parameters and results on component mount
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const params = await apiDemoControllerGetDemoParams();
        dispatch(demoSlice.setDemoParams(params));

        const results = await apiDemoControllerGetDemoResults();
        dispatch(demoSlice.setDemoResults(results));
      } catch (error) {
        console.error("Error fetching initial demo data:", error);
      }
    };

    fetchInitialData();
  }, [dispatch]);

  // Periodically fetch results when demo is running
  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(async () => {
        try {
          const results = await apiDemoControllerGetDemoResults();
          dispatch(demoSlice.setDemoResults(results));
        } catch (error) {
          console.error("Error fetching demo results:", error);
        }
      }, 1000); // Update every second
    }
    return () => clearInterval(interval);
  }, [isRunning, dispatch]);

  const handleTabChange = (event, newValue) => {
    dispatch(demoSlice.setTabIndex(newValue));
  };

  const handleStartDemo = async () => {
    try {
      // First set parameters
      const params = {
        maxRangeX,
        maxRangeY,
        scanningScheme,
        illuminationMode,
        gridRows,
        gridColumns,
        spiralShells,
        numRandomPositions,
        dwellTime,
        totalRunTime,
      };
      await apiDemoControllerSetDemoParams(params);

      // Then start demo
      await apiDemoControllerStartDemo();
      dispatch(demoSlice.setIsRunning(true));
    } catch (error) {
      console.error("Error starting demo:", error);
    }
  };

  const handleStopDemo = async () => {
    try {
      await apiDemoControllerStopDemo();
      dispatch(demoSlice.setIsRunning(false));
    } catch (error) {
      console.error("Error stopping demo:", error);
    }
  };

  const handleUpdateParameters = async () => {
    try {
      const params = {
        maxRangeX,
        maxRangeY,
        scanningScheme,
        illuminationMode,
        gridRows,
        gridColumns,
        spiralShells,
        numRandomPositions,
        dwellTime,
        totalRunTime,
      };
      await apiDemoControllerSetDemoParams(params);
      console.log("Demo parameters updated successfully");
    } catch (error) {
      console.error("Error updating demo parameters:", error);
    }
  };

  return (
    <Paper>
      <Typography
        variant="h6"
        sx={{ p: 2, borderBottom: 1, borderColor: "divider" }}
      >
        {title}
      </Typography>

      <Tabs
        value={tabIndex}
        onChange={handleTabChange}
        aria-label="Demo Controller Tabs"
      >
        <Tab label="Parameters" />
        <Tab label="Live View" />
        <Tab label="Status" />
      </Tabs>

      <TabPanel value={tabIndex} index={0}>
        <Grid container spacing={3}>
          {/* Control Buttons */}
          <Grid item xs={12}>
            <Box display="flex" gap={2} mb={2}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleStartDemo}
                disabled={isRunning}
                startIcon={<PlayArrowIcon />}
              >
                Start Demo
              </Button>
              <Button
                variant="contained"
                color="secondary"
                onClick={handleStopDemo}
                disabled={!isRunning}
                startIcon={<StopIcon />}
              >
                Stop Demo
              </Button>
              <Button
                variant="outlined"
                onClick={handleUpdateParameters}
                disabled={isRunning}
              >
                Update Parameters
              </Button>
            </Box>

            {/* Status Indicator */}
            <Box display="flex" alignItems="center" mb={2}>
              <Typography variant="h6">Status: </Typography>
              {isRunning ? (
                <>
                  <CheckCircleIcon
                    style={{ color: green[500], marginLeft: "10px" }}
                  />
                  <Typography sx={{ ml: 1 }} color="success.main">
                    Running
                  </Typography>
                </>
              ) : (
                <>
                  <CancelIcon style={{ color: red[500], marginLeft: "10px" }} />
                  <Typography sx={{ ml: 1 }} color="error.main">
                    Stopped
                  </Typography>
                </>
              )}
            </Box>
          </Grid>

          {/* Range Parameters */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Scanning Range (micrometers)
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      label="Max Range X"
                      type="number"
                      value={maxRangeX}
                      onChange={(e) =>
                        dispatch(
                          demoSlice.setMaxRangeX(parseFloat(e.target.value))
                        )
                      }
                      fullWidth
                      disabled={isRunning}
                      inputProps={{ step: 10, min: 0 }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Max Range Y"
                      type="number"
                      value={maxRangeY}
                      onChange={(e) =>
                        dispatch(
                          demoSlice.setMaxRangeY(parseFloat(e.target.value))
                        )
                      }
                      fullWidth
                      disabled={isRunning}
                      inputProps={{ step: 10, min: 0 }}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Scanning Parameters */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Scanning Configuration
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <FormControl fullWidth disabled={isRunning}>
                      <InputLabel>Scanning Scheme</InputLabel>
                      <Select
                        value={scanningScheme}
                        onChange={(e) =>
                          dispatch(demoSlice.setScanningScheme(e.target.value))
                        }
                        label="Scanning Scheme"
                      >
                        <MenuItem value="random">Random</MenuItem>
                        <MenuItem value="grid">Grid</MenuItem>
                        <MenuItem value="spiral">Spiral</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl fullWidth disabled={isRunning}>
                      <InputLabel>Illumination Mode</InputLabel>
                      <Select
                        value={illuminationMode}
                        onChange={(e) =>
                          dispatch(
                            demoSlice.setIlluminationMode(e.target.value)
                          )
                        }
                        label="Illumination Mode"
                      >
                        <MenuItem value="random">Random</MenuItem>
                        <MenuItem value="continuous">Continuous</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Scheme-specific Parameters */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Scheme-specific Parameters
                </Typography>
                <Grid container spacing={2}>
                  {scanningScheme === "random" && (
                    <Grid item xs={12}>
                      <TextField
                        label="Number of Random Positions"
                        type="number"
                        value={numRandomPositions}
                        onChange={(e) =>
                          dispatch(
                            demoSlice.setNumRandomPositions(
                              parseInt(e.target.value)
                            )
                          )
                        }
                        fullWidth
                        disabled={isRunning}
                        inputProps={{ min: 1, max: 100 }}
                      />
                    </Grid>
                  )}
                  {scanningScheme === "grid" && (
                    <>
                      <Grid item xs={6}>
                        <TextField
                          label="Grid Rows"
                          type="number"
                          value={gridRows}
                          onChange={(e) =>
                            dispatch(
                              demoSlice.setGridRows(parseInt(e.target.value))
                            )
                          }
                          fullWidth
                          disabled={isRunning}
                          inputProps={{ min: 1, max: 20 }}
                        />
                      </Grid>
                      <Grid item xs={6}>
                        <TextField
                          label="Grid Columns"
                          type="number"
                          value={gridColumns}
                          onChange={(e) =>
                            dispatch(
                              demoSlice.setGridColumns(parseInt(e.target.value))
                            )
                          }
                          fullWidth
                          disabled={isRunning}
                          inputProps={{ min: 1, max: 20 }}
                        />
                      </Grid>
                    </>
                  )}
                  {scanningScheme === "spiral" && (
                    <Grid item xs={12}>
                      <TextField
                        label="Spiral Shells"
                        type="number"
                        value={spiralShells}
                        onChange={(e) =>
                          dispatch(
                            demoSlice.setSpiralShells(parseInt(e.target.value))
                          )
                        }
                        fullWidth
                        disabled={isRunning}
                        inputProps={{ min: 1, max: 10 }}
                      />
                    </Grid>
                  )}
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Timing Parameters */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Timing Parameters
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      label="Dwell Time (seconds)"
                      type="number"
                      value={dwellTime}
                      onChange={(e) =>
                        dispatch(
                          demoSlice.setDwellTime(parseFloat(e.target.value))
                        )
                      }
                      fullWidth
                      disabled={isRunning}
                      inputProps={{ step: 0.1, min: 0.1, max: 60 }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      label="Total Run Time (seconds)"
                      type="number"
                      value={totalRunTime}
                      onChange={(e) =>
                        dispatch(
                          demoSlice.setTotalRunTime(parseFloat(e.target.value))
                        )
                      }
                      fullWidth
                      disabled={isRunning}
                      inputProps={{ step: 1, min: 1, max: 3600 }}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabIndex} index={1}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Live View
            </Typography>
            <Box sx={{ border: "1px solid #ccc", minHeight: 400 }}>
              <LiveViewControlWrapper />
            </Box>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabIndex} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Demo Status
                </Typography>
                <Box display="flex" alignItems="center" mb={2}>
                  <Typography variant="body1">Status: </Typography>
                  {isRunning ? (
                    <>
                      <CheckCircleIcon
                        style={{ color: green[500], marginLeft: "10px" }}
                      />
                      <Typography sx={{ ml: 1 }} color="success.main">
                        Running
                      </Typography>
                    </>
                  ) : (
                    <>
                      <CancelIcon
                        style={{ color: red[500], marginLeft: "10px" }}
                      />
                      <Typography sx={{ ml: 1 }} color="error.main">
                        Stopped
                      </Typography>
                    </>
                  )}
                </Box>
                <Typography variant="body2" gutterBottom>
                  Progress: {completedPositions} / {totalPositions} positions
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Elapsed Time: {Math.round(elapsedTime)}s
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Remaining Time: {Math.round(remainingTime)}s
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Current Position
                </Typography>
                <Typography variant="body2" gutterBottom>
                  X: {currentPosition?.x?.toFixed(2)} μm
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Y: {currentPosition?.y?.toFixed(2)} μm
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>
    </Paper>
  );
};

export default DemoController;
