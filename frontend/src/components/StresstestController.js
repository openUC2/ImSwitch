import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Paper,
  Tabs,
  Tab,
  Box,
  Typography,
  TextField,
  Button,
  Slider,
  Grid,
  Switch,
  FormControlLabel,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  LinearProgress,
} from "@mui/material";
import { green, red } from "@mui/material/colors";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import * as stresstestSlice from "../state/slices/StresstestSlice.js";
import apiStresstestControllerGetParams from "../backendapi/apiStresstestControllerGetParams.js";
import apiStresstestControllerSetParams from "../backendapi/apiStresstestControllerSetParams.js";
import apiStresstestControllerGetResults from "../backendapi/apiStresstestControllerGetResults.js";
import apiStresstestControllerStart from "../backendapi/apiStresstestControllerStart.js";
import apiStresstestControllerStop from "../backendapi/apiStresstestControllerStop.js";

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );
};

const StresstestController = ({ WindowTitle }) => {
  // Redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const stresstestState = useSelector(stresstestSlice.getStresstestState);

  // Destructure state for easier access
  const {
    // Parameters
    minPosX,
    maxPosX,
    minPosY,
    maxPosY,
    numRandomPositions,
    numCycles,
    timeInterval,
    illuminationIntensity,
    exposureTime,
    saveImages,
    outputPath,
    enableImageBasedError,
    numImagesPerPosition,
    imageRegistrationMethod,
    pixelSizeUM,

    // Results
    totalPositions,
    completedPositions,
    averagePositionError,
    maxPositionError,
    positionErrors,
    isRunning,
    averageImageError,
    maxImageError,

    // UI
    tabIndex,
  } = stresstestState;

  // Fetch parameters and results on component mount and periodically
  useEffect(() => {
    const fetchParams = async () => {
      try {
        const params = await apiStresstestControllerGetParams();
        dispatch(stresstestSlice.setStresstestParams(params));
      } catch (error) {
        console.error("Error fetching stresstest parameters:", error);
      }
    };

    const fetchResults = async () => {
      try {
        const results = await apiStresstestControllerGetResults();
        dispatch(stresstestSlice.setStresstestResults(results));
      } catch (error) {
        console.error("Error fetching stresstest results:", error);
      }
    };

    // Initial fetch
    fetchParams();
    fetchResults();

    // Set up periodic fetch for results while running
    const interval = setInterval(() => {
      fetchResults();
    }, 2000); // Every 2 seconds

    return () => clearInterval(interval);
  }, [dispatch]);

  // Handle parameter updates
  const updateParameters = async () => {
    try {
      const params = {
        minPosX,
        maxPosX,
        minPosY,
        maxPosY,
        numRandomPositions,
        numCycles,
        timeInterval,
        illuminationIntensity,
        exposureTime,
        saveImages,
        outputPath,
        enableImageBasedError,
        numImagesPerPosition,
        imageRegistrationMethod,
        pixelSizeUM,
      };

      await apiStresstestControllerSetParams(params);
      console.log("Parameters updated successfully");
    } catch (error) {
      console.error("Error updating parameters:", error);
    }
  };

  // Start stresstest
  const startStresstest = async () => {
    try {
      await updateParameters(); // Update parameters first
      await apiStresstestControllerStart();
      console.log("Stresstest started");
    } catch (error) {
      console.error("Error starting stresstest:", error);
    }
  };

  // Stop stresstest
  const stopStresstest = async () => {
    try {
      await apiStresstestControllerStop();
      console.log("Stresstest stopped");
    } catch (error) {
      console.error("Error stopping stresstest:", error);
    }
  };

  const handleTabChange = (event, newValue) => {
    dispatch(stresstestSlice.setTabIndex(newValue));
  };

  // Calculate progress percentage
  const progressPercentage =
    totalPositions > 0 ? (completedPositions / totalPositions) * 100 : 0;

  return (
    <Paper>
      <Typography variant="h6" gutterBottom>
        {WindowTitle || "Stresstest Controller"}
      </Typography>

      <Tabs
        value={tabIndex}
        onChange={handleTabChange}
        aria-label="stresstest settings tabs"
      >
        <Tab label="Position Parameters" />
        <Tab label="Acquisition Settings" />
        <Tab label="Results & Status" />
      </Tabs>

      {/* Position Parameters Tab */}
      <TabPanel value={tabIndex} index={0}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="h6">Position Range</Typography>
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Min X Position (μm)"
              type="number"
              value={minPosX}
              onChange={(e) =>
                dispatch(stresstestSlice.setMinPosX(parseFloat(e.target.value)))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Max X Position (μm)"
              type="number"
              value={maxPosX}
              onChange={(e) =>
                dispatch(stresstestSlice.setMaxPosX(parseFloat(e.target.value)))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Min Y Position (μm)"
              type="number"
              value={minPosY}
              onChange={(e) =>
                dispatch(stresstestSlice.setMinPosY(parseFloat(e.target.value)))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Max Y Position (μm)"
              type="number"
              value={maxPosY}
              onChange={(e) =>
                dispatch(stresstestSlice.setMaxPosY(parseFloat(e.target.value)))
              }
              fullWidth
            />
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6">Test Configuration</Typography>
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Random Positions per Cycle"
              type="number"
              value={numRandomPositions}
              onChange={(e) =>
                dispatch(
                  stresstestSlice.setNumRandomPositions(
                    parseInt(e.target.value)
                  )
                )
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Number of Cycles"
              type="number"
              value={numCycles}
              onChange={(e) =>
                dispatch(stresstestSlice.setNumCycles(parseInt(e.target.value)))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Time Interval (s)"
              type="number"
              value={timeInterval}
              onChange={(e) =>
                dispatch(
                  stresstestSlice.setTimeInterval(parseFloat(e.target.value))
                )
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              label="Output Path"
              value={outputPath}
              onChange={(e) =>
                dispatch(stresstestSlice.setOutputPath(e.target.value))
              }
              fullWidth
            />
          </Grid>
        </Grid>
      </TabPanel>

      {/* Acquisition Settings Tab */}
      <TabPanel value={tabIndex} index={1}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="h6">Camera & Illumination</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography>
              Illumination Intensity: {illuminationIntensity}%
            </Typography>
            <Slider
              value={illuminationIntensity}
              onChange={(e, value) =>
                dispatch(stresstestSlice.setIlluminationIntensity(value))
              }
              max={100}
              step={1}
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Exposure Time (s)"
              type="number"
              value={exposureTime}
              onChange={(e) =>
                dispatch(
                  stresstestSlice.setExposureTime(parseFloat(e.target.value))
                )
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={saveImages}
                  onChange={(e) =>
                    dispatch(stresstestSlice.setSaveImages(e.target.checked))
                  }
                />
              }
              label="Save Images"
            />
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6">Image-Based Error Estimation</Typography>
          </Grid>
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={enableImageBasedError}
                  onChange={(e) =>
                    dispatch(
                      stresstestSlice.setEnableImageBasedError(e.target.checked)
                    )
                  }
                />
              }
              label="Enable Image-Based Error Estimation"
            />
          </Grid>
          {enableImageBasedError && (
            <>
              <Grid item xs={6}>
                <TextField
                  label="Images per Position"
                  type="number"
                  value={numImagesPerPosition}
                  onChange={(e) =>
                    dispatch(
                      stresstestSlice.setNumImagesPerPosition(
                        parseInt(e.target.value)
                      )
                    )
                  }
                  fullWidth
                />
              </Grid>
              <Grid item xs={6}>
                <FormControl fullWidth>
                  <InputLabel>Registration Method</InputLabel>
                  <Select
                    value={imageRegistrationMethod}
                    onChange={(e) =>
                      dispatch(
                        stresstestSlice.setImageRegistrationMethod(
                          e.target.value
                        )
                      )
                    }
                  >
                    <MenuItem value="fft">FFT</MenuItem>
                    <MenuItem value="correlation">Correlation</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Pixel Size (μm)"
                  type="number"
                  value={pixelSizeUM}
                  onChange={(e) =>
                    dispatch(
                      stresstestSlice.setPixelSizeUM(parseFloat(e.target.value))
                    )
                  }
                  fullWidth
                />
              </Grid>
            </>
          )}
        </Grid>
      </TabPanel>

      {/* Results & Status Tab */}
      <TabPanel value={tabIndex} index={2}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Box display="flex" alignItems="center" gap={2}>
              {isRunning ? (
                <CancelIcon style={{ color: red[500] }} />
              ) : (
                <CheckCircleIcon style={{ color: green[500] }} />
              )}
              <Typography variant="h6" color={isRunning ? "error" : "primary"}>
                Status: {isRunning ? "Running" : "Stopped"}
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box display="flex" gap={2}>
              <Button
                variant="contained"
                color="primary"
                onClick={startStresstest}
                disabled={isRunning}
              >
                Start Stresstest
              </Button>
              <Button
                variant="contained"
                color="secondary"
                onClick={stopStresstest}
                disabled={!isRunning}
              >
                Stop Stresstest
              </Button>
              <Button variant="outlined" onClick={updateParameters}>
                Update Parameters
              </Button>
            </Box>
          </Grid>

          {totalPositions > 0 && (
            <>
              <Grid item xs={12}>
                <Typography variant="body1">
                  Progress: {completedPositions} / {totalPositions} positions
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={progressPercentage}
                />
              </Grid>
            </>
          )}

          <Grid item xs={12}>
            <Typography variant="h6">Position Error Results</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              Average Position Error: {averagePositionError.toFixed(3)} μm
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              Max Position Error: {maxPositionError.toFixed(3)} μm
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              Total Measurements: {positionErrors.length}
            </Typography>
          </Grid>

          {enableImageBasedError && (
            <>
              <Grid item xs={12}>
                <Typography variant="h6">Image-Based Error Results</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Average Image Error: {averageImageError.toFixed(3)} μm
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  Max Image Error: {maxImageError.toFixed(3)} μm
                </Typography>
              </Grid>
            </>
          )}
        </Grid>
      </TabPanel>
    </Paper>
  );
};

export default StresstestController;
