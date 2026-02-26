import React, { useEffect, useState, useCallback } from "react";
import { 
  Paper, 
  Grid, 
  TextField, 
  Button, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem,
  Typography,
  Box,
  Divider,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress
} from "@mui/material";
import Plot from "react-plotly.js";
import { useDispatch, useSelector } from "react-redux";
import * as autofocusSlice from "../state/slices/AutofocusSlice.js";
import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import apiAutofocusControllerStartLiveMonitoring from "../backendapi/apiAutofocusControllerStartLiveMonitoring.js";
import apiAutofocusControllerStopLiveMonitoring from "../backendapi/apiAutofocusControllerStopLiveMonitoring.js";
import apiAutofocusControllerSetLiveMonitoringParameters from "../backendapi/apiAutofocusControllerSetLiveMonitoringParameters.js";


const AutofocusController = ({ hostIP, hostPort }) => {
  const dispatch = useDispatch();
  
  // Local state for backend sync and status
  const [backendState, setBackendState] = useState(null);
  const [currentZ, setCurrentZ] = useState(null);
  const [positionError, setPositionError] = useState(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  
  // Access autofocus state from Redux
  const autofocusState = useSelector(autofocusSlice.getAutofocusState);
  const { 
    rangeZ, 
    resolutionZ, 
    defocusZ, 
    illuminationChannel,
    tSettle,
    isDebug,
    nGauss,
    nCropsize,
    focusAlgorithm,
    staticOffset,
    twoStage,
    isRunning, 
    plotData, 
    showPlot,
    isLiveMonitoring,
    liveFocusValue,
    liveMonitoringPeriod,
    liveMonitoringMethod,
    liveMonitoringCropsize
  } = autofocusState;
  
  // Access parameter range state for available illumination sources
  const parameterRangeState = useSelector(parameterRangeSlice.getParameterRangeState);
  const connectionSettingsState = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  
  // Get available illumination sources and find currently active ones
  // Ensure it's always an array
  const availableIlluminations = Array.isArray(parameterRangeState.illuSources) 
    ? parameterRangeState.illuSources 
    : [];
  
  // Function to get currently active illumination (first one that's on)
  const getCurrentlyActiveIllumination = async () => {
    if (availableIlluminations.length === 0) return null;
    
    const ip = connectionSettingsState.ip || hostIP;
    const port = connectionSettingsState.apiPort || hostPort;
    
    if (!ip || !port) return availableIlluminations[0]; // Fallback to first available
    
    try {
      for (const illumination of availableIlluminations) {
        const encodedName = encodeURIComponent(illumination);
        const response = await fetch(`${ip}:${port}/imswitch/api/LaserController/getLaserValue?laserName=${encodedName}`);
        if (response.ok) {
          const value = await response.json();
          if (value > 0) {
            return illumination; // Return the first active illumination
          }
        }
      }
      return availableIlluminations[0]; // Fallback to first if none active
    } catch (error) {
      console.error("Error checking active illuminations:", error);
      return availableIlluminations[0]; // Fallback to first available
    }
  };
  
  // Set default illumination channel when component mounts or illumination sources change
  useEffect(() => {
    if (availableIlluminations.length > 0 && !illuminationChannel) {
      // If selected illumination is not available, find currently active one
      if (!availableIlluminations.includes(illuminationChannel)) {
        getCurrentlyActiveIllumination().then(activeIllumination => {
          if (activeIllumination) {
            dispatch(autofocusSlice.setIlluminationChannel(activeIllumination));
          }
        });
      }
    }
  }, [availableIlluminations, illuminationChannel, dispatch]);

  // Fetch backend autofocus status on mount and periodically
  const fetchAutofocusStatus = useCallback(async () => {
    try {
      const response = await fetch(`${hostIP}:${hostPort}/imswitch/api/AutofocusController/getAutofocusStatus`);
      if (response.ok) {
        const status = await response.json();
        setBackendState(status.state);
        setCurrentZ(status.currentZ);
        setPositionError(status.positionError);
        
        // Sync running state with backend
        if (status.isRunning !== isRunning) {
          dispatch(autofocusSlice.setIsRunning(status.isRunning));
        }
        if (status.isLiveMonitoring !== isLiveMonitoring) {
          dispatch(autofocusSlice.setIsLiveMonitoring(status.isLiveMonitoring));
        }
      }
    } catch (error) {
      console.error("Error fetching autofocus status:", error);
    }
  }, [hostIP, hostPort, isRunning, isLiveMonitoring, dispatch]);

  // Fetch status on mount
  useEffect(() => {
    fetchAutofocusStatus();
  }, []);

  // Periodic status sync while running
  useEffect(() => {
    if (isRunning || isStarting) {
      const interval = setInterval(fetchAutofocusStatus, 1000);
      return () => clearInterval(interval);
    }
  }, [isRunning, isStarting, fetchAutofocusStatus]);

  const handleStart = async () => {
    // Prevent double-clicks
    if (isRunning || isStarting) {
      console.warn("Autofocus already running or starting");
      return;
    }
    
    setIsStarting(true);
    setPositionError(null);
    
    try {
      // Use selected illumination channel or fallback to currently active one
      const selectedChannel = illuminationChannel || availableIlluminations[0];
      const url = `${hostIP}:${hostPort}/imswitch/api/AutofocusController/autoFocus?rangez=${rangeZ}&resolutionz=${resolutionZ}&defocusz=${defocusZ}&illuminationChannel=${encodeURIComponent(selectedChannel || '')}&tSettle=${tSettle}&isDebug=${isDebug}&nGauss=${nGauss}&nCropsize=${nCropsize}&focusAlgorithm=${focusAlgorithm}&static_offset=${staticOffset}&twoStage=${twoStage}`;
      const response = await fetch(url, { method: "GET" });
      const result = await response.json();
      
      if (result.status === "error") {
        setPositionError(result.message);
        console.error("Autofocus start error:", result.message);
      } else if (result.status === "started") {
        dispatch(autofocusSlice.setIsRunning(true));
        dispatch(autofocusSlice.setShowPlot(false)); // Hide plot when starting a new run
        dispatch(autofocusSlice.clearPlotData());  // Clear old data
      }
    } catch (error) {
      console.error("Error starting autofocus:", error);
      setPositionError("Failed to start autofocus: " + error.message);
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    if (!isRunning || isStopping) {
      return;
    }
    
    setIsStopping(true);
    
    try {
      const url = `${hostIP}:${hostPort}/imswitch/api/AutofocusController/stopAutoFocus`;
      const response = await fetch(url, { method: "GET" });
      const result = await response.json();
      
      if (result.status === "stopped") {
        dispatch(autofocusSlice.setIsRunning(false));
        setBackendState(result.state);
      }
    } catch (error) {
      console.error("Error stopping autofocus:", error);
    } finally {
      setIsStopping(false);
      // Fetch current status to ensure sync
      setTimeout(fetchAutofocusStatus, 500);
    }
  };

  const togglePlot = () => {
    dispatch(autofocusSlice.toggleShowPlot());
  };

  const handleStartLiveMonitoring = async () => {
    try {
      const result = await apiAutofocusControllerStartLiveMonitoring(liveMonitoringPeriod, liveMonitoringMethod, liveMonitoringCropsize);
      if (result.status === "started" || result.status === "already_running") {
        dispatch(autofocusSlice.setIsLiveMonitoring(true));
      }
    } catch (error) {
      console.error("Error starting live monitoring:", error);
    }
  };

  const handleStopLiveMonitoring = async () => {
    try {
      const result = await apiAutofocusControllerStopLiveMonitoring();
      if (result.status === "stopped") {
        dispatch(autofocusSlice.setIsLiveMonitoring(false));
        dispatch(autofocusSlice.setLiveFocusValue(null));
      }
    } catch (error) {
      console.error("Error stopping live monitoring:", error);
    }
  };

  const handlePeriodChange = async (newPeriod) => {
    dispatch(autofocusSlice.setLiveMonitoringPeriod(newPeriod));
    if (isLiveMonitoring) {
      try {
        await apiAutofocusControllerSetLiveMonitoringParameters(newPeriod, null);
      } catch (error) {
        console.error("Error updating period:", error);
      }
    }
  };

  const handleMethodChange = async (newMethod) => {
    dispatch(autofocusSlice.setLiveMonitoringMethod(newMethod));
    if (isLiveMonitoring) {
      try {
        await apiAutofocusControllerSetLiveMonitoringParameters(null, newMethod);
      } catch (error) {
        console.error("Error updating method:", error);
      }
    }
  };

  return (
    <Paper style={{ padding: "20px" }}>
      <Grid container spacing={2}>
        {/* Autofocus Scan Section */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            Autofocus Scan
          </Typography>
        </Grid>
        
        <Grid item xs={3}>
          <TextField
            label="Range Z"
            value={rangeZ}
            onChange={(e) => dispatch(autofocusSlice.setRangeZ(e.target.value))}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="Resolution Z"
            value={resolutionZ}
            onChange={(e) => dispatch(autofocusSlice.setResolutionZ(e.target.value))}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="Defocus Z"
            value={defocusZ}
            onChange={(e) => dispatch(autofocusSlice.setDefocusZ(e.target.value))}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <FormControl fullWidth>
            <InputLabel>Illumination Channel</InputLabel>
            <Select
              value={illuminationChannel || ''}
              onChange={(e) => dispatch(autofocusSlice.setIlluminationChannel(e.target.value))}
              label="Illumination Channel"
            >
              {availableIlluminations.map((illumination) => (
                <MenuItem key={illumination} value={illumination}>
                  {illumination}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Advanced Parameters Section */}
        <Grid item xs={12}>
          <Typography variant="subtitle2" gutterBottom style={{ marginTop: "10px" }}>
            Advanced Parameters
          </Typography>
        </Grid>

        <Grid item xs={3}>
          <TextField
            label="Settle Time (s)"
            type="number"
            value={tSettle}
            onChange={(e) => dispatch(autofocusSlice.setTSettle(parseFloat(e.target.value)))}
            inputProps={{ step: 0.01, min: 0, max: 10 }}
            fullWidth
          />
        </Grid>

        <Grid item xs={3}>
          <TextField
            label="Gaussian Blur Sigma"
            type="number"
            value={nGauss}
            onChange={(e) => dispatch(autofocusSlice.setNGauss(parseInt(e.target.value)))}
            inputProps={{ step: 1, min: 0, max: 20 }}
            fullWidth
          />
        </Grid>

        <Grid item xs={3}>
          <TextField
            label="Crop Size"
            type="number"
            value={nCropsize}
            onChange={(e) => dispatch(autofocusSlice.setNCropsize(parseInt(e.target.value)))}
            inputProps={{ step: 128, min: 256, max: 4096 }}
            fullWidth
          />
        </Grid>

        <Grid item xs={3}>
          <FormControl fullWidth>
            <InputLabel>Focus Algorithm</InputLabel>
            <Select
              value={focusAlgorithm}
              onChange={(e) => dispatch(autofocusSlice.setFocusAlgorithm(e.target.value))}
              label="Focus Algorithm"
            >
              <MenuItem value="LAPE">LAPE (Laplacian)</MenuItem>
              <MenuItem value="GLVA">GLVA (Variance)</MenuItem>
              <MenuItem value="JPEG">JPEG (Compression)</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={3}>
          <TextField
            label="Static Offset"
            type="number"
            value={staticOffset}
            onChange={(e) => dispatch(autofocusSlice.setStaticOffset(parseFloat(e.target.value)))}
            inputProps={{ step: 0.1, min: -100, max: 100 }}
            fullWidth
          />
        </Grid>

        <Grid item xs={3}>
          <FormControlLabel
            control={
              <Switch
                checked={isDebug}
                onChange={(e) => dispatch(autofocusSlice.setIsDebug(e.target.checked))}
                color="primary"
              />
            }
            label="Debug Mode"
          />
        </Grid>

        <Grid item xs={3}>
          <FormControlLabel
            control={
              <Switch
                checked={twoStage}
                onChange={(e) => dispatch(autofocusSlice.setTwoStage(e.target.checked))}
                color="primary"
              />
            }
            label="Two-Stage Focus"
          />
        </Grid>

        {/* Status Display */}
        {(currentZ !== null || positionError || backendState) && (
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
              {currentZ !== null && (
                <Typography variant="body2" color="textSecondary">
                  Current Z: {currentZ.toFixed(2)}
                </Typography>
              )}
              {backendState && backendState !== 'idle' && (
                <Typography variant="body2" color="primary">
                  State: {backendState}
                </Typography>
              )}
            </Box>
          </Grid>
        )}

        {/* Error Display */}
        {positionError && (
          <Grid item xs={12}>
            <Alert severity="error" onClose={() => setPositionError(null)}>
              {positionError}
            </Alert>
          </Grid>
        )}

        <Grid item xs={12}>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleStart}
            disabled={isRunning || isStarting}
            startIcon={isStarting ? <CircularProgress size={16} color="inherit" /> : null}
          >
            {isStarting ? "Starting..." : "Start Autofocus"}
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={handleStop}
            style={{ marginLeft: "10px" }}
            disabled={!isRunning || isStopping}
            startIcon={isStopping ? <CircularProgress size={16} color="inherit" /> : null}
          >
            {isStopping ? "Stopping..." : "Stop Autofocus"}
          </Button>
          {plotData && (
            <Button
              variant="contained"
              style={{ marginLeft: "10px" }}
              onClick={togglePlot}
            >
              {showPlot ? "Close Plot" : "Show Plot"}
            </Button>
          )}
          {isRunning && (
            <Typography 
              variant="body2" 
              color="primary" 
              style={{ marginLeft: "20px", display: 'inline' }}
            >
              Autofocus running...
            </Typography>
          )}
        </Grid>

        {showPlot && plotData && (
          <Grid item xs={12}>
            <Plot
              data={[
                {
                  x: plotData.x,
                  y: plotData.y,
                  type: "scatter",
                  mode: "lines+markers",
                  marker: { color: "red" },
                },
              ]}
              layout={{
                title: "Focus vs Contrast",
                xaxis: { title: "Focus Position" },
                yaxis: { title: "Contrast Value" },
              }}
              style={{ width: "100%", height: "400px" }}
            />
          </Grid>
        )}

        {/* Divider */}
        <Grid item xs={12}>
          <Divider style={{ margin: "20px 0" }} />
        </Grid>

        {/* Live Monitoring Section */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            Live Focus Monitoring
          </Typography>
        </Grid>

        <Grid item xs={4}>
          <TextField
            label="Update Period (s)"
            type="number"
            value={liveMonitoringPeriod}
            onChange={(e) => handlePeriodChange(parseFloat(e.target.value))}
            inputProps={{ step: 0.1, min: 0.1, max: 10 }}
            fullWidth
            disabled={isLiveMonitoring}
          />
        </Grid>

        <Grid item xs={4}>
          <FormControl fullWidth disabled={isLiveMonitoring}>
            <InputLabel>Focus Method</InputLabel>
            <Select
              value={liveMonitoringMethod}
              onChange={(e) => handleMethodChange(e.target.value)}
              label="Focus Method"
            >
              <MenuItem value="LAPE">LAPE (Laplacian)</MenuItem>
              <MenuItem value="GLVA">GLVA (Variance)</MenuItem>
              <MenuItem value="JPEG">JPEG (Compression)</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={4}>
          <TextField
            label="Crop Size"
            type="number"
            value={liveMonitoringCropsize}
            onChange={(e) => dispatch(autofocusSlice.setLiveMonitoringCropsize(parseInt(e.target.value)))}
            inputProps={{ step: 128, min: 256, max: 4096 }}
            fullWidth
            disabled={isLiveMonitoring}
          />
        </Grid>

        <Grid item xs={12}>
          <Box display="flex" alignItems="center" height="100%">
            <FormControlLabel
              control={
                <Switch
                  checked={isLiveMonitoring}
                  onChange={isLiveMonitoring ? handleStopLiveMonitoring : handleStartLiveMonitoring}
                  color="primary"
                />
              }
              label={isLiveMonitoring ? "Monitoring Active" : "Start Monitoring"}
            />
          </Box>
        </Grid>

        {/* Live Focus Value Display */}
        {isLiveMonitoring && (
          <Grid item xs={12}>
            <Paper 
              elevation={3} 
              style={{ 
                padding: "20px", 
                backgroundColor: "primary.main",
                textAlign: "center"
              }}
            >
              <Typography variant="h4" color="primary" gutterBottom>
                {liveFocusValue ? liveFocusValue.focus_value.toFixed(2) : "---"}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Current Focus Value ({liveMonitoringMethod})
              </Typography>
              {liveFocusValue && (
                <Typography variant="caption" color="textSecondary" display="block">
                  Last updated: {new Date(liveFocusValue.timestamp * 1000).toLocaleTimeString()}
                </Typography>
              )}
            </Paper>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
};

export default AutofocusController;
