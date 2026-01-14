// src/components/DPCController.js
// DPC (Differential Phase Contrast) Controller Component
// Provides live camera view, processed DPC stream, and parameter controls

import React, { useState, useEffect, useCallback, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Paper,
  Typography,
  Slider,
  Button,
  Grid,
  TextField,
  Switch,
  FormControlLabel,
  Divider,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import SaveIcon from "@mui/icons-material/Save";

// Redux slice
import * as dpcSlice from "../state/slices/dpcSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice";

// API imports
import {
  getDpcParams,
  setDpcParams,
  getDpcState,
  startDpcProcessing,
  stopDpcProcessing,
  getDpcStreamUrl,
} from "../backendapi/dpcApi";

const DPCController = () => {
  const dispatch = useDispatch();
  const theme = useTheme();

  // Redux state
  const dpcState = useSelector(dpcSlice.getDpcState);
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  // Local state
  const [localParams, setLocalParams] = useState({});

  // Refs for canvas/image display
  const rawImageRef = useRef(null);
  const processedImageRef = useRef(null);

  // Build stream URLs - these need full URLs for video streaming
  const baseUrl = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api`;
  const rawStreamUrl = `${baseUrl}/RecordingController/video_feeder`;
  const processedStreamUrl = getDpcStreamUrl(baseUrl, 85);

  // Load initial parameters and state on mount
  useEffect(() => {
    loadParameters();
    loadState();
    
    // Poll state every 2 seconds when processing
    const interval = setInterval(() => {
      if (dpcState.is_processing) {
        loadState();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [dpcState.is_processing]);

  // Load parameters from backend
  const loadParameters = useCallback(async () => {
    try {
      const params = await getDpcParams(baseUrl);
      dispatch(dpcSlice.updateDpcParams(params));
      setLocalParams(params);
    } catch (error) {
      console.error("Failed to load DPC parameters:", error);
      dispatch(dpcSlice.setError("Failed to load parameters"));
    }
  }, [baseUrl, dispatch]);

  // Load processing state from backend
  const loadState = useCallback(async () => {
    try {
      const state = await getDpcState(baseUrl);
      dispatch(dpcSlice.updateDpcState(state));
    } catch (error) {
      console.error("Failed to load DPC state:", error);
    }
  }, [baseUrl, dispatch]);

  // Start processing
  const handleStartProcessing = useCallback(async () => {
    try {
      dispatch(dpcSlice.setLoading(true));
      const response = await startDpcProcessing(baseUrl);
      dispatch(dpcSlice.updateDpcState(response.state));
      loadState();
    } catch (error) {
      console.error("Failed to start DPC processing:", error);
      dispatch(dpcSlice.setError("Failed to start processing"));
    } finally {
      dispatch(dpcSlice.setLoading(false));
    }
  }, [baseUrl, dispatch, loadState]);

  // Stop processing
  const handleStopProcessing = useCallback(async () => {
    try {
      dispatch(dpcSlice.setLoading(true));
      const response = await stopDpcProcessing(baseUrl);
      dispatch(dpcSlice.updateDpcState(response.state));
      loadState();
    } catch (error) {
      console.error("Failed to stop DPC processing:", error);
      dispatch(dpcSlice.setError("Failed to stop processing"));
    } finally {
      dispatch(dpcSlice.setLoading(false));
    }
  }, [baseUrl, dispatch, loadState]);

  // Update parameter (local state)
  const handleParamChange = (paramName, value) => {
    setLocalParams((prev) => ({ ...prev, [paramName]: value }));
  };

  // Apply parameters to backend
  const handleApplyParams = useCallback(async () => {
    try {
      dispatch(dpcSlice.setLoading(true));
      const updatedParams = await setDpcParams(baseUrl, localParams);
      dispatch(dpcSlice.updateDpcParams(updatedParams));
      console.log("DPC parameters updated:", updatedParams);
    } catch (error) {
      console.error("Failed to update DPC parameters:", error);
      dispatch(dpcSlice.setError("Failed to update parameters"));
    } finally {
      dispatch(dpcSlice.setLoading(false));
    }
  }, [baseUrl, localParams, dispatch]);

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" gutterBottom>
        DPC Controller
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Differential Phase Contrast imaging with 4-pattern LED matrix illumination
      </Typography>

      {/* Status Card */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2">Processing Status</Typography>
              <Chip
                label={dpcState.is_processing ? "Running" : "Stopped"}
                color={dpcState.is_processing ? "success" : "default"}
                size="small"
              />
            </Grid>
            <Grid item xs={6} md={2}>
              <Typography variant="caption" color="text.secondary">
                Frames Captured
              </Typography>
              <Typography variant="h6">{dpcState.frame_count}</Typography>
            </Grid>
            <Grid item xs={6} md={2}>
              <Typography variant="caption" color="text.secondary">
                Processed
              </Typography>
              <Typography variant="h6">{dpcState.processed_count}</Typography>
            </Grid>
            <Grid item xs={6} md={2}>
              <Typography variant="caption" color="text.secondary">
                Processing FPS
              </Typography>
              <Typography variant="h6">
                {dpcState.processing_fps.toFixed(2)}
              </Typography>
            </Grid>
            <Grid item xs={6} md={2}>
              <Typography variant="caption" color="text.secondary">
                Last Process Time
              </Typography>
              <Typography variant="h6">
                {(dpcState.last_process_time * 1000).toFixed(0)} ms
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Control Buttons */}
      <Box sx={{ mb: 2, display: "flex", gap: 2 }}>
        <Button
          variant="contained"
          color={dpcState.is_processing ? "error" : "primary"}
          startIcon={dpcState.is_processing ? <StopIcon /> : <PlayArrowIcon />}
          onClick={
            dpcState.is_processing ? handleStopProcessing : handleStartProcessing
          }
          disabled={dpcState.isLoading}
        >
          {dpcState.is_processing ? "Stop Processing" : "Start Processing"}
        </Button>
        <Button
          variant="outlined"
          startIcon={<SaveIcon />}
          onClick={handleApplyParams}
          disabled={dpcState.isLoading || dpcState.is_processing}
        >
          Apply Parameters
        </Button>
      </Box>

      {/* Main Content Grid */}
      <Grid container spacing={2}>
        {/* Left Column - Video Streams */}
        <Grid item xs={12} md={8}>
          {/* Raw Camera Stream */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Raw Camera Stream
            </Typography>
            <Box
              sx={{
                width: "100%",
                aspectRatio: "4/3",
                bgcolor: "black",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
                overflow: "hidden",
              }}
            >
              <img
                ref={rawImageRef}
                src={rawStreamUrl}
                alt="Raw camera stream"
                style={{
                  maxWidth: "100%",
                  maxHeight: "100%",
                  objectFit: "contain",
                }}
                onError={(e) => {
                  console.error("Raw stream failed to load");
                  e.target.style.display = "none";
                }}
              />
            </Box>
          </Paper>

          {/* Processed DPC Stream */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Processed DPC Stream
            </Typography>
            <Box
              sx={{
                width: "100%",
                aspectRatio: "4/3",
                bgcolor: "black",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {dpcState.is_processing ? (
                <img
                  ref={processedImageRef}
                  src={processedStreamUrl}
                  alt="Processed DPC stream"
                  style={{
                    maxWidth: "100%",
                    maxHeight: "100%",
                    objectFit: "contain",
                  }}
                  onError={(e) => {
                    console.error("Processed stream failed to load");
                    e.target.style.display = "none";
                  }}
                />
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Start processing to view DPC reconstruction
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Right Column - Parameters */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              DPC Parameters
            </Typography>

            {/* Optical Parameters */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Optical Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <TextField
                    label="Pixel Size (μm)"
                    type="number"
                    value={localParams.pixelsize ?? dpcState.pixelsize}
                    onChange={(e) =>
                      handleParamChange("pixelsize", parseFloat(e.target.value))
                    }
                    inputProps={{ step: 0.01, min: 0.01, max: 50 }}
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="Wavelength (μm)"
                    type="number"
                    value={localParams.wavelength ?? dpcState.wavelength}
                    onChange={(e) =>
                      handleParamChange("wavelength", parseFloat(e.target.value))
                    }
                    inputProps={{ step: 0.01, min: 0.3, max: 1.0 }}
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="Numerical Aperture (NA)"
                    type="number"
                    value={localParams.na ?? dpcState.na}
                    onChange={(e) =>
                      handleParamChange("na", parseFloat(e.target.value))
                    }
                    inputProps={{ step: 0.01, min: 0.01, max: 1.6 }}
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="Illumination NA (NAi)"
                    type="number"
                    value={localParams.nai ?? dpcState.nai}
                    onChange={(e) =>
                      handleParamChange("nai", parseFloat(e.target.value))
                    }
                    inputProps={{ step: 0.01, min: 0.01, max: 1.6 }}
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="Refractive Index (n)"
                    type="number"
                    value={localParams.n ?? dpcState.n}
                    onChange={(e) =>
                      handleParamChange("n", parseFloat(e.target.value))
                    }
                    inputProps={{ step: 0.01, min: 1.0, max: 2.0 }}
                    size="small"
                    fullWidth
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Acquisition Parameters */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Acquisition Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <Box>
                    <Typography gutterBottom>
                      LED Intensity Red: {localParams.led_intensity_r ?? dpcState.led_intensity_r}
                    </Typography>
                    <Slider
                      value={localParams.led_intensity_r ?? dpcState.led_intensity_r}
                      onChange={(e, value) =>
                        handleParamChange("led_intensity_r", value)
                      }
                      min={0}
                      max={255}
                      step={1}
                      marks={[
                        { value: 0, label: "0" },
                        { value: 255, label: "255" },
                      ]}
                    />
                  </Box>
                  <Box>
                    <Typography gutterBottom>
                      LED Intensity Green: {localParams.led_intensity_g ?? dpcState.led_intensity_g}
                    </Typography>
                    <Slider
                      value={localParams.led_intensity_g ?? dpcState.led_intensity_g}
                      onChange={(e, value) =>
                        handleParamChange("led_intensity_g", value)
                      }
                      min={0}
                      max={255}
                      step={1}
                      marks={[
                        { value: 0, label: "0" },
                        { value: 255, label: "255" },
                      ]}
                    />
                  </Box>
                  <Box>
                    <Typography gutterBottom>
                      LED Intensity Blue: {localParams.led_intensity_b ?? dpcState.led_intensity_b}
                    </Typography>
                    <Slider
                      value={localParams.led_intensity_b ?? dpcState.led_intensity_b}
                      onChange={(e, value) =>
                        handleParamChange("led_intensity_b", value)
                      }
                      min={0}
                      max={255}
                      step={1}
                      marks={[
                        { value: 0, label: "0" },
                        { value: 255, label: "255" },
                      ]}
                    />
                  </Box>
                  <TextField
                    label="Wait Time (s)"
                    type="number"
                    value={localParams.wait_time ?? dpcState.wait_time}
                    onChange={(e) =>
                      handleParamChange("wait_time", parseFloat(e.target.value))
                    }
                    inputProps={{ step: 0.01, min: 0.01, max: 2.0 }}
                    size="small"
                    fullWidth
                    helperText="Time between LED pattern changes"
                  />
                  <TextField
                    label="Frame Sync"
                    type="number"
                    value={localParams.frame_sync ?? dpcState.frame_sync}
                    onChange={(e) =>
                      handleParamChange("frame_sync", parseInt(e.target.value))
                    }
                    inputProps={{ step: 1, min: 1, max: 10 }}
                    size="small"
                    fullWidth
                    helperText="Number of frames to wait for fresh frame"
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Reconstruction Parameters */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Reconstruction Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <TextField
                    label="Regularization (Absorption)"
                    type="number"
                    value={localParams.reg_u ?? dpcState.reg_u}
                    onChange={(e) =>
                      handleParamChange("reg_u", parseFloat(e.target.value))
                    }
                    inputProps={{ step: 0.001, min: 0.0001, max: 1.0 }}
                    size="small"
                    fullWidth
                    helperText="Tikhonov regularization for absorption"
                  />
                  <TextField
                    label="Regularization (Phase)"
                    type="number"
                    value={localParams.reg_p ?? dpcState.reg_p}
                    onChange={(e) =>
                      handleParamChange("reg_p", parseFloat(e.target.value))
                    }
                    inputProps={{ step: 0.001, min: 0.0001, max: 1.0 }}
                    size="small"
                    fullWidth
                    helperText="Tikhonov regularization for phase"
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Data Saving */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Data Saving</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localParams.save_images ?? dpcState.save_images}
                        onChange={(e) =>
                          handleParamChange("save_images", e.target.checked)
                        }
                      />
                    }
                    label="Save Images"
                  />
                  <TextField
                    label="Save Directory"
                    value={localParams.save_directory ?? dpcState.save_directory}
                    onChange={(e) =>
                      handleParamChange("save_directory", e.target.value)
                    }
                    size="small"
                    fullWidth
                    disabled={!(localParams.save_images ?? dpcState.save_images)}
                  />
                </Box>
              </AccordionDetails>
            </Accordion>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DPCController;
