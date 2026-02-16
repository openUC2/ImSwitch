// src/components/OffAxisHoloController.js
// Off-Axis Hologram Processing Controller Component
// Provides FFT visualization, sideband selection, phase unwrap, and digital refocus

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useTheme } from "@mui/material/styles";
import {
  Box,
  Button,
  Card,
  CardContent,
  Slider,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Divider,
  Chip,
  Stack,
  Paper,
  IconButton,
  Tooltip,
  Tab,
  Tabs,
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  CenterFocusStrong as CenterFocusStrongIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
} from "@mui/icons-material";

// Redux slice
import * as offAxisHoloSlice from "../state/slices/OffAxisHoloSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice";

// Components - use LiveViewControlWrapper for automatic format selection
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";

// API imports
import {
  apiOffAxisHoloControllerGetParams,
  apiOffAxisHoloControllerSetParams,
  apiOffAxisHoloControllerGetState,
  apiOffAxisHoloControllerStartProcessing,
  apiOffAxisHoloControllerStopProcessing,
  apiOffAxisHoloControllerPauseProcessing,
  apiOffAxisHoloControllerResumeProcessing,
  apiOffAxisHoloControllerSetDz,
  apiOffAxisHoloControllerSetRoi,
  apiOffAxisHoloControllerSetCcRoi,
  apiOffAxisHoloControllerSetApodization,
  apiOffAxisHoloControllerSetBinning,
  apiOffAxisHoloControllerSetPixelsize,
  apiOffAxisHoloControllerSetWavelength,
} from "../backendapi/apiOffAxisHoloController";

// Tab panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`stream-tabpanel-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

const OffAxisHoloController = () => {
  const dispatch = useDispatch();
  const theme = useTheme();

  // Redux state
  const offAxisState = useSelector(offAxisHoloSlice.getOffAxisHoloState);
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  // Local state for CC (sideband) ROI selection in FFT space
  const [ccSelection, setCcSelection] = useState({
    centerX: 100, // FFT pixel coords
    centerY: 100,
    sizeX: 200,
    sizeY: 200,
  });

  // Local state for sensor ROI
  const [roiSelection, setRoiSelection] = useState({
    centerX: 0,
    centerY: 0,
    size: 512,
  });

  // State for image dimensions
  const [imageSize, setImageSize] = useState({ width: 1920, height: 1080 });
  const [fftImageSize, setFftImageSize] = useState({ width: 512, height: 512 });

  // Tab state for processed stream selection
  const [activeStreamTab, setActiveStreamTab] = useState(0);

  // Refs for stream images
  const fftImageRef = useRef(null);
  const magImageRef = useRef(null);
  const phaseImageRef = useRef(null);

  // Compute scaling factors
  const totalScalingFactor = useMemo(() => {
    const streamSubsampling =
      liveStreamState.streamSettings?.jpeg?.subsampling?.factor ||
      liveStreamState.streamSettings?.jpeg?.subsampling_factor ||
      liveStreamState.streamSettings?.binary?.subsampling?.factor ||
      1;
    const binningFactor = offAxisState.binning || 1;
    return streamSubsampling * binningFactor;
  }, [liveStreamState.streamSettings, offAxisState.binning]);

  // ROI size in preview pixels
  const roiSizeInPreview = useMemo(() => {
    return roiSelection.size / totalScalingFactor;
  }, [roiSelection.size, totalScalingFactor]);

  // Build stream URLs
  const baseUrl = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api`;
  const fftStreamUrl = `${baseUrl}/OffAxisHoloController/mjpeg_stream_offaxisholo_fft?startStream=true&jpeg_quality=85`;
  const magStreamUrl = `${baseUrl}/OffAxisHoloController/mjpeg_stream_offaxisholo_mag?startStream=true&jpeg_quality=85`;
  const phaseStreamUrl = `${baseUrl}/OffAxisHoloController/mjpeg_stream_offaxisholo_phase?startStream=true&jpeg_quality=85`;

  // Load initial parameters and state on mount
  useEffect(() => {
    loadParameters();
    loadState();

    // Poll state periodically
    const stateInterval = setInterval(loadState, 5000);

    return () => {
      clearInterval(stateInterval);
    };
  }, []);

  // Load parameters from backend
  const loadParameters = useCallback(async () => {
    try {
      const params = await apiOffAxisHoloControllerGetParams();

      // Update Redux state
      dispatch(offAxisHoloSlice.setPixelsize(params.pixelsize || 3.45e-6));
      dispatch(offAxisHoloSlice.setWavelength(params.wavelength || 633e-9));
      dispatch(offAxisHoloSlice.setNa(params.na || 0.3));
      dispatch(offAxisHoloSlice.setDz(params.dz || 0.0));
      dispatch(offAxisHoloSlice.setBinning(params.binning || 1));
      dispatch(offAxisHoloSlice.setRoiCenter(params.roi_center || [0, 0]));
      dispatch(offAxisHoloSlice.setRoiSize(params.roi_size || 512));
      dispatch(offAxisHoloSlice.setCcCenter(params.cc_center || [100, 100]));
      dispatch(offAxisHoloSlice.setCcSize(params.cc_size || [50, 50]));
      dispatch(
        offAxisHoloSlice.setApodizationEnabled(params.apodization_enabled || false)
      );
      dispatch(
        offAxisHoloSlice.setApodizationType(params.apodization_type || "tukey")
      );
      dispatch(offAxisHoloSlice.setApodizationAlpha(params.apodization_alpha || 0.1));

      // Update local selections
      setRoiSelection({
        centerX: params.roi_center ? params.roi_center[0] : 0,
        centerY: params.roi_center ? params.roi_center[1] : 0,
        size: params.roi_size || 512,
      });
      setCcSelection({
        centerX: params.cc_center ? params.cc_center[0] : 100,
        centerY: params.cc_center ? params.cc_center[1] : 100,
        sizeX: params.cc_size ? params.cc_size[0] : 50,
        sizeY: params.cc_size ? params.cc_size[1] : 50,
      });
    } catch (error) {
      console.error("Failed to load off-axis holo parameters:", error);
    }
  }, [dispatch]);

  // Load processing state from backend
  const loadState = useCallback(async () => {
    try {
      const state = await apiOffAxisHoloControllerGetState();

      dispatch(offAxisHoloSlice.setIsProcessing(state.is_processing || false));
      dispatch(offAxisHoloSlice.setIsPaused(state.is_paused || false));
      dispatch(offAxisHoloSlice.setIsStreamingFft(state.is_streaming_fft || false));
      dispatch(
        offAxisHoloSlice.setIsStreamingMagnitude(state.is_streaming_magnitude || false)
      );
      dispatch(offAxisHoloSlice.setIsStreamingPhase(state.is_streaming_phase || false));
      dispatch(offAxisHoloSlice.setFftShape(state.fft_shape || [512, 512]));

      if (state.fft_shape) {
        setFftImageSize({ width: state.fft_shape[1], height: state.fft_shape[0] });
      }
    } catch (error) {
      console.error("Failed to load off-axis holo state:", error);
    }
  }, [dispatch]);

  // Start processing
  const handleStartProcessing = useCallback(async () => {
    try {
      await apiOffAxisHoloControllerStartProcessing();
      dispatch(offAxisHoloSlice.setIsProcessing(true));
      dispatch(offAxisHoloSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to start processing:", error);
    }
  }, [dispatch, loadState]);

  // Stop processing
  const handleStopProcessing = useCallback(async () => {
    try {
      await apiOffAxisHoloControllerStopProcessing();
      dispatch(offAxisHoloSlice.setIsProcessing(false));
      dispatch(offAxisHoloSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to stop processing:", error);
    }
  }, [dispatch, loadState]);

  // Pause processing
  const handlePauseProcessing = useCallback(async () => {
    try {
      await apiOffAxisHoloControllerPauseProcessing();
      dispatch(offAxisHoloSlice.setIsPaused(true));
      await loadState();
    } catch (error) {
      console.error("Failed to pause processing:", error);
    }
  }, [dispatch, loadState]);

  // Resume processing
  const handleResumeProcessing = useCallback(async () => {
    try {
      await apiOffAxisHoloControllerResumeProcessing();
      dispatch(offAxisHoloSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to resume processing:", error);
    }
  }, [dispatch, loadState]);

  // Update dz parameter
  const handleDzChange = useCallback(
    (event, value) => {
      dispatch(offAxisHoloSlice.setDz(value));
    },
    [dispatch]
  );

  const handleDzCommit = useCallback(async (event, value) => {
    try {
      await apiOffAxisHoloControllerSetDz(value);
    } catch (error) {
      console.error("Failed to update dz:", error);
    }
  }, []);

  // Apply CC ROI (sideband selection)
  const handleApplyCcRoi = useCallback(async () => {
    try {
      await apiOffAxisHoloControllerSetCcRoi({
        center_x: ccSelection.centerX,
        center_y: ccSelection.centerY,
        size_x: ccSelection.sizeX,
        size_y: ccSelection.sizeY,
      });
      dispatch(
        offAxisHoloSlice.setCcCenter([ccSelection.centerX, ccSelection.centerY])
      );
      dispatch(offAxisHoloSlice.setCcSize([ccSelection.sizeX, ccSelection.sizeY]));
    } catch (error) {
      console.error("Failed to apply CC ROI:", error);
    }
  }, [dispatch, ccSelection]);

  // Apply sensor ROI
  const handleApplyRoi = useCallback(async () => {
    try {
      const streamSubsampling =
        liveStreamState.streamSettings?.jpeg?.subsampling?.factor ||
        liveStreamState.streamSettings?.jpeg?.subsampling_factor ||
        liveStreamState.streamSettings?.binary?.subsampling?.factor ||
        1;

      const streamedWidth = imageSize.width;
      const streamedHeight = imageSize.height;

      const absoluteXInStream = streamedWidth / 2 + roiSelection.centerX;
      const absoluteYInStream = streamedHeight / 2 + roiSelection.centerY;

      const absoluteXFullSensor = Math.round(absoluteXInStream * streamSubsampling);
      const absoluteYFullSensor = Math.round(absoluteYInStream * streamSubsampling);
      const finalSize = Math.min(roiSelection.size, 2048);

      await apiOffAxisHoloControllerSetRoi({
        center_x: absoluteXFullSensor,
        center_y: absoluteYFullSensor,
        size: finalSize,
      });
      dispatch(
        offAxisHoloSlice.setRoiCenter([roiSelection.centerX, roiSelection.centerY])
      );
      dispatch(offAxisHoloSlice.setRoiSize(roiSelection.size));
    } catch (error) {
      console.error("Failed to apply ROI:", error);
    }
  }, [dispatch, roiSelection, liveStreamState.streamSettings, imageSize]);

  // Toggle apodization
  const handleToggleApodization = useCallback(
    async (event) => {
      const enabled = event.target.checked;
      try {
        await apiOffAxisHoloControllerSetApodization(
          enabled,
          offAxisState.apodizationType,
          offAxisState.apodizationAlpha
        );
        dispatch(offAxisHoloSlice.setApodizationEnabled(enabled));
      } catch (error) {
        console.error("Failed to toggle apodization:", error);
      }
    },
    [dispatch, offAxisState.apodizationType, offAxisState.apodizationAlpha]
  );

  // Change apodization type
  const handleApodizationTypeChange = useCallback(
    async (event) => {
      const type = event.target.value;
      try {
        await apiOffAxisHoloControllerSetApodization(
          offAxisState.apodizationEnabled,
          type,
          offAxisState.apodizationAlpha
        );
        dispatch(offAxisHoloSlice.setApodizationType(type));
      } catch (error) {
        console.error("Failed to change apodization type:", error);
      }
    },
    [dispatch, offAxisState.apodizationEnabled, offAxisState.apodizationAlpha]
  );

  // Change apodization alpha
  const handleApodizationAlphaCommit = useCallback(
    async (event, value) => {
      try {
        await apiOffAxisHoloControllerSetApodization(
          offAxisState.apodizationEnabled,
          offAxisState.apodizationType,
          value
        );
        dispatch(offAxisHoloSlice.setApodizationAlpha(value));
      } catch (error) {
        console.error("Failed to change apodization alpha:", error);
      }
    },
    [dispatch, offAxisState.apodizationEnabled, offAxisState.apodizationType]
  );

  // Handle image load from viewer
  const handleImageLoad = useCallback((width, height) => {
    setImageSize({ width, height });
  }, []);

  // Handle click on camera stream for ROI selection
  const handleLiveViewClick = useCallback(
    (pixelX, pixelY, imgWidth, imgHeight) => {
      const relativeX = pixelX - imgWidth / 2;
      const relativeY = pixelY - imgHeight / 2;

      setRoiSelection((prev) => ({
        ...prev,
        centerX: Math.round(relativeX),
        centerY: Math.round(relativeY),
      }));
    },
    []
  );

  // Handle click on FFT stream for sideband selection
  const handleFftClick = useCallback(
    (event) => {
      if (!fftImageRef.current) return;

      const rect = fftImageRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Scale to FFT image coordinates
      const scaleX = fftImageSize.width / rect.width;
      const scaleY = fftImageSize.height / rect.height;

      const fftX = Math.round(x * scaleX);
      const fftY = Math.round(y * scaleY);

      console.log("FFT Click:", { fftX, fftY, fftImageSize });

      setCcSelection((prev) => ({
        ...prev,
        centerX: fftX,
        centerY: fftY,
      }));

      // Auto-apply CC ROI
      (async () => {
        try {
          await apiOffAxisHoloControllerSetCcRoi({
            center_x: fftX,
            center_y: fftY,
            size_x: ccSelection.sizeX,
            size_y: ccSelection.sizeY,
          });
        } catch (error) {
          console.error("Failed to auto-apply CC ROI:", error);
        }
      })();
    },
    [fftImageSize, ccSelection.sizeX, ccSelection.sizeY]
  );

  // ROI overlay for camera stream
  const roiOverlay = useMemo(() => {
    if (!imageSize.width || !imageSize.height) return null;

    const centerXRel = (roiSelection.centerX + imageSize.width / 2) / imageSize.width;
    const centerYRel = (roiSelection.centerY + imageSize.height / 2) / imageSize.height;
    const sizeXRel = roiSizeInPreview / imageSize.width;
    const sizeYRel = roiSizeInPreview / imageSize.height;

    return (
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
      >
        <rect
          x={(centerXRel - sizeXRel / 2) * 100}
          y={(centerYRel - sizeYRel / 2) * 100}
          width={sizeXRel * 100}
          height={sizeYRel * 100}
          fill="none"
          stroke="red"
          strokeWidth="0.5"
          opacity="0.8"
        />
        <line
          x1={(centerXRel - 0.02) * 100}
          y1={centerYRel * 100}
          x2={(centerXRel + 0.02) * 100}
          y2={centerYRel * 100}
          stroke="red"
          strokeWidth="0.3"
        />
        <line
          x1={centerXRel * 100}
          y1={(centerYRel - 0.02) * 100}
          x2={centerXRel * 100}
          y2={(centerYRel + 0.02) * 100}
          stroke="red"
          strokeWidth="0.3"
        />
      </svg>
    );
  }, [imageSize, roiSelection, roiSizeInPreview]);

  // CC ROI overlay for FFT stream
  const ccOverlay = useMemo(() => {
    if (!fftImageSize.width || !fftImageSize.height) return null;

    const centerXRel = ccSelection.centerX / fftImageSize.width;
    const centerYRel = ccSelection.centerY / fftImageSize.height;
    const sizeXRel = ccSelection.sizeX / fftImageSize.width;
    const sizeYRel = ccSelection.sizeY / fftImageSize.height;

    return (
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
      >
        <rect
          x={(centerXRel - sizeXRel / 2) * 100}
          y={(centerYRel - sizeYRel / 2) * 100}
          width={sizeXRel * 100}
          height={sizeYRel * 100}
          fill="none"
          stroke="cyan"
          strokeWidth="0.5"
          opacity="0.9"
        />
        <circle
          cx={centerXRel * 100}
          cy={centerYRel * 100}
          r="1"
          fill="cyan"
          opacity="0.9"
        />
      </svg>
    );
  }, [fftImageSize, ccSelection]);

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: "100%" }}>
      <Typography variant="h5" gutterBottom>
        Off-Axis Hologram Processing
      </Typography>

      {/* Control Buttons */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
          <Button
            variant="contained"
            color="primary"
            startIcon={<PlayArrowIcon />}
            onClick={handleStartProcessing}
            disabled={offAxisState.isProcessing && !offAxisState.isPaused}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Start
          </Button>

          {offAxisState.isPaused ? (
            <Button
              variant="contained"
              color="success"
              startIcon={<PlayArrowIcon />}
              onClick={handleResumeProcessing}
              disabled={!offAxisState.isProcessing}
              sx={{ minWidth: { xs: "100%", sm: "auto" } }}
            >
              Resume
            </Button>
          ) : (
            <Button
              variant="contained"
              color="warning"
              startIcon={<PauseIcon />}
              onClick={handlePauseProcessing}
              disabled={!offAxisState.isProcessing || offAxisState.isPaused}
              sx={{ minWidth: { xs: "100%", sm: "auto" } }}
            >
              Pause
            </Button>
          )}

          <Button
            variant="contained"
            color="error"
            startIcon={<StopIcon />}
            onClick={handleStopProcessing}
            disabled={!offAxisState.isProcessing}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Stop
          </Button>

          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadParameters}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Refresh
          </Button>
        </Stack>

        {/* Status Chips */}
        <Stack direction="row" spacing={1} mt={2} flexWrap="wrap" useFlexGap>
          <Chip
            label={offAxisState.isProcessing ? "Processing" : "Stopped"}
            color={offAxisState.isProcessing ? "success" : "default"}
            size="small"
          />
          {offAxisState.isPaused && (
            <Chip label="Paused" color="warning" size="small" />
          )}
          <Chip
            label={`FFT: ${fftImageSize.width}×${fftImageSize.height}`}
            variant="outlined"
            size="small"
          />
        </Stack>
      </Paper>

      {/* Video Streams Grid */}
      <Grid container spacing={2} mb={2}>
        {/* Camera Stream */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Camera Stream (Click to set ROI)
              </Typography>
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  paddingTop: "75%",
                  backgroundColor: "#000",
                  borderRadius: 1,
                  overflow: "hidden",
                }}
              >
                <LiveViewControlWrapper
                  onClick={handleLiveViewClick}
                  onImageLoad={handleImageLoad}
                  overlayContent={roiOverlay}
                  enableStageMovement={false}
                />
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                ROI: {roiSelection.size}px | Preview: {Math.round(roiSizeInPreview)}px
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* FFT Stream */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                FFT Magnitude (Click to select sideband)
              </Typography>
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  paddingTop: "100%", // Square for FFT
                  backgroundColor: "#000",
                  borderRadius: 1,
                  overflow: "hidden",
                  cursor: "crosshair",
                }}
                onClick={handleFftClick}
              >
                <img
                  ref={fftImageRef}
                  src={fftStreamUrl}
                  alt="FFT Stream"
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                  }}
                  onLoad={(e) => {
                    if (e.target.naturalWidth && e.target.naturalHeight) {
                      setFftImageSize({
                        width: e.target.naturalWidth,
                        height: e.target.naturalHeight,
                      });
                    }
                  }}
                />
                {ccOverlay}
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                Sideband: ({ccSelection.centerX}, {ccSelection.centerY}) Size: {ccSelection.sizeX}×{ccSelection.sizeY}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Processed Streams Tabs */}
      <Paper elevation={2} sx={{ mb: 2 }}>
        <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
          <Tabs
            value={activeStreamTab}
            onChange={(e, v) => setActiveStreamTab(v)}
            variant="fullWidth"
          >
            <Tab label="Magnitude (Amplitude)" />
            <Tab label="Phase (Unwrapped)" />
          </Tabs>
        </Box>
        <TabPanel value={activeStreamTab} index={0}>
          <Box
            sx={{
              position: "relative",
              width: "100%",
              paddingTop: "75%",
              backgroundColor: "#000",
              borderRadius: 1,
              overflow: "hidden",
            }}
          >
            <img
              ref={magImageRef}
              src={magStreamUrl}
              alt="Magnitude Stream"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                objectFit: "contain",
              }}
            />
          </Box>
        </TabPanel>
        <TabPanel value={activeStreamTab} index={1}>
          <Box
            sx={{
              position: "relative",
              width: "100%",
              paddingTop: "75%",
              backgroundColor: "#000",
              borderRadius: 1,
              overflow: "hidden",
            }}
          >
            <img
              ref={phaseImageRef}
              src={phaseStreamUrl}
              alt="Phase Stream"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                objectFit: "contain",
              }}
            />
          </Box>
        </TabPanel>
      </Paper>

      {/* Digital Refocus Slider */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Digital Refocus (dz)
        </Typography>
        <Box sx={{ px: 2 }}>
          <Slider
            value={offAxisState.dz}
            onChange={handleDzChange}
            onChangeCommitted={handleDzCommit}
            min={-5000e-6}
            max={5000e-6}
            step={1e-6}
            valueLabelDisplay="on"
            valueLabelFormat={(value) => `${(value * 1e6).toFixed(1)} µm`}
            marks={[
              { value: -5000e-6, label: "-5000 µm" },
              { value: 0, label: "0" },
              { value: 5000e-6, label: "5000 µm" },
            ]}
          />
        </Box>
      </Paper>

      {/* Sideband ROI Controls */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Sideband (CC) Selection</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <TextField
                label="Center X"
                type="number"
                value={ccSelection.centerX}
                onChange={(e) =>
                  setCcSelection((prev) => ({
                    ...prev,
                    centerX: parseInt(e.target.value) || 0,
                  }))
                }
                size="small"
                fullWidth
              />
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField
                label="Center Y"
                type="number"
                value={ccSelection.centerY}
                onChange={(e) =>
                  setCcSelection((prev) => ({
                    ...prev,
                    centerY: parseInt(e.target.value) || 0,
                  }))
                }
                size="small"
                fullWidth
              />
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField
                label="Size X"
                type="number"
                value={ccSelection.sizeX}
                onChange={(e) =>
                  setCcSelection((prev) => ({
                    ...prev,
                    sizeX: parseInt(e.target.value) || 50,
                  }))
                }
                size="small"
                fullWidth
              />
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField
                label="Size Y"
                type="number"
                value={ccSelection.sizeY}
                onChange={(e) =>
                  setCcSelection((prev) => ({
                    ...prev,
                    sizeY: parseInt(e.target.value) || 50,
                  }))
                }
                size="small"
                fullWidth
              />
            </Grid>
            <Grid item xs={12}>
              <Button variant="contained" onClick={handleApplyCcRoi}>
                Apply Sideband ROI
              </Button>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Apodization Controls */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Apodization (Edge Damping)</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={offAxisState.apodizationEnabled}
                    onChange={handleToggleApodization}
                  />
                }
                label="Enable Apodization"
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Window Type</InputLabel>
                <Select
                  value={offAxisState.apodizationType}
                  onChange={handleApodizationTypeChange}
                  label="Window Type"
                  disabled={!offAxisState.apodizationEnabled}
                >
                  <MenuItem value="tukey">Tukey</MenuItem>
                  <MenuItem value="hann">Hann</MenuItem>
                  <MenuItem value="hamming">Hamming</MenuItem>
                  <MenuItem value="blackman">Blackman</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Typography variant="caption">
                Alpha (Tukey): {offAxisState.apodizationAlpha.toFixed(2)}
              </Typography>
              <Slider
                value={offAxisState.apodizationAlpha}
                onChange={(e, v) => dispatch(offAxisHoloSlice.setApodizationAlpha(v))}
                onChangeCommitted={handleApodizationAlphaCommit}
                min={0}
                max={1}
                step={0.01}
                disabled={
                  !offAxisState.apodizationEnabled ||
                  offAxisState.apodizationType !== "tukey"
                }
                size="small"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Advanced Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">
            <SettingsIcon sx={{ mr: 1, verticalAlign: "middle" }} />
            Advanced Settings
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={4}>
              <TextField
                label="Pixel Size (µm)"
                type="number"
                value={(offAxisState.pixelsize * 1e6).toFixed(2)}
                onChange={async (e) => {
                  const value = parseFloat(e.target.value) * 1e-6;
                  if (!isNaN(value) && value > 0) {
                    await apiOffAxisHoloControllerSetPixelsize(value);
                    dispatch(offAxisHoloSlice.setPixelsize(value));
                  }
                }}
                size="small"
                fullWidth
                inputProps={{ step: 0.01 }}
              />
            </Grid>
            <Grid item xs={6} sm={4}>
              <TextField
                label="Wavelength (nm)"
                type="number"
                value={(offAxisState.wavelength * 1e9).toFixed(0)}
                onChange={async (e) => {
                  const value = parseFloat(e.target.value) * 1e-9;
                  if (!isNaN(value) && value > 0) {
                    await apiOffAxisHoloControllerSetWavelength(value);
                    dispatch(offAxisHoloSlice.setWavelength(value));
                  }
                }}
                size="small"
                fullWidth
                inputProps={{ step: 1 }}
              />
            </Grid>
            <Grid item xs={6} sm={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Binning</InputLabel>
                <Select
                  value={offAxisState.binning}
                  onChange={async (e) => {
                    const value = e.target.value;
                    await apiOffAxisHoloControllerSetBinning(value);
                    dispatch(offAxisHoloSlice.setBinning(value));
                  }}
                  label="Binning"
                >
                  <MenuItem value={1}>1×1</MenuItem>
                  <MenuItem value={2}>2×2</MenuItem>
                  <MenuItem value={4}>4×4</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={4}>
              <TextField
                label="ROI Size (px)"
                type="number"
                value={roiSelection.size}
                onChange={(e) =>
                  setRoiSelection((prev) => ({
                    ...prev,
                    size: parseInt(e.target.value) || 512,
                  }))
                }
                size="small"
                fullWidth
                inputProps={{ min: 64, max: 2048, step: 64 }}
              />
            </Grid>
            <Grid item xs={6} sm={4}>
              <Button variant="outlined" onClick={handleApplyRoi} fullWidth>
                Apply ROI
              </Button>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default OffAxisHoloController;
