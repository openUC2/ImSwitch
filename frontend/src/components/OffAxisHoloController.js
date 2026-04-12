// src/components/OffAxisHoloController.js
// Off-Axis Hologram Processing Controller Component
// Provides FFT visualization, sideband selection, phase unwrap, and digital refocus

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useTheme } from "@mui/material/styles";
import {
  Box,
  Button,
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
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
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
  apiOffAxisHoloControllerSetUnwrapPhase,
  apiOffAxisHoloControllerSetShowFftSpace,
} from "../backendapi/apiOffAxisHoloController";

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
      dispatch(offAxisHoloSlice.setPhaseUnwrapEnabled(params.unwrap_phase !== undefined ? params.unwrap_phase : true));
      dispatch(offAxisHoloSlice.setShowFftSpace(params.show_fft_space || false));

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

      // roiSelection.centerX/Y are in natural (subsampled) image coords (offset from center)
      // Convert to absolute sensor pixel coordinates
      const absoluteXFullSensor = Math.round((imageSize.width / 2 + roiSelection.centerX) * streamSubsampling);
      const absoluteYFullSensor = Math.round((imageSize.height / 2 + roiSelection.centerY) * streamSubsampling);
      // roiSelection.size is entered directly in sensor pixels
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
      // Update UI immediately for better responsiveness
      dispatch(offAxisHoloSlice.setApodizationEnabled(enabled));
      try {
        // Use timeout to prevent UI blocking
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error("Request timeout")), 5000)
        );
        await Promise.race([
          apiOffAxisHoloControllerSetApodization(
            enabled,
            offAxisState.apodizationType,
            offAxisState.apodizationAlpha
          ),
          timeoutPromise,
        ]);
      } catch (error) {
        console.error("Failed to toggle apodization:", error);
        // Revert on error
        dispatch(offAxisHoloSlice.setApodizationEnabled(!enabled));
      }
    },
    [dispatch, offAxisState.apodizationType, offAxisState.apodizationAlpha]
  );

  // Change apodization type
  const handleApodizationTypeChange = useCallback(
    async (event) => {
      const type = event.target.value;
      const previousType = offAxisState.apodizationType;
      // Update UI immediately
      dispatch(offAxisHoloSlice.setApodizationType(type));
      try {
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error("Request timeout")), 5000)
        );
        await Promise.race([
          apiOffAxisHoloControllerSetApodization(
            offAxisState.apodizationEnabled,
            type,
            offAxisState.apodizationAlpha
          ),
          timeoutPromise,
        ]);
      } catch (error) {
        console.error("Failed to change apodization type:", error);
        // Revert on error
        dispatch(offAxisHoloSlice.setApodizationType(previousType));
      }
    },
    [dispatch, offAxisState.apodizationEnabled, offAxisState.apodizationAlpha, offAxisState.apodizationType]
  );

  // Change apodization alpha
  const handleApodizationAlphaCommit = useCallback(
    async (event, value) => {
      try {
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error("Request timeout")), 5000)
        );
        await Promise.race([
          apiOffAxisHoloControllerSetApodization(
            offAxisState.apodizationEnabled,
            offAxisState.apodizationType,
            value
          ),
          timeoutPromise,
        ]);
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

  // Handle click on camera stream for ROI selection and immediate backend update
  const handleLiveViewClick = useCallback(
    (pixelX, pixelY, imgWidth, imgHeight) => {
      // pixelX/Y and imgWidth/imgHeight are in natural image coords (subsampled sensor pixels)
      const relativeX = Math.round(pixelX - imgWidth / 2);
      const relativeY = Math.round(pixelY - imgHeight / 2);

      setRoiSelection((prev) => {
        const streamSubsampling =
          liveStreamState.streamSettings?.jpeg?.subsampling?.factor ||
          liveStreamState.streamSettings?.jpeg?.subsampling_factor ||
          liveStreamState.streamSettings?.binary?.subsampling?.factor ||
          1;
        // Convert subsampled coords to real sensor pixel coords
        const sensorCenterX = Math.round((imgWidth / 2 + relativeX) * streamSubsampling);
        const sensorCenterY = Math.round((imgHeight / 2 + relativeY) * streamSubsampling);
        // Immediately send to backend (size stays as previously set, already in sensor pixels)
        apiOffAxisHoloControllerSetRoi({
          center_x: sensorCenterX,
          center_y: sensorCenterY,
          size: prev.size,
        }).catch((err) => console.error("Failed to auto-send ROI:", err));
        return { ...prev, centerX: relativeX, centerY: relativeY };
      });
    },
    [liveStreamState.streamSettings]
  );

  // Handle click on FFT stream for sideband selection
  // fftImageSize tracks the actual FFT dimensions (from backend state), so click coords
  // map directly to full-resolution FFT pixel coordinates without extra scaling.
  const handleFftClick = useCallback(
    (event) => {
      if (!fftImageRef.current) return;

      const rect = fftImageRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Map display click to actual FFT coordinates using the real FFT shape from state
      const actualFftW = offAxisState.fftShape ? offAxisState.fftShape[1] : fftImageSize.width;
      const actualFftH = offAxisState.fftShape ? offAxisState.fftShape[0] : fftImageSize.height;

      const fftX = Math.round(x / rect.width * actualFftW);
      const fftY = Math.round(y / rect.height * actualFftH);

      console.log("FFT Click:", { fftX, fftY, actualFftW, actualFftH });

      setCcSelection((prev) => {
        const updated = { ...prev, centerX: fftX, centerY: fftY };
        // Auto-apply CC ROI to backend
        apiOffAxisHoloControllerSetCcRoi({
          center_x: updated.centerX,
          center_y: updated.centerY,
          size_x: updated.sizeX,
          size_y: updated.sizeY,
        }).catch((error) => console.error("Failed to auto-apply CC ROI:", error));
        return updated;
      });
    },
    [fftImageSize, offAxisState.fftShape]
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
    <Box sx={{ p: 1, maxWidth: "100%" }}>

      {/* ── Compact header + controls ── */}
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, mr: 0.5, whiteSpace: "nowrap" }}>
          Off-Axis Holography
        </Typography>
        <Button size="small" variant="contained" color="primary" startIcon={<PlayArrowIcon />}
          onClick={handleStartProcessing} disabled={offAxisState.isProcessing && !offAxisState.isPaused}>
          Start
        </Button>
        {offAxisState.isPaused ? (
          <Button size="small" variant="contained" color="success" startIcon={<PlayArrowIcon />}
            onClick={handleResumeProcessing} disabled={!offAxisState.isProcessing}>
            Resume
          </Button>
        ) : (
          <Button size="small" variant="contained" color="warning" startIcon={<PauseIcon />}
            onClick={handlePauseProcessing} disabled={!offAxisState.isProcessing || offAxisState.isPaused}>
            Pause
          </Button>
        )}
        <Button size="small" variant="contained" color="error" startIcon={<StopIcon />}
          onClick={handleStopProcessing} disabled={!offAxisState.isProcessing}>
          Stop
        </Button>
        <Tooltip title="Reload parameters from backend">
          <IconButton size="small" onClick={loadParameters}><RefreshIcon fontSize="small" /></IconButton>
        </Tooltip>
        <Chip label={offAxisState.isProcessing ? "Processing" : "Stopped"}
          color={offAxisState.isProcessing ? "success" : "default"} size="small" />
        {offAxisState.isPaused && <Chip label="Paused" color="warning" size="small" />}
        <Chip label={`FFT: ${fftImageSize.width}×${fftImageSize.height}`} variant="outlined" size="small" />
      </Stack>

      {/* ── 3-panel stream row – equal height via flex stretch ── */}
      <Box sx={{ display: "flex", gap: 1, mb: 1, alignItems: "stretch" }}>

        {/* Camera stream (2 parts) */}
        <Box sx={{ flex: "2 1 0", minWidth: 0, display: "flex", flexDirection: "column", gap: 0.25 }}>
          <Typography variant="caption" color="text.secondary">
            Camera — click to center ROI &nbsp;
            <Box component="span" sx={{ opacity: 0.7 }}>ROI: {roiSelection.size}px · preview {Math.round(roiSizeInPreview)}px</Box>
          </Typography>
          <Box sx={{ flex: 1, position: "relative", bgcolor: "#000", borderRadius: 1, overflow: "hidden", minHeight: 180, lineHeight: 0 }}>
            <LiveViewControlWrapper
              onClick={handleLiveViewClick}
              onImageLoad={handleImageLoad}
              overlayContent={roiOverlay}
              enableStageMovement={false}
            />
          </Box>
        </Box>

        {/* FFT Magnitude (1 part) */}
        <Box sx={{ flex: "1 1 0", minWidth: 0, display: "flex", flexDirection: "column", gap: 0.25 }}>
          <Typography variant="caption" color="text.secondary">
            FFT — click sideband &nbsp;
            <Box component="span" sx={{ opacity: 0.7 }}>
              ({ccSelection.centerX}, {ccSelection.centerY}) {ccSelection.sizeX}×{ccSelection.sizeY}
            </Box>
          </Typography>
          <Box
            sx={{ flex: 1, position: "relative", bgcolor: "#000", borderRadius: 1, overflow: "hidden", cursor: "crosshair", minHeight: 180 }}
            onClick={handleFftClick}
          >
            <img
              ref={fftImageRef}
              src={fftStreamUrl}
              alt="FFT"
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated" }}
            />
            {ccOverlay}
          </Box>
        </Box>

        {/* Reconstructed magnitude / phase (1 part) */}
        <Box sx={{ flex: "1 1 0", minWidth: 0, display: "flex", flexDirection: "column", gap: 0.25 }}>
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Chip size="small" label="Magnitude" clickable
              onClick={() => setActiveStreamTab(0)}
              color={activeStreamTab === 0 ? "primary" : "default"}
              variant={activeStreamTab === 0 ? "filled" : "outlined"} />
            <Chip size="small" label="Phase" clickable
              onClick={() => setActiveStreamTab(1)}
              color={activeStreamTab === 1 ? "secondary" : "default"}
              variant={activeStreamTab === 1 ? "filled" : "outlined"} />
            <Box component="span" sx={{ typography: "caption", color: "text.secondary", opacity: 0.7, ml: 0.5 }}>
              {offAxisState.showFftSpace ? "FFT crop" : "reconstructed"} · dz={`${(offAxisState.dz * 1e6).toFixed(0)}`}µm
            </Box>
          </Stack>
          <Box sx={{ flex: 1, position: "relative", bgcolor: "#000", borderRadius: 1, overflow: "hidden", minHeight: 180 }}>
            {/* Both streams kept mounted to avoid reconnection on tab switch */}
            <img ref={magImageRef} src={magStreamUrl} alt="Magnitude"
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated",
                       display: activeStreamTab === 0 ? "block" : "none" }} />
            <img ref={phaseImageRef} src={phaseStreamUrl} alt="Phase"
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated",
                       display: activeStreamTab === 1 ? "block" : "none" }} />
          </Box>
        </Box>
      </Box>

      {/* ── Compact control strip: CC coords + dz + toggles ── */}
      <Paper elevation={1} sx={{ p: 1, mb: 1 }}>
        <Grid container spacing={1} alignItems="center">

          {/* Sideband CC coordinates (4 compact number fields) */}
          <Grid item xs={6} sm={3} md={3}>
            <Grid container spacing={0.75}>
              <Grid item xs={6}>
                <TextField label="CC X" type="number" value={ccSelection.centerX}
                  onChange={(e) => { const v = parseInt(e.target.value) || 0; setCcSelection(prev => { const u = { ...prev, centerX: v }; apiOffAxisHoloControllerSetCcRoi({ center_x: u.centerX, center_y: u.centerY, size_x: u.sizeX, size_y: u.sizeY }).catch(console.error); return u; }); }}
                  size="small" fullWidth inputProps={{ style: { padding: "4px 6px" } }} />
              </Grid>
              <Grid item xs={6}>
                <TextField label="CC Y" type="number" value={ccSelection.centerY}
                  onChange={(e) => { const v = parseInt(e.target.value) || 0; setCcSelection(prev => { const u = { ...prev, centerY: v }; apiOffAxisHoloControllerSetCcRoi({ center_x: u.centerX, center_y: u.centerY, size_x: u.sizeX, size_y: u.sizeY }).catch(console.error); return u; }); }}
                  size="small" fullWidth inputProps={{ style: { padding: "4px 6px" } }} />
              </Grid>
              <Grid item xs={6}>
                <TextField label="Size X" type="number" value={ccSelection.sizeX}
                  onChange={(e) => { const v = parseInt(e.target.value) || 50; setCcSelection(prev => { const u = { ...prev, sizeX: v }; apiOffAxisHoloControllerSetCcRoi({ center_x: u.centerX, center_y: u.centerY, size_x: u.sizeX, size_y: u.sizeY }).catch(console.error); return u; }); }}
                  size="small" fullWidth inputProps={{ style: { padding: "4px 6px" } }} />
              </Grid>
              <Grid item xs={6}>
                <TextField label="Size Y" type="number" value={ccSelection.sizeY}
                  onChange={(e) => { const v = parseInt(e.target.value) || 50; setCcSelection(prev => { const u = { ...prev, sizeY: v }; apiOffAxisHoloControllerSetCcRoi({ center_x: u.centerX, center_y: u.centerY, size_x: u.sizeX, size_y: u.sizeY }).catch(console.error); return u; }); }}
                  size="small" fullWidth inputProps={{ style: { padding: "4px 6px" } }} />
              </Grid>
            </Grid>
          </Grid>

          {/* Toggles */}
          <Grid item xs={6} sm={3} md={2}>
            <Stack spacing={0.25}>
              <FormControlLabel sx={{ m: 0 }} control={
                <Switch size="small" checked={offAxisState.phaseUnwrapEnabled}
                  onChange={async (e) => {
                    const en = e.target.checked;
                    dispatch(offAxisHoloSlice.setPhaseUnwrapEnabled(en));
                    try { await apiOffAxisHoloControllerSetUnwrapPhase(en); }
                    catch { dispatch(offAxisHoloSlice.setPhaseUnwrapEnabled(!en)); }
                  }} />
              } label={<Typography variant="caption">Unwrap Phase</Typography>} />
              <FormControlLabel sx={{ m: 0 }} control={
                <Switch size="small" checked={offAxisState.showFftSpace}
                  onChange={async (e) => {
                    const en = e.target.checked;
                    dispatch(offAxisHoloSlice.setShowFftSpace(en));
                    try { await apiOffAxisHoloControllerSetShowFftSpace(en); }
                    catch { dispatch(offAxisHoloSlice.setShowFftSpace(!en)); }
                  }} />
              } label={<Typography variant="caption">{offAxisState.showFftSpace ? "Show FFT Crop" : "Show Reconstructed"}</Typography>} />
            </Stack>
          </Grid>

          {/* dz refocus slider */}
          <Grid item xs={12} sm={6} md={7}>
            <Typography variant="caption" color="text.secondary">
              Digital Refocus dz: <strong>{(offAxisState.dz * 1e6).toFixed(0)} µm</strong>
            </Typography>
            <Slider
              value={offAxisState.dz}
              onChange={handleDzChange}
              onChangeCommitted={handleDzCommit}
              min={-15000e-6} max={15000e-6} step={1e-6}
              valueLabelDisplay="auto"
              valueLabelFormat={(v) => `${(v * 1e6).toFixed(0)} µm`}
              marks={[{ value: -15000e-6, label: "-15mm" }, { value: 0, label: "0" }, { value: 15000e-6, label: "+15mm" }]}
              size="small"
            />
          </Grid>
        </Grid>
      </Paper>

      {/* ── Combined settings accordion (ROI · Optics · Apodization) ── */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="body2">
            <SettingsIcon sx={{ fontSize: 15, mr: 0.5, verticalAlign: "middle" }} />
            Settings — ROI · Optics · Apodization
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ pt: 1 }}>
          <Grid container spacing={1} alignItems="center">

            {/* ROI */}
            <Grid item xs={6} sm={3} md={2}>
              <TextField label="ROI Size (px)" type="number" value={roiSelection.size}
                onChange={(e) => setRoiSelection((prev) => ({ ...prev, size: parseInt(e.target.value) || 512 }))}
                size="small" fullWidth inputProps={{ min: 64, max: 2048, step: 64 }} />
            </Grid>
            <Grid item xs={6} sm={2} md={1}>
              <Button variant="outlined" size="small" onClick={handleApplyRoi} fullWidth sx={{ height: "100%" }}>
                Apply
              </Button>
            </Grid>

            {/* Optics */}
            <Grid item xs={6} sm={3} md={2}>
              <TextField label="Pixel Size (µm)" type="number"
                value={(offAxisState.pixelsize * 1e6).toFixed(2)}
                onChange={async (e) => { const v = parseFloat(e.target.value) * 1e-6; if (!isNaN(v) && v > 0) { await apiOffAxisHoloControllerSetPixelsize(v); dispatch(offAxisHoloSlice.setPixelsize(v)); } }}
                size="small" fullWidth inputProps={{ step: 0.01 }} />
            </Grid>
            <Grid item xs={6} sm={3} md={2}>
              <TextField label="Wavelength (nm)" type="number"
                value={(offAxisState.wavelength * 1e9).toFixed(0)}
                onChange={async (e) => { const v = parseFloat(e.target.value) * 1e-9; if (!isNaN(v) && v > 0) { await apiOffAxisHoloControllerSetWavelength(v); dispatch(offAxisHoloSlice.setWavelength(v)); } }}
                size="small" fullWidth inputProps={{ step: 1 }} />
            </Grid>
            <Grid item xs={6} sm={3} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Binning</InputLabel>
                <Select value={offAxisState.binning}
                  onChange={async (e) => { const v = e.target.value; await apiOffAxisHoloControllerSetBinning(v); dispatch(offAxisHoloSlice.setBinning(v)); }}
                  label="Binning">
                  <MenuItem value={1}>1×1</MenuItem>
                  <MenuItem value={2}>2×2</MenuItem>
                  <MenuItem value={4}>4×4</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}><Divider /></Grid>

            {/* Apodization */}
            <Grid item xs={12} sm={4} md={3}>
              <FormControlLabel control={
                <Switch size="small" checked={offAxisState.apodizationEnabled} onChange={handleToggleApodization} />
              } label={<Typography variant="body2">Apodization (Edge Damping)</Typography>} />
            </Grid>
            <Grid item xs={12} sm={4} md={3}>
              <FormControl fullWidth size="small" disabled={!offAxisState.apodizationEnabled}>
                <InputLabel>Window</InputLabel>
                <Select value={offAxisState.apodizationType} onChange={handleApodizationTypeChange} label="Window">
                  <MenuItem value="tukey">Tukey</MenuItem>
                  <MenuItem value="hann">Hann</MenuItem>
                  <MenuItem value="hamming">Hamming</MenuItem>
                  <MenuItem value="blackman">Blackman</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={4} md={6}>
              <Typography variant="caption" color="text.secondary">
                Tukey α: {offAxisState.apodizationAlpha.toFixed(2)}
              </Typography>
              <Slider
                value={offAxisState.apodizationAlpha}
                onChange={(e, v) => dispatch(offAxisHoloSlice.setApodizationAlpha(v))}
                onChangeCommitted={handleApodizationAlphaCommit}
                min={0} max={1} step={0.01} size="small"
                disabled={!offAxisState.apodizationEnabled || offAxisState.apodizationType !== "tukey"}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default OffAxisHoloController;
