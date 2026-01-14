// src/components/HoloController.js
// Inline Hologram Processing Controller Component
// Provides live camera view, processed hologram stream, and parameter controls

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
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  CenterFocusStrong as CenterFocusStrongIcon,
} from "@mui/icons-material";

// Redux slice
import * as holoSlice from "../state/slices/HoloSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice";

// Components - use LiveViewControlWrapper for automatic format selection (JPEG/Binary/WebRTC)
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";

// API imports
import apiInLineHoloControllerGetParams from "../backendapi/apiInLineHoloControllerGetParams";
import apiInLineHoloControllerSetParams from "../backendapi/apiInLineHoloControllerSetParams";
import apiInLineHoloControllerGetState from "../backendapi/apiInLineHoloControllerGetState";
import apiInLineHoloControllerStartProcessing from "../backendapi/apiInLineHoloControllerStartProcessing";
import apiInLineHoloControllerStopProcessing from "../backendapi/apiInLineHoloControllerStopProcessing";
import apiInLineHoloControllerPauseProcessing from "../backendapi/apiInLineHoloControllerPauseProcessing";
import apiInLineHoloControllerResumeProcessing from "../backendapi/apiInLineHoloControllerResumeProcessing";
import apiInLineHoloControllerSetRoi from "../backendapi/apiInLineHoloControllerSetRoi";
import apiInLineHoloControllerSetDz from "../backendapi/apiInLineHoloControllerSetDz";
import apiInLineHoloControllerSetBinning from "../backendapi/apiInLineHoloControllerSetBinning";
import apiInLineHoloControllerSetPixelsize from "../backendapi/apiInLineHoloControllerSetPixelsize";
import apiInLineHoloControllerSetWavelength from "../backendapi/apiInLineHoloControllerSetWavelength";

const HoloController = () => {
  const dispatch = useDispatch();
  const theme = useTheme();

  // Redux state
  const holoState = useSelector(holoSlice.getHoloState);
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  // Local state for ROI selection
  // Note: size is stored in BACKEND pixels (after scaling), not preview pixels
  // This is the actual size that will be processed by the hologram controller
  const [roiSelection, setRoiSelection] = useState({
    isSelecting: false,
    centerX: 0,  // Relative to preview image center
    centerY: 0,  // Relative to preview image center  
    size: 256,   // Backend pixels (max 1024)
  });

  // State for image dimensions from the stream viewer
  const [imageSize, setImageSize] = useState({ width: 1920, height: 1080 });
  const [displayInfo, setDisplayInfo] = useState(null);

  // Compute the total scaling factor: subsampling (from stream) × binning (from holo processing)
  // This factor converts between preview pixels and backend/sensor pixels
  const totalScalingFactor = useMemo(() => {
    const streamSubsampling = liveStreamState.streamSettings?.jpeg?.subsampling?.factor || 
                              liveStreamState.streamSettings?.jpeg?.subsampling_factor ||
                              liveStreamState.streamSettings?.binary?.subsampling?.factor || 1;
    const binningFactor = holoState.binning || 1;
    return streamSubsampling * binningFactor;
  }, [liveStreamState.streamSettings, holoState.binning]);

  // ROI size in preview pixels (for overlay display)
  // roiSelection.size stores backend pixels, divide by scaling to get preview pixels
  const roiSizeInPreview = useMemo(() => {
    return roiSelection.size / totalScalingFactor;
  }, [roiSelection.size, totalScalingFactor]);

  // Ref for processed stream image
  const processedImageRef = useRef(null);

  // Build stream URLs - these need full URLs for video streaming
  const baseUrl = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api`;
  const processedStreamUrl = `${baseUrl}/InLineHoloController/mjpeg_stream_inlineholo?startStream=true&jpeg_quality=85`;

  // Load initial parameters and state on mount
  useEffect(() => {
    loadParameters();
    loadState();
    
    // Set stream URLs
    dispatch(holoSlice.setProcessedStreamUrl(processedStreamUrl));
    
    // Poll state periodically
    const stateInterval = setInterval(loadState, 5000);
    
    return () => {
      clearInterval(stateInterval);
    };
  }, []);

  // Load parameters from backend
  const loadParameters = useCallback(async () => {
    try {
      const params = await apiInLineHoloControllerGetParams();
      
      // Update Redux state with backend parameters
      dispatch(holoSlice.setPixelsize(params.pixelsize || 3.45e-6));
      dispatch(holoSlice.setWavelength(params.wavelength || 488e-9));
      dispatch(holoSlice.setNa(params.na || 0.3));
      dispatch(holoSlice.setDz(params.dz || 0.0));
      dispatch(holoSlice.setBinning(params.binning || 1));
      dispatch(holoSlice.setRoiCenter(params.roi_center || [0, 0]));
      dispatch(holoSlice.setRoiSize(params.roi_size || 256));
      dispatch(holoSlice.setColorChannel(params.color_channel || "green"));
      dispatch(holoSlice.setFlipX(params.flip_x || false));
      dispatch(holoSlice.setFlipY(params.flip_y || false));
      dispatch(holoSlice.setRotation(params.rotation || 0));
      dispatch(holoSlice.setUpdateFreq(params.update_freq || 10.0));
      
      // Update local ROI selection state
      setRoiSelection({
        ...roiSelection,
        centerX: params.roi_center ? params.roi_center[0] : 0,
        centerY: params.roi_center ? params.roi_center[1] : 0,
        size: params.roi_size || 256,
      });
    } catch (error) {
      console.error("Failed to load hologram parameters:", error);
    }
  }, [dispatch]);

  // Load processing state from backend
  const loadState = useCallback(async () => {
    try {
      const state = await apiInLineHoloControllerGetState();
      
      dispatch(holoSlice.setIsProcessing(state.is_processing || false));
      dispatch(holoSlice.setIsPaused(state.is_paused || false));
      dispatch(holoSlice.setIsStreaming(state.is_streaming || false));
      dispatch(holoSlice.setLastProcessTime(state.last_process_time || 0.0));
      dispatch(holoSlice.setFrameCount(state.frame_count || 0));
      dispatch(holoSlice.setProcessedCount(state.processed_count || 0));
    } catch (error) {
      console.error("Failed to load hologram state:", error);
    }
  }, [dispatch]);

  // Start processing
  const handleStartProcessing = useCallback(async () => {
    try {
      await apiInLineHoloControllerStartProcessing();
      dispatch(holoSlice.setIsProcessing(true));
      dispatch(holoSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to start processing:", error);
    }
  }, [dispatch, loadState]);

  // Stop processing
  const handleStopProcessing = useCallback(async () => {
    try {
      await apiInLineHoloControllerStopProcessing();
      dispatch(holoSlice.setIsProcessing(false));
      dispatch(holoSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to stop processing:", error);
    }
  }, [dispatch, loadState]);

  // Pause processing - switch to binning 1
  const handlePauseProcessing = useCallback(async () => {
    try {
      // Store current binning before pause
      dispatch(holoSlice.setPreviousBinning(holoState.binning));
      
      // Set binning to 1 for pause mode
      await apiInLineHoloControllerSetBinning(1);
      dispatch(holoSlice.setBinning(1));
      
      // Pause processing
      await apiInLineHoloControllerPauseProcessing();
      dispatch(holoSlice.setIsPaused(true));
      await loadState();
    } catch (error) {
      console.error("Failed to pause processing:", error);
    }
  }, [dispatch, holoState.binning, loadState]);

  // Resume processing - restore previous binning
  const handleResumeProcessing = useCallback(async () => {
    try {
      // Restore previous binning
      const previousBinning = holoState.previousBinning || 1;
      await apiInLineHoloControllerSetBinning(previousBinning);
      dispatch(holoSlice.setBinning(previousBinning));
      
      // Resume processing
      await apiInLineHoloControllerResumeProcessing();
      dispatch(holoSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to resume processing:", error);
    }
  }, [dispatch, holoState.previousBinning, loadState]);

  // Update dz parameter
  const handleDzChange = useCallback(
    async (event, value) => {
      dispatch(holoSlice.setDz(value));
    },
    [dispatch]
  );

  const handleDzCommit = useCallback(
    async (event, value) => {
      try {
        await apiInLineHoloControllerSetDz(value);
      } catch (error) {
        console.error("Failed to update dz:", error);
      }
    },
    []
  );

  // Update ROI parameters
  const handleRoiCenterXChange = useCallback((event, value) => {
    setRoiSelection((prev) => ({ ...prev, centerX: value }));
  }, []);

  const handleRoiCenterYChange = useCallback((event, value) => {
    setRoiSelection((prev) => ({ ...prev, centerY: value }));
  }, []);

  const handleRoiSizeChange = useCallback((event, value) => {
    setRoiSelection((prev) => ({ ...prev, size: value }));
  }, []);

  // Apply ROI changes to backend
  // ROI coordinates are relative to image center (0,0 = center)
  // Backend expects absolute pixel coordinates in the FULL sensor resolution
  const handleApplyRoi = useCallback(async () => {
    try {
      // Get stream subsampling factor (for coordinate scaling)
      const streamSubsampling = liveStreamState.streamSettings?.jpeg?.subsampling?.factor || 
                                liveStreamState.streamSettings?.jpeg?.subsampling_factor ||
                                liveStreamState.streamSettings?.binary?.subsampling?.factor || 1;
      
      // Use imageSize from viewer (this is the streamed/displayed image size)
      const streamedWidth = imageSize.width;
      const streamedHeight = imageSize.height;
      
      // roiSelection.centerX/Y are relative to the STREAMED image center (in preview pixels)
      // Convert to absolute pixel coordinates in the STREAMED image space
      const streamedCenterX = streamedWidth / 2;
      const streamedCenterY = streamedHeight / 2;
      
      const absoluteXInStream = streamedCenterX + roiSelection.centerX;
      const absoluteYInStream = streamedCenterY + roiSelection.centerY;
      
      // Scale up coordinates to full sensor space (using stream subsampling only, not binning)
      const absoluteXFullSensor = Math.round(absoluteXInStream * streamSubsampling);
      const absoluteYFullSensor = Math.round(absoluteYInStream * streamSubsampling);
      
      // roiSelection.size is already in backend pixels (includes binning consideration)
      // Just clamp to max 1024
      const finalSize = Math.min(roiSelection.size, 1024);
      
      console.log('ROI Parameters:', {
        relativeCenter: [roiSelection.centerX, roiSelection.centerY],
        streamedImageSize: [streamedWidth, streamedHeight],
        absoluteInStream: [absoluteXInStream, absoluteYInStream],
        streamSubsampling,
        binning: holoState.binning,
        totalScalingFactor,
        absoluteFullSensor: [absoluteXFullSensor, absoluteYFullSensor],
        roiSizeBackend: finalSize,
        roiSizePreview: roiSizeInPreview
      });
      
      // Backend API expects: center_x, center_y (absolute pixel coords in full sensor space)
      // and size in backend pixels
      const roiParams = {
        center_x: absoluteXFullSensor,
        center_y: absoluteYFullSensor,
        size: finalSize,
      };
      
      await apiInLineHoloControllerSetRoi(roiParams);
      dispatch(holoSlice.setRoiCenter([roiSelection.centerX, roiSelection.centerY]));
      dispatch(holoSlice.setRoiSize(roiSelection.size));
    } catch (error) {
      console.error("Failed to apply ROI:", error);
    }
  }, [dispatch, roiSelection, liveStreamState.streamSettings, imageSize, holoState.binning, totalScalingFactor, roiSizeInPreview]);

  // Reset ROI to center
  const handleResetRoi = useCallback(async () => {
    try {
      // Get stream subsampling factor for coordinate conversion
      const streamSubsampling = liveStreamState.streamSettings?.jpeg?.subsampling?.factor || 
                                liveStreamState.streamSettings?.jpeg?.subsampling_factor ||
                                liveStreamState.streamSettings?.binary?.subsampling?.factor || 1;
      
      // Use imageSize from viewer
      const streamedWidth = imageSize.width;
      const streamedHeight = imageSize.height;
      
      // Calculate full sensor center (using stream subsampling for coordinate conversion)
      const fullSensorCenterX = Math.round((streamedWidth / 2) * streamSubsampling);
      const fullSensorCenterY = Math.round((streamedHeight / 2) * streamSubsampling);
      
      // Reset to 256 backend pixels
      const defaultSize = 256;
      
      const roiParams = {
        center_x: fullSensorCenterX,
        center_y: fullSensorCenterY,
        size: defaultSize,
      };
      
      await apiInLineHoloControllerSetRoi(roiParams);
      dispatch(holoSlice.setRoiCenter([0, 0]));
      dispatch(holoSlice.setRoiSize(defaultSize));
      setRoiSelection({
        ...roiSelection,
        centerX: 0,
        centerY: 0,
        size: defaultSize,  // Backend pixels
      });
    } catch (error) {
      console.error("Failed to reset ROI:", error);
    }
  }, [dispatch, roiSelection, liveStreamState.streamSettings, imageSize]);

  // Update developer parameters
  const handleUpdateParameter = useCallback(
    async (paramName, value) => {
      try {
        // Call individual API endpoint based on parameter
        switch (paramName) {
          case "pixelsize":
            await apiInLineHoloControllerSetPixelsize(value);
            dispatch(holoSlice.setPixelsize(value));
            break;
          case "wavelength":
            await apiInLineHoloControllerSetWavelength(value);
            dispatch(holoSlice.setWavelength(value));
            break;
          case "binning":
            await apiInLineHoloControllerSetBinning(value);
            dispatch(holoSlice.setBinning(value));
            break;
          case "na":
          case "color_channel":
          case "flip_x":
          case "flip_y":
          case "rotation":
          case "update_freq":
            // These use the bulk set_parameters endpoint
            await apiInLineHoloControllerSetParams({ [paramName]: value });
            // Update Redux state
            switch (paramName) {
              case "na":
                dispatch(holoSlice.setNa(value));
                break;
              case "color_channel":
                dispatch(holoSlice.setColorChannel(value));
                break;
              case "flip_x":
                dispatch(holoSlice.setFlipX(value));
                break;
              case "flip_y":
                dispatch(holoSlice.setFlipY(value));
                break;
              case "rotation":
                dispatch(holoSlice.setRotation(value));
                break;
              case "update_freq":
                dispatch(holoSlice.setUpdateFreq(value));
                break;
              default:
                break;
            }
            break;
          default:
            console.warn(`Unknown parameter: ${paramName}`);
            break;
        }
      } catch (error) {
        console.error(`Failed to update ${paramName}:`, error);
      }
    },
    [dispatch]
  );

  // Handle image load from viewer - updates imageSize state
  const handleImageLoad = useCallback((width, height) => {
    console.log('HoloController: Image loaded with dimensions:', width, height);
    setImageSize({ width, height });
  }, []);

  // Handle click from LiveViewControlWrapper for ROI selection
  // Viewer provides pixel coordinates in the actual image space
  // We need to account for flipX, flipY, rotation transforms applied to the viewer
  const handleLiveViewClick = useCallback((pixelX, pixelY, imgWidth, imgHeight, displayInfoData) => {
    // pixelX, pixelY are in the STREAMED image coordinate space (0 to imgWidth, 0 to imgHeight)
    // The CSS transform (flipX, flipY, rotation) is applied to the viewer container,
    // but the click coordinates come from the viewer which doesn't know about the CSS transform.
    // We need to reverse the CSS transform to get the coordinates in the backend's coordinate space.
    
    let adjustedX = pixelX;
    let adjustedY = pixelY;
    
    // Apply inverse transforms in reverse order (CSS does: scaleX, scaleY, rotate)
    // So we need to: un-rotate, then un-flip
    
    // First, handle rotation (counter-rotate)
    const rotation = holoState.rotation || 0;
    if (rotation !== 0) {
      // Rotate around center
      const centerX = imgWidth / 2;
      const centerY = imgHeight / 2;
      const rad = -rotation * Math.PI / 180; // Negative for inverse
      const dx = adjustedX - centerX;
      const dy = adjustedY - centerY;
      adjustedX = centerX + dx * Math.cos(rad) - dy * Math.sin(rad);
      adjustedY = centerY + dx * Math.sin(rad) + dy * Math.cos(rad);
    }
    
    // Then handle flips
    if (holoState.flipX) {
      adjustedX = imgWidth - adjustedX;
    }
    if (holoState.flipY) {
      adjustedY = imgHeight - adjustedY;
    }
    
    // Convert to relative coordinates (center = 0,0)
    const imageCenterX = imgWidth / 2;
    const imageCenterY = imgHeight / 2;
    
    const relativeX = adjustedX - imageCenterX;
    const relativeY = adjustedY - imageCenterY;
    
    console.log('HoloController: LiveView click:', {
      rawPixelCoords: [pixelX, pixelY],
      adjustedPixelCoords: [adjustedX, adjustedY],
      transforms: { flipX: holoState.flipX, flipY: holoState.flipY, rotation: holoState.rotation },
      imageSize: [imgWidth, imgHeight],
      relativeCoords: [relativeX, relativeY]
    });
    
    // Store display info for overlay rendering
    if (displayInfoData) {
      setDisplayInfo(displayInfoData);
    }
    
    // Update ROI selection with relative coordinates and auto-apply
    const newCenterX = Math.round(relativeX);
    const newCenterY = Math.round(relativeY);
    
    setRoiSelection((prev) => ({
      ...prev,
      centerX: newCenterX,
      centerY: newCenterY,
    }));
    
    // Auto-apply ROI when clicking on canvas for convenience
    // We need to apply with the new values directly since state update is async
    (async () => {
      try {
        // Get stream subsampling for coordinate conversion only
        const streamSubsampling = liveStreamState.streamSettings?.jpeg?.subsampling?.factor || 
                                  liveStreamState.streamSettings?.jpeg?.subsampling_factor ||
                                  liveStreamState.streamSettings?.binary?.subsampling?.factor || 1;
        
        const streamedCenterX = imgWidth / 2;
        const streamedCenterY = imgHeight / 2;
        
        const absoluteXInStream = streamedCenterX + newCenterX;
        const absoluteYInStream = streamedCenterY + newCenterY;
        
        const absoluteXFullSensor = Math.round(absoluteXInStream * streamSubsampling);
        const absoluteYFullSensor = Math.round(absoluteYInStream * streamSubsampling);
        
        // roiSelection.size is already in backend pixels, just clamp to 1024
        const currentSize = roiSelection.size || 256;
        const finalSize = Math.min(currentSize, 1024);
        
        console.log('Auto-applying ROI on click:', {
          newCenter: [newCenterX, newCenterY],
          absoluteFullSensor: [absoluteXFullSensor, absoluteYFullSensor],
          streamSubsampling,
          binning: holoState.binning,
          roiSizeBackend: finalSize
        });
        
        await apiInLineHoloControllerSetRoi({
          center_x: absoluteXFullSensor,
          center_y: absoluteYFullSensor,
          size: finalSize,
        });
      } catch (error) {
        console.error('Failed to auto-apply ROI:', error);
      }
    })();
  }, [liveStreamState.streamSettings, roiSelection.size, holoState.flipX, holoState.flipY, holoState.rotation]);

  // Generate ROI overlay SVG for the stream viewer
  // This renders on top of the canvas to show the selected ROI
  // Uses roiSizeInPreview which is the size in preview pixels (backend size / totalScalingFactor)
  const roiOverlay = useMemo(() => {
    if (!imageSize.width || !imageSize.height) return null;
    
    // Calculate ROI position in relative coordinates (0-1 range from top-left)
    const centerXRel = (roiSelection.centerX + imageSize.width / 2) / imageSize.width;
    const centerYRel = (roiSelection.centerY + imageSize.height / 2) / imageSize.height;
    // Use preview size for overlay (roiSizeInPreview = backend size / scaling factor)
    const sizeXRel = roiSizeInPreview / imageSize.width;
    const sizeYRel = roiSizeInPreview / imageSize.height;
    
    return (
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
      >
        {/* ROI rectangle */}
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
        {/* Center crosshair */}
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

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: "100%" }}>
      <Typography variant="h5" gutterBottom>
        Inline Hologram Processing
      </Typography>

      {/* Control Buttons */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
          <Button
            variant="contained"
            color="primary"
            startIcon={<PlayArrowIcon />}
            onClick={handleStartProcessing}
            disabled={holoState.isProcessing && !holoState.isPaused}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Start
          </Button>
          
          {holoState.isPaused ? (
            <Button
              variant="contained"
              color="success"
              startIcon={<PlayArrowIcon />}
              onClick={handleResumeProcessing}
              disabled={!holoState.isProcessing}
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
              disabled={!holoState.isProcessing || holoState.isPaused}
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
            disabled={!holoState.isProcessing}
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
            label={holoState.isProcessing ? "Processing" : "Stopped"}
            color={holoState.isProcessing ? "success" : "default"}
            size="small"
          />
          {holoState.isPaused && (
            <Chip label="Paused" color="warning" size="small" />
          )}
          <Chip
            label={`Frames: ${holoState.frameCount}`}
            variant="outlined"
            size="small"
          />
          <Chip
            label={`Processed: ${holoState.processedCount}`}
            variant="outlined"
            size="small"
          />
        </Stack>
      </Paper>

      {/* Video Streams */}
      <Grid container spacing={2} mb={2}>
        {/* Raw Camera Stream */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Camera Stream
              </Typography>
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  paddingTop: "75%", // 4:3 aspect ratio for larger display
                  backgroundColor: "#000",
                  borderRadius: 1,
                  overflow: "hidden",
                }}
              >
                {/* Container for transform (flip/rotation) */}
                <Box
                  sx={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    transform: `scaleX(${holoState.flipX ? -1 : 1}) scaleY(${holoState.flipY ? -1 : 1}) rotate(${holoState.rotation || 0}deg)`,
                  }}
                >
                  {/* Use LiveViewControlWrapper for the camera stream with ROI overlay */}
                  {/* Auto-selects JPEG/Binary/WebRTC viewer based on stream format */}
                  <LiveViewControlWrapper
                    onClick={handleLiveViewClick}
                    onImageLoad={handleImageLoad}
                    overlayContent={roiOverlay}
                    enableStageMovement={false} // We handle our own click logic for ROI
                  />
                </Box>
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Click to set ROI center (auto-applies). Preview: {imageSize.width}×{imageSize.height}px | ROI: {roiSelection.size}px → {Math.round(roiSizeInPreview)}px preview
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Processed Hologram Stream */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processed Hologram
              </Typography>
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  paddingTop: "75%", // 4:3 aspect ratio - matches raw stream
                  backgroundColor: "#000",
                  borderRadius: 1,
                  overflow: "hidden",
                }}
              >
                <img
                  ref={processedImageRef}
                  src={processedStreamUrl}
                  alt="Processed Hologram Stream"
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
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Parameter Control - dz Slider */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Propagation Distance (dz)
        </Typography>
        <Box sx={{ px: 2 }}>
          <Slider
            value={holoState.dz}
            onChange={handleDzChange}
            onChangeCommitted={handleDzCommit}
            min={0}
            max={15000e-6}
            step={1e-6}
            valueLabelDisplay="on"
            valueLabelFormat={(value) => `${(value * 1e6).toFixed(1)} µm`}
            marks={[
              { value: 0, label: "0 µm" },
              { value: 15000e-6, label: "15000 µm" },
            ]}
            sx={{
              "& .MuiSlider-thumb": {
                width: 24,
                height: 24,
              },
            }}
          />
        </Box>
      </Paper>

      {/* ROI Controls */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6">ROI Selection</Typography>
          <Tooltip title="Reset ROI to center">
            <IconButton onClick={handleResetRoi} size="small">
              <CenterFocusStrongIcon />
            </IconButton>
          </Tooltip>
        </Box>

        <Grid container spacing={2}>
          {/* Center X - dynamic range based on image width */}
          <Grid item xs={12} sm={4}>
            <Typography gutterBottom>Center X (relative to center)</Typography>
            <Slider
              value={roiSelection.centerX}
              onChange={handleRoiCenterXChange}
              min={-Math.floor(imageSize.width / 2)}
              max={Math.floor(imageSize.width / 2)}
              step={1}
              valueLabelDisplay="auto"
              marks={[
                { value: -Math.floor(imageSize.width / 2), label: `${-Math.floor(imageSize.width / 2)}` },
                { value: 0, label: "0" },
                { value: Math.floor(imageSize.width / 2), label: `${Math.floor(imageSize.width / 2)}` },
              ]}
              sx={{
                "& .MuiSlider-thumb": {
                  width: 20,
                  height: 20,
                },
              }}
            />
          </Grid>

          {/* Center Y - dynamic range based on image height */}
          <Grid item xs={12} sm={4}>
            <Typography gutterBottom>Center Y (relative to center)</Typography>
            <Slider
              value={roiSelection.centerY}
              onChange={handleRoiCenterYChange}
              min={-Math.floor(imageSize.height / 2)}
              max={Math.floor(imageSize.height / 2)}
              step={1}
              valueLabelDisplay="auto"
              marks={[
                { value: -Math.floor(imageSize.height / 2), label: `${-Math.floor(imageSize.height / 2)}` },
                { value: 0, label: "0" },
                { value: Math.floor(imageSize.height / 2), label: `${Math.floor(imageSize.height / 2)}` },
              ]}
              sx={{
                "& .MuiSlider-thumb": {
                  width: 20,
                  height: 20,
                },
              }}
            />
          </Grid>

          {/* Size - ROI size in backend pixels (max 1024) */}
          {/* Preview shows scaled version: backend_size / (subsampling × binning) */}
          <Grid item xs={12} sm={4}>
            <Typography gutterBottom>
              ROI Size: {roiSelection.size}px (backend) / {Math.round(roiSizeInPreview)}px (preview)
            </Typography>
            <Slider
              value={roiSelection.size}
              onChange={handleRoiSizeChange}
              min={64}
              max={1024}
              step={64}
              valueLabelDisplay="auto"
              valueLabelFormat={(v) => `${v}px`}
              marks={[
                { value: 64, label: "64" },
                { value: 256, label: "256" },
                { value: 512, label: "512" },
                { value: 1024, label: "1024" },
              ]}
              sx={{
                "& .MuiSlider-thumb": {
                  width: 20,
                  height: 20,
                },
              }}
            />
            <Typography variant="caption" color="text.secondary">
              Scaling: {totalScalingFactor}× (subsampling: {liveStreamState.streamSettings?.jpeg?.subsampling?.factor || 1}, binning: {holoState.binning || 1})
            </Typography>
          </Grid>
        </Grid>

        <Button
          variant="contained"
          color="primary"
          onClick={handleApplyRoi}
          fullWidth
          sx={{ mt: 2 }}
        >
          Apply ROI
        </Button>
      </Paper>

      {/* Developer Options */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <SettingsIcon sx={{ mr: 1 }} />
          <Typography>Developer Options</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {/* Pixel Size */}
            <Grid item xs={12} sm={6}>
              <TextField
                label="Pixel Size (µm)"
                type="number"
                value={(holoState.pixelsize * 1e6).toFixed(3)}
                onChange={(e) =>
                  handleUpdateParameter("pixelsize", parseFloat(e.target.value) * 1e-6)
                }
                fullWidth
                inputProps={{ step: 0.001 }}
              />
            </Grid>

            {/* Wavelength */}
            <Grid item xs={12} sm={6}>
              <TextField
                label="Wavelength (nm)"
                type="number"
                value={(holoState.wavelength * 1e9).toFixed(1)}
                onChange={(e) =>
                  handleUpdateParameter("wavelength", parseFloat(e.target.value) * 1e-9)
                }
                fullWidth
                inputProps={{ step: 1 }}
              />
            </Grid>

            {/* Numerical Aperture */}
            <Grid item xs={12} sm={6}>
              <TextField
                label="Numerical Aperture (NA)"
                type="number"
                value={holoState.na}
                onChange={(e) =>
                  handleUpdateParameter("na", parseFloat(e.target.value))
                }
                fullWidth
                inputProps={{ step: 0.01, min: 0, max: 1 }}
              />
            </Grid>

            {/* Binning */}
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Binning</InputLabel>
                <Select
                  value={holoState.binning}
                  label="Binning"
                  onChange={(e) => handleUpdateParameter("binning", e.target.value)}
                >
                  <MenuItem value={1}>1x1</MenuItem>
                  <MenuItem value={2}>2x2</MenuItem>
                  <MenuItem value={4}>4x4</MenuItem>
                  <MenuItem value={8}>8x8</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            {/* Color Channel */}
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Color Channel</InputLabel>
                <Select
                  value={holoState.colorChannel}
                  label="Color Channel"
                  onChange={(e) => handleUpdateParameter("color_channel", e.target.value)}
                >
                  <MenuItem value="red">Red</MenuItem>
                  <MenuItem value="green">Green</MenuItem>
                  <MenuItem value="blue">Blue</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            {/* Update Frequency */}
            <Grid item xs={12} sm={6}>
              <TextField
                label="Update Frequency (Hz)"
                type="number"
                value={holoState.updateFreq}
                onChange={(e) =>
                  handleUpdateParameter("update_freq", parseFloat(e.target.value))
                }
                fullWidth
                inputProps={{ step: 1, min: 1, max: 60 }}
              />
            </Grid>

            {/* Flip X */}
            <Grid item xs={12} sm={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={holoState.flipX}
                    onChange={(e) => handleUpdateParameter("flip_x", e.target.checked)}
                  />
                }
                label="Flip X"
              />
            </Grid>

            {/* Flip Y */}
            <Grid item xs={12} sm={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={holoState.flipY}
                    onChange={(e) => handleUpdateParameter("flip_y", e.target.checked)}
                  />
                }
                label="Flip Y"
              />
            </Grid>

            {/* Rotation */}
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth>
                <InputLabel>Rotation</InputLabel>
                <Select
                  value={holoState.rotation}
                  label="Rotation"
                  onChange={(e) => handleUpdateParameter("rotation", e.target.value)}
                >
                  <MenuItem value={0}>0°</MenuItem>
                  <MenuItem value={90}>90°</MenuItem>
                  <MenuItem value={180}>180°</MenuItem>
                  <MenuItem value={270}>270°</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default HoloController;
