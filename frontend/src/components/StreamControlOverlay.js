import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Box,
  Paper,
  Typography,
  Chip,
  Slider,
  Button,
  Collapse,
  IconButton,
  Divider,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  useTheme,
} from "@mui/material";
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  AutoMode as AutoModeIcon,
  Settings as SettingsIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
} from "@mui/icons-material";
import { useSelector, useDispatch } from "react-redux";
import {
  setMinVal,
  setMaxVal,
  setGamma,
  getLiveStreamState,
  setStreamSettings,
  setImageFormat,
  setCropSize,
} from "../state/slices/LiveStreamSlice.js";
import apiLiveViewControllerSetStreamParameters from "../backendapi/apiLiveViewControllerSetStreamParameters";
import apiLiveViewControllerGetStreamParameters from "../backendapi/apiLiveViewControllerGetStreamParameters";
import apiLiveViewControllerStopLiveView from "../backendapi/apiLiveViewControllerStopLiveView";
import apiLiveViewControllerStartLiveView from "../backendapi/apiLiveViewControllerStartLiveView";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";

const StreamControlOverlay = ({
  stats,
  featureSupport,
  isWebGL,
  imageSize,
  viewTransform,
  forceExpanded = false,
}) => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const liveStreamState = useSelector(getLiveStreamState);
  const [isExpanded, setIsExpanded] = useState(forceExpanded);
  const [activeTab, setActiveTab] = useState(0); // 0 = Controls, 1 = Settings, 2 = Info

  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );

  // Destructure liveStreamState FIRST before using its values
  const {
    minVal = 0,
    maxVal = 255,
    gamma = 1.0,
    imageFormat = "binary", // This is the correct field name in Redux
    streamSettings = {},
  } = liveStreamState;

  // Draft mode for settings - initialize from Redux streamSettings
  const [draftSettings, setDraftSettings] = useState({
    ...streamSettings,
    crop_size: liveStreamState.cropSize || 0
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  // Determine if we're in JPEG mode
  const isJpeg = imageFormat === "jpeg";
  const maxRange = isJpeg ? 255 : 65535;
  const formatLabel = isJpeg ? "JPEG" : "Binary";
  const rangeLabel = `0–${maxRange}`;

  // Debug log on mount to verify persisted state
  useEffect(() => {
    console.log("[StreamControlOverlay] Mounted with persisted state:", {
      imageFormat,
      streamSettings,
      minVal,
      maxVal,
    });
  }, []);

  // Get current frame ID from stats
  const currentFrameId = liveStreamState?.stats?.currentFrameId;

  // Load settings from backend on mount - use a ref to track if already loaded
  const hasLoadedSettings = useRef(false);

  useEffect(() => {
    // Only load once on mount
    if (hasLoadedSettings.current) return;

    const loadBackendSettings = async () => {
      try {
        // Get stream parameters from new LiveViewController endpoint
        const response = await apiLiveViewControllerGetStreamParameters();
        console.log("Stream parameters from LiveViewController:", response);

        // Extract protocol data from new API response format
        const allParams = response.protocols || response;
        const currentProtocol =
          response.current_protocol || imageFormat || "jpeg"; // Use persisted format as fallback

        console.log("Current active protocol from backend:", currentProtocol);

        // Transform backend response to frontend format
        const loadedSettings = {
          binary: {
            enabled:
              allParams.binary?.is_active || currentProtocol === "binary",
            compression: {
              algorithm: allParams.binary?.compression_algorithm || "lz4",
              level: allParams.binary?.compression_level || 0,
            },
            subsampling: { factor: allParams.binary?.subsampling_factor || 4 },
            throttle_ms: allParams.binary?.throttle_ms || 100,
          },
          jpeg: {
            enabled: allParams.jpeg?.is_active || currentProtocol === "jpeg",
            quality: allParams.jpeg?.jpeg_quality || 85,
            subsampling: { factor: allParams.jpeg?.subsampling_factor || 1 },
            throttle_ms: allParams.jpeg?.throttle_ms || 100,
          },
          webrtc: {
            enabled:
              allParams.webrtc?.is_active || currentProtocol === "webrtc",
            max_width: allParams.webrtc?.max_width || 1280,
            throttle_ms: allParams.webrtc?.throttle_ms || 33,
            subsampling_factor: allParams.webrtc?.subsampling_factor || 1,
          },
        };

        // Add crop_size from backend response (shared across all protocols)
        loadedSettings.crop_size = allParams.binary?.crop_size || allParams.jpeg?.crop_size || allParams.webrtc?.crop_size || 0;

        setDraftSettings(loadedSettings);

        // ONLY update format if it differs from persisted state
        // This preserves user's last selected format across sessions
        if (currentProtocol !== imageFormat) {
          console.log(
            "Backend protocol differs from persisted format, updating to:",
            currentProtocol
          );
          dispatch(setImageFormat(currentProtocol));

          // Set appropriate min/max values based on format
          if (currentProtocol === "jpeg") {
            dispatch(setMinVal(0));
            dispatch(setMaxVal(255));
          } else {
            // Binary mode: auto-stretch to max range
            dispatch(setMinVal(0));
            dispatch(setMaxVal(65535));
          }
        } else {
          console.log("Using persisted format:", imageFormat);
        }

        dispatch(setStreamSettings(loadedSettings));
      } catch (error) {
        console.warn("Failed to load backend settings:", error);
        // Use Redux state as fallback or set defaults to JPEG
        const fallbackSettings = streamSettings || {
          binary: {
            enabled: false,
            compression: { algorithm: "lz4", level: 0 },
            subsampling: { factor: 4 },
            throttle_ms: 100,
          },
          jpeg: {
            enabled: true,
            quality: 85,
            subsampling: { factor: 1 },
            throttle_ms: 100,
          },
          webrtc: {
            enabled: false,
            max_width: 1280,
            throttle_ms: 33,
            subsampling_factor: 1,
          },
        };
        setDraftSettings(fallbackSettings);

        // Only set default format if not already persisted
        if (!imageFormat || imageFormat === "binary") {
          console.log("Setting default format to jpeg (fallback)");
          dispatch(setImageFormat("jpeg"));
          dispatch(setMinVal(0));
          dispatch(setMaxVal(255));
        } else {
          console.log("Using persisted format:", imageFormat);
        }
        dispatch(setStreamSettings(fallbackSettings));

        hasLoadedSettings.current = true;
      }
    };

    loadBackendSettings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only load on mount, imageFormat is captured from closure

  // Auto contrast function
  const handleAutoContrast = () => {
    if (isJpeg) {
      dispatch(setMinVal(0));
      dispatch(setMaxVal(255));
    } else {
      dispatch(setMinVal(0));
      dispatch(setMaxVal(32768));
    }
  };

  // Settings handlers
  const handleSubmitSettings = async () => {
    setIsSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(false);

    // Capture current Redux format BEFORE making any changes
    const previousFormat = liveStreamState.imageFormat || "binary";

    try {
      // Determine the format and prepare parameters
      const isJpegMode = draftSettings.jpeg?.enabled === true;
      const isWebRTCMode = draftSettings.webrtc?.enabled === true;
      const newFormat = isWebRTCMode
        ? "webrtc"
        : isJpegMode
        ? "jpeg"
        : "binary";

      console.log(
        "Submitting settings for format:",
        newFormat,
        "Draft settings:",
        draftSettings
      );

      // Submit to backend FIRST using new LiveViewController API
      if (isWebRTCMode) {
        // Set WebRTC stream parameters
        await apiLiveViewControllerSetStreamParameters("webrtc", {
          max_width: draftSettings.webrtc?.max_width || 1280,
          throttle_ms: draftSettings.webrtc?.throttle_ms || 33,
          subsampling_factor: draftSettings.webrtc?.subsampling_factor || 1,
          crop_size: draftSettings.crop_size || 0,
        });
      } else if (isJpegMode) {
        // Set JPEG stream parameters
        await apiLiveViewControllerSetStreamParameters("jpeg", {
          jpeg_quality: draftSettings.jpeg?.quality || 85,
          subsampling_factor: draftSettings.jpeg?.subsampling?.factor || 1,
          throttle_ms: draftSettings.jpeg?.throttle_ms || 100,
          crop_size: draftSettings.crop_size || 0,
        });
      } else {
        // Set binary stream parameters
        await apiLiveViewControllerSetStreamParameters("binary", {
          compression_algorithm:
            draftSettings.binary?.compression?.algorithm || "lz4",
          compression_level: draftSettings.binary?.compression?.level || 0,
          subsampling_factor: draftSettings.binary?.subsampling?.factor || 4,
          throttle_ms: draftSettings.binary?.throttle_ms || 100,
          crop_size: draftSettings.crop_size || 0,
        });
      }

      console.log(
        "Stream parameters updated successfully for protocol:",
        newFormat
      );

      // Check if format changed by comparing with previousFormat (captured at start of function)
      const formatChanged = previousFormat !== newFormat;

      // Update Redux FIRST before starting stream so frontend knows the correct format
      dispatch(setImageFormat(newFormat));
      dispatch(setStreamSettings(draftSettings));

      // Restart stream if format changed
      if (formatChanged) {
        console.log(
          `Format changed from ${previousFormat} to ${newFormat}, restarting stream...`
        );

        // Stop current stream
        await apiLiveViewControllerStopLiveView(null, false);

        // Wait a bit for cleanup
        await new Promise((resolve) => setTimeout(resolve, 100));

        // Start new stream with NEW protocol (not default binary!)
        await apiLiveViewControllerStartLiveView(null, newFormat);

        console.log(`Stream restarted with protocol: ${newFormat}`);
      }

      // Update min/max values based on format
      if (isJpegMode) {
        dispatch(setMinVal(0));
        dispatch(setMaxVal(255));
      } else if (isWebRTCMode) {
        // WebRTC mode: no intensity control needed (handled by camera)
        dispatch(setMinVal(0));
        dispatch(setMaxVal(255));
      } else {
        // Binary mode: keep current values or reset to max range
        if (liveStreamState.maxVal > 65535 || liveStreamState.maxVal === 255) {
          dispatch(setMinVal(0));
          dispatch(setMaxVal(65535));
        }
      }

      // Update backend capabilities based on mode
      dispatch({
        type: "liveStreamState/setBackendCapabilities",
        payload: {
          binaryStreaming: !isJpegMode && !isWebRTCMode,
          webglSupported: !isJpegMode && !isWebRTCMode,
        },
      });

      // Update Redux with new crop_size
      dispatch(setCropSize(draftSettings.crop_size || 0));

      setSubmitSuccess(true);
      setTimeout(() => setSubmitSuccess(false), 3000);
    } catch (error) {
      console.error("Error submitting stream settings:", error);
      setSubmitError(error.message || "Failed to submit settings");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResetSettings = () => {
    setDraftSettings({});
    setSubmitError(null);
    setSubmitSuccess(false);
  };

  return (
    <Paper
      elevation={forceExpanded ? 0 : 3}
      sx={{
        position: "relative",
        width: isExpanded || forceExpanded ? "100%" : 48,
        height: isExpanded || forceExpanded ? "auto" : 48,
        maxHeight: isExpanded || forceExpanded ? "none" : 48,
        backgroundColor: forceExpanded
          ? "transparent"
          : theme.palette.background.paper,
        color: theme.palette.text.primary,
        border: forceExpanded ? "none" : `1px solid ${theme.palette.divider}`,
        borderRadius: forceExpanded ? 0 : 2,
        zIndex: forceExpanded ? "auto" : 1000,
        transition: forceExpanded ? "none" : "all 0.3s ease-in-out",
        cursor: isExpanded || forceExpanded ? "default" : "pointer",
        overflow: forceExpanded ? "visible" : isExpanded ? "auto" : "hidden",
        display: "flex",
        flexDirection: "column",
      }}
      onClick={
        !isExpanded && !forceExpanded ? () => setIsExpanded(true) : undefined
      }
    >
      {/* Header */}
      <Box
        sx={{
          p: isExpanded || forceExpanded ? 2 : 0,
          pb: isExpanded || forceExpanded ? 1 : 0,
          flexShrink: 0,
          height: isExpanded || forceExpanded ? "auto" : 48,
        }}
      >
        {!isExpanded && !forceExpanded ? (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
            }}
          >
            <SettingsIcon sx={{ color: "primary.main" }} />
          </Box>
        ) : (
          <>
            {!forceExpanded && (
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  mb: 1,
                }}
              >
                <Typography
                  variant="h6"
                  sx={{ fontWeight: "bold", fontSize: "1rem" }}
                >
                  Stream Controls
                </Typography>
                <IconButton
                  size="small"
                  onClick={() => setIsExpanded(false)}
                  sx={{ ml: "auto" }}
                >
                  <ExpandLessIcon />
                </IconButton>
              </Box>
            )}

            <Box
              sx={{
                display: "flex",
                gap: 1,
                flexWrap: "wrap",
                mb: forceExpanded ? 2 : 0,
              }}
            >
              <Chip
                label={`Format: ${formatLabel}`}
                size="small"
                color={isJpeg ? "primary" : "secondary"}
              />
              <Chip
                label={`Range: ${rangeLabel}`}
                size="small"
                variant="outlined"
              />
              <Chip
                label={`Window: ${minVal}–${maxVal}`}
                size="small"
                color="info"
              />
              {currentFrameId !== null && currentFrameId !== undefined && (
                <Chip
                  label={`Frame: ${currentFrameId}`}
                  size="small"
                  color="success"
                />
              )}
            </Box>
          </>
        )}
      </Box>

      {/* Body */}
      <Collapse
        in={isExpanded || forceExpanded}
        sx={{ flex: 1, minHeight: 0, display: "flex" }}
      >
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            flex: 1,
            minHeight: 0,
          }}
        >
          <Divider sx={{ flexShrink: 0 }} />

          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
            variant="fullWidth"
            sx={{ borderBottom: 1, borderColor: "divider", flexShrink: 0 }}
          >
            <Tab label="Controls" />
            <Tab label="Settings" />
            <Tab label="Info" />
          </Tabs>

          {/* Scrollable content */}
          <Box
            sx={{
              flex: 1,
              minHeight: 0,
              overflowY: "auto",
              "&::-webkit-scrollbar": { width: 6 },
              "&::-webkit-scrollbar-track": { backgroundColor: "transparent" },
              "&::-webkit-scrollbar-thumb": {
                backgroundColor:
                  theme.palette.mode === "dark"
                    ? "rgba(255,255,255,0.3)"
                    : "rgba(0,0,0,0.3)",
                borderRadius: 3,
              },
              "&::-webkit-scrollbar-thumb:hover": {
                backgroundColor:
                  theme.palette.mode === "dark"
                    ? "rgba(255,255,255,0.5)"
                    : "rgba(0,0,0,0.5)",
              },
            }}
          >
            {/* Controls Tab */}
            {activeTab === 0 && (
              <Box sx={{ p: 2 }}>
                <Box sx={{ mb: 2 }}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 1,
                    }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: "medium" }}>
                      Window/Level
                    </Typography>
                    <Button
                      size="small"
                      variant="outlined"
                      startIcon={<AutoModeIcon />}
                      onClick={handleAutoContrast}
                      sx={{ ml: "auto", fontSize: "0.75rem", py: 0.25 }}
                    >
                      Auto
                    </Button>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" color="textSecondary">
                      Window: {minVal}
                    </Typography>
                    <Slider
                      value={minVal || 0}
                      onChange={(_, value) => dispatch(setMinVal(value))}
                      min={0}
                      max={maxRange}
                      step={1}
                      valueLabelDisplay="auto"
                      size="small"
                    />
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" color="textSecondary">
                      Level: {maxVal}
                    </Typography>
                    <Slider
                      value={maxVal || maxRange}
                      onChange={(_, value) => dispatch(setMaxVal(value))}
                      min={0}
                      max={maxRange}
                      step={1}
                      valueLabelDisplay="auto"
                      size="small"
                    />
                  </Box>

                  {!isJpeg && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="caption" color="textSecondary">
                        Gamma: {gamma?.toFixed(2) || 1.0}
                      </Typography>
                      <Slider
                        value={gamma || 1.0}
                        onChange={(_, value) => dispatch(setGamma(value))}
                        min={0.1}
                        max={3.0}
                        step={0.1}
                        valueLabelDisplay="auto"
                        size="small"
                      />
                    </Box>
                  )}
                </Box>
              </Box>
            )}

            {/* Settings Tab */}
            {activeTab === 1 && (
              <Box sx={{ p: 2, pb: 0 }}>
                <Typography
                  variant="body2"
                  sx={{ mb: 2, fontWeight: "medium" }}
                >
                  Stream Settings
                </Typography>

                {/* Crop Control - Shared across all formats */}
                <Box sx={{ mb: 3 }}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 1,
                    }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: "medium" }}>
                      Crop Region
                    </Typography>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => setDraftSettings(prev => ({ ...prev, crop_size: 0 }))}
                      disabled={draftSettings.crop_size === 0}
                      sx={{ ml: "auto", fontSize: "0.75rem", py: 0.25 }}
                    >
                      Reset FOV
                    </Button>
                  </Box>

                  <Box sx={{ mb: 1 }}>
                    <Typography variant="caption" color="textSecondary">
                      Crop Size: {draftSettings.crop_size === 0 ? "Full FOV" : `${draftSettings.crop_size}px`}
                    </Typography>
                    <Slider
                      value={draftSettings.crop_size || 0}
                      onChange={(_, value) => {
                        setDraftSettings(prev => ({ ...prev, crop_size: value }));
                      }}
                      min={0}
                      max={Math.max(imageSize?.width || 2048, imageSize?.height || 2048)}
                      step={16}
                      valueLabelDisplay="auto"
                      size="small"
                    />
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block", mt: 0.5, fontSize: "0.7rem" }}
                    >
                      Crops quadratic region around center (applied before subsampling)
                    </Typography>
                  </Box>
                </Box>

                {/* Stream Format Dropdown */}
                <Box sx={{ mb: 3 }}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Stream Format</InputLabel>
                    <Select
                      value={
                        draftSettings?.webrtc?.enabled
                          ? "webrtc"
                          : draftSettings?.jpeg?.enabled
                          ? "jpeg"
                          : "binary"
                      }
                      label="Stream Format"
                      onChange={(e) => {
                        const newFormat = e.target.value;
                        const isJpeg = newFormat === "jpeg";
                        const isWebRTC = newFormat === "webrtc";

                        // Update ONLY draft settings - do NOT update Redux yet
                        // Redux will be updated when user clicks Submit
                        setDraftSettings((prev) => ({
                          ...prev,
                          binary: {
                            ...prev?.binary,
                            enabled: !isJpeg && !isWebRTC,
                          },
                          jpeg: { ...prev?.jpeg, enabled: isJpeg },
                          webrtc: { ...prev?.webrtc, enabled: isWebRTC },
                        }));

                        console.log("Format changed to:", newFormat);
                      }}
                    >
                      <MenuItem value="binary">
                        Binary (16-bit) - High Quality
                      </MenuItem>
                      <MenuItem value="jpeg">JPEG (8-bit) - Legacy</MenuItem>
                      <MenuItem value="webrtc">WebRTC - Low Latency</MenuItem>
                    </Select>
                  </FormControl>
                  <Typography
                    variant="caption"
                    color="textSecondary"
                    sx={{ display: "block", mt: 1 }}
                  >
                    Current:{" "}
                    {draftSettings?.webrtc?.enabled
                      ? "WebRTC"
                      : draftSettings?.binary?.enabled
                      ? "Binary"
                      : "JPEG"}
                  </Typography>
                </Box>

                {/* Binary Settings */}
                {draftSettings?.binary?.enabled && (
                  <Box sx={{ mb: 3 }}>
                    <FormControl fullWidth sx={{ mb: 2 }} size="small">
                      <InputLabel>Compression</InputLabel>
                      <Select
                        value={
                          draftSettings.binary?.compression?.algorithm || "lz4"
                        }
                        label="Compression"
                        onChange={(e) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            binary: {
                              ...prev.binary,
                              compression: {
                                ...prev.binary?.compression,
                                algorithm: e.target.value,
                              },
                            },
                          }))
                        }
                      >
                        <MenuItem value="lz4">LZ4</MenuItem>
                        <MenuItem value="zstd">Zstandard</MenuItem>
                        <MenuItem value="none">None</MenuItem>
                      </Select>
                    </FormControl>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Compression Level:{" "}
                        {draftSettings.binary?.compression?.level || 0}
                      </Typography>
                      <Slider
                        value={draftSettings.binary?.compression?.level || 0}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            binary: {
                              ...prev.binary,
                              compression: {
                                ...prev.binary?.compression,
                                level: value,
                              },
                            },
                          }))
                        }
                        min={0}
                        max={9}
                        step={1}
                        marks
                        valueLabelDisplay="auto"
                        size="small"
                      />
                    </Box>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Subsampling:{" "}
                        {draftSettings.binary?.subsampling?.factor || 4}x
                      </Typography>
                      <Slider
                        value={draftSettings.binary?.subsampling?.factor || 4}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            binary: {
                              ...prev.binary,
                              subsampling: {
                                ...prev.binary?.subsampling,
                                factor: value,
                              },
                            },
                          }))
                        }
                        min={1}
                        max={8}
                        step={1}
                        marks
                        valueLabelDisplay="auto"
                        size="small"
                      />
                    </Box>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Throttle: {draftSettings.binary?.throttle_ms || 100}ms
                      </Typography>
                      <Slider
                        value={draftSettings.binary?.throttle_ms || 100}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            binary: { ...prev.binary, throttle_ms: value },
                          }))
                        }
                        min={16}
                        max={1000}
                        step={16}
                        valueLabelDisplay="auto"
                        size="small"
                      />
                    </Box>
                  </Box>
                )}

                {/* JPEG Settings */}
                {draftSettings?.jpeg?.enabled && (
                  <Box sx={{ mb: 3 }}>
                    <Typography
                      variant="body2"
                      sx={{ mb: 2, fontWeight: "medium" }}
                    >
                      JPEG Settings
                    </Typography>

                    <Alert severity="info" sx={{ mb: 2 }}>
                      JPEG mode provides 8-bit images. Quality setting affects
                      compression ratio.
                    </Alert>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Quality: {draftSettings.jpeg?.quality || 85}%
                      </Typography>
                      <Slider
                        value={draftSettings.jpeg?.quality || 85}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            jpeg: { ...prev.jpeg, quality: value },
                          }))
                        }
                        min={1}
                        max={100}
                        step={5}
                        marks={[
                          { value: 1, label: "Low" },
                          { value: 50, label: "Medium" },
                          { value: 100, label: "High" },
                        ]}
                        valueLabelDisplay="auto"
                        size="small"
                      />
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mt: 1 }}
                      >
                        Lower quality = smaller files, faster streaming
                      </Typography>
                    </Box>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Subsampling:{" "}
                        {draftSettings.jpeg?.subsampling?.factor || 1}x
                      </Typography>
                      <Slider
                        value={draftSettings.jpeg?.subsampling?.factor || 1}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            jpeg: {
                              ...prev.jpeg,
                              subsampling: {
                                ...prev.jpeg?.subsampling,
                                factor: value,
                              },
                            },
                          }))
                        }
                        min={1}
                        max={8}
                        step={1}
                        marks
                        valueLabelDisplay="auto"
                        size="small"
                      />
                    </Box>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Throttle: {draftSettings.jpeg?.throttle_ms || 100}ms
                      </Typography>
                      <Slider
                        value={draftSettings.jpeg?.throttle_ms || 100}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            jpeg: { ...prev.jpeg, throttle_ms: value },
                          }))
                        }
                        min={16}
                        max={1000}
                        step={16}
                        valueLabelDisplay="auto"
                        size="small"
                      />
                    </Box>
                  </Box>
                )}

                {/* WebRTC Settings */}
                {draftSettings?.webrtc?.enabled && (
                  <Box sx={{ mb: 3 }}>
                    <Typography
                      variant="body2"
                      sx={{ mb: 2, fontWeight: "medium" }}
                    >
                      WebRTC Settings
                    </Typography>

                    <Alert severity="info" sx={{ mb: 2 }}>
                      WebRTC mode provides real-time low-latency streaming with
                      adaptive bitrate.
                    </Alert>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Max Width: {draftSettings.webrtc?.max_width || 1280}px
                      </Typography>
                      <Slider
                        value={draftSettings.webrtc?.max_width || 1280}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            webrtc: { ...prev.webrtc, max_width: value },
                          }))
                        }
                        min={320}
                        max={1920}
                        step={160}
                        marks={[
                          { value: 320, label: "320p" },
                          { value: 640, label: "640p" },
                          { value: 1280, label: "1280p" },
                          { value: 1920, label: "1080p" },
                        ]}
                        valueLabelDisplay="auto"
                        size="small"
                      />
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mt: 1 }}
                      >
                        Higher resolution = better quality, more bandwidth
                      </Typography>
                    </Box>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Subsampling:{" "}
                        {draftSettings.webrtc?.subsampling_factor || 1}x
                      </Typography>
                      <Slider
                        value={draftSettings.webrtc?.subsampling_factor || 1}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            webrtc: {
                              ...prev.webrtc,
                              subsampling_factor: value,
                            },
                          }))
                        }
                        min={1}
                        max={8}
                        step={1}
                        marks
                        valueLabelDisplay="auto"
                        size="small"
                      />
                    </Box>

                    <Box sx={{ mb: 2 }}>
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mb: 1 }}
                      >
                        Throttle: {draftSettings.webrtc?.throttle_ms || 33}ms
                      </Typography>
                      <Slider
                        value={draftSettings.webrtc?.throttle_ms || 33}
                        onChange={(_, value) =>
                          setDraftSettings((prev) => ({
                            ...prev,
                            webrtc: { ...prev.webrtc, throttle_ms: value },
                          }))
                        }
                        min={16}
                        max={1000}
                        step={16}
                        valueLabelDisplay="auto"
                        size="small"
                      />
                      <Typography
                        variant="caption"
                        color="textSecondary"
                        sx={{ display: "block", mt: 1 }}
                      >
                        Frame interval: ~
                        {Math.round(
                          1000 / (draftSettings.webrtc?.throttle_ms || 33)
                        )}{" "}
                        FPS
                      </Typography>
                    </Box>
                  </Box>
                )}

                {/* Sticky actions */}
                <Box
                  sx={{
                    position: "sticky",
                    bottom: 0,
                    pt: 1,
                    pb: 1,
                    mt: 3,
                    backgroundColor: theme.palette.background.paper,
                    borderTop: `1px solid ${theme.palette.divider}`,
                    display: "flex",
                    gap: 1,
                    zIndex: 1,
                  }}
                >
                  <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={handleSubmitSettings}
                    disabled={isSubmitting}
                    size="small"
                    sx={{ flex: 1 }}
                  >
                    {isSubmitting ? <CircularProgress size={16} /> : "Submit"}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={handleResetSettings}
                    size="small"
                  >
                    Reset
                  </Button>
                </Box>

                {submitError && (
                  <Alert
                    severity="error"
                    sx={{ mt: 1 }}
                    onClose={() => setSubmitError(null)}
                  >
                    {submitError}
                  </Alert>
                )}

                {submitSuccess && (
                  <Alert
                    severity="success"
                    sx={{ mt: 1 }}
                    onClose={() => setSubmitSuccess(false)}
                  >
                    Settings submitted successfully!
                  </Alert>
                )}
              </Box>
            )}

            {/* Info Tab */}
            {activeTab === 2 && (
              <Box sx={{ p: 2 }}>
                <Box sx={{ mb: 2 }}>
                  <Typography
                    variant="body2"
                    sx={{ mb: 1, fontWeight: "medium" }}
                  >
                    Performance
                  </Typography>
                  <Box
                    sx={{
                      p: 1,
                      backgroundColor:
                        theme.palette.mode === "dark"
                          ? "rgba(255,255,255,0.05)"
                          : "rgba(0,0,0,0.05)",
                      borderRadius: 1,
                    }}
                  >
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block" }}
                    >
                      {isWebGL ? "WebGL2" : "Canvas2D"} |{" "}
                      {featureSupport?.lz4 ? "LZ4" : "No LZ4"}
                    </Typography>
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block" }}
                    >
                      FPS: {stats?.fps || 0} |{" "}
                      {((stats?.bps || 0) / 1000000).toFixed(1)} Mbps
                    </Typography>
                    {currentFrameId !== null &&
                      currentFrameId !== undefined && (
                        <Typography
                          variant="caption"
                          color="textSecondary"
                          sx={{ display: "block" }}
                        >
                          Frame ID: {currentFrameId}
                        </Typography>
                      )}
                  </Box>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography
                    variant="body2"
                    sx={{ mb: 1, fontWeight: "medium" }}
                  >
                    Image
                  </Typography>
                  <Box
                    sx={{
                      p: 1,
                      backgroundColor:
                        theme.palette.mode === "dark"
                          ? "rgba(255,255,255,0.05)"
                          : "rgba(0,0,0,0.05)",
                      borderRadius: 1,
                    }}
                  >
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block" }}
                    >
                      Resolution: {imageSize?.width || 0}x
                      {imageSize?.height || 0}
                    </Typography>
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block" }}
                    >
                      Zoom: {(viewTransform?.scale || 1).toFixed(2)}x
                    </Typography>
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block" }}
                    >
                      Pan: X={(viewTransform?.translateX || 0).toFixed(0)}, Y=
                      {(viewTransform?.translateY || 0).toFixed(0)}
                    </Typography>
                  </Box>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography
                    variant="body2"
                    sx={{ mb: 1, fontWeight: "medium" }}
                  >
                    Backend
                  </Typography>
                  <Box
                    sx={{
                      p: 1,
                      backgroundColor:
                        theme.palette.mode === "dark"
                          ? "rgba(255,255,255,0.05)"
                          : "rgba(0,0,0,0.05)",
                      borderRadius: 1,
                    }}
                  >
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block" }}
                    >
                      Host: {connectionSettingsState.ip}:
                      {connectionSettingsState.apiPort}/imswitch
                    </Typography>
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block" }}
                    >
                      Legacy: {liveStreamState.isLegacyBackend ? "Yes" : "No"}
                    </Typography>
                    <Typography
                      variant="caption"
                      color="textSecondary"
                      sx={{ display: "block" }}
                    >
                      Binary Support:{" "}
                      {liveStreamState.backendCapabilities?.binaryStreaming
                        ? "Yes"
                        : "No"}
                    </Typography>
                  </Box>
                </Box>

                {streamSettings && (
                  <Box
                    sx={{
                      mt: 2,
                      pt: 1,
                      borderTop: `1px solid ${theme.palette.divider}`,
                    }}
                  >
                    <Typography variant="caption" color="textSecondary">
                      Quality: {streamSettings.quality || "N/A"} | FPS:{" "}
                      {streamSettings.fps || "N/A"}
                    </Typography>
                  </Box>
                )}
              </Box>
            )}
          </Box>
        </Box>
      </Collapse>
    </Paper>
  );
};

export default StreamControlOverlay;
