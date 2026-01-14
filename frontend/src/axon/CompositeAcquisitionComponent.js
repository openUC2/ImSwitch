// src/axon/CompositeAcquisitionComponent.js
// Multi-illumination composite acquisition UI component
// Allows configuration of illumination steps, RGB mapping, and preview of fused images

import React, { useState, useEffect, useCallback, useRef } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  Box,
  Paper,
  Typography,
  Button,
  IconButton,
  Slider,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormControlLabel,
  Switch,
  Chip,
  Tooltip,
  Grid,
  Divider,
  Collapse,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
} from "@mui/material";
import {
  PlayArrow,
  Stop,
  CameraAlt,
  Add,
  Delete,
  ArrowUpward,
  ArrowDownward,
  Settings,
  ExpandMore,
  ExpandLess,
  Visibility,
  VisibilityOff,
  Refresh,
  Lightbulb,
  WbIncandescent,
} from "@mui/icons-material";

// Import API functions
import apiCompositeControllerGetIlluminationSources from "../backendapi/apiCompositeControllerGetIlluminationSources";
import apiCompositeControllerGetParameters from "../backendapi/apiCompositeControllerGetParameters";
import apiCompositeControllerSetParameters from "../backendapi/apiCompositeControllerSetParameters";
import apiCompositeControllerGetState from "../backendapi/apiCompositeControllerGetState";
import apiCompositeControllerStart from "../backendapi/apiCompositeControllerStart";
import apiCompositeControllerStop from "../backendapi/apiCompositeControllerStop";
import apiCompositeControllerAddStep from "../backendapi/apiCompositeControllerAddStep";
import apiCompositeControllerRemoveStep from "../backendapi/apiCompositeControllerRemoveStep";
import apiCompositeControllerSetMapping from "../backendapi/apiCompositeControllerSetMapping";
import apiCompositeControllerCaptureSingle from "../backendapi/apiCompositeControllerCaptureSingle";

// Import Redux slice
import * as compositeSlice from "../state/slices/CompositeAcquisitionSlice";

// Color mapping for channel visualization
const CHANNEL_COLORS = {
  R: "#ff4444",
  G: "#44ff44",
  B: "#4444ff",
};

/**
 * IlluminationStepItem - Single illumination step configuration row
 */
const IlluminationStepItem = ({
  step,
  index,
  sources,
  onUpdate,
  onRemove,
  onMoveUp,
  onMoveDown,
  onToggleEnabled,
  isFirst,
  isLast,
  isRunning,
}) => {
  return (
    <ListItem
      sx={{
        bgcolor: step.enabled ? "background.paper" : "action.disabledBackground",
        borderRadius: 1,
        mb: 0.5,
        opacity: step.enabled ? 1 : 0.6,
      }}
    >
      <ListItemIcon>
        <IconButton
          size="small"
          onClick={() => onToggleEnabled(index)}
          disabled={isRunning}
        >
          {step.enabled ? <Visibility /> : <VisibilityOff />}
        </IconButton>
      </ListItemIcon>
      
      <ListItemText
        primary={
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <Select
                value={step.illumination || ""}
                onChange={(e) =>
                  onUpdate(index, { illumination: e.target.value })
                }
                disabled={isRunning}
                displayEmpty
              >
                <MenuItem value="">
                  <em>Select source</em>
                </MenuItem>
                {sources.map((source) => (
                  <MenuItem key={source} value={source}>
                    {source}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Box sx={{ width: 120 }}>
              <Typography variant="caption" color="text.secondary">
                Intensity: {(step.intensity * 100).toFixed(0)}%
              </Typography>
              <Slider
                size="small"
                value={step.intensity}
                min={0}
                max={1}
                step={0.01}
                onChange={(_, v) => onUpdate(index, { intensity: v })}
                disabled={isRunning}
              />
            </Box>
            
            <TextField
              size="small"
              label="Settle (ms)"
              type="number"
              value={step.settleMs}
              onChange={(e) =>
                onUpdate(index, { settleMs: parseFloat(e.target.value) || 0 })
              }
              disabled={isRunning}
              sx={{ width: 80 }}
              inputProps={{ min: 0, step: 5 }}
            />
          </Box>
        }
      />
      
      <ListItemSecondaryAction>
        <IconButton
          size="small"
          onClick={() => onMoveUp(index)}
          disabled={isFirst || isRunning}
        >
          <ArrowUpward fontSize="small" />
        </IconButton>
        <IconButton
          size="small"
          onClick={() => onMoveDown(index)}
          disabled={isLast || isRunning}
        >
          <ArrowDownward fontSize="small" />
        </IconButton>
        <IconButton
          size="small"
          onClick={() => onRemove(index)}
          disabled={isRunning}
          color="error"
        >
          <Delete fontSize="small" />
        </IconButton>
      </ListItemSecondaryAction>
    </ListItem>
  );
};

/**
 * ChannelMappingSelector - RGB channel mapping configuration
 */
const ChannelMappingSelector = ({ mapping, sources, onSetMapping, disabled }) => {
  return (
    <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", alignItems: "center" }}>
      {["R", "G", "B"].map((channel) => (
        <FormControl key={channel} size="small" sx={{ minWidth: 140 }}>
          <InputLabel
            sx={{ color: CHANNEL_COLORS[channel] }}
          >
            {channel === "R" ? "Red" : channel === "G" ? "Green" : "Blue"}
          </InputLabel>
          <Select
            value={mapping[channel] || ""}
            onChange={(e) =>
              onSetMapping({ [`${channel.toLowerCase()}_source`]: e.target.value })
            }
            disabled={disabled}
            label={channel === "R" ? "Red" : channel === "G" ? "Green" : "Blue"}
            sx={{
              "& .MuiOutlinedInput-notchedOutline": {
                borderColor: CHANNEL_COLORS[channel],
              },
            }}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            {sources.map((source) => (
              <MenuItem key={source} value={source}>
                {source}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      ))}
    </Box>
  );
};

/**
 * CompositeAcquisitionComponent - Main component for composite acquisition
 */
const CompositeAcquisitionComponent = ({ showPreview = true }) => {
  const dispatch = useDispatch();
  const compositeState = useSelector(compositeSlice.getCompositeState);
  
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const imageRef = useRef(null);
  const pollingRef = useRef(null);

  // Fetch initial data on mount
  useEffect(() => {
    fetchIlluminationSources();
    fetchParameters();
    fetchState();
    
    return () => {
      // Cleanup polling on unmount
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  // Start polling state when running
  useEffect(() => {
    if (compositeState.isRunning) {
      pollingRef.current = setInterval(fetchState, 500);
    } else if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [compositeState.isRunning]);

  const fetchIlluminationSources = async () => {
    try {
      const sources = await apiCompositeControllerGetIlluminationSources();
      dispatch(compositeSlice.setIlluminationSources(sources || []));
    } catch (error) {
      console.error("Failed to fetch illumination sources:", error);
    }
  };

  const fetchParameters = async () => {
    try {
      const params = await apiCompositeControllerGetParameters();
      dispatch(compositeSlice.updateParameters(params));
    } catch (error) {
      console.error("Failed to fetch parameters:", error);
    }
  };

  const fetchState = async () => {
    try {
      const state = await apiCompositeControllerGetState();
      dispatch(compositeSlice.updateState(state));
    } catch (error) {
      console.error("Failed to fetch state:", error);
    }
  };

  const handleStart = async () => {
    dispatch(compositeSlice.setIsLoading(true));
    try {
      // First sync parameters to backend
      await syncParametersToBackend();
      // Then start acquisition
      const result = await apiCompositeControllerStart();
      dispatch(compositeSlice.updateState(result));
    } catch (error) {
      console.error("Failed to start composite acquisition:", error);
      dispatch(compositeSlice.setErrorMessage(error.message));
    } finally {
      dispatch(compositeSlice.setIsLoading(false));
    }
  };

  const handleStop = async () => {
    dispatch(compositeSlice.setIsLoading(true));
    try {
      const result = await apiCompositeControllerStop();
      dispatch(compositeSlice.updateState(result));
    } catch (error) {
      console.error("Failed to stop composite acquisition:", error);
    } finally {
      dispatch(compositeSlice.setIsLoading(false));
    }
  };

  const handleCaptureSingle = async () => {
    dispatch(compositeSlice.setIsCapturing(true));
    try {
      // Sync parameters first
      await syncParametersToBackend();
      // Capture single composite
      const result = await apiCompositeControllerCaptureSingle();
      if (result.status === "success") {
        setPreviewImage(`data:image/jpeg;base64,${result.image_base64}`);
        dispatch(compositeSlice.setLastCompositeImage(result.image_base64));
        dispatch(compositeSlice.setLastCaptureMetadata(result.metadata));
      } else {
        dispatch(compositeSlice.setErrorMessage(result.message));
      }
    } catch (error) {
      console.error("Failed to capture single composite:", error);
      dispatch(compositeSlice.setErrorMessage(error.message));
    } finally {
      dispatch(compositeSlice.setIsCapturing(false));
    }
  };

  const syncParametersToBackend = async () => {
    // Convert frontend format to backend format
    const backendParams = {
      steps: compositeState.steps.map((s) => ({
        illumination: s.illumination,
        intensity: s.intensity,
        exposure_ms: s.exposureMs,
        settle_ms: s.settleMs,
        enabled: s.enabled,
      })),
      mapping: compositeState.mapping,
      fps_target: compositeState.fpsTarget,
      jpeg_quality: compositeState.jpegQuality,
      normalize_channels: compositeState.normalizeChannels,
      auto_exposure: compositeState.autoExposure,
    };
    
    await apiCompositeControllerSetParameters(backendParams);
  };

  const handleAddStep = () => {
    const newStep = {
      illumination: compositeState.illuminationSources[0] || "",
      intensity: 0.5,
      exposureMs: null,
      settleMs: 10.0,
      enabled: true,
    };
    dispatch(compositeSlice.addStep(newStep));
  };

  const handleUpdateStep = (index, updates) => {
    dispatch(compositeSlice.updateStep({ index, updates }));
  };

  const handleRemoveStep = (index) => {
    dispatch(compositeSlice.removeStep(index));
  };

  const handleToggleStepEnabled = (index) => {
    dispatch(compositeSlice.toggleStepEnabled(index));
  };

  const handleMoveStepUp = (index) => {
    dispatch(compositeSlice.moveStepUp(index));
  };

  const handleMoveStepDown = (index) => {
    dispatch(compositeSlice.moveStepDown(index));
  };

  const handleSetMapping = async (mappingUpdate) => {
    // Update local state
    dispatch(compositeSlice.setMapping({
      R: mappingUpdate.r_source !== undefined ? mappingUpdate.r_source : compositeState.mapping.R,
      G: mappingUpdate.g_source !== undefined ? mappingUpdate.g_source : compositeState.mapping.G,
      B: mappingUpdate.b_source !== undefined ? mappingUpdate.b_source : compositeState.mapping.B,
    }));
    
    // Sync to backend if running
    if (compositeState.isRunning) {
      try {
        await apiCompositeControllerSetMapping(mappingUpdate);
      } catch (error) {
        console.error("Failed to set mapping:", error);
      }
    }
  };

  const enabledStepsCount = compositeState.steps.filter((s) => s.enabled).length;

  return (
    <Paper sx={{ p: 2 }}>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <WbIncandescent color="primary" />
          <Typography variant="h6">Composite Acquisition</Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          {compositeState.isRunning && (
            <Chip
              label={`${compositeState.averageFps.toFixed(1)} FPS`}
              color="success"
              size="small"
            />
          )}
          <Tooltip title="Refresh from backend">
            <IconButton onClick={() => { fetchParameters(); fetchState(); }} size="small">
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Error message */}
      {compositeState.errorMessage && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => dispatch(compositeSlice.setErrorMessage(""))}>
          {compositeState.errorMessage}
        </Alert>
      )}

      {/* Control buttons */}
      <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
        {!compositeState.isRunning ? (
          <Button
            variant="contained"
            color="primary"
            startIcon={compositeState.isLoading ? <CircularProgress size={20} /> : <PlayArrow />}
            onClick={handleStart}
            disabled={compositeState.isLoading || enabledStepsCount === 0}
          >
            Start
          </Button>
        ) : (
          <Button
            variant="contained"
            color="error"
            startIcon={compositeState.isLoading ? <CircularProgress size={20} /> : <Stop />}
            onClick={handleStop}
            disabled={compositeState.isLoading}
          >
            Stop
          </Button>
        )}
        
        <Button
          variant="outlined"
          startIcon={compositeState.isCapturing ? <CircularProgress size={20} /> : <CameraAlt />}
          onClick={handleCaptureSingle}
          disabled={compositeState.isRunning || compositeState.isCapturing || enabledStepsCount === 0}
        >
          Capture Single
        </Button>
      </Box>

      {/* Illumination Steps */}
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
            Illumination Steps ({enabledStepsCount} active)
          </Typography>
          <Button
            size="small"
            startIcon={<Add />}
            onClick={handleAddStep}
            disabled={compositeState.isRunning}
          >
            Add Step
          </Button>
        </Box>
        
        {compositeState.steps.length === 0 ? (
          <Alert severity="info">
            No illumination steps configured. Click "Add Step" to begin.
          </Alert>
        ) : (
          <List dense sx={{ bgcolor: "background.default", borderRadius: 1, p: 1 }}>
            {compositeState.steps.map((step, index) => (
              <IlluminationStepItem
                key={index}
                step={step}
                index={index}
                sources={compositeState.illuminationSources}
                onUpdate={handleUpdateStep}
                onRemove={handleRemoveStep}
                onMoveUp={handleMoveStepUp}
                onMoveDown={handleMoveStepDown}
                onToggleEnabled={handleToggleStepEnabled}
                isFirst={index === 0}
                isLast={index === compositeState.steps.length - 1}
                isRunning={compositeState.isRunning}
              />
            ))}
          </List>
        )}
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* RGB Channel Mapping */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
          RGB Channel Mapping
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Map illumination sources to RGB output channels for the composite image.
        </Typography>
        <ChannelMappingSelector
          mapping={compositeState.mapping}
          sources={compositeState.steps
            .filter((s) => s.enabled && s.illumination)
            .map((s) => s.illumination)}
          onSetMapping={handleSetMapping}
          disabled={false}
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Advanced Settings */}
      <Box>
        <Button
          size="small"
          onClick={() => setShowAdvanced(!showAdvanced)}
          endIcon={showAdvanced ? <ExpandLess /> : <ExpandMore />}
          startIcon={<Settings />}
        >
          Advanced Settings
        </Button>
        
        <Collapse in={showAdvanced}>
          <Box sx={{ mt: 2, display: "flex", flexDirection: "column", gap: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  size="small"
                  label="Target FPS"
                  type="number"
                  value={compositeState.fpsTarget}
                  onChange={(e) =>
                    dispatch(compositeSlice.setFpsTarget(parseFloat(e.target.value) || 1))
                  }
                  disabled={compositeState.isRunning}
                  inputProps={{ min: 0.1, max: 30, step: 0.5 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  size="small"
                  label="JPEG Quality"
                  type="number"
                  value={compositeState.jpegQuality}
                  onChange={(e) =>
                    dispatch(compositeSlice.setJpegQuality(parseInt(e.target.value) || 85))
                  }
                  inputProps={{ min: 1, max: 100, step: 5 }}
                />
              </Grid>
            </Grid>
            
            <FormControlLabel
              control={
                <Switch
                  checked={compositeState.normalizeChannels}
                  onChange={(e) =>
                    dispatch(compositeSlice.setNormalizeChannels(e.target.checked))
                  }
                  disabled={compositeState.isRunning}
                />
              }
              label="Normalize channels (auto-contrast)"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={compositeState.autoExposure}
                  onChange={(e) =>
                    dispatch(compositeSlice.setAutoExposure(e.target.checked))
                  }
                  disabled={compositeState.isRunning}
                />
              }
              label="Per-step exposure override"
            />
          </Box>
        </Collapse>
      </Box>

      {/* Preview Image */}
      {showPreview && previewImage && (
        <Box sx={{ mt: 2 }}>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
            Last Captured Composite
          </Typography>
          <Box
            sx={{
              position: "relative",
              width: "100%",
              maxWidth: 400,
              border: "1px solid",
              borderColor: "divider",
              borderRadius: 1,
              overflow: "hidden",
            }}
          >
            <img
              ref={imageRef}
              src={previewImage}
              alt="Composite preview"
              style={{ width: "100%", height: "auto", display: "block" }}
            />
          </Box>
        </Box>
      )}

      {/* Status info */}
      {compositeState.isRunning && (
        <Box sx={{ mt: 2, p: 1, bgcolor: "background.default", borderRadius: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Cycle: {compositeState.cycleCount} | 
            Step: {compositeState.currentStep + 1}/{enabledStepsCount} | 
            Cycle time: {compositeState.lastCycleTimeMs.toFixed(1)} ms
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default CompositeAcquisitionComponent;
