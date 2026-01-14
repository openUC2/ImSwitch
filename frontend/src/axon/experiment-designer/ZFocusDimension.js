import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  TextField,
  FormControl,
  FormControlLabel,
  Radio,
  RadioGroup,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Select,
  MenuItem,
  InputLabel,
  Slider,
  Switch,
  Tooltip,
  IconButton,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import InfoIcon from "@mui/icons-material/Info";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PreviewIcon from "@mui/icons-material/Preview";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";
import { DIMENSIONS, Z_FOCUS_MODES } from "../../state/slices/ExperimentUISlice";
import apiFocusLockControllerGetCurrentFocusValue from "../../backendapi/apiFocusLockControllerGetCurrentFocusValue";

/**
 * ZFocusDimension - Z/Focus configuration interface
 *
 * Provides mutually exclusive mode selector:
 * - Single Z: No Z movement
 * - Autofocus: Software or hardware autofocus
 * - Z-Stack: Acquire multiple Z planes
 * - Z-Stack + Autofocus: Z-Stack with autofocus at each position
 *
 * Only parameters relevant to selected mode are visible
 */
const ZFocusDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const experimentUI = useSelector(experimentUISlice.getExperimentUIState);
  const parameterRange = useSelector(parameterRangeSlice.getParameterRangeState);

  const parameterValue = experimentState.parameterValue;
  const zFocusMode = experimentUI.dimensions[DIMENSIONS.Z_FOCUS]?.mode || Z_FOCUS_MODES.SINGLE_Z;

  // Calculate Z stack info
  const zStackRange = parameterValue.zStackMax - parameterValue.zStackMin;
  const zStackSteps = Math.max(1, Math.ceil(zStackRange / (parameterValue.zStackStepSize || 1)) + 1);

  // Update summary based on mode and settings
  useEffect(() => {
    let summary = "Single Z";
    
    switch (zFocusMode) {
      case Z_FOCUS_MODES.SINGLE_Z:
        summary = "Single Z";
        break;
      case Z_FOCUS_MODES.AUTOFOCUS:
        summary = parameterValue.autoFocusMode === "hardware" 
          ? "Autofocus (HW)" 
          : "Autofocus (SW)";
        break;
      case Z_FOCUS_MODES.Z_STACK:
        summary = `Z-Stack (${zStackSteps} planes)`;
        break;
      case Z_FOCUS_MODES.Z_STACK_AUTOFOCUS:
        summary = `Z-Stack + AF (${zStackSteps} planes)`;
        break;
    }

    dispatch(experimentUISlice.setDimensionSummary({
      dimension: DIMENSIONS.Z_FOCUS,
      summary,
    }));
  }, [zFocusMode, parameterValue, zStackSteps, dispatch]);

  // Sync mode with experiment state
  useEffect(() => {
    // Enable autofocus in experiment state when autofocus modes are selected
    const shouldEnableAF = zFocusMode === Z_FOCUS_MODES.AUTOFOCUS || 
                           zFocusMode === Z_FOCUS_MODES.Z_STACK_AUTOFOCUS;
    if (parameterValue.autoFocus !== shouldEnableAF) {
      dispatch(experimentSlice.setAutoFocus(shouldEnableAF));
    }

    // Enable Z-stack in experiment state when Z-stack modes are selected
    const shouldEnableZStack = zFocusMode === Z_FOCUS_MODES.Z_STACK || 
                               zFocusMode === Z_FOCUS_MODES.Z_STACK_AUTOFOCUS;
    if (parameterValue.zStack !== shouldEnableZStack) {
      dispatch(experimentSlice.setZStack(shouldEnableZStack));
    }
  }, [zFocusMode, dispatch]);

  // Poll current focus value for hardware autofocus
  const pollCurrentFocusValue = async () => {
    try {
      const result = await apiFocusLockControllerGetCurrentFocusValue();
      const focusValue = result.focus_value;
      dispatch(experimentSlice.setAutoFocusTargetSetpoint(focusValue));
      console.log(`Polled focus value: ${focusValue}`);
    } catch (error) {
      console.error("Failed to poll current focus value:", error);
      alert("Failed to get current focus value. Make sure FocusLock measurement is running.");
    }
  };

  // Handle mode change
  const handleModeChange = (event) => {
    const newMode = event.target.value;
    dispatch(experimentUISlice.setZFocusMode(newMode));
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      {/* Mode Selector */}
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
        Select Offset
      </Typography>
      
      <FormControl component="fieldset" sx={{ mb: 3 }}>
        <RadioGroup value={zFocusMode} onChange={handleModeChange}>
          <FormControlLabel
            value={Z_FOCUS_MODES.SINGLE_Z}
            control={<Radio size="small" />}
            label={
              <Box>
                <Typography variant="body2">Single Z</Typography>
                <Typography variant="caption" color="textSecondary">
                  Acquire at current Z position only
                </Typography>
              </Box>
            }
            sx={{ mb: 1 }}
          />
          
          <FormControlLabel
            value={Z_FOCUS_MODES.AUTOFOCUS}
            control={<Radio size="small" />}
            label={
              <Box>
                <Typography variant="body2">Autofocus</Typography>
                <Typography variant="caption" color="textSecondary">
                  Find optimal focus before each acquisition
                </Typography>
              </Box>
            }
            sx={{ mb: 1 }}
          />
          
          <FormControlLabel
            value={Z_FOCUS_MODES.Z_STACK}
            control={<Radio size="small" />}
            label={
              <Box>
                <Typography variant="body2">Z-Stack</Typography>
                <Typography variant="caption" color="textSecondary">
                  Acquire multiple Z planes
                </Typography>
              </Box>
            }
            sx={{ mb: 1 }}
          />
          
          <FormControlLabel
            value={Z_FOCUS_MODES.Z_STACK_AUTOFOCUS}
            control={<Radio size="small" />}
            label={
              <Box>
                <Typography variant="body2">Z-Stack + Autofocus</Typography>
                <Typography variant="caption" color="textSecondary">
                  Autofocus, then acquire Z-stack at each position
                </Typography>
              </Box>
            }
          />
        </RadioGroup>
      </FormControl>

      {/* Z-Stack Parameters (visible when Z-Stack selected) */}
      {(zFocusMode === Z_FOCUS_MODES.Z_STACK || zFocusMode === Z_FOCUS_MODES.Z_STACK_AUTOFOCUS) && (
        <Box
          sx={{
            p: 2,
            mb: 2,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 1,
            backgroundColor: alpha(theme.palette.background.default, 0.5),
          }}
        >
          <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
            Z-Stack Settings
          </Typography>

          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {/* Start position */}
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Start</InputLabel>
              <Select
                value={parameterValue.zStackMin}
                onChange={(e) => dispatch(experimentSlice.setZStackMin(Number(e.target.value)))}
                label="Start"
              >
                {[-100, -50, -20, 0, 20, 50, 80, 100].map((val) => (
                  <MenuItem key={val} value={val}>{val} μm</MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Stop position */}
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Stop</InputLabel>
              <Select
                value={parameterValue.zStackMax}
                onChange={(e) => dispatch(experimentSlice.setZStackMax(Number(e.target.value)))}
                label="Stop"
              >
                {[20, 40, 60, 80, 100, 150, 200].map((val) => (
                  <MenuItem key={val} value={val}>{val} μm</MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Step size */}
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Step Size</InputLabel>
              <Select
                value={parameterValue.zStackStepSize}
                onChange={(e) => dispatch(experimentSlice.setZStackStepSize(Number(e.target.value)))}
                label="Step Size"
              >
                {[0.5, 1, 2, 3, 5, 10, 20].map((val) => (
                  <MenuItem key={val} value={val}>{val} μm</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </Box>
      )}

      {/* Autofocus Parameters (visible when Autofocus selected) */}
      {(zFocusMode === Z_FOCUS_MODES.AUTOFOCUS || zFocusMode === Z_FOCUS_MODES.Z_STACK_AUTOFOCUS) && (
        <Box
          sx={{
            p: 2,
            mb: 2,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 1,
            backgroundColor: alpha(theme.palette.background.default, 0.5),
          }}
        >
          <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
            Autofocus Settings
          </Typography>

          {/* Autofocus Mode */}
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
              <Typography variant="caption" sx={{ fontWeight: 500 }}>
                Autofocus Mode
              </Typography>
              <Tooltip title="Software: Z-sweep autofocus (scans through Z positions). Hardware: One-shot autofocus using FocusLock controller.">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <FormControl size="small" fullWidth>
              <Select
                value={parameterValue.autoFocusMode || "software"}
                onChange={(e) => dispatch(experimentSlice.setAutoFocusMode(e.target.value))}
              >
                <MenuItem value="software">Software (Z-Sweep)</MenuItem>
                <MenuItem value="hardware">Hardware (FocusLock One-Shot)</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Hardware autofocus specific parameters */}
          {parameterValue.autoFocusMode === "hardware" && (
            <>
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: "block" }}>
                  Max Attempts
                </Typography>
                <TextField
                  type="number"
                  size="small"
                  value={parameterValue.autofocus_max_attempts || 3}
                  onChange={(e) => dispatch(experimentSlice.setAutoFocusMaxAttempts(Number(e.target.value)))}
                  inputProps={{ min: 1, max: 10 }}
                  sx={{ width: 100 }}
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                  <Typography variant="caption" sx={{ fontWeight: 500 }}>
                    Target Focus Setpoint
                  </Typography>
                  <Tooltip title="Enter the target focus value manually or poll the current value from FocusLock controller">
                    <IconButton size="small">
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Box sx={{ display: "flex", gap: 1 }}>
                  <TextField
                    type="number"
                    size="small"
                    value={parameterValue.autofocus_target_focus_setpoint || 0}
                    onChange={(e) => dispatch(experimentSlice.setAutoFocusTargetSetpoint(Number(e.target.value)))}
                    inputProps={{ step: 0.1 }}
                    sx={{ flex: 1 }}
                  />
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={pollCurrentFocusValue}
                  >
                    📊 Poll Current
                  </Button>
                </Box>
              </Box>
            </>
          )}

          {/* Illumination Channel for Autofocus */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: "block" }}>
              Illumination Channel
            </Typography>
            <FormControl size="small" fullWidth>
              <Select
                value={parameterValue.autoFocusIlluminationChannel || ""}
                onChange={(e) => dispatch(experimentSlice.setAutoFocusIlluminationChannel(e.target.value))}
              >
                <MenuItem value="">Auto (use active channel)</MenuItem>
                {parameterRange.illuSources?.map((source) => (
                  <MenuItem key={source} value={source}>
                    {source}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </Box>
      )}

      {/* Advanced Settings */}
      <Accordion
        disableGutters
        sx={{
          boxShadow: "none",
          border: `1px solid ${theme.palette.divider}`,
          "&:before": { display: "none" },
        }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="body2">Advanced Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {/* Software autofocus parameters */}
            {parameterValue.autoFocusMode !== "hardware" && (
              <>
                <TextField
                  label="Settle Time (s)"
                  type="number"
                  size="small"
                  value={parameterValue.autoFocusSettleTime || 0.1}
                  onChange={(e) => dispatch(experimentSlice.setAutoFocusSettleTime(Number(e.target.value)))}
                  inputProps={{ step: 0.01, min: 0, max: 10 }}
                />

                <TextField
                  label="Range (±μm)"
                  type="number"
                  size="small"
                  value={parameterValue.autoFocusRange || 100}
                  onChange={(e) => dispatch(experimentSlice.setAutoFocusRange(Number(e.target.value)))}
                  inputProps={{ step: 1, min: 1 }}
                />

                <TextField
                  label="Resolution (μm)"
                  type="number"
                  size="small"
                  value={parameterValue.autoFocusResolution || 10}
                  onChange={(e) => dispatch(experimentSlice.setAutoFocusResolution(Number(e.target.value)))}
                  inputProps={{ step: 0.1, min: 0.1 }}
                />

                <TextField
                  label="Crop Size (px)"
                  type="number"
                  size="small"
                  value={parameterValue.autoFocusCropsize || 2048}
                  onChange={(e) => dispatch(experimentSlice.setAutoFocusCropsize(Number(e.target.value)))}
                  inputProps={{ step: 128, min: 256, max: 4096 }}
                />

                <FormControl size="small">
                  <InputLabel>Focus Algorithm</InputLabel>
                  <Select
                    value={parameterValue.autoFocusAlgorithm || "LAPE"}
                    onChange={(e) => dispatch(experimentSlice.setAutoFocusAlgorithm(e.target.value))}
                    label="Focus Algorithm"
                  >
                    <MenuItem value="LAPE">LAPE (Laplacian)</MenuItem>
                    <MenuItem value="GLVA">GLVA (Variance)</MenuItem>
                    <MenuItem value="JPEG">JPEG (Compression)</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  label="Static Offset (μm)"
                  type="number"
                  size="small"
                  value={parameterValue.autoFocusStaticOffset || 0.0}
                  onChange={(e) => dispatch(experimentSlice.setAutoFocusStaticOffset(Number(e.target.value)))}
                  inputProps={{ step: 0.1, min: -100, max: 100 }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={parameterValue.autoFocusTwoStage || false}
                      onChange={(e) => dispatch(experimentSlice.setAutoFocusTwoStage(e.target.checked))}
                    />
                  }
                  label={<Typography variant="caption">Two-Stage Focus</Typography>}
                />
              </>
            )}
          </Box>
        </AccordionDetails>
      </Accordion>
      
      {/* Information about scan pattern */}
      <Box
        sx={{
          mt: 2,
          p: 1.5,
          borderRadius: 1,
          backgroundColor: alpha(theme.palette.info.main, 0.08),
        }}
      >
        <Typography variant="caption" color="textSecondary" sx={{ fontStyle: "italic" }}>
          💡 Note: Scan pattern (snake vs. raster) is configured in the Positions dimension.
        </Typography>
      </Box>
    </Box>
  );
};

export default ZFocusDimension;

