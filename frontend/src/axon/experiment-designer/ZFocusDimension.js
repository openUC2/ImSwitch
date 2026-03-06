import React, { useEffect, useState } from "react";
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
  Switch,
  Tooltip,
  IconButton,
  Alert,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import InfoIcon from "@mui/icons-material/Info";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PreviewIcon from "@mui/icons-material/Preview";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";
import * as focusMapSlice from "../../state/slices/FocusMapSlice";
import { DIMENSIONS, Z_FOCUS_MODES } from "../../state/slices/ExperimentUISlice";
import apiFocusLockControllerGetCurrentFocusValue from "../../backendapi/apiFocusLockControllerGetCurrentFocusValue";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";

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

  // Check if focus map is enabled (for mutual-exclusion warning)
  const focusMapEnabled = useSelector(focusMapSlice.isFocusMapEnabled);

  // Local string states for Z-stack inputs so the user can type negative
  // values (e.g. "-10") without the field resetting to 0 on each keystroke.
  const [zMinRaw, setZMinRaw] = useState(String(parameterValue.zStackMin ?? -10));
  const [zMaxRaw, setZMaxRaw] = useState(String(parameterValue.zStackMax ?? 10));
  const [zStepRaw, setZStepRaw] = useState(String(parameterValue.zStackStepSize ?? 1));

  // Keep local strings in sync when Redux changes from outside (e.g. reset)
  useEffect(() => { setZMinRaw(String(parameterValue.zStackMin)); }, [parameterValue.zStackMin]);
  useEffect(() => { setZMaxRaw(String(parameterValue.zStackMax)); }, [parameterValue.zStackMax]);
  useEffect(() => { setZStepRaw(String(parameterValue.zStackStepSize)); }, [parameterValue.zStackStepSize]);

  // Commit helper: parse raw string and dispatch; on NaN restore to Redux value
  const commitZMin  = () => { const v = parseFloat(zMinRaw);  if (!isNaN(v)) dispatch(experimentSlice.setZStackMin(v));  else setZMinRaw(String(parameterValue.zStackMin)); };
  const commitZMax  = () => { const v = parseFloat(zMaxRaw);  if (!isNaN(v)) dispatch(experimentSlice.setZStackMax(v));  else setZMaxRaw(String(parameterValue.zStackMax)); };
  const commitZStep = () => { const v = parseFloat(zStepRaw); if (!isNaN(v) && v > 0) dispatch(experimentSlice.setZStackStepSize(v)); else setZStepRaw(String(parameterValue.zStackStepSize)); };

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
      {/* Mutual exclusion warning: Focus Map + per-position AF */}
      {focusMapEnabled && (zFocusMode === Z_FOCUS_MODES.AUTOFOCUS || zFocusMode === Z_FOCUS_MODES.Z_STACK_AUTOFOCUS) && (
        <Alert
          severity="warning"
          icon={<WarningAmberIcon />}
          sx={{ mb: 2 }}
        >
          <strong>Focus Map is enabled.</strong> Using per-position autofocus together with Focus Mapping is redundant.
          The focus map already provides Z correction at every XY position. Consider using &quot;Single Z&quot; or &quot;Z-Stack&quot; mode instead.
        </Alert>
      )}

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
          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
            Z-Stack Settings
          </Typography>

          {/* Relative-position hint */}
          <Typography variant="caption" color="textSecondary" sx={{ display: "block", mb: 2 }}>
            Values are <strong>relative offsets</strong> from the current Z position.
            {focusMapEnabled
              ? " With Focus Map enabled, this offset is applied on top of the per-position interpolated Z."
              : " The stack is centred on the Z origin at each scan position."}
          </Typography>

          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", alignItems: "flex-start" }}>
            {/* Start offset */}
            <TextField
              label="Start (µm)"
              type="number"
              size="small"
              value={zMinRaw}
              onChange={(e) => {
                setZMinRaw(e.target.value);
                const v = parseFloat(e.target.value);
                if (!isNaN(v)) dispatch(experimentSlice.setZStackMin(v));
              }}
              onBlur={commitZMin}
              inputProps={{ step: 1 }}
              helperText="Relative start offset"
              sx={{ width: 130 }}
            />

            {/* Stop offset */}
            <TextField
              label="Stop (µm)"
              type="number"
              size="small"
              value={zMaxRaw}
              onChange={(e) => {
                setZMaxRaw(e.target.value);
                const v = parseFloat(e.target.value);
                if (!isNaN(v)) dispatch(experimentSlice.setZStackMax(v));
              }}
              onBlur={commitZMax}
              inputProps={{ step: 1 }}
              helperText="Relative stop offset"
              sx={{ width: 130 }}
            />

            {/* Step size */}
            <TextField
              label="Step Size (µm)"
              type="number"
              size="small"
              value={zStepRaw}
              onChange={(e) => {
                setZStepRaw(e.target.value);
                const v = parseFloat(e.target.value);
                if (!isNaN(v) && v > 0) dispatch(experimentSlice.setZStackStepSize(v));
              }}
              onBlur={commitZStep}
              inputProps={{ step: 0.5, min: 0.01 }}
              helperText="Distance per step"
              sx={{ width: 130 }}
            />
          </Box>

          {/* Summary: planes & total range */}
          <Box
            sx={{
              mt: 2,
              px: 1.5,
              py: 1,
              borderRadius: 1,
              backgroundColor: alpha(theme.palette.primary.main, 0.08),
              display: "flex",
              gap: 3,
              flexWrap: "wrap",
            }}
          >
            <Typography variant="caption">
              <strong>{zStackSteps}</strong> planes
            </Typography>
            <Typography variant="caption">
              Range: <strong>{(parameterValue.zStackMax - parameterValue.zStackMin).toFixed(1)} µm</strong>
            </Typography>
            <Typography variant="caption">
              Step: <strong>{Number(parameterValue.zStackStepSize).toFixed(2)} µm</strong>
            </Typography>
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
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                  <Typography variant="caption" sx={{ fontWeight: 500 }}>
                    Max Attempts
                  </Typography>
                  <Tooltip title="Maximum number of hardware autofocus attempts before giving up. Increase if your sample has weak reflections.">
                    <IconButton size="small" sx={{ p: 0 }}>
                      <InfoIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Tooltip>
                </Box>
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
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
              <Typography variant="caption" sx={{ fontWeight: 500 }}>
                Illumination Channel
              </Typography>
              <Tooltip title="Select which illumination source to use during autofocus. 'Auto' uses the currently active channel.">
                <IconButton size="small" sx={{ p: 0 }}>
                  <InfoIcon sx={{ fontSize: 14 }} />
                </IconButton>
              </Tooltip>
            </Box>
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

