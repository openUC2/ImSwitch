import React, { useEffect, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Tooltip,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

/**
 * TimeDimension - Time-lapse configuration interface
 *
 * Contains:
 * - Interval / duration / repetitions
 * - Advanced timing options (hidden by default):
 *   - Focus every N frames
 *   - Channel-specific timing
 *   - Drift correction
 */
const TimeDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const parameterValue = experimentState.parameterValue;

  // Calculate total duration
  const totalDuration = useMemo(() => {
    const interval = parameterValue.timeLapsePeriod || 0;
    const count = parameterValue.numberOfImages || 1;
    const totalSeconds = interval * (count - 1);
    
    if (totalSeconds < 60) {
      return `${Math.round(totalSeconds)} seconds`;
    } else if (totalSeconds < 3600) {
      return `${(totalSeconds / 60).toFixed(1)} minutes`;
    } else {
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.round((totalSeconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  }, [parameterValue.timeLapsePeriod, parameterValue.numberOfImages]);

  // Update summary when parameters change
  useEffect(() => {
    const count = parameterValue.numberOfImages || 1;
    const interval = parameterValue.timeLapsePeriod || 0;
    
    let summary;
    if (count <= 1) {
      summary = "Single timepoint";
    } else {
      summary = `Period ${interval}s · ${count} timepoints`;
    }

    dispatch(experimentUISlice.setDimensionSummary({
      dimension: DIMENSIONS.TIME,
      summary,
    }));
    dispatch(experimentUISlice.setDimensionConfigured({
      dimension: DIMENSIONS.TIME,
      configured: count > 1,
    }));
  }, [parameterValue.numberOfImages, parameterValue.timeLapsePeriod, dispatch]);

  // Interval presets
  const intervalPresets = [
    { label: "1s", value: 1 },
    { label: "5s", value: 5 },
    { label: "10s", value: 10 },
    { label: "30s", value: 30 },
    { label: "1m", value: 60 },
    { label: "5m", value: 300 },
    { label: "10m", value: 600 },
    { label: "30m", value: 1800 },
    { label: "1h", value: 3600 },
  ];

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      {/* Quick Info */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1,
          mb: 3,
          p: 1.5,
          borderRadius: 1,
          backgroundColor: alpha(theme.palette.warning.main, 0.08),
        }}
      >
        <Typography variant="body2" color="textSecondary">
          Estimated minimum duration:
        </Typography>
        <Chip
          label={totalDuration}
          size="small"
          color="warning"
          variant="outlined"
        />
        <Typography variant="caption" color="textSecondary" sx={{ fontStyle: "italic" }}>
          (Actual scan may take longer due to stage movement, autofocus, and acquisition time)
        </Typography>
      </Box>

      {/* Period between acquisitions */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
          Period (time between acquisitions)
        </Typography>
        
        {/* Preset buttons */}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 1.5 }}>
          {intervalPresets.map((preset) => (
            <Chip
              key={preset.value}
              label={preset.label}
              size="small"
              onClick={() => dispatch(experimentSlice.setTimeLapsePeriod(preset.value))}
              color={parameterValue.timeLapsePeriod === preset.value ? "primary" : "default"}
              variant={parameterValue.timeLapsePeriod === preset.value ? "filled" : "outlined"}
              sx={{ cursor: "pointer" }}
            />
          ))}
        </Box>

        {/* Custom period input */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <TextField
            label="Period"
            type="number"
            size="small"
            value={parameterValue.timeLapsePeriod}
            onChange={(e) => dispatch(experimentSlice.setTimeLapsePeriod(Number(e.target.value)))}
            inputProps={{ min: 0, step: 0.1 }}
            sx={{ width: 150 }}
          />
          <Typography variant="body2" color="textSecondary">
            seconds
          </Typography>
        </Box>
        <Typography variant="caption" color="textSecondary" sx={{ mt: 0.5, fontStyle: "italic" }}>
          Time between the start of one acquisition and the start of the next.
          If the acquisition takes longer than the period, the next one starts immediately.
        </Typography>
      </Box>

      {/* Number of timepoints */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
          Number of Timepoints
        </Typography>
        
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Slider
            value={parameterValue.numberOfImages || 1}
            min={1}
            max={100}
            onChange={(e, val) => dispatch(experimentSlice.setNumberOfImages(val))}
            sx={{ flex: 1, maxWidth: 300 }}
            marks={[
              { value: 1, label: "1" },
              { value: 25, label: "25" },
              { value: 50, label: "50" },
              { value: 75, label: "75" },
              { value: 100, label: "100" },
            ]}
          />
          <TextField
            type="number"
            size="small"
            value={parameterValue.numberOfImages || 1}
            onChange={(e) => dispatch(experimentSlice.setNumberOfImages(Number(e.target.value)))}
            inputProps={{ min: 1 }}
            sx={{ width: 80 }}
          />
        </Box>
      </Box>

      {/* Quick duration calculator */}
      <Box
        sx={{
          p: 2,
          mb: 2,
          border: `1px solid ${theme.palette.divider}`,
          borderRadius: 1,
          backgroundColor: alpha(theme.palette.background.default, 0.5),
        }}
      >
        <Typography variant="caption" color="textSecondary" sx={{ mb: 1, display: "block" }}>
          Duration Calculator
        </Typography>
        <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
          <Box>
            <Typography variant="caption" color="textSecondary">
              First acquisition
            </Typography>
            <Typography variant="body2">0:00</Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="textSecondary">
              Last acquisition
            </Typography>
            <Typography variant="body2">{totalDuration}</Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="textSecondary">
              Total frames
            </Typography>
            <Typography variant="body2">{parameterValue.numberOfImages || 1}</Typography>
          </Box>
        </Box>
      </Box>

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
            {/* Focus every N frames */}
            <Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  Focus every N frames
                </Typography>
                <Tooltip title="Run autofocus every N timepoints to correct for focus drift during long time-lapse acquisitions." arrow>
                  <InfoOutlinedIcon sx={{ fontSize: 14, color: "text.disabled", cursor: "help" }} />
                </Tooltip>
              </Box>
              <TextField
                type="number"
                size="small"
                defaultValue={1}
                inputProps={{ min: 1 }}
                sx={{ width: 100 }}
                helperText="Run autofocus at this interval"
              />
            </Box>

            {/* Drift correction placeholder */}
            <FormControlLabel
              control={<Switch size="small" disabled />}
              label={
                <Box>
                  <Typography variant="body2">Drift Correction</Typography>
                  <Typography variant="caption" color="textSecondary">
                    Compensate for XY drift between timepoints (coming soon)
                  </Typography>
                </Box>
              }
            />

            {/* Channel-specific timing placeholder */}
            <FormControlLabel
              control={<Switch size="small" disabled />}
              label={
                <Box>
                  <Typography variant="body2">Channel-specific Timing</Typography>
                  <Typography variant="caption" color="textSecondary">
                    Different intervals per channel (coming soon)
                  </Typography>
                </Box>
              }
            />
          </Box>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default TimeDimension;
