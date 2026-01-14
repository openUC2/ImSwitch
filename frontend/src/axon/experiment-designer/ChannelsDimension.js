import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../../backendapi/createAxiosInstance';
import {
  Box,
  Typography,
  Slider,
  TextField,
  Switch,
  FormControlLabel,
  IconButton,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Collapse,
  Chip,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import SettingsIcon from "@mui/icons-material/Settings";
import DeleteIcon from "@mui/icons-material/Delete";
import AddIcon from "@mui/icons-material/Add";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";
import * as connectionSettingsSlice from "../../state/slices/ConnectionSettingsSlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";
import fetchLaserControllerCurrentValues from "../../middleware/fetchLaserControllerCurrentValues";

/**
 * Single channel block - collapsible card for each illumination source
 */
const ChannelBlock = ({
  channelName,
  channelIndex,
  intensity,
  exposure,
  gain,
  minIntensity,
  maxIntensity,
  isExpanded,
  onToggleExpand,
  onIntensityChange,
  onExposureChange,
  onGainChange,
  onRemove,
}) => {
  const theme = useTheme();

  return (
    <Box
      sx={{
        border: `1px solid ${theme.palette.divider}`,
        borderRadius: 1,
        mb: 1,
        overflow: "hidden",
        backgroundColor: alpha(theme.palette.background.paper, 0.5),
      }}
    >
      {/* Channel Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          padding: "8px 12px",
          backgroundColor: alpha(theme.palette.primary.main, 0.05),
          cursor: "pointer",
          "&:hover": {
            backgroundColor: alpha(theme.palette.primary.main, 0.1),
          },
        }}
        onClick={onToggleExpand}
      >
        {/* Drag handle placeholder */}
        <DragIndicatorIcon
          sx={{
            color: theme.palette.text.disabled,
            fontSize: 18,
            mr: 1,
            cursor: "grab",
          }}
        />

        {/* Channel name and badge */}
        <Chip
          label={channelName}
          size="small"
          color="primary"
          variant="outlined"
          sx={{ mr: 1.5, fontWeight: 600 }}
        />

        {/* Quick info when collapsed */}
        {!isExpanded && (
          <Typography
            variant="caption"
            sx={{ color: theme.palette.text.secondary, flex: 1 }}
          >
            {intensity} mW · {exposure} ms · Gain {gain}
          </Typography>
        )}

        {/* Expand/collapse indicator */}
        <ExpandMoreIcon
          sx={{
            transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
            color: theme.palette.text.secondary,
          }}
        />
      </Box>

      {/* Channel Parameters */}
      <Collapse in={isExpanded}>
        <Box sx={{ p: 2 }}>
          {/* Intensity Slider */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: "block" }}>
              Intensity
            </Typography>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Slider
                value={intensity}
                min={minIntensity}
                max={maxIntensity}
                onChange={(e, val) => onIntensityChange(val)}
                sx={{ flex: 1 }}
              />
              <Typography
                variant="body2"
                sx={{ minWidth: "60px", textAlign: "right", fontWeight: 500 }}
              >
                {intensity} mW
              </Typography>
            </Box>
          </Box>

          {/* Exposure and Gain side by side */}
          <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
            {/* Exposure */}
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: "block" }}>
                Exposure
              </Typography>
              <FormControl size="small" fullWidth>
                <Select
                  value={exposure}
                  onChange={(e) => onExposureChange(e.target.value)}
                >
                  {[10, 20, 50, 100, 200, 500, 1000, 2000].map((val) => (
                    <MenuItem key={val} value={val}>
                      {val} ms
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            {/* Gain */}
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: "block" }}>
                Gain
              </Typography>
              <FormControl size="small" fullWidth>
                <Select
                  value={gain}
                  onChange={(e) => onGainChange(e.target.value)}
                >
                  {[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 23].map((val) => (
                    <MenuItem key={val} value={val}>
                      {val}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          </Box>

          {/* Performance Mode toggle removed - now in OutputDimension */}
        </Box>
      </Collapse>
    </Box>
  );
};

/**
 * ChannelsDimension - Channel configuration interface
 *
 * Contains:
 * - Collapsible channel blocks for each illumination source
 * - Intensity, exposure, gain per channel
 * - Performance mode toggle
 * - Copy settings to all channels action
 * - Advanced camera parameters (hidden by default)
 */
const ChannelsDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const parameterRange = useSelector(parameterRangeSlice.getParameterRangeState);
  const connectionSettings = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  const experimentUI = useSelector(experimentUISlice.getExperimentUIState);

  // Local state for expanded channels
  const [expandedChannels, setExpandedChannels] = useState({});

  // Parameter values from experiment state
  const parameterValue = experimentState.parameterValue;
  const intensities = parameterValue.illuIntensities || [];
  const exposures = parameterValue.exposureTimes || [];
  const gains = parameterValue.gains || [];
  const illuSources = parameterRange.illuSources || [];

  // Initialize arrays if needed
  useEffect(() => {
    if (illuSources.length > 0) {
      const initIntensities = illuSources.map((_, idx) => intensities[idx] ?? 0);
      const initExposures = illuSources.map((_, idx) => exposures[idx] ?? 100);
      const initGains = illuSources.map((_, idx) => gains[idx] ?? 0);

      if (JSON.stringify(intensities) !== JSON.stringify(initIntensities)) {
        dispatch(experimentSlice.setIlluminationIntensities(initIntensities));
      }
      if (JSON.stringify(exposures) !== JSON.stringify(initExposures)) {
        dispatch(experimentSlice.setExposureTimes(initExposures));
      }
      if (JSON.stringify(gains) !== JSON.stringify(initGains)) {
        dispatch(experimentSlice.setGains(initGains));
      }
      dispatch(experimentSlice.setIllumination(illuSources));
    }
  }, [illuSources]);

  // Update summary when channels change
  useEffect(() => {
    const channelCount = illuSources.length;
    const summary =
      channelCount === 0
        ? "No channels available"
        : channelCount === 1
        ? "1 channel"
        : `${channelCount} channels`;

    dispatch(
      experimentUISlice.setDimensionSummary({
        dimension: DIMENSIONS.CHANNELS,
        summary,
      })
    );
    dispatch(
      experimentUISlice.setDimensionConfigured({
        dimension: DIMENSIONS.CHANNELS,
        configured: channelCount > 0,
      })
    );
  }, [illuSources, dispatch]);

  // Handler for intensity change
  const handleIntensityChange = async (index, value) => {
    const arr = [...intensities];
    arr[index] = value;
    dispatch(experimentSlice.setIlluminationIntensities(arr));

    // Update backend immediately for real-time feedback
    const laserName = illuSources[index];
    if (laserName && connectionSettings.ip && connectionSettings.apiPort) {
      try {
        // First, turn off all other channels
        const api = createAxiosInstance();
        for (let i = 0; i < illuSources.length; i++) {
          if (i !== index) {
            const otherLaserName = illuSources[i];
            const encodedOtherLaserName = encodeURIComponent(otherLaserName);
            await api.get(
              `/LaserController/setLaserValue?laserName=${encodedOtherLaserName}&value=0`
            );
          }
        }

        // Then set the selected channel with its intensity
        const encodedLaserName = encodeURIComponent(laserName);
        await api.get(
          `/LaserController/setLaserValue?laserName=${encodedLaserName}&value=${value}`
        );

        // Also send exposure time and gain for this channel
        const currentExposure = exposures[index] ?? 100;
        const currentGain = gains[index] ?? 0;
        
        await api.get(
          `/SettingsController/setDetectorExposureTime?exposureTime=${currentExposure}`
        );
        
        await api.get(
          `/SettingsController/setDetectorGain?gain=${currentGain}`
        );
      } catch (error) {
        console.error("Failed to update laser intensity:", error);
      }
    }
  };


  // Handler for exposure change
  const handleExposureChange = (index, value) => {
    const arr = [...exposures];
    arr[index] = Number(value);
    dispatch(experimentSlice.setExposureTimes(arr));
    // update backend immediately for real-time feedback
    if ( connectionSettings.ip && connectionSettings.apiPort) {
      const api = createAxiosInstance();
      api.get(
        `/SettingsController/setDetectorExposureTime?exposureTime=${value}`
      ).catch((error) => {
        console.error("Failed to update detector exposure time:", error);
      });
    }
  };

  // Handler for gain change
  const handleGainChange = (index, value) => {
    const arr = [...gains];
    arr[index] = Number(value);
    dispatch(experimentSlice.setGains(arr));
    // update backend immediately for real-time feedback
    if (connectionSettings.ip && connectionSettings.apiPort) {
      const api = createAxiosInstance();
      api.get(
        `/SettingsController/setDetectorGain?gain=${value}`
      ).catch((error) => {
        console.error("Failed to update detector gain:", error);
      });
    }
  };

  // Copy settings from first channel to all
  const handleCopyToAll = () => {
    if (illuSources.length <= 1) return;

    const firstIntensity = intensities[0] ?? 0;
    const firstExposure = exposures[0] ?? 100;
    const firstGain = gains[0] ?? 0;

    const newIntensities = illuSources.map(() => firstIntensity);
    const newExposures = illuSources.map(() => firstExposure);
    const newGains = illuSources.map(() => firstGain);

    dispatch(experimentSlice.setIlluminationIntensities(newIntensities));
    dispatch(experimentSlice.setExposureTimes(newExposures));
    dispatch(experimentSlice.setGains(newGains));
  };

  // Toggle channel expansion
  const toggleChannelExpand = (index) => {
    setExpandedChannels((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      {/* Header with actions */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2,
        }}
      >
        <Typography variant="body2" color="textSecondary">
          {illuSources.length} {illuSources.length === 1 ? "channel" : "channels"} selected
        </Typography>

        <Tooltip title="Copy settings from first channel to all others">
          <Button
            size="small"
            variant="outlined"
            startIcon={<ContentCopyIcon />}
            onClick={handleCopyToAll}
            disabled={illuSources.length <= 1}
          >
            Copy settings to all channels
          </Button>
        </Tooltip>
      </Box>

      {/* Channel Blocks */}
      {illuSources.length === 0 ? (
        <Box
          sx={{
            textAlign: "center",
            py: 4,
            color: theme.palette.text.secondary,
          }}
        >
          <Typography variant="body2">
            No illumination sources available.
          </Typography>
          <Typography variant="caption">
            Connect to a microscope to see available channels.
          </Typography>
        </Box>
      ) : (
        illuSources.map((source, idx) => (
          <ChannelBlock
            key={`channel-${source}-${idx}`}
            channelName={source}
            channelIndex={idx}
            intensity={intensities[idx] ?? 0}
            exposure={exposures[idx] ?? 100}
            gain={gains[idx] ?? 0}
            minIntensity={parameterRange.illuSourceMinIntensities?.[idx] ?? 0}
            maxIntensity={parameterRange.illuSourceMaxIntensities?.[idx] ?? 1023}
            isExpanded={expandedChannels[idx] ?? idx === 0}
            onToggleExpand={() => toggleChannelExpand(idx)}
            onIntensityChange={(val) => handleIntensityChange(idx, val)}
            onExposureChange={(val) => handleExposureChange(idx, val)}
            onGainChange={(val) => handleGainChange(idx, val)}
          />
        ))
      )}

      {/* Advanced Settings */}
      <Accordion
        disableGutters
        sx={{
          mt: 2,
          boxShadow: "none",
          border: `1px solid ${theme.palette.divider}`,
          "&:before": { display: "none" },
        }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <SettingsIcon sx={{ mr: 1, fontSize: 18, color: theme.palette.text.secondary }} />
          <Typography variant="body2">Advanced Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="caption" color="textSecondary">
            Advanced camera and illumination parameters will be shown here.
            These settings are typically configured once and rarely changed.
          </Typography>
          {/* Placeholder for advanced settings */}
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default ChannelsDimension;
