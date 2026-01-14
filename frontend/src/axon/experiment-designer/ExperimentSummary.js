import React, { useMemo } from "react";
import { useSelector } from "react-redux";
import { Box, Typography, Divider, Chip } from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import LocationOnIcon from "@mui/icons-material/LocationOn";
import TuneIcon from "@mui/icons-material/Tune";
import LayersIcon from "@mui/icons-material/Layers";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import TimerIcon from "@mui/icons-material/Timer";
import StorageIcon from "@mui/icons-material/Storage";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice";
import { DIMENSIONS, Z_FOCUS_MODES } from "../../state/slices/ExperimentUISlice";

/**
 * Summary stat item with icon
 */
const SummaryStat = ({ icon: Icon, label, value, color = "primary" }) => {
  const theme = useTheme();
  
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 0.75,
        padding: "4px 8px",
        borderRadius: "4px",
        backgroundColor: alpha(theme.palette[color]?.main || theme.palette.primary.main, 0.08),
      }}
    >
      <Icon 
        sx={{ 
          fontSize: 16, 
          color: theme.palette[color]?.main || theme.palette.primary.main,
        }} 
      />
      <Box sx={{ display: "flex", flexDirection: "column" }}>
        <Typography
          variant="caption"
          sx={{
            fontSize: "0.65rem",
            color: theme.palette.text.secondary,
            lineHeight: 1.2,
          }}
        >
          {label}
        </Typography>
        <Typography
          variant="body2"
          sx={{
            fontWeight: 600,
            fontSize: "0.8rem",
            color: theme.palette.text.primary,
            lineHeight: 1.2,
          }}
        >
          {value}
        </Typography>
      </Box>
    </Box>
  );
};

/**
 * ExperimentSummary - Always-visible compact summary panel
 * 
 * Shows:
 * - Number of positions
 * - Channels count
 * - Z planes
 * - Timepoints
 * - Estimated duration
 * - Estimated data size
 */
const ExperimentSummary = () => {
  const theme = useTheme();
  
  // Get experiment state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const experimentUI = useSelector(experimentUISlice.getExperimentUIState);
  const parameterRange = useSelector(parameterRangeSlice.getParameterRangeState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  
  // Calculate summary values
  const summaryData = useMemo(() => {
    const dimensions = experimentUI.dimensions;
    const params = experimentState.parameterValue;
    
    // Positions
    const totalPositions = experimentState.pointList?.length || 0;
    
    // Channels (count enabled illumination sources)
    const enabledChannels = dimensions[DIMENSIONS.CHANNELS]?.enabled
      ? (params.illumination?.length || parameterRange.illuSources?.length || 0)
      : 1; // Single channel when disabled
    
    // Z planes
    let zPlanes = 1;
    const zFocusEnabled = dimensions[DIMENSIONS.Z_FOCUS]?.enabled;
    const zFocusMode = dimensions[DIMENSIONS.Z_FOCUS]?.mode;
    
    if (zFocusEnabled && (zFocusMode === Z_FOCUS_MODES.Z_STACK || zFocusMode === Z_FOCUS_MODES.Z_STACK_AUTOFOCUS)) {
      const zRange = params.zStackMax - params.zStackMin;
      const zStep = params.zStackStepSize || 1;
      zPlanes = Math.max(1, Math.ceil(zRange / zStep) + 1);
    }
    
    // Timepoints
    const timeEnabled = dimensions[DIMENSIONS.TIME]?.enabled;
    const timepoints = timeEnabled ? (params.numberOfImages || 1) : 1;
    
    // Calculate estimates
    const avgExposureMs = Array.isArray(params.exposureTimes)
      ? params.exposureTimes.reduce((a, b) => a + b, 0) / params.exposureTimes.length
      : params.exposureTimes || 100;
    
    const moveTimePerPositionMs = 500; // Estimated stage move time
    const timeLapseIntervalS = params.timeLapsePeriod || 0;
    
    // Total acquisition time estimation
    const acquisitionsPerPosition = enabledChannels * zPlanes;
    const timePerPositionMs = acquisitionsPerPosition * (avgExposureMs + 50) + moveTimePerPositionMs;
    const singleCycleTimeMs = totalPositions * timePerPositionMs;
    const totalTimeMs = timepoints > 1
      ? singleCycleTimeMs + (timepoints - 1) * Math.max(singleCycleTimeMs, timeLapseIntervalS * 1000)
      : singleCycleTimeMs;
    
    // Convert to human readable
    const totalMinutes = totalTimeMs / 1000 / 60;
    let durationStr;
    if (totalMinutes < 1) {
      durationStr = `${Math.round(totalTimeMs / 1000)}s`;
    } else if (totalMinutes < 60) {
      durationStr = `${totalMinutes.toFixed(1)} min`;
    } else {
      const hours = Math.floor(totalMinutes / 60);
      const mins = Math.round(totalMinutes % 60);
      durationStr = `${hours}h ${mins}m`;
    }
    
    // Estimate data size (rough: 2MB per 16-bit 2048x2048 image)
    const pixelWidth = objectiveState?.fov?.width || 2048;
    const pixelHeight = objectiveState?.fov?.height || 2048;
    const bytesPerPixel = 2; // 16-bit
    const imageSizeMB = (pixelWidth * pixelHeight * bytesPerPixel) / (1024 * 1024);
    const totalImages = totalPositions * enabledChannels * zPlanes * timepoints;
    const totalDataMB = totalImages * imageSizeMB;
    
    let dataSizeStr;
    if (totalDataMB < 1024) {
      dataSizeStr = `${Math.round(totalDataMB)} MB`;
    } else {
      dataSizeStr = `${(totalDataMB / 1024).toFixed(1)} GB`;
    }
    
    return {
      positions: totalPositions,
      channels: enabledChannels,
      zPlanes,
      timepoints,
      duration: durationStr,
      dataSize: dataSizeStr,
      totalImages,
    };
  }, [experimentState, experimentUI, parameterRange, objectiveState]);

  return (
    <Box
      sx={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 1,
        padding: "8px 12px",
        backgroundColor: alpha(theme.palette.background.default, 0.5),
        borderTop: `1px solid ${theme.palette.divider}`,
        minHeight: "48px",
      }}
    >
      {/* Title */}
      <Typography
        variant="caption"
        sx={{
          fontWeight: 600,
          color: theme.palette.text.secondary,
          textTransform: "uppercase",
          fontSize: "0.65rem",
          mr: 1,
        }}
      >
        Summary
      </Typography>
      
      <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />
      
      {/* Stats */}
      <SummaryStat
        icon={LocationOnIcon}
        label="Positions"
        value={summaryData.positions}
        color="primary"
      />
      
      <SummaryStat
        icon={TuneIcon}
        label="Channels"
        value={summaryData.channels}
        color="secondary"
      />
      
      <SummaryStat
        icon={LayersIcon}
        label="Z Planes"
        value={summaryData.zPlanes}
        color="info"
      />
      
      <SummaryStat
        icon={AccessTimeIcon}
        label="Timepoints"
        value={summaryData.timepoints}
        color="warning"
      />
      
      <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />
      
      <SummaryStat
        icon={TimerIcon}
        label="Est. Duration"
        value={summaryData.duration}
        color="success"
      />
      
      <SummaryStat
        icon={StorageIcon}
        label="Est. Size"
        value={summaryData.dataSize}
        color="error"
      />
      
      {/* Total images chip */}
      <Chip
        size="small"
        label={`${summaryData.totalImages} images`}
        sx={{
          ml: "auto",
          fontSize: "0.7rem",
          height: "22px",
        }}
      />
    </Box>
  );
};

export default ExperimentSummary;
