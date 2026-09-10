import React, { useEffect, useMemo, useState } from "react";
import { useSelector } from "react-redux";
import { Box, Typography, Divider, Chip } from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import LocationOnIcon from "@mui/icons-material/LocationOn";
import TuneIcon from "@mui/icons-material/Tune";
import LayersIcon from "@mui/icons-material/Layers";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import TimerIcon from "@mui/icons-material/Timer";
import StorageIcon from "@mui/icons-material/Storage";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import FilterCenterFocusIcon from "@mui/icons-material/FilterCenterFocus";
import Tooltip from "@mui/material/Tooltip";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice";
import * as wellSelectorSlice from "../../state/slices/WellSelectorSlice";
import { getStorageState } from "../../state/slices/StorageSlice";
import apiExperimentControllerGetFocusMapSummary from "../../backendapi/apiExperimentControllerGetFocusMapSummary";
import { DIMENSIONS, Z_FOCUS_MODES } from "../../state/slices/ExperimentUISlice";
import * as coordinateCalculator from "../CoordinateCalculator";

const formatSize = (megabytes) =>
  megabytes < 1024
    ? `${Math.round(megabytes)} MB`
    : `${(megabytes / 1024).toFixed(1)} GB`;

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
 * Does this run fit on the drive it will be written to? Returns null when we
 * have no free-space reading. The estimate was always shown but never compared
 * to the drive, so a run could only fail once the disk filled up.
 */
export const checkFitsOnDrive = (dataSizeMB, device) => {
  const freeBytes = device?.usage?.free;
  if (typeof freeBytes !== "number") return null;
  const freeMB = freeBytes / (1024 * 1024);
  return {
    name: device.label || device.path || "the active drive",
    freeText: formatSize(freeMB),
    freeMB,
    fits: dataSizeMB <= freeMB,
  };
};

/**
 * Positions x channels x Z planes x timepoints, and what that costs in time and
 * bytes. Exported so the Start guard can check the same number the summary bar
 * shows instead of estimating it a second way.
 */
export const computeSummary = ({
  experimentState, experimentUI, parameterRange, objectiveState, wellSelectorState,
}) => {
    const dimensions = experimentUI.dimensions;
    const params = experimentState.parameterValue;
    
    // Positions – use coordinate calculator to count all tiles, not just
    // the raw pointList length (which is the number of scan areas).
    let totalPositions = 0;
    try {
      const scanConfig = coordinateCalculator.calculateScanCoordinates(
        experimentState,
        objectiveState,
        wellSelectorState
      );
      totalPositions = scanConfig.metadata.totalPositions || 0;
    } catch {
      // Fallback: use raw point list length
      totalPositions = experimentState.pointList?.length || 0;
    }
    
    // Channels – count only those enabled for the experiment
    const channelEnabled = params.channelEnabledForExperiment || [];
    const allChannelCount = params.illumination?.length || parameterRange.illuSources?.length || 0;
    const enabledChannels = dimensions[DIMENSIONS.CHANNELS]?.enabled
      ? (channelEnabled.filter(Boolean).length || allChannelCount)
      : 1; // Single channel when dimension disabled
    
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
    
    // Estimate data size using FOV and pixel size to derive sensor dimensions.
    // objectiveState.fovX / fovY are in µm; pixelsize is µm/px.
    const effectivePixelSize = objectiveState?.pixelsize || 1;
    const pixelWidth = effectivePixelSize > 0 && objectiveState?.fovX
      ? Math.round(objectiveState.fovX / effectivePixelSize)
      : 2048;
    const pixelHeight = effectivePixelSize > 0 && objectiveState?.fovY
      ? Math.round(objectiveState.fovY / effectivePixelSize)
      : 2048;
    const bytesPerPixel = 2; // 16-bit
    const imageSizeMB = (pixelWidth * pixelHeight * bytesPerPixel) / (1024 * 1024);
    const totalImages = totalPositions * enabledChannels * zPlanes * timepoints;
    const totalDataMB = totalImages * imageSizeMB;
    
    const dataSizeStr = formatSize(totalDataMB);
    
    return {
      positions: totalPositions,
      channels: enabledChannels,
      zPlanes,
      timepoints,
      duration: durationStr,
      dataSize: dataSizeStr,
      dataSizeMB: totalDataMB,
      totalImages,
  };
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
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const storageState = useSelector(getStorageState);

  // Which focus map each region would actually run on. Refreshed on the same
  // cadence as the rest of the bar so it reflects the latest fit.
  const [focusMaps, setFocusMaps] = useState(null);
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      apiExperimentControllerGetFocusMapSummary()
        .then((data) => !cancelled && setFocusMaps(data))
        .catch(() => !cancelled && setFocusMaps(null));
    load();
    const timer = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const summaryData = useMemo(
    () => computeSummary({
      experimentState, experimentUI, parameterRange, objectiveState, wellSelectorState,
    }),
    [experimentState, experimentUI, parameterRange, objectiveState, wellSelectorState],
  );

  const storage = useMemo(
    () => checkFitsOnDrive(summaryData.dataSizeMB, storageState?.status?.active_device),
    [storageState, summaryData.dataSizeMB],
  );

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
        value={
          storage
            ? `${summaryData.dataSize} / ${storage.freeText} free`
            : summaryData.dataSize
        }
        color={storage && !storage.fits ? "error" : "success"}
      />

      {storage && !storage.fits && (
        <Tooltip
          title={`This run needs about ${summaryData.dataSize}, but only ${storage.freeText} is free on ${storage.name}.`}
        >
          <Chip
            size="small"
            icon={<WarningAmberIcon />}
            color="error"
            label="Will not fit"
            sx={{ fontSize: "0.7rem", height: "22px" }}
          />
        </Tooltip>
      )}
      
      {focusMaps?.focus_map_active && focusMaps.regions?.length > 0 && (
        <Tooltip
          title={
            <Box sx={{ whiteSpace: "pre-line" }}>
              {focusMaps.regions
                .map(
                  (r) =>
                    `${r.region_name} [${r.region_id}]: ${
                      r.has_map ? `${r.method}, n=${r.n_points}` : "no map"
                    } — ${r.quality}`,
                )
                .join("\n")}
            </Box>
          }
        >
          <Chip
            size="small"
            icon={<FilterCenterFocusIcon />}
            color={
              focusMaps.regions.every((r) => r.has_map && !r.reason)
                ? "success"
                : "warning"
            }
            variant="outlined"
            label={`Focus map: ${
              focusMaps.regions.filter((r) => r.has_map).length
            }/${focusMaps.regions.length} regions`}
            sx={{ fontSize: "0.7rem", height: "22px" }}
          />
        </Tooltip>
      )}

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
