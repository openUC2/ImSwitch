import React, { useRef, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../../backendapi/createAxiosInstance';
import {
  Box,
  Typography,
  Button,
  ButtonGroup,
  LinearProgress,
  IconButton,
  Tooltip,
  Divider,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import StopIcon from "@mui/icons-material/Stop";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import VisibilityIcon from "@mui/icons-material/Visibility";

// Dimension components
import DimensionBar from "./DimensionBar";
import ExperimentSummary from "./ExperimentSummary";
import PositionsDimension from "./PositionsDimension";
import ChannelsDimension from "./ChannelsDimension";
import ZFocusDimension from "./ZFocusDimension";
import TimeDimension from "./TimeDimension";
import TilingDimension from "./TilingDimension";
import OutputDimension from "./OutputDimension";
import FocusMapDimension from "./FocusMapDimension";

// Utilities
import * as coordinateCalculator from "../CoordinateCalculator";
import InfoPopup from "../InfoPopup";

// State slices
import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as experimentStatusSlice from "../../state/slices/ExperimentStatusSlice";
import * as experimentStateSlice from "../../state/slices/ExperimentStateSlice";
import * as wellSelectorSlice from "../../state/slices/WellSelectorSlice";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice";
import * as connectionSettingsSlice from "../../state/slices/ConnectionSettingsSlice";
import * as vizarrViewerSlice from "../../state/slices/VizarrViewerSlice";
import * as focusMapSlice from "../../state/slices/FocusMapSlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

// API
import apiExperimentControllerStartWellplateExperiment from "../../backendapi/apiExperimentControllerStartWellplateExperiment";
import apiExperimentControllerStopExperiment from "../../backendapi/apiExperimentControllerStopExperiment";
import apiExperimentControllerPauseWorkflow from "../../backendapi/apiExperimentControllerPauseWorkflow";
import apiExperimentControllerResumeExperiment from "../../backendapi/apiExperimentControllerResumeExperiment";
import fetchGetExperimentStatus from "../../middleware/fetchExperimentControllerGetExperimentStatus";

// Status enum
const Status = Object.freeze({
  IDLE: "idle",
  RUNNING: "running",
  PAUSED: "paused",
  STOPPING: "stopping",
});

/**
 * ExperimentDesigner - Main container for the dimension-based experiment UI
 * 
 * Structure:
 * - Header with experiment controls (Start/Pause/Stop)
 * - Dimension bar for navigation
 * - Active dimension panel
 * - Persistent experiment summary
 */
const ExperimentDesigner = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const infoPopupRef = useRef(null);

  // Redux state
  const connectionSettings = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const experimentUI = useSelector(experimentUISlice.getExperimentUIState);
  const experimentStatus = useSelector(experimentStatusSlice.getExperimentStatusState);
  const experimentWorkflow = useSelector(experimentStateSlice.getExperimentState);
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const focusMapConfig = useSelector(focusMapSlice.getFocusMapConfig);

  // Progress tracking
  const [cachedStepId, setCachedStepId] = useState(0);
  const [cachedTotalSteps, setCachedTotalSteps] = useState(undefined);
  const [cachedStepName, setCachedStepName] = useState("");

  // Periodic status fetch
  useEffect(() => {
    fetchGetExperimentStatus(dispatch);
    const intervalId = setInterval(() => {
      fetchGetExperimentStatus(dispatch);
    }, 3000);
    return () => clearInterval(intervalId);
  }, [dispatch]);

  // Update cached progress values
  useEffect(() => {
    if (experimentWorkflow.totalSteps !== undefined) {
      setCachedTotalSteps(experimentWorkflow.totalSteps);
    }
    if (experimentWorkflow.stepId !== undefined) {
      setCachedStepId(experimentWorkflow.stepId);
    }
    if (experimentWorkflow.stepName) {
      setCachedStepName(experimentWorkflow.stepName);
    }
  }, [experimentWorkflow.totalSteps, experimentWorkflow.stepId, experimentWorkflow.stepName]);

  // Calculate progress
  const progress = cachedTotalSteps && cachedTotalSteps > 0
    ? Math.floor((cachedStepId / cachedTotalSteps) * 100)
    : 0;

  // Dimension to component mapping
  const dimensionComponents = {
    [DIMENSIONS.POSITIONS]: PositionsDimension,
    [DIMENSIONS.CHANNELS]: ChannelsDimension,
    [DIMENSIONS.Z_FOCUS]: ZFocusDimension,
    [DIMENSIONS.FOCUS_MAP]: FocusMapDimension,
    [DIMENSIONS.TIME]: TimeDimension,
    [DIMENSIONS.TILING]: TilingDimension,
    [DIMENSIONS.OUTPUT]: OutputDimension,
  };

  // Get active dimension component
  const ActiveDimensionComponent = dimensionComponents[experimentUI.expandedDimension];
  const isActiveDimensionEnabled = experimentUI.dimensions[experimentUI.expandedDimension]?.enabled ?? true;

  // Control handlers
  const handleStart = () => {
    console.log("Experiment started");
    dispatch(experimentSlice.setIsSnakescan(wellSelectorState.areaSelectSnakescan));

    if (wellSelectorState.mode === "area") {
      dispatch(experimentSlice.setOverlapWidth(wellSelectorState.areaSelectOverlap));
      dispatch(experimentSlice.setOverlapHeight(wellSelectorState.areaSelectOverlap));
    }

    const scanConfig = coordinateCalculator.calculateScanCoordinates(
      experimentState,
      objectiveState,
      wellSelectorState
    );

    console.log("Scan configuration:", scanConfig);
    console.log(`Total positions: ${scanConfig.metadata.totalPositions}`);

    // Zero out intensities for channels that are not enabled for experiment acquisition
    const channelEnabled = experimentState.parameterValue.channelEnabledForExperiment || [];
    const rawIntensities = experimentState.parameterValue.illuIntensities || [];
    const filteredIntensities = rawIntensities.map((val, idx) =>
      channelEnabled[idx] === true ? val : 0    );

    const experimentRequest = {
      name: experimentState.name,
      parameterValue: {
        ...experimentState.parameterValue,
        illuIntensities: filteredIntensities,
        resortPointListToSnakeCoordinates: false,
        is_snakescan: wellSelectorState.areaSelectSnakescan,
        overlapWidth: wellSelectorState.mode === "area" 
          ? wellSelectorState.areaSelectOverlap 
          : experimentState.parameterValue.overlapWidth,
        overlapHeight: wellSelectorState.mode === "area" 
          ? wellSelectorState.areaSelectOverlap 
          : experimentState.parameterValue.overlapHeight,
      },
      scanAreas: scanConfig.scanAreas,
      scanMetadata: scanConfig.metadata,
      pointList: coordinateCalculator.convertToBackendFormat(scanConfig, experimentState).pointList,
      focusMap: focusMapConfig.enabled ? focusMapConfig : undefined,
    };

    apiExperimentControllerStartWellplateExperiment(experimentRequest)
      .then(() => {
        dispatch(experimentStatusSlice.setStatus(Status.RUNNING));
        infoPopupRef.current?.showMessage("Experiment started");
      })
      .catch(() => {
        infoPopupRef.current?.showMessage("Start Experiment failed");
      });
  };

  const handlePause = () => {
    apiExperimentControllerPauseWorkflow()
      .then(() => {
        dispatch(experimentStatusSlice.setStatus(Status.PAUSED));
        infoPopupRef.current?.showMessage("Experiment paused");
      })
      .catch(() => {
        infoPopupRef.current?.showMessage("Pause Experiment failed");
      });
  };

  const handleResume = () => {
    apiExperimentControllerResumeExperiment()
      .then(() => {
        dispatch(experimentStatusSlice.setStatus(Status.RUNNING));
        infoPopupRef.current?.showMessage("Experiment resumed");
      })
      .catch(() => {
        infoPopupRef.current?.showMessage("Resume Experiment failed");
      });
  };

  const handleStop = () => {
    dispatch(experimentStatusSlice.setStatus(Status.IDLE));
    apiExperimentControllerStopExperiment()
      .then(() => {
        infoPopupRef.current?.showMessage("Experiment stopped");
      })
      .catch(() => {
        infoPopupRef.current?.showMessage("Stop Experiment failed");
      });
  };

  const handleOpenVizarr = () => {
    const api = createAxiosInstance();
    api.get(
      `/ExperimentController/getLastScanAsOMEZARR`
    )
      .then((res) => res.data)
      .then((data) => {
        const lastZarrPath = data || "";
        if (lastZarrPath) {
          dispatch(vizarrViewerSlice.openViewer({
            url: lastZarrPath,
            fileName: lastZarrPath.split("/").pop() || "OME-Zarr",
          }));
          infoPopupRef.current?.showMessage("Opening OME-Zarr in integrated viewer");
        } else {
          infoPopupRef.current?.showMessage("No OME-Zarr data available");
        }
      })
      .catch(() => {
        infoPopupRef.current?.showMessage("Failed to open OME-Zarr");
      });
  };

  // Button visibility helpers
  const showStart = experimentStatus.status === Status.IDLE || experimentStatus.status === Status.STOPPING;
  const showPause = experimentStatus.status === Status.RUNNING;
  const showResume = experimentStatus.status === Status.PAUSED;
  const showStop = experimentStatus.status !== Status.IDLE;

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        backgroundColor: theme.palette.background.default,
      }}
    >
      {/* Header with controls */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 2,
          p: 1.5,
          borderBottom: `1px solid ${theme.palette.divider}`,
          backgroundColor: alpha(theme.palette.background.paper, 0.8),
        }}
      >
        {/* Control Buttons */}
        <ButtonGroup size="small" variant="contained">
          <Tooltip title="Start experiment">
            <span>
              <Button
                onClick={handleStart}
                disabled={!showStart}
                color="success"
                startIcon={<PlayArrowIcon />}
              >
                Start
              </Button>
            </span>
          </Tooltip>
          <Tooltip title="Pause experiment">
            <span>
              <Button onClick={handlePause} disabled={!showPause}>
                <PauseIcon />
              </Button>
            </span>
          </Tooltip>
          <Tooltip title="Resume experiment">
            <span>
              <Button onClick={handleResume} disabled={!showResume}>
                <RestartAltIcon />
              </Button>
            </span>
          </Tooltip>
          <Tooltip title="Stop experiment">
            <span>
              <Button onClick={handleStop} disabled={!showStop} color="error">
                <StopIcon />
              </Button>
            </span>
          </Tooltip>
        </ButtonGroup>

        {/* Status */}
        <Typography
          variant="body2"
          sx={{
            px: 1.5,
            py: 0.5,
            borderRadius: 1,
            backgroundColor: alpha(
              experimentStatus.status === Status.RUNNING
                ? theme.palette.success.main
                : experimentStatus.status === Status.PAUSED
                ? theme.palette.warning.main
                : theme.palette.grey[500],
              0.15
            ),
            fontWeight: 500,
          }}
        >
          {experimentStatus.status}
        </Typography>

        {/* Progress */}
        {cachedTotalSteps && cachedTotalSteps > 0 && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flex: 1, maxWidth: 300 }}>
            <Typography
              variant="caption"
              sx={{
                maxWidth: 100,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
              title={cachedStepName}
            >
              {cachedStepName}
            </Typography>
            <Box sx={{ flex: 1 }}>
              <LinearProgress variant="determinate" value={progress} />
            </Box>
            <Typography variant="caption">{progress}%</Typography>
          </Box>
        )}

        {/* Spacer */}
        <Box sx={{ flex: 1 }} />

        {/* Viewer buttons */}
        <Tooltip title="Open Vizarr viewer">
          <Button 
            size="small" 
            variant="outlined" 
            onClick={handleOpenVizarr} 
            startIcon={<VisibilityIcon />}
          >
            Open Vizarr
          </Button>
        </Tooltip>
      </Box>

      {/* Dimension Bar */}
      <DimensionBar />

      {/* Active Dimension Panel */}
      <Box
        sx={{
          flex: 1,
          overflow: "auto",
          p: 2,
        }}
      >
        {ActiveDimensionComponent && isActiveDimensionEnabled ? (
          <ActiveDimensionComponent />
        ) : (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: theme.palette.text.secondary,
            }}
          >
            <Typography variant="body1" sx={{ mb: 1 }}>
              This dimension is disabled
            </Typography>
            <Typography variant="caption">
              Click on the dimension tab to enable it
            </Typography>
          </Box>
        )}
      </Box>

      {/* Experiment Summary */}
      <ExperimentSummary />

      <InfoPopup ref={infoPopupRef} />
    </Box>
  );
};

export default ExperimentDesigner;
