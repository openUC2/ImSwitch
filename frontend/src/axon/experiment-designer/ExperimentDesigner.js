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
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";

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
import ObjectiveDimension from "./ObjectiveDimension";

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
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";
import * as positionSlice from "../../state/slices/PositionSlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

// API
import apiExperimentControllerStartWellplateExperiment from "../../backendapi/apiExperimentControllerStartWellplateExperiment";
import apiExperimentControllerStopExperiment from "../../backendapi/apiExperimentControllerStopExperiment";
import apiExperimentControllerPauseWorkflow from "../../backendapi/apiExperimentControllerPauseWorkflow";
import apiExperimentControllerResumeExperiment from "../../backendapi/apiExperimentControllerResumeExperiment";
import apiExperimentControllerInterruptFocusMap from "../../backendapi/apiExperimentControllerInterruptFocusMap";
import apiExperimentControllerRunAshlarStitching from "../../backendapi/apiExperimentControllerRunAshlarStitching";
import apiExperimentControllerGetOverviewAsyncStatus from "../../backendapi/apiExperimentControllerGetOverviewAsyncStatus";
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
  const parameterRange = useSelector(parameterRangeSlice.getParameterRangeState);
  const positionState = useSelector(positionSlice.getPositionState);

  // Progress tracking
  const [cachedStepId, setCachedStepId] = useState(0);
  const [cachedTotalSteps, setCachedTotalSteps] = useState(undefined);
  const [cachedStepName, setCachedStepName] = useState("");
  const [ashlarRunning, setAshlarRunning] = useState(false);
  const [ashlarInterrupted, setAshlarInterrupted] = useState(false);

  // Periodic status fetch
  useEffect(() => {
    fetchGetExperimentStatus(dispatch);
    const intervalId = setInterval(() => {
      fetchGetExperimentStatus(dispatch);
    }, 3000);
    return () => clearInterval(intervalId);
  }, [dispatch]);

  // Trigger Ashlar stitching automatically when the experiment finishes.
  // Use a boolean ref so we catch any IDLE transition that follows a RUNNING
  // phase — including RUNNING→STOPPING→IDLE flows.
  const wasRunningRef = useRef(false);
  useEffect(() => {
    const curr = experimentStatus.status;
    if (curr === Status.RUNNING) {
      wasRunningRef.current = true;
    } else if (wasRunningRef.current && curr === Status.IDLE) {
      wasRunningRef.current = false;
      if (experimentState.parameterValue.ome_write_ashlar_stitch) {
        // Backend auto-starts stitching once all tiles are written.
        // Start polling so we can show progress as soon as it begins.
        setAshlarRunning(true);
      }
    }
  }, [
    experimentStatus.status,
    experimentState.parameterValue.ome_write_ashlar_stitch,
    experimentState.parameterValue.ashlar_pixel_size,
    experimentState.parameterValue.ashlar_maximum_shift,
    experimentState.parameterValue.ashlar_align_channel,
  ]);

  // Poll stitching progress until the background job finishes
  useEffect(() => {
    if (!ashlarRunning) return;
    const intervalId = setInterval(() => {
      apiExperimentControllerGetOverviewAsyncStatus()
        .then((status) => {
          if (status?.message)
            infoPopupRef.current?.showMessage(`Stitching: ${status.message}`);
          if (!status?.running) {
            setAshlarRunning(false);
            if (status?.error)
              infoPopupRef.current?.showMessage(`Stitching failed: ${status.error.slice(0, 120)}`);
            else
              infoPopupRef.current?.showMessage("Ashlar stitching complete");
          }
        })
        .catch(() => {});
    }, 3000);
    return () => clearInterval(intervalId);
  }, [ashlarRunning]);

  // Warn the user before leaving/refreshing the page while an experiment is running
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (
        experimentStatus.status === Status.RUNNING ||
        experimentStatus.status === Status.PAUSED
      ) {
        e.preventDefault();
        // Legacy browsers need returnValue to be set
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [experimentStatus.status]);

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
    [DIMENSIONS.OBJECTIVE]: ObjectiveDimension,
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

    // "Override per-group Z with current Z" (Tiling tab): rewrite every stored
    // Z with the microscope's current stage Z. Done here on the frontend (the
    // backend just consumes the coordinates). Skipped when a focus map is
    // active, since the focus map drives Z per-XY.
    if (experimentState.parameterValue.overrideZWithCurrentZ && !focusMapConfig.enabled) {
      const currentZ = positionState?.z ?? 0;
      (scanConfig.scanAreas || []).forEach((area) => {
        if (area.centerPosition) area.centerPosition.z = currentZ;
        (area.positions || []).forEach((pos) => { pos.z = currentZ; });
      });
    }

    console.log("Scan configuration:", scanConfig);
    console.log(`Total positions: ${scanConfig.metadata.totalPositions}`);

    // Zero out intensities for channels that are not enabled for experiment acquisition
    const pv = experimentState.parameterValue;
    const channelEnabled = pv.channelEnabledForExperiment || [];
    const rawIntensities = pv.illuIntensities || [];
    const exposureTimes = pv.exposureTimes || [];
    const gains = pv.gains || [];
    const illuminationParamsState = pv.illuminationParams || {};
    const filteredIntensities = rawIntensities.map((val, idx) =>
      channelEnabled[idx] === true ? val : 0    );

    // Split the flat channel list into conventional sources + a dedicated
    // synthetic (ring/DPC) list. The Channels dimension renders one merged
    // list [default..., synthetic...]; here we cut it back at the default-source
    // boundary so the REST payload keeps `illumination` conventional-only and
    // carries ring/DPC in `syntheticChannels` (single source of truth, no
    // RGB→intensity promotion on the backend).
    const defaultSourceNames = Array.isArray(parameterRange.illuSources) ? parameterRange.illuSources : [];
    const syntheticDefs = Array.isArray(parameterRange.syntheticChannels) ? parameterRange.syntheticChannels : [];
    const nDefault = defaultSourceNames.length;
    const syntheticChannelsPayload = syntheticDefs.map((s, j) => {
      const idx = nDefault + j;
      const params = illuminationParamsState[s.name] || {};
      return {
        name: s.name,
        kind: s.kind,
        enabled: channelEnabled[idx] === true,
        intensityR: params.intensityR ?? s.intensityR ?? 0,
        intensityG: params.intensityG ?? s.intensityG ?? 0,
        intensityB: params.intensityB ?? s.intensityB ?? 0,
        radius: params.radius ?? s.radius ?? null,
        exposure: exposureTimes[idx] ?? null,
        gain: gains[idx] ?? null,
      };
    });

    const experimentRequest = {
      name: experimentState.name,
      parameterValue: {
        ...pv,
        // Conventional channels only; synthetic channels travel separately.
        illumination: defaultSourceNames,
        illuIntensities: filteredIntensities.slice(0, nDefault),
        exposureTimes: exposureTimes.slice(0, nDefault),
        gains: gains.slice(0, nDefault),
        syntheticChannels: syntheticChannelsPayload,
        resortPointListToSnakeCoordinates: false,
        is_snakescan: wellSelectorState.areaSelectSnakescan,
        overlapWidth: wellSelectorState.mode === "area"
          ? wellSelectorState.areaSelectOverlap
          : pv.overlapWidth,
        overlapHeight: wellSelectorState.mode === "area"
          ? wellSelectorState.areaSelectOverlap
          : pv.overlapHeight,
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

  const handleStopStitch = () => {
    const axiosInstance = createAxiosInstance();
    axiosInstance.get("/ExperimentController/stopAshlarStitching")
      .then((res) => {
        if (res.data?.stopped) {
          setAshlarRunning(false);
          setAshlarInterrupted(true);
          infoPopupRef.current?.showMessage("Ashlar stitching stopped");
        } else {
          infoPopupRef.current?.showMessage(res.data?.message ?? "No stitching process running");
        }
      })
      .catch(() => {
        infoPopupRef.current?.showMessage("Failed to stop Ashlar stitching");
      });
  };

  const handleRestartStitch = () => {
    setAshlarInterrupted(false);
    apiExperimentControllerRunAshlarStitching({
      pixelSize: experimentState.parameterValue.ashlar_pixel_size,
      maximumShift: experimentState.parameterValue.ashlar_maximum_shift,
      alignChannel: experimentState.parameterValue.ashlar_align_channel,
    })
      .then((data) => {
        if (data?.started) {
          setAshlarRunning(true);
          infoPopupRef.current?.showMessage("Ashlar stitching restarted");
        } else {
          setAshlarInterrupted(true);
          infoPopupRef.current?.showMessage(
            `Could not restart stitching: ${data?.error ?? "unknown error"}`
          );
        }
      })
      .catch(() => {
        setAshlarInterrupted(true);
        infoPopupRef.current?.showMessage("Failed to restart Ashlar stitching");
      });
  };

  const handleStop = () => {
    dispatch(experimentStatusSlice.setStatus(Status.IDLE));
    // Interrupt focus map immediately (in case we are in focus map phase)
    apiExperimentControllerInterruptFocusMap().catch(() => {});
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

        {/* Stitching control button — Stop while running, Restart after interrupted */}
        {ashlarRunning && (
          <Tooltip title="Kill the running Ashlar stitching process">
            <Button
              size="small"
              variant="outlined"
              color="warning"
              startIcon={<AutoFixHighIcon />}
              onClick={handleStopStitch}
            >
              Stop Stitching
            </Button>
          </Tooltip>
        )}
        {ashlarInterrupted && !ashlarRunning && (
          <Tooltip title="Restart Ashlar stitching from the beginning">
            <Button
              size="small"
              variant="outlined"
              color="secondary"
              startIcon={<AutoFixHighIcon />}
              onClick={handleRestartStitch}
            >
              Restart Stitching
            </Button>
          </Tooltip>
        )}

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
