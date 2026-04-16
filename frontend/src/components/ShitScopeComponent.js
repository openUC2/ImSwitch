import React, { useState, useEffect, useRef, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Button,
  Box,
  Typography,
  LinearProgress,
  Paper,
  Chip,
  Alert,
  CircularProgress,
  Slider,
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Home as HomeIcon,
  FolderOpen as FolderOpenIcon,
} from "@mui/icons-material";

import * as experimentSlice from "../state/slices/ExperimentSlice.js";
import * as experimentStatusSlice from "../state/slices/ExperimentStatusSlice.js";
import * as experimentStateSlice from "../state/slices/ExperimentStateSlice.js";
import * as wellSelectorSlice from "../state/slices/WellSelectorSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import * as coordinateCalculator from "../axon/CoordinateCalculator.js";

import apiExperimentControllerStartWellplateExperiment from "../backendapi/apiExperimentControllerStartWellplateExperiment.js";
import apiExperimentControllerStopExperiment from "../backendapi/apiExperimentControllerStopExperiment.js";
import apiExperimentControllerHomeAllAxes from "../backendapi/apiExperimentControllerHomeAllAxes.js";
import fetchGetExperimentStatus from "../middleware/fetchExperimentControllerGetExperimentStatus.js";

import WellSelectorCanvas from "../axon/WellSelectorCanvas.js";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";
import InfoPopup from "../axon/InfoPopup.js";

// Status enum matching ExperimentComponent
const Status = Object.freeze({
  IDLE: "idle",
  RUNNING: "running",
  PAUSED: "paused",
  STOPPING: "stopping",
});

// Hardcoded ShitScope scan area dimensions (micrometers)
const SHITSCOPE_SCAN_WIDTH = 15000; // 15 mm
const SHITSCOPE_SCAN_HEIGHT = 7000; // 7 mm

/**
 * ShitScope - Dedicated single-button scan application
 *
 * Simplified scan interface with:
 * - Fixed rectangular scan area (15x7 mm)
 * - Live view with overview canvas showing current position
 * - Pre-experiment homing of all axes
 * - Single start button to launch paving scan
 */
const ShitScopeComponent = ({ onOpenFileManager }) => {
  const dispatch = useDispatch();
  const infoPopupRef = useRef(null);
  const canvasRef = useRef(null);

  // Homing state
  const [isHoming, setIsHoming] = useState(false);

  // Workflow step tracking
  const [cachedStepId, setCachedStepId] = useState(0);
  const [cachedTotalSteps, setCachedTotalSteps] = useState(undefined);
  const [cachedStepName, setCachedStepName] = useState("");

  // Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const experimentWorkflowState = useSelector(
    experimentStateSlice.getExperimentState
  );
  const experimentStatusState = useSelector(
    experimentStatusSlice.getExperimentStatusState
  );
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const positionState = useSelector(positionSlice.getPositionState);

  // Initialize the ShitScope layout on mount
  useEffect(() => {
    // Set the well layout to the shitscope single-area rectangle
    dispatch(
      experimentSlice.setWellLayout({
        name: "ShitScope",
        unit: "um",
        width: SHITSCOPE_SCAN_WIDTH * 1.2, // Canvas padding
        height: SHITSCOPE_SCAN_HEIGHT * 1.2,
        wells: [
          {
            id: "A1",
            name: "Scan Area",
            shape: "rectangle",
            x: SHITSCOPE_SCAN_WIDTH * 1.2 / 2,
            y: SHITSCOPE_SCAN_HEIGHT * 1.2 / 2,
            width: SHITSCOPE_SCAN_WIDTH,
            height: SHITSCOPE_SCAN_HEIGHT,
            row: 0,
            col: 0,
          },
        ],
      })
    );

    // Set mode to MOVE_CAMERA so canvas clicks move the stage instead of adding points
    dispatch(wellSelectorSlice.setMode("camera"));
    dispatch(wellSelectorSlice.setAreaSelectSnakescan(true));

    // Create a single point covering the entire scan area
    dispatch(experimentSlice.setPointList([]));
    dispatch(
      experimentSlice.createPoint({
        x: SHITSCOPE_SCAN_WIDTH / 2,
        y: SHITSCOPE_SCAN_HEIGHT / 2,
        z: 0,
        name: "ShitScope Scan",
        shape: "rectangle",
        rectPlusX: SHITSCOPE_SCAN_WIDTH / 2,
        rectPlusY: SHITSCOPE_SCAN_HEIGHT / 2,
        rectMinusX: SHITSCOPE_SCAN_WIDTH / 2,
        rectMinusY: SHITSCOPE_SCAN_HEIGHT / 2,
      })
    );
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Periodic experiment status polling
  useEffect(() => {
    fetchGetExperimentStatus(dispatch);
    const intervalId = setInterval(() => {
      fetchGetExperimentStatus(dispatch);
    }, 3000);
    return () => clearInterval(intervalId);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Cache workflow step updates
  useEffect(() => {
    if (experimentWorkflowState.totalSteps !== undefined) {
      setCachedTotalSteps(experimentWorkflowState.totalSteps);
    }
    if (experimentWorkflowState.stepId !== undefined) {
      setCachedStepId(experimentWorkflowState.stepId);
    }
    if (experimentWorkflowState.stepName !== undefined) {
      setCachedStepName(experimentWorkflowState.stepName);
    }
  }, [
    experimentWorkflowState.totalSteps,
    experimentWorkflowState.stepId,
    experimentWorkflowState.stepName,
  ]);

  // Progress calculation
  const progress =
    cachedTotalSteps && cachedTotalSteps > 0
      ? Math.floor((cachedStepId / cachedTotalSteps) * 100)
      : 0;

  const isRunning = experimentStatusState.status === Status.RUNNING;
  const isIdle = experimentStatusState.status === Status.IDLE;

  // ── Start experiment ───────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    try {
      // Step 1: Sync state – raster scan (never snake)
      dispatch(experimentSlice.setIsSnakescan(true  ));
      dispatch(
        experimentSlice.setOverlapWidth(wellSelectorState.areaSelectOverlap)
      );
      dispatch(
        experimentSlice.setOverlapHeight(wellSelectorState.areaSelectOverlap)
      );

      // Step 2: "We are here" – use current stage position as scan center so
      // the scan starts from where the stage currently is without homing.
      const weAreHerePoint = experimentState.pointList[0]
        ? {
            ...experimentState.pointList[0],
            x: positionState.x,
            y: positionState.y,
          }
        : null;
      if (weAreHerePoint) {
        // Also update canvas visualisation
        dispatch(
          experimentSlice.replacePoint({ index: 0, newPoint: weAreHerePoint })
        );
      }

      // Step 3: Calculate scan coordinates using current stage position
      const scanExperimentState = weAreHerePoint
        ? { ...experimentState, pointList: [weAreHerePoint] }
        : experimentState;

      const scanConfig = coordinateCalculator.calculateScanCoordinates(
        scanExperimentState,
        objectiveState,
        wellSelectorState
      );

      console.log(
        `[ShitScope] Scan: ${scanConfig.scanAreas.length} areas, ${scanConfig.metadata.totalPositions} positions`
      );

      // Step 4: Filter illumination intensities
      const channelEnabled =
        scanExperimentState.parameterValue.channelEnabledForExperiment || [];
      const rawIntensities =
        scanExperimentState.parameterValue.illuIntensities || [];
      const filteredIntensities = rawIntensities.map((val, idx) =>
        channelEnabled[idx] === true ? val : 0
      );

      // Step 5: Build experiment request
      const experimentRequest = {
        name: scanExperimentState.name || "ShitScope_Scan",
        parameterValue: {
          ...scanExperimentState.parameterValue,
          illuIntensities: filteredIntensities,
          resortPointListToSnakeCoordinates: false,
          is_snakescan: true, // always raster
          overlapWidth: wellSelectorState.areaSelectOverlap,
          overlapHeight: wellSelectorState.areaSelectOverlap,
        },
        scanAreas: scanConfig.scanAreas,
        scanMetadata: scanConfig.metadata,
        pointList: coordinateCalculator.convertToBackendFormat(
          scanConfig,
          scanExperimentState
        ).pointList,
      };

      // Step 6: Send to backend
      await apiExperimentControllerStartWellplateExperiment(experimentRequest);
      dispatch(experimentStatusSlice.setStatus(Status.RUNNING));

      if (infoPopupRef.current) {
        infoPopupRef.current.showMessage("ShitScope scan started!");
      }
    } catch (err) {
      console.error("[ShitScope] Start failed:", err);
      if (infoPopupRef.current) {
        infoPopupRef.current.showMessage(
          "Failed to start scan: " + (err.message || "Unknown error")
        );
      }
    }
  }, [
    dispatch,
    experimentState,
    objectiveState,
    positionState,
    wellSelectorState,
  ]);

  // ── Stop experiment ────────────────────────────────────────────────────
  const handleStop = useCallback(() => {
    apiExperimentControllerStopExperiment()
      .then(() => {
        dispatch(experimentStatusSlice.setStatus(Status.IDLE));
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Scan stopped.");
        }
      })
      .catch((err) => {
        console.error("[ShitScope] Stop failed:", err);
      });
  }, [dispatch]);

  // FOV info for display
  const fovX = objectiveState.fovX || 0;
  const fovY = objectiveState.fovY || 0;
  const pixelSize = objectiveState.pixelsize || 0;
  const tilesX = fovX > 0 ? Math.ceil(SHITSCOPE_SCAN_WIDTH / fovX) : "?";
  const tilesY = fovY > 0 ? Math.ceil(SHITSCOPE_SCAN_HEIGHT / fovY) : "?";
  const totalTiles =
    typeof tilesX === "number" && typeof tilesY === "number"
      ? tilesX * tilesY
      : "?";

  return (
    <Box sx={{ width: "100%", p: 1 }}>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          mb: 1,
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: "bold" }}>
          ShitScope
        </Typography>
      </Box>

      {/* Scan info chips */}
      <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
        <Chip
          label={`Scan Area: ${SHITSCOPE_SCAN_WIDTH / 1000} × ${SHITSCOPE_SCAN_HEIGHT / 1000} mm`}
          variant="outlined"
          size="small"
        />
        <Chip
          label={`FOV: ${(fovX / 1000).toFixed(1)} × ${(fovY / 1000).toFixed(1)} mm`}
          variant="outlined"
          size="small"
          color="info"
        />
        <Chip
          label={`Pixel Size: ${pixelSize.toFixed(2)} µm`}
          variant="outlined"
          size="small"
          color="info"
        />
        <Chip
          label={`Tiles: ${tilesX} × ${tilesY} = ${totalTiles}`}
          variant="outlined"
          size="small"
          color="secondary"
        />
      </Box>

      {/* Main layout: canvas + live view + controls */}
      <Box sx={{ display: "flex", gap: 2 }}>
        {/* Left panel: Canvas overview always visible, live view always below */}
        <Box sx={{ flex: 3, display: "flex", flexDirection: "row", gap: 1 }}>
          <Box sx={{ flex: 1, minHeight: 220 }}>
            <Typography variant="caption" color="text.secondary">
              Overview – click to move stage
            </Typography>
            <WellSelectorCanvas ref={canvasRef} />
          </Box>
          <Box sx={{ flex: 1, minHeight: 220 }}>
            <Typography variant="caption" color="text.secondary">
              Live View
            </Typography>
            <LiveViewControlWrapper />
          </Box>
        </Box>

        {/* Right panel: Controls */}
        <Box sx={{ flex: 1, minWidth: 240 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Scan Control
            </Typography>

            {/* Status */}
            <Alert severity={isRunning ? "info" : "success"} sx={{ mb: 2 }}>
              {isRunning ? "Scan running" : "Ready"}
            </Alert>

            {/* Start / Stop button */}
            <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
              <Button
                variant="contained"
                color="primary"
                size="large"
                fullWidth
                startIcon={<PlayArrowIcon />}
                onClick={handleStart}
                disabled={!isIdle}
              >
                Start Scan
              </Button>

              <Button
                variant="contained"
                color="error"
                size="large"
                startIcon={<StopIcon />}
                onClick={handleStop}
                disabled={!isRunning}
              >
                Stop
              </Button>
            </Box>

            {/* Home button (manual) */}
            <Button
              variant="outlined"
              fullWidth
              startIcon={
                isHoming ? (
                  <CircularProgress size={16} color="inherit" />
                ) : (
                  <HomeIcon />
                )
              }
              onClick={async () => {
                setIsHoming(true);
                try {
                  await apiExperimentControllerHomeAllAxes();
                  if (infoPopupRef.current) {
                    infoPopupRef.current.showMessage("Homing complete.");
                  }
                } catch (err) {
                  if (infoPopupRef.current) {
                    infoPopupRef.current.showMessage("Homing failed.");
                  }
                } finally {
                  setIsHoming(false);
                }
              }}
              disabled={isRunning || isHoming}
              sx={{ mb: 2 }}
            >
              Home All Axes
            </Button>

            {/* Overlap slider */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                Tile Overlap:{" "}
                <strong>
                  {Math.round((wellSelectorState.areaSelectOverlap || 0) * 100)}%
                </strong>
              </Typography>
              <Slider
                min={-50}
                max={50}
                step={1}
                value={Math.round((wellSelectorState.areaSelectOverlap || 0) * 100)}
                onChange={(_, v) => {
                  const pct = v / 100;
                  dispatch(wellSelectorSlice.setAreaSelectOverlap(pct));
                  dispatch(experimentSlice.setOverlapWidth(pct));
                  dispatch(experimentSlice.setOverlapHeight(pct));
                }}
                valueLabelDisplay="auto"
                valueLabelFormat={(v) => `${v}%`}
                disabled={isRunning || isHoming}
                size="small"
              />
            </Box>

            {/* Progress */}
            {isRunning && cachedTotalSteps > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" noWrap title={cachedStepName}>
                  {cachedStepName}
                </Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Box sx={{ flex: 1 }}>
                    <LinearProgress variant="determinate" value={progress} />
                  </Box>
                  <Typography variant="body2">{progress}%</Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  Step {cachedStepId} / {cachedTotalSteps}
                </Typography>
              </Box>
            )}

            {/* Open latest scan in file manager */}
            {onOpenFileManager && (
              <Button
                variant="outlined"
                fullWidth
                startIcon={<FolderOpenIcon />}
                onClick={() => onOpenFileManager("/ExperimentController")}
                sx={{ mt: 1 }}
              >
                Open Scans Folder
              </Button>
            )}
          </Paper>
        </Box>
      </Box>

      <InfoPopup ref={infoPopupRef} />
    </Box>
  );
};

export default ShitScopeComponent;
