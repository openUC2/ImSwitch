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
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Home as HomeIcon,
} from "@mui/icons-material";

import * as experimentSlice from "../state/slices/ExperimentSlice.js";
import * as experimentStatusSlice from "../state/slices/ExperimentStatusSlice.js";
import * as experimentStateSlice from "../state/slices/ExperimentStateSlice.js";
import * as wellSelectorSlice from "../state/slices/WellSelectorSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as coordinateCalculator from "../axon/CoordinateCalculator.js";

import apiExperimentControllerStartWellplateExperiment from "../backendapi/apiExperimentControllerStartWellplateExperiment.js";
import apiExperimentControllerStopExperiment from "../backendapi/apiExperimentControllerStopExperiment.js";
import apiExperimentControllerHomeAllAxes from "../backendapi/apiExperimentControllerHomeAllAxes.js";
import fetchGetExperimentStatus from "../middleware/fetchExperimentControllerGetExperimentStatus.js";

import WellSelectorCanvas from "../axon/WellSelectorCanvas.js";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";
import GenericTabBar from "../axon/GenericTabBar.js";
import PictureInPicture, { PiPToggleButton } from "../axon/PictureInPicture.js";
import InfoPopup from "../axon/InfoPopup.js";

// Status enum matching ExperimentComponent
const Status = Object.freeze({
  IDLE: "idle",
  RUNNING: "running",
  PAUSED: "paused",
  STOPPING: "stopping",
});

// Hardcoded ShitScope scan area dimensions (micrometers)
const SHITSCOPE_SCAN_WIDTH = 3000; // 30 mm
const SHITSCOPE_SCAN_HEIGHT = 1000; // 10 mm

/**
 * ShitScope - Dedicated single-button scan application
 *
 * Simplified scan interface with:
 * - Fixed rectangular scan area (30x10 mm)
 * - Live view with overview canvas showing current position
 * - Pre-experiment homing of all axes
 * - Single start button to launch paving scan
 */
const ShitScopeComponent = () => {
  const dispatch = useDispatch();
  const infoPopupRef = useRef(null);
  const canvasRef = useRef(null);

  // PiP live preview
  const [pipVisible, setPipVisible] = useState(false);

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
            x: SHITSCOPE_SCAN_WIDTH / 2,
            y: SHITSCOPE_SCAN_HEIGHT / 2,
            width: SHITSCOPE_SCAN_WIDTH,
            height: SHITSCOPE_SCAN_HEIGHT,
            row: 0,
            col: 0,
          },
        ],
      })
    );

    // Set overlap from wellSelector area mode
    dispatch(wellSelectorSlice.setMode("area"));
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

  // ── Start experiment with homing pre-hook ──────────────────────────────
  const handleStart = useCallback(async () => {
    try {
      // Step 1: Home all axes
      setIsHoming(true);
      if (infoPopupRef.current) {
        infoPopupRef.current.showMessage("Homing all axes...");
      }

      await apiExperimentControllerHomeAllAxes();
      setIsHoming(false);

      if (infoPopupRef.current) {
        infoPopupRef.current.showMessage("Homing complete. Starting scan...");
      }

      // Step 2: Sync state
      dispatch(
        experimentSlice.setIsSnakescan(wellSelectorState.areaSelectSnakescan)
      );
      dispatch(
        experimentSlice.setOverlapWidth(wellSelectorState.areaSelectOverlap)
      );
      dispatch(
        experimentSlice.setOverlapHeight(wellSelectorState.areaSelectOverlap)
      );

      // Step 3: Calculate scan coordinates
      const scanConfig = coordinateCalculator.calculateScanCoordinates(
        experimentState,
        objectiveState,
        wellSelectorState
      );

      console.log(
        `[ShitScope] Scan: ${scanConfig.scanAreas.length} areas, ${scanConfig.metadata.totalPositions} positions`
      );

      // Step 4: Filter illumination intensities
      const channelEnabled =
        experimentState.parameterValue.channelEnabledForExperiment || [];
      const rawIntensities =
        experimentState.parameterValue.illuIntensities || [];
      const filteredIntensities = rawIntensities.map((val, idx) =>
        channelEnabled[idx] === true ? val : 0
      );

      // Step 5: Build experiment request
      const experimentRequest = {
        name: experimentState.name || "ShitScope_Scan",
        parameterValue: {
          ...experimentState.parameterValue,
          illuIntensities: filteredIntensities,
          resortPointListToSnakeCoordinates: false,
          is_snakescan: wellSelectorState.areaSelectSnakescan,
          overlapWidth: wellSelectorState.areaSelectOverlap,
          overlapHeight: wellSelectorState.areaSelectOverlap,
        },
        scanAreas: scanConfig.scanAreas,
        scanMetadata: scanConfig.metadata,
        pointList: coordinateCalculator.convertToBackendFormat(
          scanConfig,
          experimentState
        ).pointList,
      };

      // Step 6: Send to backend
      await apiExperimentControllerStartWellplateExperiment(experimentRequest);
      dispatch(experimentStatusSlice.setStatus(Status.RUNNING));

      if (infoPopupRef.current) {
        infoPopupRef.current.showMessage("ShitScope scan started!");
      }
    } catch (err) {
      setIsHoming(false);
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
          justifyContent: "space-between",
          mb: 1,
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: "bold" }}>
          ShitScope
        </Typography>
        <PiPToggleButton
          active={pipVisible}
          onClick={() => setPipVisible((v) => !v)}
        />
      </Box>

      {/* PiP overlay */}
      <PictureInPicture
        visible={pipVisible}
        onClose={() => setPipVisible(false)}
      />

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

      {/* Main layout: canvas + live view */}
      <Box sx={{ display: "flex", gap: 2 }}>
        {/* Left panel: Well selector canvas (overview) */}
        <Box sx={{ flex: 3 }}>
          <GenericTabBar
            id="shitscope-left"
            tabNames={["Overview", "Live View"]}
          >
            <WellSelectorCanvas ref={canvasRef} />
            <LiveViewControlWrapper />
          </GenericTabBar>
        </Box>

        {/* Right panel: Controls */}
        <Box sx={{ flex: 1, minWidth: 240 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Scan Control
            </Typography>

            {/* Status */}
            <Alert
              severity={
                isRunning
                  ? "info"
                  : isHoming
                    ? "warning"
                    : "success"
              }
              sx={{ mb: 2 }}
            >
              {isHoming
                ? "Homing axes..."
                : isRunning
                  ? "Scan running"
                  : "Ready"}
            </Alert>

            {/* Start / Stop button */}
            <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
              <Button
                variant="contained"
                color="primary"
                size="large"
                fullWidth
                startIcon={
                  isHoming ? (
                    <CircularProgress size={20} color="inherit" />
                  ) : (
                    <PlayArrowIcon />
                  )
                }
                onClick={handleStart}
                disabled={!isIdle || isHoming}
              >
                {isHoming ? "Homing..." : "Start Scan"}
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
          </Paper>
        </Box>
      </Box>

      <InfoPopup ref={infoPopupRef} />
    </Box>
  );
};

export default ShitScopeComponent;
