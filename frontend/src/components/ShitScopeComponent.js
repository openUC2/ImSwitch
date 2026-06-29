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
  Tabs,
  Tab,
  TextField,
  Divider,
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
 * Build an ordered list of (x,y) stage positions for a snake-pattern tile scan.
 * The origin is the current stage position; tiles are placed at stepSizeX / stepSizeY intervals.
 */
function buildTileEditorSnakePositions(baseX, baseY, numTilesX, numTilesY, stepSizeX, stepSizeY) {
  const positions = [];
  for (let iY = 0; iY < numTilesY; iY++) {
    const row = [];
    for (let iX = 0; iX < numTilesX; iX++) {
      row.push({
        x: baseX + iX * stepSizeX,
        y: baseY + iY * stepSizeY,
        z: 0,
        iX,
        iY,
      });
    }
    // Reverse every other row for snake pattern
    if (iY % 2 === 1) row.reverse();
    positions.push(...row);
  }
  return positions;
}

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

  // Tab: 0 = Overview, 1 = Tile Editor
  const [activeTab, setActiveTab] = useState(0);

  // Tile editor parameters
  const [tilesX, setTilesX] = useState(3);
  const [tilesY, setTilesY] = useState(3);
  const [stepSizeX, setStepSizeX] = useState(0);
  const [stepSizeY, setStepSizeY] = useState(0);

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

  // FOV info (needed by handleStart and the tile editor UI)
  const fovX = objectiveState.fovX || 0;
  const fovY = objectiveState.fovY || 0;
  const pixelSize = objectiveState.pixelsize || 0;

  // Keep stepSizeX/Y in sync with FOV as the suggested default (only while
  // the user has not manually overridden them, i.e. while they equal 0).
  useEffect(() => {
    if (fovX > 0 && stepSizeX === 0) setStepSizeX(fovX);
  }, [fovX]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (fovY > 0 && stepSizeY === 0) setStepSizeY(fovY);
  }, [fovY]); // eslint-disable-line react-hooks/exhaustive-deps

  const tilesX_computed = fovX > 0 ? Math.ceil(SHITSCOPE_SCAN_WIDTH / fovX) : "?";
  const tilesY_computed = fovY > 0 ? Math.ceil(SHITSCOPE_SCAN_HEIGHT / fovY) : "?";
  const totalTiles_computed =
    typeof tilesX_computed === "number" && typeof tilesY_computed === "number"
      ? tilesX_computed * tilesY_computed
      : "?";

  // ── Start experiment ───────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    try {
      // ── TILE EDITOR MODE ─────────────────────────────────────────────
      if (activeTab === 1) {
        const numX = Math.max(1, Math.round(tilesX));
        const numY = Math.max(1, Math.round(tilesY));
        const sX = stepSizeX > 0 ? stepSizeX : fovX;
        const sY = stepSizeY > 0 ? stepSizeY : fovY;
        if (sX <= 0 || sY <= 0) throw new Error('Tile Editor requires a positive step size (set Step X/Y or ensure FOV is available).');
        const snakePositions = buildTileEditorSnakePositions(
          positionState.x,
          positionState.y,
          numX,
          numY,
          sX,
          sY,
        );

        const scanArea = {
          areaId: "tile_editor_scan",
          areaName: "Tile Editor Scan",
          areaType: "tile_scan",
          wellId: null,
          centerPosition: {
            x: positionState.x + ((numX - 1) * sX) / 2,
            y: positionState.y + ((numY - 1) * sY) / 2,
            z: 0,
          },
          bounds: {
            minX: positionState.x,
            maxX: positionState.x + (numX - 1) * sX,
            minY: positionState.y,
            maxY: positionState.y + (numY - 1) * sY,
            width: (numX - 1) * sX,
            height: (numY - 1) * sY,
          },
          scanPattern: "snake",
          positions: snakePositions.map((pos, idx) => ({ ...pos, index: idx })),
        };

        const pointList = [
          {
            id: "tile_editor_scan",
            name: "Tile Editor Scan",
            x: scanArea.centerPosition.x,
            y: scanArea.centerPosition.y,
            z: 0,
            iX: 0,
            iY: 0,
            wellId: null,
            areaType: "tile_scan",
            neighborPointList: snakePositions.map((pos) => ({
              x: pos.x,
              y: pos.y,
              z: 0,
              iX: pos.iX,
              iY: pos.iY,
            })),
          },
        ];

        const channelEnabled =
          experimentState.parameterValue.channelEnabledForExperiment || [];
        const rawIntensities =
          experimentState.parameterValue.illuIntensities || [];
        const filteredIntensities = rawIntensities.map((val, idx) =>
          channelEnabled[idx] === true ? val : 0
        );

        const experimentRequest = {
          name: experimentState.name || "ShitScope_TileEditor",
          parameterValue: {
            ...experimentState.parameterValue,
            illuIntensities: filteredIntensities,
            resortPointListToSnakeCoordinates: false,
            is_snakescan: true,
            overlapWidth: 0,
            overlapHeight: 0,
          },
          scanAreas: [scanArea],
          scanMetadata: {
            totalPositions: snakePositions.length,
            fovX: fovX,
            fovY: fovY,
            overlapWidth: 0,
            overlapHeight: 0,
            scanPattern: "snake",
          },
          pointList,
        };

        console.log(
          `[ShitScope TileEditor] ${numX}×${numY} = ${snakePositions.length} tiles, step ${sX}×${sY} µm`
        );

        await apiExperimentControllerStartWellplateExperiment(experimentRequest);
        dispatch(experimentStatusSlice.setStatus(Status.RUNNING));

        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage(
            `Tile scan started: ${numX}×${numY} = ${snakePositions.length} tiles`
          );
        }
        return;
      }

      // ── OVERVIEW MODE (existing flow) ────────────────────────────────
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
    activeTab,
    tilesX,
    tilesY,
    stepSizeX,
    stepSizeY,
    fovX,
    fovY,
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
          label={`Tiles (overview): ${tilesX_computed} × ${tilesY_computed} = ${totalTiles_computed}`}
          variant="outlined"
          size="small"
          color="secondary"
        />
      </Box>

      {/* Main layout: canvas + live view + controls */}
      <Box sx={{ display: "flex", gap: 2 }}>
        {/* Left panel: tabbed (Overview / Tile Editor) + live view */}
        <Box sx={{ flex: 3, display: "flex", flexDirection: "row", gap: 1 }}>
          <Box sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
            {/* Tab bar */}
            <Tabs
              value={activeTab}
              onChange={(_, v) => setActiveTab(v)}
              sx={{ mb: 1, minHeight: 36 }}
              variant="fullWidth"
            >
              <Tab label="Overview – click to move" sx={{ minHeight: 36, py: 0.5 }} />
              <Tab label="Tile Editor" sx={{ minHeight: 36, py: 0.5 }} />
            </Tabs>

            {/* Tab 0: Overview canvas */}
            {activeTab === 0 && (
              <Box sx={{ minHeight: 220 }}>
                <WellSelectorCanvas ref={canvasRef} />
              </Box>
            )}

            {/* Tab 1: Tile Editor */}
            {activeTab === 1 && (
              <Box sx={{ p: 1 }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Set number of tiles and step size. The scan starts from the
                  current stage position and runs in a snake pattern.
                </Typography>

                <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
                  <TextField
                    label="Tiles X"
                    type="number"
                    size="small"
                    value={tilesX}
                    onChange={(e) => setTilesX(Math.max(1, parseInt(e.target.value) || 1))}
                    inputProps={{ min: 1, step: 1 }}
                    sx={{ flex: 1 }}
                    disabled={isRunning}
                  />
                  <TextField
                    label="Tiles Y"
                    type="number"
                    size="small"
                    value={tilesY}
                    onChange={(e) => setTilesY(Math.max(1, parseInt(e.target.value) || 1))}
                    inputProps={{ min: 1, step: 1 }}
                    sx={{ flex: 1 }}
                    disabled={isRunning}
                  />
                </Box>

                <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
                  <TextField
                    label="Step X (µm)"
                    type="number"
                    size="small"
                    value={stepSizeX}
                    onChange={(e) => setStepSizeX(parseFloat(e.target.value) || 0)}
                    helperText={fovX > 0 ? `Suggested: ${fovX.toFixed(1)} µm (FOV)` : ""}
                    inputProps={{ min: 1, step: 1 }}
                    sx={{ flex: 1 }}
                    disabled={isRunning}
                  />
                  <TextField
                    label="Step Y (µm)"
                    type="number"
                    size="small"
                    value={stepSizeY}
                    onChange={(e) => setStepSizeY(parseFloat(e.target.value) || 0)}
                    helperText={fovY > 0 ? `Suggested: ${fovY.toFixed(1)} µm (FOV)` : ""}
                    inputProps={{ min: 1, step: 1 }}
                    sx={{ flex: 1 }}
                    disabled={isRunning}
                  />
                </Box>

                <Divider sx={{ my: 1 }} />

                {/* Computed summary */}
                <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                  <Chip
                    label={`Total tiles: ${tilesX * tilesY}`}
                    size="small"
                    color="primary"
                  />
                  <Chip
                    label={`Scan area: ${((tilesX * (stepSizeX || fovX)) / 1000).toFixed(2)} × ${((tilesY * (stepSizeY || fovY)) / 1000).toFixed(2)} mm`}
                    size="small"
                    color="secondary"
                  />
                  <Chip
                    label={`Base: (${positionState.x?.toFixed(0) ?? "?"}, ${positionState.y?.toFixed(0) ?? "?"}) µm`}
                    size="small"
                    variant="outlined"
                  />
                </Box>
              </Box>
            )}
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

            {/* Progress – always visible */}
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                <Typography variant="body2" color="text.secondary" noWrap title={cachedStepName}>
                  {isRunning ? cachedStepName || "Running…" : "Idle"}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {cachedStepId} / {cachedTotalSteps ?? "–"}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={progress}
                color={isRunning ? "primary" : "inherit"}
                sx={{ height: 8, borderRadius: 1 }}
              />
              <Typography variant="caption" color="text.secondary">
                {isRunning
                  ? `${progress}% complete`
                  : progress > 0
                  ? `Last scan: ${progress}% (${cachedStepId}/${cachedTotalSteps})`
                  : "No scan data yet"}
              </Typography>
            </Box>

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
