import React, { useEffect, useState, useCallback, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  TextField,
  FormControl,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  InputLabel,
  Slider,
  Button,
  ButtonGroup,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Alert,
  CircularProgress,
  Tooltip,
  Divider,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DeleteIcon from "@mui/icons-material/Delete";
import RefreshIcon from "@mui/icons-material/Refresh";
import InfoIcon from "@mui/icons-material/Info";
import GridOnIcon from "@mui/icons-material/GridOn";
import PlaceIcon from "@mui/icons-material/Place";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import PendingIcon from "@mui/icons-material/Pending";
import StopIcon from "@mui/icons-material/Stop";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import AddCircleOutlineIcon from "@mui/icons-material/AddCircleOutline";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import EditIcon from "@mui/icons-material/Edit";
import SaveIcon from "@mui/icons-material/Save";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import HeightIcon from "@mui/icons-material/Height";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";
import RestoreIcon from "@mui/icons-material/Restore";
import CloseIcon from "@mui/icons-material/Close";

// State slices
import * as focusMapSlice from "../../state/slices/FocusMapSlice";
import MeasuredFocusPoints from "./MeasuredFocusPoints";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";
import * as positionSlice from "../../state/slices/PositionSlice";
import * as wellSelectorSlice from "../../state/slices/WellSelectorSlice";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice";

// Coordinate calculation
import { calculateScanCoordinates } from "../CoordinateCalculator";
import {
  computePlannedFocusPoints,
  filterRemovedPlannedPoints,
  plannedPointKey,
} from "./plannedFocusGrid";

// API
import apiExperimentControllerComputeFocusMap from "../../backendapi/apiExperimentControllerComputeFocusMap";
import apiExperimentControllerGetFocusMap from "../../backendapi/apiExperimentControllerGetFocusMap";
import apiExperimentControllerClearFocusMap from "../../backendapi/apiExperimentControllerClearFocusMap";
import apiExperimentControllerGetFocusMapPreview from "../../backendapi/apiExperimentControllerGetFocusMapPreview";
import apiExperimentControllerInterruptFocusMap from "../../backendapi/apiExperimentControllerInterruptFocusMap";
import apiExperimentControllerComputeFocusMapFromPoints from "../../backendapi/apiExperimentControllerComputeFocusMapFromPoints";
import apiExperimentControllerMeasureFocusMapFromPoints from "../../backendapi/apiExperimentControllerMeasureFocusMapFromPoints";
import apiExperimentControllerSaveFocusMaps from "../../backendapi/apiExperimentControllerSaveFocusMaps";
import apiExperimentControllerLoadFocusMaps from "../../backendapi/apiExperimentControllerLoadFocusMaps";
import apiPositionerControllerMovePositioner from "../../backendapi/apiPositionerControllerMovePositioner";
import apiPositionerControllerMovePositionerXYZ from "../../backendapi/apiPositionerControllerMovePositionerXYZ";
import apiPositionerControllerGetPositions from "../../backendapi/apiPositionerControllerGetPositions";
import apiAutofocusControllerDoAutofocusBackground, { waitForAutofocusComplete } from "../../backendapi/apiAutofocusControllerDoAutofocusBackground";


// Visualization
import FocusMapVisualization, { colorForValue } from "./FocusMapVisualization";

/**
 * Extract Z from positions API response which may be nested:
 * e.g. {VirtualStage: {X: ..., Y: ..., Z: ...}} or {Z: ...}
 * The positioner name ("VirtualStage") is dynamic.
 */
const extractZFromPositions = (positions) => {
  if (!positions) return null;
  if (positions.Z !== undefined) return positions.Z;
  const keys = Object.keys(positions);
  for (const key of keys) {
    if (positions[key] && typeof positions[key] === "object" && positions[key].Z !== undefined) {
      return positions[key].Z;
    }
  }
  return null;
};

/**
 * FocusMapDimension – Experiment Designer panel for Focus Mapping configuration.
 *
 * Provides:
 * - Enable/disable toggle
 * - Grid configuration (rows, cols, margin)
 * - Fit method selection (spline, RBF, constant)
 * - Z offset and clamping
 * - Per-illumination-channel Z offsets
 * - Interrupt button during computation
 * - Autofocus settings (shared with ZFocusDimension)
 * - Mutual-exclusion warning for AF-per-position vs Focus Map
 * - Manual focus point collection
 * - Per-group compute/clear actions
 * - Expanded fit statistics display
 * - Visualization of measured points and fitted surface
 */
// The backend reports no error number when it cannot measure one — below four
// points the residuals are taken against the very points that were fitted.
const fitErrorLabel = (stats) =>
  stats.mean_abs_error == null
    ? "not validated"
    : `MAE ${stats.mean_abs_error.toFixed(2)} µm`;

const fitErrorColor = (stats) => {
  if (stats.mean_abs_error == null) return "default";
  if (stats.mean_abs_error < 1) return "success";
  return stats.mean_abs_error < 5 ? "warning" : "error";
};

const FocusMapDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // ── Redux state ──────────────────────────────────────────────────────
  const focusMapState = useSelector(focusMapSlice.getFocusMapState);
  const { config, results, ui, manualPoints } = focusMapState;

  const experimentState = useSelector(experimentSlice.getExperimentState);
  const parameterValue = experimentState.parameterValue;
  const parameterRange = useSelector(parameterRangeSlice.getParameterRangeState);
  const positionState = useSelector(positionSlice.getPositionState);
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const showOverlayOnWellplate = useSelector(focusMapSlice.getShowOverlayOnWellplate);
  const manualPlacementActive = useSelector(focusMapSlice.getManualPlacementActive);
  const highlightedPoint = useSelector(focusMapSlice.getFocusMapHighlightedPoint);
  const plannedRemovedKeys = useSelector(focusMapSlice.getPlannedRemovedKeys);

  // Detect mutual exclusion: per-position AF enabled while Focus Map is also enabled
  const isAutoFocusPerPosition = parameterValue.autoFocus === true;

  // Local state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showAFSettings, setShowAFSettings] = useState(false);
  const [showChannelOffsets, setShowChannelOffsets] = useState(false);
  // showManualPoints and showMeasuredPoints are persisted in Redux (focusMap.ui)
  const showManualPoints = ui.showManualPoints ?? false;
  const showMeasuredPoints = ui.showMeasuredPoints ?? false;
  const showPlannedPoints = ui.showPlannedPoints ?? true;
  const setShowManualPoints = (val) => dispatch(focusMapSlice.setShowManualPoints(val));
  const setShowMeasuredPoints = (val) => dispatch(focusMapSlice.setShowMeasuredPoints(val));
  const setShowPlannedPoints = (val) => dispatch(focusMapSlice.setShowPlannedPoints(val));
  const [previewData, setPreviewData] = useState(null);
  const [expandedFitGroup, setExpandedFitGroup] = useState(null);
  const [goToInProgress, setGoToInProgress] = useState(null); // "groupId-pointIndex"

  // ── Dimension summary ────────────────────────────────────────────────
  useEffect(() => {
    let summary = "Disabled";
    if (config.enabled) {
      const groupCount = Object.keys(results).length;
      const readyCount = Object.values(results).filter((r) => r.status === "ready").length;
      if (groupCount > 0) {
        summary = `${readyCount}/${groupCount} groups mapped (${config.method})`;
      } else {
        summary = `${config.rows}×${config.cols} grid, ${config.method}`;
      }
    }
    dispatch(
      experimentUISlice.setDimensionSummary({
        dimension: "focusMap",
        summary,
      })
    );
  }, [config, results, dispatch]);

  // Fetch existing focus maps on mount
  useEffect(() => {
    if (config.enabled) {
      apiExperimentControllerGetFocusMap()
        .then((data) => {
          dispatch(focusMapSlice.setFocusMapResults(data));
        })
        .catch(() => {
          // Ignore – no maps yet
        });
    }
  }, [config.enabled, dispatch]);

  // ── Handlers ─────────────────────────────────────────────────────────

  // Build scan areas from the current experiment/wellplate state so the
  // backend knows the correct XY bounds even before an experiment is started.
  const buildScanAreas = useCallback(() => {
    try {
      const scanConfig = calculateScanCoordinates(
        experimentState,
        objectiveState,
        wellSelectorState
      );
      if (scanConfig?.scanAreas?.length > 0) {
        return scanConfig.scanAreas.map((area) => ({
          areaId: area.areaId,
          areaName: area.areaName,
          bounds: area.bounds,
        }));
      }
    } catch (err) {
      console.warn("Could not build scan areas for focus map:", err);
    }
    return null;
  }, [experimentState, objectiveState, wellSelectorState]);

  // ── Planned automatic-grid preview ──────────────────────────────────
  // The positions the automatic measurement WILL visit (per scan region),
  // recomputed live from the current selection + grid config so they can be
  // shown — and pruned — before anything moves.
  const {
    enabled: fmEnabled,
    use_manual_map: fmUseManualMap,
    rows: fmRows,
    cols: fmCols,
    add_margin: fmAddMargin,
  } = config;

  // Grid vs Points: the same flag the backend gates on, named for what it means.
  const isPointsMode = Boolean(config.use_manual_map);
  const hasManualMap = Object.keys(results).some(
    (k) => results[k]?.status === "ready" && (k === "manual" || k === "global"),
  );

  const plannedPoints = useMemo(
    () =>
      computePlannedFocusPoints(experimentState, objectiveState, wellSelectorState, {
        enabled: fmEnabled,
        use_manual_map: fmUseManualMap,
        rows: fmRows,
        cols: fmCols,
        add_margin: fmAddMargin,
      }),
    [
      experimentState,
      objectiveState,
      wellSelectorState,
      fmEnabled,
      fmUseManualMap,
      fmRows,
      fmCols,
      fmAddMargin,
    ],
  );

  const keptPlannedPoints = useMemo(
    () => filterRemovedPlannedPoints(plannedPoints, plannedRemovedKeys),
    [plannedPoints, plannedRemovedKeys],
  );
  const removedPlannedCount = plannedPoints.length - keptPlannedPoints.length;

  // Delete a planned grid point (before measurement). Only the point's key is
  // stored; the grid itself always mirrors the current selection.
  const handleRemovePlannedPoint = useCallback(
    (pt) => {
      // Indices shift after removal – drop any stale highlight
      dispatch(focusMapSlice.setFocusMapHighlightedPoint(null));
      dispatch(focusMapSlice.removePlannedPointKey(plannedPointKey(pt)));
    },
    [dispatch],
  );

  // Move the stage to a planned point (XY only – Z is unknown until measured)
  const handleGoToPlannedPoint = useCallback(async (pt, idx) => {
    const key = `planned-${idx}`;
    setGoToInProgress(key);
    try {
      await apiPositionerControllerMovePositionerXYZ({
        x: pt.x, y: pt.y, isAbsolute: true, isBlocking: false, speed: 15000,
      });
    } catch (err) {
      console.error("Failed to move to planned point:", err);
    } finally {
      setGoToInProgress(null);
    }
  }, []);

  // Compute focus map for all groups
  const handleComputeAll = useCallback(async () => {
    dispatch(focusMapSlice.setFocusMapComputing({ isComputing: true, groupId: null }));
    dispatch(focusMapSlice.clearFocusMapError());

    try {
      // Attach scan_areas so the backend uses the correct wellplate bounds
      const scanAreas = buildScanAreas();
      const configWithAreas = { ...config };
      if (scanAreas) {
        configWithAreas.scan_areas = scanAreas;
      }

      // Send the experiment's own settings, not a field-by-field copy of them:
      // the backend reads autofocus from exactly the object the Z / Focus tab
      // edits, so the two cannot drift.
      configWithAreas.autofocus = parameterValue;

      // Send the (possibly pruned) planned grid so the backend measures
      // exactly the previewed positions. Scan areas whose planned points were
      // all deleted are skipped by the backend.
      if (!config.use_manual_map && keptPlannedPoints.length > 0) {
        configWithAreas.grid_points = keptPlannedPoints.map((p) => ({
          x: p.x,
          y: p.y,
          group_id: p.groupId,
        }));
      }

      const data = await apiExperimentControllerComputeFocusMap(configWithAreas);
      dispatch(focusMapSlice.setFocusMapResults(data));
    } catch (err) {
      dispatch(focusMapSlice.setFocusMapError(err.message || "Failed to compute focus map"));
    } finally {
      dispatch(focusMapSlice.setFocusMapComputing({ isComputing: false }));
    }
  }, [config, parameterValue, buildScanAreas, keptPlannedPoints, dispatch]);

  // Interrupt ongoing computation
  const handleInterrupt = useCallback(async () => {
    try {
      await apiExperimentControllerInterruptFocusMap();
    } catch (err) {
      console.error("Failed to interrupt focus map:", err);
    }
  }, []);

  // Clear all focus maps.
  // "Clear All" must FULLY disable focus mapping, otherwise the next
  // experiment still re-sends focusMap.enabled and the backend re-measures /
  // re-applies a Z surface (the stage appears to "snap back" to old focus
  // points even after the user manually changed Z).  So in addition to
  // clearing the backend maps + cached results we:
  //   - disable the focus-map dimension (enabled=false) so it is excluded
  //     from the experiment payload, and
  //   - drop any manually added focus points.
  const handleClearAll = useCallback(async () => {
    try {
      await apiExperimentControllerClearFocusMap();
      dispatch(focusMapSlice.clearFocusMapResults());
      dispatch(focusMapSlice.clearManualPoints());
      dispatch(focusMapSlice.restorePlannedPoints());
      dispatch(focusMapSlice.setFocusMapEnabled(false));
      setPreviewData(null);
    } catch (err) {
      dispatch(focusMapSlice.setFocusMapError(err.message || "Failed to clear focus maps"));
    }
  }, [dispatch]);

  // Load preview for a specific group
  const handlePreviewGroup = useCallback(
    async (groupId) => {
      try {
        const data = await apiExperimentControllerGetFocusMapPreview(groupId, 30);
        setPreviewData({ groupId, ...data });
        dispatch(focusMapSlice.setFocusMapSelectedGroup(groupId));
      } catch (err) {
        console.error("Failed to load preview:", err);
      }
    },
    [dispatch]
  );

  // Add a manual focus point at current stage position
  const handleAddManualPoint = useCallback(() => {
    dispatch(
      focusMapSlice.addManualPoint({
        x: positionState?.x ?? 0,
        y: positionState?.y ?? 0,
        z: positionState?.z ?? 0,
      })
    );
  }, [dispatch, positionState]);

  // Move stage to a measured focus point
  const handleGoToPoint = useCallback(
    async (pt, groupId, pointIndex) => {
      const key = `${groupId}-${pointIndex}`;
      setGoToInProgress(key);
      try {
        // One coordinated XYZ move to the stored point
        await apiPositionerControllerMovePositionerXYZ({
          x: pt.x,
          y: pt.y,
          z: pt.z,
          isAbsolute: true,
          isBlocking: false,
          speed: 15000,
        });
      } catch (err) {
        console.error("Failed to move to focus point:", err);
      } finally {
        setGoToInProgress(null);
      }
    },
    []
  );

  // Run autofocus at a specific measured point's XY, then update its Z
  const handleAutofocusAtPoint = useCallback(
    async (pt, groupId, pointIndex) => {
      const key = `${groupId}-${pointIndex}`;
      setGoToInProgress(key);
      try {
        // Move stage to point XY
        await apiPositionerControllerMovePositionerXYZ({
          x: pt.x, y: pt.y, isAbsolute: true, isBlocking: false, speed: 15000,
        });

        // Run autofocus with current ExperimentSlice settings
        // autoFocus starts the AF in a background thread; we must poll until done.
        await apiAutofocusControllerDoAutofocusBackground({
          rangez: parameterValue.autoFocusRange ?? 100,
          resolutionz: parameterValue.autoFocusResolution ?? 10,
          nCropsize: parameterValue.autoFocusCropsize ?? 2048,
          focusAlgorithm: parameterValue.autoFocusAlgorithm || "LAPE",
          tSettle: parameterValue.autoFocusSettleTime ?? 0.1,
          static_offset: parameterValue.autoFocusStaticOffset ?? 0,
          twoStage: parameterValue.autoFocusTwoStage ?? false,
        });

        // Wait for autofocus to finish (polls getAutofocusStatus)
        const afStatus = await waitForAutofocusComplete(500, 120000);

        // Read resulting Z from AF status or positions API
        let newZ = afStatus?.currentZ ?? null;
        if (newZ == null) {
          const positions = await apiPositionerControllerGetPositions();
          newZ = extractZFromPositions(positions) ?? pt.z;
        }

        // Update the point's Z in Redux
        const result = results[groupId];
        if (result?.points) {
          const updatedPts = [...result.points];
          updatedPts[pointIndex] = { ...pt, z: newZ };
          dispatch(
            focusMapSlice.updateFocusMapGroupResult({
              groupId,
              result: { ...result, points: updatedPts },
            })
          );
        }
      } catch (err) {
        console.error("Autofocus at point failed:", err);
      } finally {
        setGoToInProgress(null);
      }
    },
    [parameterValue, results, dispatch]
  );

  // ── Manual-point row actions ────────────────────────────────
  // Move the stage to a manual point (XY always; Z only if it has been measured).
  const handleGoToManualPoint = useCallback(async (pt, idx) => {
    const key = `manual-${idx}`;
    setGoToInProgress(key);
    try {
      // XY blocking so the stage has arrived before Z is driven to the stored
      // focus height; Z only moves when this point has actually been measured.
      await apiPositionerControllerMovePositionerXYZ({
        x: pt.x, y: pt.y, isAbsolute: true, isBlocking: true, speed: 15000,
      });
      if (pt.z != null) {
        await apiPositionerControllerMovePositionerXYZ({
          z: pt.z, isAbsolute: true, isBlocking: false, speed: 15000,
        });
      }
    } catch (err) {
      console.error("Failed to move to manual point:", err);
    } finally {
      setGoToInProgress(null);
    }
  }, []);

  // Drive to a manual point's XY, run autofocus with the current settings, and
  // return the measured Z.
  const measureManualPointZ = useCallback(async (pt) => {
    await apiPositionerControllerMovePositionerXYZ({
      x: pt.x, y: pt.y, isAbsolute: true, isBlocking: false, speed: 15000,
    });
    await apiAutofocusControllerDoAutofocusBackground({
      rangez: parameterValue.autoFocusRange ?? 100,
      resolutionz: parameterValue.autoFocusResolution ?? 10,
      nCropsize: parameterValue.autoFocusCropsize ?? 2048,
      focusAlgorithm: parameterValue.autoFocusAlgorithm || "LAPE",
      tSettle: parameterValue.autoFocusSettleTime ?? 0.1,
      static_offset: parameterValue.autoFocusStaticOffset ?? 0,
      twoStage: parameterValue.autoFocusTwoStage ?? false,
    });
    const afStatus = await waitForAutofocusComplete(500, 120000);
    let newZ = afStatus?.currentZ ?? null;
    if (newZ == null) {
      const positions = await apiPositionerControllerGetPositions();
      newZ = extractZFromPositions(positions) ?? pt.z ?? 0;
    }
    return newZ;
  }, [parameterValue]);

  // Autofocus a single manual point and store its measured Z.
  const handleAutofocusManualPoint = useCallback(async (pt, idx) => {
    const key = `manual-${idx}`;
    setGoToInProgress(key);
    try {
      const newZ = await measureManualPointZ(pt);
      dispatch(focusMapSlice.updateManualPointZ({ index: idx, z: newZ }));
    } catch (err) {
      console.error("Autofocus at manual point failed:", err);
    } finally {
      setGoToInProgress(null);
    }
  }, [measureManualPointZ, dispatch]);

  // Measure Z (autofocus) at every manual point's XY and fit, then enable the
  // manual map. The whole move→autofocus→fit loop runs on the BACKEND as one
  // blocking call (the browser can't reliably sense per-point blocking), and a
  // map-placed point (Z=null) gets its Z measured rather than guessed.
  const handleMeasureAndFitManual = useCallback(async () => {
    if (!manualPoints || manualPoints.length === 0) return;
    dispatch(focusMapSlice.setFocusMapComputing({ isComputing: true, groupId: "manual" }));
    dispatch(focusMapSlice.clearFocusMapError());
    try {
      const cfg = {
        ...config,
        // Strip a null Z so the backend autofocus-measures it; keep a real Z.
        points: manualPoints.map((p) => ({
          x: p.x,
          y: p.y,
          ...(p.z == null ? {} : { z: p.z }),
        })),
        autofocus: parameterValue,
      };
      const data = await apiExperimentControllerMeasureFocusMapFromPoints(cfg);
      const result = data?.manual;
      if (result) {
        dispatch(focusMapSlice.updateFocusMapGroupResult({ groupId: "manual", result }));
        // Reflect the measured Z values back into the editable table.
        if (Array.isArray(result.points)) {
          result.points.forEach((rp, i) => {
            if (rp && rp.z != null && i < manualPoints.length) {
              dispatch(focusMapSlice.updateManualPointZ({ index: i, z: rp.z }));
            }
          });
        }
      }
      dispatch(focusMapSlice.setFocusMapUseManualMap(true));
    } catch (err) {
      dispatch(focusMapSlice.setFocusMapError(err.message || "Measure & fit failed"));
    } finally {
      dispatch(focusMapSlice.setFocusMapComputing({ isComputing: false }));
    }
  }, [manualPoints, config, parameterValue, dispatch]);

  // Step Z up/down by a fixed amount (5 µm) and update the point
  const STEP_Z_SIZE = 5;
  const handleStepZ = useCallback(
    async (pt, groupId, pointIndex, direction) => {
      const delta = direction === "up" ? STEP_Z_SIZE : -STEP_Z_SIZE;
      const key = `${groupId}-${pointIndex}`;
      setGoToInProgress(key);
      try {
        // Move Z relative
        await apiPositionerControllerMovePositioner({
          axis: "Z",
          dist: delta,
          isAbsolute: false,
          isBlocking: false,
        });

        // Read actual Z from positions API (nested structure)
        const positions = await apiPositionerControllerGetPositions();
        const newZ = extractZFromPositions(positions) ?? pt.z + delta;

        // Update point
        const result = results[groupId];
        if (result?.points) {
          const updatedPts = [...result.points];
          updatedPts[pointIndex] = { ...pt, z: newZ };
          dispatch(
            focusMapSlice.updateFocusMapGroupResult({
              groupId,
              result: { ...result, points: updatedPts },
            })
          );
        }
      } catch (err) {
        console.error("Step Z failed:", err);
      } finally {
        setGoToInProgress(null);
      }
    },
    [results, dispatch]
  );

  // Set a point's Z to the current stage Z position (uses Redux state which is always up-to-date)
  const handleSetCurrentZ = useCallback(
    (pt, groupId, pointIndex) => {
      const newZ = positionState.z;

      const result = results[groupId];
      if (result?.points) {
        const updatedPts = [...result.points];
        updatedPts[pointIndex] = { ...pt, z: newZ };
        dispatch(
          focusMapSlice.updateFocusMapGroupResult({
            groupId,
            result: { ...result, points: updatedPts },
          })
        );
      }
    },
    [positionState, results, dispatch]
  );

  // Set a point's XYZ to the current stage position
  const handleSetCurrentXYZ = useCallback(
    (pt, groupId, pointIndex) => {
      const result = results[groupId];
      if (result?.points) {
        const updatedPts = [...result.points];
        updatedPts[pointIndex] = {
          ...pt,
          x: positionState?.x ?? pt.x,
          y: positionState?.y ?? pt.y,
          z: positionState?.z ?? pt.z,
        };
        dispatch(
          focusMapSlice.updateFocusMapGroupResult({
            groupId,
            result: { ...result, points: updatedPts },
          })
        );
      }
    },
    [positionState, results, dispatch]
  );

  // Refit a group's focus map using its measured points (possibly with edited Z values)
  const handleRefitGroup = useCallback(
    async (groupId, points) => {
      dispatch(focusMapSlice.setFocusMapComputing({ isComputing: true, groupId }));
      dispatch(focusMapSlice.clearFocusMapError());
      try {
        const result = await apiExperimentControllerComputeFocusMapFromPoints({
          points: points.map((pt) => ({ x: pt.x, y: pt.y, z: pt.z })),
          group_id: groupId,
          group_name: results[groupId]?.group_name || groupId,
          method: config.method,
          smoothing_factor: config.smoothing_factor,
          z_offset: config.z_offset,
          clamp_enabled: config.clamp_enabled,
          z_min: config.z_min,
          z_max: config.z_max,
        });
        dispatch(focusMapSlice.updateFocusMapGroupResult({ groupId, result }));
        // Update preview data so the heatmap visualization refreshes immediately
        setPreviewData({ groupId, ...result });
      } catch (err) {
        dispatch(
          focusMapSlice.setFocusMapError(err.message || "Failed to refit focus map")
        );
      } finally {
        dispatch(focusMapSlice.setFocusMapComputing({ isComputing: false }));
      }
    },
    [config, results, dispatch]
  );

  // Delete a measured (auto-generated) calibration point, e.g. one that
  // landed outside the sample. The surface is refitted from the remaining
  // points right away so the map never reflects the deleted point. If no
  // points remain, the whole group map is cleared (backend + frontend).
  const handleDeleteMeasuredPoint = useCallback(
    async (groupId, pointIndex) => {
      const result = results[groupId];
      if (!result?.points) return;
      const remaining = result.points.filter((_, i) => i !== pointIndex);

      // Indices shift after removal – drop any stale highlight
      dispatch(focusMapSlice.setFocusMapHighlightedPoint(null));

      if (remaining.length === 0) {
        try {
          await apiExperimentControllerClearFocusMap(groupId);
        } catch (err) {
          console.error("Failed to clear focus map group:", err);
        }
        dispatch(focusMapSlice.clearFocusMapResults(groupId));
        setPreviewData((prev) => (prev?.groupId === groupId ? null : prev));
        return;
      }

      dispatch(focusMapSlice.removeMeasuredPoint({ groupId, index: pointIndex }));
      await handleRefitGroup(groupId, remaining);
    },
    [results, handleRefitGroup, dispatch]
  );

  // Hover helpers linking list rows to the wellplate / heatmap markers
  const highlightPoint = useCallback(
    (source, groupId, index) => {
      dispatch(
        focusMapSlice.setFocusMapHighlightedPoint({ source, groupId, index })
      );
    },
    [dispatch]
  );
  const clearHighlight = useCallback(() => {
    dispatch(focusMapSlice.setFocusMapHighlightedPoint(null));
  }, [dispatch]);

  // Status icon helper
  const StatusIcon = ({ status }) => {
    switch (status) {
      case "ready":
        return <CheckCircleIcon fontSize="small" color="success" />;
      case "measuring":
      case "fitting":
        return <CircularProgress size={16} />;
      case "error":
        return <ErrorIcon fontSize="small" color="error" />;
      default:
        return <PendingIcon fontSize="small" color="disabled" />;
    }
  };

  const groupEntries = Object.entries(results);
  const channelOffsetEntries = Object.entries(config.channel_offsets || {});
  const illuSources = Array.isArray(parameterRange.illuSources) ? parameterRange.illuSources : [];

  return (
    <Box>
      {/* ── Main toggle ──────────────────────────────────────────────── */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={config.enabled}
              onChange={(e) => dispatch(focusMapSlice.setFocusMapEnabled(e.target.checked))}
              color="primary"
            />
          }
          label={
            <Typography variant="subtitle1" fontWeight={500}>
              Enable Focus Mapping
            </Typography>
          }
        />
        <Tooltip title="Measure focus at a grid of positions per scan region, fit a Z surface, and automatically set Z during acquisition.">
          <InfoIcon fontSize="small" color="action" />
        </Tooltip>
      </Box>

      {!config.enabled && (
        <Typography variant="body2" color="text.secondary">
          When enabled, a Z-focus surface is measured before acquisition to automatically set the
          correct Z height at every XY position.
        </Typography>
      )}

      {config.enabled && (
        <>
          {/* ── Overlay toggle for wellplate viewer ──────────────────── */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={showOverlayOnWellplate}
                  onChange={(e) =>
                    dispatch(focusMapSlice.setShowOverlayOnWellplate(e.target.checked))
                  }
                  size="small"
                />
              }
              label={
                <Typography variant="body2">
                  {showOverlayOnWellplate ? (
                    <><VisibilityIcon fontSize="inherit" sx={{ verticalAlign: "middle", mr: 0.5 }} />Show points on wellplate</>
                  ) : (
                    <><VisibilityOffIcon fontSize="inherit" sx={{ verticalAlign: "middle", mr: 0.5 }} />Show points on wellplate</>
                  )}
                </Typography>
              }
            />
          </Box>

          {/* ── Mutual exclusion warning ──────────────────────────────── */}
          {isAutoFocusPerPosition && (
            <Alert
              severity="warning"
              icon={<WarningAmberIcon />}
              sx={{ mb: 2 }}
            >
              <strong>Autofocus per position</strong> is also enabled in the Z/Focus tab.
              Using both simultaneously is redundant – the focus map already provides Z correction
              at every position. Consider disabling per-position autofocus for faster acquisition.
            </Alert>
          )}

          {/* ── Mode ─────────────────────────────────────────────────────
              Grid and Points are two different jobs with no shared controls.
              They used to be interleaved in one scroll behind a switch buried
              in the toggles below, so half the panel was always inert and it
              was not obvious which half. */}
          <ToggleButtonGroup
            exclusive
            fullWidth
            size="small"
            value={isPointsMode ? "points" : "grid"}
            onChange={(e, mode) =>
              mode && dispatch(focusMapSlice.setFocusMapUseManualMap(mode === "points"))
            }
            sx={{ mb: 2 }}
          >
            <ToggleButton value="grid">
              <GridOnIcon fontSize="small" sx={{ mr: 1 }} />
              Grid — measure {config.rows}×{config.cols} per region
            </ToggleButton>
            <ToggleButton value="points">
              <PlaceIcon fontSize="small" sx={{ mr: 1 }} />
              Points — measure where I put them
            </ToggleButton>
          </ToggleButtonGroup>

          {isPointsMode && (
            <Alert severity="info" variant="outlined" sx={{ mb: 2 }} icon={<InfoIcon />}>
              One surface is fitted from your points and interpolated onto every
              region.{" "}
              {hasManualMap ? (
                <Chip label="Map fitted ✓" size="small" color="success" sx={{ ml: 0.5 }} />
              ) : (
                <Chip label="No map yet" size="small" color="warning" sx={{ ml: 0.5 }} />
              )}
            </Alert>
          )}

          {/* ── Points mode: the points the operator places ───────────── */}
          {isPointsMode && (<>
          <Accordion
            // Open while empty: in Points mode this holds the only way to add points.
            expanded={showManualPoints || (manualPoints || []).length === 0}
            onChange={() => setShowManualPoints(!showManualPoints)}
            variant="outlined"
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2">
                Manual Focus Points
                {(manualPoints || []).length > 0 && (
                  <Chip
                    label={`${manualPoints.length} point(s)`}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                )}
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: "block" }}>
                Pick where the focus is measured. "Place on Map" and then click on the
                plate map (any drawing mode, also with a freehand region; Shift+click works
                too): XY comes from the click, Z is measured by autofocus in "Measure Z &amp; Fit".
                "Add Current Position" takes the stage XYZ as it is.
              </Typography>

              {/* Manual points table */}
              {(manualPoints || []).length > 0 && (
                <TableContainer component={Paper} variant="outlined" sx={{ mb: 1 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>#</TableCell>
                        <TableCell>X (µm)</TableCell>
                        <TableCell>Y (µm)</TableCell>
                        <TableCell>Z (µm)</TableCell>
                        <TableCell align="right"></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {manualPoints.map((pt, idx) => (
                        <TableRow
                          key={idx}
                          hover
                          selected={
                            highlightedPoint?.source === "manual" &&
                            highlightedPoint?.index === idx
                          }
                          onMouseEnter={() => highlightPoint("manual", null, idx)}
                          onMouseLeave={clearHighlight}
                        >
                          <TableCell>{idx + 1}</TableCell>
                          <TableCell>
                            <TextField
                              type="number"
                              size="small"
                              value={pt.x}
                              onChange={(e) => {
                                const newPts = [...manualPoints];
                                newPts[idx] = { ...pt, x: parseFloat(e.target.value) || 0 };
                                // Dispatch individual update
                                dispatch(focusMapSlice.clearManualPoints());
                                newPts.forEach((p) => dispatch(focusMapSlice.addManualPoint(p)));
                              }}
                              variant="standard"
                              sx={{ width: 80 }}
                            />
                          </TableCell>
                          <TableCell>
                            <TextField
                              type="number"
                              size="small"
                              value={pt.y}
                              onChange={(e) => {
                                const newPts = [...manualPoints];
                                newPts[idx] = { ...pt, y: parseFloat(e.target.value) || 0 };
                                dispatch(focusMapSlice.clearManualPoints());
                                newPts.forEach((p) => dispatch(focusMapSlice.addManualPoint(p)));
                              }}
                              variant="standard"
                              sx={{ width: 80 }}
                            />
                          </TableCell>
                          <TableCell>
                            <TextField
                              type="number"
                              size="small"
                              value={pt.z ?? ""}
                              placeholder="auto"
                              onChange={(e) =>
                                dispatch(
                                  focusMapSlice.updateManualPointZ({
                                    index: idx,
                                    z: e.target.value === "" ? null : parseFloat(e.target.value),
                                  })
                                )
                              }
                              variant="standard"
                              sx={{ width: 80 }}
                            />
                          </TableCell>
                          <TableCell align="right">
                            <Tooltip title="Go to this position">
                              <span>
                                <IconButton
                                  size="small"
                                  disabled={goToInProgress === `manual-${idx}`}
                                  onClick={() => handleGoToManualPoint(pt, idx)}
                                >
                                  <MyLocationIcon fontSize="small" />
                                </IconButton>
                              </span>
                            </Tooltip>
                            <Tooltip title="Use current stage Z">
                              <span>
                                <IconButton
                                  size="small"
                                  onClick={() =>
                                    dispatch(
                                      focusMapSlice.updateManualPointZ({
                                        index: idx,
                                        z: positionState?.z ?? 0,
                                      })
                                    )
                                  }
                                >
                                  <HeightIcon fontSize="small" />
                                </IconButton>
                              </span>
                            </Tooltip>
                            <Tooltip title="Autofocus here (measure Z)">
                              <span>
                                <IconButton
                                  size="small"
                                  disabled={goToInProgress === `manual-${idx}`}
                                  onClick={() => handleAutofocusManualPoint(pt, idx)}
                                >
                                  <CenterFocusStrongIcon fontSize="small" />
                                </IconButton>
                              </span>
                            </Tooltip>
                            <IconButton
                              size="small"
                              onClick={() => dispatch(focusMapSlice.removeManualPoint(idx))}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}

              {manualPlacementActive && (
                <Alert severity="info" sx={{ mb: 1 }}>
                  Click on the wellplate / sample map to drop focus points. Only
                  XY is recorded — Z is measured by autofocus when you press
                  "Measure Z &amp; Fit". Click "Stop Placing" when done.
                </Alert>
              )}

              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                <Button
                  size="small"
                  variant={manualPlacementActive ? "contained" : "outlined"}
                  color={manualPlacementActive ? "secondary" : "primary"}
                  startIcon={<MyLocationIcon />}
                  onClick={() =>
                    dispatch(
                      focusMapSlice.setManualPlacementActive(
                        !manualPlacementActive,
                      ),
                    )
                  }
                >
                  {manualPlacementActive ? "Stop Placing" : "Place on Map"}
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<AddCircleOutlineIcon />}
                  onClick={handleAddManualPoint}
                >
                  Add Current Position
                </Button>
                {(manualPoints || []).length >= 3 && (
                  <Button
                    size="small"
                    variant="contained"
                    color="secondary"
                    startIcon={
                      ui.isComputing ? (
                        <CircularProgress size={16} color="inherit" />
                      ) : (
                        <CenterFocusStrongIcon />
                      )
                    }
                    disabled={ui.isComputing}
                    title="Drive to each point, autofocus to measure Z, then fit and enable the manual map"
                    onClick={handleMeasureAndFitManual}
                  >
                    Measure Z &amp; Fit
                  </Button>
                )}
                {(manualPoints || []).length >= 3 &&
                  !(manualPoints || []).some((p) => p.z == null) && (
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<PlayArrowIcon />}
                    disabled={ui.isComputing}
                    title="Fit using the Z values already in the table (no autofocus)"
                    onClick={async () => {
                      dispatch(focusMapSlice.setFocusMapComputing({ isComputing: true, groupId: "manual" }));
                      dispatch(focusMapSlice.clearFocusMapError());
                      try {
                        const result = await apiExperimentControllerComputeFocusMapFromPoints({
                          points: manualPoints,
                          group_id: "manual",
                          group_name: "Manual Points",
                          method: config.method,
                          smoothing_factor: config.smoothing_factor,
                          z_offset: config.z_offset,
                          clamp_enabled: config.clamp_enabled,
                          z_min: config.z_min,
                          z_max: config.z_max,
                        });
                        dispatch(focusMapSlice.updateFocusMapGroupResult({ groupId: "manual", result }));
                        // Auto-enable "use manual map" so the fitted plane is actually used
                        dispatch(focusMapSlice.setFocusMapUseManualMap(true));
                      } catch (err) {
                        dispatch(focusMapSlice.setFocusMapError(err.message || "Failed to fit from manual points"));
                      } finally {
                        dispatch(focusMapSlice.setFocusMapComputing({ isComputing: false }));
                      }
                    }}
                  >
                    Fit from Points
                  </Button>
                )}
                {(manualPoints || []).length > 0 && (
                  <Button
                    size="small"
                    variant="outlined"
                    color="warning"
                    startIcon={<DeleteIcon />}
                    onClick={() => dispatch(focusMapSlice.clearManualPoints())}
                  >
                    Clear Points
                  </Button>
                )}
              </Box>
            </AccordionDetails>
          </Accordion>
          </>)}

          {/* ── Grid configuration ───────────────────────────────────── */}
          {!isPointsMode && (
          <>
          <Box sx={{ display: "flex", gap: 2, mb: 2, flexWrap: "wrap" }}>
            <TextField
              label="Grid Rows"
              type="number"
              size="small"
              value={config.rows}
              onChange={(e) => dispatch(focusMapSlice.setFocusMapRows(parseInt(e.target.value) || 1))}
              inputProps={{ min: 1, max: 20 }}
              sx={{ width: 100 }}
            />
            <TextField
              label="Grid Cols"
              type="number"
              size="small"
              value={config.cols}
              onChange={(e) => dispatch(focusMapSlice.setFocusMapCols(parseInt(e.target.value) || 1))}
              inputProps={{ min: 1, max: 20 }}
              sx={{ width: 100 }}
            />
          </Box>

          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 2 }}>
            {config.rows * config.cols < 3
              ? "At least 3 points are needed for a tilted plane, and they must not lie on one line — below that the map is a flat Z."
              : config.rows * config.cols < 4
                ? "3 points give a tilted plane; 4 or more are needed for a curved surface."
                : `${config.rows * config.cols} points per region — enough for a ${config.method} surface.`}
          </Typography>

          {/* Fit mode toggles */}
          <Box sx={{ display: "flex", gap: 2, mb: 2, alignItems: "center", flexWrap: "wrap" }}>
            <FormControlLabel
              control={
                <Switch
                  checked={config.fit_by_region}
                  onChange={(e) =>
                    dispatch(focusMapSlice.setFocusMapFitByRegion(e.target.checked))
                  }
                  size="small"
                />
              }
              label="Fit per region"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={config.add_margin}
                  onChange={(e) =>
                    dispatch(focusMapSlice.setFocusMapAddMargin(e.target.checked))
                  }
                  size="small"
                />
              }
              label="Add margin"
            />
          </Box>

          {/* ── Planned Grid Points (automatic-mode preview) ─────────── */}
          {(
            <Accordion
              expanded={showPlannedPoints}
              onChange={() => setShowPlannedPoints(!showPlannedPoints)}
              variant="outlined"
              sx={{ mb: 2 }}
            >
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="body2">
                  Planned Grid Points
                  <Chip
                    label={`${keptPlannedPoints.length} point(s)`}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                  {removedPlannedCount > 0 && (
                    <Chip
                      label={`${removedPlannedCount} removed`}
                      size="small"
                      color="warning"
                      variant="outlined"
                      sx={{ ml: 0.5 }}
                    />
                  )}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ mb: 1, display: "block" }}
                >
                  Positions "Automatic Measure and Fit" will visit (one{" "}
                  {config.rows}×{config.cols} grid per scan region), shown as
                  crosses on the plate map. Remove points that should not be
                  measured (e.g. outside the sample) before starting the scan;
                  a region with all points removed is skipped.
                </Typography>

                {plannedPoints.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    No scan areas selected yet — select positions on the plate
                    map first.
                  </Typography>
                ) : (
                  <>
                    <TableContainer
                      component={Paper}
                      variant="outlined"
                      sx={{ mb: 1, maxHeight: 300 }}
                    >
                      <Table size="small" stickyHeader>
                        <TableHead>
                          <TableRow>
                            <TableCell>#</TableCell>
                            <TableCell>Region</TableCell>
                            <TableCell>X (µm)</TableCell>
                            <TableCell>Y (µm)</TableCell>
                            <TableCell align="right"></TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {keptPlannedPoints.map((pt, idx) => (
                            <TableRow
                              key={plannedPointKey(pt)}
                              hover
                              selected={
                                highlightedPoint?.source === "planned" &&
                                highlightedPoint?.index === idx
                              }
                              onMouseEnter={() =>
                                highlightPoint("planned", pt.groupId, idx)
                              }
                              onMouseLeave={clearHighlight}
                            >
                              <TableCell>{idx + 1}</TableCell>
                              <TableCell>
                                <Typography variant="caption">
                                  {pt.groupName}
                                </Typography>
                              </TableCell>
                              <TableCell>{pt.x.toFixed(1)}</TableCell>
                              <TableCell>{pt.y.toFixed(1)}</TableCell>
                              <TableCell align="right">
                                <Tooltip title="Move stage to this XY">
                                  <span>
                                    <IconButton
                                      size="small"
                                      disabled={
                                        goToInProgress === `planned-${idx}`
                                      }
                                      onClick={() =>
                                        handleGoToPlannedPoint(pt, idx)
                                      }
                                    >
                                      <MyLocationIcon fontSize="small" />
                                    </IconButton>
                                  </span>
                                </Tooltip>
                                <Tooltip title="Remove this point from the automatic scan">
                                  <IconButton
                                    size="small"
                                    color="error"
                                    onClick={() => handleRemovePlannedPoint(pt)}
                                  >
                                    <CloseIcon fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>

                    {removedPlannedCount > 0 && (
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={<RestoreIcon />}
                        onClick={() =>
                          dispatch(focusMapSlice.restorePlannedPoints())
                        }
                      >
                        Restore All ({removedPlannedCount})
                      </Button>
                    )}
                  </>
                )}
              </AccordionDetails>
            </Accordion>
          )}
          </>
          )}

          {/* Autofocus is configured once, on the Z / Focus tab. This panel
              rendered the same controls a second time, and until now the
              focus-map phase used its own FocusMapConfig.af_* copy instead of
              the values shown here — so the panel edited settings it ignored. */}
          <Alert severity="info" variant="outlined" sx={{ mb: 2 }}>
            Focus mapping uses the autofocus configured on the{" "}
            <strong>Z / Focus</strong> tab — range, step, algorithm and channel
            are set there.
          </Alert>

          {/* ── Per-channel Z offsets ─────────────────────────────────── */}
          <Accordion
            expanded={showChannelOffsets}
            onChange={() => setShowChannelOffsets(!showChannelOffsets)}
            variant="outlined"
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2">
                Per-Channel Z Offsets
                {channelOffsetEntries.length > 0 && (
                  <Chip
                    label={`${channelOffsetEntries.length} channel(s)`}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                )}
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: "block" }}>
                Add a Z offset per illumination channel to compensate for chromatic focal shift.
                The offset is added on top of the interpolated Z from the focus map.
              </Typography>
              {illuSources.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No illumination sources detected. Start the experiment setup first.
                </Typography>
              ) : (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  {illuSources.map((src) => {
                    const offset = config.channel_offsets?.[src];
                    const hasOffset = offset !== undefined;
                    return (
                      <Box
                        key={src}
                        sx={{ display: "flex", alignItems: "center", gap: 1 }}
                      >
                        <Typography variant="body2" sx={{ minWidth: 120 }}>
                          {src}
                        </Typography>
                        <TextField
                          type="number"
                          size="small"
                          value={hasOffset ? offset : ""}
                          placeholder="0"
                          onChange={(e) => {
                            const val = e.target.value;
                            if (val === "" || val === undefined) {
                              dispatch(focusMapSlice.removeChannelOffset(src));
                            } else {
                              dispatch(
                                focusMapSlice.setFocusMapChannelOffset({
                                  channel: src,
                                  offset: parseFloat(val) || 0,
                                })
                              );
                            }
                          }}
                          inputProps={{ step: 0.5 }}
                          sx={{ width: 100 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          µm
                        </Typography>
                        {hasOffset && (
                          <IconButton
                            size="small"
                            onClick={() => dispatch(focusMapSlice.removeChannelOffset(src))}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        )}
                      </Box>
                    );
                  })}
                </Box>
              )}
            </AccordionDetails>
          </Accordion>

          {/* ── Advanced settings ─────────────────────────────────────── */}
          <Accordion
            expanded={showAdvanced}
            onChange={() => setShowAdvanced(!showAdvanced)}
            variant="outlined"
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2">Advanced Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
                  <FormControl size="small" sx={{ minWidth: 120 }}>
                    <InputLabel>Fit Method</InputLabel>
                    <Select
                      value={config.method}
                      label="Fit Method"
                      onChange={(e) =>
                        dispatch(focusMapSlice.setFocusMapMethod(e.target.value))
                      }
                    >
                      <MenuItem value="spline">Spline</MenuItem>
                      <MenuItem value="rbf">RBF</MenuItem>
                      <MenuItem value="constant">Constant</MenuItem>
                    </Select>
                  </FormControl>
                  <TextField
                    label="Smoothing"
                    type="number"
                    size="small"
                    value={config.smoothing_factor}
                    onChange={(e) =>
                      dispatch(
                        focusMapSlice.setFocusMapSmoothingFactor(parseFloat(e.target.value) || 0)
                      )
                    }
                    inputProps={{ step: 0.01, min: 0 }}
                    sx={{ width: 110 }}
                  />
                  <TextField
                    label="Z Offset (µm)"
                    type="number"
                    size="small"
                    value={config.z_offset}
                    onChange={(e) =>
                      dispatch(focusMapSlice.setFocusMapZOffset(parseFloat(e.target.value) || 0))
                    }
                    inputProps={{ step: 0.5 }}
                    sx={{ width: 120 }}
                  />
                  <TextField
                    label="Settle (ms)"
                    type="number"
                    size="small"
                    value={config.settle_ms}
                    onChange={(e) =>
                      dispatch(focusMapSlice.setFocusMapSettleMs(parseInt(e.target.value) || 0))
                    }
                    inputProps={{ min: 0 }}
                    sx={{ width: 110 }}
                  />
                </Box>

                {/* Z Clamping */}
                <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.clamp_enabled}
                        onChange={(e) =>
                          dispatch(focusMapSlice.setFocusMapClampEnabled(e.target.checked))
                        }
                        size="small"
                      />
                    }
                    label="Clamp Z"
                  />
                  {config.clamp_enabled && (
                    <>
                      <TextField
                        label="Z Min"
                        type="number"
                        size="small"
                        value={config.z_min}
                        onChange={(e) =>
                          dispatch(focusMapSlice.setFocusMapZMin(parseFloat(e.target.value) || 0))
                        }
                        sx={{ width: 100 }}
                      />
                      <TextField
                        label="Z Max"
                        type="number"
                        size="small"
                        value={config.z_max}
                        onChange={(e) =>
                          dispatch(focusMapSlice.setFocusMapZMax(parseFloat(e.target.value) || 0))
                        }
                        sx={{ width: 100 }}
                      />
                    </>
                  )}
                </Box>

                <FormControlLabel
                  control={
                    <Switch
                      checked={config.apply_during_scan}
                      onChange={(e) =>
                        dispatch(focusMapSlice.setFocusMapApplyDuringScan(e.target.checked))
                      }
                      size="small"
                    />
                  }
                  label="Apply Z correction during scan"
                />
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* ── Action Buttons ────────────────────────────────────────── */}
          <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
            <Button
              variant="contained"
              size="small"
              startIcon={
                ui.isComputing ? <CircularProgress size={16} color="inherit" /> : <PlayArrowIcon />
              }
              onClick={handleComputeAll}
              disabled={ui.isComputing || config.use_manual_map}
              title={config.use_manual_map ? "Disabled in manual map mode — use 'Fit from Points' instead" : ""}
            >
              {ui.isComputing ? "Measuring..." : "Automatic Measure and Fit"} {/* TODO: We should only be able to click this button if we have not already entered items for the manual scan*/}
            </Button>

            {/* Interrupt button – only visible during computation */}
            {ui.isComputing && (
              <Button
                variant="contained"
                size="small"
                color="error"
                startIcon={<StopIcon />}
                onClick={handleInterrupt}
              >
                Interrupt
              </Button>
            )}

            <Button
              variant="outlined"
              size="small"
              startIcon={<DeleteIcon />}
              onClick={handleClearAll}
              disabled={ui.isComputing}
              color="warning"
            >
              Clear All
            </Button>
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshIcon />}
              onClick={() => {
                apiExperimentControllerGetFocusMap()
                  .then((data) => dispatch(focusMapSlice.setFocusMapResults(data)))
                  .catch(() => {});
              }}
            >
              Refresh
            </Button>

            {/* Save / Load focus maps to/from disk */}
            <Tooltip title="Save all focus maps to disk (~/ ImSwitch/focus_maps). Allows reuse across sessions." arrow>
              <Button
                variant="outlined"
                size="small"
                startIcon={<SaveIcon />}
                onClick={async () => {
                  try {
                    const res = await apiExperimentControllerSaveFocusMaps();
                    dispatch(focusMapSlice.clearFocusMapError());
                    alert(`Saved ${res.count} focus map(s) to ${res.path}`);
                  } catch (err) {
                    dispatch(focusMapSlice.setFocusMapError(err.message || "Failed to save focus maps"));
                  }
                }}
                disabled={ui.isComputing || Object.keys(results).length === 0}
              >
                Save Maps
              </Button>
            </Tooltip>

            <Tooltip title="Load previously saved focus maps from disk (~/ ImSwitch/focus_maps)." arrow>
              <Button
                variant="outlined"
                size="small"
                startIcon={<FolderOpenIcon />}
                onClick={async () => {
                  try {
                    const res = await apiExperimentControllerLoadFocusMaps();
                    if (res.maps) {
                      dispatch(focusMapSlice.setFocusMapResults(res.maps));
                    }
                    dispatch(focusMapSlice.clearFocusMapError());
                    alert(`Loaded ${res.loaded_count} focus map(s) from ${res.path}`);
                  } catch (err) {
                    dispatch(focusMapSlice.setFocusMapError(err.message || "Failed to load focus maps"));
                  }
                }}
                disabled={ui.isComputing}
              >
                Load Maps
              </Button>
            </Tooltip>
          </Box>

          {/* ── Error display ─────────────────────────────────────────── */}
          {ui.error && (
            <Alert severity="error" onClose={() => dispatch(focusMapSlice.clearFocusMapError())} sx={{ mb: 2 }}>
              {ui.error}
            </Alert>
          )}

          {/* ── Group results list with expanded fit statistics ────────── */}
          {groupEntries.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Focus Map Results
              </Typography>
              {groupEntries.map(([groupId, result]) => (
                <Box key={groupId} sx={{ mb: 1 }}>
                  {/* Group summary row */}
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      p: 1,
                      borderRadius: 1,
                      backgroundColor:
                        ui.selectedGroupId === groupId
                          ? alpha(theme.palette.primary.main, 0.08)
                          : "transparent",
                      border: `1px solid ${theme.palette.divider}`,
                      cursor: "pointer",
                      "&:hover": {
                        backgroundColor: alpha(theme.palette.primary.main, 0.04),
                      },
                    }}
                    onClick={() => {
                      handlePreviewGroup(groupId);
                      setExpandedFitGroup(expandedFitGroup === groupId ? null : groupId);
                    }}
                  >
                    <StatusIcon status={result.status} />
                    <Typography variant="body2" sx={{ flex: 1 }}>
                      {result.group_name || groupId}
                    </Typography>
                    {result.fit_stats && result.status === "ready" && (
                      <>
                        <Chip
                          label={`${result.fit_stats.method} · n=${result.fit_stats.n_points}`}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          label={fitErrorLabel(result.fit_stats)}
                          size="small"
                          color={fitErrorColor(result.fit_stats)}
                          variant="outlined"
                        />
                      </>
                    )}
                    <ExpandMoreIcon
                      fontSize="small"
                      sx={{
                        transform: expandedFitGroup === groupId ? "rotate(180deg)" : "none",
                        transition: "transform 0.2s",
                      }}
                    />
                  </Box>

                  {result.fit_stats?.fallback_reason && (
                    <Typography
                      variant="caption"
                      color="warning.main"
                      sx={{ display: "block", ml: 4, mb: 0.5 }}
                    >
                      {result.fit_stats.fallback_reason}
                    </Typography>
                  )}

                  {/* Expanded fit statistics panel */}
                  {expandedFitGroup === groupId && result.fit_stats && (
                    <Box
                      sx={{
                        ml: 2,
                        mt: 0.5,
                        p: 1.5,
                        borderLeft: `3px solid ${theme.palette.primary.main}`,
                        backgroundColor: alpha(theme.palette.background.default, 0.5),
                        borderRadius: "0 4px 4px 0",
                      }}
                    >
                      <Typography variant="caption" fontWeight={600} sx={{ mb: 1, display: "block" }}>
                        Fit Statistics
                      </Typography>
                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 0.5 }}>
                        <Typography variant="caption" color="text.secondary">
                          Method:
                        </Typography>
                        <Typography variant="caption">
                          {result.fit_stats.method}
                          {result.fit_stats.fallback_used && " (fallback)"}
                        </Typography>

                        <Typography variant="caption" color="text.secondary">
                          Points:
                        </Typography>
                        <Typography variant="caption">{result.fit_stats.n_points}</Typography>

                        <Typography variant="caption" color="text.secondary">
                          MAE:
                        </Typography>
                        <Typography variant="caption">
                          {result.fit_stats.mean_abs_error == null
                            ? "not enough points to validate"
                            : `${result.fit_stats.mean_abs_error.toFixed(4)} µm (${
                                result.fit_stats.error_is_cross_validated
                                  ? "cross-validated"
                                  : "in-sample"
                              })`}
                        </Typography>

                        {result.fit_stats.r_squared != null && (
                          <>
                            <Typography variant="caption" color="text.secondary">
                              R²:
                            </Typography>
                            <Typography variant="caption">
                              {result.fit_stats.r_squared?.toFixed(4) ?? "N/A"}
                            </Typography>
                          </>
                        )}

                        {result.fit_stats.max_abs_error !== undefined && (
                          <>
                            <Typography variant="caption" color="text.secondary">
                              Max Error:
                            </Typography>
                            <Typography variant="caption">
                              {result.fit_stats.max_abs_error?.toFixed(4) ?? "N/A"} µm
                            </Typography>
                          </>
                        )}

                        {result.fit_stats.bounds_z && (
                          <>
                            <Typography variant="caption" color="text.secondary">
                              Z Range:
                            </Typography>
                            <Typography variant="caption">
                              [{result.fit_stats.bounds_z[0]?.toFixed(1)},
                              {" "}{result.fit_stats.bounds_z[1]?.toFixed(1)}] µm
                            </Typography>
                          </>
                        )}

                        {result.fit_stats.residuals && result.fit_stats.residuals.length > 0 && (
                          <>
                            <Typography variant="caption" color="text.secondary">
                              Residuals (std):
                            </Typography>
                            <Typography variant="caption">
                              {Math.sqrt(
                                result.fit_stats.residuals.reduce((s, r) => s + r * r, 0) /
                                  result.fit_stats.residuals.length
                              ).toFixed(4)}{" "}
                              µm
                            </Typography>
                          </>
                        )}

                        {result.fit_stats.fallback_reason && (
                          <>
                            <Typography variant="caption" color="text.secondary">
                              Fallback Reason:
                            </Typography>
                            <Typography variant="caption" color="warning.main">
                              {result.fit_stats.fallback_reason}
                            </Typography>
                          </>
                        )}
                      </Box>
                    </Box>
                  )}
                </Box>
              ))}
            </Box>
          )}

          {/* ── Measured Points (own component) ─────────────────────── */}
          <MeasuredFocusPoints
            groupEntries={groupEntries}
            ui={ui}
            showMeasuredPoints={showMeasuredPoints}
            setShowMeasuredPoints={setShowMeasuredPoints}
            highlightedPoint={highlightedPoint}
            dispatch={dispatch}
            handleGoToPoint={handleGoToPoint}
            handleDeleteMeasuredPoint={handleDeleteMeasuredPoint}
            handleAutofocusAtPoint={handleAutofocusAtPoint}
            handleRefitGroup={handleRefitGroup}
            handleSetCurrentZ={handleSetCurrentZ}
            handleSetCurrentXYZ={handleSetCurrentXYZ}
            handleStepZ={handleStepZ}
            goToInProgress={goToInProgress}
            highlightPoint={highlightPoint}
            clearHighlight={clearHighlight}
          />

          {/* ── Visualization ─────────────────────────────────────────── */}
          {previewData && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Focus Map Preview – {previewData.groupId}
              </Typography>
              <FocusMapVisualization
                data={previewData}
                highlightIndex={
                  highlightedPoint?.source === "measured" &&
                  highlightedPoint?.groupId === previewData.groupId
                    ? highlightedPoint.index
                    : null
                }
                onClickPosition={async (worldX, worldY) => {
                  // Move stage to clicked position on the heatmap, including
                  // interpolated Z from the preview grid.
                  try {
                    // Interpolate Z from the preview grid (bilinear)
                    let targetZ = null;
                    const grid = previewData?.preview_grid || (
                      previewData?.x && previewData?.y && previewData?.z
                        ? { x: previewData.x, y: previewData.y, z: previewData.z }
                        : null
                    );
                    if (grid && grid.x && grid.y && grid.z) {
                      const xs = grid.x;
                      const ys = grid.y;
                      const zs = grid.z;
                      // Find bounding grid cell indices
                      let ix0 = 0, ix1 = xs.length - 1;
                      for (let i = 0; i < xs.length - 1; i++) {
                        if (worldX >= xs[i] && worldX <= xs[i + 1]) { ix0 = i; ix1 = i + 1; break; }
                      }
                      let iy0 = 0, iy1 = ys.length - 1;
                      for (let i = 0; i < ys.length - 1; i++) {
                        if (worldY >= ys[i] && worldY <= ys[i + 1]) { iy0 = i; iy1 = i + 1; break; }
                      }
                      // Bilinear interpolation fractions
                      const tx = xs[ix1] !== xs[ix0] ? (worldX - xs[ix0]) / (xs[ix1] - xs[ix0]) : 0;
                      const ty = ys[iy1] !== ys[iy0] ? (worldY - ys[iy0]) / (ys[iy1] - ys[iy0]) : 0;
                      const z00 = zs[iy0]?.[ix0] ?? 0;
                      const z10 = zs[iy0]?.[ix1] ?? z00;
                      const z01 = zs[iy1]?.[ix0] ?? z00;
                      const z11 = zs[iy1]?.[ix1] ?? z00;
                      targetZ = z00 * (1 - tx) * (1 - ty)
                              + z10 * tx * (1 - ty)
                              + z01 * (1 - tx) * ty
                              + z11 * tx * ty;
                    }

                    // Single move to the clicked XY, carrying Z along when the
                    // surface could be interpolated there.
                    await apiPositionerControllerMovePositionerXYZ({
                      x: worldX,
                      y: worldY,
                      ...(targetZ != null && isFinite(targetZ)
                        ? { z: targetZ }
                        : {}),
                      isAbsolute: true,
                      isBlocking: false,
                      speed: 15000,
                    });
                  } catch (err) {
                    console.error("Failed to move stage to heatmap position:", err);
                  }
                }}
              />
            </Box>
          )}
        </>
      )}
    </Box>
  );
};

export default FocusMapDimension;
