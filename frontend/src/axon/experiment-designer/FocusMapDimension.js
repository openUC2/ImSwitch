import React, { useEffect, useState, useCallback } from "react";
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
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DeleteIcon from "@mui/icons-material/Delete";
import RefreshIcon from "@mui/icons-material/Refresh";
import InfoIcon from "@mui/icons-material/Info";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import PendingIcon from "@mui/icons-material/Pending";

// State slices
import * as focusMapSlice from "../../state/slices/FocusMapSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";

// API
import apiExperimentControllerComputeFocusMap from "../../backendapi/apiExperimentControllerComputeFocusMap";
import apiExperimentControllerGetFocusMap from "../../backendapi/apiExperimentControllerGetFocusMap";
import apiExperimentControllerClearFocusMap from "../../backendapi/apiExperimentControllerClearFocusMap";
import apiExperimentControllerGetFocusMapPreview from "../../backendapi/apiExperimentControllerGetFocusMapPreview";

// Visualization
import FocusMapVisualization from "./FocusMapVisualization";

/**
 * FocusMapDimension – Experiment Designer panel for Focus Mapping configuration.
 *
 * Provides:
 * - Enable/disable toggle
 * - Grid configuration (rows, cols, margin)
 * - Fit method selection (spline, RBF, constant)
 * - Z offset and clamping
 * - Per-group compute/clear actions
 * - Visualization of measured points and fitted surface
 */
const FocusMapDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // Redux state
  const focusMapState = useSelector(focusMapSlice.getFocusMapState);
  const { config, results, ui } = focusMapState;

  // Local state for advanced settings visibility
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [previewData, setPreviewData] = useState(null);

  // Update dimension summary when config changes
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

  // Compute focus map for all groups
  const handleComputeAll = useCallback(async () => {
    dispatch(focusMapSlice.setFocusMapComputing({ isComputing: true, groupId: null }));
    dispatch(focusMapSlice.clearFocusMapError());

    try {
      const data = await apiExperimentControllerComputeFocusMap(config);
      dispatch(focusMapSlice.setFocusMapResults(data));
    } catch (err) {
      dispatch(focusMapSlice.setFocusMapError(err.message || "Failed to compute focus map"));
    } finally {
      dispatch(focusMapSlice.setFocusMapComputing({ isComputing: false }));
    }
  }, [config, dispatch]);

  // Clear all focus maps
  const handleClearAll = useCallback(async () => {
    try {
      await apiExperimentControllerClearFocusMap();
      dispatch(focusMapSlice.clearFocusMapResults());
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

  return (
    <Box>
      {/* Main toggle */}
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
          {/* Grid configuration */}
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
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Fit Method</InputLabel>
              <Select
                value={config.method}
                label="Fit Method"
                onChange={(e) => dispatch(focusMapSlice.setFocusMapMethod(e.target.value))}
              >
                <MenuItem value="spline">Spline</MenuItem>
                <MenuItem value="rbf">RBF</MenuItem>
                <MenuItem value="constant">Constant</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Fit mode */}
          <Box sx={{ display: "flex", gap: 2, mb: 2, alignItems: "center" }}>
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

          {/* Advanced settings */}
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

          {/* Action Buttons */}
          <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
            <Button
              variant="contained"
              size="small"
              startIcon={
                ui.isComputing ? <CircularProgress size={16} color="inherit" /> : <PlayArrowIcon />
              }
              onClick={handleComputeAll}
              disabled={ui.isComputing}
            >
              {ui.isComputing ? "Computing..." : "Compute All"}
            </Button>
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
          </Box>

          {/* Error display */}
          {ui.error && (
            <Alert severity="error" onClose={() => dispatch(focusMapSlice.clearFocusMapError())} sx={{ mb: 2 }}>
              {ui.error}
            </Alert>
          )}

          {/* Group results list */}
          {groupEntries.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Focus Map Results
              </Typography>
              {groupEntries.map(([groupId, result]) => (
                <Box
                  key={groupId}
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                    p: 1,
                    mb: 0.5,
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
                  onClick={() => handlePreviewGroup(groupId)}
                >
                  <StatusIcon status={result.status} />
                  <Typography variant="body2" sx={{ flex: 1 }}>
                    {result.group_name || groupId}
                  </Typography>
                  {result.fit_stats && result.status === "ready" && (
                    <>
                      <Chip
                        label={`${result.fit_stats.method}`}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        label={`MAE: ${result.fit_stats.mean_abs_error?.toFixed(3) ?? "?"}`}
                        size="small"
                        color={
                          result.fit_stats.mean_abs_error < 1
                            ? "success"
                            : result.fit_stats.mean_abs_error < 5
                            ? "warning"
                            : "error"
                        }
                        variant="outlined"
                      />
                      <Chip
                        label={`n=${result.fit_stats.n_points}`}
                        size="small"
                        variant="outlined"
                      />
                    </>
                  )}
                  {result.fit_stats?.fallback_used && (
                    <Tooltip title={result.fit_stats.fallback_reason}>
                      <Chip label="Fallback" size="small" color="warning" />
                    </Tooltip>
                  )}
                </Box>
              ))}
            </Box>
          )}

          {/* Visualization */}
          {previewData && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Focus Map Preview – {previewData.groupId}
              </Typography>
              <FocusMapVisualization data={previewData} />
            </Box>
          )}
        </>
      )}
    </Box>
  );
};

export default FocusMapDimension;
