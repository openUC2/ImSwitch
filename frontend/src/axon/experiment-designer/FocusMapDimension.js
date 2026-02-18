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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
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
import StopIcon from "@mui/icons-material/Stop";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import AddCircleOutlineIcon from "@mui/icons-material/AddCircleOutline";

// State slices
import * as focusMapSlice from "../../state/slices/FocusMapSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";

// API
import apiExperimentControllerComputeFocusMap from "../../backendapi/apiExperimentControllerComputeFocusMap";
import apiExperimentControllerGetFocusMap from "../../backendapi/apiExperimentControllerGetFocusMap";
import apiExperimentControllerClearFocusMap from "../../backendapi/apiExperimentControllerClearFocusMap";
import apiExperimentControllerGetFocusMapPreview from "../../backendapi/apiExperimentControllerGetFocusMapPreview";
import apiExperimentControllerInterruptFocusMap from "../../backendapi/apiExperimentControllerInterruptFocusMap";
import apiExperimentControllerComputeFocusMapFromPoints from "../../backendapi/apiExperimentControllerComputeFocusMapFromPoints";

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
 * - Per-illumination-channel Z offsets
 * - Interrupt button during computation
 * - Autofocus settings (shared with ZFocusDimension)
 * - Mutual-exclusion warning for AF-per-position vs Focus Map
 * - Manual focus point collection
 * - Per-group compute/clear actions
 * - Expanded fit statistics display
 * - Visualization of measured points and fitted surface
 */
const FocusMapDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // ── Redux state ──────────────────────────────────────────────────────
  const focusMapState = useSelector(focusMapSlice.getFocusMapState);
  const { config, results, ui, manualPoints } = focusMapState;

  const experimentState = useSelector(experimentSlice.getExperimentState);
  const parameterValue = experimentState.parameterValue;
  const parameterRange = useSelector(parameterRangeSlice.getParameterRangeState);

  // Detect mutual exclusion: per-position AF enabled while Focus Map is also enabled
  const isAutoFocusPerPosition = parameterValue.autoFocus === true;

  // Local state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showAFSettings, setShowAFSettings] = useState(false);
  const [showChannelOffsets, setShowChannelOffsets] = useState(false);
  const [showManualPoints, setShowManualPoints] = useState(false);
  const [previewData, setPreviewData] = useState(null);
  const [expandedFitGroup, setExpandedFitGroup] = useState(null);

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

  // Interrupt ongoing computation
  const handleInterrupt = useCallback(async () => {
    try {
      await apiExperimentControllerInterruptFocusMap();
    } catch (err) {
      console.error("Failed to interrupt focus map:", err);
    }
  }, []);

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

  // Add a manual focus point at current stage position
  const handleAddManualPoint = useCallback(() => {
    // Placeholder: user fills in coordinates manually or from current position
    dispatch(focusMapSlice.addManualPoint({ x: 0, y: 0, z: 0 }));
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
  const illuSources = parameterRange.illuSources || [];

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

          {/* ── Grid configuration ───────────────────────────────────── */}
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

          {/* Fit mode toggles */}
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

          {/* ── Autofocus Settings (shared with Z/Focus tab) ─────────── */}
          <Accordion
            expanded={showAFSettings}
            onChange={() => setShowAFSettings(!showAFSettings)}
            variant="outlined"
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2">
                Autofocus Settings
                <Chip
                  label={parameterValue.autoFocusMode || "software"}
                  size="small"
                  sx={{ ml: 1 }}
                />
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: "block" }}>
                These settings are shared with the Z/Focus tab. Changes apply to both.
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {/* Mode selector */}
                <FormControl size="small" sx={{ minWidth: 140 }}>
                  <InputLabel>AF Mode</InputLabel>
                  <Select
                    value={parameterValue.autoFocusMode || "software"}
                    label="AF Mode"
                    onChange={(e) => dispatch(experimentSlice.setAutoFocusMode(e.target.value))}
                  >
                    <MenuItem value="software">Software (Z-sweep)</MenuItem>
                    <MenuItem value="hardware">Hardware (FocusLock)</MenuItem>
                  </Select>
                </FormControl>

                {/* Software AF parameters */}
                {(parameterValue.autoFocusMode || "software") === "software" && (
                  <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
                    <TextField
                      label="Range (µm)"
                      type="number"
                      size="small"
                      value={parameterValue.autoFocusRange ?? 100}
                      onChange={(e) => dispatch(experimentSlice.setAutoFocusRange(Number(e.target.value)))}
                      sx={{ width: 110 }}
                    />
                    <TextField
                      label="Resolution"
                      type="number"
                      size="small"
                      value={parameterValue.autoFocusResolution ?? 10}
                      onChange={(e) => dispatch(experimentSlice.setAutoFocusResolution(Number(e.target.value)))}
                      sx={{ width: 110 }}
                    />
                    <TextField
                      label="Crop Size"
                      type="number"
                      size="small"
                      value={parameterValue.autoFocusCropsize ?? 2048}
                      onChange={(e) => dispatch(experimentSlice.setAutoFocusCropsize(Number(e.target.value)))}
                      sx={{ width: 110 }}
                    />
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel>Algorithm</InputLabel>
                      <Select
                        value={parameterValue.autoFocusAlgorithm || "LAPE"}
                        label="Algorithm"
                        onChange={(e) => dispatch(experimentSlice.setAutoFocusAlgorithm(e.target.value))}
                      >
                        <MenuItem value="LAPE">LAPE</MenuItem>
                        <MenuItem value="GLVA">GLVA</MenuItem>
                        <MenuItem value="JPEG">JPEG</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                )}

                {/* Hardware AF parameters */}
                {parameterValue.autoFocusMode === "hardware" && (
                  <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
                    <TextField
                      label="Max Attempts"
                      type="number"
                      size="small"
                      value={parameterValue.autofocus_max_attempts ?? 3}
                      onChange={(e) => dispatch(experimentSlice.setAutoFocusMaxAttempts(Number(e.target.value)))}
                      inputProps={{ min: 1, max: 20 }}
                      sx={{ width: 120 }}
                    />
                    <TextField
                      label="Target Setpoint"
                      type="number"
                      size="small"
                      value={parameterValue.autofocus_target_focus_setpoint ?? 0}
                      onChange={(e) => dispatch(experimentSlice.setAutoFocusTargetSetpoint(Number(e.target.value)))}
                      inputProps={{ step: 0.1 }}
                      sx={{ width: 130 }}
                    />
                  </Box>
                )}

                {/* Illumination channel */}
                <FormControl size="small" sx={{ minWidth: 180 }}>
                  <InputLabel>AF Illumination Channel</InputLabel>
                  <Select
                    value={parameterValue.autoFocusIlluminationChannel || ""}
                    label="AF Illumination Channel"
                    onChange={(e) => dispatch(experimentSlice.setAutoFocusIlluminationChannel(e.target.value))}
                  >
                    <MenuItem value="">Auto (active channel)</MenuItem>
                    {illuSources.map((src) => (
                      <MenuItem key={src} value={src}>
                        {src}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                {/* Common AF parameters */}
                <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
                  <TextField
                    label="Settle Time (s)"
                    type="number"
                    size="small"
                    value={parameterValue.autoFocusSettleTime ?? 0.1}
                    onChange={(e) => dispatch(experimentSlice.setAutoFocusSettleTime(Number(e.target.value)))}
                    inputProps={{ step: 0.05, min: 0 }}
                    sx={{ width: 130 }}
                  />
                  <TextField
                    label="Static Offset (µm)"
                    type="number"
                    size="small"
                    value={parameterValue.autoFocusStaticOffset ?? 0}
                    onChange={(e) => dispatch(experimentSlice.setAutoFocusStaticOffset(Number(e.target.value)))}
                    inputProps={{ step: 0.5 }}
                    sx={{ width: 140 }}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={parameterValue.autoFocusTwoStage ?? false}
                        onChange={(e) => dispatch(experimentSlice.setAutoFocusTwoStage(e.target.checked))}
                        size="small"
                      />
                    }
                    label="Two-stage AF"
                  />
                </Box>
              </Box>
            </AccordionDetails>
          </Accordion>

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

          {/* ── Manual Focus Points ──────────────────────────────────── */}
          <Accordion
            expanded={showManualPoints}
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
                Manually define XYZ reference points for focus map fitting.
                Move the stage to the desired position, focus, and click "Add Current Position",
                or enter coordinates manually. You can also click on the wellplate viewer to add
                points (XY from click, Z from current stage position).
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
                        <TableRow key={idx}>
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
                              value={pt.z}
                              onChange={(e) =>
                                dispatch(
                                  focusMapSlice.updateManualPointZ({
                                    index: idx,
                                    z: parseFloat(e.target.value) || 0,
                                  })
                                )
                              }
                              variant="standard"
                              sx={{ width: 80 }}
                            />
                          </TableCell>
                          <TableCell align="right">
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

              <Box sx={{ display: "flex", gap: 1 }}>
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
                    startIcon={<PlayArrowIcon />}
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

          {/* ── Action Buttons ────────────────────────────────────────── */}
          <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
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
                    <ExpandMoreIcon
                      fontSize="small"
                      sx={{
                        transform: expandedFitGroup === groupId ? "rotate(180deg)" : "none",
                        transition: "transform 0.2s",
                      }}
                    />
                  </Box>

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
                          {result.fit_stats.mean_abs_error?.toFixed(4) ?? "N/A"} µm
                        </Typography>

                        {result.fit_stats.r_squared !== undefined && (
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

          {/* ── Visualization ─────────────────────────────────────────── */}
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
