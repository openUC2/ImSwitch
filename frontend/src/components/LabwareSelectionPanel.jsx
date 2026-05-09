// src/components/LabwareSelectionPanel.jsx
//
// Opentrons-style labware selection panel. Lets the user pick a labware
// loadName, select wells (by clicking chips, range pattern, rows, columns,
// or "all"), tag wells with free-text condition labels, and push the
// resulting points into the experiment's pointList — all via the new
// `LabwareManager` backend endpoints.
//
// Mount this above (or alongside) the WellSelectorCanvas inside
// WellSelectorComponent.

import React, { useEffect, useMemo, useState, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Button,
  Chip,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Stack,
  TextField,
  Tooltip,
  Typography,
  Alert,
  CircularProgress,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

import * as wellSelectorSlice from "../state/slices/WellSelectorSlice.js";
import * as experimentSlice from "../state/slices/ExperimentSlice.js";

import apiExperimentControllerGetLabwareList from "../backendapi/apiExperimentControllerGetLabwareList.js";
import apiExperimentControllerGetLabwareDefinition from "../backendapi/apiExperimentControllerGetLabwareDefinition.js";
import apiExperimentControllerSelectWellsByPattern from "../backendapi/apiExperimentControllerSelectWellsByPattern.js";
import apiExperimentControllerApplyWellSelectionToExperiment from "../backendapi/apiExperimentControllerApplyWellSelectionToExperiment.js";

/**
 * Convert a LabwareDefinition (µm) into the legacy wellLayout object the
 * existing WellSelectorCanvas expects, so picking a labware also re-paints
 * the canvas.
 */
function labwareDefinitionToWellLayout(def) {
  if (!def || !def.wells) return null;
  const wells = Object.values(def.wells).map((w) => {
    const g = w.geometry || {};
    if (g.shape === "rectangle") {
      return {
        id: w.id,
        name: w.id,
        x: w.x,
        y: w.y,
        shape: "rectangle",
        width: g.width || 0,
        height: g.height || 0,
        row: w.row,
        col: w.column,
      };
    }
    return {
      id: w.id,
      name: w.id,
      x: w.x,
      y: w.y,
      shape: "circle",
      radius: g.radius || 0,
      row: w.row,
      col: w.column,
    };
  });
  return {
    name: def.load_name,
    unit: "um",
    width: (def.dimensions && def.dimensions.x) || 0,
    height: (def.dimensions && def.dimensions.y) || 0,
    wells,
  };
}

const LabwareSelectionPanel = ({ defaultExpanded = true }) => {
  const dispatch = useDispatch();
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);

  const [labwareList, setLabwareList] = useState([]);
  const [labwareDef, setLabwareDef] = useState(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDef, setLoadingDef] = useState(false);
  const [error, setError] = useState(null);

  const [patternInput, setPatternInput] = useState(""); // e.g. "A1:C3, B5"
  const [conditionInput, setConditionInput] = useState("");
  const [applyMode, setApplyMode] = useState("append"); // "append" | "replace"
  const [pointNameTemplate, setPointNameTemplate] = useState("{well_id}");
  const [busyApply, setBusyApply] = useState(false);

  const loadName = wellSelectorState.labwareLoadName;
  const selectedWellIds = useMemo(
    () => wellSelectorState.selectedWellIds || [],
    [wellSelectorState.selectedWellIds]
  );
  const conditionLabels = useMemo(
    () => wellSelectorState.conditionLabels || {},
    [wellSelectorState.conditionLabels]
  );
  const offsetX = wellSelectorState.layoutOffsetX || 0;
  const offsetY = wellSelectorState.layoutOffsetY || 0;

  // ── Load labware list once ────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setLoadingList(true);
    apiExperimentControllerGetLabwareList()
      .then((items) => {
        if (cancelled) return;
        setLabwareList(items || []);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(`Failed to load labware list: ${e?.message || e}`);
      })
      .finally(() => !cancelled && setLoadingList(false));
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Load full definition when selection changes ───────────────────────
  useEffect(() => {
    if (!loadName) {
      setLabwareDef(null);
      return;
    }
    let cancelled = false;
    setLoadingDef(true);
    setError(null);
    apiExperimentControllerGetLabwareDefinition(loadName, offsetX, offsetY)
      .then((def) => {
        if (cancelled) return;
        setLabwareDef(def);
        // Push to canvas as the active wellLayout for visualisation.
        const layout = labwareDefinitionToWellLayout(def);
        if (layout) {
          dispatch(experimentSlice.setWellLayout(layout));
        }
      })
      .catch((e) => {
        if (cancelled) return;
        setError(`Failed to load definition: ${e?.message || e}`);
      })
      .finally(() => !cancelled && setLoadingDef(false));
    return () => {
      cancelled = true;
    };
  }, [loadName, offsetX, offsetY, dispatch]);

  // Derived: well IDs in row-major order
  const allWellIds = useMemo(() => {
    if (!labwareDef) return [];
    if (Array.isArray(labwareDef.well_names_flat)) {
      return labwareDef.well_names_flat;
    }
    return Object.keys(labwareDef.wells || {});
  }, [labwareDef]);

  const handleLabwareChange = (event) => {
    dispatch(wellSelectorSlice.setLabwareLoadName(event.target.value));
    dispatch(wellSelectorSlice.clearSelectedWellIds());
    dispatch(wellSelectorSlice.clearConditionLabels());
  };

  const handleToggleWell = (wellId) => {
    dispatch(wellSelectorSlice.toggleSelectedWellId(wellId));
  };

  const handleSelectAll = () => {
    dispatch(wellSelectorSlice.setSelectedWellIds(allWellIds));
  };

  const handleClearSelection = () => {
    dispatch(wellSelectorSlice.clearSelectedWellIds());
  };

  // Build the WellSelectionPattern payload (combines explicit chips + the
  // free-text pattern field in one go so the user can mix them).
  const buildPattern = useCallback(() => {
    const ranges = patternInput
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    return {
      wells: selectedWellIds.length > 0 ? selectedWellIds : undefined,
      ranges: ranges.length > 0 ? ranges : undefined,
    };
  }, [patternInput, selectedWellIds]);

  // ── Resolve pattern preview (also expands ranges into chips) ──────────
  const handleResolvePattern = async () => {
    if (!loadName) return;
    setError(null);
    try {
      const result = await apiExperimentControllerSelectWellsByPattern({
        loadName,
        pattern: buildPattern(),
        offsetXUm: offsetX,
        offsetYUm: offsetY,
      });
      const ids = (result?.wells || []).map((w) => w.well_id);
      dispatch(wellSelectorSlice.setSelectedWellIds(ids));
      setPatternInput(""); // ranges have been expanded into chips
    } catch (e) {
      setError(`Pattern resolution failed: ${e?.message || e}`);
    }
  };

  const handleApplyConditionToSelected = () => {
    if (!conditionInput) return;
    const next = { ...conditionLabels };
    for (const id of selectedWellIds) next[id] = conditionInput;
    dispatch(wellSelectorSlice.setConditionLabels(next));
  };

  const handleClearConditions = () => {
    dispatch(wellSelectorSlice.clearConditionLabels());
  };

  const handleApply = async () => {
    if (!loadName) return;
    setError(null);
    setBusyApply(true);
    try {
      const result = await apiExperimentControllerApplyWellSelectionToExperiment({
        loadName,
        pattern: buildPattern(),
        offsetXUm: offsetX,
        offsetYUm: offsetY,
        conditionLabels,
        pointNameTemplate: pointNameTemplate || "{well_id}",
      });
      const points = result?.points || [];
      if (applyMode === "replace") {
        dispatch(experimentSlice.replacePoints(points));
      } else {
        dispatch(experimentSlice.appendPoints(points));
      }
    } catch (e) {
      setError(`Apply failed: ${e?.message || e}`);
    } finally {
      setBusyApply(false);
    }
  };

  return (
    <Accordion defaultExpanded={defaultExpanded} sx={{ mb: 1 }}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="body2">
          Labware (Opentrons) – {loadName || "no plate selected"}
          {selectedWellIds.length > 0 && ` · ${selectedWellIds.length} well(s)`}
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={1.5}>
          {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}

          {/* Labware picker */}
          <FormControl size="small" fullWidth>
            <InputLabel>Labware</InputLabel>
            <Select
              label="Labware"
              value={loadName || ""}
              onChange={handleLabwareChange}
              disabled={loadingList}
            >
              <MenuItem value="">
                <em>{loadingList ? "Loading…" : "(none)"}</em>
              </MenuItem>
              {labwareList.map((item) => (
                <MenuItem key={item.load_name} value={item.load_name}>
                  {item.display_name || item.load_name}
                  {item.format ? ` · ${item.format}` : ""}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {loadingDef && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="caption">Loading definition…</Typography>
            </Box>
          )}

          {/* Selection controls */}
          {labwareDef && (
            <>
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                <Button size="small" variant="outlined" onClick={handleSelectAll}>
                  Select all ({allWellIds.length})
                </Button>
                <Button size="small" variant="outlined" onClick={handleClearSelection}>
                  Clear
                </Button>
                <Tooltip title="Examples: A1:C3, B5, A1:A12">
                  <TextField
                    size="small"
                    label="Pattern"
                    value={patternInput}
                    onChange={(e) => setPatternInput(e.target.value)}
                    placeholder="A1:C3, B5"
                    sx={{ minWidth: 200 }}
                  />
                </Tooltip>
                <Button size="small" variant="contained" onClick={handleResolvePattern}>
                  Resolve
                </Button>
              </Box>

              {/* Well chip grid */}
              <Box
                sx={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 0.5,
                  maxHeight: 220,
                  overflowY: "auto",
                  border: "1px solid #ddd",
                  borderRadius: 1,
                  p: 1,
                  backgroundColor: "primary.light",
                }}
              >
                {allWellIds.map((id) => {
                  const selected = selectedWellIds.includes(id);
                  const label = conditionLabels[id];
                  return (
                    <Chip
                      key={id}
                      label={label ? `${id}·${label}` : id}
                      size="small"
                      color={selected ? "primary" : "secondary"}
                      variant={selected ? "filled" : "outlined"}
                      onClick={() => handleToggleWell(id)}
                    />
                  );
                })}
              </Box>

              {/* Condition labelling */}
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
                <TextField
                  size="small"
                  label="Condition label"
                  value={conditionInput}
                  onChange={(e) => setConditionInput(e.target.value)}
                  placeholder="e.g. Donor1+DMSO"
                  sx={{ minWidth: 200 }}
                />
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleApplyConditionToSelected}
                  disabled={!conditionInput || selectedWellIds.length === 0}
                >
                  Tag selected
                </Button>
                <Button size="small" variant="text" onClick={handleClearConditions}>
                  Clear all conditions
                </Button>
              </Box>

              {/* Apply controls */}
              <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
                <FormControl>
                  <RadioGroup
                    row
                    value={applyMode}
                    onChange={(e) => setApplyMode(e.target.value)}
                  >
                    <FormControlLabel value="append" control={<Radio size="small" />} label="Append" />
                    <FormControlLabel value="replace" control={<Radio size="small" />} label="Replace" />
                  </RadioGroup>
                </FormControl>
                <TextField
                  size="small"
                  label="Point name template"
                  value={pointNameTemplate}
                  onChange={(e) => setPointNameTemplate(e.target.value)}
                  helperText="{well_id} {row} {column} {label}"
                  sx={{ minWidth: 220 }}
                />
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleApply}
                  disabled={busyApply || (selectedWellIds.length === 0 && !patternInput)}
                  startIcon={busyApply ? <CircularProgress size={14} color="inherit" /> : null}
                >
                  Apply to experiment
                </Button>
              </Box>
            </>
          )}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
};

export default LabwareSelectionPanel;
