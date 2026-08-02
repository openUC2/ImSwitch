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
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
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

/**
 * Small SVG preview of the sub-position pattern inside a single well.
 * Renders the first well of the labware (circle or rectangle) and overlays
 * the dot offsets so the user sees exactly what will be applied.
 */
function WellSubPositionPreview({ labwareDef, offsets }) {
  if (!labwareDef || !labwareDef.wells) return null;
  const firstId = (labwareDef.well_names_flat && labwareDef.well_names_flat[0]) ||
    Object.keys(labwareDef.wells)[0];
  const w = labwareDef.wells[firstId];
  if (!w) return null;
  const g = w.geometry || {};
  const isCircle = g.shape === "circle";
  const halfW = isCircle ? (g.radius || 0) : (g.width || 0) / 2;
  const halfH = isCircle ? (g.radius || 0) : (g.height || 0) / 2;
  const maxOffX = offsets.reduce((m, o) => Math.max(m, Math.abs(o.dx)), 0);
  const maxOffY = offsets.reduce((m, o) => Math.max(m, Math.abs(o.dy)), 0);
  const viewHalfW = Math.max(halfW, maxOffX + 50) * 1.15;
  const viewHalfH = Math.max(halfH, maxOffY + 50) * 1.15;
  const size = 120;
  return (
    <Box sx={{ ml: "auto" }}>
      <Typography variant="caption" sx={{ display: "block", textAlign: "center" }}>
        Preview ({firstId})
      </Typography>
      <svg
        width={size}
        height={size}
        viewBox={`${-viewHalfW} ${-viewHalfH} ${2 * viewHalfW} ${2 * viewHalfH}`}
        style={{ border: "1px solid #ccc", background: "#fff" }}
      >
        {isCircle ? (
          <circle cx={0} cy={0} r={halfW} fill="#e3f2fd" stroke="#1976d2" strokeWidth={viewHalfW * 0.01} />
        ) : (
          <rect
            x={-halfW} y={-halfH} width={2 * halfW} height={2 * halfH}
            fill="#e3f2fd" stroke="#1976d2" strokeWidth={viewHalfW * 0.01}
          />
        )}
        {offsets.map((o, i) => (
          <circle key={i} cx={o.dx} cy={o.dy} r={Math.max(viewHalfW, viewHalfH) * 0.04} fill="#d32f2f" />
        ))}
      </svg>
    </Box>
  );
}

/**
 * Shows the currently-applied points grouped by well. Each well row is
 * collapsible: clicking it reveals the individual sub-positions (their
 * absolute X/Y/Z and shared ``areaId`` / ``groupId``). This makes the
 * per-well grouping that the zarr/tif writer uses visible to the user.
 */
function PerWellPointsOverview({ points }) {
  const [expandedWell, setExpandedWell] = useState(null);
  const grouped = useMemo(() => {
    const map = new Map();
    for (const p of points || []) {
      const key = p.wellId || p.areaId || p.groupId || p.name || `pt_${p.id}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(p);
    }
    return Array.from(map.entries());
  }, [points]);
  if (grouped.length === 0) return null;
  return (
    <Box sx={{ mt: 1, border: "1px solid #eee", borderRadius: 1, p: 1 }}>
      <Typography variant="caption" sx={{ display: "block", mb: 0.5 }}>
        Points per well ({grouped.length} group{grouped.length !== 1 ? "s" : ""})
      </Typography>
      <Stack spacing={0.25}>
        {grouped.map(([wellKey, pts]) => {
          const isOpen = expandedWell === wellKey;
          return (
            <Box key={wellKey}>
              <Box
                onClick={() => setExpandedWell(isOpen ? null : wellKey)}
                sx={{
                  display: "flex",
                  alignItems: "center",
                  cursor: "pointer",
                  px: 0.5,
                  py: 0.25,
                  borderRadius: 0.5,
                  "&:hover": { background: "#f5f5f5" },
                }}
              >
                <Typography variant="body2" sx={{ flex: 1 }}>
                  <strong>{wellKey}</strong>
                  {pts[0]?.conditionLabel && ` · ${pts[0].conditionLabel}`}
                </Typography>
                <Chip size="small" label={`${pts.length} pt`} />
              </Box>
              {isOpen && (
                <Box sx={{ pl: 2, pb: 0.5 }}>
                  {pts.map((p) => (
                    <Typography
                      key={p.id}
                      variant="caption"
                      sx={{ display: "block", fontFamily: "monospace" }}
                    >
                      {p.name || "(unnamed)"} · X={Math.round(p.x)} Y=
                      {Math.round(p.y)} Z={Math.round(p.z)}
                    </Typography>
                  ))}
                </Box>
              )}
            </Box>
          );
        })}
      </Stack>
    </Box>
  );
}

const LabwareSelectionPanel = ({ defaultExpanded = true }) => {
  const dispatch = useDispatch();
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const experimentState = useSelector(experimentSlice.getExperimentState);

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

  // Sub-position generation (per-well): "center" | "grid"
  const [subPosMode, setSubPosMode] = useState("center");
  const [subNx, setSubNx] = useState(2);
  const [subNy, setSubNy] = useState(2);
  const [subSpacingX, setSubSpacingX] = useState(500); // µm
  const [subSpacingY, setSubSpacingY] = useState(500); // µm

  // Pending labware change awaiting user confirmation
  const [pendingLoadName, setPendingLoadName] = useState(null);

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
    const newName = event.target.value;
    if (newName === loadName) return;
    const hasPoints = (experimentState?.pointList?.length || 0) > 0;
    const hasSelection = (selectedWellIds?.length || 0) > 0;
    if (hasPoints || hasSelection) {
      setPendingLoadName(newName);
      return;
    }
    applyLabwareChange(newName);
  };

  const applyLabwareChange = (newName) => {
    dispatch(wellSelectorSlice.setLabwareLoadName(newName));
    dispatch(wellSelectorSlice.clearSelectedWellIds());
    dispatch(wellSelectorSlice.clearConditionLabels());
  };

  const handleConfirmLabwareChange = () => {
    const target = pendingLoadName;
    setPendingLoadName(null);
    if (target == null) return;
    // Also clear the experiment's pointList — the previous wells are no
    // longer meaningful in the new plate's coordinate system.
    dispatch(experimentSlice.setPointList([]));
    applyLabwareChange(target);
  };

  const handleCancelLabwareChange = () => setPendingLoadName(null);

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

  // Compute sub-position offsets (µm) relative to a well centre.
  const subOffsets = useMemo(() => {
    if (subPosMode !== "grid") return [{ dx: 0, dy: 0, ix: 0, iy: 0 }];
    const nx = Math.max(1, Math.floor(Number(subNx) || 1));
    const ny = Math.max(1, Math.floor(Number(subNy) || 1));
    const sx = Number(subSpacingX) || 0;
    const sy = Number(subSpacingY) || 0;
    const out = [];
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const dx = (ix - (nx - 1) / 2) * sx;
        const dy = (iy - (ny - 1) / 2) * sy;
        out.push({ dx, dy, ix, iy });
      }
    }
    return out;
  }, [subPosMode, subNx, subNy, subSpacingX, subSpacingY]);

  const expandPointsWithSubPositions = useCallback(
    (points) => {
      if (subPosMode !== "grid" || subOffsets.length <= 1) return points;
      const expanded = [];
      for (const p of points) {
        // All sub-positions of one well share an areaId / groupId so the
        // downstream zarr/tif writer co-locates their tiles in a single
        // per-well folder (mirrors the area-select grouping).
        const wellKey = p.wellId || p.name || `${p.x}_${p.y}`;
        const areaId = `well_${wellKey}`;
        for (const off of subOffsets) {
          const suffix = `_r${off.iy}c${off.ix}`;
          expanded.push({
            ...p,
            x: (p.x || 0) + off.dx,
            y: (p.y || 0) + off.dy,
            name: (p.name || "") + suffix,
            areaId,
            groupId: areaId,
            areaType: p.areaType || "well_subpositions",
          });
        }
      }
      return expanded;
    },
    [subPosMode, subOffsets]
  );

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
      const points = expandPointsWithSubPositions(result?.points || []);
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

              {/* Per-well sub-positions */}
              <Box sx={{ display: "flex", gap: 2, alignItems: "flex-start", flexWrap: "wrap" }}>
                <FormControl size="small">
                  <Typography variant="caption">Per-well points</Typography>
                  <RadioGroup
                    row
                    value={subPosMode}
                    onChange={(e) => setSubPosMode(e.target.value)}
                  >
                    <FormControlLabel value="center" control={<Radio size="small" />} label="Center" />
                    <FormControlLabel value="grid" control={<Radio size="small" />} label="N×M grid" />
                  </RadioGroup>
                </FormControl>
                {subPosMode === "grid" && (
                  <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
                    <TextField
                      size="small" type="number" label="Nx"
                      value={subNx}
                      onChange={(e) => setSubNx(Math.max(1, parseInt(e.target.value || "1", 10)))}
                      sx={{ width: 70 }}
                    />
                    <TextField
                      size="small" type="number" label="Ny"
                      value={subNy}
                      onChange={(e) => setSubNy(Math.max(1, parseInt(e.target.value || "1", 10)))}
                      sx={{ width: 70 }}
                    />
                    <TextField
                      size="small" type="number" label="Δx (µm)"
                      value={subSpacingX}
                      onChange={(e) => setSubSpacingX(parseFloat(e.target.value || "0"))}
                      sx={{ width: 100 }}
                    />
                    <TextField
                      size="small" type="number" label="Δy (µm)"
                      value={subSpacingY}
                      onChange={(e) => setSubSpacingY(parseFloat(e.target.value || "0"))}
                      sx={{ width: 100 }}
                    />
                    <Typography variant="caption" sx={{ color: "text.secondary" }}>
                      = {subOffsets.length} pt/well
                    </Typography>
                  </Box>
                )}
                <WellSubPositionPreview
                  labwareDef={labwareDef}
                  offsets={subOffsets}
                />
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

              {/* Structured per-well point overview. Groups currently
                  applied points by their ``wellId`` (sub-positions of one
                  well share an areaId / groupId so they end up in the same
                  zarr/tif folder downstream). */}
              <PerWellPointsOverview points={experimentState?.pointList || []} />
            </>
          )}
        </Stack>
      </AccordionDetails>
      <Dialog
        open={pendingLoadName != null}
        onClose={handleCancelLabwareChange}
      >
        <DialogTitle>Switch labware?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Switching to <strong>{pendingLoadName}</strong> will clear the
            current well selection
            {(experimentState?.pointList?.length || 0) > 0 && (
              <> and remove all {experimentState.pointList.length} point(s) from the experiment</>
            )}
            . This cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelLabwareChange}>Cancel</Button>
          <Button onClick={handleConfirmLabwareChange} color="error" variant="contained">
            Switch and clear
          </Button>
        </DialogActions>
      </Dialog>
    </Accordion>
  );
};

export default LabwareSelectionPanel;
