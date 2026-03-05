import React, { useState, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  FormControlLabel,
  Switch,
  Checkbox,
  Grid,
  Chip,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

import Frame3DViewer from "../components/Frame3DViewer.jsx";
import * as frame3DSlice from "../state/slices/Frame3DViewerSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";

/**
 * Frame3DViewerPanel – full panel that wraps the 3D viewer with
 * configuration controls for axis mapping (offset / scale / invert)
 * and visibility toggles.  Reads live positions from the global
 * Redux position slice.
 */
const AXIS_LABELS = {
  stageX: "Stage → X",
  stageY: "Stage → Y",
  stageZ: "Stage → Z",
  turretX: "Turret → Focus",
};

const MICROSCOPE_AXES = ["x", "y", "z", "a"];
const MODEL_AXES = ["x", "y", "z"];

const Frame3DViewerPanel = () => {
  const dispatch = useDispatch();

  // Redux state
  const positions = useSelector(positionSlice.getPositionState);
  const viewerState = useSelector(frame3DSlice.getFrame3DViewerState);
  const { axisConfig, cameraState, visibility } = viewerState;

  // Local expanded state for the config accordion
  const [configOpen, setConfigOpen] = useState(false);

  // Camera change handler (debounced inside the viewer)
  const handleCameraChange = useCallback(
    (newCam) => dispatch(frame3DSlice.setCameraState(newCam)),
    [dispatch]
  );

  // Axis config change helper
  const updateAxisConfig = (key, field, value) => {
    dispatch(frame3DSlice.setAxisConfig({ key, config: { [field]: value } }));
  };

  // Visibility toggle helper
  const toggleVisibility = (group, checked) => {
    dispatch(frame3DSlice.setVisibility({ group, visible: checked }));
  };

  return (
    <Box sx={{ p: 1 }}>
      <Typography variant="h6" gutterBottom>
        FRAME 3D Digital Twin
      </Typography>

      {/* Live position readout */}
      <Box sx={{ display: "flex", gap: 1, mb: 1, flexWrap: "wrap" }}>
        <Chip label={`X: ${(positions.x ?? 0).toFixed(1)}`} size="small" variant="outlined" />
        <Chip label={`Y: ${(positions.y ?? 0).toFixed(1)}`} size="small" variant="outlined" />
        <Chip label={`Z: ${(positions.z ?? 0).toFixed(1)}`} size="small" variant="outlined" />
        <Chip label={`A: ${(positions.a ?? 0).toFixed(1)}`} size="small" variant="outlined" />
      </Box>

      {/* 3D Viewer */}
      <Frame3DViewer
        positions={positions}
        axisConfig={axisConfig}
        visibility={visibility}
        cameraState={cameraState}
        onCameraChange={handleCameraChange}
        width={600}
        height={420}
      />

      {/* Visibility toggles */}
      <Box sx={{ display: "flex", gap: 2, mt: 1, flexWrap: "wrap" }}>
        <FormControlLabel
          control={
            <Checkbox
              checked={visibility.base}
              onChange={(e) => toggleVisibility("base", e.target.checked)}
              size="small"
            />
          }
          label="Base"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={visibility.stage}
              onChange={(e) => toggleVisibility("stage", e.target.checked)}
              size="small"
            />
          }
          label="XY Stage"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={visibility.turret}
              onChange={(e) => toggleVisibility("turret", e.target.checked)}
              size="small"
            />
          }
          label="Obj. Turret"
        />
      </Box>

      {/* Axis Configuration (collapsible) */}
      <Accordion expanded={configOpen} onChange={() => setConfigOpen(!configOpen)} sx={{ mt: 1 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">Axis Mapping Configuration</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={1}>
            {Object.entries(AXIS_LABELS).map(([key, label]) => {
              const cfg = axisConfig[key] || {};
              return (
                <Grid item xs={12} key={key}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      flexWrap: "wrap",
                      borderBottom: "1px solid #eee",
                      pb: 0.5,
                      mb: 0.5,
                    }}
                  >
                    <Typography variant="body2" sx={{ minWidth: 120 }}>
                      {label}
                    </Typography>

                    {/* Microscope axis selector */}
                    <TextField
                      select
                      label="Src Axis"
                      size="small"
                      value={cfg.microscopeAxis || "x"}
                      onChange={(e) => updateAxisConfig(key, "microscopeAxis", e.target.value)}
                      SelectProps={{ native: true }}
                      sx={{ width: 90 }}
                    >
                      {MICROSCOPE_AXES.map((a) => (
                        <option key={a} value={a}>
                          {a.toUpperCase()}
                        </option>
                      ))}
                    </TextField>

                    {/* Model axis selector */}
                    <TextField
                      select
                      label="3D Axis"
                      size="small"
                      value={cfg.modelAxis || "x"}
                      onChange={(e) => updateAxisConfig(key, "modelAxis", e.target.value)}
                      SelectProps={{ native: true }}
                      sx={{ width: 90 }}
                    >
                      {MODEL_AXES.map((a) => (
                        <option key={a} value={a}>
                          {a.toUpperCase()}
                        </option>
                      ))}
                    </TextField>

                    {/* Offset */}
                    <TextField
                      label="Offset"
                      type="number"
                      size="small"
                      value={cfg.offset ?? 0}
                      onChange={(e) => updateAxisConfig(key, "offset", parseFloat(e.target.value) || 0)}
                      sx={{ width: 90 }}
                      inputProps={{ step: 1 }}
                    />

                    {/* Scale */}
                    <TextField
                      label="Scale"
                      type="number"
                      size="small"
                      value={cfg.scale ?? 0.001}
                      onChange={(e) => updateAxisConfig(key, "scale", parseFloat(e.target.value) || 0.001)}
                      sx={{ width: 90 }}
                      inputProps={{ step: 0.0001 }}
                    />

                    {/* Invert toggle */}
                    <FormControlLabel
                      control={
                        <Switch
                          checked={cfg.invert || false}
                          onChange={(e) => updateAxisConfig(key, "invert", e.target.checked)}
                          size="small"
                        />
                      }
                      label="+/-"
                    />
                  </Box>
                </Grid>
              );
            })}
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default Frame3DViewerPanel;
