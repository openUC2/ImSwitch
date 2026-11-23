import React, { useState } from "react";
import { Box, Button, ButtonGroup, TextField, Typography } from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import * as lightsheetSlice from "../state/slices/LightsheetSlice";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner";

const AXES = [
  ["X", "sample"],
  ["Y", "sample"],
  ["Z", "objective"],
  ["A", "sample"],
];

/** Compact jog pad: one row per axis (position, −/+ by the step). */
const LightsheetPositionControls = () => {
  const dispatch = useDispatch();
  const positions = useSelector(lightsheetSlice.getLightsheetState).stagePositions || {};
  const [stepSize, setStepSize] = useState(100);

  const moveAxis = async (axis, distance) => {
    try {
      await apiPositionerControllerMovePositioner({ axis, dist: distance, isAbsolute: false });
      const key = axis.toLowerCase();
      dispatch(lightsheetSlice.setStagePosition({ axis: key, value: (positions[key] || 0) + distance }));
    } catch (error) {
      console.error(`Error moving ${axis}:`, error);
    }
  };

  return (
    <Box>
      <TextField
        label="Step (µm)"
        type="number"
        value={stepSize}
        onChange={(e) => setStepSize(parseFloat(e.target.value) || 100)}
        size="small"
        inputProps={{ step: 10, min: 1 }}
        sx={{ width: 120, mb: 1 }}
      />
      {AXES.map(([axis, what]) => (
        <Box key={axis} sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
          <Typography variant="body2" sx={{ width: 22, fontWeight: 600 }}>{axis}</Typography>
          <Typography variant="body2" sx={{ width: 90, fontFamily: "monospace", textAlign: "right" }}>
            {(positions[axis.toLowerCase()] ?? 0).toFixed(1)}
          </Typography>
          <ButtonGroup size="small" variant="outlined">
            <Button onClick={() => moveAxis(axis, -stepSize)}>−</Button>
            <Button onClick={() => moveAxis(axis, stepSize)}>+</Button>
          </ButtonGroup>
          <Typography variant="caption" color="text.secondary">{what}</Typography>
        </Box>
      ))}
    </Box>
  );
};

export default LightsheetPositionControls;
