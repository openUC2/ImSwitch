import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Button, Input, Select, MenuItem, FormControl, FormControlLabel, Checkbox, Typography, Box } from "@mui/material";

import * as experimentSlice from "../state/slices/ExperimentSlice";

import { Shape } from "./WellSelectorCanvas.js";

//##################################################################################
const PointListShapeEditorComponent = () => {
  const Shape_Off = "off"; //Note: MenuItems cant have emty value (""/Shape.EPMTY) as default value
  const Shape_CenterOnly = "center_only"; // New mode for center-only positions
  const Shape_WellPattern = "well_pattern"; // New mode for patterns within wells
  
  // States for shape, neighborsX, and neighborsY
  const [shape, setShape] = useState(Shape_Off);
  const [rectPlusX, setRectPlusX] = useState(0);
  const [rectMinusX, setRectMinusX] = useState(0);
  const [rectPlusY, setRectPlusY] = useState(0);
  const [rectMinusY, setRectMinusY] = useState(0);
  const [circleRadiusX, setCircleRadiusX] = useState(0);
  const [circleRadiusY, setCircleRadiusY] = useState(0);
  
  // New states for well pattern mode
  const [wellPatternType, setWellPatternType] = useState("circle"); // "circle" or "rectangle"
  const [patternRadius, setPatternRadius] = useState(50); // For circle pattern
  const [patternWidth, setPatternWidth] = useState(100); // For rectangle pattern
  const [patternHeight, setPatternHeight] = useState(100); // For rectangle pattern
  const [patternOverlap, setPatternOverlap] = useState(0.1); // Overlap between positions (0.0 - 1.0)

  //redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);

  //##################################################################################
  const handleSetAllClick = () => {
    // Your logic here
    console.log("Set All button clicked", experimentState);

    let updatedPointList = [];
    if (shape == Shape_Off) {
      updatedPointList = experimentState.pointList.map((point) => ({
        ...point,
        shape: shape,
      }));
    } else if (shape == Shape_CenterOnly) {
      updatedPointList = experimentState.pointList.map((point) => ({
        ...point,
        shape: shape,
        wellMode: "center_only"
      }));
    } else if (shape == Shape_WellPattern) {
      updatedPointList = experimentState.pointList.map((point) => ({
        ...point,
        shape: shape,
        wellMode: "pattern",
        patternType: wellPatternType,
        patternRadius: patternRadius,
        patternWidth: patternWidth,
        patternHeight: patternHeight,
        patternOverlap: patternOverlap
      }));
    } else if (shape == Shape.RECTANGLE) {
      updatedPointList = experimentState.pointList.map((point) => ({
        ...point,
        shape: shape,
        rectPlusX: rectPlusX,
        rectPlusY: rectPlusY,
        rectMinusX: rectMinusX,
        rectMinusY: rectMinusY,
      }));
    } else if (shape == Shape.CIRCLE) {
      updatedPointList = experimentState.pointList.map((point) => ({
        ...point,
        shape: shape,
        circleRadiusX: circleRadiusX,
        circleRadiusY: circleRadiusY,
      }));
    }

    // Update the state with the modified list
    dispatch(experimentSlice.setPointList(updatedPointList));
  };

  //##################################################################################
  return (
    <div
      style={{
        textAlign: "left",
        padding: "10px",
        margin: "0px",
      }}
    >
      {/* Header */}
      <h4 style={{ margin: "0", padding: "0" }}>Point List Shape Editor</h4>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          //border: "1px solid #cfc",
        }}
      >
        <Box display="flex" alignItems="center" gap={2}>
          <FormControl>
            <Select
              sx={{ height: "32px", minWidth: "120px" }}
              value={shape}
              onChange={(e) => setShape(e.target.value)} // directly call setShape
            >
              <MenuItem value={Shape_Off}>Off</MenuItem>
              <MenuItem value={Shape_CenterOnly}>Cup Center Only</MenuItem>
              <MenuItem value={Shape_WellPattern}>Cup Pattern</MenuItem>
              <MenuItem value={Shape.RECTANGLE}>Rect</MenuItem>
              <MenuItem value={Shape.CIRCLE}>Circle</MenuItem>
            </Select>
          </FormControl>

          <Button variant="contained" onClick={handleSetAllClick}>
            Set All
          </Button>
        </Box>

        {/* Cup Pattern Mode Configuration */}
        {shape === Shape_WellPattern && (
          <Box 
            sx={{ 
              border: "1px solid #ddd", 
              borderRadius: "4px", 
              padding: "10px",
              backgroundColor: "#f9f9f9" 
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Pattern Configuration
            </Typography>
            
            <Box display="flex" flexDirection="column" gap={2}>
              {/* Pattern Type Selection */}
              <Box display="flex" alignItems="center" gap={2}>
                <Typography variant="body2" sx={{ minWidth: "80px" }}>
                  Pattern Type:
                </Typography>
                <FormControl>
                  <Select
                    size="small"
                    value={wellPatternType}
                    onChange={(e) => setWellPatternType(e.target.value)}
                  >
                    <MenuItem value="circle">Circle</MenuItem>
                    <MenuItem value="rectangle">Rectangle</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              {/* Circle Pattern Parameters */}
              {wellPatternType === "circle" && (
                <Box display="flex" alignItems="center" gap={2}>
                  <Typography variant="body2" sx={{ minWidth: "80px" }}>
                    Radius (μm):
                  </Typography>
                  <Input
                    type="number"
                    value={patternRadius}
                    onChange={(e) => setPatternRadius(parseFloat(e.target.value) || 0)}
                    size="small"
                    sx={{ width: "80px" }}
                  />
                </Box>
              )}

              {/* Rectangle Pattern Parameters */}
              {wellPatternType === "rectangle" && (
                <>
                  <Box display="flex" alignItems="center" gap={2}>
                    <Typography variant="body2" sx={{ minWidth: "80px" }}>
                      Width (μm):
                    </Typography>
                    <Input
                      type="number"
                      value={patternWidth}
                      onChange={(e) => setPatternWidth(parseFloat(e.target.value) || 0)}
                      size="small"
                      sx={{ width: "80px" }}
                    />
                  </Box>
                  <Box display="flex" alignItems="center" gap={2}>
                    <Typography variant="body2" sx={{ minWidth: "80px" }}>
                      Height (μm):
                    </Typography>
                    <Input
                      type="number"
                      value={patternHeight}
                      onChange={(e) => setPatternHeight(parseFloat(e.target.value) || 0)}
                      size="small"
                      sx={{ width: "80px" }}
                    />
                  </Box>
                </>
              )}

              {/* Overlap Parameter */}
              <Box display="flex" alignItems="center" gap={2}>
                <Typography variant="body2" sx={{ minWidth: "80px" }}>
                  Overlap (%):
                </Typography>
                <Input
                  type="number"
                  value={Math.round(patternOverlap * 100)}
                  onChange={(e) => setPatternOverlap((parseFloat(e.target.value) || 0) / 100)}
                  size="small"
                  sx={{ width: "80px" }}
                  inputProps={{ min: 0, max: 90, step: 5 }}
                />
                <Typography variant="caption" color="text.secondary">
                  (0-90%)
                </Typography>
              </Box>
            </Box>
          </Box>
        )}

        {/* Traditional Rectangle Mode */}
        {shape === Shape.RECTANGLE && (
          <Box 
            sx={{ 
              border: "1px solid #ddd", 
              borderRadius: "4px", 
              padding: "10px",
              backgroundColor: "#f9f9f9" 
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Rectangle Configuration
            </Typography>
            <Box display="flex" flexDirection="column" gap={1}>
              <Box display="flex" alignItems="center" gap={2}>
                <Typography variant="body2" sx={{ minWidth: "30px" }}>
                  -X:
                </Typography>
                <Input
                  type="number"
                  value={rectMinusX}
                  min="-1000000"
                  max="0"
                  onChange={(e) => setRectMinusX(parseInt(e.target.value, 10))}
                  size="small"
                  sx={{ width: "80px" }}
                />
                <Typography variant="body2" sx={{ minWidth: "30px" }}>
                  +X:
                </Typography>
                <Input
                  type="number"
                  value={rectPlusX}
                  min="0"
                  max="1000000"
                  onChange={(e) => setRectPlusX(parseInt(e.target.value, 10))}
                  size="small"
                  sx={{ width: "80px" }}
                />
              </Box>
              <Box display="flex" alignItems="center" gap={2}>
                <Typography variant="body2" sx={{ minWidth: "30px" }}>
                  -Y:
                </Typography>
                <Input
                  type="number"
                  value={rectMinusY}
                  min="-1000000"
                  max="0"
                  onChange={(e) => setRectMinusY(parseInt(e.target.value, 10))}
                  size="small"
                  sx={{ width: "80px" }}
                />
                <Typography variant="body2" sx={{ minWidth: "30px" }}>
                  +Y:
                </Typography>
                <Input
                  type="number"
                  value={rectPlusY}
                  min="0"
                  max="1000000"
                  onChange={(e) => setRectPlusY(parseInt(e.target.value, 10))}
                  size="small"
                  sx={{ width: "80px" }}
                />
              </Box>
            </Box>
          </Box>
        )}

        {/* Traditional Circle Mode */}
        {shape === Shape.CIRCLE && (
          <Box 
            sx={{ 
              border: "1px solid #ddd", 
              borderRadius: "4px", 
              padding: "10px",
              backgroundColor: "#f9f9f9" 
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Circle Configuration
            </Typography>
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="body2" sx={{ minWidth: "30px" }}>
                rX:
              </Typography>
              <Input
                type="number"
                value={circleRadiusX}
                min="0"
                max="1000000"
                onChange={(e) => setCircleRadiusX(parseInt(e.target.value, 10))}
                size="small"
                sx={{ width: "80px" }}
              />
              <Typography variant="body2" sx={{ minWidth: "30px" }}>
                rY:
              </Typography>
              <Input
                type="number"
                value={circleRadiusY}
                min="0"
                max="1000000"
                onChange={(e) => setCircleRadiusY(parseInt(e.target.value, 10))}
                size="small"
                sx={{ width: "80px" }}
              />
            </Box>
          </Box>
        )}
      </div>
    </div>
  );
};

export default PointListShapeEditorComponent;
