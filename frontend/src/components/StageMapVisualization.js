import React from "react";
import { useSelector, useDispatch } from "react-redux";
import { Box, Paper, Typography } from "@mui/material";
import { MyLocation as LocationIcon } from "@mui/icons-material";
import * as stageCenterCalibrationSlice from "../state/slices/StageCenterCalibrationSlice";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner";

const StageMapVisualization = ({ hostIP, hostPort, width = 400, height = 400 }) => {
  const dispatch = useDispatch();
  const stageCenterState = useSelector(stageCenterCalibrationSlice.getStageCenterCalibrationState);
  
  const {
    currentX,
    currentY,
    stageMapWidth,
    stageMapHeight,
    stageMapCenterX,
    stageMapCenterY,
    foundCenterX,
    foundCenterY,
    manualCenterX,
    manualCenterY,
  } = stageCenterState;

  // Convert stage coordinates to SVG coordinates
  const stageToSvg = (stageX, stageY) => {
    const svgX = ((stageX - stageMapCenterX + stageMapWidth / 2) / stageMapWidth) * width;
    const svgY = height - ((stageY - stageMapCenterY + stageMapHeight / 2) / stageMapHeight) * height;
    return { x: svgX, y: svgY };
  };

  // Convert SVG coordinates to stage coordinates
  const svgToStage = (svgX, svgY) => {
    const stageX = ((svgX / width) * stageMapWidth) - (stageMapWidth / 2) + stageMapCenterX;
    const stageY = (((height - svgY) / height) * stageMapHeight) - (stageMapHeight / 2) + stageMapCenterY;
    return { x: Math.round(stageX), y: Math.round(stageY) };
  };

  const handleMapClick = async (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const svgX = event.clientX - rect.left;
    const svgY = event.clientY - rect.top;
    
    const { x: stageX, y: stageY } = svgToStage(svgX, svgY);
    
    dispatch(stageCenterCalibrationSlice.setIsLoading(true));
    try {
      // Move to clicked position
      await apiPositionerControllerMovePositioner({
        positionerName: "ESP32Stage",
        axis: "X",
        dist: stageX,
        isAbsolute: true,
        isBlocking: false,
        speed: 1000,
      });

      await apiPositionerControllerMovePositioner({
        positionerName: "ESP32Stage",
        axis: "Y",
        dist: stageY,
        isAbsolute: true,
        isBlocking: false,
        speed: 1000,
      });

      dispatch(stageCenterCalibrationSlice.setSuccessMessage(`Moving to (${stageX}, ${stageY})`));
    } catch (error) {
      console.error("Error moving to clicked position:", error);
      dispatch(stageCenterCalibrationSlice.setError("Failed to move to clicked position"));
    } finally {
      dispatch(stageCenterCalibrationSlice.setIsLoading(false));
    }
  };

  // Calculate positions for display
  const currentPos = stageToSvg(currentX, currentY);
  const manualCenterPos = manualCenterX && manualCenterY ? stageToSvg(parseFloat(manualCenterX), parseFloat(manualCenterY)) : null;
  const foundCenterPos = foundCenterX !== null && foundCenterY !== null ? stageToSvg(foundCenterX, foundCenterY) : null;

  return (
    <Paper elevation={2} sx={{ p: 2, backgroundColor: '#f8f9fa' }}>
      <Typography variant="h6" gutterBottom sx={{ textAlign: 'center' }}>
        Interactive Stage Map
      </Typography>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <svg
          width={width}
          height={height}
          style={{
            border: '2px solid #ddd',
            borderRadius: '8px',
            cursor: 'crosshair',
            backgroundColor: '#ffffff',
          }}
          onClick={handleMapClick}
        >
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e0e0e0" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
          
          {/* Center crosshair */}
          <line x1={width/2 - 10} y1={height/2} x2={width/2 + 10} y2={height/2} stroke="#999" strokeWidth="1" />
          <line x1={width/2} y1={height/2 - 10} x2={width/2} y2={height/2 + 10} stroke="#999" strokeWidth="1" />
          
          {/* Current position */}
          <circle
            cx={currentPos.x}
            cy={currentPos.y}
            r="8"
            fill="#2196F3"
            stroke="#ffffff"
            strokeWidth="2"
          />
          <text
            x={currentPos.x + 12}
            y={currentPos.y + 4}
            fontSize="12"
            fill="#2196F3"
            fontWeight="bold"
          >
            Current ({currentX.toFixed(0)}, {currentY.toFixed(0)})
          </text>
          
          {/* Manual center position */}
          {manualCenterPos && (
            <>
              <circle
                cx={manualCenterPos.x}
                cy={manualCenterPos.y}
                r="6"
                fill="#FF9800"
                stroke="#ffffff"
                strokeWidth="2"
              />
              <text
                x={manualCenterPos.x + 10}
                y={manualCenterPos.y - 8}
                fontSize="10"
                fill="#FF9800"
                fontWeight="bold"
              >
                Manual Center
              </text>
            </>
          )}
          
          {/* Found center position */}
          {foundCenterPos && (
            <>
              <circle
                cx={foundCenterPos.x}
                cy={foundCenterPos.y}
                r="6"
                fill="#4CAF50"
                stroke="#ffffff"
                strokeWidth="2"
              />
              <text
                x={foundCenterPos.x + 10}
                y={foundCenterPos.y + 16}
                fontSize="10"
                fill="#4CAF50"
                fontWeight="bold"
              >
                Found Center
              </text>
            </>
          )}
        </svg>
      </Box>
      
      {/* Legend */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, flexWrap: 'wrap' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#2196F3', border: '2px solid #fff' }} />
          <Typography variant="caption">Current Position</Typography>
        </Box>
        {manualCenterPos && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#FF9800', border: '2px solid #fff' }} />
            <Typography variant="caption">Manual Center</Typography>
          </Box>
        )}
        {foundCenterPos && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#4CAF50', border: '2px solid #fff' }} />
            <Typography variant="caption">Found Center</Typography>
          </Box>
        )}
      </Box>
      
      <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', mt: 1, color: '#666' }}>
        Click anywhere on the map to move the stage to that position
        <br />
        Map range: ±{(stageMapWidth/2).toFixed(0)}μm (X), ±{(stageMapHeight/2).toFixed(0)}μm (Y)
      </Typography>
    </Paper>
  );
};

export default StageMapVisualization;