import React, { useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { Box, Typography, Paper } from "@mui/material";
import WellSelectorCanvas from "../axon/WellSelectorCanvas";
import { Mode } from "../axon/WellSelectorCanvas";
import * as wellSelectorSlice from "../state/slices/WellSelectorSlice";
import * as experimentSlice from "../state/slices/ExperimentSlice";

const StageMapCanvas = ({ width = 500, height = 400 }) => {
  const dispatch = useDispatch();
  
  // Initialize the well selector state for camera movement mode
  useEffect(() => {
    // Set the mode to MOVE_CAMERA so clicking moves the stage
    dispatch(wellSelectorSlice.setMode(Mode.MOVE_CAMERA));
  }, [dispatch]);

  return (
    <Paper elevation={2} sx={{ p: 2, backgroundColor: '#f8f9fa' }}>
      <Typography variant="h6" gutterBottom sx={{ textAlign: 'center' }}>
        Interactive Stage Map
      </Typography>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <div style={{ width, height, border: '2px solid #ddd', borderRadius: '8px' }}>
          <WellSelectorCanvas 
            ref={null}
            width={width}
            height={height}
          />
        </div>
      </Box>
      
      <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', mt: 1, color: '#666' }}>
        Click anywhere on the map to move the stage to that position
        <br />
        Use the mouse wheel to zoom in/out and drag to pan the view
      </Typography>
    </Paper>
  );
};

export default StageMapCanvas;