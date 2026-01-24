import React, { useState } from "react";
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Checkbox,
  FormControlLabel,
  Grid,
  TextField,
  Typography,
  Button
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useDispatch, useSelector } from "react-redux";
import * as lightsheetSlice from "../state/slices/LightsheetSlice";

/**
 * AxisConfigurationMenu - Advanced menu for configuring axis mappings
 * Allows setting offset, scaling factor, and inversion for each axis (X, Y, Z, A)
 * Settings are stored in Redux and persist during the session
 */
const AxisConfigurationMenu = () => {
  const dispatch = useDispatch();
  const lightsheetState = useSelector(lightsheetSlice.getLightsheetState);
  const axisConfig = lightsheetState.axisConfig;

  const axes = ['x', 'y', 'z', 'a'];
  const axisLabels = {
    x: 'X Axis → Sample X',
    y: 'Y Axis → Sample Z',
    z: 'Z Axis → Objective Y',
    a: 'A Axis → Sample Y'
  };

  // Handle offset change
  const handleOffsetChange = (axis, value) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      dispatch(lightsheetSlice.setAxisOffset({ axis, offset: numValue }));
    }
  };

  // Handle scale change
  const handleScaleChange = (axis, value) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue) && numValue !== 0) {
      dispatch(lightsheetSlice.setAxisScale({ axis, scale: numValue }));
    }
  };

  // Handle invert toggle
  const handleInvertChange = (axis, checked) => {
    dispatch(lightsheetSlice.setAxisInvert({ axis, invert: checked }));
  };

  // Reset all configurations to defaults
  const handleResetAll = () => {
    axes.forEach(axis => {
      dispatch(lightsheetSlice.setAxisConfig({ 
        axis, 
        config: { offset: 0, scale: 1, invert: false } 
      }));
    });
  };

  return (
    <Accordion>
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls="axis-config-content"
        id="axis-config-header"
      >
        <Typography variant="h6">Advanced Axis Configuration</Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Box sx={{ width: "100%" }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Configure offset, scaling, and direction for each axis to match your microscope setup.
          </Typography>
          
          {axes.map((axis) => (
            <Box 
              key={axis} 
              sx={{ 
                mb: 3, 
                p: 2, 
                border: "1px solid #444",
                borderRadius: 1,
                backgroundColor: "background.paper"
              }}
            >
              <Typography variant="subtitle1" gutterBottom>
                {axisLabels[axis]}
              </Typography>
              
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={4}>
                  <TextField
                    label="Offset"
                    type="number"
                    value={axisConfig[axis]?.offset || 0}
                    onChange={(e) => handleOffsetChange(axis, e.target.value)}
                    fullWidth
                    variant="outlined"
                    size="small"
                    inputProps={{ step: 0.1 }}
                    helperText="Position offset (µm)"
                  />
                </Grid>
                
                <Grid item xs={12} sm={4}>
                  <TextField
                    label="Scale Factor"
                    type="number"
                    value={axisConfig[axis]?.scale || 1}
                    onChange={(e) => handleScaleChange(axis, e.target.value)}
                    fullWidth
                    variant="outlined"
                    size="small"
                    inputProps={{ step: 0.1, min: 0.01 }}
                    helperText="Scaling multiplier"
                  />
                </Grid>
                
                <Grid item xs={12} sm={4}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={axisConfig[axis]?.invert || false}
                        onChange={(e) => handleInvertChange(axis, e.target.checked)}
                        color="primary"
                      />
                    }
                    label="Invert Direction"
                  />
                </Grid>
              </Grid>
            </Box>
          ))}

          <Box sx={{ mt: 2, display: "flex", justifyContent: "flex-end" }}>
            <Button 
              variant="outlined" 
              onClick={handleResetAll}
              size="small"
            >
              Reset All to Defaults
            </Button>
          </Box>
        </Box>
      </AccordionDetails>
    </Accordion>
  );
};

export default AxisConfigurationMenu;
