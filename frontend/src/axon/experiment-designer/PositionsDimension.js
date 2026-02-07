import React, { useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  Button,
  ButtonGroup,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Slider,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import AddLocationIcon from "@mui/icons-material/AddLocation";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import RestartAltIcon from "@mui/icons-material/RestartAlt";

import WellSelectorCanvas, { Mode } from "../WellSelectorCanvas";
import * as wsUtils from "../WellSelectorUtils";
import InfoPopup from "../InfoPopup";

import * as wellSelectorSlice from "../../state/slices/WellSelectorSlice";
import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as positionSlice from "../../state/slices/PositionSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

/**
 * PositionsDimension - Position/well selection interface
 * 
 * Contains:
 * - Well plate / XY canvas
 * - Selection modes: Single, Area, Pattern, Imported
 * - Navigation actions (move camera, add/remove, calibrate)
 * 
 * Explicitly excludes: Channels, Z, Time, Autofocus parameters
 */
const PositionsDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  
  // Refs
  const canvasRef = useRef();
  const infoPopupRef = useRef();
  
  // Redux state
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const positionState = useSelector(positionSlice.getPositionState);
  
  // Update summary when points change
  React.useEffect(() => {
    const pointCount = experimentState.pointList?.length || 0;
    const summary = pointCount === 0 
      ? "No positions defined"
      : pointCount === 1
        ? "1 position"
        : `${pointCount} positions`;
    
    dispatch(experimentUISlice.setDimensionSummary({ 
      dimension: DIMENSIONS.POSITIONS, 
      summary 
    }));
    dispatch(experimentUISlice.setDimensionConfigured({ 
      dimension: DIMENSIONS.POSITIONS, 
      configured: pointCount > 0 
    }));
  }, [experimentState.pointList, dispatch]);

  // Mode change handler
  const handleModeChange = (mode) => {
    dispatch(wellSelectorSlice.setMode(mode));
  };

  // Layout change handler
  const handleLayoutChange = (event) => {
    const layoutName = event.target.value;
    const offsetX = wellSelectorState.layoutOffsetX || 0;
    const offsetY = wellSelectorState.layoutOffsetY || 0;
    
    let wellLayout;
    
    switch (layoutName) {
      case "Default":
        wellLayout = wsUtils.wellLayoutDefault;
        break;
      case "Heidstar 4x Histosample":
        wellLayout = wsUtils.wellLayoutDevelopment;
        break;
      case "Wellplate 384":
        wellLayout = wsUtils.generateWellLayout384({ offsetX, offsetY });
        break;
      case "Ropod":
        wellLayout = wsUtils.ropodLayout;
        break;
      default:
        return;
    }
    
    // Apply offsets
    wellLayout = wsUtils.applyLayoutOffset(wellLayout, offsetX, offsetY);
    dispatch(experimentSlice.setWellLayout(wellLayout));
  };

  // Add current position
  const handleAddCurrentPosition = () => {
    const currentX = positionState.x;
    const currentY = positionState.y;
    
    dispatch(experimentSlice.createPoint({
      x: currentX,
      y: currentY,
      name: `Position ${experimentState.pointList.length + 1}`,
      shape: ""
    }));
    
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(`Added position: X=${currentX}, Y=${currentY}`);
    }
  };

  // Calibrate offset
  const handleCalibrateOffset = () => {
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(
        "Right-click on the map where you are and select 'We are here' to calibrate the stage offset."
      );
    }
  };

  // Reset view
  const handleResetView = () => {
    canvasRef.current?.resetView();
  };

  // Reset history
  const handleResetHistory = () => {
    canvasRef.current?.resetHistory();
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Info message about Well Selector tab */}
      <Box
        sx={{
          p: 2,
          mb: 2,
          border: `1px solid ${theme.palette.info.main}`,
          borderRadius: 1,
          backgroundColor: alpha(theme.palette.info.main, 0.08),
        }}
      >
        <Typography variant="body2" sx={{ mb: 0.5, fontWeight: 600 }}>
          📍 Position Selection
        </Typography>
        <Typography variant="body2" color="textSecondary">
          To select positions on the well plate, please switch to the <strong>Well Selector</strong> tab.
          The positions you define there will appear in this experiment.
        </Typography>
      </Box>

      {/* Quick add current position */}
      <Box sx={{ mb: 2 }}>
        <Button
          size="small"
          variant="outlined"
          startIcon={<AddLocationIcon />}
          onClick={handleAddCurrentPosition}
          fullWidth
        >
          Add Current Stage Position
        </Button>
      </Box>

      {/* Position list (placeholder for save/load functionality) */}
      <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
        <Button
          size="small"
          variant="outlined"
          fullWidth
          disabled
        >
          💾 Save Positions
        </Button>
        <Button
          size="small"
          variant="outlined"
          fullWidth
          disabled
        >
          📂 Load Positions
        </Button>
      </Box>

      <Typography variant="caption" color="textSecondary" sx={{ fontStyle: "italic" }}>
        Save/Load functionality coming soon
      </Typography>

      <InfoPopup ref={infoPopupRef} />
    </Box>
  );
};

export default PositionsDimension;
