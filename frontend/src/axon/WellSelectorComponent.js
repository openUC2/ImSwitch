import React, { useState, useRef, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import WellSelectorCanvas, { Mode } from "./WellSelectorCanvas.js";

import * as wsUtils from "./WellSelectorUtils.js";

import InfoPopup from "./InfoPopup.js";


import * as wellSelectorSlice from "../state/slices/WellSelectorSlice.js";
import * as experimentSlice from "../state/slices/ExperimentSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js"; 
import * as overviewRegSlice from '../state/slices/OverviewRegistrationSlice.js';

import apiDownloadJson from "../backendapi/apiDownloadJson.js";
import apiGetOverviewOverlayData from "../backendapi/apiGetOverviewOverlayData.js";
import OverviewRegistrationWizard from "./OverviewRegistrationWizard.js";

import {
  Button,
  Typography,
  Box,
  Input,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormHelperText,
  FormControlLabel,
  ButtonGroup,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
} from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

//##################################################################################
const WellSelectorComponent = () => {
  //local state
  const [wellLayoutFileList, setWellLayoutFileList] = useState([
    "image/test.json",//TODO remove test
    "image/test1.json",//TODO remove test
  ]);

  //child ref
  const childRef = useRef();//canvas 
  const infoPopupRef = useRef();

  //redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const positionState = useSelector(positionSlice.getPositionState); 
  const overviewRegState = useSelector(overviewRegSlice.getOverviewRegistrationState);


  //##################################################################################
  const handleOpenOverviewWizard = () => {
    dispatch(overviewRegSlice.setWizardOpen(true));
  };

  const handleOverlayToggle = (event) => {
    dispatch(overviewRegSlice.setOverlayEnabled(event.target.checked));
    // Load overlay data if enabling and not yet loaded
    if (event.target.checked && (!overviewRegState.overlayData || !overviewRegState.overlayData.slides || Object.keys(overviewRegState.overlayData.slides || {}).length === 0)) {
      loadOverlayData();
    }
  };

  const handleOverlayOpacityChange = (event, newValue) => {
    dispatch(overviewRegSlice.setOverlayOpacity(newValue));
  };

  const loadOverlayData = async () => {
    try {
      const data = await apiGetOverviewOverlayData(
        overviewRegState.cameraName,
        overviewRegState.layoutName || experimentState.wellLayout.name
      );
      dispatch(overviewRegSlice.setOverlayData(data));
    } catch (e) {
      console.warn("Failed to load overlay data:", e);
    }
  };


  //##################################################################################
  const handleModeChange = (mode) => {
    // Update Redux state
    dispatch(wellSelectorSlice.setMode(mode)); // Update Redux state
  };

  //##################################################################################
  const handleResetView = () => {
    //call child methode
    childRef.current.resetView();
  };

  //##################################################################################
  const handleResetHistory = () => {
    //call child method to reset position history
    childRef.current.resetHistory();
  };

  //##################################################################################
  const handleShowOverlapChange = (event) => {
    dispatch(wellSelectorSlice.setShowOverlap(event.target.checked));
  };

  //##################################################################################
  const handleShowShapeChange = (event) => {
    dispatch(wellSelectorSlice.setShowShape(event.target.checked));
  };

  //##################################################################################
  const handleLayoutChange = (event) => {
    console.log("handleLayoutChange");
    //select layout
    let wellLayout; // = wsUtils.wellLayoutDefault;

    // Get current offsets from Redux state
    const offsetX = wellSelectorState.layoutOffsetX || 0;
    const offsetY = wellSelectorState.layoutOffsetY || 0;

    //check defaults
    if (event.target.value === "Default") {
      wellLayout = wsUtils.wellLayoutDefault;
    } else if (event.target.value === "Heidstar 4x Histosample") {
      wellLayout = wsUtils.wellLayoutDevelopment;
   } else if (event.target.value === "Wellplate 384") {
      // Generate 384 layout with offsets
      wellLayout = wsUtils.generateWellLayout384({
        offsetX: offsetX,
        offsetY: offsetY
      });
    } else if (event.target.value === "DEP Chip") {
      // Generate DEP Chip layout with offsets
      wellLayout = wsUtils.generateWellLayoutDEPChip({
        offsetX: offsetX,
        offsetY: offsetY
      });
    } else if (event.target.value === "Ropod") {
      wellLayout = wsUtils.ropodLayout;      
    } else {
      //donwload layout
      apiDownloadJson(event.target.value) // Pass the JSON file path
        .then((data) => {
          console.log("apiDownloadJson", data);
          //handle layout
          //TODO
        //set popup
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("TODO impl me");
        } 
          console.error("-----------------------------------------------TODO impl me------------------------------------------------------------");
        })
        .catch((err) => {
          //handle error if needed
          console.log(err);
        });

      return;
    }

    // Apply offsets to the layout
    wellLayout = wsUtils.applyLayoutOffset(wellLayout, offsetX, offsetY);


    //set new layout
    dispatch(experimentSlice.setWellLayout(wellLayout));
  };

  //##################################################################################
  const handleAddCurrentPosition = () => {
    // Get current position from Redux state
    const currentX = positionState.x;
    const currentY = positionState.y;
    
    // Create a new point with current position
    dispatch(experimentSlice.createPoint({
      x: currentX,
      y: currentY,
      name: `Position ${experimentState.pointList.length + 1}`,
      shape: ""
    }));
    
    // Show confirmation message
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(`Added position: X=${currentX}, Y=${currentY}`);
    }
  };

  //##################################################################################
  const handleCalibrateOffset = () => {
    // Stage offset calibration is now available via right-click context menu on the canvas.
    // Right-click on the map and select "We are here (Calibrate Offset)" to set the stage offset.
    // This uses the clicked position as the known position and transmits it to the backend
    // via the setStageOffsetAxis API (similar to StageOffsetCalibrationController.jsx).
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage("Right-click on the map where you are and select 'We are here' to calibrate the stage offset.");
    }
  }

  //##################################################################################
  const handleLayoutOffsetXChange = (event) => {
    const value = parseFloat(event.target.value);
    dispatch(wellSelectorSlice.setLayoutOffsetX(value));
    
    // Re-apply the current layout with new offset
    handleLayoutChange({ target: { value: experimentState.wellLayout.name } });
  };

  //##################################################################################
  const handleLayoutOffsetYChange = (event) => {
    const value = parseFloat(event.target.value);
    dispatch(wellSelectorSlice.setLayoutOffsetY(value));
    
    // Re-apply the current layout with new offset
    handleLayoutChange({ target: { value: experimentState.wellLayout.name } });
  };

  //##################################################################################
  const handleAreaSelectSnakescanChange = (event) => {
    dispatch(wellSelectorSlice.setAreaSelectSnakescan(event.target.checked));
  };

  //##################################################################################
  const handleAreaSelectOverlapChange = (event) => {
    const value = parseFloat(event.target.value);
    dispatch(wellSelectorSlice.setAreaSelectOverlap(value));
  };

  //##################################################################################
  const handleCupSelectShapeChange = (event) => {
    dispatch(wellSelectorSlice.setCupSelectShape(event.target.value));
  };

  //##################################################################################
  const handleCupSelectOverlapChange = (event) => {
    const value = parseFloat(event.target.value);
    dispatch(wellSelectorSlice.setCupSelectOverlap(value));
  };

  //##################################################################################
  return (
    <div style={{ border: "0px solid #eee", padding: "10px" }}>
      
        {/* LAYOUT */}
            {/* WellSelectorComponent with mode passed as prop width: "100%", height: "100%", display: "block"*/}
            <WellSelectorCanvas ref={childRef} style={{}} />

{/* PARAMETER*/}
{/* Add a little spacer between the wellselector and layout */}
<div style={{ height: "16px" }} />

{/* PARAMETER */}
{/* Add a little spacer between the wellselector and layout */}
<div style={{ marginBottom: "15px" }}>
<div/>
        <FormControl>
          <InputLabel>Layout</InputLabel>
          <Select
            label="Layout"
            value={experimentState.wellLayout.name}
            onChange={handleLayoutChange}
          >
            {/* current layout */}
            <MenuItem
              style={{ backgroundColor: "lightblue" }}
              value={experimentState.wellLayout.name}
            >
              {experimentState.wellLayout.name}
            </MenuItem>
            {/* hard coded layouts */}
            <MenuItem value="Default">Default</MenuItem>
            <MenuItem value="Heidstar 4x Histosample">4 Slide Heidstar</MenuItem>
            <MenuItem value="Ropod">Ropod</MenuItem>  
            <MenuItem value="Wellplate 384">Wellplate 384</MenuItem>
            <MenuItem value="DEP Chip">DEP Chip (8x6)</MenuItem>
            {/* online layouts */}
            {wellLayoutFileList.map((file, index) => (
              <MenuItem value={file}>{file}</MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* VIEW - All controls in one row */}
        <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
          <Button 
            variant="contained" 
            size="small"
            onClick={() => handleResetView()}
          >
            Reset View
          </Button>

          <Button 
            variant="contained" 
            size="small"
            onClick={() => handleResetHistory()}
          >
            Reset History
          </Button>

          <label style={{ fontSize: "14px", display: "flex", alignItems: "center", gap: "4px" }}>
            <input
              type="checkbox"
              checked={wellSelectorState.showOverlap}
              onChange={handleShowOverlapChange}
            />
            Show Overlap
          </label>

          <label style={{ fontSize: "14px", display: "flex", alignItems: "center", gap: "4px" }}>
            <input
              type="checkbox"
              checked={wellSelectorState.showShape}
              onChange={handleShowShapeChange}
            />
            Show Shape
          </label>
        </Box>

      </div>

      {/* MODE */}
      <div
        style={{
          marginBottom: "10px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <ButtonGroup>
          <Button
            variant="contained"
            style={{}}
            onClick={() => handleModeChange(Mode.SINGLE_SELECT)}
            disabled={wellSelectorState.mode == Mode.SINGLE_SELECT}
          >
            SINGLE select
          </Button>

          <Button
            variant="contained"
            style={{}}
            onClick={() => handleModeChange(Mode.AREA_SELECT)}
            disabled={wellSelectorState.mode == Mode.AREA_SELECT}
          >
            AREA select
          </Button>

          <Button
            variant="contained"
            style={{}}
            onClick={() => handleModeChange(Mode.CUP_SELECT)}
            disabled={wellSelectorState.mode == Mode.CUP_SELECT}
          >
            Well select
          </Button>

          <Button
            variant="contained"
            style={{}}
            onClick={() => handleModeChange(Mode.MOVE_CAMERA)}
            disabled={wellSelectorState.mode == Mode.MOVE_CAMERA}
          >
            MOVE CAMERA
          </Button>

          <Button
            variant="contained"
            style={{}}
            onClick={() => handleAddCurrentPosition()}
          >
            ADD CURRENT POSITION
          </Button>

          <Button
            variant="contained"
            style={{}}
            onClick={() => handleCalibrateOffset()}
          >
            CALIBRATE OFFSET
          </Button>
        </ButtonGroup>
      </div>

      <InfoPopup ref={infoPopupRef}/>

      {/* Overview Camera Overlay Controls */}
      <Accordion sx={{ mt: 1 }} defaultExpanded={false}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="body2">Overview Camera Overlay</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap", mb: 1 }}>
            <Button
              variant="contained"
              size="small"
              onClick={handleOpenOverviewWizard}
            >
              Overview Overlay Wizard
            </Button>
            <label style={{ fontSize: "14px", display: "flex", alignItems: "center", gap: "4px" }}>
              <input
                type="checkbox"
                checked={overviewRegState.overlayEnabled}
                onChange={handleOverlayToggle}
              />
              Show Overlay
            </label>
          </Box>
          {overviewRegState.overlayEnabled && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 2, mt: 1 }}>
              <Typography variant="caption" sx={{ minWidth: 60 }}>Opacity:</Typography>
              <Slider
                value={overviewRegState.overlayOpacity}
                onChange={handleOverlayOpacityChange}
                min={0}
                max={1}
                step={0.05}
                size="small"
                sx={{ maxWidth: 200 }}
              />
              <Typography variant="caption">{Math.round(overviewRegState.overlayOpacity * 100)}%</Typography>
            </Box>
          )}
        </AccordionDetails>
      </Accordion>

      {/* Overview Registration Wizard Dialog */}
      <OverviewRegistrationWizard />
    </div>
  );
};

//##################################################################################
export default WellSelectorComponent;
