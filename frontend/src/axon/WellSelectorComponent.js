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
import LabwareSelectionPanel from "../components/LabwareSelectionPanel.jsx";

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
  Slider,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

//##################################################################################
const WellSelectorComponent = () => {
  //local state
  const [wellLayoutFileList, setWellLayoutFileList] = useState([
    "image/test.json",//TODO remove test
    "image/test1.json",//TODO remove test
  ]);

  // Pending layout switch awaiting user confirmation when pointList is non-empty
  const [pendingLayoutEvent, setPendingLayoutEvent] = useState(null);

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

  // Convert the current freehand polygon (drawn on the canvas) into
  // experiment scan points using the current FOV and area-scan overlap.
  const handleConvertFreehandToPoints = () => {
    if (!childRef.current || !childRef.current.generateFreehandScanPositions) {
      if (infoPopupRef.current) {
        infoPopupRef.current.showMessage("Freehand drawing not available.");
      }
      return;
    }
    const overlap = wellSelectorState.areaSelectOverlap || 0;
    const positions = childRef.current.generateFreehandScanPositions(overlap);
    if (!positions || positions.length === 0) {
      if (infoPopupRef.current) {
        infoPopupRef.current.showMessage(
          "Draw a closed freehand region first (FREEHAND mode, click+drag)."
        );
      }
      return;
    }
    // All points generated from a single freehand polygon belong to one
    // logical scan area, so they share the same ``areaId`` (mirrors the
    // grouping that area-select / labware sub-positions use). Downstream
    // writers will store these tiles in a common zarr/tif folder.
    const areaId = `freehand_${Date.now()}`;
    positions.forEach((p, idx) => {
      dispatch(
        experimentSlice.createPoint({
          x: p.x,
          y: p.y,
          name: `Freehand_${idx + 1}`,
          shape: "",
          areaType: "free_scan",
          areaId,
          groupId: areaId,
        })
      );
    });
    childRef.current.clearFreehand && childRef.current.clearFreehand();
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(
        `Created ${positions.length} freehand scan point(s).`
      );
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
    // Warn the user before swapping layouts when there are existing points,
    // since their well coordinates will not match the new plate.
    const newName = event?.target?.value;
    if (
      newName &&
      newName !== experimentState?.wellLayout?.name &&
      (experimentState?.pointList?.length || 0) > 0
    ) {
      setPendingLayoutEvent({ target: { value: newName } });
      return;
    }
    applyLayoutChange(event);
  };

  const handleConfirmLayoutChange = () => {
    const evt = pendingLayoutEvent;
    setPendingLayoutEvent(null);
    if (!evt) return;
    dispatch(experimentSlice.setPointList([]));
    dispatch(wellSelectorSlice.clearSelectedWellIds());
    dispatch(wellSelectorSlice.clearConditionLabels());
    applyLayoutChange(evt);
  };

  const handleCancelLayoutChange = () => setPendingLayoutEvent(null);

  const applyLayoutChange = (event) => {
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

        {/* Opentrons-style labware selection (loadName + well chips + condition labels) */}
        <LabwareSelectionPanel defaultExpanded={false} />

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
              style={{ backgroundColor: "primary.main" }}
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
            onClick={() => handleModeChange(Mode.FREEHAND_DRAW)}
            disabled={wellSelectorState.mode == Mode.FREEHAND_DRAW}
          >
            FREEHAND DRAW
          </Button>

          <Button
            variant="contained"
            style={{}}
            onClick={handleConvertFreehandToPoints}
          >
            CONVERT FREEHAND TO SCAN POINTS
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

      {/* Overview Registration Wizard Dialog */}
      <OverviewRegistrationWizard />

      {/* Confirm before swapping layout (clears existing points) */}
      <Dialog
        open={pendingLayoutEvent != null}
        onClose={handleCancelLayoutChange}
      >
        <DialogTitle>Switch layout?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Switching to <strong>{pendingLayoutEvent?.target?.value}</strong> will
            remove all {experimentState?.pointList?.length || 0} point(s) from the
            current experiment, since their coordinates are tied to the old
            layout. Continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelLayoutChange}>Cancel</Button>
          <Button onClick={handleConfirmLayoutChange} color="error" variant="contained">
            Switch and clear
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

//##################################################################################
export default WellSelectorComponent;
