import React, { useState, useRef, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import WellSelectorCanvas, { Mode } from "./WellSelectorCanvas.js";

import * as wsUtils from "./WellSelectorUtils.js";

import InfoPopup from "./InfoPopup.js";

import * as wellSelectorSlice from "../state/slices/WellSelectorSlice.js";
import * as experimentSlice from "../state/slices/ExperimentSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import * as overviewRegSlice from "../state/slices/OverviewRegistrationSlice.js";
import apiGetOverviewOverlayData from "../backendapi/apiGetOverviewOverlayData.js";

import apiDownloadJson from "../backendapi/apiDownloadJson.js";
import fetchObjectiveControllerGetStatus from "../middleware/fetchObjectiveControllerGetStatus.js";
import LabwareSelectionPanel from "../components/LabwareSelectionPanel.jsx";

import {
  Button,
  Typography,
  Box,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@mui/material";
import PlaceIcon from "@mui/icons-material/Place";
import HighlightAltIcon from "@mui/icons-material/HighlightAlt";
import RadioButtonCheckedIcon from "@mui/icons-material/RadioButtonChecked";
import GestureIcon from "@mui/icons-material/Gesture";
import PanToolIcon from "@mui/icons-material/PanTool";
import ScatterPlotIcon from "@mui/icons-material/ScatterPlot";
import AddLocationIcon from "@mui/icons-material/AddLocation";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";

//##################################################################################
const WellSelectorComponent = () => {
  //local state
  const [wellLayoutFileList] = useState([
    "image/test.json", //TODO remove test
    "image/test1.json", //TODO remove test
  ]);

  // Pending layout switch awaiting user confirmation when pointList is non-empty
  const [pendingLayoutEvent, setPendingLayoutEvent] = useState(null);

  //child ref
  const childRef = useRef(); //canvas
  const infoPopupRef = useRef();

  //redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const positionState = useSelector(positionSlice.getPositionState);
  const overviewRegState = useSelector(
    overviewRegSlice.getOverviewRegistrationState,
  );

  // Toggle the Overview camera overlay (stitched overview image) on the plate
  // map; lazily fetch the overlay data the first time it is switched on.
  const handleToggleOverviewOverlay = async () => {
    const next = !overviewRegState.overlayEnabled;
    dispatch(overviewRegSlice.setOverlayEnabled(next));
    const slides = overviewRegState.overlayData?.slides;
    if (next && (!slides || Object.keys(slides).length === 0)) {
      try {
        const cam = overviewRegState.cameraName || "";
        const layout = overviewRegState.layoutName || "Heidstar 4x Histosample";
        const data = await apiGetOverviewOverlayData(cam, layout);
        dispatch(overviewRegSlice.setOverlayData(data));
      } catch (e) {
        // best-effort; the Overview tab can load/refresh the overlay explicitly
      }
    }
  };

  // Opening the wellplate view should refresh the objective state so the
  // current pixel size / FOV (which drives tiling, overlap and freehand step
  // sizes) is applied. The tab bar mounts this component fresh on open, so a
  // mount effect is enough.
  useEffect(() => {
    fetchObjectiveControllerGetStatus(dispatch);
  }, [dispatch]);

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
          "Draw a closed freehand region first (FREEHAND mode, click+drag).",
        );
      }
      return;
    }
    // A freehand polygon is ONE logical scan area (like an area-select
    // rectangle), so it becomes a SINGLE point-list entry whose interior scan
    // positions ride along in ``neighborPointList``.  This is what makes the
    // scan treat it as one group (one zarr/tif folder) instead of dozens of
    // separate single-tile points.
    const areaId = `freehand_${Date.now()}`;
    const cx = positions.reduce((s, p) => s + p.x, 0) / positions.length;
    const cy = positions.reduce((s, p) => s + p.y, 0) / positions.length;
    dispatch(
      experimentSlice.createPoint({
        x: cx,
        y: cy,
        name: "Freehand",
        shape: "",
        areaType: "free_scan",
        areaId,
        groupId: areaId,
        neighborPointList: positions.map((p) => ({
          x: p.x,
          y: p.y,
          z: p.z ?? 0,
          iX: 0,
          iY: 0,
        })),
      }),
    );
    childRef.current.clearFreehand && childRef.current.clearFreehand();
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(
        `Created 1 freehand region with ${positions.length} scan position(s).`,
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
    if (event.target.value === "Blank" || event.target.value === "Default") {
      wellLayout = wsUtils.wellLayoutDefault;
    } else if (event.target.value === "Heidstar 4x Histosample") {
      wellLayout = wsUtils.wellLayoutDevelopment;
    } else if (event.target.value === "Wellplate 384") {
      // Generate 384 layout with offsets
      wellLayout = wsUtils.generateWellLayout384({
        offsetX: offsetX,
        offsetY: offsetY,
      });
    } else if (event.target.value === "DEP Chip") {
      // Generate DEP Chip layout with offsets
      wellLayout = wsUtils.generateWellLayoutDEPChip({
        offsetX: offsetX,
        offsetY: offsetY,
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
          console.error(
            "-----------------------------------------------TODO impl me------------------------------------------------------------",
          );
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
    const currentZ = positionState.z;
    // Create a new point with current position
    dispatch(
      experimentSlice.createPoint({
        x: currentX,
        y: currentY,
        z: currentZ,
        name: `Position ${experimentState.pointList.length + 1}`,
        shape: "",
      }),
    );

    // Show confirmation message
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(
        `Added position: X=${currentX}, Y=${currentY}`,
      );
    }
  };

  //##################################################################################
  const handleCalibrateOffset = () => {
    // Stage offset calibration is now available via right-click context menu on the canvas.
    // Right-click on the map and select "We are here (Calibrate Offset)" to set the stage offset.
    // This uses the clicked position as the known position and transmits it to the backend
    // via the setStageOffsetAxis API (single source of truth for offsets).
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(
        "Right-click on the map where you are and select 'We are here' to calibrate the stage offset.",
      );
    }
  };

  //##################################################################################
  const handleMoveCameraSpeedXYChange = (event) => {
    dispatch(wellSelectorSlice.setMoveCameraSpeedXY(event.target.value));
    // Keep the experiment scan speed in sync so "Move Camera Speed" drives the
    // XY scan too (single source of truth shared with the Tiling tab control).
    const v = parseFloat(event.target.value);
    if (!isNaN(v) && v > 0) {
      dispatch(experimentSlice.setSpeed(v));
    }
  };

  //##################################################################################
  const handleMoveCameraSpeedZChange = (event) => {
    dispatch(wellSelectorSlice.setMoveCameraSpeedZ(event.target.value));
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
        <div />
        <FormControl>
          <InputLabel>Layout</InputLabel>
          <Select
            label="Layout"
            value={experimentState.wellLayout.name}
            onChange={handleLayoutChange}
          >
            {/* hard coded layouts */}
            <MenuItem value="Blank">Blank</MenuItem>
            <MenuItem value="Heidstar 4x Histosample">
              4 Slide Heidstar
            </MenuItem>
            <MenuItem value="Ropod">Ropod</MenuItem>
            <MenuItem value="Wellplate 384">Wellplate 384</MenuItem>
            <MenuItem value="DEP Chip">DEP Chip (8x6)</MenuItem>
            {/* online layouts */}
            {wellLayoutFileList.map((file) => (
              <MenuItem value={file}>{file}</MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* VIEW - All controls in one row */}
        <Box
          sx={{
            display: "flex",
            gap: 1,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
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

          <label
            style={{
              fontSize: "14px",
              display: "flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            <input
              type="checkbox"
              checked={wellSelectorState.showOverlap}
              onChange={handleShowOverlapChange}
            />
            Show Overlap
          </label>

          <label
            style={{
              fontSize: "14px",
              display: "flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            <input
              type="checkbox"
              checked={wellSelectorState.showShape}
              onChange={handleShowShapeChange}
            />
            Show Shape
          </label>
        </Box>
      </div>

      {/* MODE — selection tools + stage actions, with explanatory icons + tooltips */}
      <Box
        sx={{
          mb: 1.5,
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          alignItems: "center",
          gap: 1,
        }}
      >
        {/* Selection / interaction modes (active one is filled) */}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
          {[
            {
              mode: Mode.SINGLE_SELECT,
              label: "Single",
              icon: <PlaceIcon />,
              tip: "Single point — click the map to add one imaging position.",
            },
            {
              mode: Mode.AREA_SELECT,
              label: "Area",
              icon: <HighlightAltIcon />,
              tip: "Area — drag a rectangle to tile-scan a whole region.",
            },
            {
              mode: Mode.CUP_SELECT,
              label: "Well",
              icon: <RadioButtonCheckedIcon />,
              tip: "Well — click wells to image the entire well.",
            },
            {
              mode: Mode.FREEHAND_DRAW,
              label: "Freehand",
              icon: <GestureIcon />,
              tip: "Freehand — draw a closed region, then press Convert to fill it with scan points.",
            },
            {
              mode: Mode.MOVE_CAMERA,
              label: "Move",
              icon: <PanToolIcon />,
              tip: "Move camera — click the map to drive the stage to that point.",
            },
          ].map((b) => (
            <Tooltip key={b.label} title={b.tip} arrow>
              <Button
                size="small"
                startIcon={b.icon}
                variant={
                  wellSelectorState.mode === b.mode ? "contained" : "outlined"
                }
                onClick={() => handleModeChange(b.mode)}
              >
                {b.label}
              </Button>
            </Tooltip>
          ))}
        </Box>

        {/* Stage actions */}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
          {wellSelectorState.mode === Mode.FREEHAND_DRAW && (
            <Tooltip
              title="Convert the drawn freehand region into tiled scan points."
              arrow
            >
              <Button
                size="small"
                variant="outlined"
                color="secondary"
                startIcon={<ScatterPlotIcon />}
                onClick={handleConvertFreehandToPoints}
              >
                Convert
              </Button>
            </Tooltip>
          )}
          <Tooltip title="Add the current stage XYZ as a new position." arrow>
            <Button
              size="small"
              variant="outlined"
              startIcon={<AddLocationIcon />}
              onClick={() => handleAddCurrentPosition()}
            >
              Add current
            </Button>
          </Tooltip>
          <Tooltip
            title="Calibrate the stage offset: right-click the map where the camera currently is and choose 'We are here'."
            arrow
          >
            <Button
              size="small"
              variant="outlined"
              startIcon={<GpsFixedIcon />}
              onClick={() => handleCalibrateOffset()}
            >
              Calibrate
            </Button>
          </Tooltip>
          <Tooltip
            title="Show/hide the Overview camera overlay (stitched overview image) on the plate map."
            arrow
          >
            <Button
              size="small"
              variant={
                overviewRegState.overlayEnabled ? "contained" : "outlined"
              }
              color="secondary"
              onClick={handleToggleOverviewOverlay}
            >
              {overviewRegState.overlayEnabled ? "Overlay on" : "Overlay off"}
            </Button>
          </Tooltip>
        </Box>
      </Box>

      {/* MOVE CAMERA speed controls – only shown when MOVE_CAMERA mode is active */}
      {wellSelectorState.mode === Mode.MOVE_CAMERA && (
        <Box
          sx={{
            display: "flex",
            gap: 2,
            alignItems: "center",
            flexWrap: "wrap",
            mb: 1,
            mt: 1,
          }}
        >
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Move Camera Speed:
          </Typography>
          <TextField
            label="XY Speed (µm/s)"
            type="number"
            size="small"
            value={wellSelectorState.moveCameraSpeedXY ?? 20000}
            onChange={handleMoveCameraSpeedXYChange}
            inputProps={{ min: 1, step: 1000 }}
            error={
              (parseFloat(wellSelectorState.moveCameraSpeedXY) || 0) > 20000
            }
            helperText={
              (parseFloat(wellSelectorState.moveCameraSpeedXY) || 0) > 20000
                ? "⚠ >20000 µm/s is highly unreliable (may lose steps/accuracy)"
                : " "
            }
            sx={{ width: 230 }}
          />
          <TextField
            label="Z Speed (µm/s)"
            type="number"
            size="small"
            value={wellSelectorState.moveCameraSpeedZ ?? 1000}
            onChange={handleMoveCameraSpeedZChange}
            inputProps={{ min: 1, step: 100 }}
            sx={{ width: 140 }}
          />
        </Box>
      )}

      <InfoPopup ref={infoPopupRef} />

      {/* Confirm before swapping layout (clears existing points) */}
      <Dialog
        open={pendingLayoutEvent != null}
        onClose={handleCancelLayoutChange}
      >
        <DialogTitle>Switch layout?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Switching to <strong>{pendingLayoutEvent?.target?.value}</strong>{" "}
            will remove all {experimentState?.pointList?.length || 0} point(s)
            from the current experiment, since their coordinates are tied to the
            old layout. Continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelLayoutChange}>Cancel</Button>
          <Button
            onClick={handleConfirmLayoutChange}
            color="error"
            variant="contained"
          >
            Switch and clear
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

//##################################################################################
export default WellSelectorComponent;
