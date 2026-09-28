import React, { useState, useRef, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import WellSelectorCanvas, { Mode } from "./WellSelectorCanvas.js";

import * as wsUtils from "./WellSelectorUtils.js";

import InfoPopup from "./InfoPopup.js";

import * as wellSelectorSlice from "../state/slices/WellSelectorSlice.js";
import * as experimentSlice from "../state/slices/ExperimentSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import * as overviewRegSlice from "../state/slices/OverviewRegistrationSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as stageMapSlice from "../state/slices/StageMapSlice.js";
import apiGetOverviewOverlayData from "../backendapi/apiGetOverviewOverlayData.js";
import { apiStageMapGetTiles } from "../backendapi/apiStageMapController.js";

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
  Tooltip as MuiTooltip,
} from "@mui/material";
import PlaceIcon from "@mui/icons-material/Place";
import HighlightAltIcon from "@mui/icons-material/HighlightAlt";
import RadioButtonCheckedIcon from "@mui/icons-material/RadioButtonChecked";
import GestureIcon from "@mui/icons-material/Gesture";
import PanToolIcon from "@mui/icons-material/PanTool";
import ScatterPlotIcon from "@mui/icons-material/ScatterPlot";
import AddLocationIcon from "@mui/icons-material/AddLocation";
import GpsFixedIcon from "@mui/icons-material/GpsFixed";
import BlurLinearIcon from "@mui/icons-material/BlurLinear";
import LayersClearIcon from "@mui/icons-material/LayersClear";
import {
  apiStageMapClear,
  apiStageMapStartPrescan,
  apiStageMapStopPrescan,
} from "../backendapi/apiStageMapController";

// The tool buttons wrap into several rows and some tips are long, so a tip
// must not pop up while the pointer merely crosses a button, must close as soon
// as it leaves (not hover-trap over the next row), and must stay narrow.
const Tooltip = (props) => (
  <MuiTooltip
    enterDelay={700}
    enterNextDelay={700}
    disableInteractive
    slotProps={{ tooltip: { sx: { maxWidth: 260 } } }}
    {...props}
  />
);

//##################################################################################
const WellSelectorComponent = () => {
  //local state
  const [wellLayoutFileList] = useState([
    "image/test.json", //TODO remove test
    "image/test1.json", //TODO remove test
  ]);

  // Pending layout switch awaiting user confirmation when pointList is non-empty

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
  const stageMapState = useSelector(stageMapSlice.getStageMapState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);

  // The backend says whether a prescan is running (status over the socket);
  // the local flag only bridges the click until that first status arrives.
  // Keeping this purely local is why the button used to stay on "Stop" for
  // ever after a prescan finished on its own.
  const [prescanPending, setPrescanPending] = useState(false);
  const prescanRunning =
    Boolean(stageMapState?.status?.prescanRunning) || prescanPending;
  useEffect(() => {
    if (stageMapState?.status?.prescanRunning !== undefined) setPrescanPending(false);
  }, [stageMapState?.status?.prescanRunning]);

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

  // Toggle drawing of collected stage-map scan tiles (e.g. a 4x overview
  // tilescan) on the plate map; lazily fetch the tiles from the backend the
  // first time it is switched on (e.g. after a page reload).
  const handleToggleStageMapOverlay = async () => {
    const next = !stageMapState.showOnWellplate;
    dispatch(stageMapSlice.setShowOnWellplate(next));
    if (next && (stageMapState.tiles || []).length === 0) {
      try {
        const data = await apiStageMapGetTiles(0, true);
        if (Array.isArray(data?.tiles) && data.tiles.length > 0) {
          dispatch(stageMapSlice.setTiles(data.tiles));
        } else if (infoPopupRef.current) {
          infoPopupRef.current.showMessage(
            "No stage map tiles collected yet — record a scan in the Stage Map app first.",
          );
        }
      } catch (e) {
        // best-effort; the Stage Map app can collect/reload tiles explicitly
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

  // A freehand region keeps its polygon, so when the objective changes (new
  // FOV) or the area overlap is edited, re-tile it on the new pitch. Without
  // this the positions computed for a 4x field stay frozen when scanning at
  // 20x — a sparse grid with gaps between the fields.
  const fovXForTiling = objectiveState?.fovX || 0;
  const fovYForTiling = objectiveState?.fovY || 0;
  const overlapForTiling = wellSelectorState.areaSelectOverlap || 0;
  useEffect(() => {
    if (fovXForTiling <= 0 || fovYForTiling <= 0) return;
    experimentState.pointList.forEach((point, index) => {
      if (!Array.isArray(point.polygon) || point.polygon.length < 3) return;
      const positions = wsUtils.generatePolygonScanPositions(
        point.polygon,
        fovXForTiling,
        fovYForTiling,
        overlapForTiling,
      );
      if (positions.length === 0) return;
      const current = point.neighborPointList || [];
      const unchanged =
        current.length === positions.length &&
        current.every(
          (n, i) =>
            Math.abs(n.x - positions[i].x) < 1e-6 &&
            Math.abs(n.y - positions[i].y) < 1e-6,
        );
      if (unchanged) return;
      dispatch(
        experimentSlice.replacePoint({
          index,
          newPoint: {
            ...point,
            neighborPointList: positions.map((p) => ({
              x: p.x,
              y: p.y,
              z: point.z ?? 0,
              iX: p.iX,
              iY: p.iY,
            })),
          },
        }),
      );
    });
  }, [dispatch, experimentState.pointList, fovXForTiling, fovYForTiling, overlapForTiling]);

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
    // Keep the outline itself, not just this tiling of it: switching the
    // objective changes the FOV, and the region must then be re-tiled on the
    // new pitch (see the re-tiling effect above and CoordinateCalculator).
    const polygon = childRef.current.getFreehandPolygon
      ? childRef.current.getFreehandPolygon()
      : [];
    dispatch(
      experimentSlice.createPoint({
        x: cx,
        y: cy,
        name: "Freehand",
        shape: "",
        areaType: "free_scan",
        areaId,
        groupId: areaId,
        polygon,
        neighborPointList: positions.map((p) => ({
          x: p.x,
          y: p.y,
          z: p.z ?? 0,
          iX: p.iX ?? 0,
          iY: p.iY ?? 0,
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
  // Changing the plate redraws it and KEEPS the points: they are absolute
  // stage micrometers, so a new plate definition does not invalidate their
  // coordinates. Only the per-point well annotations belonged to the old plate.
  const handleLayoutChange = (event) => {
    if (event?.target?.value !== experimentState?.wellLayout?.name) {
      dispatch(wellSelectorSlice.clearSelectedWellIds());
      dispatch(wellSelectorSlice.clearConditionLabels());
    }
    applyLayoutChange(event);
  };

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

  // ── Fast prescan overlay ─────────────────────────────────────────────
  // Sweeps the bounding box of whatever is currently selected and drops the
  // result onto the map through the existing stage-map tile overlay, so there
  // is something real to trace over. Illumination and exposure are used as the
  // operator left them.
  const selectionBounds = () => {
    const xs = [];
    const ys = [];
    (experimentState.pointList || []).forEach((p) => {
      xs.push(p.x - (p.rectMinusX || 0), p.x + (p.rectPlusX || 0));
      ys.push(p.y - (p.rectMinusY || 0), p.y + (p.rectPlusY || 0));
      (p.neighborPointList || []).forEach((n) => {
        xs.push(n.x);
        ys.push(n.y);
      });
    });
    if (xs.length === 0) return null;
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  };

  // Line spacing is the tile spacing already in use (objective FOV minus the
  // configured overlap) and the sweep speed is the stage speed already set —
  // a prescan should cover the sample the same way the scan will.
  const prescanDy = Math.max(
    1,
    Math.round(
      (objectiveState?.fovY || 0) * (1 - (wellSelectorState.areaSelectOverlap || 0)),
    ) || 500,
  );
  // Exposure pitch along the sweep: the same X tile spacing the scan uses, so
  // one exposure lands per tile step and the strip has no gaps and no overlap.
  const prescanDx = Math.max(
    1,
    Math.round(
      (objectiveState?.fovX || 0) * (1 - (wellSelectorState.areaSelectOverlap || 0)),
    ) || 0,
  );
  const prescanSpeed = Math.max(
    1,
    parseFloat(wellSelectorState.moveCameraSpeedXY) || 20000,
  );

  // Slot 0/1 with the lower magnification — a prescan wants the widest field.
  const lowMagSlot =
    (objectiveState?.magnification2 || 0) > 0 &&
    (objectiveState?.magnification2 || 0) < (objectiveState?.magnification1 || 0)
      ? 1
      : 0;

  const handleStartPrescan = () => {
    const bounds = selectionBounds();
    if (!bounds) {
      infoPopupRef.current?.showMessage(
        "Select an area or place positions first — the prescan sweeps their bounding box.",
      );
      return;
    }

    // Moving the turret is a physical change, so ask before doing it. The
    // backend restores the objective, X, Y and Z afterwards either way.
    let objectiveSlot;
    const current = objectiveState?.currentObjective;
    if (current != null && current !== lowMagSlot) {
      const lowMag =
        lowMagSlot === 0
          ? objectiveState?.magnification1
          : objectiveState?.magnification2;
      // eslint-disable-next-line no-alert
      if (
        window.confirm(
          `A prescan is quicker and covers more with the lower magnification` +
            `${lowMag ? ` (${lowMag}x)` : ""}.\n\n` +
            `Switch the objective for the prescan? Its own saved focus is applied, ` +
            `and the objective and stage position are restored afterwards.\n\n` +
            `Cancel to prescan with the current objective.`,
        )
      ) {
        objectiveSlot = lowMagSlot;
      }
    }

    // The strips arrive over the socket line by line; make sure they are visible.
    dispatch(stageMapSlice.setShowOnWellplate(true));
    setPrescanPending(true);
    apiStageMapStartPrescan({
      ...bounds,
      dx: prescanDx,
      dy: prescanDy,
      speedX: prescanSpeed,
      ...(objectiveSlot === undefined ? {} : { objectiveSlot }),
    })
      .then((res) => {
        if (res?.success) {
          infoPopupRef.current?.showMessage(`Prescan running — ${res.lines} line(s)`);
        } else {
          setPrescanPending(false);
          infoPopupRef.current?.showMessage(res?.error || "Prescan failed to start");
        }
      })
      .catch(() => {
        setPrescanPending(false);
        infoPopupRef.current?.showMessage("Prescan failed to start");
      });
  };

  const handleStopPrescan = () => {
    apiStageMapStopPrescan().finally(() => setPrescanPending(false));
  };

  // Throw the overlay away — backend tiles and the local copy — so the next
  // prescan starts on a clean map instead of layering over the old one.
  const handleClearPrescan = () => {
    dispatch(stageMapSlice.clearTiles());
    apiStageMapClear()
      .then(() => infoPopupRef.current?.showMessage("Prescan overlay cleared"))
      .catch(() => infoPopupRef.current?.showMessage("Could not clear the overlay"));
  };

  // ── Boundary from dropped points (freehand mode) ─────────────────────
  const handleAddCurrentAsVertex = () => {
    childRef.current?.addFreehandVertex?.({
      x: positionState.x,
      y: positionState.y,
    });
    infoPopupRef.current?.showMessage(
      `Vertex at X=${positionState.x}, Y=${positionState.y}`,
    );
  };

  const handleCloseFreehand = () => {
    const closed = childRef.current?.closeFreehand?.();
    infoPopupRef.current?.showMessage(
      closed ? "Region closed — press Convert to tile it."
             : "Need at least 3 vertices to close a region.",
    );
  };

  const handleWrapPoints = () => {
    const n = childRef.current?.wrapPointsIntoFreehand?.() || 0;
    infoPopupRef.current?.showMessage(
      n ? `Wrapped ${n} boundary vertices around the existing positions.`
        : "Need at least 3 positions to wrap.",
    );
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
            <>
              <Tooltip
                title="Add the current stage XY as a boundary vertex — drive to a tissue edge, tap, repeat."
                arrow
              >
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<AddLocationIcon />}
                  onClick={handleAddCurrentAsVertex}
                >
                  Add vertex
                </Button>
              </Tooltip>
              <Tooltip
                title="Close the boundary you have been tapping out."
                arrow
              >
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleCloseFreehand}
                >
                  Close region
                </Button>
              </Tooltip>
              <Tooltip
                title="Wrap the existing positions in a boundary (convex hull)."
                arrow
              >
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleWrapPoints}
                >
                  Wrap points
                </Button>
              </Tooltip>
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
            </>
          )}
          <Tooltip
            title={`Sweep the selected area fast and lay the result under the map, so you can see the tissue and trace it. Uses the tile spacing (${prescanDy} µm), the stage speed (${prescanSpeed} µm/s) and the illumination you already have set.`}
            arrow
          >
            <Button
              size="small"
              variant={prescanRunning ? "contained" : "outlined"}
              color="info"
              startIcon={<BlurLinearIcon />}
              onClick={prescanRunning ? handleStopPrescan : handleStartPrescan}
            >
              {prescanRunning ? "Stop prescan" : "Prescan"}
            </Button>
          </Tooltip>
          {(stageMapState?.tiles?.length || 0) > 0 && (
            <Tooltip title="Discard the overlay so the next prescan starts on a clean map." arrow>
              <Button
                size="small"
                variant="outlined"
                color="warning"
                startIcon={<LayersClearIcon />}
                onClick={handleClearPrescan}
                disabled={prescanRunning}
              >
                Clear overlay
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
          <Tooltip
            title="Show/hide the collected stage-map scan tiles (e.g. a low-magnification overview scan) at their true stage positions, so a ROI for a higher-magnification scan can be selected on the sample image."
            arrow
          >
            <Button
              size="small"
              variant={stageMapState.showOnWellplate ? "contained" : "outlined"}
              color="secondary"
              onClick={handleToggleStageMapOverlay}
            >
              {stageMapState.showOnWellplate ? "Tiles on" : "Tiles off"}
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

    </div>
  );
};

//##################################################################################
export default WellSelectorComponent;
