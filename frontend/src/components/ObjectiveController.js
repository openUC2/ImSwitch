import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Paper, Grid, Button, Typography, TextField, Box } from "@mui/material";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import ObjectiveCalibrationWizard from "./ObjectiveCalibrationWizard";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import { useTheme } from "@mui/material/styles";

import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";
import apiObjectiveControllerSetPositions from "../backendapi/apiObjectiveControllerSetPositions.js";
import apiObjectiveControllerCalibrateObjective from "../backendapi/apiObjectiveControllerCalibrateObjective.js";
import apiObjectiveControllerGetCurrentObjective from "../backendapi/apiObjectiveControllerGetCurrentObjective.js";
import apiObjectiveControllerMoveToObjective from "../backendapi/apiObjectiveControllerMoveToObjective.js";
import apiObjectiveControllerGetStatus from "../backendapi/apiObjectiveControllerGetStatus.js";
import apiPositionerControllerGetPositions from "../backendapi/apiPositionerControllerGetPositions.js";
import apiSettingsControllerGetDetectorNames from "../backendapi/apiSettingsControllerGetDetectorNames.js";
import apiObjectiveControllerSetObjectiveParameters from "../backendapi/apiObjectiveControllerSetObjectiveParameters.js";

import fetchObjectiveControllerGetStatus from "../middleware/fetchObjectiveControllerGetStatus.js";
import fetchObjectiveControllerGetCurrentObjective from "../middleware/fetchObjectiveControllerGetCurrentObjective.js";

const ExtendedObjectiveController = () => {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  //redux dispatcher
  const dispatch = useDispatch();
  const theme = useTheme(); // get MUI theme for color mode

  // Access global Redux state
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  ////const State = useSelector(Slice.getState);

  // Access state from Redux instead of local state
  const currentA = objectiveState.currentA;
  const currentZ = objectiveState.currentZ;
  const imageUrls = objectiveState.imageUrls;
  const detectors = objectiveState.detectors;
  const manualZ0 = objectiveState.manualZ0;
  const manualZ1 = objectiveState.manualZ1;

  // Local state for wizard
  const [wizardOpen, setWizardOpen] = useState(false);

  // Local state for editable objective metadata
  const [editMeta, setEditMeta] = useState({ 0: {}, 1: {} });

  // Get positions from Redux instead of local state
  const positionState = useSelector(positionSlice.getPositionState);
  const positions = {
    X: positionState.x,
    Y: positionState.y,
    Z: positionState.z,
    A: positionState.a,
  };

  // Remove all WebSocket handlers - now handled by WebSocketHandler.js
  // - sigObjectiveChanged: Already handled in WebSocketHandler -> objectiveSlice
  // - sigUpdateImage: Already handled in WebSocketHandler -> liveStreamSlice
  // - sigUpdateMotorPosition: Already handled in WebSocketHandler -> positionSlice

  useEffect(() => {
    //fetch current objective
    fetchObjectiveControllerGetCurrentObjective(dispatch);
    //refresh status
    refreshStatus();
    // Get detector names
    apiSettingsControllerGetDetectorNames()
      .then((data) => {
        dispatch(objectiveSlice.setDetectors(data));
      })
      .catch((err) => {
        console.log("Failed to fetch detector names", err); // Handle the error
      });
  }, [hostIP, hostPort]); // on host ip/port change

  // Fetch objective status (x0 and x1) from backend
  const refreshStatus = () => {
    //request fetch status
    fetchObjectiveControllerGetStatus(dispatch);
  };

  // Calibrate objective (home) and update state
  const handleCalibrate = () => {
    //request calibrate
    apiObjectiveControllerCalibrateObjective()
      .then((data) => {
        console.info("Calibrate response");
        //fetch current objective
        fetchObjectiveControllerGetCurrentObjective(dispatch);
      })
      .catch((err) => {
        console.error("Failed to calibrate the objective"); // Handle the error
      });
  };

  // Switch objective (slot should be 0 or 1)
  const handleSwitchObjective = (slot, skipZ) => {
    // Warn user if the target positions have not been configured yet
    const x0 = objectiveState.posX0;
    const x1 = objectiveState.posX1;
    const positionsConfigured =
      x0 !== null && x0 !== undefined && x0 !== 0 &&
      x1 !== null && x1 !== undefined && x1 !== 0;
    if (!positionsConfigured) {
      alert(
        "Objective positions (X0 / X1) are not configured yet.\n" +
        "Please use the Calibration Wizard to set them before switching."
      );
      // return;
    }
    apiObjectiveControllerMoveToObjective(slot, skipZ)
      .then((data) => {
        dispatch(objectiveSlice.setCurrentObjective(slot)); //setCurrentObjective(slot);
        refreshStatus();
      })
      .catch((err) => {
        console.error(`Error switching to objective ${slot}:`, err);
      });
  };

  const movePositioner = (dist, axis = "A") => {
    apiPositionerControllerMovePositioner({
      axis,
      dist,
      isAbsolute: false,
      isBlocking: false,
    })
      .then((r) => console.log(`Move ${axis} by ${dist}:`, r))
      .catch((e) => console.log(`Move ${axis} by ${dist} error:`, e));
  };

  const handleSetPosition = (key, value, label) => {
    const numericValue = Number(value);
    if (isNaN(numericValue)) {
      console.error(`${label} must be a number`);
      return;
    }
    if (!window.confirm(`Set ${label} to ${numericValue}?`)) return;
    apiObjectiveControllerSetPositions({
      [key]: numericValue,
      isBlocking: false,
    })
      .then(() => refreshStatus())
      .catch((err) => console.error(`Error setting ${label}:`, err));
  };

  const handleSetX0 = (value) => handleSetPosition("x0", value, "Position 1 (X0)");
  const handleSetX1 = (value) => handleSetPosition("x1", value, "Position 2 (X1)");
  const handleSetZ0 = (value) => handleSetPosition("z0", value, "Focus 1 (Z0)");
  const handleSetZ1 = (value) => handleSetPosition("z1", value, "Focus 2 (Z1)");

  const handleSetCurrentAs = async (which) => {
    apiPositionerControllerGetPositions()
      .then((data) => {
        let posA, posZ;
        const stage = data.ESP32Stage || data.VirtualStage;
        if (stage) {
          posZ = stage.Z;
          posA = stage.A;
          dispatch(objectiveSlice.setCurrentZ(posZ));
          dispatch(objectiveSlice.setCurrentA(posA));
        }
        if (which === "x0") handleSetX0(posA);
        else if (which === "x1") handleSetX1(posA);
        else if (which === "z0") handleSetZ0(posZ);
        else if (which === "z1") handleSetZ1(posZ);
      })
      .catch((err) => {
        console.error(`Error setting current position as ${which}:`, err);
      });
  };

  // Position updates now come automatically via WebSocketHandler -> positionSlice
  // No need to fetch positions or listen to socket events - data is in Redux

  const handleSaveObjectiveMeta = (slot) => {
    const meta = editMeta[slot] || {};
    const params = { objectiveSlot: slot };
    if (meta.name !== undefined && meta.name !== "") params.objectiveName = meta.name;
    if (meta.NA !== undefined && meta.NA !== "") params.NA = parseFloat(meta.NA);
    if (meta.magnification !== undefined && meta.magnification !== "") params.magnification = parseInt(meta.magnification, 10);
    if (meta.pixelsize !== undefined && meta.pixelsize !== "") params.pixelsize = parseFloat(meta.pixelsize);
    const summary = Object.entries(params)
      .filter(([k]) => k !== "objectiveSlot")
      .map(([k, v]) => `${k}: ${v}`)
      .join(", ");
    if (!window.confirm(`Save slot ${slot} metadata?\n${summary}`)) return;
    apiObjectiveControllerSetObjectiveParameters(params)
      .then(() => {
        refreshStatus();
        setEditMeta((prev) => ({ ...prev, [slot]: {} }));
      })
      .catch((err) => console.error("Error saving objective metadata:", err));
  };

  return (
    <Paper style={{ padding: "24px", maxWidth: 1100, margin: "0 auto" }}>
      <Grid container spacing={3}>
        {/* Objective Information */}
        <Grid item xs={12}>
          <Typography variant="h5" gutterBottom>
            Objective Controller
          </Typography>
          <Typography>
            <b>Current Objective:</b>{" "}
            {objectiveState.currentObjective !== null
              ? objectiveState.currentObjective
              : "Unknown"}{" "}
            ({objectiveState.objectivName || "Unknown"})
          </Typography>
          <Typography>
            <b>Pixelsize:</b> {objectiveState.pixelsize ?? "Unknown"},{" "}
            <b>NA:</b> {objectiveState.NA ?? "Unknown"},{" "}
            <b>Magnification:</b> {objectiveState.magnification || "Unknown"}
          </Typography>

          {/* Per-slot metadata cards */}
          <Grid container spacing={2} sx={{ mt: 1 }}>
            {[0, 1].map((slot) => (
              <Grid item xs={12} md={6} key={slot}>
                <Box sx={{ border: "1px solid #ddd", borderRadius: 2, p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    <b>Slot {slot} — {objectiveState.availableObjectivesNames?.[slot] || `Obj ${slot + 1}`}</b>
                  </Typography>
                  <Typography variant="body2">
                    Magnification: {objectiveState.availableObjectiveMagnifications?.[slot] ?? "—"} &nbsp;|
                    NA: {objectiveState.availableObjectiveNAs?.[slot] ?? "—"} &nbsp;|
                    Pixelsize: {objectiveState.availableObjectivePixelSizes?.[slot] ?? "—"} µm/px
                  </Typography>
                  <Box sx={{ display: "flex", gap: 1, mt: 1, flexWrap: "wrap" }}>
                    <TextField
                      label="Name"
                      size="small"
                      value={editMeta[slot]?.name ?? ""}
                      placeholder={objectiveState.availableObjectivesNames?.[slot] || ""}
                      onChange={(e) => setEditMeta((p) => ({ ...p, [slot]: { ...p[slot], name: e.target.value } }))}
                      sx={{ width: 110 }}
                    />
                    <TextField
                      label="Magnification"
                      size="small"
                      type="number"
                      value={editMeta[slot]?.magnification ?? ""}
                      placeholder={String(objectiveState.availableObjectiveMagnifications?.[slot] ?? "")}
                      onChange={(e) => setEditMeta((p) => ({ ...p, [slot]: { ...p[slot], magnification: e.target.value } }))}
                      sx={{ width: 110 }}
                    />
                    <TextField
                      label="NA"
                      size="small"
                      type="number"
                      value={editMeta[slot]?.NA ?? ""}
                      placeholder={String(objectiveState.availableObjectiveNAs?.[slot] ?? "")}
                      onChange={(e) => setEditMeta((p) => ({ ...p, [slot]: { ...p[slot], NA: e.target.value } }))}
                      sx={{ width: 80 }}
                    />
                    <TextField
                      label="Pixelsize"
                      size="small"
                      type="number"
                      value={editMeta[slot]?.pixelsize ?? ""}
                      placeholder={String(objectiveState.availableObjectivePixelSizes?.[slot] ?? "")}
                      onChange={(e) => setEditMeta((p) => ({ ...p, [slot]: { ...p[slot], pixelsize: e.target.value } }))}
                      sx={{ width: 90 }}
                    />
                    <Button variant="contained" size="small" onClick={() => handleSaveObjectiveMeta(slot)}>
                      Save
                    </Button>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Detector Live View (Stream) and Current Positions */}
        <Grid item xs={12}>
          <Grid container spacing={2} alignItems="flex-start">
            {/* Live Stream */}
            <Grid item xs={12} md={7}>
              <Typography variant="h6" gutterBottom>
                Live Stream
              </Typography>
              <Box
                sx={{
                  border: "1px solid #ddd",
                  borderRadius: 2,
                  p: 1,
                  mb: 2,
                  maxWidth: 600,
                  maxHeight: 400,
                  overflow: "auto",
                  background: "#fafbfc",
                }}
              >
                <LiveViewControlWrapper />
              </Box>
            </Grid>
            {/* Current Positions */}
            <Grid item xs={12} md={5}>
              <Box
                sx={{
                  border: "1px solid #eee",
                  borderRadius: 2,
                  p: 2,
                  mb: 2,
                  minWidth: 180,
                  background:
                    theme.palette.mode === "dark"
                      ? theme.palette.background.paper
                      : "#f8fafd",
                }}
              >
                <Typography variant="subtitle1" gutterBottom>
                  Current Stage Positions
                </Typography>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  <Typography>
                    <b>X:</b>{" "}
                    {positions.X !== undefined ? positions.X : "Unknown"}
                  </Typography>
                  <Typography>
                    <b>Y:</b>{" "}
                    {positions.Y !== undefined ? positions.Y : "Unknown"}
                  </Typography>
                  <Typography>
                    <b>Z:</b>{" "}
                    {positions.Z !== undefined ? positions.Z : "Unknown"}
                  </Typography>
                  <Typography>
                    <b>A:</b>{" "}
                    {positions.A !== undefined ? positions.A : "Unknown"}
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Grid>

        {/* Calibration and Switching */}
        <Grid item xs={12}>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <Button
                variant="contained"
                color="success"
                onClick={() => setWizardOpen(true)}
              >
                Start Calibration Wizard
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant="contained"
                color="primary"
                onClick={handleCalibrate}
              >
                Calibrate/Home Objective
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={refreshStatus}>
                Refresh Positions
              </Button>
            </Grid>
          </Grid>
        </Grid>

        {/* Objective Lens Movement */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Move Objective Lens (Axis A)
          </Typography>
          <Grid container spacing={1}>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(-1000)}>
                ←←← (1000)
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(-100)}>
                ←← (100)
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(-10)}>
                ← (10)
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(10)}>
                (10) →
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(100)}>
                (100) →→
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(1000)}>
                (1000) →→→
              </Button>
            </Grid>
          </Grid>
        </Grid>

        {/* Z Focus Movement */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Move Focus (Axis Z)
          </Typography>
          <Grid container spacing={1}>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(-1000, "Z")}>
                ↓↓↓ (1000)
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(-100, "Z")}>
                ↓↓ (100)
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(-10, "Z")}>
                ↓ (10)
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(10, "Z")}>
                (10) ↑
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(100, "Z")}>
                (100) ↑↑
              </Button>
            </Grid>
            <Grid item>
              <Button variant="outlined" onClick={() => movePositioner(1000, "Z")}>
                (1000) ↑↑↑
              </Button>
            </Grid>
          </Grid>
        </Grid>

        {/* Objective Positions (X0, X1, Z0, Z1) */}
        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>
            Objective Positions
          </Typography>
          <Grid container spacing={2}>
            {/* X0 */}
            <Grid item xs={12} md={6} lg={3}>
              <Box
                sx={{ border: "1px solid #eee", borderRadius: 2, p: 2, mb: 2 }}
              >
                <Typography variant="body1">
                  <b>Position 1 ({objectiveState.availableObjectivesNames?.[0] || "Obj 1"}):</b>{" "}
                  {objectiveState.posX0 !== null ? objectiveState.posX0 : "Unknown"}
                </Typography>
                <TextField
                  label="Set Position 1"
                  value={objectiveState.manualX0}
                  onChange={(e) =>
                    dispatch(objectiveSlice.setManualX0(e.target.value))
                  }
                  size="small"
                  fullWidth
                  sx={{ my: 1 }}
                />
                <Button
                  variant="contained"
                  onClick={() => handleSetX0(objectiveState.manualX0)}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Set Position 1
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => handleSetCurrentAs("x0")}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Set Current as Position 1
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={() => handleSwitchObjective(0, true)}
                  fullWidth
                >
                  Switch to Objective 1
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={() => handleSwitchObjective(0, false)}
                  fullWidth
                >
                  Switch to Objective 1 (incl. Z)
                </Button>                
              </Box>
            </Grid>
            {/* X1 */}
            <Grid item xs={12} md={6} lg={3}>
              <Box
                sx={{ border: "1px solid #eee", borderRadius: 2, p: 2, mb: 2 }}
              >
                <Typography variant="body1">
                  <b>Position 2 ({objectiveState.availableObjectivesNames?.[1] || "Obj 2"}):</b>{" "}
                  {objectiveState.posX1 !== null ? objectiveState.posX1 : "Unknown"}
                </Typography>
                <TextField
                  label="Set Position 2"
                  value={objectiveState.manualX1}
                  onChange={(e) =>
                    dispatch(objectiveSlice.setManualX1(e.target.value))
                  }
                  size="small"
                  fullWidth
                  sx={{ my: 1 }}
                />
                <Button
                  variant="contained"
                  onClick={() => handleSetX1(objectiveState.manualX1)}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Set Position 2
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => handleSetCurrentAs("x1")}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Set Current as Position 2
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={() => handleSwitchObjective(1, true  )}
                  fullWidth
                >
                  Switch to Objective 2
                </Button>
              <Button
                  variant="outlined"
                  color="secondary"
                  onClick={() => handleSwitchObjective(1, false  )}
                  fullWidth
                >
                  Switch to Objective 2 (incl. Z)
                </Button>                
              </Box>
            </Grid>
            {/* Z0 */}
            <Grid item xs={12} md={6} lg={3}>
              <Box
                sx={{ border: "1px solid #eee", borderRadius: 2, p: 2, mb: 2 }}
              >
                <Typography variant="body1">
                  <b>Focus 1 ({objectiveState.availableObjectivesNames?.[0] || "Obj 1"}):</b>{" "}
                  {objectiveState.posZ0 !== null ? objectiveState.posZ0 : "Unknown"}
                </Typography>
                <TextField
                  label="Set Focus 1"
                  value={manualZ0}
                  onChange={(e) =>
                    dispatch(objectiveSlice.setManualZ0(e.target.value))
                  }
                  size="small"
                  fullWidth
                  sx={{ my: 1 }}
                />
                <Button
                  variant="contained"
                  onClick={() => handleSetZ0(manualZ0)}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Set Focus 1
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => handleSetCurrentAs("z0")}
                  fullWidth
                >
                  Set Current as Focus 1
                </Button>
              </Box>
            </Grid>
            {/* Z1 */}
            <Grid item xs={12} md={6} lg={3}>
              <Box
                sx={{ border: "1px solid #eee", borderRadius: 2, p: 2, mb: 2 }}
              >
                <Typography variant="body1">
                  <b>Focus 2 ({objectiveState.availableObjectivesNames?.[1] || "Obj 2"}):</b>{" "}
                  {objectiveState.posZ1 !== null ? objectiveState.posZ1 : "Unknown"}
                </Typography>
                <TextField
                  label="Set Focus 2"
                  value={manualZ1}
                  onChange={(e) =>
                    dispatch(objectiveSlice.setManualZ1(e.target.value))
                  }
                  size="small"
                  fullWidth
                  sx={{ my: 1 }}
                />
                <Button
                  variant="contained"
                  onClick={() => handleSetZ1(manualZ1)}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Set Focus 2
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => handleSetCurrentAs("z1")}
                  fullWidth
                >
                  Set Current as Focus 2
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Calibration Wizard */}
      <ObjectiveCalibrationWizard
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
      />
    </Paper>
  );
};

export default ExtendedObjectiveController;
