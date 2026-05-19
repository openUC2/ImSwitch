import React, { useState, useEffect } from "react";
import {
  Paper,
  Grid,
  Button,
  CircularProgress,
  Typography,
} from "@mui/material";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import { useDispatch, useSelector } from "react-redux";
import fetchObjectiveControllerGetStatus from "../middleware/fetchObjectiveControllerGetStatus.js";

// Hypothetical backend helpers (adjust import paths if different)
import apiObjectiveControllerMoveToObjective from "../backendapi/apiObjectiveControllerMoveToObjective.js";
import apiObjectiveControllerGetCurrentObjective from "../backendapi/apiObjectiveControllerGetCurrentObjective.js";

export default function ObjectiveSwitcher({ hostIP, hostPort }) {
  const dispatch = useDispatch();
  const [currentSlot, setCurrentSlot] = useState(null); // Let's say local state, or read from Redux
  const [isSwitching, setIsSwitching] = useState(false); // Show spinner while switching

  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const name0 = objectiveState.availableObjectivesNames?.[0] || "Obj 1";
  const name1 = objectiveState.availableObjectivesNames?.[1] || "Obj 2";
  const mag0 = objectiveState.availableObjectiveMagnifications?.[0];
  const mag1 = objectiveState.availableObjectiveMagnifications?.[1];
  const label0 = mag0 ? `${name0} (${mag0}×)` : name0;
  const label1 = mag1 ? `${name1} (${mag1}×)` : name1;
  // Fetch objective status on mount
  useEffect(() => {
    fetchObjectiveControllerGetStatus(dispatch);
  }, [dispatch]);

  // Track Redux state changes (e.g. from socket updates) and update
  // local state + clear spinner when the objective has changed.
  useEffect(() => {
    if (objectiveState.currentObjective != null) {
      setCurrentSlot(objectiveState.currentObjective);
      setIsSwitching(false); // Objective update received => done switching
    }
  }, [objectiveState.currentObjective]);

  useEffect(() => {
      //refresh status
      refreshStatus();
    }, [hostIP, hostPort]); // on host ip/port change
  
  // Switch to a different objective, show spinner until we get update from the socket
  const switchTo = async (slot, skipZ = 0) => {
    try {
      setIsSwitching(true);
      await apiObjectiveControllerMoveToObjective(slot, skipZ);
      // Move completed – update slot and clear spinner immediately; no need to wait for
      // a socket event that may never arrive.
      dispatch(objectiveSlice.setCurrentObjective(slot));
      setCurrentSlot(slot);
      setIsSwitching(false);
      // Also refresh full status to sync pixelsize / FOV etc.
      fetchObjectiveControllerGetStatus(dispatch);
    } catch (e) {
      console.error(`Error switching to objective ${slot}`, e);
      setIsSwitching(false);
    }
  };

  // Fetch objective status (x1 and x2) from backend
  const refreshStatus = () => {
      //request fetch status
      fetchObjectiveControllerGetStatus(dispatch);
    };
  

  return (
    <Paper sx={{ p: 2 }}>
      <Grid container spacing={2} alignItems="center">
        {/* Info row about the current objective */}
        <Grid item xs={12}>
          <Typography>
            {/* Show numeric slot + name from Redux */}
            Current:{" "}
            {objectiveState.objectivName && `(${objectiveState.objectivName})`}
          </Typography>
          <Typography sx={{ fontSize: "0.9rem" }}>
            {objectiveState.magnification != null &&
              `Mag: ${objectiveState.magnification}×  • `}
            {objectiveState.NA != null && `NA ${objectiveState.NA}  • `}
            {objectiveState.pixelsize != null &&
              `Pixel ${objectiveState.pixelsize} µm`}
          </Typography>
        </Grid>

        {/* Buttons to switch objective */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item>
              <Button
                variant="contained"
                color={currentSlot === 0 ? "secondary" : "primary"}
                onClick={() => switchTo(0)}
              >
                {isSwitching && currentSlot !== 0 ? (
                  <CircularProgress size={14} sx={{ color: "#fff", ml: 1 }} />
                ) : (
                  label0
                )}
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant="outlined"
                color={currentSlot === 0 ? "secondary" : "primary"}
                onClick={() => switchTo(0, false)}
              >
                {label0} + Z
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant="contained"
                color={currentSlot === 1 ? "secondary" : "primary"}
                onClick={() => switchTo(1)}
              >
                {isSwitching && currentSlot !== 1 ? (
                  <CircularProgress size={14} sx={{ color: "#fff", ml: 1 }} />
                ) : (
                  label1
                )}
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant="outlined"
                color={currentSlot === 1 ? "secondary" : "primary"}
                onClick={() => switchTo(1, false)}
              >
                {label1} + Z
              </Button>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Paper>
  );
}
