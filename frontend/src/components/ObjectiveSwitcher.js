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

  // Access global Redux state (objectiveState from objectiveSlice)
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const magnification1 = objectiveState.magnification1;
  const magnification2 = objectiveState.magnification2;
  // Sample effect to track which slot is active
  // If we have data from store, set the local currentSlot
  useEffect(() => {
    fetchObjectiveControllerGetStatus(dispatch);
    if (objectiveState.currentObjective != null) {
      setCurrentSlot(objectiveState.currentObjective);
      setIsSwitching(false); // We got an update => done switching
    }
  }, [dispatch]);

  useEffect(() => {
      //refresh status
      refreshStatus();
    }, [hostIP, hostPort]); // on host ip/port change
  
  // Switch to a different objective, show spinner until we get update from the socket
  const switchTo = async (slot) => {
    try {
      setIsSwitching(true);
      await apiObjectiveControllerMoveToObjective(slot, 0);
      // When the socket update arrives, it should set objectiveState.currentObjective => triggers useEffect above
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
                {/* Button text */}
                Switch to{" "}
                {isSwitching && currentSlot !== 0 ? (
                  <CircularProgress
                    size={14}
                    sx={{ color: "#fff", marginLeft: 8 }}
                  />
                ) : (
                  magnification1 && `(Mag: ${magnification1}×)`
                )}
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant="contained"
                color={currentSlot === 2 ? "secondary" : "primary"}
                onClick={() => switchTo(1)}
              >
                {/* Button text */}
                Switch to{" "}
                {isSwitching && currentSlot !== 1 ? (
                  <CircularProgress
                    size={14}
                    sx={{ color: "#fff", marginLeft: 8 }}
                  />
                ) : (
                  magnification2 && `(Mag: ${magnification2}×)`
                )}
              </Button>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Paper>
  );
}
