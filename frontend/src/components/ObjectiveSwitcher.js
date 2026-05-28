import React, { useState, useEffect } from "react";
import {
  Paper,
  Grid,
  Box,
  Button,
  ButtonGroup,
  FormControlLabel,
  Switch,
  CircularProgress,
  Typography,
} from "@mui/material";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import { useDispatch, useSelector } from "react-redux";
import fetchObjectiveControllerGetStatus from "../middleware/fetchObjectiveControllerGetStatus.js";

// Hypothetical backend helpers (adjust import paths if different)
import apiObjectiveControllerMoveToObjective from "../backendapi/apiObjectiveControllerMoveToObjective.js";

export default function ObjectiveSwitcher({ hostIP, hostPort }) {
  const dispatch = useDispatch();
  const [currentSlot, setCurrentSlot] = useState(null); // Let's say local state, or read from Redux
  const [isSwitching, setIsSwitching] = useState(false); // Show spinner while switching
  const [zLevelingEnabled, setZLevelingEnabled] = useState(true);

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
    fetchObjectiveControllerGetStatus(dispatch);
  }, [hostIP, hostPort, dispatch]); // on host ip/port change

  // Switch to a different objective, show spinner until we get update from the socket
  const switchTo = async (slot) => {
    try {
      setIsSwitching(true);
      const skipZ = !zLevelingEnabled;
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

  return (
    <Paper sx={{ p: 2 }}>
      <Grid container spacing={2} alignItems="center">
        {/* Info row about the current objective */}
        <Grid item xs={12}>
          <Typography>
            {/* Show numeric slot + name from Redux */}
            Current:{" "}
            {objectiveState.objectivName && `(${objectiveState.objectivName})`}
            {objectiveState.magnification != null &&
              ` Mag: ${objectiveState.magnification}×  • `}
            {objectiveState.NA != null && `NA ${objectiveState.NA}  • `}
            {objectiveState.pixelsize != null &&
              `Pixel ${objectiveState.pixelsize} µm`}
          </Typography>
        </Grid>

        {/* Objective selection + Z leveling option */}
        <Grid item xs={12}>
          <Grid container spacing={1.5} alignItems="center">
            <Grid item xs={12}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 2,
                  flexWrap: "wrap",
                }}
              >
                <ButtonGroup variant="outlined" size="small" color="primary">
                  <Button
                    variant={currentSlot === 0 ? "contained" : "outlined"}
                    color="primary"
                    onClick={() => switchTo(0)}
                    disabled={isSwitching}
                  >
                    <Box
                      sx={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 0.75,
                      }}
                    >
                      {isSwitching && currentSlot !== 0 ? (
                        <CircularProgress size={14} sx={{ color: "#fff" }} />
                      ) : null}
                      {label0}
                    </Box>
                  </Button>

                  <Button
                    variant={currentSlot === 1 ? "contained" : "outlined"}
                    color="primary"
                    onClick={() => switchTo(1)}
                    disabled={isSwitching}
                  >
                    <Box
                      sx={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 0.75,
                      }}
                    >
                      {isSwitching && currentSlot !== 1 ? (
                        <CircularProgress size={14} sx={{ color: "#fff" }} />
                      ) : null}
                      {label1}
                    </Box>
                  </Button>
                </ButtonGroup>

                <FormControlLabel
                  control={
                    <Switch
                      checked={zLevelingEnabled}
                      onChange={(e) => setZLevelingEnabled(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Z leveling"
                  sx={{ ml: 0 }}
                />
              </Box>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="caption" color="text.secondary">
                Switching will apply{" "}
                {zLevelingEnabled ? "with Z leveling" : "without Z leveling"}.
              </Typography>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Paper>
  );
}
