import React, { useState, useEffect, useRef } from "react";
import {
  Paper,
  Grid,
  Box,
  Button,
  ButtonGroup,
  Chip,
  Switch,
  CircularProgress,
  Typography,
} from "@mui/material";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as laserSlice from "../state/slices/LaserSlice.js";
import * as stormSlice from "../state/slices/STORMSlice.js";
import * as detectorParametersSlice from "../state/slices/DetectorParametersSlice.js";
import { setNotification } from "../state/slices/NotificationSlice.js";
import { useDispatch, useSelector } from "react-redux";
import fetchObjectiveControllerGetStatus from "../middleware/fetchObjectiveControllerGetStatus.js";
import apiObjectiveControllerMoveToObjective from "../backendapi/apiObjectiveControllerMoveToObjective.js";
import {
  rememberObjectiveIllumination,
  restoreObjectiveIllumination,
} from "../middleware/objectiveIlluminationPresets.js";

const OBJECTIVE_SWITCH_TIMEOUT_MS = 15000;

export default function ObjectiveSwitcher({ hostIP, hostPort }) {
  const dispatch = useDispatch();
  const [currentSlot, setCurrentSlot] = useState(null);
  const [pendingSlot, setPendingSlot] = useState(null);
  const [isSwitching, setIsSwitching] = useState(false);
  const [zLevelingEnabled, setZLevelingEnabled] = useState(true);
  const switchTimeoutRef = useRef(null);
  const latestObjectiveRef = useRef(null);
  const isSwitchingRef = useRef(false);

  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const laserState = useSelector(laserSlice.getLaserState);
  const name0 = objectiveState.availableObjectivesNames?.[0] || "Obj 1";
  const name1 = objectiveState.availableObjectivesNames?.[1] || "Obj 2";
  const mag0 = objectiveState.availableObjectiveMagnifications?.[0];
  const mag1 = objectiveState.availableObjectiveMagnifications?.[1];
  const label0 = mag0 ? `${name0} (${mag0}×)` : name0;
  const label1 = mag1 ? `${name1} (${mag1}×)` : name1;
  const isObjectiveSwitchBlocked = isSwitching;

  const clearSwitchTimeout = () => {
    if (switchTimeoutRef.current) {
      clearTimeout(switchTimeoutRef.current);
      switchTimeoutRef.current = null;
    }
  };

  useEffect(() => {
    fetchObjectiveControllerGetStatus(dispatch);
  }, [dispatch]);

  useEffect(() => {
    latestObjectiveRef.current = objectiveState.currentObjective;

    if (objectiveState.currentObjective != null) {
      clearSwitchTimeout();
      setCurrentSlot(objectiveState.currentObjective);
      setPendingSlot(null);
      setIsSwitching(false);
    }
  }, [objectiveState.currentObjective]);

  useEffect(() => {
    isSwitchingRef.current = isSwitching;
  }, [isSwitching]);

  useEffect(() => {
    return () => {
      clearSwitchTimeout();
    };
  }, []);

  useEffect(() => {
    fetchObjectiveControllerGetStatus(dispatch);
  }, [hostIP, hostPort, dispatch]);

  const switchTo = async (slot) => {
    if (isObjectiveSwitchBlocked) {
      return;
    }

    if (objectiveState.currentObjective === slot) {
      dispatch(
        setNotification({
          message: "This objective is already selected.",
          type: "info",
        }),
      );
      return;
    }

    try {
      await rememberObjectiveIllumination({
        objectiveSlot: objectiveState.currentObjective,
        laserState,
        hostIP,
        hostPort,
      });
      setIsSwitching(true);
      setPendingSlot(slot);
      setCurrentSlot(null);
      const skipZ = !zLevelingEnabled;
      await apiObjectiveControllerMoveToObjective(slot, skipZ);
      const restoreResult = await restoreObjectiveIllumination({
        objectiveSlot: slot,
        hostIP,
        hostPort,
        dispatch,
        laserSlice,
        stormSlice,
        detectorParametersSlice,
      });

      if (restoreResult.errors.length > 0) {
        dispatch(
          setNotification({
            message: `Objective switched, but some illumination values could not be restored: ${restoreResult.errors.join("; ")}`,
            type: "warning",
          }),
        );
      }

      clearSwitchTimeout();
      switchTimeoutRef.current = setTimeout(() => {
        switchTimeoutRef.current = null;
        if (!isSwitchingRef.current) {
          return;
        }

        setPendingSlot(null);
        setCurrentSlot(latestObjectiveRef.current ?? null);
        setIsSwitching(false);
        dispatch(
          setNotification({
            message:
              "Objective switch timed out. Please check hardware connection and try again.",
            type: "error",
          }),
        );
      }, OBJECTIVE_SWITCH_TIMEOUT_MS);
    } catch (e) {
      console.error(`Error switching to objective ${slot}`, e);
      clearSwitchTimeout();
      dispatch(
        setNotification({
          message: `Objective switch failed: ${e?.message || e}`,
          type: "error",
        }),
      );
      fetchObjectiveControllerGetStatus(dispatch);
      setPendingSlot(null);
      setCurrentSlot(
        latestObjectiveRef.current ?? objectiveState.currentObjective,
      );
      setIsSwitching(false);
    }
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12}>
          {isObjectiveSwitchBlocked ? (
            <Typography color="text.secondary">
              The objective is switching right now... please wait.
            </Typography>
          ) : (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                flexWrap: "wrap",
              }}
            >
              <Typography>Current objective:</Typography>
              <Chip
                size="small"
                variant="outlined"
                label={
                  <Box
                    sx={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 0.75,
                    }}
                  >
                    <span>
                      {objectiveState.objectivName &&
                        `(${objectiveState.objectivName})`}
                    </span>
                    {objectiveState.magnification != null && (
                      <span>Mag: {objectiveState.magnification}×</span>
                    )}
                    {objectiveState.NA != null && (
                      <span>NA {objectiveState.NA}</span>
                    )}
                    {objectiveState.pixelsize != null && (
                      <span>Pixel {objectiveState.pixelsize} µm</span>
                    )}
                  </Box>
                }
              />
            </Box>
          )}
        </Grid>

        <Grid item xs={12}>
          <Grid container spacing={1.5} alignItems="center">
            <Grid item xs={12}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 2,
                  flexWrap: "wrap",
                  width: "100%",
                }}
              >
                <Typography variant="subtitle1" sx={{ whiteSpace: "nowrap" }}>
                  Objective Switcher
                </Typography>

                <ButtonGroup
                  variant="outlined"
                  size="small"
                  color="primary"
                  disabled={isObjectiveSwitchBlocked}
                >
                  <Button
                    variant={currentSlot === 0 ? "contained" : "outlined"}
                    color="primary"
                    onClick={() => switchTo(0)}
                  >
                    <Box
                      sx={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 0.75,
                      }}
                    >
                      {isSwitching && pendingSlot === 0 ? (
                        <CircularProgress size={14} sx={{ color: "#fff" }} />
                      ) : null}
                      {label0}
                    </Box>
                  </Button>

                  <Button
                    variant={currentSlot === 1 ? "contained" : "outlined"}
                    color="primary"
                    onClick={() => switchTo(1)}
                  >
                    <Box
                      sx={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 0.75,
                      }}
                    >
                      {isSwitching && pendingSlot === 1 ? (
                        <CircularProgress size={14} sx={{ color: "#fff" }} />
                      ) : null}
                      {label1}
                    </Box>
                  </Button>
                </ButtonGroup>
              </Box>
            </Grid>

            <Grid item xs={12}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 2,
                  flexWrap: "wrap",
                  width: "100%",
                }}
              >
                <Switch
                  checked={zLevelingEnabled}
                  onChange={(e) => setZLevelingEnabled(e.target.checked)}
                  size="small"
                />
                <Typography
                  variant="body2"
                  component="span"
                  sx={{ whiteSpace: "nowrap" }}
                >
                  Z leveling
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Switching will apply{" "}
                  {zLevelingEnabled ? "with Z leveling" : "without Z leveling"}.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Paper>
  );
}
