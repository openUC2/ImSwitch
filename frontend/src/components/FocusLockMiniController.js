import React, { useState, useEffect, useRef, useCallback } from "react";
import { Paper, Grid, Button, Typography, TextField, Card, CardContent, Box } from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import * as focusLockSlice from "../state/slices/FocusLockSlice.js";

import apiFocusLockControllerGetCurrentFocusValue from "../backendapi/apiFocusLockControllerGetCurrentFocusValue.js";
import apiFocusLockControllerGetFocusLockState from "../backendapi/apiFocusLockControllerGetFocusLockState.js";
import apiFocusLockControllerReturnLastImage from "../backendapi/apiFocusLockControllerReturnLastImage.js";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";

const FocusLockMiniController = () => {
  const dispatch = useDispatch();

  const focusLockUI = useSelector((state) => ({
    currentFocusValue: state.focusLockState.currentFocusValue,
    lastImage: state.focusLockState.lastImage,
    isLoadingImage: state.focusLockState.isLoadingImage,
    setPoint: state.focusLockState.setPoint,
    status: state.focusLockState.isFocusLocked ? "Locked" : "Unlocked",
  }));

  const imgRef = useRef(null);

  // load backend status
  const loadStatus = useCallback(async () => {
    try {
      const state = await apiFocusLockControllerGetFocusLockState();
      // could store more state here if needed
    } catch (e) {
      console.error("status error", e);
    }
  }, []);

  // poll focus value
  const pollFocus = useCallback(async () => {
    try {
      const r = await apiFocusLockControllerGetCurrentFocusValue();
      if (r && r.focus_value !== undefined) {
        dispatch(focusLockSlice.setCurrentFocusValue(r.focus_value));
      }
    } catch (e) {}
  }, [dispatch]);

  useEffect(() => {
    loadStatus();
    const t = setInterval(() => {
      pollFocus();
    }, 500);
    return () => clearInterval(t);
  }, [loadStatus, pollFocus]);

  // load image
  const loadLastImage = useCallback(async () => {
    try {
      dispatch(focusLockSlice.setIsLoadingImage(true));
      const blob = await apiFocusLockControllerReturnLastImage();
      const url = URL.createObjectURL(blob);
      dispatch(focusLockSlice.setLastImage(url));
    } catch (e) {
      console.error(e);
    } finally {
      dispatch(focusLockSlice.setIsLoadingImage(false));
    }
  }, [dispatch]);

  // set setpoint
  const [manualSetpoint, setManualSetpoint] = useState("");

  const setSetpoint = () => {
    if (manualSetpoint === "") return;
    const val = parseFloat(manualSetpoint);
    if (!isNaN(val)) dispatch(focusLockSlice.setSetPoint(val));
  };

  const useCurrentAsSetpoint = () => {
    dispatch(focusLockSlice.setSetPoint(focusLockUI.currentFocusValue));
  };

  // move Z
  const moveZ = async (d) => {
    try {
      await apiPositionerControllerMovePositioner({
        positionerName: "ESP32Stage",
        axis: "Z",
        dist: d,
        isAbsolute: false,
        isBlocking: false,
        speed: 1000,
      });
    } catch (e) {
      console.error("moveZ error", e);
    }
  };

  return (
    <Paper style={{ padding: 20 }}>
      <Grid container spacing={2}>
        {/* FOCUS + STATUS */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6">Focus Status</Typography>

              <Typography sx={{ mt: 1 }}>
                Current Focus: {focusLockUI.currentFocusValue.toFixed(3)}
              </Typography>

              <Typography>Status: {focusLockUI.status}</Typography>

              {/* Setpoint controls */}
              <Box sx={{ mt: 2 }}>
                <TextField
                  label="Setpoint"
                  size="small"
                  value={manualSetpoint}
                  onChange={(e) => setManualSetpoint(e.target.value)}
                  fullWidth
                />
                <Button sx={{ mt: 1 }} variant="contained" onClick={setSetpoint}>
                  Set Setpoint
                </Button>

                <Button sx={{ mt: 1 }} variant="outlined" onClick={useCurrentAsSetpoint}>
                  Use Current Focus as Setpoint
                </Button>

                <Typography sx={{ mt: 1 }}>
                  Current SetPoint: {focusLockUI.setPoint.toFixed(3)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* IMAGE */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6">Latest Image</Typography>

              <Button sx={{ mt: 1, mb: 2 }} variant="contained" onClick={loadLastImage}>
                {focusLockUI.isLoadingImage ? "Loading..." : "Load Last Image"}
              </Button>

              {focusLockUI.lastImage ? (
                <img
                  ref={imgRef}
                  src={focusLockUI.lastImage}
                  alt="focus"
                  style={{ maxWidth: "100%", border: "1px solid #999" }}
                />
              ) : (
                <Typography>No Image Loaded</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Z MOVES */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6">Z Movement</Typography>

              <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mt: 2 }}>
                <Button variant="contained" onClick={() => moveZ(+10)}>Z +10</Button>
                <Button variant="contained" onClick={() => moveZ(-10)}>Z -10</Button>
                <Button variant="contained" color="secondary" onClick={() => moveZ(+100)}>Z +100</Button>
                <Button variant="contained" color="secondary" onClick={() => moveZ(-100)}>Z -100</Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default FocusLockMiniController;
