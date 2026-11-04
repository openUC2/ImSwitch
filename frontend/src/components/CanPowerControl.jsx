import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box, Chip, FormControlLabel, Switch, Typography } from "@mui/material";
import BoltIcon from "@mui/icons-material/Bolt";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import { setNotification } from "../state/slices/NotificationSlice";
import apiUC2ConfigControllerGetBusStatus from "../backendapi/apiUC2ConfigControllerGetBusStatus";
import apiUC2ConfigControllerSetBusPower from "../backendapi/apiUC2ConfigControllerSetBusPower";

// CAN-bus power switch for the live view. Shown only while a UC2 ESP32 is connected;
// state comes from the same store as the System page (sigBusStatusUpdate -> UC2Slice).
const CanPowerControl = () => {
  const dispatch = useDispatch();
  const { uc2Connected, backendConnected, busPower, busAvailable, emergencyActive } =
    useSelector(uc2Slice.getUc2State);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      const status = await apiUC2ConfigControllerGetBusStatus();
      if (status && typeof status === "object") dispatch(uc2Slice.setBusStatus(status));
    } catch (e) {
      console.error("getBusStatus failed:", e);
    }
  };

  useEffect(() => {
    if (uc2Connected && backendConnected) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uc2Connected, backendConnected]);

  if (!uc2Connected) return null;

  const toggle = async (enable) => {
    setBusy(true);
    try {
      const res = await apiUC2ConfigControllerSetBusPower(enable);
      if (res?.status === "success") {
        dispatch(uc2Slice.setBusPower(res.power));
      } else {
        dispatch(setNotification({ message: `Failed to set bus power: ${res?.message || "unknown error"}`, type: "error" }));
      }
      refresh();
    } catch (e) {
      dispatch(setNotification({ message: "setBusPower failed: " + (e.message || e), type: "error" }));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Box data-tour="can-power" sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <BoltIcon fontSize="small" color="action" />
      <Typography variant="h6">CAN Power</Typography>
      <FormControlLabel
        sx={{ ml: 1 }}
        control={<Switch checked={busPower === 1} disabled={busy || emergencyActive} onChange={(e) => toggle(e.target.checked)} />}
        label={busPower === 1 ? "On" : busPower === 0 ? "Off" : "?"}
      />
      {emergencyActive ? (
        <Chip label="E-stop" color="error" size="small" />
      ) : (
        !busAvailable && <Chip label="Bus unavailable" color="warning" size="small" variant="outlined" />
      )}
    </Box>
  );
};

export default CanPowerControl;
