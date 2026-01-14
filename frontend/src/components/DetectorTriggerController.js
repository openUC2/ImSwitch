import React, { useState, useEffect, useMemo } from "react";
import {
  Paper,
  Grid,
  Button,
  CircularProgress,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import axios from "axios";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

/* Axios factory bound to the camera host */
const createAxiosInstance = (ip, port) =>
  axios.create({
    baseURL: `${ip}:${port}/imswitch/api`,
    timeout: 6000,
  });

const DetectorTriggerController = () => {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  // Memoize axios instance for current connection settings
  const api = useMemo(
    () => createAxiosInstance(hostIP, hostPort),
    [hostIP, hostPort]
  );

  const [triggerTypes, setTriggerTypes] = useState([]);
  const [currentTrigger, setCurrentTrigger] = useState("");
  const [busyInit, setBusyInit] = useState(true);
  const [busySwitch, setBusySwitch] = useState(false);
  const [busyShot, setBusyShot] = useState(false);

  /* Initial load */
  useEffect(() => {
    (async () => {
      try {
        const [types, current] = await Promise.all([
          api.get("/SettingsController/getDetectorTriggerTypes"),
          api.get("/SettingsController/getDetectorCurrentTriggerType"),
        ]);
        setTriggerTypes(types.data || []);
        setCurrentTrigger(current.data || "");
      } catch (err) {
        console.error("Init error", err);
      } finally {
        setBusyInit(false);
      }
    })();
  }, [api]);

  /* Change trigger type */
  const changeTriggerType = async (type) => {
    setBusySwitch(true);
    try {
      await api.get(
        `/SettingsController/setDetectorTriggerType?triggerType=${encodeURIComponent(
          type
        )}`
      );
      setCurrentTrigger(type);
    } catch (err) {
      console.error("Switch error", err);
    } finally {
      setBusySwitch(false);
    }
  };

  /* Software trigger */
  const shoot = async () => {
    setBusyShot(true);
    try {
      await api.get("/SettingsController/sendSoftwareTrigger");
    } catch (err) {
      console.error("Shot error", err);
    } finally {
      setBusyShot(false);
    }
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={12}>
          <Typography variant="h6">Detector Trigger</Typography>
        </Grid>

        <Grid item xs={12} md={6}>
          <FormControl fullWidth disabled={busyInit || busySwitch}>
            <InputLabel id="trigger-select-label">Trigger Type</InputLabel>
            <Select
              labelId="trigger-select-label"
              value={currentTrigger}
              label="Trigger Type"
              onChange={(e) => changeTriggerType(e.target.value)}
            >
              {triggerTypes.map((t) => (
                <MenuItem key={t} value={t}>
                  {t}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={6}>
          <Button
            fullWidth
            variant="contained"
            disabled={
              !currentTrigger.includes("Software") ||
              busyShot ||
              busyInit ||
              busySwitch
            }
            onClick={shoot}
          >
            {busyShot ? (
              <CircularProgress size={20} sx={{ color: "#fff" }} />
            ) : (
              "Acquire Frame"
            )}
          </Button>
        </Grid>

        {(busyInit || busySwitch) && (
          <Grid item xs={12}>
            <CircularProgress />
          </Grid>
        )}
      </Grid>
    </Paper>
  );
};

export default DetectorTriggerController;
