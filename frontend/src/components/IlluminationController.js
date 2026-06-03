import React, { useEffect, useRef, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box, Checkbox, Slider, Typography, Paper, Grid } from "@mui/material";

import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as laserSlice from "../state/slices/LaserSlice.js";
import fetchExperimentControllerGetCurrentExperimentParams from "../middleware/fetchExperimentControllerGetCurrentExperimentParams.js";
import apiLaserControllerGetLaserChannelIndex from "../backendapi/apiLaserControllerGetLaserChannelIndex.js";

export default function IlluminationController({ hostIP, hostPort }) {
  const dispatch = useDispatch();

  // Get state from Redux
  const parameterRangeState = useSelector(
    parameterRangeSlice.getParameterRangeState,
  );
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState,
  );

  // Get laser state from Redux (updated via WebSocket sigUpdateLaserPower)
  const laserState = useSelector(laserSlice.getLaserState);
  const lasers = laserState.lasers;

  // Debounce refs for laser value updates to prevent serial overload
  const laserTimeoutRefs = useRef({});
  const LASER_UPDATE_DEBOUNCE_MS = 300; // Wait 300ms after user stops adjusting

  // Initialize experiment params and laser values on mount
  useEffect(() => {
    // Fetch experiment parameters which includes laser sources
    fetchExperimentControllerGetCurrentExperimentParams(dispatch);
  }, [dispatch]);

  // Fetch laser names and value ranges when laser sources change
  useEffect(() => {
    if (parameterRangeState.illuSources.length > 0) {
      // Initialize timeout refs object for each laser
      parameterRangeState.illuSources.forEach((laserName) => {
        if (!laserTimeoutRefs.current[laserName]) {
          laserTimeoutRefs.current[laserName] = null;
        }
      });
    }
  }, [parameterRangeState.illuSources]);

  // Cleanup timeout refs on unmount
  useEffect(() => {
    return () => {
      // Clear all pending timeouts
      Object.values(laserTimeoutRefs.current).forEach((timeoutRef) => {
        if (timeoutRef) {
          clearTimeout(timeoutRef);
        }
      });
    };
  }, []);

  // Fetch per-laser initial power/enabled state from the backend.
  //
  // IMPORTANT: this used to also overwrite `illuSources` / min / max from
  // `LaserController.getLaserNames` + `getLaserValueRanges`.  That clobbered
  // the longer channel list (real lasers + LED-matrix synthetic channels
  // "LED Matrix Ring" / "LED Matrix DPC") that `getHardwareParameters`
  // returns from the ExperimentController.  `ExperimentController.getHardwareParameters`
  // is now the single source of truth for the channel list, kinds, and
  // value ranges — `fetchExperimentControllerGetCurrentExperimentParams`
  // (called in the previous effect) populates them.  This effect only
  // fetches per-laser *runtime* state (current power, enabled) for the
  // real-laser entries (kind === "default") — synthetic LED-matrix channels
  // have no LaserController endpoints to hit, so we skip them.
  useEffect(() => {
    async function fetchLaserRuntimeState() {
      const sources = parameterRangeState.illuSources || [];
      const kinds = parameterRangeState.illuSourceKinds || [];
      if (sources.length === 0) return;
      for (let idx = 0; idx < sources.length; idx++) {
        const name = sources[idx];
        const kind = kinds[idx] || "default";
        if (kind !== "default") {
          // Synthetic LED-matrix channels (ring/dpc) have no LaserController
          // endpoint — seed the slice with neutral defaults so the UI has a
          // consistent shape, then move on.
          dispatch(laserSlice.setLaserState({ laserName: name, power: 0, enabled: false }));
          continue;
        }
        const encodedName = encodeURIComponent(name);
        try {
          const valueRes = await fetch(
            `${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserValue?laserName=${encodedName}`,
          );
          const power = await valueRes.json();
          const activeRes = await fetch(
            `${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserActive?laserName=${encodedName}`,
          );
          const enabled = await activeRes.json();
          dispatch(laserSlice.setLaserState({ laserName: name, power, enabled }));
        } catch (err) {
          console.warn(`Failed to fetch initial state for laser ${name}:`, err);
          dispatch(laserSlice.setLaserState({ laserName: name, power: 0, enabled: false }));
        }
      }
    }
    fetchLaserRuntimeState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hostIP, hostPort, dispatch, parameterRangeState.illuSources, parameterRangeState.illuSourceKinds]);

  // Debounced laser value update to prevent serial overload
  const debouncedSetLaserValue = useCallback(
    (laserName, val) => {
      // Update Redux state immediately for UI responsiveness
      dispatch(laserSlice.setLaserPower({ laserName, power: val }));

      // Clear existing timeout for this laser
      if (laserTimeoutRefs.current[laserName]) {
        clearTimeout(laserTimeoutRefs.current[laserName]);
      }

      // Set new timeout to send to backend after user stops adjusting
      laserTimeoutRefs.current[laserName] = setTimeout(async () => {
        const ip = connectionSettingsState.ip || hostIP;
        const port = connectionSettingsState.apiPort || hostPort;

        if (ip && port) {
          try {
            const encodedLaserName = encodeURIComponent(laserName);
            await fetch(
              `${ip}:${port}/imswitch/api/LaserController/setLaserValue?laserName=${encodedLaserName}&value=${val}`,
            );
            console.log(`${laserName} intensity updated to: ${val}`);
          } catch (error) {
            console.error("Failed to set laser value in backend:", error);
          }
        }
      }, LASER_UPDATE_DEBOUNCE_MS);
    },
    [
      dispatch,
      connectionSettingsState.ip,
      connectionSettingsState.apiPort,
      hostIP,
      hostPort,
    ],
  );

  // Update laser active state
  const setLaserActive = async (laserName, active) => {
    // Update Redux state immediately
    dispatch(laserSlice.setLaserEnabled({ laserName, enabled: active }));

    // Update backend
    const ip = connectionSettingsState.ip || hostIP;
    const port = connectionSettingsState.apiPort || hostPort;

    if (ip && port) {
      try {
        const encodedLaserName = encodeURIComponent(laserName);
        await fetch(
          `${ip}:${port}/imswitch/api/LaserController/setLaserActive?laserName=${encodedLaserName}&active=${active}`,
        );
        console.log(`${laserName} active state updated to: ${active}`);
      } catch (error) {
        console.error("Failed to set laser active state in backend:", error);
      }
    }
  };

  // Get laser data from Redux state.
  // We deliberately filter out non-"default" kinds (e.g. LED-matrix ring/DPC
  // synthetic channels) because this panel directly maps slider → LaserController,
  // and synthetic channels have no LaserController endpoints.  They're still
  // present in illuSources for the experiment designer / channel selector,
  // but they don't belong in the live-view laser sliders.
  const allIlluSources = parameterRangeState.illuSources || [];
  const allIlluKinds = parameterRangeState.illuSourceKinds || [];
  const allMinValues = parameterRangeState.illuSourceMinIntensities || [];
  const allMaxValues = parameterRangeState.illuSourceMaxIntensities || [];
  // Build parallel arrays restricted to default-kind (real laser) sources.
  const _laserIndices = allIlluSources
    .map((_, i) => i)
    .filter((i) => (allIlluKinds[i] || "default") === "default");
  const laserSources = _laserIndices.map((i) => allIlluSources[i]);
  const laserMinValues = _laserIndices.map((i) => allMinValues[i]);
  const laserMaxValues = _laserIndices.map((i) => allMaxValues[i]);

  return (
    <Paper sx={{ p: 2 }}>
      <Grid container direction="column" spacing={2}>
        {laserSources.length ? (
          laserSources.map((laserName, idx) => {
            // Get laser state from Redux (updated via WebSocket).
            // Guard against stale persisted state where values may be objects
            // (e.g. {detail: "..."} from a backend error) if the laser no longer
            // exists on the current backend session.
            const laserData = lasers[laserName] || { power: 0, enabled: false };
            const currentValue = typeof laserData.power === "number" ? laserData.power : 0;
            const isActive = !!laserData.enabled;
            const minValue = laserMinValues[idx] || 0;
            const maxValue = laserMaxValues[idx] || 1023;

            return (
              <Grid
                item
                key={laserName}
                sx={{ display: "flex", alignItems: "center", gap: 2 }}
              >
                {/* Laser name */}
                <Typography sx={{ minWidth: 120 }}>{laserName}</Typography>

                {/* Slider with dynamic min and max */}
                <Box sx={{ flex: 1, px: 1 }}>
                  <Slider
                    value={currentValue}
                    min={minValue}
                    max={maxValue}
                    onChange={(e) =>
                      debouncedSetLaserValue(laserName, e.target.value)
                    }
                    sx={{ width: "100%" }}
                    valueLabelDisplay="auto"
                  />
                  <Box
                    sx={{ display: "flex", justifyContent: "space-between" }}
                  >
                    <Typography variant="caption" color="textSecondary">
                      {minValue}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {maxValue}
                    </Typography>
                  </Box>
                </Box>

                {/* Current slider value */}
                <Typography sx={{ minWidth: 60, textAlign: "center" }}>
                  {currentValue}
                </Typography>

                {/* Active checkbox */}
                <Checkbox
                  checked={isActive}
                  onChange={(e) => setLaserActive(laserName, e.target.checked)}
                />
              </Grid>
            );
          })
        ) : (
          <Typography>Loading laser names…</Typography>
        )}
      </Grid>
    </Paper>
  );
}
