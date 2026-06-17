import React, { useEffect, useRef, useCallback, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Switch,
  Slider,
  Typography,
  Paper,
  Grid,
  Input,
} from "@mui/material";

import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as laserSlice from "../state/slices/LaserSlice.js";
import fetchExperimentControllerGetCurrentExperimentParams from "../middleware/fetchExperimentControllerGetCurrentExperimentParams.js";
import fetchLaserRuntimeState from "../middleware/fetchLaserRuntimeState.js";

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
    const timeoutRefs = laserTimeoutRefs.current;

    return () => {
      // Clear all pending timeouts
      Object.values(timeoutRefs).forEach((timeoutRef) => {
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
    async function syncLaserRuntimeState() {
      const sources = parameterRangeState.illuSources || [];
      const kinds = parameterRangeState.illuSourceKinds || [];
      if (sources.length === 0) return;
      const runtimeStates = await fetchLaserRuntimeState({
        hostIP,
        hostPort,
        sources,
        kinds,
      });

      runtimeStates.forEach(({ laserName, power, enabled, ok, error }) => {
        if (!ok && error) {
          console.warn(
            `Failed to fetch initial state for laser ${laserName}:`,
            error,
          );
        }

        dispatch(laserSlice.setLaserState({ laserName, power, enabled }));
      });
    }
    syncLaserRuntimeState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    hostIP,
    hostPort,
    dispatch,
    parameterRangeState.illuSources,
    parameterRangeState.illuSourceKinds,
  ]);

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

  // Track container width to hide slider when too narrow
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(null);
  const SLIDER_MIN_WIDTH = 320;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      setContainerWidth(entries[0].contentRect.width);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const showSlider =
    containerWidth === null || containerWidth >= SLIDER_MIN_WIDTH;

  return (
    <Paper sx={{ p: 2 }} ref={containerRef}>
      <Grid container direction="column" spacing={1.25}>
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
            const marks = [
              { value: minValue, label: `${minValue}` },
              { value: maxValue, label: `${maxValue}` },
            ];

            return (
              <Grid
                item
                key={laserName}
                sx={{ display: "flex", alignItems: "center", gap: 1.25 }}
              >
                {/* Laser name */}
                <Typography sx={{ minWidth: 60 }}>{laserName}</Typography>

                {/* Slider with dynamic min and max */}
                <Box sx={{ flex: 1 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    {showSlider && (
                      <Slider
                        value={currentValue}
                        min={minValue}
                        max={maxValue}
                        marks={marks}
                        onChange={(_event, value) => {
                          const nextValue = Array.isArray(value)
                            ? value[0]
                            : value;
                          debouncedSetLaserValue(laserName, Number(nextValue));
                        }}
                        sx={{
                          flex: 1,
                          "& .MuiSlider-markLabel[data-index='0']": {
                            transform: "translateX(0%)",
                          },
                          "& .MuiSlider-markLabel[data-index='1']": {
                            transform: "translateX(-100%)",
                          },
                        }}
                      />
                    )}
                    <Input
                      value={currentValue}
                      size="small"
                      onChange={(e) => {
                        if (e.target.value === "") {
                          debouncedSetLaserValue(laserName, minValue);
                          return;
                        }
                        const parsed = Number(e.target.value);
                        if (!Number.isFinite(parsed)) return;
                        const clamped = Math.max(
                          minValue,
                          Math.min(maxValue, parsed),
                        );
                        debouncedSetLaserValue(laserName, clamped);
                      }}
                      inputProps={{
                        step: 1,
                        min: minValue,
                        max: maxValue,
                        type: "number",
                        "aria-label": `${laserName} intensity`,
                      }}
                      sx={{ width: 72, fontWeight: 600 }}
                    />
                  </Box>
                </Box>

                {/* Active switch with explicit status */}
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 0.5,
                    minWidth: 112,
                    justifyContent: "flex-end",
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      fontWeight: 700,
                      color: isActive ? "success.main" : "text.secondary",
                      minWidth: 26,
                      textAlign: "right",
                    }}
                  >
                    {isActive ? "ON" : "OFF"}
                  </Typography>
                  <Switch
                    checked={isActive}
                    onChange={(e) =>
                      setLaserActive(laserName, e.target.checked)
                    }
                    inputProps={{
                      "aria-label": `${laserName} illumination enabled`,
                    }}
                  />
                </Box>
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
