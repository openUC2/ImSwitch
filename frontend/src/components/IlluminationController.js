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
  const parameterRangeState = useSelector(parameterRangeSlice.getParameterRangeState);
  const connectionSettingsState = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  
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
      parameterRangeState.illuSources.forEach(laserName => {
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
      Object.values(laserTimeoutRefs.current).forEach(timeoutRef => {
        if (timeoutRef) {
          clearTimeout(timeoutRef);
        }
      });
    };
  }, []);

  // Fetch laser names, value ranges from backend and update Redux state
  useEffect(() => {
    async function fetchIlluminationData() {
      try {
        // Fetch laser names
        const namesRes = await fetch(`${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserNames`);
        const names = await namesRes.json();
        dispatch(parameterRangeSlice.setIlluSources(names));

        // Fetch value ranges for each laser
        const minArr = [];
        const maxArr = [];
        for (const name of names) {
          const encodedName = encodeURIComponent(name);
          // Value range
          const rangeRes = await fetch(`${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserValueRanges?laserName=${encodedName}`);
          const range = await rangeRes.json();
          minArr.push(range[0]);
          maxArr.push(range[1]);
          
          // Fetch initial laser state (power and active)
          try {
            const valueRes = await fetch(`${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserValue?laserName=${encodedName}`);
            const power = await valueRes.json();
            
            const activeRes = await fetch(`${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserActive?laserName=${encodedName}`);
            const enabled = await activeRes.json();
            
            // Initialize laser state in Redux
            dispatch(laserSlice.setLaserState({ laserName: name, power, enabled }));
          } catch (err) {
            console.warn(`Failed to fetch initial state for laser ${name}:`, err);
            // Initialize with defaults
            dispatch(laserSlice.setLaserState({ laserName: name, power: 0, enabled: false }));
          }
        }
        dispatch(parameterRangeSlice.setIlluSourceMinIntensities(minArr));
        dispatch(parameterRangeSlice.setIlluSourceMaxIntensities(maxArr));
      } catch (e) {
        console.error("Failed to fetch illumination data", e);
      }
    }
    fetchIlluminationData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hostIP, hostPort, dispatch]);


  // Debounced laser value update to prevent serial overload
  const debouncedSetLaserValue = useCallback((laserName, val) => {
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
          await fetch(`${ip}:${port}/imswitch/api/LaserController/setLaserValue?laserName=${encodedLaserName}&value=${val}`);
          console.log(`${laserName} intensity updated to: ${val}`);
        } catch (error) {
          console.error("Failed to set laser value in backend:", error);
        }
      }
    }, LASER_UPDATE_DEBOUNCE_MS);
  }, [dispatch, connectionSettingsState.ip, connectionSettingsState.apiPort, hostIP, hostPort]);

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
        await fetch(`${ip}:${port}/imswitch/api/LaserController/setLaserActive?laserName=${encodedLaserName}&active=${active}`);
        console.log(`${laserName} active state updated to: ${active}`);
      } catch (error) {
        console.error("Failed to set laser active state in backend:", error);
      }
    }
  };

  // Get laser data from Redux state
  const laserSources = parameterRangeState.illuSources || [];
  const laserMinValues = parameterRangeState.illuSourceMinIntensities || [];
  const laserMaxValues = parameterRangeState.illuSourceMaxIntensities || [];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Laser power and enabled state are updated in real-time via WebSocket.
        Adjust laser intensity and observe the color change on the detector stream.
      </Typography>
      
      <Grid container direction="column" spacing={2}>
        
        {laserSources.length ? (
          laserSources.map((laserName, idx) => {
            // Get laser state from Redux (updated via WebSocket)
            const laserData = lasers[laserName] || { power: 0, enabled: false };
            const currentValue = laserData.power;
            const isActive = laserData.enabled;
            const minValue = laserMinValues[idx] || 0;
            const maxValue = laserMaxValues[idx] || 1023;
            
            return (
              <Grid
                item
                key={laserName}
                sx={{ display: "flex", alignItems: "center", gap: 2 }}
              >
                {/* Laser name */}
                <Typography sx={{ minWidth: 120 }}>
                  {laserName}
                </Typography>

                {/* Slider with dynamic min and max */}
                <Box sx={{ flex: 1, px: 1 }}>
                  <Slider
                    value={currentValue}
                    min={minValue}
                    max={maxValue}
                    onChange={(e) => debouncedSetLaserValue(laserName, e.target.value)}
                    sx={{ width: '100%' }}
                    valueLabelDisplay="auto"
                  />
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="textSecondary">{minValue}</Typography>
                    <Typography variant="caption" color="textSecondary">{maxValue}</Typography>
                  </Box>
                </Box>

                {/* Current slider value */}
                <Typography sx={{ minWidth: 60, textAlign: 'center' }}>{currentValue}</Typography>

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
