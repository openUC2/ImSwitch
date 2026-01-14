// src/components/LiveViewSettings.js
import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../backendapi/createAxiosInstance';
import {
  Box,
  Typography,
  TextField,
  Slider,
  Card,
  CardContent,
  Checkbox,
  FormControlLabel,
  Grid,
} from "@mui/material";
import * as stormSlice from "../state/slices/STORMSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";

const LiveViewSettings = () => {
  const dispatch = useDispatch();

  // Redux state
  const stormState = useSelector(stormSlice.getSTORMState);
  const connectionSettingsState = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  const parameterRangeState = useSelector(parameterRangeSlice.getParameterRangeState);

  // Local state
  const [detectorGain, setDetectorGain] = useState(0);
  const [laserActiveStates, setLaserActiveStates] = useState({});

  // Redux state
  const exposureTime = stormState.exposureTime;
  const laserIntensities = stormState.laserIntensities;

  // Fetch initial detector gain
  useEffect(() => {
    const fetchDetectorGain = async () => {
      if (connectionSettingsState.ip && connectionSettingsState.apiPort) {
        try {
          const api = createAxiosInstance();
          const response = await api.get(
            `/SettingsController/getDetectorParameters`
          );
          const data = response.data;
          setDetectorGain(data.gain || 0);
        } catch (error) {
          console.error("Error fetching detector gain:", error);
        }
      }
    };

    fetchDetectorGain();
  }, [connectionSettingsState.ip, connectionSettingsState.apiPort]);

  // Initialize laser intensities when parameter range is available
  useEffect(() => {
    if (parameterRangeState.illuSources.length > 0) {
      const initializeLasers = async () => {
        const initialLaserIntensities = {};
        const initialLaserActiveStates = {};
        
        // Initialize all lasers with 0 if not already set
        parameterRangeState.illuSources.forEach((laserName) => {
          if (!(laserName in laserIntensities)) {
            initialLaserIntensities[laserName] = 0;
          }
          if (!(laserName in laserActiveStates)) {
            initialLaserActiveStates[laserName] = false;
          }
        });
        
        // Fetch current laser values from backend
        if (connectionSettingsState.ip && connectionSettingsState.apiPort) {
          try {
            const api = createAxiosInstance();
            const laserValues = await Promise.all(
              parameterRangeState.illuSources.map(async (laserName) => {
                try {
                  const encodedLaserName = encodeURIComponent(laserName);
                  const response = await api.get(
                    `/LaserController/getLaserValue?laserName=${encodedLaserName}`
                  );
                  
                  const value = response.data;
                  return { laserName, value: typeof value === 'number' ? value : 0 };
                } catch (error) {
                  console.warn(`Failed to fetch value for laser ${laserName}:`, error);
                }
                return { laserName, value: 0 };
              })
            );
            
            // Build laser intensities object
            const fetchedIntensities = {};
            laserValues.forEach(({ laserName, value }) => {
              fetchedIntensities[laserName] = value;
            });
            
            dispatch(stormSlice.setLaserIntensities({
              ...laserIntensities,
              ...initialLaserIntensities,
              ...fetchedIntensities
            }));
            
            setLaserActiveStates({
              ...laserActiveStates,
              ...initialLaserActiveStates
            });
          } catch (error) {
            console.error("Failed to fetch laser values:", error);
            // Just use initial values
            dispatch(stormSlice.setLaserIntensities({
              ...laserIntensities,
              ...initialLaserIntensities
            }));
            setLaserActiveStates({
              ...laserActiveStates,
              ...initialLaserActiveStates
            });
          }
        } else {
          // No connection, just use initial values
          dispatch(stormSlice.setLaserIntensities({
            ...laserIntensities,
            ...initialLaserIntensities
          }));
          setLaserActiveStates({
            ...laserActiveStates,
            ...initialLaserActiveStates
          });
        }
      };
      
      initializeLasers();
    }
  }, [dispatch, parameterRangeState.illuSources, connectionSettingsState.ip, connectionSettingsState.apiPort]);

  // --- Exposure Time and Gain boundaries from parameterRangeState ---
  const minExposure = parameterRangeState?.exposureTimeMin ?? 1;
  const maxExposure = parameterRangeState?.exposureTimeMax ?? 1000;
  const minGain = parameterRangeState?.detectorGainMin ?? 0;
  const maxGain = parameterRangeState?.detectorGainMax ?? 100;

  // --- Synchronized Exposure Time ---
  const setExposureTime = async (value) => {
    // Clamp value to min/max
    let newValue = Math.max(minExposure, Math.min(maxExposure, Number(value)));
    dispatch(stormSlice.setExposureTime(newValue));
    // Update backend immediately
    if (connectionSettingsState.ip && connectionSettingsState.apiPort) {
      try {
        const api = createAxiosInstance();
        await api.get(
          `/SettingsController/setDetectorExposureTime?exposureTime=${newValue}`
        );
      } catch (error) {
        console.error("Failed to update exposure time in backend:", error);
      }
    }
  };

  // Handle laser intensity changes
  const setLaserIntensity = async (laserName, value) => {
    dispatch(stormSlice.setLaserIntensity({ laserName, intensity: value }));
    
    // Also update backend immediately if connected
    if (connectionSettingsState.ip && connectionSettingsState.apiPort) {
      try {
        const api = createAxiosInstance();
        const encodedLaserName = encodeURIComponent(laserName);
        await api.get(
          `/LaserController/setLaserValue?laserName=${encodedLaserName}&value=${value}`
        );
      } catch (error) {
        console.error("Failed to update laser intensity in backend:", error);
      }
    }
  };

  // Handle laser active state changes
  const setLaserActive = async (laserName, active) => {
    setLaserActiveStates(prev => ({ ...prev, [laserName]: active }));
    
    // Also update backend immediately if connected
    if (connectionSettingsState.ip && connectionSettingsState.apiPort) {
      try {
        const api = createAxiosInstance();
        const encodedLaserName = encodeURIComponent(laserName);
        await api.get(
          `/LaserController/setLaserActive?laserName=${encodedLaserName}&active=${active}`
        );
      } catch (error) {
        console.error("Failed to update laser active state in backend:", error);
      }
    }
  };

  // Handle detector gain changes
  const updateDetectorGain = async (value) => {
    setDetectorGain(value);
    
    // Also update backend immediately if connected
    if (connectionSettingsState.ip && connectionSettingsState.apiPort) {
      try {
        const api = createAxiosInstance();
        await api.get(
          `/SettingsController/setDetectorGain?gain=${value}`
        );
      } catch (error) {
        console.error("Failed to update detector gain in backend:", error);
      }
    }
  };

  return (
    <Card sx={{ mt: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>Live View Settings</Typography>
        
        {/* Experiment Name */}
        <Box sx={{ mb: 2 }}>
          <TextField
            label="Experiment Name"
            value={stormState.experimentName}
            onChange={(e) => dispatch(stormSlice.setExperimentName(e.target.value))}
            fullWidth
            size="small"
          />
        </Box>

        {/* Exposure Time */}
        <Box sx={{ mb: 2 }}>
          <Typography gutterBottom variant="body2">
            Exposure Time: {exposureTime} ms
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Slider
              value={exposureTime}
              min={minExposure}
              max={maxExposure}
              onChange={(e, value) => {
                const v = Math.max(minExposure, Math.min(maxExposure, value));
                setExposureTime(v);
              }}
              valueLabelDisplay="auto"
              sx={{ flex: 1 }}
              size="small"
            />
            <TextField
              type="number"
              value={exposureTime}
              onChange={e => {
                const v = parseInt(e.target.value) || minExposure;
                setExposureTime(Math.max(minExposure, Math.min(maxExposure, v)));
              }}
              inputProps={{
                min: minExposure,
                max: maxExposure,
                style: { width: 60 }
              }}
              size="small"
            />
          </Box>
        </Box>

        {/* Detector Gain */}
        <Box sx={{ mb: 2 }}>
          <Typography gutterBottom variant="body2">
            Detector Gain: {detectorGain}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Slider
              value={detectorGain}
              min={minGain}
              max={maxGain}
              onChange={(e, value) => {
                const v = Math.max(minGain, Math.min(maxGain, value));
                updateDetectorGain(v);
              }}
              valueLabelDisplay="auto"
              sx={{ flex: 1 }}
              size="small"
            />
            <TextField
              type="number"
              value={detectorGain}
              onChange={e => {
                const v = parseInt(e.target.value) || minGain;
                updateDetectorGain(Math.max(minGain, Math.min(maxGain, v)));
              }}
              inputProps={{
                min: minGain,
                max: maxGain,
                style: { width: 60 }
              }}
              size="small"
            />
          </Box>
        </Box>

        {/* Laser Controls */}
        <Box>
          <Typography variant="body1" gutterBottom>
            Laser Control
          </Typography>
          {Object.entries(laserIntensities).map(([laserName, intensity]) => {
            const isActive = laserActiveStates[laserName] || false;
            const minValue = parameterRangeState.illuSourceMinIntensities?.[parameterRangeState.illuSources?.indexOf(laserName)] || 0;
            const maxValue = parameterRangeState.illuSourceMaxIntensities?.[parameterRangeState.illuSources?.indexOf(laserName)] || 1023;
            
            return (
              <Box key={laserName} sx={{ mb: 1 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="body2" sx={{ minWidth: 60 }}>
                    {laserName}
                  </Typography>
                  <Slider
                    value={intensity}
                    onChange={(e, value) => setLaserIntensity(laserName, value)}
                    min={minValue}
                    max={maxValue}
                    valueLabelDisplay="auto"
                    sx={{ flex: 1 }}
                    size="small"
                  />
                  <Typography variant="body2" sx={{ mx: 1, minWidth: 40 }}>
                    {intensity}
                  </Typography>
                  <Checkbox
                    checked={isActive}
                    onChange={(e) => setLaserActive(laserName, e.target.checked)}
                    size="small"
                  />
                </Box>
              </Box>
            );
          })}
          
          {Object.keys(laserIntensities).length === 0 && (
            <Typography variant="body2" color="textSecondary">
              No lasers configured. Laser controls will appear when connected to backend.
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default LiveViewSettings;
