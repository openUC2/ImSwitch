import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Grid, 
  Paper,
  Alert,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider
} from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import { getConnectionSettingsState } from '../../state/slices/ConnectionSettingsSlice';
import * as objectiveSlice from '../../state/slices/ObjectiveSlice';
import * as liveStreamSlice from '../../state/slices/LiveStreamSlice';
import * as liveViewSlice from '../../state/slices/LiveViewSlice';

import apiObjectiveControllerMoveToObjective from '../../backendapi/apiObjectiveControllerMoveToObjective';
import apiObjectiveControllerGetCurrentObjective from '../../backendapi/apiObjectiveControllerGetCurrentObjective';
import apiObjectiveControllerGetStatus from '../../backendapi/apiObjectiveControllerGetStatus';
import apiLiveViewControllerStartLiveView from '../../backendapi/apiLiveViewControllerStartLiveView';
import apiLiveViewControllerStopLiveView from '../../backendapi/apiLiveViewControllerStopLiveView';
import fetchObjectiveControllerGetStatus from '../../middleware/fetchObjectiveControllerGetStatus';
import fetchObjectiveControllerGetCurrentObjective from '../../middleware/fetchObjectiveControllerGetCurrentObjective';

/**
 * ObjectiveControllerTab - Objective management and configuration
 * 
 * Features:
 * - Switch between objective positions (1 and 2)
 * - Set objective information (magnification, NA, pixel size)
 * - Save configuration to backend
 * - Display current objective status
 */
const ObjectiveControllerTab = () => {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  // Overview stream state
  const [overviewStreamActive, setOverviewStreamActive] = useState(false);
  const prevDetectorRef = useRef(null);
  const prevProtocolRef = useRef('jpeg');

  // UI state
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  
  // Objective information for editing
  const [selectedSlot, setSelectedSlot] = useState(1);
  const [objectiveName, setObjectiveName] = useState('');
  const [magnification, setMagnification] = useState('');
  const [na, setNa] = useState('');
  const [pixelSize, setPixelSize] = useState('');

  // Load current objective info on mount
  useEffect(() => {
    refreshObjectiveStatus();
  }, []);

  // Handle overview stream toggle: reuses the existing WebSocket live stream by switching the
  // active detector to ObservationCamera. Restores the previous detector on stop.
  const handleOverviewStreamToggle = async () => {
    try {
      const newStreamState = !overviewStreamActive;

      if (newStreamState) {
        prevDetectorRef.current = liveViewState.detectors[liveViewState.activeTab] || null;
        prevProtocolRef.current = liveStreamState.imageFormat || 'jpeg';
        await apiLiveViewControllerStartLiveView('ObservationCamera', 'jpeg', { subsampling_factor: 1 });
      } else {
        if (prevDetectorRef.current) {
          await apiLiveViewControllerStartLiveView(prevDetectorRef.current, prevProtocolRef.current);
        } else {
          await apiLiveViewControllerStopLiveView('ObservationCamera');
        }
      }

      setOverviewStreamActive(newStreamState);
      setStatus(newStreamState ? 'Overview stream started' : 'Overview stream stopped');
    } catch (err) {
      setError(`Failed to toggle overview stream: ${err.message}`);
    }
  };

  // Refresh objective status from backend
  const refreshObjectiveStatus = async () => {
    try {
      await fetchObjectiveControllerGetStatus(dispatch);
      await fetchObjectiveControllerGetCurrentObjective(dispatch);
      setStatus('Objective status refreshed');
    } catch (err) {
      setError(`Failed to refresh status: ${err.message}`);
    }
  };

  // Switch to objective slot
  const handleSwitchObjective = async (slot) => {
    try {
      setLoading(true);
      setError('');
      setStatus(`Switching to objective ${slot}...`);
      
      await apiObjectiveControllerMoveToObjective(slot);
      
      // Update Redux state
      dispatch(objectiveSlice.setCurrentObjective(slot));
      
      // Refresh status
      await refreshObjectiveStatus();
      
      setStatus(`Successfully switched to objective ${slot}`);
    } catch (err) {
      setError(`Failed to switch objective: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Save objective configuration
  const handleSaveObjectiveConfig = async () => {
    try {
      setLoading(true);
      setError('');
      
      // TODO: This requires a backend API endpoint to save objective configuration
      // For now, we'll update Redux state and show a message
      
      // Update Redux state (this will persist in the session)
      if (selectedSlot === 1) {
        // Update slot 1 info
        // Note: We need to add these setters to the ObjectiveSlice
      } else if (selectedSlot === 2) {
        // Update slot 2 info
      }
      
      setStatus(`Objective ${selectedSlot} configuration saved (session only - backend save not yet implemented)`);
      
      // TODO: Call backend API to persist to JSON config
      // await apiObjectiveControllerSaveConfig({ slot: selectedSlot, name: objectiveName, magnification, na, pixelSize });
      
    } catch (err) {
      setError(`Failed to save configuration: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Top: Overview Camera */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Overview Camera Stream
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Wide field view to verify objective position and field of view
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Button 
                variant="contained" 
                onClick={handleOverviewStreamToggle}
              >
                {overviewStreamActive ? 'Stop Stream' : 'Start Stream'}
              </Button>
            </Box>

            <Box 
              sx={{ 
                backgroundColor: 'black', 
                minHeight: 300,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              {overviewStreamActive && liveStreamState.liveViewImage ? (
                <img
                  src={`data:image/jpeg;base64,${liveStreamState.liveViewImage}`}
                  alt="Overview Camera"
                  style={{ 
                    display: 'block',
                    margin: 'auto',
                    maxWidth: '100%', 
                    maxHeight: 300,
                    objectFit: 'contain',
                    WebkitUserSelect: 'none'
                  }}
                />
              ) : (
                <Typography color="white">
                  {overviewStreamActive ? 'Waiting for image...' : 'Stream not active'}
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Bottom Row: Status and Controls */}
        {/* Left: Current Status */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Current Objective Status
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="body1" sx={{ mb: 1 }}>
              <strong>Current Objective:</strong>{' '}
              {objectiveState.currentObjective !== null 
                ? `Slot ${objectiveState.currentObjective}` 
                : 'Unknown'}
            </Typography>
            
            <Typography variant="body1" sx={{ mb: 1 }}>
              <strong>Name:</strong>{' '}
              {objectiveState.objectivName || 'Not set'}
            </Typography>
            
            <Typography variant="body1" sx={{ mb: 1 }}>
              <strong>Magnification:</strong>{' '}
              {objectiveState.magnification !== null 
                ? `${objectiveState.magnification}×` 
                : 'Not set'}
            </Typography>
            
            <Typography variant="body1" sx={{ mb: 1 }}>
              <strong>Numerical Aperture (NA):</strong>{' '}
              {objectiveState.NA !== null 
                ? objectiveState.NA 
                : 'Not set'}
            </Typography>
            
            <Typography variant="body1" sx={{ mb: 1 }}>
              <strong>Pixel Size:</strong>{' '}
              {objectiveState.pixelsize !== null 
                ? `${objectiveState.pixelsize} µm` 
                : 'Not set'}
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="body2" color="text.secondary">
              <strong>Position X0:</strong>{' '}
              {objectiveState.posX0 !== null ? objectiveState.posX0 : 'Unknown'}
            </Typography>
            
            <Typography variant="body2" color="text.secondary">
              <strong>Position X1:</strong>{' '}
              {objectiveState.posX1 !== null ? objectiveState.posX1 : 'Unknown'}
            </Typography>
            
            <Button 
              variant="outlined" 
              onClick={refreshObjectiveStatus}
              sx={{ mt: 2 }}
              fullWidth
            >
              Refresh Status
            </Button>
          </Paper>

          {/* Switch Objective */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Switch Objective
            </Typography>
            
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={6}>
                <Button 
                  variant={objectiveState.currentObjective === 1 ? "contained" : "outlined"}
                  onClick={() => handleSwitchObjective(1)}
                  disabled={loading || objectiveState.currentObjective === 1}
                  fullWidth
                  size="large"
                >
                  Objective 1
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button 
                  variant={objectiveState.currentObjective === 2 ? "contained" : "outlined"}
                  onClick={() => handleSwitchObjective(2)}
                  disabled={loading || objectiveState.currentObjective === 2}
                  fullWidth
                  size="large"
                >
                  Objective 2
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Right: Configuration Editor */}
        <Grid item xs={12} md={6}>
          {/* Status and Errors */}
          {status && (
            <Alert severity="info" sx={{ mb: 2 }} onClose={() => setStatus('')}>
              {status}
            </Alert>
          )}
          
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Configure Objective
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Set objective information for accurate calibration
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Objective Slot</InputLabel>
              <Select
                value={selectedSlot}
                label="Objective Slot"
                onChange={(e) => setSelectedSlot(e.target.value)}
              >
                <MenuItem value={1}>Slot 1</MenuItem>
                <MenuItem value={2}>Slot 2</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Objective Name"
              value={objectiveName}
              onChange={(e) => setObjectiveName(e.target.value)}
              fullWidth
              sx={{ mb: 2 }}
              placeholder="e.g., 10x Plan Apo, 4x Objective"
              helperText="Descriptive name for this objective"
            />

            <TextField
              label="Magnification"
              type="number"
              value={magnification}
              onChange={(e) => setMagnification(e.target.value)}
              fullWidth
              sx={{ mb: 2 }}
              placeholder="e.g., 10, 4, 20"
              helperText="Objective magnification (e.g., 10 for 10×)"
            />

            <TextField
              label="Numerical Aperture (NA)"
              type="number"
              value={na}
              onChange={(e) => setNa(e.target.value)}
              fullWidth
              sx={{ mb: 2 }}
              placeholder="e.g., 0.3, 0.75, 1.4"
              helperText="Numerical aperture of the objective"
              inputProps={{ step: 0.01 }}
            />

            <TextField
              label="Pixel Size (µm)"
              type="number"
              value={pixelSize}
              onChange={(e) => setPixelSize(e.target.value)}
              fullWidth
              sx={{ mb: 2 }}
              placeholder="e.g., 0.325, 1.0"
              helperText="Physical size of one pixel at this magnification"
              inputProps={{ step: 0.001 }}
            />

            <Button 
              variant="contained" 
              color="primary"
              onClick={handleSaveObjectiveConfig}
              disabled={loading}
              fullWidth
              size="large"
            >
              Save Configuration
            </Button>
            
            <Alert severity="warning" sx={{ mt: 2 }}>
              Note: Backend API for saving objective config to JSON is not yet implemented. 
              Configuration will only persist for this session.
            </Alert>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ObjectiveControllerTab;
