import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Grid, 
  Paper,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useSelector } from 'react-redux';
import { getConnectionSettingsState } from '../../state/slices/ConnectionSettingsSlice';

import apiPositionerControllerHomeAxis from '../../backendapi/apiPositionerControllerHomeAxis';
import apiPixelCalibrationControllerOverviewVerifyHoming from '../../backendapi/apiPixelCalibrationControllerOverviewVerifyHoming';
import apiPixelCalibrationControllerOverviewStream from '../../backendapi/apiPixelCalibrationControllerOverviewStream';

/**
 * TestHomingTab - Axis homing verification
 * 
 * Features:
 * - Per-axis homing controls
 * - Overview camera display for visual verification
 * - Developer options for custom homing parameters
 * - Automated homing verification process
 */
const TestHomingTab = () => {
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  // Stream state
  const [streamUrl, setStreamUrl] = useState('');
  const [streamActive, setStreamActive] = useState(false);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [developerMode, setDeveloperMode] = useState(false);
  
  // Homing parameters - basic
  const [selectedAxis, setSelectedAxis] = useState('X');
  const [positionerName, setPositionerName] = useState('');
  
  // Homing parameters - advanced (developer mode)
  const [homeDirection, setHomeDirection] = useState('');
  const [homeSpeed, setHomeSpeed] = useState('');
  const [homeEndstopPolarity, setHomeEndstopPolarity] = useState('');
  const [homeEndposRelease, setHomeEndposRelease] = useState('');
  const [homeTimeout, setHomeTimeout] = useState('');
  
  // Verification parameters
  const [maxTimeS, setMaxTimeS] = useState(20.0);
  
  // Status and results
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  // Reference for MJPEG image
  const imgRef = useRef(null);

  // Set up stream URL
  useEffect(() => {
    if (hostIP && hostPort) {
      setStreamUrl(`${hostIP}:${hostPort}/imswitch/api/PixelCalibrationController/overviewStream`);
    }
  }, [hostIP, hostPort]);

  // Handle stream toggle
  const handleStreamToggle = async () => {
    try {
      const newStreamState = !streamActive;
      
      if (!newStreamState) {
        // Stop stream via API
        await apiPixelCalibrationControllerOverviewStream(false);
      }
      
      setStreamActive(newStreamState);
      setStatus(newStreamState ? 'Stream started' : 'Stream stopped');
    } catch (err) {
      setError(`Failed to toggle stream: ${err.message}`);
    }
  };

  // Home single axis
  const handleHomeAxis = async (axis) => {
    try {
      setLoading(true);
      setError('');
      setStatus(`Homing ${axis} axis...`);
      
      // Build parameters object
      const params = {
        axis,
        isBlocking: true // Wait for homing to complete
      };
      
      if (positionerName) params.positionerName = positionerName;
      
      // Add developer options if provided
      if (developerMode) {
        if (homeDirection !== '') params.homeDirection = parseInt(homeDirection);
        if (homeSpeed !== '') params.homeSpeed = parseFloat(homeSpeed);
        if (homeEndstopPolarity !== '') params.homeEndstoppolarity = parseInt(homeEndstopPolarity);
        if (homeEndposRelease !== '') params.homeEndposRelease = parseFloat(homeEndposRelease);
        if (homeTimeout !== '') params.homeTimeout = parseInt(homeTimeout);
      }
      
      const response = await apiPositionerControllerHomeAxis(params);
      
      setStatus(`${axis} axis homing complete`);
      setResult(response);
    } catch (err) {
      setError(`Failed to home ${axis} axis: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Run automated homing verification
  const handleVerifyHoming = async () => {
    try {
      setLoading(true);
      setError('');
      setStatus('Running homing verification...');
      
      const response = await apiPixelCalibrationControllerOverviewVerifyHoming(maxTimeS);
      
      setStatus('Homing verification complete');
      setResult(response);
    } catch (err) {
      setError(`Homing verification failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Left: Overview Camera */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Overview Camera
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Visual verification of stage position during homing
            </Typography>

            {/* Stream Control */}
            <Box sx={{ mb: 2 }}>
              <Button 
                variant="contained" 
                onClick={handleStreamToggle}
              >
                {streamActive ? 'Stop Stream' : 'Start Stream'}
              </Button>
            </Box>

            {/* MJPEG Stream Display */}
            <Box 
              sx={{ 
                backgroundColor: 'black', 
                minHeight: 500,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
              }}
            >
              {streamActive ? (
                <img
                  ref={imgRef}
                  src={streamUrl}
                  alt="Overview Camera"
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: 600,
                    objectFit: 'contain'
                  }}
                  onError={() => {
                    // Stream might not be active
                  }}
                />
              ) : (
                <Typography color="white">
                  Stream not active
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Right: Homing Controls */}
        <Grid item xs={12} md={5}>
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

          {/* Basic Homing Controls */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Axis Homing
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Home individual axes to calibrate stage position
            </Typography>

            <TextField
              label="Positioner Name (optional)"
              value={positionerName}
              onChange={(e) => setPositionerName(e.target.value)}
              fullWidth
              sx={{ mb: 2 }}
              placeholder="Leave empty for default"
            />

            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Button 
                  variant="contained" 
                  onClick={() => handleHomeAxis('X')}
                  disabled={loading}
                  fullWidth
                >
                  {loading && selectedAxis === 'X' ? (
                    <CircularProgress size={24} />
                  ) : (
                    'Home X Axis'
                  )}
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button 
                  variant="contained" 
                  onClick={() => handleHomeAxis('Y')}
                  disabled={loading}
                  fullWidth
                >
                  {loading && selectedAxis === 'Y' ? (
                    <CircularProgress size={24} />
                  ) : (
                    'Home Y Axis'
                  )}
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button 
                  variant="contained" 
                  onClick={() => handleHomeAxis('Z')}
                  disabled={loading}
                  fullWidth
                >
                  {loading && selectedAxis === 'Z' ? (
                    <CircularProgress size={24} />
                  ) : (
                    'Home Z Axis'
                  )}
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button 
                  variant="contained" 
                  onClick={() => handleHomeAxis('A')}
                  disabled={loading}
                  fullWidth
                >
                  {loading && selectedAxis === 'A' ? (
                    <CircularProgress size={24} />
                  ) : (
                    'Home A Axis'
                  )}
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {/* Developer Options */}
          <Accordion expanded={developerMode} onChange={() => setDeveloperMode(!developerMode)}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Developer Options</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Advanced Homing Parameters
                </Typography>
                
                <Typography variant="caption" display="block" sx={{ mb: 2, color: 'text.secondary' }}>
                  Leave fields empty to use default values
                </Typography>

                <TextField
                  label="Home Direction (-1, 0, 1)"
                  type="number"
                  value={homeDirection}
                  onChange={(e) => setHomeDirection(e.target.value)}
                  fullWidth
                  size="small"
                  sx={{ mb: 2 }}
                />

                <TextField
                  label="Home Speed"
                  type="number"
                  value={homeSpeed}
                  onChange={(e) => setHomeSpeed(e.target.value)}
                  fullWidth
                  size="small"
                  sx={{ mb: 2 }}
                />

                <TextField
                  label="Endstop Polarity (0 or 1)"
                  type="number"
                  value={homeEndstopPolarity}
                  onChange={(e) => setHomeEndstopPolarity(e.target.value)}
                  fullWidth
                  size="small"
                  sx={{ mb: 2 }}
                />

                <TextField
                  label="Endpos Release"
                  type="number"
                  value={homeEndposRelease}
                  onChange={(e) => setHomeEndposRelease(e.target.value)}
                  fullWidth
                  size="small"
                  sx={{ mb: 2 }}
                />

                <TextField
                  label="Timeout (seconds)"
                  type="number"
                  value={homeTimeout}
                  onChange={(e) => setHomeTimeout(e.target.value)}
                  fullWidth
                  size="small"
                />
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Automated Verification */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Automated Homing Verification
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Run comprehensive homing test for all axes
            </Typography>

            <TextField
              label="Max Time (seconds)"
              type="number"
              value={maxTimeS}
              onChange={(e) => setMaxTimeS(parseFloat(e.target.value))}
              fullWidth
              sx={{ mb: 2 }}
            />

            <Button 
              variant="contained" 
              color="success"
              onClick={handleVerifyHoming}
              disabled={loading}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Verify Homing'}
            </Button>
          </Paper>

          {/* Results Display */}
          {result && (
            <Paper sx={{ p: 2, mt: 2, backgroundColor: '#f5f5f5' }}>
              <Typography variant="h6" gutterBottom>
                Result
              </Typography>
              <pre style={{ fontSize: '0.85em', overflow: 'auto', maxHeight: 300 }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default TestHomingTab;
