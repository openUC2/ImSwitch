/**
 * AcceptanceTestComponent.jsx
 * 
 * Step-by-step wizard for acceptance testing of microscope hardware.
 * Tests motion, lighting, camera, and autofocus with user confirmation.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Grid,
  Alert,
  CircularProgress,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Divider,
} from '@mui/material';
import createAxiosInstance from '../backendapi/createAxiosInstance';
import {
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  PlayArrow as PlayArrowIcon,
  Home as HomeIcon,
  Navigation as NavigationIcon,
  Lightbulb as LightbulbIcon,
  Camera as CameraIcon,
  CenterFocusStrong as AutofocusIcon,
  Assessment as ReportIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { useDispatch, useSelector } from "react-redux";

// Import LiveView component
import LiveViewControlWrapper from '../axon/LiveViewControlWrapper';

import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
// Import API functions
import apiAcceptanceTestControllerHomeAxisX from '../backendapi/apiAcceptanceTestControllerHomeAxisX';
import apiAcceptanceTestControllerHomeAxisY from '../backendapi/apiAcceptanceTestControllerHomeAxisY';
import apiAcceptanceTestControllerHomeAxisZ from '../backendapi/apiAcceptanceTestControllerHomeAxisZ';
import apiAcceptanceTestControllerHomeAxisA from '../backendapi/apiAcceptanceTestControllerHomeAxisA';
import apiAcceptanceTestControllerMoveToTestPosition from '../backendapi/apiAcceptanceTestControllerMoveToTestPosition';
import apiAcceptanceTestControllerMoveXPlus from '../backendapi/apiAcceptanceTestControllerMoveXPlus';
import apiAcceptanceTestControllerMoveXMinus from '../backendapi/apiAcceptanceTestControllerMoveXMinus';
import apiAcceptanceTestControllerGetAvailableLightSources from '../backendapi/apiAcceptanceTestControllerGetAvailableLightSources';
import apiAcceptanceTestControllerTurnOnLight from '../backendapi/apiAcceptanceTestControllerTurnOnLight';
import apiAcceptanceTestControllerTurnOffLight from '../backendapi/apiAcceptanceTestControllerTurnOffLight';
import apiAcceptanceTestControllerGetCameraInfo from '../backendapi/apiAcceptanceTestControllerGetCameraInfo';
import apiAcceptanceTestControllerGetCurrentExposureAndGain from '../backendapi/apiAcceptanceTestControllerGetCurrentExposureAndGain';
import apiAcceptanceTestControllerRunAutofocus from '../backendapi/apiAcceptanceTestControllerRunAutofocus';
import apiAcceptanceTestControllerRecordTestResult from '../backendapi/apiAcceptanceTestControllerRecordTestResult';
import apiAcceptanceTestControllerGetTestReport from '../backendapi/apiAcceptanceTestControllerGetTestReport';
import apiAcceptanceTestControllerResetTestResults from '../backendapi/apiAcceptanceTestControllerResetTestResults';
import apiAcceptanceTestControllerSetLaserActive from '../backendapi/apiAcceptanceTestControllerSetLaserActive';

const AcceptanceTestComponent = () => {
  // Main stepper state
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Centralized decision storage - THIS IS THE KEY FIX
  // Structure: { testId: 'yes' | 'no' | 'skip' }
  const [decisions, setDecisions] = useState({});
  
  const connectionSettings = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  
  // Track which actions have been started (for enabling Yes/No buttons)
  // Structure: { testId: boolean }
  const [actionStarted, setActionStarted] = useState({});
  
  // Test-specific state
  const [lightSources, setLightSources] = useState([]);
  const [cameraInfo, setCameraInfo] = useState(null);
  const [currentTestAnimation, setCurrentTestAnimation] = useState(null);
  
  // Dialog state
  const [showAbortDialog, setShowAbortDialog] = useState(false);
  const [showReportDialog, setShowReportDialog] = useState(false);
  const [finalReport, setFinalReport] = useState(null);
  
  // Report state
  const [report, setReport] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);

  // Test steps configuration
  const steps = [
    {
      label: 'Motion: Homing',
      icon: <HomeIcon />,
      description: 'Test stage homing for all axes',
      category: 'motion'
    },
    {
      label: 'Motion: Positioning',
      icon: <NavigationIcon />,
      description: 'Test absolute and relative movements',
      category: 'motion'
    },
    {
      label: 'Lighting',
      icon: <LightbulbIcon />,
      description: 'Test light source control',
      category: 'lighting'
    },
    {
      label: 'Camera',
      icon: <CameraIcon />,
      description: 'Test camera detection and settings',
      category: 'camera'
    },
    {
      label: 'Autofocus',
      icon: <AutofocusIcon />,
      description: 'Test software autofocus',
      category: 'autofocus'
    },
    {
      label: 'Report',
      icon: <ReportIcon />,
      description: 'Generate acceptance test report',
      category: 'report'
    }
  ];

  // Initialize test on component mount
  useEffect(() => {
    initializeTest();
  }, []);

  const initializeTest = async () => {
    try {
      await apiAcceptanceTestControllerResetTestResults();
      setDecisions({});
      setActionStarted({});
    } catch (err) {
      console.error('Error initializing test:', err);
    }
  };

  // Central decision handler - THIS IS THE KEY FIX
  const setDecision = (testId, decision) => {
    console.log('Setting decision for', testId, ':', decision);
    setDecisions(prev => ({ ...prev, [testId]: decision }));
    
    // Also record to backend
    // Parse testId properly: motion_home_X -> category=motion, testName=home_X
    // motion_X -> category=motion, testName=X
    // lighting_LED_toggle -> category=lighting, testName=LED_toggle
    const parts = testId.split('_');
    const category = parts[0]; // 'motion', 'lighting', 'camera', 'autofocus'
    const testName = parts.slice(1).join('_'); // everything after first underscore
    const passed = decision === 'yes';
    const notes = decision === 'skip' ? 'Skipped' : '';
    
    console.log('Recording test result:', { category, testName, passed, notes });
    apiAcceptanceTestControllerRecordTestResult(category, testName, passed, notes)
      .catch(err => console.error('Error recording result:', err));
  };
  
  // Mark action as started
  const markActionStarted = (testId) => {
    console.log('Marking action started for', testId);
    setActionStarted(prev => ({ ...prev, [testId]: true }));
  };

  // Show animation for current test
  const showAnimation = (animationName) => {
    setCurrentTestAnimation(animationName);
  };

  const hideAnimation = () => {
    setCurrentTestAnimation(null);
  };

  // Generate report function (used by ReportStep and abort dialog)
  const generateReport = async () => {
    setLoadingReport(true);
    try {
      const response = await apiAcceptanceTestControllerGetTestReport();
      if (response.status === 'success') {
        setReport(response);
        setFinalReport(response);
      }
    } catch (err) {
      setError(`Error generating report: ${err.message}`);
    }
    setLoadingReport(false);
  };

  // ==================== Motion: Homing Step ====================
  const HomingStep = () => {
    const axes = ['X', 'Y', 'Z', 'A'];
    const testIds = axes.map(axis => `motion_home_${axis}`);

    const handleHomeAxis = async (axis) => {
      const testId = `motion_home_${axis}`;
      console.log('handleHomeAxis called for', axis, 'testId:', testId);
      
      setLoading(true);
      markActionStarted(testId);
      showAnimation(`home_${axis.toLowerCase()}`);
      
      try {
        let response;
        if (axis === 'X') response = await apiAcceptanceTestControllerHomeAxisX();
        else if (axis === 'Y') response = await apiAcceptanceTestControllerHomeAxisY();
        else if (axis === 'Z') response = await apiAcceptanceTestControllerHomeAxisZ();
        else if (axis === 'A') response = await apiAcceptanceTestControllerHomeAxisA();

        if (response.status === 'success') {
          setLoading(false);
          hideAnimation();
        } else {
          throw new Error(response.message);
        }
      } catch (err) {
        setError(`Error homing ${axis}: ${err.message}`);
        setLoading(false);
        hideAnimation();
      }
    };

    // Check if all axes have decisions
    const allHomingComplete = testIds.every(testId => decisions[testId] != null);
    const completedCount = testIds.filter(testId => decisions[testId] != null).length;

    return (
      <Box>
        <Alert severity="info" sx={{ mb: 2 }}>
          The motors don't have sensors to indicate they're running. Please watch the stage 
          and confirm whether each axis successfully moves to its home position.
        </Alert>

        {currentTestAnimation && (
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <img 
              src={`/assets/animations/${currentTestAnimation}.gif`} 
              alt="Homing animation"
              style={{ maxWidth: '300px', height: 'auto' }}
            />
          </Box>
        )}

        <Grid container spacing={2}>
          {axes.map(axis => {
            const testId = `motion_home_${axis}`;
            return (
            <Grid item xs={12} key={axis}>
              <Card variant="outlined">
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Typography variant="h6">Home {axis} Axis</Typography>
                    {decisions[testId] != null && (
                      <Chip
                        icon={decisions[testId] === 'yes' ? <CheckCircleIcon /> : <CancelIcon />}
                        label={decisions[testId] === 'skip' ? 'Skipped' : (decisions[testId] === 'yes' ? 'Success' : 'Failed')}
                        color={decisions[testId] === 'no' ? 'error' : 'success'}
                      />
                    )}
                  </Box>
                  
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      startIcon={<HomeIcon />}
                      onClick={() => handleHomeAxis(axis)}
                      disabled={loading || decisions[testId] != null}
                      fullWidth
                      sx={{ mb: 2 }}
                    >
                      Home {axis}
                    </Button>
                    
                    {loading && actionStarted[testId] && decisions[testId] == null && (
                      <Box sx={{ mb: 2 }}>
                        <LinearProgress sx={{ mb: 1 }} />
                        <Typography variant="caption" color="textSecondary" align="center" display="block">
                          Homing in progress...
                        </Typography>
                      </Box>
                    )}
                    
                    {decisions[testId] == null && (
                      <Box>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          Did the stage successfully home?
                        </Typography>
                        <Grid container spacing={1}>
                          <Grid item xs={4}>
                            <Button
                              variant="outlined"
                              color="success"
                              startIcon={<CheckCircleIcon />}
                              onClick={() => setDecision(testId, 'yes')}
                              disabled={!actionStarted[testId] || loading}
                              fullWidth
                            >
                              Yes
                            </Button>
                          </Grid>
                          <Grid item xs={4}>
                            <Button
                              variant="outlined"
                              color="error"
                              startIcon={<CancelIcon />}
                              onClick={() => setDecision(testId, 'no')}
                              disabled={!actionStarted[testId] || loading}
                              fullWidth
                            >
                              No
                            </Button>
                          </Grid>
                          <Grid item xs={4}>
                            <Button
                              variant="outlined"
                              color="warning"
                              onClick={() => setDecision(testId, 'skip')}
                              disabled={loading}
                              fullWidth
                            >
                              Skip
                            </Button>
                          </Grid>
                        </Grid>
                      </Box>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )})}
        </Grid>

        <Box sx={{ mt: 3 }}>
          <Alert severity={allHomingComplete ? 'success' : 'info'} sx={{ mb: 2 }}>
            {allHomingComplete 
              ? 'All axes tested! Click Next to continue.' 
              : `Progress: ${completedCount} / ${axes.length} axes completed`
            }
          </Alert>
          <Button
            variant="contained"
            color="primary"
            onClick={() => setActiveStep(prev => prev + 1)}
            disabled={!allHomingComplete}
            fullWidth
            size="large"
          >
            Next: Position Testing
          </Button>
        </Box>
      </Box>
    );
  };

  // ==================== Motion: Positioning Step ====================
  const PositioningStep = () => {
    // Separate test for each axis (X, Y, Z)
    const axes = [
      { name: 'X', label: 'left', distance: 10000 },
      { name: 'Y', label: 'up', distance: 10000 },
      { name: 'Z', label: 'down', distance: 10000 }
    ];
    const testIds = axes.map(axis => `motion_${axis.name}`);

    const handleMoveAxis = async (axisName) => {
      const testId = `motion_${axisName}`;
      setLoading(true);
      markActionStarted(testId);
      showAnimation(`move_${axisName.toLowerCase()}`);
      
      try {
        // Call API with only the specified axis moving
        const params = {};
        params[axisName.toLowerCase()] = 10000; // 10000 µm movement
        
        const response = await apiAcceptanceTestControllerMoveToTestPosition(
          null, 
          params.x || 0, 
          params.y || 0, 
          params.z || 0, 
          0
        );
        
        if (response.status === 'success') {
          setLoading(false);
          hideAnimation();
        } else {
          throw new Error(response.message);
        }
      } catch (err) {
        setError(`Error moving ${axisName}: ${err.message}`);
        setLoading(false);
        hideAnimation();
      }
    };

    const allPositioningComplete = testIds.every(testId => decisions[testId] != null);
    const completedCount = testIds.filter(testId => decisions[testId] != null).length;

    return (
      <Box>
        <Alert severity="info" sx={{ mb: 2 }}>
          Test each axis separately. Each axis will move 10000 µm. Please confirm the movement direction.
        </Alert>

        {currentTestAnimation && (
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <img 
              src={`/assets/animations/${currentTestAnimation}.gif`} 
              alt="Movement animation"
              style={{ maxWidth: '300px', height: 'auto' }}
            />
          </Box>
        )}

        <Grid container spacing={2}>
          {axes.map((axis) => {
            const testId = `motion_${axis.name}`;
            return (
              <Grid item xs={12} key={axis.name}>
                <Card variant="outlined">
                  <CardContent>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography variant="h6">
                        {axis.name} Axis Movement ({axis.distance} µm {axis.label})
                      </Typography>
                      {decisions[testId] != null && (
                        <Chip
                          icon={decisions[testId] === 'yes' ? <CheckCircleIcon /> : <CancelIcon />}
                          label={decisions[testId] === 'skip' ? 'Skipped' : (decisions[testId] === 'yes' ? 'Success' : 'Failed')}
                          color={decisions[testId] === 'no' ? 'error' : 'success'}
                        />
                      )}
                    </Box>
                    
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="contained"
                        startIcon={<NavigationIcon />}
                        onClick={() => handleMoveAxis(axis.name)}
                        disabled={loading || decisions[testId] != null}
                        fullWidth
                        sx={{ mb: 2 }}
                      >
                        Move {axis.name} Axis
                      </Button>
                      
                      {loading && actionStarted[testId] && decisions[testId] == null && (
                        <Box sx={{ mb: 2 }}>
                          <LinearProgress sx={{ mb: 1 }} />
                          <Typography variant="caption" color="textSecondary" align="center" display="block">
                            Moving {axis.name} axis...
                          </Typography>
                        </Box>
                      )}
                      
                      {decisions[testId] == null && (
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            Did the {axis.name} axis move {axis.label} correctly?
                          </Typography>
                          <Grid container spacing={1}>
                            <Grid item xs={4}>
                              <Button
                                variant="outlined"
                                color="success"
                                startIcon={<CheckCircleIcon />}
                                onClick={() => setDecision(testId, 'yes')}
                                disabled={!actionStarted[testId] || loading}
                                fullWidth
                              >
                                Yes
                              </Button>
                            </Grid>
                            <Grid item xs={4}>
                              <Button
                                variant="outlined"
                                color="error"
                                startIcon={<CancelIcon />}
                                onClick={() => setDecision(testId, 'no')}
                                disabled={!actionStarted[testId] || loading}
                                fullWidth
                              >
                                No
                              </Button>
                            </Grid>
                            <Grid item xs={4}>
                              <Button
                                variant="outlined"
                                color="warning"
                                onClick={() => setDecision(testId, 'skip')}
                                disabled={loading}
                                fullWidth
                              >
                                Skip
                              </Button>
                            </Grid>
                          </Grid>
                        </Box>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>

        <Box sx={{ mt: 3 }}>
          <Alert severity={allPositioningComplete ? 'success' : 'info'} sx={{ mb: 2 }}>
            {allPositioningComplete 
              ? 'All axis tests complete! Click Next to continue.' 
              : `Progress: ${completedCount} / ${axes.length} axes tested`
            }
          </Alert>
          <Button
            variant="contained"
            color="primary"
            onClick={() => setActiveStep(prev => prev + 1)}
            disabled={!allPositioningComplete}
            fullWidth
            size="large"
          >
            Next: Lighting Test
          </Button>
        </Box>
      </Box>
    );
  };

  // ==================== Lighting Step ====================
  const LightingStep = () => {
    const [loadingLights, setLoadingLights] = useState(false);

    useEffect(() => {
      // Only load if we don't have light sources yet
      if (lightSources.length === 0 && !loadingLights) {
        loadLightSources();
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Empty deps - only run on mount

    const loadLightSources = async () => {
      setLoadingLights(true);
      try {
        const response = await apiAcceptanceTestControllerGetAvailableLightSources();
        if (response.status === 'success') {
          setLightSources(response.light_sources);
        }
      } catch (err) {
        setError(`Error loading light sources: ${err.message}`);
      }
      setLoadingLights(false);
    };

    const handleToggleLight = async (laserName, turnOn) => {
      setLoading(true);
      showAnimation(turnOn ? 'light_on' : 'light_off');
      
      try {
        const response = turnOn
          ? await apiAcceptanceTestControllerTurnOnLight(laserName)
          : await apiAcceptanceTestControllerTurnOffLight(laserName);
        
        if (response.status === 'success') {
          setLoading(false);
          hideAnimation();
        } else {
          throw new Error(response.message);
        }
      } catch (err) {
        setError(`Error toggling light: ${err.message}`);
        setLoading(false);
        hideAnimation();
      }
    };

    // Decisions are now handled by the central setDecision function

    // Calculate completion based on decisions - now just one toggle test per source
    const requiredTestIds = lightSources.map(source => `lighting_${source.name}_toggle`);
    const completedTests = requiredTestIds.filter(testId => decisions[testId] != null).length;
    const allLightingComplete = requiredTestIds.length > 0 && completedTests >= requiredTestIds.length;

    return (
      <Box>
        <Alert severity="info" sx={{ mb: 2 }}>
          We'll turn each light source on and off. Please confirm if the light responds correctly.
        </Alert>

        {currentTestAnimation && (
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <img 
              src={`/assets/animations/${currentTestAnimation}.gif`} 
              alt="Lighting animation"
              style={{ maxWidth: '300px', height: 'auto' }}
            />
          </Box>
        )}

        {loadingLights ? (
          <CircularProgress />
        ) : (
          <Grid container spacing={2}>
            {lightSources.map(source => (
              <Grid item xs={12} key={source.name}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6">{source.name}</Typography>
                    
                    {/* Toggle ON/OFF test */}
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2">Toggle Test (ON → OFF)</Typography>
                      <Button
                        variant="contained"
                        startIcon={<LightbulbIcon />}
                        onClick={() => handleToggleLight(source.name, true)}
                        disabled={loading || decisions[`lighting_${source.name}_toggle`] != null}
                        fullWidth
                        sx={{ mb: 1, mt: 1 }}
                      >
                        Test Light (ON then OFF)
                      </Button>
                      
                      {!loading && decisions[`lighting_${source.name}_toggle`] == null && (
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            Did the light turn ON and then OFF correctly?
                          </Typography>
                          <Grid container spacing={1}>
                            <Grid item xs={6}>
                              <Button
                                variant="outlined"
                                color="success"
                                startIcon={<CheckCircleIcon />}
                                onClick={() => setDecision(`lighting_${source.name}_toggle`, 'yes')}
                                fullWidth
                              >
                                Yes
                              </Button>
                            </Grid>
                            <Grid item xs={6}>
                              <Button
                                variant="outlined"
                                color="error"
                                startIcon={<CancelIcon />}
                                onClick={() => setDecision(`lighting_${source.name}_toggle`, 'no')}
                                fullWidth
                              >
                                No
                              </Button>
                            </Grid>
                          </Grid>
                        </Box>
                      )}
                      {decisions[`lighting_${source.name}_toggle`] != null && (
                        <Chip
                          icon={decisions[`lighting_${source.name}_toggle`] === 'yes' ? <CheckCircleIcon /> : <CancelIcon />}
                          label={decisions[`lighting_${source.name}_toggle`] === 'yes' ? 'Success' : 'Failed'}
                          color={decisions[`lighting_${source.name}_toggle`] === 'yes' ? 'success' : 'error'}
                          sx={{ mt: 1 }}
                        />
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}

        {allLightingComplete && (
          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => setActiveStep(prev => prev + 1)}
              fullWidth
            >
              Next: Camera Test
            </Button>
          </Box>
        )}
      </Box>
    );
  };

  // ==================== Camera Step ====================
  const CameraStep = () => {
    const [loadingCamera, setLoadingCamera] = useState(false);
    const [exposure, setExposure] = useState('');
    const [gain, setGain] = useState('');

    useEffect(() => {
      // Only load if we don't have camera info yet
      if (!cameraInfo && !loadingCamera) {
        loadCameraInfo();
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const loadCameraInfo = async () => {
      setLoadingCamera(true);
      try {
        const response = await apiAcceptanceTestControllerGetCameraInfo();
        if (response.status === 'success') {
          setCameraInfo(response.cameras);
        }
      } catch (err) {
        setError(`Error loading camera info: ${err.message}`);
      }
      setLoadingCamera(false);
    };

    // Decisions are now handled by the central setDecision function
    const cameraTestIds = ['camera_specs', 'camera_imageVisible', 'camera_exposureGain'];
    const allCameraComplete = cameraTestIds.every(testId => decisions[testId] != null);

    return (
      <Box>
        <Alert severity="info" sx={{ mb: 2 }}>
          Please insert a sample and verify that the camera is working correctly.
        </Alert>

        {/* Display camera specifications */}
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Camera Specifications</Typography>
            {loadingCamera ? (
              <CircularProgress size={24} />
            ) : cameraInfo && cameraInfo.length > 0 ? (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Model</TableCell>
                      <TableCell>Resolution</TableCell>
                      <TableCell>Pixel Size</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {cameraInfo.map((camera, idx) => (
                      <TableRow key={idx}>
                        <TableCell>{camera.name}</TableCell>
                        <TableCell>{camera.model || 'N/A'}</TableCell>
                        <TableCell>
                          {camera.width_pixels && camera.height_pixels
                            ? `${camera.width_pixels} x ${camera.height_pixels}`
                            : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {camera.pixel_size_um ? `${camera.pixel_size_um} µm` : 'N/A'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography>No camera detected</Typography>
            )}
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Are the camera specifications correct?
              </Typography>
              {decisions['camera_specs'] == null && (
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Button
                      variant="outlined"
                      color="success"
                      startIcon={<CheckCircleIcon />}
                      onClick={() => setDecision('camera_specs', 'yes')}
                      fullWidth
                    >
                      Yes
                    </Button>
                  </Grid>
                  <Grid item xs={6}>
                    <Button
                      variant="outlined"
                      color="error"
                      startIcon={<CancelIcon />}
                      onClick={() => setDecision('camera_specs', 'no')}
                      fullWidth
                    >
                      No
                    </Button>
                  </Grid>
                </Grid>
              )}
              {decisions['camera_specs'] != null && (
                <Chip
                  icon={decisions['camera_specs'] === 'yes' ? <CheckCircleIcon /> : <CancelIcon />}
                  label={decisions['camera_specs'] === 'yes' ? 'Correct' : 'Incorrect'}
                  color={decisions['camera_specs'] === 'yes' ? 'success' : 'error'}
                />
              )}
            </Box>
          </CardContent>
        </Card>

        {/* Image visibility test */}
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Image Visibility</Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Please check the live view. Is an image visible from the camera?
            </Typography>
            
            {/* Live View */}
            {decisions['camera_specs'] != null && (
              <Box sx={{ mb: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                <LiveViewControlWrapper />
              </Box>
            )}
            
            {decisions['camera_imageVisible'] == null && decisions['camera_specs'] != null && (
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Button
                    variant="outlined"
                    color="success"
                    startIcon={<CheckCircleIcon />}
                    onClick={() => setDecision('camera_imageVisible', 'yes')}
                    fullWidth
                  >
                    Yes, image visible
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<CancelIcon />}
                    onClick={() => setDecision('camera_imageVisible', 'no')}
                    fullWidth
                  >
                    No image
                  </Button>
                </Grid>
              </Grid>
            )}
            {decisions['camera_imageVisible'] != null && (
              <Chip
                icon={decisions['camera_imageVisible'] === 'yes' ? <CheckCircleIcon /> : <CancelIcon />}
                label={decisions['camera_imageVisible'] === 'yes' ? 'Image visible' : 'No image'}
                color={decisions['camera_imageVisible'] === 'yes' ? 'success' : 'error'}
              />
            )}
          </CardContent>
        </Card>

        {/* Exposure and gain test */}
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Exposure & Gain Control</Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Use the joystick to move the stage and find a feature. Then adjust exposure and gain 
              to ensure the image isn't overexposed. Do the controls work properly?
            </Typography>
            
            {/* Illumination Controls for Stream Testing */}
            {decisions['camera_imageVisible'] != null && decisions['camera_exposureGain'] == null && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
                  Stream & Illumination Controls
                </Typography>
                
                {/* Stream Controls */}
                <Box sx={{ mb: 2 }}>
                  <Typography variant="caption" color="textSecondary" sx={{ mb: 1, display: 'block' }}>
                    Camera Stream (automatically running above)
                  </Typography>
                </Box>
                
                {/* Light Source Controls */}
                <Typography variant="caption" color="textSecondary" sx={{ mb: 1, display: 'block' }}>
                  Toggle Light Sources:
                </Typography>
                <Grid container spacing={1} sx={{ mb: 2 }}>
                  {lightSources.map(source => (
                    <Grid item xs={6} sm={4} key={source.name}>
                      <Button
                        variant={source.enabled ? "contained" : "outlined"}
                        size="small"
                        onClick={async () => {
                          try {
                            // Toggle the light
                            const newState = !source.enabled;
                            await apiAcceptanceTestControllerSetLaserActive(source.name, newState, 255);
                            // Update local state
                            source.enabled = newState;
                            // Force re-render
                            setLightSources([...lightSources]);
                          } catch (err) {
                            console.error('Error toggling light:', err);
                          }
                        }}
                        fullWidth
                      >
                        {source.name} {source.enabled ? '(ON)' : '(OFF)'}
                      </Button>
                    </Grid>
                  ))}
                </Grid>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
                  Exposure & Gain Controls
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" gutterBottom>Exposure Time</Typography>
                    <input
                      type="number"
                      value={exposure}
                      onChange={async (e) => {
                        const value = e.target.value;
                        setExposure(value);
                        try {
                          const api = createAxiosInstance();
                          await api.get(`/SettingsController/setDetectorExposureTime?exposureTime=${value}`);
                        } catch (err) {
                          console.error('Error setting exposure:', err);
                        }
                      }}
                      style={{ width: '100%', padding: '8px', fontSize: '14px' }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" gutterBottom>Gain</Typography>
                    <input
                      type="number"
                      value={gain}
                      onChange={async (e) => {
                        const value = e.target.value;
                        setGain(value);
                        try {
                          const api = createAxiosInstance();
                          await api.get(`/SettingsController/setDetectorGain?gain=${value}`);
                        } catch (err) {
                          console.error('Error setting gain:', err);
                        }
                      }}
                      style={{ width: '100%', padding: '8px', fontSize: '14px' }}
                    />
                  </Grid>
                </Grid>
              </Box>
            )}
            
            {decisions['camera_exposureGain'] == null && decisions['camera_imageVisible'] != null && (
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Button
                    variant="outlined"
                    color="success"
                    startIcon={<CheckCircleIcon />}
                    onClick={() => setDecision('camera_exposureGain', 'yes')}
                    fullWidth
                  >
                    Yes, working
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<CancelIcon />}
                    onClick={() => setDecision('camera_exposureGain', 'no')}
                    fullWidth
                  >
                    Not working
                  </Button>
                </Grid>
              </Grid>
            )}
            {decisions['camera_exposureGain'] != null && (
              <Chip
                icon={decisions['camera_exposureGain'] === 'yes' ? <CheckCircleIcon /> : <CancelIcon />}
                label={decisions['camera_exposureGain'] === 'yes' ? 'Working' : 'Not working'}
                color={decisions['camera_exposureGain'] === 'yes' ? 'success' : 'error'}
              />
            )}
          </CardContent>
        </Card>

        {allCameraComplete && (
          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => setActiveStep(prev => prev + 1)}
              fullWidth
            >
              Next: Autofocus Test
            </Button>
          </Box>
        )}
      </Box>
    );
  };

  // ==================== Autofocus Step ====================
  const AutofocusStep = () => {
    const [runningAutofocus, setRunningAutofocus] = useState(false);

    const handleRunAutofocus = async () => {
      setRunningAutofocus(true);
      showAnimation('autofocus');
      
      try {
        const response = await apiAcceptanceTestControllerRunAutofocus();
        if (response.status === 'success' || response.status === 'error') {
          setRunningAutofocus(false);
          hideAnimation();
        }
      } catch (err) {
        setError(`Error running autofocus: ${err.message}`);
        setRunningAutofocus(false);
        hideAnimation();
      }
    };

    // Decisions are now handled by the central setDecision function

    return (
      <Box>
        <Alert severity="info" sx={{ mb: 2 }}>
          We'll test the software-based autofocus. This should automatically focus the image.
        </Alert>

        {currentTestAnimation && (
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <img 
              src={`/assets/animations/${currentTestAnimation}.gif`} 
              alt="Autofocus animation"
              style={{ maxWidth: '300px', height: 'auto' }}
            />
          </Box>
        )}

        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>Software Autofocus</Typography>
            
            <Button
              variant="contained"
              startIcon={<AutofocusIcon />}
              onClick={handleRunAutofocus}
              disabled={runningAutofocus || decisions['autofocus_test'] != null}
              fullWidth
              sx={{ mb: 2 }}
            >
              Run Autofocus
            </Button>

            {runningAutofocus && <LinearProgress sx={{ mb: 2 }} />}
            
            {!runningAutofocus && decisions['autofocus_test'] == null && (
              <Box>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Did the autofocus work correctly?
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Button
                      variant="outlined"
                      color="success"
                      startIcon={<CheckCircleIcon />}
                      onClick={() => setDecision('autofocus_test', 'yes')}
                      fullWidth
                    >
                      Yes
                    </Button>
                  </Grid>
                  <Grid item xs={6}>
                    <Button
                      variant="outlined"
                      color="error"
                      startIcon={<CancelIcon />}
                      onClick={() => setDecision('autofocus_test', 'no')}
                      fullWidth
                    >
                      No
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            )}
            
            {decisions['autofocus_test'] != null && (
              <Chip
                icon={decisions['autofocus_test'] === 'yes' ? <CheckCircleIcon /> : <CancelIcon />}
                label={decisions['autofocus_test'] === 'yes' ? 'Success' : 'Failed'}
                color={decisions['autofocus_test'] === 'yes' ? 'success' : 'error'}
              />
            )}
          </CardContent>
        </Card>

        {decisions['autofocus_test'] != null && (
          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => setActiveStep(prev => prev + 1)}
              fullWidth
            >
              Generate Report
            </Button>
          </Box>
        )}
      </Box>
    );
  };

  // ==================== Report Step ====================
  const ReportStep = () => {
    useEffect(() => {
      // Only generate report if we don't have one yet
      if (!report && !loadingReport) {
        generateReport();
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Only run on mount

    const downloadReport = () => {
      if (!report) return;

      // Create a formatted text report
      let reportText = '=== MICROSCOPE ACCEPTANCE TEST REPORT ===\n\n';
      reportText += `Date: ${new Date().toLocaleString()}\n\n`;
      reportText += `Summary:\n`;
      reportText += `  Total Tests: ${report.summary.total_tests}\n`;
      reportText += `  Passed: ${report.summary.passed_tests}\n`;
      reportText += `  Failed: ${report.summary.failed_tests}\n`;
      reportText += `  Pass Rate: ${report.summary.pass_rate.toFixed(1)}%\n\n`;
      
      reportText += '=== DETAILED RESULTS ===\n\n';
      
      if (report.test_results) {
        Object.entries(report.test_results).forEach(([category, tests]) => {
          if (typeof tests === 'object' && Object.keys(tests).length > 0) {
          reportText += `${category.toUpperCase()}:\n`;
          Object.entries(tests).forEach(([testName, result]) => {
            if (typeof result === 'object' && 'passed' in result) {
              reportText += `  ${testName}: ${result.passed ? 'PASS' : 'FAIL'}`;
              if (result.notes) reportText += ` (${result.notes})`;
              reportText += '\n';
            }
          });
            reportText += '\n';
          }
        });
      }

      // Create download
      const blob = new Blob([reportText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `acceptance_test_report_${Date.now()}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };

    return (
      <Box>
        <Alert severity="success" sx={{ mb: 2 }}>
          Acceptance testing complete! Review the results below.
        </Alert>

        {loadingReport ? (
          <Box sx={{ textAlign: 'center', py: 3 }}>
            <CircularProgress />
          </Box>
        ) : report ? (
          <>
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Test Summary</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="textSecondary">Total Tests</Typography>
                    <Typography variant="h4">{report.summary.total_tests}</Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="textSecondary">Passed</Typography>
                    <Typography variant="h4" color="success.main">{report.summary.passed_tests}</Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="textSecondary">Failed</Typography>
                    <Typography variant="h4" color="error.main">{report.summary.failed_tests}</Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="textSecondary">Pass Rate</Typography>
                    <Typography variant="h4">{report.summary.pass_rate.toFixed(1)}%</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>Detailed Results</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Category</TableCell>
                        <TableCell>Test</TableCell>
                        <TableCell>Result</TableCell>
                        <TableCell>Notes</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {report.test_results && Object.entries(report.test_results).map(([category, tests]) => {
                        if (typeof tests !== 'object') return null;
                        return Object.entries(tests).map(([testName, result]) => {
                          if (typeof result !== 'object' || !('passed' in result)) return null;
                          return (
                            <TableRow key={`${category}_${testName}`}>
                              <TableCell>{category}</TableCell>
                              <TableCell>{testName}</TableCell>
                              <TableCell>
                                <Chip
                                  icon={result.passed ? <CheckCircleIcon /> : <CancelIcon />}
                                  label={result.passed ? 'PASS' : 'FAIL'}
                                  color={result.passed ? 'success' : 'error'}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell>{result.notes || '-'}</TableCell>
                            </TableRow>
                          );
                        });
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>

            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                startIcon={<DownloadIcon />}
                onClick={downloadReport}
                fullWidth
              >
                Download Report (TXT)
              </Button>
              <Button
                variant="outlined"
                onClick={() => {
                  setActiveStep(0);
                  initializeTest();
                }}
                fullWidth
              >
                Start New Test
              </Button>
            </Box>
          </>
        ) : (
          <Alert severity="error">Failed to generate report</Alert>
        )}
      </Box>
    );
  };

  // Render step content
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return <HomingStep />;
      case 1:
        return <PositioningStep />;
      case 2:
        return <LightingStep />;
      case 3:
        return <CameraStep />;
      case 4:
        return <AutofocusStep />;
      case 5:
        return <ReportStep />;
      default:
        return <Typography>Unknown step</Typography>;
    }
  };

  // Main render
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Microscope Acceptance Test
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Step-by-step validation of microscope hardware and software functionality.
        You can abort the test at any time.
      </Typography>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Stepper activeStep={activeStep} orientation="vertical">
        {steps.map((step, index) => (
          <Step key={step.label}>
            <StepLabel
              icon={step.icon}
              optional={
                index === steps.length - 1 ? (
                  <Typography variant="caption">Final step</Typography>
                ) : null
              }
            >
              {step.label}
            </StepLabel>
            <StepContent>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                {step.description}
              </Typography>
              {getStepContent(index)}
              
              {index > 0 && index < steps.length - 1 && (
                <Box sx={{ mt: 2 }}>
                  <Button
                    onClick={() => setActiveStep(prev => prev - 1)}
                    sx={{ mr: 1 }}
                  >
                    Back
                  </Button>
                  <Button
                    variant="outlined"
                    color="error"
                    onClick={() => setShowAbortDialog(true)}
                  >
                    Abort Test
                  </Button>
                </Box>
              )}
            </StepContent>
          </Step>
        ))}
      </Stepper>

      {/* Abort confirmation dialog */}
      <Dialog open={showAbortDialog} onClose={() => setShowAbortDialog(false)}>
        <DialogTitle>Abort Acceptance Test?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to abort the acceptance test? Your progress will be saved,
            but you'll need to restart from the beginning for a complete test.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAbortDialog(false)}>Cancel</Button>
          <Button
            onClick={async () => {
              await generateReport();
              setShowAbortDialog(false);
              setActiveStep(5); // Jump to report
            }}
            color="error"
            variant="contained"
          >
            Abort & Generate Report
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AcceptanceTestComponent;
