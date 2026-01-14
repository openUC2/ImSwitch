// src/components/LightsheetController.js
import CancelIcon from "@mui/icons-material/Cancel";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import ViewInArIcon from "@mui/icons-material/ViewInAr";
import DownloadIcon from "@mui/icons-material/Download";
import FlipIcon from "@mui/icons-material/Flip";
import RotateRightIcon from "@mui/icons-material/RotateRight";
import VideocamIcon from "@mui/icons-material/Videocam";
import VideocamOffIcon from "@mui/icons-material/VideocamOff";
import {
  Box,
  Button,
  FormControl,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Tab,
  Tabs,
  TextField,
  Typography,
  Alert,
  Chip,
  Slider,
  Switch,
  FormControlLabel,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
  Tooltip,
  Divider,
} from "@mui/material";
import { green, red, orange } from "@mui/material/colors";
import { useCallback, useEffect, useState, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as lightsheetSlice from "../state/slices/LightsheetSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import ErrorBoundary from "./ErrorBoundary.js";
import VtkViewer from "./VtkViewer.js";
import Lightsheet3DViewer from "./Lightsheet3DViewer.jsx";
import AxisConfigurationMenu from "./AxisConfigurationMenu.jsx";
import LightsheetPositionControls from "./LightsheetPositionControls.jsx";
import VizarrViewer from "./VizarrViewer.jsx";
import apiPositionerControllerGetPositions from "../backendapi/apiPositionerControllerGetPositions.js";
import {
  apiStartStepAcquireScan,
  apiStartContinuousScanWithZarr,
  apiGetScanStatus,
  apiGetAvailableScanModes,
  apiGetAvailableStorageFormats,
  apiGetLatestZarrPath,
  apiGetObjectiveFOV,
} from "../backendapi/apiLightsheetController.js";
import {
  apiLightsheetControllerObservationStreamControl,
  apiLightsheetControllerSetObservationExposure,
  apiLightsheetControllerGetObservationExposure,
  apiLightsheetControllerSetObservationGain,
  apiLightsheetControllerGetObservationGain,
  apiLightsheetControllerSetStreamTransform,
} from "../backendapi/apiLightsheetControllerObservationStream.js";
import {
  apiGetCameraExposureTime,
  apiSetCameraExposureTime,
  apiGetCameraGain,
  apiSetCameraGain,
} from "../backendapi/apiLightsheetCameraSettings.js";

// Import Socket.IO client for real-time updates
import { io } from "socket.io-client";

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );
};

/**
 * ImSwitch Lightsheet Controller Component
 * Manages 3D lightsheet microscopy scanning and visualization
 * Follows Copilot Instructions for Redux state management and API communication
 * 
 * Features:
 * - Continuous scan mode (original fast scan)
 * - Step-Acquire mode (Go-Stop-Acquire for high-quality Z-stacks)
 * - OME-Zarr and TIFF storage formats
 * - Real-time progress updates via Socket.IO
 * - 3D visualization using VizarrViewer
 */
const LightsheetController = () => {
  // Access ImSwitch backend connection settings from Redux - following Copilot Instructions
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const hostIP = connectionSettingsState.ip;
  const hostPort = connectionSettingsState.apiPort;

  // Redux dispatcher and lightsheet state
  const dispatch = useDispatch();

  // Access global Redux state
  const lightsheetState = useSelector(lightsheetSlice.getLightsheetState);

  // Use Redux state instead of local useState
  const tabIndex = lightsheetState.tabIndex;
  const minPos = lightsheetState.minPos;
  const maxPos = lightsheetState.maxPos;
  const speed = lightsheetState.speed;
  const stepSize = lightsheetState.stepSize;
  const axis = lightsheetState.axis;
  const illuSource = lightsheetState.illuSource;
  const illuValue = lightsheetState.illuValue;
  const isRunning = lightsheetState.isRunning;
  const scanMode = lightsheetState.scanMode;
  const storageFormat = lightsheetState.storageFormat;
  const experimentName = lightsheetState.experimentName;
  const scanStatus = lightsheetState.scanStatus;
  const availableScanModes = lightsheetState.availableScanModes;
  const availableStorageFormats = lightsheetState.availableStorageFormats;
  const latestZarrPath = lightsheetState.latestZarrPath;
  
  // Tiling and timelapse state from Redux
  const enableTiling = lightsheetState.enableTiling;
  const tilesXPositive = lightsheetState.tilesXPositive;
  const tilesXNegative = lightsheetState.tilesXNegative;
  const tilesYPositive = lightsheetState.tilesYPositive;
  const tilesYNegative = lightsheetState.tilesYNegative;
  const tileStepSizeX = lightsheetState.tileStepSizeX;
  const tileStepSizeY = lightsheetState.tileStepSizeY;
  const tileOverlap = lightsheetState.tileOverlap;
  const timepoints = lightsheetState.timepoints;
  const timeLapsePeriod = lightsheetState.timeLapsePeriod;
  const objectiveFOV = lightsheetState.objectiveFOV;

  // Local state for socket connection
  const [socketConnected, setSocketConnected] = useState(false);
  const [showZarrViewer, setShowZarrViewer] = useState(false);

  // Observation camera stream state
  const [observationStreamActive, setObservationStreamActive] = useState(false);
  const [observationStreamUrl, setObservationStreamUrl] = useState('');
  const [observationExposure, setObservationExposure] = useState(50);
  const [observationGain, setObservationGain] = useState(1);
  const [flipX, setFlipX] = useState(false);
  const [flipY, setFlipY] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [streamError, setStreamError] = useState('');
  const observationImgRef = useRef(null);

  // Camera settings state for live view (2D lightsheet camera)
  const [cameraExposure, setCameraExposure] = useState(100);
  const [cameraGain, setCameraGain] = useState(0);

  // Initialize Socket.IO connection for real-time updates
  useEffect(() => {
    if (!hostIP || !hostPort) return;

    // Build socket URL
    const socketUrl = `${hostIP}:${hostPort}`;
    const socket = io(socketUrl, {
      path: '/imswitch/socket.io/',
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socket.on('connect', () => {
      console.log('Lightsheet Controller: Socket.IO connected');
      setSocketConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('Lightsheet Controller: Socket.IO disconnected');
      setSocketConnected(false);
    });

    // Listen for lightsheet status updates from backend
    socket.on('lightsheet_status', (data) => {
      console.log('Lightsheet status update:', data);
      dispatch(lightsheetSlice.setScanStatus(data));
    });

    return () => {
      socket.disconnect();
    };
  }, [hostIP, hostPort, dispatch]);

  // Set up observation stream URL
  useEffect(() => {
    if (hostIP && hostPort) {
      setObservationStreamUrl(`${hostIP}:${hostPort}/imswitch/api/LightsheetController/observationStream`);
    }
  }, [hostIP, hostPort]);

  // Fetch initial observation camera settings
  useEffect(() => {
    const fetchCameraSettings = async () => {
      try {
        const exposure = await apiLightsheetControllerGetObservationExposure();
        if (typeof exposure === 'number') {
          setObservationExposure(exposure);
        }
        const gain = await apiLightsheetControllerGetObservationGain();
        if (typeof gain === 'number') {
          setObservationGain(gain);
        }
      } catch (err) {
        console.warn('Failed to fetch observation camera settings:', err);
      }
    };
    
    if (hostIP && hostPort) {
      fetchCameraSettings();
    }
  }, [hostIP, hostPort]);

  // Fetch initial live view camera settings (2D lightsheet camera)
  useEffect(() => {
    const fetchLiveViewCameraSettings = async () => {
      try {
        const exposureResult = await apiGetCameraExposureTime();
        if (exposureResult.success && typeof exposureResult.exposureTime === 'number') {
          setCameraExposure(exposureResult.exposureTime);
        }
        const gainResult = await apiGetCameraGain();
        if (gainResult.success && typeof gainResult.gain === 'number') {
          setCameraGain(gainResult.gain);
        }
      } catch (err) {
        console.warn('Failed to fetch live view camera settings:', err);
      }
    };
    
    if (hostIP && hostPort) {
      fetchLiveViewCameraSettings();
    }
  }, [hostIP, hostPort]);

  // Handle observation stream toggle
  const handleObservationStreamToggle = async () => {
    try {
      setStreamError('');
      const newState = !observationStreamActive;
      
      if (newState) {
        // Start stream - simply toggle state, img tag will connect automatically
        setObservationStreamActive(true);
      } else {
        // Stop stream via API
        try {
          await apiLightsheetControllerObservationStreamControl(false);
        } catch (err) {
          console.warn('Error stopping stream:', err);
        }
        setObservationStreamActive(false);
      }
    } catch (err) {
      setStreamError(`Failed to ${observationStreamActive ? 'stop' : 'start'} stream: ${err.message}`);
    }
  };

  // Handle exposure change
  const handleExposureChange = async (event, value) => {
    setObservationExposure(value);
  };

  const handleExposureChangeCommitted = async (event, value) => {
    try {
      await apiLightsheetControllerSetObservationExposure(value);
    } catch (err) {
      console.error('Failed to set exposure:', err);
    }
  };

  // Handle gain change
  const handleGainChange = async (event, value) => {
    setObservationGain(value);
  };

  const handleGainChangeCommitted = async (event, value) => {
    try {
      await apiLightsheetControllerSetObservationGain(value);
    } catch (err) {
      console.error('Failed to set gain:', err);
    }
  };

  // Handle flip/rotate changes
  const handleFlipXToggle = async () => {
    const newFlipX = !flipX;
    setFlipX(newFlipX);
    try {
      await apiLightsheetControllerSetStreamTransform({ flipX: newFlipX, flipY, rotation });
    } catch (err) {
      console.error('Failed to set transform:', err);
    }
  };

  const handleFlipYToggle = async () => {
    const newFlipY = !flipY;
    setFlipY(newFlipY);
    try {
      await apiLightsheetControllerSetStreamTransform({ flipX, flipY: newFlipY, rotation });
    } catch (err) {
      console.error('Failed to set transform:', err);
    }
  };

  const handleRotationChange = async (event, newRotation) => {
    if (newRotation !== null) {
      setRotation(newRotation);
      try {
        await apiLightsheetControllerSetStreamTransform({ flipX, flipY, rotation: newRotation });
      } catch (err) {
        console.error('Failed to set rotation:', err);
      }
    }
  };

  // Handle live view camera exposure change
  const handleCameraExposureChange = async (event, value) => {
    const newExposure = value || cameraExposure;
    try {
      const result = await apiSetCameraExposureTime(newExposure);
      if (result.success) {
        console.log(`Live view camera exposure set to ${newExposure} ms`);
      } else {
        console.warn('Failed to set camera exposure:', result.message);
      }
    } catch (err) {
      console.error('Error setting camera exposure:', err);
    }
  };

  // Handle live view camera gain change
  const handleCameraGainChange = async (event, value) => {
    const newGain = value || cameraGain;
    try {
      const result = await apiSetCameraGain(newGain);
      if (result.success) {
        console.log(`Live view camera gain set to ${newGain}`);
      } else {
        console.warn('Failed to set camera gain:', result.message);
      }
    } catch (err) {
      console.error('Error setting camera gain:', err);
    }
  };

  // Fetch available scan modes and storage formats on mount
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const modes = await apiGetAvailableScanModes();
        if (Array.isArray(modes)) {
          dispatch(lightsheetSlice.setAvailableScanModes(modes));
        }
        
        const formats = await apiGetAvailableStorageFormats();
        if (Array.isArray(formats)) {
          dispatch(lightsheetSlice.setAvailableStorageFormats(formats));
        }
        
        // Fetch objective FOV for tiling hints
        const fovInfo = await apiGetObjectiveFOV();
        if (fovInfo.success) {
          dispatch(lightsheetSlice.setObjectiveFOV(fovInfo));
          // Auto-calculate tiling step size based on FOV with overlap
          const overlapFactor = 1 - fovInfo.suggestedOverlap;
          dispatch(lightsheetSlice.setTileStepSizeX(fovInfo.fovX * overlapFactor));
          dispatch(lightsheetSlice.setTileStepSizeY(fovInfo.fovY * overlapFactor));
        }
      } catch (error) {
        console.error("Error fetching options:", error);
      }
    };

    if (hostIP && hostPort) {
      fetchOptions();
    }
  }, [hostIP, hostPort, dispatch]);

  // Poll scan status when not using sockets
  useEffect(() => {
    if (!hostIP || !hostPort || socketConnected) return;

    const pollStatus = async () => {
      if (isRunning) {
        try {
          const status = await apiGetScanStatus();
          dispatch(lightsheetSlice.setScanStatus(status));
        } catch (error) {
          console.error("Error polling scan status:", error);
        }
      }
    };

    const interval = setInterval(pollStatus, 1000);
    return () => clearInterval(interval);
  }, [hostIP, hostPort, isRunning, socketConnected, dispatch]);

  // Poll current positions for 3D visualization
  // Fetch immediately on mount, then poll periodically
  useEffect(() => {
    if (!hostIP || !hostPort) return;

    const fetchPositions = async () => {
      try {
        const positionsData = await apiPositionerControllerGetPositions();
        
        const positions = {};
        if (positionsData) {
          ['X', 'Y', 'Z', 'A'].forEach(axis => {
            if (typeof positionsData[axis] !== 'undefined') {
              positions[axis.toLowerCase()] = positionsData[axis];
            }
          });
          
          if (Object.keys(positions).length > 0) {
            dispatch(lightsheetSlice.setAllStagePositions(positions));
          }
        }
      } catch (error) {
        console.error("Error fetching stage positions:", error);
      }
    };

    // Fetch immediately on mount for 3D viewer initial positioning
    fetchPositions();
    
    // Then poll every 20 seconds
    const interval = setInterval(fetchPositions, 20000);
    return () => clearInterval(interval);
  }, [hostIP, hostPort, dispatch]);

  // Sync global positionSlice with lightsheet-specific stagePositions
  // This ensures 3D model updates immediately when position changes via ANY route (buttons, websockets, etc.)
  const globalPositionState = useSelector(positionSlice.getPositionState);
  useEffect(() => {
    const positions = {};
    ['x', 'y', 'z', 'a'].forEach(axis => {
      if (typeof globalPositionState[axis] !== 'undefined') {
        positions[axis] = globalPositionState[axis];
      }
    });
    
    if (Object.keys(positions).length > 0) {
      dispatch(lightsheetSlice.setAllStagePositions(positions));
    }
  }, [globalPositionState.x, globalPositionState.y, globalPositionState.z, globalPositionState.a, dispatch]);

  // Start scanning based on selected mode
  const startScanning = useCallback(async () => {
    try {
      let result;
      
      if (scanMode === "step_acquire") {
        result = await apiStartStepAcquireScan({
          minPos: parseFloat(minPos),
          maxPos: parseFloat(maxPos),
          stepSize: parseFloat(stepSize),
          axis,
          illuSource: illuSource.toString(),
          illuValue: parseFloat(illuValue),
          storageFormat,
          experimentName,
          enableTiling,
          tilesXPositive: parseInt(tilesXPositive),
          tilesXNegative: parseInt(tilesXNegative),
          tilesYPositive: parseInt(tilesYPositive),
          tilesYNegative: parseInt(tilesYNegative),
          tileStepSizeX: parseFloat(tileStepSizeX),
          tileStepSizeY: parseFloat(tileStepSizeY),
          tileOverlap: parseFloat(tileOverlap),
          timepoints: parseInt(timepoints),
          timeLapsePeriod: parseFloat(timeLapsePeriod)
        });
      } else {
        result = await apiStartContinuousScanWithZarr({
          minPos: parseFloat(minPos),
          maxPos: parseFloat(maxPos),
          speed: parseFloat(speed),
          axis,
          illuSource: illuSource.toString(),
          illuValue: parseFloat(illuValue),
          storageFormat,
          experimentName,
        });
      }

      console.log("Scan started:", result);
      if (result.success) {
        dispatch(lightsheetSlice.setIsRunning(true));
      }
    } catch (error) {
      console.error("Error starting scan:", error);
    }
  }, [scanMode, minPos, maxPos, stepSize, speed, axis, illuSource, illuValue, storageFormat, experimentName, 
      enableTiling, tilesXPositive, tilesXNegative, tilesYPositive, tilesYNegative, 
      tileStepSizeX, tileStepSizeY, tileOverlap, timepoints, timeLapsePeriod, dispatch]);

  // Fetch and show latest Zarr for visualization
  const openZarrViewer = useCallback(async () => {
    try {
      const zarrInfo = await apiGetLatestZarrPath();
      if (zarrInfo.exists && zarrInfo.zarrPath) {
        dispatch(lightsheetSlice.setLatestZarrPath(zarrInfo));
        setShowZarrViewer(true);
      } else {
        alert("No Zarr data available yet. Run a scan first.");
      }
    } catch (error) {
      console.error("Error getting Zarr path:", error);
    }
  }, [dispatch]);

  const handleTabChange = (event, newValue) => {
    dispatch(lightsheetSlice.setTabIndex(newValue));
  };

  // Format display labels for scan modes
  const getScanModeLabel = (mode) => {
    const labels = {
      continuous: "Continuous (Fast)",
      step_acquire: "Step-Acquire (High Quality)",
    };
    return labels[mode] || mode;
  };

  // Format display labels for storage formats
  const getStorageFormatLabel = (format) => {
    const labels = {
      tiff: "TIFF Stack",
      ome_zarr: "OME-Zarr",
      both: "Both (TIFF + Zarr)",
    };
    return labels[format] || format;
  };

  return (
    <Paper>
      <Tabs
        value={tabIndex}
        onChange={handleTabChange}
        aria-label="Lightsheet Controller Tabs"
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab label="Scanning Parameters" />
        <Tab label="Observation Camera" />
        <Tab label="Galvo Scanner" />
        <Tab label="View Latest Stack" />
        <Tab label="3D Zarr Viewer" />
        <Tab label="VTK Viewer" />
      </Tabs>

      <TabPanel value={tabIndex} index={0}>
        <Grid container spacing={3}>
          {/* Connection Status */}
          <Grid item xs={12}>
            <Box display="flex" gap={1} alignItems="center" mb={2}>
              <Chip
                label={socketConnected ? "Socket Connected" : "Polling Mode"}
                color={socketConnected ? "success" : "warning"}
                size="small"
              />
              {scanStatus.scanMode && (
                <Chip
                  label={`Mode: ${getScanModeLabel(scanStatus.scanMode)}`}
                  color="info"
                  size="small"
                />
              )}
            </Box>
          </Grid>

          {/* 2D Live View */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Live View (2D)
            </Typography>
            <LiveViewControlWrapper />
          </Grid>

          {/* 3D Visualization */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              3D Assembly View
            </Typography>
            <Lightsheet3DViewer
              positions={lightsheetState.stagePositions}
              axisConfig={lightsheetState.axisConfig}
              cameraState={lightsheetState.cameraState}
              onCameraChange={(newCameraState) => {
                dispatch(lightsheetSlice.setCameraState(newCameraState));
              }}
              width={600}
              height={400}
            />
          </Grid>

          {/* Axis Configuration Menu */}
          <Grid item xs={12}>
            <AxisConfigurationMenu />
          </Grid>

          {/* Position Controls */}
          <Grid item xs={12}>
            <LightsheetPositionControls />
          </Grid>

          {/* Scan Mode Selection */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Scan Configuration
            </Typography>
          </Grid>

          <Grid item xs={12} md={4}>
            <FormControl fullWidth variant="outlined">
              <InputLabel>Scan Mode</InputLabel>
              <Select
                value={scanMode}
                onChange={(e) => dispatch(lightsheetSlice.setScanMode(e.target.value))}
                label="Scan Mode"
              >
                {availableScanModes.map((mode) => (
                  <MenuItem key={mode} value={mode}>
                    {getScanModeLabel(mode)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <FormControl fullWidth variant="outlined">
              <InputLabel>Storage Format</InputLabel>
              <Select
                value={storageFormat}
                onChange={(e) => dispatch(lightsheetSlice.setStorageFormat(e.target.value))}
                label="Storage Format"
              >
                {availableStorageFormats.map((format) => (
                  <MenuItem key={format} value={format}>
                    {getStorageFormatLabel(format)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              label="Experiment Name"
              value={experimentName}
              onChange={(e) => dispatch(lightsheetSlice.setExperimentName(e.target.value))}
              fullWidth
              variant="outlined"
            />
          </Grid>

          {/* Scanning Parameters */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Scanning Parameters
            </Typography>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <TextField
              label="Min Position (µm)"
              value={minPos}
              onChange={(e) => dispatch(lightsheetSlice.setMinPos(e.target.value))}
              fullWidth
              type="number"
              variant="outlined"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              label="Max Position (µm)"
              value={maxPos}
              onChange={(e) => dispatch(lightsheetSlice.setMaxPos(e.target.value))}
              fullWidth
              type="number"
              variant="outlined"
            />
          </Grid>
          
          {/* Show step size for step-acquire mode, speed for continuous */}
          {scanMode === "step_acquire" ? (
            <Grid item xs={12} md={4}>
              <TextField
                label="Step Size (µm)"
                value={stepSize}
                onChange={(e) => dispatch(lightsheetSlice.setStepSize(e.target.value))}
                fullWidth
                type="number"
                variant="outlined"
                inputProps={{ min: 0.1, step: 0.1 }}
              />
            </Grid>
          ) : (
            <Grid item xs={12} md={4}>
              <TextField
                label="Speed"
                value={speed}
                onChange={(e) => dispatch(lightsheetSlice.setSpeed(e.target.value))}
                fullWidth
                type="number"
                variant="outlined"
              />
            </Grid>
          )}

          <Grid item xs={12} md={4}>
            <FormControl fullWidth variant="outlined">
              <InputLabel>Axis</InputLabel>
              <Select
                value={axis}
                onChange={(e) => dispatch(lightsheetSlice.setAxis(e.target.value))}
                label="Axis"
              >
                <MenuItem value="A">A Axis</MenuItem>
                <MenuItem value="X">X Axis</MenuItem>
                <MenuItem value="Y">Y Axis</MenuItem>
                <MenuItem value="Z">Z Axis</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              label="Illumination Source"
              value={illuSource}
              onChange={(e) => dispatch(lightsheetSlice.setIlluSource(e.target.value))}
              fullWidth
              type="number"
              variant="outlined"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              label="Illumination Value"
              value={illuValue}
              onChange={(e) => dispatch(lightsheetSlice.setIlluValue(e.target.value))}
              fullWidth
              type="number"
              variant="outlined"
            />
          </Grid>

          {/* Camera Settings for Live View */}
          <Grid item xs={12}>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom>
              Camera Settings (Live View)
            </Typography>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="body2" gutterBottom>
              Exposure Time (ms)
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Slider
                value={cameraExposure}
                onChange={(e, value) => setCameraExposure(value)}
                onChangeCommitted={handleCameraExposureChange}
                min={1}
                max={1000}
                valueLabelDisplay="auto"
              />
              <TextField
                value={cameraExposure}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  if (!isNaN(val)) {
                    setCameraExposure(val);
                    handleCameraExposureChange(null, val);
                  }
                }}
                type="number"
                size="small"
                sx={{ width: 100 }}
              />
            </Box>
          </Grid>

          <Grid item xs={12} md={6}>
            <Typography variant="body2" gutterBottom>
              Gain
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Slider
                value={cameraGain}
                onChange={(e, value) => setCameraGain(value)}
                onChangeCommitted={handleCameraGainChange}
                min={0}
                max={100}
                valueLabelDisplay="auto"
              />
              <TextField
                value={cameraGain}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  if (!isNaN(val)) {
                    setCameraGain(val);
                    handleCameraGainChange(null, val);
                  }
                }}
                type="number"
                size="small"
                sx={{ width: 100 }}
              />
            </Box>
          </Grid>

          {/* Tiling Configuration (only for step-acquire mode) */}
          {scanMode === "step_acquire" && (
            <>
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Typography variant="h6" gutterBottom>
                    XY Tiling Configuration
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={enableTiling}
                        onChange={(e) => dispatch(lightsheetSlice.setEnableTiling(e.target.checked))}
                        color="primary"
                      />
                    }
                    label="Enable Tiling"
                  />
                </Box>
                <Typography variant="caption" color="text.secondary">
                  Suggested FOV: {objectiveFOV.fovX.toFixed(1)} x {objectiveFOV.fovY.toFixed(1)} µm
                </Typography>
              </Grid>

              {enableTiling && (
                <>
                  <Grid item xs={12} md={3}>
                    <TextField
                      label="Tiles +X"
                      value={tilesXPositive}
                      onChange={(e) => dispatch(lightsheetSlice.setTilesXPositive(parseInt(e.target.value) || 0))}
                      fullWidth
                      type="number"
                      variant="outlined"
                      inputProps={{ min: 0 }}
                      helperText="Positive X direction"
                    />
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <TextField
                      label="Tiles -X"
                      value={tilesXNegative}
                      onChange={(e) => dispatch(lightsheetSlice.setTilesXNegative(parseInt(e.target.value) || 0))}
                      fullWidth
                      type="number"
                      variant="outlined"
                      inputProps={{ min: 0 }}
                      helperText="Negative X direction"
                    />
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <TextField
                      label="Tiles +Y"
                      value={tilesYPositive}
                      onChange={(e) => dispatch(lightsheetSlice.setTilesYPositive(parseInt(e.target.value) || 0))}
                      fullWidth
                      type="number"
                      variant="outlined"
                      inputProps={{ min: 0 }}
                      helperText="Positive Y direction"
                    />
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <TextField
                      label="Tiles -Y"
                      value={tilesYNegative}
                      onChange={(e) => dispatch(lightsheetSlice.setTilesYNegative(parseInt(e.target.value) || 0))}
                      fullWidth
                      type="number"
                      variant="outlined"
                      inputProps={{ min: 0 }}
                      helperText="Negative Y direction"
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <TextField
                      label="Step Size X (µm)"
                      value={tileStepSizeX}
                      onChange={(e) => dispatch(lightsheetSlice.setTileStepSizeX(parseFloat(e.target.value) || 0))}
                      fullWidth
                      type="number"
                      variant="outlined"
                      inputProps={{ min: 0, step: 10 }}
                      helperText={`Suggested: ${(objectiveFOV.fovX * (1 - tileOverlap)).toFixed(1)} µm`}
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <TextField
                      label="Step Size Y (µm)"
                      value={tileStepSizeY}
                      onChange={(e) => dispatch(lightsheetSlice.setTileStepSizeY(parseFloat(e.target.value) || 0))}
                      fullWidth
                      type="number"
                      variant="outlined"
                      inputProps={{ min: 0, step: 10 }}
                      helperText={`Suggested: ${(objectiveFOV.fovY * (1 - tileOverlap)).toFixed(1)} µm`}
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <TextField
                      label="Overlap (%)"
                      value={(tileOverlap * 100).toFixed(0)}
                      onChange={(e) => dispatch(lightsheetSlice.setTileOverlap(parseFloat(e.target.value) / 100 || 0.1))}
                      fullWidth
                      type="number"
                      variant="outlined"
                      inputProps={{ min: 0, max: 50, step: 5 }}
                      helperText="Overlap between tiles"
                    />
                  </Grid>

                  {/* Tiling Grid Visualization */}
                  <Grid item xs={12}>
                    <Box sx={{ mt: 2, p: 2, border: '1px solid #ddd', borderRadius: 1, backgroundColor: 'Primary' }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Tiling Grid Preview
                      </Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                        <table style={{ borderCollapse: 'collapse' }}>
                          <tbody>
                            {Array.from({ length: tilesYPositive + tilesYNegative + 1 }, (_, row) => {
                              const yIndex = tilesYPositive - row;
                              return (
                                <tr key={row}>
                                  {Array.from({ length: tilesXNegative + tilesXPositive + 1 }, (_, col) => {
                                    const xIndex = col - tilesXNegative;
                                    const isCenter = xIndex === 0 && yIndex === 0;
                                    return (
                                      <td
                                        key={col}
                                        style={{
                                          width: '40px',
                                          height: '40px',
                                          border: '1px solid #666',
                                          backgroundColor: isCenter ? '#4CAF50' : '#fff',
                                          textAlign: 'center',
                                          verticalAlign: 'middle',
                                          fontSize: '10px',
                                          fontWeight: isCenter ? 'bold' : 'normal',
                                          color: isCenter ? '#fff' : '#333',
                                        }}
                                        title={`X: ${xIndex}, Y: ${yIndex}${isCenter ? ' (Center)' : ''}`}
                                      >
                                        {xIndex},{yIndex}
                                      </td>
                                    );
                                  })}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </Box>
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
                        Total tiles: {(tilesXNegative + tilesXPositive + 1) * (tilesYNegative + tilesYPositive + 1)} 
                        {' '}({tilesXNegative + tilesXPositive + 1} × {tilesYNegative + tilesYPositive + 1})
                        {' • '}Green cell (0,0) is the center/origin
                      </Typography>
                    </Box>
                  </Grid>
                </>
              )}
            </>
          )}

          {/* Timelapse Configuration (only for step-acquire mode) */}
          {scanMode === "step_acquire" && (
            <>
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Timelapse Configuration
                </Typography>
              </Grid>

              <Grid item xs={12} md={6}>
                <TextField
                  label="Timepoints"
                  value={timepoints}
                  onChange={(e) => dispatch(lightsheetSlice.setTimepoints(parseInt(e.target.value) || 1))}
                  fullWidth
                  type="number"
                  variant="outlined"
                  inputProps={{ min: 1, step: 1 }}
                  helperText="Number of timepoints to acquire"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Period (seconds)"
                  value={timeLapsePeriod}
                  onChange={(e) => dispatch(lightsheetSlice.setTimeLapsePeriod(parseFloat(e.target.value) || 60))}
                  fullWidth
                  type="number"
                  variant="outlined"
                  inputProps={{ min: 1, step: 10 }}
                  helperText="Time between timepoints"
                  disabled={timepoints <= 1}
                />
              </Grid>
            </>
          )}

          {/* Progress Display */}
          {(isRunning || scanStatus.progress > 0) && (
            <Grid item xs={12}>
              <Box sx={{ width: '100%' }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {isRunning 
                    ? `Scanning... Frame ${scanStatus.currentFrame}/${scanStatus.totalPositions || '?'} at position ${scanStatus.currentPosition?.toFixed(1) || 0} µm`
                    : "Scan complete"}
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={scanStatus.progress || 0}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>
            </Grid>
          )}

          {/* Error Display */}
          {scanStatus.errorMessage && (
            <Grid item xs={12}>
              <Alert severity="error">{scanStatus.errorMessage}</Alert>
            </Grid>
          )}

          {/* Control Buttons */}
          <Grid item xs={12}>
            <Box display="flex" alignItems="center" gap={2}>
              <Button
                variant="contained"
                color="primary"
                onClick={startScanning}
                disabled={isRunning}
                size="large"
                startIcon={<PlayArrowIcon />}
              >
                {scanMode === "step_acquire" ? "Start Step-Acquire" : "Start Continuous Scan"}
              </Button>
              
              {isRunning && (
                <Button
                  variant="contained"
                  color="error"
                  size="large"
                  startIcon={<StopIcon />}
                  onClick={() => {
                    // TODO: Implement stop functionality via API
                    fetch(`${hostIP}:${hostPort}/imswitch/api/LightsheetController/stopLightsheet`);
                  }}
                >
                  Stop
                </Button>
              )}

              <Button
                variant="outlined"
                color="secondary"
                onClick={openZarrViewer}
                disabled={isRunning}
                startIcon={<ViewInArIcon />}
              >
                View Latest Zarr
              </Button>

              {isRunning ? (
                <CheckCircleIcon style={{ color: green[500] }} />
              ) : (
                <CancelIcon style={{ color: scanStatus.progress >= 100 ? green[500] : red[500] }} />
              )}
              <Typography variant="body2">
                {isRunning 
                  ? "Scanning in progress..." 
                  : scanStatus.progress >= 100 
                    ? "Scan complete" 
                    : "Ready to scan"}
              </Typography>
            </Box>
          </Grid>

          {/* Result Paths */}
          {(scanStatus.zarrPath || scanStatus.tiffPath) && (
            <Grid item xs={12}>
              <Box sx={{ mt: 2, p: 2, bgcolor: 'secondary', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Output Files:
                </Typography>
                {scanStatus.zarrPath && (
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    Zarr: {scanStatus.zarrPath}
                  </Typography>
                )}
                {scanStatus.tiffPath && (
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    TIFF: {scanStatus.tiffPath}
                  </Typography>
                )}
              </Box>
            </Grid>
          )}
        </Grid>
      </TabPanel>

      {/* Observation Camera Tab */}
      <TabPanel value={tabIndex} index={1}>
        <Grid container spacing={3}>
          {/* Stream Error Display */}
          {streamError && (
            <Grid item xs={12}>
              <Alert severity="error" onClose={() => setStreamError('')}>
                {streamError}
              </Alert>
            </Grid>
          )}

          {/* Observation Camera Stream */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Observation Camera Stream
              </Typography>
              
              {/* Stream Controls */}
              <Box sx={{ mb: 2, display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
                <Button 
                  variant="contained" 
                  color={observationStreamActive ? "error" : "primary"}
                  onClick={handleObservationStreamToggle}
                  startIcon={observationStreamActive ? <VideocamOffIcon /> : <VideocamIcon />}
                >
                  {observationStreamActive ? 'Stop Stream' : 'Start Stream'}
                </Button>

                <Divider orientation="vertical" flexItem />

                {/* Flip Controls */}
                <Tooltip title="Flip Horizontal">
                  <IconButton 
                    onClick={handleFlipXToggle}
                    color={flipX ? "primary" : "default"}
                  >
                    <FlipIcon />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title="Flip Vertical">
                  <IconButton 
                    onClick={handleFlipYToggle}
                    color={flipY ? "primary" : "default"}
                    sx={{ transform: 'rotate(90deg)' }}
                  >
                    <FlipIcon />
                  </IconButton>
                </Tooltip>

                <Divider orientation="vertical" flexItem />

                {/* Rotation Controls */}
                <ToggleButtonGroup
                  value={rotation}
                  exclusive
                  onChange={handleRotationChange}
                  size="small"
                >
                  <ToggleButton value={0}>0°</ToggleButton>
                  <ToggleButton value={90}>90°</ToggleButton>
                  <ToggleButton value={180}>180°</ToggleButton>
                  <ToggleButton value={270}>270°</ToggleButton>
                </ToggleButtonGroup>
              </Box>

              {/* MJPEG Stream Display */}
              <Box 
                sx={{ 
                  backgroundColor: 'black', 
                  minHeight: 400,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative',
                  borderRadius: 1,
                  overflow: 'hidden'
                }}
              >
                {observationStreamActive ? (
                  <img
                    ref={observationImgRef}
                    src={observationStreamUrl}
                    alt="Observation Camera"
                    style={{ 
                      display: 'block',
                      margin: 'auto',
                      maxWidth: '100%', 
                      maxHeight: 500,
                      objectFit: 'contain',
                      WebkitUserSelect: 'none',
                      transform: `scaleX(${flipX ? -1 : 1}) scaleY(${flipY ? -1 : 1}) rotate(${rotation}deg)`
                    }}
                    onError={() => setStreamError('Failed to load stream. Check if observation camera is available.')}
                  />
                ) : (
                  <Typography color="white">
                    Stream not active. Click "Start Stream" to begin.
                  </Typography>
                )}
              </Box>
            </Paper>
          </Grid>

          {/* Camera Controls */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Camera Settings
              </Typography>

              {/* Exposure Time Slider */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Exposure Time (ms)
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Slider
                    value={observationExposure}
                    onChange={handleExposureChange}
                    onChangeCommitted={handleExposureChangeCommitted}
                    min={1}
                    max={1000}
                    valueLabelDisplay="auto"
                    disabled={!observationStreamActive}
                  />
                  <TextField
                    value={observationExposure}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val)) {
                        setObservationExposure(val);
                        handleExposureChangeCommitted(null, val);
                      }
                    }}
                    type="number"
                    size="small"
                    sx={{ width: 80 }}
                    disabled={!observationStreamActive}
                  />
                </Box>
              </Box>

              {/* Gain Slider */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Gain
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Slider
                    value={observationGain}
                    onChange={handleGainChange}
                    onChangeCommitted={handleGainChangeCommitted}
                    min={0}
                    max={100}
                    valueLabelDisplay="auto"
                    disabled={!observationStreamActive}
                  />
                  <TextField
                    value={observationGain}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val)) {
                        setObservationGain(val);
                        handleGainChangeCommitted(null, val);
                      }
                    }}
                    type="number"
                    size="small"
                    sx={{ width: 80 }}
                    disabled={!observationStreamActive}
                  />
                </Box>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Transform Status */}
              <Typography variant="subtitle2" gutterBottom>
                Current Transform
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Chip 
                  label={`Flip X: ${flipX ? 'On' : 'Off'}`} 
                  size="small" 
                  color={flipX ? 'primary' : 'default'}
                />
                <Chip 
                  label={`Flip Y: ${flipY ? 'On' : 'Off'}`} 
                  size="small" 
                  color={flipY ? 'primary' : 'default'}
                />
                <Chip 
                  label={`Rotation: ${rotation}°`} 
                  size="small" 
                  color={rotation !== 0 ? 'primary' : 'default'}
                />
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Galvo Scanner Tab - now index 2 */}
      <TabPanel value={tabIndex} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Galvo Scanner Configuration
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <LiveViewControlWrapper />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Channel"
              value={lightsheetState.galvoChannel || 2}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoChannel(parseInt(e.target.value))
                )
              }
              fullWidth
              type="number"
              variant="outlined"
              inputProps={{ min: 1, max: 2 }}
              helperText="Channel: 1 or 2"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Frequency"
              value={lightsheetState.galvoFrequency || 20}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoFrequency(parseFloat(e.target.value))
                )
              }
              fullWidth
              type="number"
              variant="outlined"
              inputProps={{ min: 0, step: 0.1 }}
              helperText="Frequency (Hz)"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Offset"
              value={lightsheetState.galvoOffset || 0}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoOffset(parseFloat(e.target.value))
                )
              }
              fullWidth
              type="number"
              variant="outlined"
              inputProps={{ step: 0.1 }}
              helperText="Offset value"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Amplitude"
              value={lightsheetState.galvoAmplitude || 2}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoAmplitude(parseFloat(e.target.value))
                )
              }
              fullWidth
              type="number"
              variant="outlined"
              inputProps={{ min: 0, step: 0.1 }}
              helperText="Amplitude value"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Clock Divider"
              value={lightsheetState.galvoClkDiv || 0}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoClkDiv(parseInt(e.target.value))
                )
              }
              fullWidth
              type="number"
              variant="outlined"
              inputProps={{ min: 0 }}
              helperText="Clock divider"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Phase"
              value={lightsheetState.galvoPhase || 0}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoPhase(parseInt(e.target.value))
                )
              }
              fullWidth
              type="number"
              variant="outlined"
              helperText="Phase value"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              label="Invert"
              value={lightsheetState.galvoInvert || 1}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoInvert(parseInt(e.target.value))
                )
              }
              fullWidth
              type="number"
              variant="outlined"
              inputProps={{ min: 0, max: 1 }}
              helperText="Invert: 0 or 1"
            />
          </Grid>
          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => {
                const channel = lightsheetState.galvoChannel || 1;
                const frequency = lightsheetState.galvoFrequency || 10;
                const offset = lightsheetState.galvoOffset || 0;
                const amplitude = lightsheetState.galvoAmplitude || 1;
                const clk_div = lightsheetState.galvoClkDiv || 0;
                const phase = lightsheetState.galvoPhase || 0;
                const invert = lightsheetState.galvoInvert || 1;

                const url = `${hostIP}:${hostPort}/imswitch/api/LightsheetController/setGalvo?channel=${channel}&frequency=${frequency}&offset=${offset}&amplitude=${amplitude}&clk_div=${clk_div}&phase=${phase}&invert=${invert}`;

                fetch(url, { method: "GET" })
                  .then((response) => response.json())
                  .then((data) => {
                    console.log("Galvo parameters set:", data);
                  })
                  .catch((error) =>
                    console.error("Error setting galvo parameters:", error)
                  );
              }}
              size="large"
            >
              Apply Galvo Settings
            </Button>
          </Grid>
        </Grid>
      </TabPanel>

      {/* View Latest Stack Tab - now index 3 */}
      <TabPanel value={tabIndex} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Download & External Viewers
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<ViewInArIcon />}
              onClick={() =>
                window.open(
                  `https://kitware.github.io/itk-vtk-viewer/app/?rotate=false&fileToLoad=${hostIP}:${hostPort}/imswitch/api/LightsheetController/getLatestLightsheetStackAsTif`,
                  "_blank"
                )
              }
            >
              Open in ITK-VTK Viewer (requires internet)
            </Button>
          </Grid>
          <Grid item xs={12}>
            <Button
              variant="contained"
              color="secondary"
              startIcon={<DownloadIcon />}
              onClick={() =>
                window.open(
                  `${hostIP}:${hostPort}/imswitch/api/LightsheetController/getLatestLightsheetStackAsTif`,
                  "_blank"
                )
              }
            >
              Download Latest TIFF Stack
            </Button>
          </Grid>
          {latestZarrPath && (
            <Grid item xs={12}>
              <Alert severity="info">
                Latest Zarr path: {latestZarrPath}
              </Alert>
            </Grid>
          )}
        </Grid>
      </TabPanel>

      {/* 3D Zarr Viewer Tab - now index 4 */}
      <TabPanel value={tabIndex} index={4}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              3D OME-Zarr Viewer
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Visualize lightsheet Z-stacks directly in the browser using the offline-capable Zarr viewer.
            </Typography>
          </Grid>
          
          <Grid item xs={12}>
            <Box display="flex" gap={2} mb={2}>
              <Button
                variant="contained"
                color="primary"
                onClick={openZarrViewer}
                startIcon={<ViewInArIcon />}
              >
                Load Latest Zarr Stack
              </Button>
              <Button
                variant="outlined"
                onClick={() => setShowZarrViewer(false)}
                disabled={!showZarrViewer}
              >
                Close Viewer
              </Button>
            </Box>
          </Grid>

          {showZarrViewer && latestZarrPath ? (
            <Grid item xs={12}>
              <ErrorBoundary>
                <Box sx={{ height: '70vh', width: '100%' }}>
                  <VizarrViewer
                    zarrUrl={latestZarrPath}
                    onClose={() => setShowZarrViewer(false)}
                    embedded={true}
                    height="100%"
                    width="100%"
                  />
                </Box>
              </ErrorBoundary>
            </Grid>
          ) : (
            <Grid item xs={12}>
              <Box sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.100', borderRadius: 2 }}>
                <Typography variant="body1" color="text.secondary">
                  {latestZarrPath 
                    ? "Click 'Load Latest Zarr Stack' to view the data"
                    : "No Zarr data available. Run a scan with OME-Zarr storage format first."}
                </Typography>
              </Box>
            </Grid>
          )}
        </Grid>
      </TabPanel>

      {/* VTK Viewer Tab - now index 5 */}
      <TabPanel value={tabIndex} index={5}>
        <ErrorBoundary>
          <Typography variant="h6" gutterBottom>
            VTK Volume Viewer (TIFF)
          </Typography>
          <VtkViewer
            tifUrl={`${hostIP}:${hostPort}/imswitch/api/LightsheetController/getLatestLightsheetStackAsTif`}
          />
        </ErrorBoundary>
      </TabPanel>
    </Paper>
  );
};

export default LightsheetController;
