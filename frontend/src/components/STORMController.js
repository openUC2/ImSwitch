// src/components/STORMController.js
import React, { useEffect, useRef, useState, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../backendapi/createAxiosInstance';
import {
  Paper,
  Tabs,
  Tab,
  Box,
  Typography,
  TextField,
  Button,
  Grid,
  Slider,
  Card,
  CardContent,
  Checkbox,
  FormControlLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { green, red } from "@mui/material/colors";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import RefreshIcon from "@mui/icons-material/Refresh";
import { useWebSocket } from "../context/WebSocketContext";
import * as stormSlice from "../state/slices/STORMSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import apiSTORMControllerStart from "../backendapi/apiSTORMControllerStart.js";
import apiSTORMControllerStop from "../backendapi/apiSTORMControllerStop.js";
import apiSTORMControllerGetStatus from "../backendapi/apiSTORMControllerGetStatus.js";
import apiSTORMControllerSetParameters from "../backendapi/apiSTORMControllerSetParameters.js";
import apiSTORMControllerTriggerReconstruction from "../backendapi/apiSTORMControllerTriggerReconstruction.js";
import apiSTORMControllerSetProcessingParameters from "../backendapi/apiSTORMControllerSetProcessingParameters.js";
import apiSTORMControllerStartReconstructionLocal from "../backendapi/apiSTORMControllerStartReconstructionLocal.js";
import apiSTORMControllerStopReconstructionLocal from "../backendapi/apiSTORMControllerStopReconstructionLocal.js";
import apiSTORMControllerGetProcessingParameters from "../backendapi/apiSTORMControllerGetProcessingParameters.js";
import apiSTORMControllerGetReconstructionStatus from "../backendapi/apiSTORMControllerGetReconstructionStatus.js";
import apiSTORMControllerGetLastReconstructedImagePath from "../backendapi/apiSTORMControllerGetLastReconstructedImagePath.js";

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`storm-tabpanel-${index}`}
      aria-labelledby={`storm-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );
};

// Utility function to perform min/max intensity stretch on an image
function stretchImageIntensity(imageData) {
  // Find min and max pixel values
  let min = 255,
    max = 0;
  for (let i = 0; i < imageData.data.length; i += 4) {
    const v = imageData.data[i]; // Assume grayscale PNG
    if (v < min) min = v;
    if (v > max) max = v;
  }
  // Stretch intensities
  const range = max - min || 1;
  for (let i = 0; i < imageData.data.length; i += 4) {
    const v = imageData.data[i];
    const stretched = Math.round(((v - min) / range) * 255);
    imageData.data[i] = stretched;
    imageData.data[i + 1] = stretched;
    imageData.data[i + 2] = stretched;
    // Alpha channel remains unchanged
  }
  return imageData;
}

const STORMController = ({ hostIP, hostPort, WindowTitle }) => {
  // Redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const stormState = useSelector(stormSlice.getSTORMState);
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const parameterRangeState = useSelector(parameterRangeSlice.getParameterRangeState);
  const liveStreamState = useSelector((state) => state.liveStreamState);

  // Local state for crop region selection and detector parameters
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [laserActiveStates, setLaserActiveStates] = useState({});
  const [detectorGain, setDetectorGain] = useState(0);
  const [cropImage, setCropImage] = useState(null); // Image for cropping that loads on mount
  // Add state for loaded image and its pixel dimensions
  const [loadedImage, setLoadedImage] = useState(null); // holds stretched PNG data URL
  const [imageDims, setImageDims] = useState({ width: 0, height: 0 });
  const [isCropping, setIsCropping] = useState(false);
  const [cropStart, setCropStart] = useState({ x: 0, y: 0 });
  const [cropPreview, setCropPreview] = useState(null); // for preview rectangle
  const canvasRef = useRef(null);

  // Use Redux state instead of local useState
  const tabIndex = stormState.tabIndex;
  const experimentName = stormState.experimentName;
  const exposureTime = stormState.exposureTime;
  const cropRegion = stormState.cropRegion;
  const laserIntensities = stormState.laserIntensities;
  const isRunning = stormState.isRunning;
  const currentFrameNumber = stormState.currentFrameNumber;
  const reconstructedImage = stormState.reconstructedImage;
  const stormParameters = stormState.stormParameters;
  const acquisitionParameters = stormState.acquisitionParameters;
  const isReconstructing = stormState.isReconstructing;

  // Use global live stream image
  const liveStreamImage = liveStreamState.liveViewImage;

  const socket = useWebSocket();

  // Load initial image for cropping on component mount
  useEffect(() => {
    const loadInitialCropImage = async () => {
      if (liveStreamImage) {
        setCropImage(liveStreamImage);
      } else if (connectionSettingsState.ip && connectionSettingsState.apiPort) {
        // Try to get a snapshot from the backend for initial cropping
        try {
          const api = createAxiosInstance();
          const response = await api.get(
            `/DetectorController/getLatestDetectorFrame`,
            { responseType: 'blob' }
          );
          const dataUrl = URL.createObjectURL(response.data);
          setCropImage(dataUrl);
        } catch (error) {
          console.warn("Could not load initial crop image:", error);
          // Use a placeholder or the current live stream image
          if (liveStreamImage) setCropImage(liveStreamImage);
        }
      }
    };

    loadInitialCropImage();
  }, [liveStreamImage, connectionSettingsState.ip, connectionSettingsState.apiPort]);

  // Fetch initial detector gain on component mount
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

  // Fetch initial status on component mount
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await apiSTORMControllerGetStatus();
        dispatch(stormSlice.setIsRunning(data.isRunning || false));
        dispatch(stormSlice.setCurrentFrameNumber(data.currentFrame || 0));
        
        // Also get reconstruction status
        await getReconstructionStatus();
      } catch (error) {
        console.error('Error fetching STORM status:', error);
      }
    };

    fetchStatus();
  }, [dispatch]);

  // Load processing parameters on component mount
  useEffect(() => {
    const loadInitialParameters = async () => {
      try {
        await loadProcessingParameters();
      } catch (error) {
        console.log("Could not load initial processing parameters, using defaults");
      }
    };

    loadInitialParameters();
  }, []);

  // Periodic status polling when reconstruction is running
  useEffect(() => {
    let statusInterval;
    
    if (isReconstructing) {
      statusInterval = setInterval(async () => {
        await getReconstructionStatus();
        await loadLastReconstructedImage();
      }, 2000); // Poll every 2 seconds
    }
    
    return () => {
      if (statusInterval) {
        clearInterval(statusInterval);
      }
    };
  }, [isReconstructing]);

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

  // WebSocket integration for STORM-specific experiment status signals
  useEffect(() => {
    if (!socket) return;

    const handleSignal = (data) => {
      try {
        const jdata = JSON.parse(data);
        
        // Handle STORM experiment status changes
        if (jdata.name === "sigSTORMExperimentStarted") {
          dispatch(stormSlice.setIsRunning(true));
        }
        
        if (jdata.name === "sigSTORMExperimentStopped") {
          dispatch(stormSlice.setIsRunning(false));
        }
      } catch (error) {
        console.error("Error parsing STORM signal data:", error);
      }
    };

    socket.on("signal", handleSignal);
    
    return () => {
      socket.off("signal", handleSignal);
    };
  }, [socket, dispatch]);

  // Start STORM experiment
  const startExperiment = async () => {
    try {
      const params = {
        session_id: experimentName,
        crop_x: cropRegion.x,
        crop_y: cropRegion.y,
        crop_width: cropRegion.width,
        crop_height: cropRegion.height,
        exposure_time: exposureTime,
        saveDirectory: "STORM",
        save_format: "tiff",
        max_frames: -1,
        process_arkitekt: false
      };
      
      await apiSTORMControllerStart(params);
      dispatch(stormSlice.setIsRunning(true));
    } catch (error) {
      console.error("Error starting STORM experiment:", error);
    }
  };

  // Stop STORM experiment
  const stopExperiment = async () => {
    try {
      await apiSTORMControllerStop();
      dispatch(stormSlice.setIsRunning(false));
    } catch (error) {
      console.error("Error stopping STORM experiment:", error);
    }
  };

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    dispatch(stormSlice.setTabIndex(newValue));
  };

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

  // Helper: Min/Max stretch for PNG (Uint8)
  const minMaxStretch = async (imgUrl) => {
    return new Promise((resolve) => {
      const img = new window.Image();
      img.crossOrigin = "Anonymous";
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        let min = 255, max = 0;
        for (let i = 0; i < data.length; i += 4) {
          const v = data[i];
          if (v < min) min = v;
          if (v > max) max = v;
        }
        const range = max - min || 1;
        for (let i = 0; i < data.length; i += 4) {
          const stretched = Math.round(((data[i] - min) / range) * 255);
          data[i] = data[i + 1] = data[i + 2] = stretched;
        }
        ctx.putImageData(imageData, 0, 0);
        resolve(canvas.toDataURL("image/png"));
      };
      img.src = imgUrl;
    });
  };

  // Load image from backend and stretch
  const handleLoadImage = async () => {
    try {
      const api = createAxiosInstance();
      const response = await api.get(
        `/RecordingController/snapNumpyToFastAPI?resizeFactor=1`,
        { responseType: 'blob' }
      );
      const url = URL.createObjectURL(response.data);
      // Get true pixel dimensions
      const img = new window.Image();
      img.onload = async () => {
        setImageDims({ width: img.naturalWidth, height: img.naturalHeight });
        const stretchedUrl = await minMaxStretch(url);
        setLoadedImage(stretchedUrl);
        // Reset crop region to full image
        dispatch(stormSlice.setCropRegion({
          x: 0,
          y: 0,
          width: img.naturalWidth,
          height: img.naturalHeight,
        }));
      };
      img.src = url;
    } catch (e) {
      alert("Could not load image: " + e.message);
    }
  };

  // Draw canvas with crop rectangle
  const drawCropCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const imgSrc = loadedImage;
    if (!canvas || !imgSrc) return;
    const ctx = canvas.getContext("2d");
    const img = new window.Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      // Draw crop rectangle
      const scaleX = canvas.width / imageDims.width;
      const scaleY = canvas.height / imageDims.height;
      const crop = cropPreview || cropRegion;
      ctx.save();
      ctx.strokeStyle = "#ff0000";
      ctx.lineWidth = 2;
      ctx.strokeRect(
        crop.x * scaleX,
        crop.y * scaleY,
        crop.width * scaleX,
        crop.height * scaleY
      );
      ctx.restore();
    };
    img.src = imgSrc;
  }, [loadedImage, imageDims, cropRegion, cropPreview]);

  useEffect(() => {
    drawCropCanvas();
  }, [drawCropCanvas]);

  // Mouse events for cropping
  const handleCropMouseDown = (e) => {
    if (!loadedImage) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * imageDims.width) / canvas.width;
    const y = ((e.clientY - rect.top) * imageDims.height) / canvas.height;
    setIsCropping(true);
    setCropStart({ x, y });
    setCropPreview({ x, y, width: 1, height: 1 });
  };
  const handleCropMouseMove = (e) => {
    if (!isCropping) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * imageDims.width) / canvas.width;
    const y = ((e.clientY - rect.top) * imageDims.height) / canvas.height;
    const cropX = Math.max(0, Math.min(cropStart.x, x));
    const cropY = Math.max(0, Math.min(cropStart.y, y));
    const cropW = Math.abs(x - cropStart.x);
    const cropH = Math.abs(y - cropStart.y);
    setCropPreview({
      x: Math.round(cropX),
      y: Math.round(cropY),
      width: Math.round(cropW),
      height: Math.round(cropH),
    });
  };
  const handleCropMouseUp = () => {
    if (isCropping && cropPreview) {
      dispatch(stormSlice.setCropRegion(cropPreview));
    }
    setIsCropping(false);
    setCropPreview(null);
  };

  // Local state for acquisition_active from backend
  const [acquisitionActive, setAcquisitionActive] = useState(false);

  // Handle STORM reconstruction parameters
  const setStormParameter = async (paramName, value) => {
    const params = { [paramName]: value };
    
    try {
      await apiSTORMControllerSetParameters(params);
      dispatch(stormSlice.setStormParameters(params));
    } catch (error) {
      console.error("Error setting STORM parameter:", error);
    }
  };

  // Handle bandpass filter parameters
  const setBandpassFilterParameter = (paramName, value) => {
    const updatedFilter = { ...stormParameters.bandpass_filter, [paramName]: value };
    dispatch(stormSlice.setBandpassFilter(updatedFilter));
  };

  // Handle blob detector parameters
  const setBlobDetectorParameter = (paramName, value) => {
    const updatedDetector = { ...stormParameters.blob_detector, [paramName]: value };
    dispatch(stormSlice.setBlobDetector(updatedDetector));
  };

  // Handle acquisition parameters
  const setAcquisitionParameter = (paramName, value) => {
    const updatedAcquisition = { ...acquisitionParameters, [paramName]: value };
    dispatch(stormSlice.setAcquisitionParameters(updatedAcquisition));
  };

  // Update processing parameters on the backend
  const updateProcessingParameters = async () => {
    try {
      await apiSTORMControllerSetProcessingParameters(stormParameters);
      console.log("Processing parameters updated successfully");
    } catch (error) {
      console.error("Error updating processing parameters:", error);
    }
  };

  // Start local reconstruction
  const startReconstructionLocal = async () => {
    try {
      const reconstructionRequest = {
        session_id: acquisitionParameters.session_id,
        acquisition_parameters: {
          ...acquisitionParameters,
          crop_x: cropRegion.x,
          crop_y: cropRegion.y,
          crop_width: cropRegion.width,
          crop_height: cropRegion.height,
          exposure_time: exposureTime,
          crop_enabled: acquisitionParameters.crop_enabled || (cropRegion.x !== 0 || cropRegion.y !== 0 || cropRegion.width !== imageDims.width || cropRegion.height !== imageDims.height),
        },
        processing_parameters: stormParameters,
        save_enabled: acquisitionParameters.save_enabled
      };
      
      await apiSTORMControllerStartReconstructionLocal(reconstructionRequest);
      dispatch(stormSlice.setIsReconstructing(true));
      console.log("Local reconstruction started successfully");
    } catch (error) {
      console.error("Error starting local reconstruction:", error);
    }
  };

  // Stop local reconstruction
  const stopReconstructionLocal = async () => {
    try {
      await apiSTORMControllerStopReconstructionLocal();
      dispatch(stormSlice.setIsReconstructing(false));
      console.log("Local reconstruction stopped successfully");
    } catch (error) {
      console.error("Error stopping local reconstruction:", error);
    }
  };

  // Load current processing parameters from backend
  const loadProcessingParameters = async () => {
    try {
      const params = await apiSTORMControllerGetProcessingParameters();
      dispatch(stormSlice.setStormParameters(params));
      console.log("Processing parameters loaded successfully");
    } catch (error) {
      console.error("Error loading processing parameters:", error);
    }
  };

  // Get reconstruction status
  const getReconstructionStatus = async () => {
    try {
      const status = await apiSTORMControllerGetReconstructionStatus();
      // Update Redux and local state
      dispatch(stormSlice.setIsReconstructing(status.isReconstructing || false));
      setAcquisitionActive(!!status.acquisition_active); // Update local state
      return status;
    } catch (error) {
      console.error("Error getting reconstruction status:", error);
      setAcquisitionActive(false);
      return null;
    }
  };

  // Get last reconstructed image path and load it
  const loadLastReconstructedImage = async () => {
    try {
      const imagePath = await apiSTORMControllerGetLastReconstructedImagePath();
      if (imagePath && connectionSettingsState.ip && connectionSettingsState.apiPort) {
        // Construct the URL to load the image - keep full URL for image preview
        const imageUrl = `${connectionSettingsState.ip}:${connectionSettingsState.apiPort}/imswitch/api/FileManager/preview/${encodeURIComponent(imagePath)}`;
        dispatch(stormSlice.setReconstructedImage(imageUrl));
      }
    } catch (error) {
      console.error("Error loading last reconstructed image:", error);
    }
  };

  // Trigger STORM reconstruction
  const triggerReconstruction = async () => {
    try {
      await apiSTORMControllerTriggerReconstruction();
      dispatch(stormSlice.setIsReconstructing(true));
    } catch (error) {
      console.error("Error triggering STORM reconstruction:", error);
    }
  };

  // Reset crop region to full image
  const resetCropRegion = () => {
    if (loadedImage && imageDims.width && imageDims.height) {
      dispatch(stormSlice.setCropRegion({
        x: 0,
        y: 0,
        width: imageDims.width,
        height: imageDims.height,
      }));
    }
  };

  // Local state for socket image
  const [reconImage, setReconImage] = useState(null);

  // Listen for socket image updates
  useEffect(() => {
    if (!socket) return;
    const handleImageUpdate = (data) => {
      try {
        const jdata = typeof data === 'string' ? JSON.parse(data) : data;
        if (jdata.name === 'sigExperimentImageUpdate' && jdata.detectorname === 'STORM' && jdata.image) {
          // Decode base64 JPEG and apply min-max stretch
          const imgSrc = `data:image/jpeg;base64,${jdata.image}`;
          minMaxStretch(imgSrc).then(stretchedUrl => setReconImage(stretchedUrl));
        }
      } catch (err) {
        console.error('Error parsing socket image update:', err);
      }
    };
    socket.on('signal', handleImageUpdate);
    return () => socket.off('signal', handleImageUpdate);
  }, [socket]);

  return (
    <Paper>
      <Grid container spacing={2} sx={{ height: '100%' }}>
        {/* Left side - Always visible livestream */}
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Live Stream
            </Typography>
            
            {/* Live stream image */}
            <Box sx={{ flex: 1, minHeight: 400, border: '1px solid #ccc', mb: 2 }}>
              <LiveViewControlWrapper />
            </Box>

            {/* Status and frame counter */}
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Box display="flex" alignItems="center">
                  <Typography variant="h6">Status: </Typography>
                  {/* Show running status from Redux */}
                  {isRunning ? (
                    <>
                      <CheckCircleIcon style={{ color: green[500], marginLeft: "10px" }} />
                      <Typography sx={{ ml: 1 }} color="success.main">Running</Typography>
                    </>
                  ) : (
                    <>
                      <CancelIcon style={{ color: red[500], marginLeft: "10px" }} />
                      <Typography sx={{ ml: 1 }} color="error.main">Stopped</Typography>
                    </>
                  )}
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="h6">
                  Frame: {currentFrameNumber}
                </Typography>
              </Grid>
            </Grid>
          </Box>
        </Grid>

        {/* Right side - Control tabs */}
        <Grid item xs={12} md={6}>
          <Tabs
            value={tabIndex}
            onChange={handleTabChange}
            aria-label="STORM controller tabs"
          >
            <Tab label="Experiment & Controls" />
            <Tab label="Crop Region" />
            <Tab label="Reconstructed Image" />
            <Tab label="STORM Reconstruction" />
            <Tab label="Live Reconstruction" />
          </Tabs>

          {/* Experiment & Controls Tab (merged exposure, gain, lasers) */}
          <TabPanel value={tabIndex} index={0}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  label="Experiment Name"
                  value={experimentName}
                  onChange={(e) => dispatch(stormSlice.setExperimentName(e.target.value))}
                  fullWidth
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>
                  Exposure Time: {exposureTime} ms
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Slider
                    value={exposureTime}
                    min={parameterRangeState.exposureTimeMin || 1}
                    max={parameterRangeState.exposureTimeMax || 1000}
                    onChange={(e, value) => {
                      // Clamp value and update
                      const min = parameterRangeState.exposureTimeMin || 1;
                      const max = parameterRangeState.exposureTimeMax || 1000;
                      let v = Math.max(min, Math.min(max, value));
                      setExposureTime(v);
                    }}
                    valueLabelDisplay="auto"
                    sx={{ flex: 1 }}
                  />
                  <TextField
                    type="number"
                    value={exposureTime}
                    onChange={e => {
                      const min = parameterRangeState.exposureTimeMin || 1;
                      const max = parameterRangeState.exposureTimeMax || 1000;
                      let v = parseInt(e.target.value) || min;
                      v = Math.max(min, Math.min(max, v));
                      setExposureTime(v);
                    }}
                    inputProps={{
                      min: parameterRangeState.exposureTimeMin || 1,
                      max: parameterRangeState.exposureTimeMax || 1000,
                      style: { width: 80 }
                    }}
                    size="small"
                  />
                </Box>
              </Grid>

              <Grid item xs={12}>
                <Typography gutterBottom>
                  Detector Gain: {detectorGain}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Slider
                    value={detectorGain}
                    min={parameterRangeState.detectorGainMin || 0}
                    max={parameterRangeState.detectorGainMax || 100}
                    onChange={(e, value) => {
                      // Clamp value and update
                      const min = parameterRangeState.detectorGainMin || 0;
                      const max = parameterRangeState.detectorGainMax || 100;
                      let v = Math.max(min, Math.min(max, value));
                      updateDetectorGain(v);
                    }}
                    valueLabelDisplay="auto"
                    sx={{ flex: 1 }}
                  />
                  <TextField
                    type="number"
                    value={detectorGain}
                    onChange={e => {
                      const min = parameterRangeState.detectorGainMin || 0;
                      const max = parameterRangeState.detectorGainMax || 100;
                      let v = parseInt(e.target.value) || min;
                      v = Math.max(min, Math.min(max, v));
                      updateDetectorGain(v);
                    }}
                    inputProps={{
                      min: parameterRangeState.detectorGainMin || 0,
                      max: parameterRangeState.detectorGainMax || 100,
                      style: { width: 80 }
                    }}
                    size="small"
                  />
                </Box>
              </Grid>

              {/* Laser Controls */}
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Laser Control
                </Typography>
                {Object.entries(laserIntensities).map(([laserName, intensity]) => {
                  const isActive = laserActiveStates[laserName] || false;
                  const minValue = parameterRangeState.illuSourceMinIntensities?.[parameterRangeState.illuSources?.indexOf(laserName)] || 0;
                  const maxValue = parameterRangeState.illuSourceMaxIntensities?.[parameterRangeState.illuSources?.indexOf(laserName)] || 1023;
                  
                  return (
                    <Grid item xs={12} key={laserName} sx={{ mb: 2 }}>
                      <Card>
                        <CardContent>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                            <Typography sx={{ minWidth: 80 }}>
                              {laserName}
                            </Typography>
                            <Slider
                              value={intensity}
                              onChange={(e, value) => setLaserIntensity(laserName, value)}
                              min={minValue}
                              max={maxValue}
                              valueLabelDisplay="auto"
                              sx={{ flex: 1 }}
                            />
                            <Typography sx={{ mx: 1, minWidth: 50 }}>
                              {intensity}
                            </Typography>
                            <Checkbox
                              checked={isActive}
                              onChange={(e) => setLaserActive(laserName, e.target.checked)}
                            />
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
                
                {Object.keys(laserIntensities).length === 0 && (
                  <Typography color="textSecondary">
                    No lasers configured. Laser controls will appear when connected to backend.
                  </Typography>
                )}
              </Grid>
              
              <Grid item xs={6}>
                <Button
                  variant="contained"
                  onClick={startExperiment}
                  disabled={isRunning}
                  color="primary"
                  fullWidth
                >
                  Start STORM
                </Button>
              </Grid>
              
              <Grid item xs={6}>
                <Button
                  variant="contained"
                  onClick={stopExperiment}
                  disabled={!isRunning}
                  color="secondary"
                  fullWidth
                >
                  Stop STORM
                </Button>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Crop Region Tab */}
          <TabPanel value={tabIndex} index={1}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h6">
                    Select Crop Region
                  </Typography>
                  <Box>
                    <Button
                      variant="outlined"
                      onClick={handleLoadImage}
                      size="small"
                      sx={{ mr: 1 }}
                    >
                      Load Image
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<RefreshIcon />}
                      onClick={resetCropRegion}
                      size="small"
                    >
                      Reset Crop
                    </Button>
                  </Box>
                </Box>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Load an image, then click and drag to select the crop region
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <canvas
                  ref={canvasRef}
                  width={imageDims.width || 400}
                  height={imageDims.height || 400}
                  style={{
                    border: '1px solid #ccc',
                    cursor: isCropping ? 'crosshair' : 'pointer',
                    maxWidth: '100%',
                  }}
                  onMouseDown={handleCropMouseDown}
                  onMouseMove={handleCropMouseMove}
                  onMouseUp={handleCropMouseUp}
                  onMouseLeave={handleCropMouseUp}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="X Position"
                  type="number"
                  value={cropRegion.x}
                  onChange={(e) => dispatch(stormSlice.setCropX(parseInt(e.target.value) || 0))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Y Position"
                  type="number"
                  value={cropRegion.y}
                  onChange={(e) => dispatch(stormSlice.setCropY(parseInt(e.target.value) || 0))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Width"
                  type="number"
                  value={cropRegion.width}
                  onChange={(e) => dispatch(stormSlice.setCropWidth(parseInt(e.target.value) || 1))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Height"
                  type="number"
                  value={cropRegion.height}
                  onChange={(e) => dispatch(stormSlice.setCropHeight(parseInt(e.target.value) || 1))}
                  fullWidth
                />
              </Grid>
            </Grid>
          </TabPanel>

          {/* Reconstructed Image Tab */}
          <TabPanel value={tabIndex} index={2}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Reconstructed STORM Image
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  This tab will display the reconstructed STORM image when available.
                </Typography>
              </Grid>
              
              <Grid item xs={12}>
                {reconstructedImage ? (
                  <img
                    src={reconstructedImage}
                    alt="Reconstructed STORM"
                    style={{ maxWidth: '100%', height: 'auto' }}
                  />
                ) : (
                  <Box
                    sx={{
                      width: '100%',
                      height: 400,
                      backgroundColor: '#f5f5f5',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: '1px dashed #ccc',
                    }}
                  >
                    <Typography color="textSecondary">
                      No reconstructed image available
                    </Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </TabPanel>

          {/* STORM Reconstruction Tab */}
          <TabPanel value={tabIndex} index={3}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  STORM Reconstruction Parameters
                </Typography>
              </Grid>
              
              {/* Main Processing Parameters */}
              <Grid item xs={12}>
                <Card sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      Main Processing Parameters
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography gutterBottom>
                          Detection Threshold: {stormParameters.threshold}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Slider
                            value={stormParameters.threshold}
                            min={0.0}
                            max={1.0}
                            step={0.01}
                            onChange={(e, value) => setStormParameter('threshold', value)}
                            valueLabelDisplay="auto"
                            sx={{ flex: 1 }}
                          />
                          <TextField
                            type="number"
                            value={stormParameters.threshold}
                            onChange={e => setStormParameter('threshold', Math.max(0.0, Math.min(1.0, parseFloat(e.target.value) || 0.0)))}
                            inputProps={{
                              min: 0.0,
                              max: 1.0,
                              step: 0.01,
                              style: { width: 80 }
                            }}
                            size="small"
                          />
                        </Box>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <Typography gutterBottom>
                          Fit ROI Size: {stormParameters.fit_roi_size}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Slider
                            value={stormParameters.fit_roi_size}
                            min={7}
                            max={99}
                            step={2}
                            onChange={(e, value) => setStormParameter('fit_roi_size', value)}
                            valueLabelDisplay="auto"
                            marks={[
                              { value: 7, label: '7' },
                              { value: 99, label: '99' }
                            ]}
                            sx={{ flex: 1 }}
                          />
                          <TextField
                            type="number"
                            value={stormParameters.fit_roi_size}
                            onChange={e => {
                              let value = parseInt(e.target.value) || 7;
                              // Ensure odd number between 7 and 99
                              value = Math.max(7, Math.min(99, value));
                              if (value % 2 === 0) value = value + 1; // Make it odd
                              setStormParameter('fit_roi_size', value);
                            }}
                            inputProps={{
                              min: 7,
                              max: 99,
                              step: 2,
                              style: { width: 80 }
                            }}
                            size="small"
                          />
                        </Box>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth size="small">
                          <InputLabel>Fitting Method</InputLabel>
                          <Select
                            value={stormParameters.fitting_method}
                            onChange={e => setStormParameter('fitting_method', e.target.value)}
                            label="Fitting Method"
                          >
                            <MenuItem value="2D_Phasor_CPU">2D Phasor CPU</MenuItem>
                            <MenuItem value="2D_Gauss_MLE_fixed_sigma">2D Gauss MLE Fixed Sigma</MenuItem>
                            <MenuItem value="2D_Gauss_MLE_free_sigma">2D Gauss MLE Free Sigma</MenuItem>
                            <MenuItem value="2D_Gauss_MLE_elliptical_sigma">2D Gauss MLE Elliptical Sigma</MenuItem>
                            <MenuItem value="3D_Gauss_MLE_cspline_sigma">3D Gauss MLE CSpline Sigma</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth size="small">
                          <InputLabel>Filter Type</InputLabel>
                          <Select
                            value={stormParameters.filter_type}
                            onChange={e => setStormParameter('filter_type', e.target.value)}
                            label="Filter Type"
                          >
                            <MenuItem value="bandpass">Bandpass</MenuItem>
                            <MenuItem value="difference_of_gaussians">Difference of Gaussians</MenuItem>
                            <MenuItem value="temporal_median">Temporal Median</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <Typography gutterBottom>
                          Update Rate: {stormParameters.update_rate}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Slider
                            value={stormParameters.update_rate}
                            min={1}
                            max={100}
                            onChange={(e, value) => setStormParameter('update_rate', value)}
                            valueLabelDisplay="auto"
                            marks={[
                              { value: 1, label: '1' },
                              { value: 50, label: '50' },
                              { value: 100, label: '100' }
                            ]}
                            sx={{ flex: 1 }}
                          />
                          <TextField
                            type="number"
                            value={stormParameters.update_rate}
                            onChange={e => setStormParameter('update_rate', Math.max(1, Math.min(100, parseInt(e.target.value) || 1)))}
                            inputProps={{
                              min: 1,
                              max: 100,
                              style: { width: 80 }
                            }}
                            size="small"
                          />
                        </Box>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={stormParameters.temporal_median_enabled}
                              onChange={e => setStormParameter('temporal_median_enabled', e.target.checked)}
                            />
                          }
                          label="Enable Temporal Median"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Pixel Size (nm)"
                          type="number"
                          value={stormParameters.pixel_size_nm}
                          onChange={e => setStormParameter('pixel_size_nm', Math.max(0.1, parseFloat(e.target.value) || 117.5))}
                          inputProps={{
                            min: 0.1,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Super Resolution Pixel Size (nm)"
                          type="number"
                          value={stormParameters.super_resolution_pixel_size_nm}
                          onChange={e => setStormParameter('super_resolution_pixel_size_nm', Math.max(0.1, parseFloat(e.target.value) || 10.0))}
                          inputProps={{
                            min: 0.1,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Bandpass Filter Parameters */}
              <Grid item xs={12}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">Bandpass Filter Parameters</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Center Frequency"
                          type="number"
                          value={stormParameters.bandpass_filter.center}
                          onChange={e => setBandpassFilterParameter('center', Math.max(0.1, parseFloat(e.target.value) || 40.0))}
                          inputProps={{
                            min: 0.1,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Filter Width"
                          type="number"
                          value={stormParameters.bandpass_filter.width}
                          onChange={e => setBandpassFilterParameter('width', Math.max(0.1, parseFloat(e.target.value) || 90.0))}
                          inputProps={{
                            min: 0.1,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth size="small">
                          <InputLabel>Bandpass Filter Type</InputLabel>
                          <Select
                            value={stormParameters.bandpass_filter.filter_type}
                            onChange={e => setBandpassFilterParameter('filter_type', e.target.value)}
                            label="Bandpass Filter Type"
                          >
                            <MenuItem value="gauss">Gauss</MenuItem>
                            <MenuItem value="butterworth">Butterworth</MenuItem>
                            <MenuItem value="ideal">Ideal</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={stormParameters.bandpass_filter.show_filter}
                              onChange={e => setBandpassFilterParameter('show_filter', e.target.checked)}
                            />
                          }
                          label="Show Filter Visualization"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>

              {/* Blob Detector Parameters */}
              <Grid item xs={12}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">Blob Detector Parameters</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Min Threshold"
                          type="number"
                          value={stormParameters.blob_detector.min_threshold}
                          onChange={e => setBlobDetectorParameter('min_threshold', Math.max(0.0, Math.min(255.0, parseFloat(e.target.value) || 0.0)))}
                          inputProps={{
                            min: 0.0,
                            max: 255.0,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Max Threshold"
                          type="number"
                          value={stormParameters.blob_detector.max_threshold}
                          onChange={e => setBlobDetectorParameter('max_threshold', Math.max(0.0, Math.min(255.0, parseFloat(e.target.value) || 255.0)))}
                          inputProps={{
                            min: 0.0,
                            max: 255.0,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Min Area"
                          type="number"
                          value={stormParameters.blob_detector.min_area}
                          onChange={e => setBlobDetectorParameter('min_area', Math.max(0.1, parseFloat(e.target.value) || 1.5))}
                          inputProps={{
                            min: 0.1,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Max Area"
                          type="number"
                          value={stormParameters.blob_detector.max_area}
                          onChange={e => setBlobDetectorParameter('max_area', Math.max(0.1, parseFloat(e.target.value) || 80.0))}
                          inputProps={{
                            min: 0.1,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Min Circularity (null to disable)"
                          type="number"
                          value={stormParameters.blob_detector.min_circularity || ''}
                          onChange={e => setBlobDetectorParameter('min_circularity', e.target.value ? Math.max(0.0, Math.min(1.0, parseFloat(e.target.value))) : null)}
                          inputProps={{
                            min: 0.0,
                            max: 1.0,
                            step: 0.01
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Min Convexity (null to disable)"
                          type="number"
                          value={stormParameters.blob_detector.min_convexity || ''}
                          onChange={e => setBlobDetectorParameter('min_convexity', e.target.value ? Math.max(0.0, Math.min(1.0, parseFloat(e.target.value))) : null)}
                          inputProps={{
                            min: 0.0,
                            max: 1.0,
                            step: 0.01
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Min Inertia Ratio (null to disable)"
                          type="number"
                          value={stormParameters.blob_detector.min_inertia_ratio || ''}
                          onChange={e => setBlobDetectorParameter('min_inertia_ratio', e.target.value ? Math.max(0.0, Math.min(1.0, parseFloat(e.target.value))) : null)}
                          inputProps={{
                            min: 0.0,
                            max: 1.0,
                            step: 0.01
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Blob Color"
                          type="number"
                          value={stormParameters.blob_detector.blob_color}
                          onChange={e => setBlobDetectorParameter('blob_color', Math.max(0, Math.min(255, parseInt(e.target.value) || 255)))}
                          inputProps={{
                            min: 0,
                            max: 255,
                            step: 1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Min Distance Between Blobs"
                          type="number"
                          value={stormParameters.blob_detector.min_dist_between_blobs}
                          onChange={e => setBlobDetectorParameter('min_dist_between_blobs', Math.max(0.0, parseFloat(e.target.value) || 0.0))}
                          inputProps={{
                            min: 0.0,
                            step: 0.1
                          }}
                          size="small"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>

              {/* Acquisition Parameters */}
              <Grid item xs={12}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">Acquisition Parameters</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Session ID"
                          value={acquisitionParameters.session_id || ''}
                          onChange={e => setAcquisitionParameter('session_id', e.target.value)}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Max Frames (-1 for unlimited)"
                          type="number"
                          value={acquisitionParameters.max_frames}
                          onChange={e => setAcquisitionParameter('max_frames', parseInt(e.target.value) || -1)}
                          inputProps={{
                            min: -1,
                            step: 1
                          }}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="Save Directory"
                          value={acquisitionParameters.save_directory}
                          onChange={e => setAcquisitionParameter('save_directory', e.target.value)}
                          size="small"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth size="small">
                          <InputLabel>Save Format</InputLabel>
                          <Select
                            value={acquisitionParameters.save_format}
                            onChange={e => setAcquisitionParameter('save_format', e.target.value)}
                            label="Save Format"
                          >
                            <MenuItem value="tiff">TIFF</MenuItem>
                            <MenuItem value="omezarr">OME-Zarr</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={acquisitionParameters.save_enabled}
                              onChange={e => setAcquisitionParameter('save_enabled', e.target.checked)}
                            />
                          }
                          label="Enable Saving"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={acquisitionParameters.process_locally}
                              onChange={e => setAcquisitionParameter('process_locally', e.target.checked)}
                            />
                          }
                          label="Process Locally"
                        />
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={acquisitionParameters.process_arkitekt}
                              onChange={e => setAcquisitionParameter('process_arkitekt', e.target.checked)}
                            />
                          }
                          label="Process with Arkitekt"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>

              {/* Reconstruction Control */}
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Typography variant="h6">
                    Reconstruction Status:
                  </Typography>
                  {/* Show acquisitionActive and isReconstructing */}
                  {acquisitionActive ? (
                    <>
                      <CheckCircleIcon style={{ color: green[500], marginLeft: "10px" }} />
                      <Typography sx={{ ml: 1 }} color="success.main">Active</Typography>
                    </>
                  ) : (
                    <>
                      <CancelIcon style={{ color: red[500], marginLeft: "10px" }} />
                      <Typography sx={{ ml: 1 }} color="error.main">Inactive</Typography>
                    </>
                  )}
                  {/* Show internal reconstructing state for debugging */}
                  {isReconstructing && (
                    <Typography sx={{ ml: 2 }} color="info.main">(Reconstructing)</Typography>
                  )}
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Button
                  variant="contained"
                  onClick={updateProcessingParameters}
                  color="secondary"
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Update Processing Parameters
                </Button>
              </Grid>

              <Grid item xs={12} md={6}>
                <Button
                  variant="contained"
                  onClick={startReconstructionLocal}
                  disabled={isReconstructing}
                  color="primary"
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  {isReconstructing ? 'Reconstructing...' : 'Start Local Reconstruction'}
                </Button>
              </Grid>

              <Grid item xs={12} md={6}>
                <Button
                  variant="contained"
                  onClick={triggerReconstruction}
                  disabled={isReconstructing}
                  color="primary"
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  {isReconstructing ? 'Reconstructing...' : 'Trigger Single Frame Reconstruction'}
                </Button>
              </Grid>

              <Grid item xs={12} md={6}>
                <Button
                  variant="contained"
                  onClick={stopReconstructionLocal}
                  disabled={!acquisitionActive}
                  color="secondary"
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Stop Local Reconstruction
                </Button>
              </Grid>

              <Grid item xs={12} md={6}>
                <Button
                  variant="outlined"
                  onClick={getReconstructionStatus}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Get Reconstruction Status
                </Button>
              </Grid>

              <Grid item xs={12} md={6}>
                <Button
                  variant="outlined"
                  onClick={loadLastReconstructedImage}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Load Last Reconstructed Image
                </Button>
              </Grid>

              <Grid item xs={12} md={6}>
                <Button
                  variant="outlined"
                  onClick={loadProcessingParameters}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  Load Current Parameters
                </Button>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Live Reconstruction Tab */}
          <TabPanel value={tabIndex} index={4}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Live STORM Reconstruction
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  This tab displays the latest reconstruction image received via socket.
                </Typography>
              </Grid>
              <Grid item xs={12}>
                {reconImage ? (
                  <img src={reconImage} alt="Live STORM Reconstruction" style={{ maxWidth: '100%', height: 'auto' }} />
                ) : (
                  <Box sx={{ width: '100%', height: 400, backgroundColor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #ccc' }}>
                    <Typography color="textSecondary">No reconstruction image received yet</Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </TabPanel>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default STORMController;
