// src/components/STORMControllerLocal.js
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
  Checkbox,
  FormControlLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import { green, red } from "@mui/material/colors";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import RefreshIcon from "@mui/icons-material/Refresh";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useWebSocket } from "../context/WebSocketContext";
import * as stormSlice from "../state/slices/STORMSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import LiveViewSettings from "./LiveViewSettings";
import STORMPlot from "./STORMPlot";
import apiSTORMControllerStartReconstructionLocal from "../backendapi/apiSTORMControllerStartReconstructionLocal.js";
import apiSTORMControllerStopReconstructionLocal from "../backendapi/apiSTORMControllerStopReconstructionLocal.js";
import apiSTORMControllerSetProcessingParameters from "../backendapi/apiSTORMControllerSetProcessingParameters.js";
import apiSTORMControllerGetProcessingParameters from "../backendapi/apiSTORMControllerGetProcessingParameters.js";
import apiSTORMControllerGetReconstructionStatus from "../backendapi/apiSTORMControllerGetReconstructionStatus.js";
import apiSTORMControllerGetLastReconstructedImagePath from "../backendapi/apiSTORMControllerGetLastReconstructedImagePath.js";

// Plot component - placeholder for now
const StormPlot = ({ data }) => {
  return <STORMPlot data={data} title="Live Localizations" />;
};

// Statistics component - placeholder for now
const StormStatistics = ({ stats }) => {
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Statistics
      </Typography>
      {stats ? (
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2">
              Total Localizations: {stats.totalLocalizations || 0}
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              Processing Rate: {stats.processingRate || 0} fps
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              Duration: {stats.duration || 0} s
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              Quality Score: {stats.qualityScore || 0}
            </Typography>
          </Grid>
        </Grid>
      ) : (
        <Typography color="textSecondary">No statistics available</Typography>
      )}
    </Box>
  );
};

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`reconstruction-tabpanel-${index}`}
      aria-labelledby={`reconstruction-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 2 }}>{children}</Box>}
    </div>
  );
};

const STORMControllerLocal = () => {
  const dispatch = useDispatch();

  // Redux state
  const stormState = useSelector(stormSlice.getSTORMState);
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const parameterRangeState = useSelector(
    parameterRangeSlice.getParameterRangeState
  );
  const liveStreamState = useSelector((state) => state.liveStreamState);

  // Local state
  const [activeStep, setActiveStep] = useState(0);
  const [expandedAccordions, setExpandedAccordions] = useState({
    0: true, // Start with first accordion expanded
    1: false,
    2: false,
    3: false,
    4: false,
    5: false,
  });
  const [reconstructionTabIndex, setReconstructionTabIndex] = useState(0);
  const [visualizationTabIndex, setVisualizationTabIndex] = useState(0);
  const [cropImage, setCropImage] = useState(null);
  const [loadedImage, setLoadedImage] = useState(null);
  const [imageDims, setImageDims] = useState({ width: 0, height: 0 });
  const [isCropping, setIsCropping] = useState(false);
  const [cropStart, setCropStart] = useState({ x: 0, y: 0 });
  const [cropPreview, setCropPreview] = useState(null);
  const [acquisitionActive, setAcquisitionActive] = useState(false);
  const [reconImage, setReconImage] = useState(null);
  const [plotData, setPlotData] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [brightfieldImage, setBrightfieldImage] = useState(null);
  const canvasRef = useRef(null);

  const socket = useWebSocket();

  // Access Redux state
  const exposureTime = stormState.exposureTime;
  const cropRegion = stormState.cropRegion;
  const isReconstructing = stormState.isReconstructing;
  const stormParameters = stormState.stormParameters;
  const acquisitionParameters = stormState.acquisitionParameters;
  const reconstructedImage = stormState.reconstructedImage;
  const localizations = stormState.localizations;

  const steps = [
    "Crop Settings",
    "Autofocus Settings",
    "Prefilter Settings",
    "Localization Settings",
    "N-Frames/Display Settings",
    "Start/Stop Acquisition",
  ];

  // WebSocket signal handling
  useEffect(() => {
    if (!socket) return;

    const handleSignal = (data) => {
      try {
        const jdata = typeof data === "string" ? JSON.parse(data) : data;

        // Handle reconstruction image updates
        if (
          jdata.name === "sigExperimentImageUpdate" &&
          jdata.detectorname === "STORM" &&
          jdata.image
        ) {
          const imgSrc = `data:image/jpeg;base64,${jdata.image}`;
          setReconImage(imgSrc);
        }

        // Handle localization data for plotting
        if (
          jdata.name === "sigSTORMLocalizationUpdate" &&
          jdata.localizations
        ) {
          setPlotData(jdata.localizations);
        }

        // Handle statistics updates
        if (jdata.name === "sigSTORMStatisticsUpdate" && jdata.statistics) {
          setStatistics(jdata.statistics);
        }
      } catch (error) {
        console.error("Error parsing STORM signal data:", error);
      }
    };

    socket.on("signal", handleSignal);
    return () => socket.off("signal", handleSignal);
  }, [socket]);

  // Load initial image for cropping
  useEffect(() => {
    const loadInitialCropImage = async () => {
      if (liveStreamState.liveViewImage) {
        setCropImage(liveStreamState.liveViewImage);
      }
    };
    loadInitialCropImage();
  }, [liveStreamState.liveViewImage]);

  // Helper: Min/Max stretch for PNG
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
        let min = 255,
          max = 0;
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
      const img = new window.Image();
      img.onload = async () => {
        setImageDims({ width: img.naturalWidth, height: img.naturalHeight });
        const stretchedUrl = await minMaxStretch(url);
        setLoadedImage(stretchedUrl);
        dispatch(
          stormSlice.setCropRegion({
            x: 0,
            y: 0,
            width: img.naturalWidth,
            height: img.naturalHeight,
          })
        );
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

  const resetCropRegion = () => {
    if (loadedImage && imageDims.width && imageDims.height) {
      dispatch(
        stormSlice.setCropRegion({
          x: 0,
          y: 0,
          width: imageDims.width,
          height: imageDims.height,
        })
      );
    }
  };

  // Parameter setters
  const setStormParameter = async (paramName, value) => {
    const params = { [paramName]: value };
    try {
      await apiSTORMControllerSetProcessingParameters({
        ...stormParameters,
        ...params,
      });
      dispatch(stormSlice.setStormParameters(params));
    } catch (error) {
      console.error("Error setting STORM parameter:", error);
    }
  };

  const setAcquisitionParameter = (paramName, value) => {
    const updatedAcquisition = { ...acquisitionParameters, [paramName]: value };
    dispatch(stormSlice.setAcquisitionParameters(updatedAcquisition));
  };

  // Reconstruction controls
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
        },
        processing_parameters: stormParameters,
        save_enabled: acquisitionParameters.save_enabled,
      };

      await apiSTORMControllerStartReconstructionLocal(reconstructionRequest);
      dispatch(stormSlice.setIsReconstructing(true));
    } catch (error) {
      console.error("Error starting local reconstruction:", error);
    }
  };

  const stopReconstructionLocal = async () => {
    try {
      await apiSTORMControllerStopReconstructionLocal();
      dispatch(stormSlice.setIsReconstructing(false));
    } catch (error) {
      console.error("Error stopping local reconstruction:", error);
    }
  };

  const getReconstructionStatus = async () => {
    try {
      const status = await apiSTORMControllerGetReconstructionStatus();
      dispatch(
        stormSlice.setIsReconstructing(status.isReconstructing || false)
      );
      setAcquisitionActive(!!status.acquisition_active);
      return status;
    } catch (error) {
      console.error("Error getting reconstruction status:", error);
      setAcquisitionActive(false);
      return null;
    }
  };

  // Load brightfield image (placeholder endpoint)
  const loadBrightfieldImage = async () => {
    try {
      // This endpoint doesn't exist yet, so placeholder
      setBrightfieldImage("/api/placeholder/brightfield.jpg");
    } catch (error) {
      console.error("Error loading brightfield image:", error);
    }
  };

  // Reset localizations
  const resetLocalizations = () => {
    dispatch(stormSlice.resetLocalizations());
  };

  // Handle direct step navigation (clicking on steps)
  const handleStepClick = (stepIndex) => {
    setActiveStep(stepIndex);
  };

  // Handle accordion expand/collapse
  const handleAccordionChange = (stepIndex) => (event, isExpanded) => {
    setExpandedAccordions((prev) => ({
      ...prev,
      [stepIndex]: isExpanded,
    }));
    if (isExpanded) {
      setActiveStep(stepIndex);
    }
  };

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleReconstructionTabChange = (event, newValue) => {
    setReconstructionTabIndex(newValue);
  };

  const handleVisualizationTabChange = (event, newValue) => {
    setVisualizationTabIndex(newValue);
  };

  const renderAccordionSteps = () => {
    return steps.map((stepTitle, index) => (
      <Accordion
        key={index}
        expanded={expandedAccordions[index]}
        onChange={handleAccordionChange(index)}
        sx={{
          mb: 1,
          "&:before": { display: "none" }, // Remove default accordion divider
          boxShadow: "none", // Remove shadow
          border: "1px solid #e0e0e0", // Add subtle border
          borderRadius: "4px !important",
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          sx={{
            backgroundColor: activeStep === index ? "#f5f5f5" : "transparent",
            "&:hover": { backgroundColor: "#f9f9f9" },
          }}
        >
          <Typography
            variant="subtitle1"
            sx={{ fontWeight: activeStep === index ? 600 : 400 }}
          >
            {index + 1}. {stepTitle}
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ pt: 0 }}>
          {renderStepContent(index)}
        </AccordionDetails>
      </Accordion>
    ));
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0: // Crop Settings
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Crop Settings
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
                  <Button
                    variant="outlined"
                    onClick={handleLoadImage}
                    size="small"
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
                <canvas
                  ref={canvasRef}
                  width={imageDims.width || 400}
                  height={imageDims.height || 300}
                  style={{
                    border: "1px solid #ccc",
                    cursor: isCropping ? "crosshair" : "pointer",
                    maxWidth: "100%",
                    maxHeight: "300px",
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
                  onChange={(e) =>
                    dispatch(stormSlice.setCropX(parseInt(e.target.value) || 0))
                  }
                  fullWidth
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Y Position"
                  type="number"
                  value={cropRegion.y}
                  onChange={(e) =>
                    dispatch(stormSlice.setCropY(parseInt(e.target.value) || 0))
                  }
                  fullWidth
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Width"
                  type="number"
                  value={cropRegion.width}
                  onChange={(e) =>
                    dispatch(
                      stormSlice.setCropWidth(parseInt(e.target.value) || 1)
                    )
                  }
                  fullWidth
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Height"
                  type="number"
                  value={cropRegion.height}
                  onChange={(e) =>
                    dispatch(
                      stormSlice.setCropHeight(parseInt(e.target.value) || 1)
                    )
                  }
                  fullWidth
                  size="small"
                />
              </Grid>
            </Grid>
          </Box>
        );

      case 1: // Autofocus Settings
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Autofocus Settings
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={acquisitionParameters.autofocus_enabled || false}
                      onChange={(e) =>
                        setAcquisitionParameter(
                          "autofocus_enabled",
                          e.target.checked
                        )
                      }
                    />
                  }
                  label="Enable Autofocus"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Autofocus Interval (frames)"
                  type="number"
                  value={acquisitionParameters.autofocus_interval || 100}
                  onChange={(e) =>
                    setAcquisitionParameter(
                      "autofocus_interval",
                      parseInt(e.target.value) || 100
                    )
                  }
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Focus Range (µm)"
                  type="number"
                  value={acquisitionParameters.focus_range || 5}
                  onChange={(e) =>
                    setAcquisitionParameter(
                      "focus_range",
                      parseFloat(e.target.value) || 5
                    )
                  }
                  size="small"
                />
              </Grid>
            </Grid>
          </Box>
        );

      case 2: // Prefilter Settings
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Prefilter Settings
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <FormControl fullWidth size="small">
                  <InputLabel>Filter Type</InputLabel>
                  <Select
                    value={stormParameters.filter_type}
                    onChange={(e) =>
                      setStormParameter("filter_type", e.target.value)
                    }
                    label="Filter Type"
                  >
                    <MenuItem value="bandpass">Bandpass</MenuItem>
                    <MenuItem value="difference_of_gaussians">
                      Difference of Gaussians
                    </MenuItem>
                    <MenuItem value="temporal_median">Temporal Median</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={stormParameters.temporal_median_enabled}
                      onChange={(e) =>
                        setStormParameter(
                          "temporal_median_enabled",
                          e.target.checked
                        )
                      }
                    />
                  }
                  label="Enable Temporal Median"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Bandpass Center"
                  type="number"
                  value={stormParameters.bandpass_filter?.center || 40}
                  onChange={(e) =>
                    setStormParameter("bandpass_filter", {
                      ...stormParameters.bandpass_filter,
                      center: parseFloat(e.target.value) || 40,
                    })
                  }
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Bandpass Width"
                  type="number"
                  value={stormParameters.bandpass_filter?.width || 90}
                  onChange={(e) =>
                    setStormParameter("bandpass_filter", {
                      ...stormParameters.bandpass_filter,
                      width: parseFloat(e.target.value) || 90,
                    })
                  }
                  size="small"
                />
              </Grid>
            </Grid>
          </Box>
        );

      case 3: // Localization Settings
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Localization Settings
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography gutterBottom>
                  Detection Threshold: {stormParameters.threshold}
                </Typography>
                <Slider
                  value={stormParameters.threshold}
                  min={0.0}
                  max={1.0}
                  step={0.01}
                  onChange={(e, value) => setStormParameter("threshold", value)}
                  valueLabelDisplay="auto"
                />
              </Grid>
              <Grid item xs={6}>
                <Typography gutterBottom>
                  Fit ROI Size: {stormParameters.fit_roi_size}
                </Typography>
                <Slider
                  value={stormParameters.fit_roi_size}
                  min={7}
                  max={99}
                  step={2}
                  onChange={(e, value) =>
                    setStormParameter("fit_roi_size", value)
                  }
                  valueLabelDisplay="auto"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControl fullWidth size="small">
                  <InputLabel>Fitting Method</InputLabel>
                  <Select
                    value={stormParameters.fitting_method}
                    onChange={(e) =>
                      setStormParameter("fitting_method", e.target.value)
                    }
                    label="Fitting Method"
                  >
                    <MenuItem value="2D_Phasor_CPU">2D Phasor CPU</MenuItem>
                    <MenuItem value="2D_Gauss_MLE_fixed_sigma">
                      2D Gauss MLE Fixed Sigma
                    </MenuItem>
                    <MenuItem value="2D_Gauss_MLE_free_sigma">
                      2D Gauss MLE Free Sigma
                    </MenuItem>
                    <MenuItem value="2D_Gauss_MLE_elliptical_sigma">
                      2D Gauss MLE Elliptical Sigma
                    </MenuItem>
                    <MenuItem value="3D_Gauss_MLE_cspline_sigma">
                      3D Gauss MLE CSpline Sigma
                    </MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        );

      case 4: // N-Frames/Display Settings
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              N-Frames/Display Settings
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Max Frames (-1 for unlimited)"
                  type="number"
                  value={acquisitionParameters.max_frames}
                  onChange={(e) =>
                    setAcquisitionParameter(
                      "max_frames",
                      parseInt(e.target.value) || -1
                    )
                  }
                  size="small"
                />
              </Grid>
              <Grid item xs={6}>
                <Typography gutterBottom>
                  Update Rate: {stormParameters.update_rate}
                </Typography>
                <Slider
                  value={stormParameters.update_rate}
                  min={1}
                  max={100}
                  onChange={(e, value) =>
                    setStormParameter("update_rate", value)
                  }
                  valueLabelDisplay="auto"
                />
              </Grid>
              <Grid item xs={6}>
                <FormControl fullWidth size="small">
                  <InputLabel>Save Format</InputLabel>
                  <Select
                    value={acquisitionParameters.save_format}
                    onChange={(e) =>
                      setAcquisitionParameter("save_format", e.target.value)
                    }
                    label="Save Format"
                  >
                    <MenuItem value="tiff">TIFF</MenuItem>
                    <MenuItem value="omezarr">OME-Zarr</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={acquisitionParameters.save_enabled}
                      onChange={(e) =>
                        setAcquisitionParameter(
                          "save_enabled",
                          e.target.checked
                        )
                      }
                    />
                  }
                  label="Enable Saving"
                />
              </Grid>
            </Grid>
          </Box>
        );

      case 5: // Start/Stop Acquisition
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Start/Stop Acquisition
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Session ID"
                  value={acquisitionParameters.session_id || ""}
                  onChange={(e) =>
                    setAcquisitionParameter("session_id", e.target.value)
                  }
                  size="small"
                  sx={{ mb: 2 }}
                />
              </Grid>
              <Grid item xs={12}>
                <Box
                  sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}
                >
                  <Typography variant="h6">Status:</Typography>
                  {acquisitionActive ? (
                    <>
                      <CheckCircleIcon style={{ color: green[500] }} />
                      <Typography color="success.main">Active</Typography>
                    </>
                  ) : (
                    <>
                      <CancelIcon style={{ color: red[500] }} />
                      <Typography color="error.main">Inactive</Typography>
                    </>
                  )}
                </Box>
              </Grid>
              <Grid item xs={6}>
                <Button
                  variant="contained"
                  onClick={startReconstructionLocal}
                  disabled={isReconstructing}
                  color="primary"
                  fullWidth
                >
                  {isReconstructing ? "Starting..." : "Start Acquisition"}
                </Button>
              </Grid>
              <Grid item xs={6}>
                <Button
                  variant="contained"
                  onClick={stopReconstructionLocal}
                  disabled={!acquisitionActive}
                  color="secondary"
                  fullWidth
                >
                  Stop Acquisition
                </Button>
              </Grid>
              <Grid item xs={12}>
                <Button
                  variant="outlined"
                  onClick={getReconstructionStatus}
                  fullWidth
                >
                  Refresh Status
                </Button>
              </Grid>
            </Grid>
          </Box>
        );

      default:
        return "Unknown step";
    }
  };

  return (
    <Box
      sx={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        backgroundColor: "transparent",
      }}
    >
      <Typography variant="h5" sx={{ p: 2, borderBottom: "1px solid #ddd" }}>
        STORM Local Controller
      </Typography>

      <Grid container sx={{ flex: 1, height: "calc(100vh - 80px)" }}>
        {/* Left Column - Live View */}
        <Grid
          item
          xs={4}
          sx={{ borderRight: "1px solid #ddd", height: "100%" }}
        >
          <Box
            sx={{
              p: 2,
              height: "100%",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Typography variant="h6" gutterBottom>
              Live View Stream
            </Typography>
            <Box
              sx={{ flex: 1, minHeight: 400, border: "1px solid #ccc", mb: 2 }}
            >
              <LiveViewControlWrapper />
            </Box>
            <Typography variant="h6" gutterBottom>
              Live View Settings
            </Typography>
            <LiveViewSettings />
          </Box>
        </Grid>

        {/* Middle Column - Acquisition Settings Flow */}
        <Grid
          item
          xs={4}
          sx={{ borderRight: "1px solid #ddd", height: "100%" }}
        >
          <Box sx={{ p: 2, height: "100%", overflow: "auto" }}>
            <Typography variant="h6" gutterBottom>
              Acquisition Settings
            </Typography>

            {/* Vertical Accordion Steps */}
            <Box sx={{ mt: 2 }}>{renderAccordionSteps()}</Box>
          </Box>
        </Grid>

        {/* Right Column - Live Reconstruction */}
        <Grid item xs={4} sx={{ height: "100%" }}>
          <Box
            sx={{
              p: 2,
              height: "100%",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Tabs
              value={reconstructionTabIndex}
              onChange={handleReconstructionTabChange}
              aria-label="reconstruction tabs"
            >
              <Tab label="Live Reconstruction" />
              <Tab label="Statistics" />
            </Tabs>

            {/* Live Reconstruction Tab */}
            <TabPanel value={reconstructionTabIndex} index={0}>
              <Tabs
                value={visualizationTabIndex}
                onChange={handleVisualizationTabChange}
                aria-label="visualization tabs"
                variant="fullWidth"
              >
                <Tab label="Rendering" />
                <Tab label="XY Plot" />
                <Tab label="Brightfield" />
              </Tabs>

              <TabPanel value={visualizationTabIndex} index={0}>
                <Box sx={{ textAlign: "center" }}>
                  <Typography variant="h6" gutterBottom>
                    Reconstruction
                  </Typography>
                  {reconImage ? (
                    <img
                      src={reconImage}
                      alt="Live STORM Reconstruction"
                      style={{ maxWidth: "100%", maxHeight: 400 }}
                    />
                  ) : (
                    <Box
                      sx={{
                        width: "100%",
                        height: 300,
                        backgroundColor: "#f5f5f5",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        border: "1px dashed #ccc",
                      }}
                    >
                      <Typography color="textSecondary">
                        No reconstruction available
                      </Typography>
                    </Box>
                  )}
                </Box>
              </TabPanel>

              <TabPanel value={visualizationTabIndex} index={1}>
                <Typography variant="h6" gutterBottom>
                  XY Plot
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Button
                    variant="outlined"
                    onClick={resetLocalizations}
                    size="small"
                  >
                    Reset Localizations
                  </Button>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Total localizations: {localizations.length}
                  </Typography>
                </Box>
                <StormPlot data={localizations} />
              </TabPanel>

              <TabPanel value={visualizationTabIndex} index={2}>
                <Box sx={{ textAlign: "center" }}>
                  <Typography variant="h6" gutterBottom>
                    Brightfield
                  </Typography>
                  <Button
                    variant="outlined"
                    onClick={loadBrightfieldImage}
                    sx={{ mb: 2 }}
                  >
                    Load Brightfield
                  </Button>
                  {brightfieldImage ? (
                    <img
                      src={brightfieldImage}
                      alt="Brightfield"
                      style={{ maxWidth: "100%", maxHeight: 400 }}
                    />
                  ) : (
                    <Box
                      sx={{
                        width: "100%",
                        height: 300,
                        backgroundColor: "#f5f5f5",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        border: "1px dashed #ccc",
                      }}
                    >
                      <Typography color="textSecondary">
                        No brightfield image loaded
                      </Typography>
                    </Box>
                  )}
                </Box>
              </TabPanel>
            </TabPanel>

            {/* Statistics Tab */}
            <TabPanel value={reconstructionTabIndex} index={1}>
              <StormStatistics stats={statistics} />
            </TabPanel>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default STORMControllerLocal;
