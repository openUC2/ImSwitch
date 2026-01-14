/**
 * VizarrViewer.jsx
 * 
 * An offline-capable OME-Zarr viewer component using zarrita + Canvas rendering.
 * This provides similar functionality to the online vizarr viewer but works
 * entirely offline within the application.
 * 
 * Uses Canvas-based rendering for maximum compatibility, avoiding deck.gl
 * version conflicts with viv.
 * 
 * Supports both multiscale and single-scale OME-Zarr files.
 */

import React, { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { useSelector, useDispatch } from "react-redux";
import * as zarr from "zarrita";
import {
  Box,
  IconButton,
  Slider,
  Typography,
  Tooltip,
  Paper,
  CircularProgress,
  Alert,
  Button,
  Stack,
  Collapse,
  Divider,
  LinearProgress,
} from "@mui/material";
import {
  ZoomIn,
  ZoomOut,
  FitScreen,
  Settings,
  Close,
  Refresh,
  OpenInNew,
  NavigateBefore,
  NavigateNext,
  Visibility,
  VisibilityOff,
} from "@mui/icons-material";

import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import * as vizarrViewerSlice from "../state/slices/VizarrViewerSlice";

// Default channel colors for multi-channel images
const DEFAULT_CHANNEL_COLORS = [
  [255, 0, 0],    // Red
  [0, 255, 0],    // Green
  [0, 0, 255],    // Blue
  [255, 255, 0],  // Yellow
  [255, 0, 255],  // Magenta
  [0, 255, 255],  // Cyan
  [255, 128, 0],  // Orange
  [128, 0, 255],  // Purple
];

/**
 * Normalize zarr dtype string to a standard format
 */
function normalizeDtype(dtype) {
  const dtypeStr = String(dtype).toLowerCase();
  if (dtypeStr.includes("uint8") || dtypeStr === "<u1" || dtypeStr === "|u1" || dtypeStr === ">u1") return "uint8";
  if (dtypeStr.includes("uint16") || dtypeStr === "<u2" || dtypeStr === ">u2") return "uint16";
  if (dtypeStr.includes("uint32") || dtypeStr === "<u4" || dtypeStr === ">u4") return "uint32";
  if (dtypeStr.includes("int8") || dtypeStr === "<i1" || dtypeStr === "|i1" || dtypeStr === ">i1") return "int8";
  if (dtypeStr.includes("int16") || dtypeStr === "<i2" || dtypeStr === ">i2") return "int16";
  if (dtypeStr.includes("int32") || dtypeStr === "<i4" || dtypeStr === ">i4") return "int32";
  if (dtypeStr.includes("float32") || dtypeStr === "<f4" || dtypeStr === ">f4") return "float32";
  if (dtypeStr.includes("float64") || dtypeStr === "<f8" || dtypeStr === ">f8") return "float64";
  return "uint8";
}

/**
 * Get max value for dtype for normalization
 */
function getMaxValueForDtype(dtype) {
  switch (dtype) {
    case "uint8": return 255;
    case "uint16": return 65535;
    case "uint32": return 4294967295;
    case "int8": return 127;
    case "int16": return 32767;
    case "int32": return 2147483647;
    case "float32":
    case "float64":
      return 1.0; // Assume normalized floats, will auto-scale
    default: return 255;
  }
}

const VizarrViewer = ({ 
  zarrUrl: propZarrUrl = null, 
  onClose = null,
  embedded = false,
  height = "100%",
  width = "100%"
}) => {
  const dispatch = useDispatch();
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  
  // Get connection settings for building full URL
  const connectionSettings = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  
  // Get Vizarr viewer state from Redux
  const vizarrState = useSelector(vizarrViewerSlice.getVizarrViewerState);
  
  // Determine the URL to use (prop takes precedence over Redux state)
  const zarrUrl = propZarrUrl || vizarrState.currentUrl;
  
  // Local component state
  const [loading, setLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [error, setError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [metadata, setMetadata] = useState(null);
  
  // Image data state
  const [imageData, setImageData] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0, channels: 1, z: 1, t: 1 });
  const [dtype, setDtype] = useState("uint8");
  
  // View state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  // Channel-specific settings
  const [channelSettings, setChannelSettings] = useState([]);
  
  // Current slice indices for z-stack and time series
  const [currentZ, setCurrentZ] = useState(0);
  const [currentT, setCurrentT] = useState(0);
  
  // Store zarr arrays for on-demand loading
  const [zarrArrays, setZarrArrays] = useState(null);
  const [axisLabels, setAxisLabels] = useState(["y", "x"]);
  
  // Build the full URL for the Zarr file
  const fullUrl = useMemo(() => {
    if (!zarrUrl) return null;
    
    // If URL already contains protocol, use as is
    if (zarrUrl.startsWith("http://") || zarrUrl.startsWith("https://")) {
      return zarrUrl;
    }
    
    // Build URL from connection settings
    // Check if zarrUrl already starts with /imswitch/data/ to avoid double prefix
    const urlPath = zarrUrl.startsWith("/imswitch/data/") || zarrUrl.startsWith("/imswitch/data") 
      ? zarrUrl 
      : `/imswitch/data${zarrUrl}`;
    return `${connectionSettings.ip}:${connectionSettings.apiPort}${urlPath}`;
  }, [zarrUrl, connectionSettings.ip, connectionSettings.apiPort]);
  
  // Initialize channel settings when dimensions change
  const initializeChannelSettings = useCallback((numChannels, dataDtype) => {
    const maxVal = getMaxValueForDtype(dataDtype);
    const settings = Array.from({ length: numChannels }, (_, i) => ({
      visible: i === 0 || numChannels <= 3, // Show first channel or all if <= 3
      color: numChannels === 1 ? [255, 255, 255] : DEFAULT_CHANNEL_COLORS[i % DEFAULT_CHANNEL_COLORS.length],
      contrastMin: 0,
      contrastMax: maxVal,
      opacity: 1.0,
    }));
    setChannelSettings(settings);
  }, []);
  
  /**
   * Load a specific Z/T slice from the zarr array
   */
  const loadSlice = useCallback(async (arr, labels, shape, z, t, numChannels, setProgress) => {
    console.log("[VizarrViewer] Loading slice z:", z, "t:", t);
    if (setProgress) setProgress(70);
    
    try {
      const xIdx = labels.indexOf("x");
      const yIdx = labels.indexOf("y");
      const cIdx = labels.indexOf("c");
      const zIdx = labels.indexOf("z");
      const tIdx = labels.indexOf("t");
      
      // Build selection for all channels at this Z/T position
      const channelData = [];
      
      for (let c = 0; c < numChannels; c++) {
        // Build selection array - null means full slice for that dimension
        const selection = shape.map((_, idx) => {
          if (idx === xIdx) return null; // Full x slice
          if (idx === yIdx) return null; // Full y slice
          if (idx === cIdx) return c;
          if (idx === zIdx) return z;
          if (idx === tIdx) return t;
          return 0; // Default for unknown dimensions
        });
        
        try {
          const result = await zarr.get(arr, selection);
          
          // Determine width and height from result shape
          // Result shape should be 2D [height, width] after slicing
          const resultShape = result.shape;
          const resultWidth = resultShape[resultShape.length - 1];
          const resultHeight = resultShape[resultShape.length - 2] || 1;
          
          channelData.push({
            data: result.data,
            width: resultWidth,
            height: resultHeight,
          });
        } catch (err) {
          console.error(`[VizarrViewer] Error loading channel ${c}:`, err);
          // Create empty channel data on error
          const imgWidth = xIdx >= 0 ? shape[xIdx] : shape[shape.length - 1];
          const imgHeight = yIdx >= 0 ? shape[yIdx] : shape[shape.length - 2];
          channelData.push({
            data: new Uint8Array(imgWidth * imgHeight),
            width: imgWidth,
            height: imgHeight,
          });
        }
        
        if (setProgress) setProgress(70 + Math.round(30 * (c + 1) / numChannels));
      }
      
      setImageData(channelData);
      console.log("[VizarrViewer] Loaded", channelData.length, "channels");
      
      return channelData;
    } catch (err) {
      console.error("[VizarrViewer] Error loading slice:", err);
      throw err;
    }
  }, []);
  
  /**
   * Load OME-Zarr metadata and structure using zarrita
   */
  useEffect(() => {
    if (!fullUrl) {
      setError("No OME-Zarr URL provided");
      setLoading(false);
      return;
    }
    
    const loadMetadata = async () => {
      console.log("[VizarrViewer] Loading OME-Zarr from:", fullUrl);
      setLoading(true);
      setLoadingProgress(0);
      setError(null);
      setImageData(null);
      
      try {
        // Fetch .zattrs to get OME-Zarr metadata
        const attrsUrl = `${fullUrl}/.zattrs`;
        let attrs = {};
        
        try {
          const attrsResponse = await fetch(attrsUrl);
          if (attrsResponse.ok) {
            attrs = await attrsResponse.json();
          }
        } catch (e) {
          console.log("[VizarrViewer] No .zattrs at root, trying zarr v3 format");
        }
        
        // Also try zarr.json for v3
        if (Object.keys(attrs).length === 0) {
          try {
            const zarrJsonUrl = `${fullUrl}/zarr.json`;
            const zarrJsonResponse = await fetch(zarrJsonUrl);
            if (zarrJsonResponse.ok) {
              const zarrJson = await zarrJsonResponse.json();
              attrs = zarrJson.attributes || {};
            }
          } catch (e) {
            console.log("[VizarrViewer] No zarr.json found");
          }
        }
        
        console.log("[VizarrViewer] Root attributes:", attrs);
        setMetadata(attrs);
        setLoadingProgress(20);
        
        let arrayPath = "0";
        let labels = ["y", "x"];
        
        // Check if this is multiscale OME-Zarr
        if (attrs.multiscales && Array.isArray(attrs.multiscales) && attrs.multiscales.length > 0) {
          console.log("[VizarrViewer] Found multiscale OME-Zarr");
          const multiscale = attrs.multiscales[0];
          const datasets = multiscale.datasets || [];
          
          // Use highest resolution (first dataset)
          if (datasets.length > 0) {
            arrayPath = datasets[0].path;
          }
          
          // Get axis information if available
          if (multiscale.axes) {
            labels = multiscale.axes.map(a => typeof a === "string" ? a : a.name);
          }
        }
        
        setAxisLabels(labels);
        setLoadingProgress(40);
        
        // Load the array using zarrita
        const arrayUrl = `${fullUrl}/${arrayPath}`;
        console.log("[VizarrViewer] Loading array from:", arrayUrl);
        
        const store = new zarr.FetchStore(arrayUrl);
        const arr = await zarr.open(store, { kind: "array" });
        
        console.log("[VizarrViewer] Loaded array:", arr.shape, "dtype:", arr.dtype);
        setLoadingProgress(60);
        
        // Extract dimensions from shape based on axis labels
        const shape = arr.shape;
        const xIdx = labels.indexOf("x");
        const yIdx = labels.indexOf("y");
        const cIdx = labels.indexOf("c");
        const zIdx = labels.indexOf("z");
        const tIdx = labels.indexOf("t");
        
        const imgWidth = xIdx >= 0 ? shape[xIdx] : shape[shape.length - 1];
        const imgHeight = yIdx >= 0 ? shape[yIdx] : shape[shape.length - 2];
        const channels = cIdx >= 0 ? shape[cIdx] : 1;
        const zSlices = zIdx >= 0 ? shape[zIdx] : 1;
        const timePoints = tIdx >= 0 ? shape[tIdx] : 1;
        
        const dataDtype = normalizeDtype(arr.dtype);
        setDtype(dataDtype);
        setDimensions({ width: imgWidth, height: imgHeight, channels, z: zSlices, t: timePoints });
        initializeChannelSettings(channels, dataDtype);
        
        // Store zarr array reference and labels for later slice loading
        setZarrArrays({ arr, labels, shape });
        
        console.log("[VizarrViewer] Dimensions:", { width: imgWidth, height: imgHeight, channels, z: zSlices, t: timePoints });
        
        // Load initial slice
        await loadSlice(arr, labels, shape, 0, 0, channels, setLoadingProgress);
        
        setLoading(false);
        
      } catch (err) {
        console.error("[VizarrViewer] Error loading OME-Zarr:", err);
        setError(`Failed to load OME-Zarr: ${err.message}`);
        setLoading(false);
      }
    };
    
    loadMetadata();
  }, [fullUrl, initializeChannelSettings, vizarrState.refreshKey, loadSlice]);
  
  // Reload slice when Z or T changes
  useEffect(() => {
    if (!zarrArrays || loading) {
      return;
    }

    const { arr, labels, shape } = zarrArrays;

    loadSlice(
      arr,
      labels,
      shape,
      currentZ,
      currentT,
      dimensions.channels,
      setLoadingProgress  // Pass the correct progress setter function
    );
  }, [currentZ, currentT, zarrArrays, loading, dimensions.channels, loadSlice, setLoadingProgress]);
  
  /**
   * Render image data to canvas
   */
  useEffect(() => {
    if (!imageData || !canvasRef.current || imageData.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    
    const { width: imgWidth, height: imgHeight } = dimensions;
    
    // Set canvas size to match image
    canvas.width = imgWidth;
    canvas.height = imgHeight;
    
    // Create output image data
    const outputData = ctx.createImageData(imgWidth, imgHeight);
    const output = outputData.data;
    
    // Combine channels
    for (let y = 0; y < imgHeight; y++) {
      for (let x = 0; x < imgWidth; x++) {
        const pixelIdx = y * imgWidth + x;
        const outputIdx = pixelIdx * 4;
        
        let r = 0, g = 0, b = 0;
        
        // Blend all visible channels
        for (let c = 0; c < imageData.length; c++) {
          const settings = channelSettings[c];
          if (!settings || !settings.visible) continue;
          
          const channelPixels = imageData[c].data;
          const rawValue = channelPixels[pixelIdx] || 0;
          
          // Apply contrast limits
          const { contrastMin, contrastMax, color } = settings;
          const normalized = Math.max(0, Math.min(1, 
            (rawValue - contrastMin) / Math.max(1, contrastMax - contrastMin)
          ));
          
          // Add channel contribution
          r += normalized * color[0];
          g += normalized * color[1];
          b += normalized * color[2];
        }
        
        // Clamp and write to output
        output[outputIdx] = Math.min(255, Math.round(r));
        output[outputIdx + 1] = Math.min(255, Math.round(g));
        output[outputIdx + 2] = Math.min(255, Math.round(b));
        output[outputIdx + 3] = 255; // Full opacity
      }
    }
    
    ctx.putImageData(outputData, 0, 0);
    
  }, [imageData, channelSettings, dimensions]);
  
  // Zoom controls
  const handleZoomIn = useCallback(() => {
    setZoom(prev => Math.min(10, prev * 1.2));
  }, []);
  
  const handleZoomOut = useCallback(() => {
    setZoom(prev => Math.max(0.1, prev / 1.2));
  }, []);
  
  const handleFitToScreen = useCallback(() => {
    if (!containerRef.current || !dimensions.width) return;
    
    const container = containerRef.current;
    const containerWidth = container.clientWidth - 40;
    const containerHeight = container.clientHeight - 40;
    
    const scaleX = containerWidth / dimensions.width;
    const scaleY = containerHeight / dimensions.height;
    const newZoom = Math.min(scaleX, scaleY, 1);
    
    setZoom(newZoom);
    setPan({ x: 0, y: 0 });
  }, [dimensions]);
  
  // Mouse handlers for panning
  const handleMouseDown = useCallback((e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }, [pan]);
  
  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  }, [isDragging, dragStart]);
  
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);
  
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.1, Math.min(10, prev * delta)));
  }, []);
  
  // Channel setting updates
  const updateChannelSetting = useCallback((channelIndex, key, value) => {
    setChannelSettings(prev => {
      const newSettings = [...prev];
      newSettings[channelIndex] = { ...newSettings[channelIndex], [key]: value };
      return newSettings;
    });
  }, []);
  
  // Toggle channel visibility
  const toggleChannelVisibility = useCallback((channelIndex) => {
    setChannelSettings(prev => {
      const newSettings = [...prev];
      newSettings[channelIndex] = { 
        ...newSettings[channelIndex], 
        visible: !newSettings[channelIndex].visible 
      };
      return newSettings;
    });
  }, []);
  
  // Auto-contrast for a channel
  const autoContrast = useCallback((channelIndex) => {
    if (!imageData || !imageData[channelIndex]) return;
    
    const data = imageData[channelIndex].data;
    let min = Infinity, max = -Infinity;
    
    // Sample data to find min/max (sample for performance)
    const step = Math.max(1, Math.floor(data.length / 10000));
    for (let i = 0; i < data.length; i += step) {
      const val = data[i];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    
    // Add some padding
    const range = max - min;
    min = Math.max(0, min - range * 0.05);
    max = max + range * 0.05;
    
    setChannelSettings(prev => {
      const newSettings = [...prev];
      newSettings[channelIndex] = { 
        ...newSettings[channelIndex], 
        contrastMin: min,
        contrastMax: max,
      };
      return newSettings;
    });
  }, [imageData]);
  
  // Auto-contrast all channels
  const autoContrastAll = useCallback(() => {
    if (!imageData) return;
    for (let i = 0; i < imageData.length; i++) {
      autoContrast(i);
    }
  }, [imageData, autoContrast]);
  
  // Refresh handler
  const handleRefresh = useCallback(() => {
    dispatch(vizarrViewerSlice.refreshViewer());
  }, [dispatch]);
  
  // Open in external viewer
  const handleOpenExternal = useCallback(() => {
    if (fullUrl) {
      const externalUrl = `https://hms-dbmi.github.io/vizarr/?source=${encodeURIComponent(fullUrl)}`;
      window.open(externalUrl, "_blank");
    }
  }, [fullUrl]);
  
  // Close handler
  const handleClose = useCallback(() => {
    if (onClose) {
      onClose();
    } else {
      dispatch(vizarrViewerSlice.closeViewer());
    }
  }, [onClose, dispatch]);
  
  // Render error state
  if (error && !loading) {
    return (
      <Box
        sx={{
          height: embedded ? height : "100vh",
          width: embedded ? width : "100%",
          display: "flex",
          flexDirection: "column",
          bgcolor: "background.default",
        }}
      >
        <Box sx={{ p: 2, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Typography variant="h6">OME-Zarr Viewer</Typography>
          <IconButton onClick={handleClose}>
            <Close />
          </IconButton>
        </Box>
        <Box sx={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", p: 3 }}>
          <Alert 
            severity="error" 
            sx={{ maxWidth: 600 }}
            action={
              <Stack direction="row" spacing={1}>
                <Button color="inherit" size="small" onClick={handleRefresh}>
                  Retry
                </Button>
                <Button 
                  color="inherit" 
                  size="small" 
                  onClick={handleOpenExternal}
                  startIcon={<OpenInNew />}
                >
                  Open in Vizarr
                </Button>
              </Stack>
            }
          >
            <Typography variant="body2">{error}</Typography>
            {fullUrl && (
              <Typography variant="caption" display="block" sx={{ mt: 1, wordBreak: "break-all" }}>
                URL: {fullUrl}
              </Typography>
            )}
          </Alert>
        </Box>
      </Box>
    );
  }
  
  // Render loading state
  if (loading) {
    return (
      <Box
        sx={{
          height: embedded ? height : "100vh",
          width: embedded ? width : "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "background.default",
        }}
      >
        <CircularProgress size={60} />
        <Typography variant="body1" sx={{ mt: 2 }}>
          Loading OME-Zarr...
        </Typography>
        <Box sx={{ width: 300, mt: 2 }}>
          <LinearProgress variant="determinate" value={loadingProgress} />
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
          {loadingProgress}%
        </Typography>
      </Box>
    );
  }
  
  // Render empty state
  if (!imageData) {
    return (
      <Box
        sx={{
          height: embedded ? height : "100vh",
          width: embedded ? width : "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "background.default",
        }}
      >
        <Typography variant="body1" color="text.secondary">
          No image data loaded
        </Typography>
      </Box>
    );
  }
  
  return (
    <Box
      sx={{
        height: embedded ? height : "100vh",
        width: embedded ? width : "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: "background.default",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <Paper
        elevation={1}
        sx={{
          p: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderRadius: 0,
        }}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
            OME-Zarr Viewer
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {dimensions.width} × {dimensions.height}
            {dimensions.channels > 1 && ` • ${dimensions.channels}ch`}
            {dimensions.z > 1 && ` • ${dimensions.z}z`}
            {dimensions.t > 1 && ` • ${dimensions.t}t`}
          </Typography>
        </Stack>
        
        <Stack direction="row" spacing={0.5}>
          <Tooltip title="Zoom In">
            <IconButton size="small" onClick={handleZoomIn}>
              <ZoomIn fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton size="small" onClick={handleZoomOut}>
              <ZoomOut fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Fit to Screen">
            <IconButton size="small" onClick={handleFitToScreen}>
              <FitScreen fontSize="small" />
            </IconButton>
          </Tooltip>
          <Typography variant="caption" sx={{ alignSelf: "center", minWidth: 50, textAlign: "center" }}>
            {Math.round(zoom * 100)}%
          </Typography>
          <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
          <Tooltip title="Settings">
            <IconButton size="small" onClick={() => setShowSettings(!showSettings)}>
              <Settings fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Open in External Vizarr">
            <IconButton size="small" onClick={handleOpenExternal}>
              <OpenInNew fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh">
            <IconButton size="small" onClick={handleRefresh}>
              <Refresh fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Close">
            <IconButton size="small" onClick={handleClose}>
              <Close fontSize="small" />
            </IconButton>
          </Tooltip>
        </Stack>
      </Paper>
      
      {/* Settings Panel */}
      <Collapse in={showSettings}>
        <Paper elevation={0} sx={{ p: 2, borderBottom: 1, borderColor: "divider" }}>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
            <Typography variant="subtitle2">
              Channel Settings
            </Typography>
            <Button size="small" onClick={autoContrastAll}>
              Auto Contrast All
            </Button>
          </Stack>
          
          <Stack spacing={2}>
            {channelSettings.map((settings, idx) => (
              <Box key={idx}>
                <Stack direction="row" alignItems="center" spacing={1}>
                  <IconButton 
                    size="small" 
                    onClick={() => toggleChannelVisibility(idx)}
                  >
                    {settings.visible ? <Visibility fontSize="small" /> : <VisibilityOff fontSize="small" />}
                  </IconButton>
                  <Box
                    sx={{
                      width: 20,
                      height: 20,
                      borderRadius: 1,
                      bgcolor: `rgb(${settings.color.join(",")})`,
                      border: 1,
                      borderColor: "divider",
                    }}
                  />
                  <Typography variant="body2" sx={{ minWidth: 80 }}>
                    Channel {idx + 1}
                  </Typography>
                  <Button size="small" onClick={() => autoContrast(idx)}>
                    Auto
                  </Button>
                </Stack>
                
                {settings.visible && (
                  <Box sx={{ ml: 5, mt: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      Contrast: {Math.round(settings.contrastMin)} - {Math.round(settings.contrastMax)}
                    </Typography>
                    <Slider
                      size="small"
                      value={[settings.contrastMin, settings.contrastMax]}
                      onChange={(_, value) => {
                        setChannelSettings(prev => {
                          const newSettings = [...prev];
                          newSettings[idx] = { 
                            ...newSettings[idx], 
                            contrastMin: value[0],
                            contrastMax: value[1],
                          };
                          return newSettings;
                        });
                      }}
                      min={0}
                      max={getMaxValueForDtype(dtype)}
                      sx={{ width: 200 }}
                    />
                  </Box>
                )}
              </Box>
            ))}
          </Stack>
          
          {/* Z-Stack Slider */}
          {dimensions.z > 1 && (
            <Box sx={{ mt: 2 }}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Typography variant="body2">Z-Slice:</Typography>
                <IconButton 
                  size="small" 
                  onClick={() => setCurrentZ(prev => Math.max(0, prev - 1))}
                  disabled={currentZ === 0}
                >
                  <NavigateBefore fontSize="small" />
                </IconButton>
                <Typography variant="body2" sx={{ minWidth: 60, textAlign: "center" }}>
                  {currentZ + 1} / {dimensions.z}
                </Typography>
                <IconButton 
                  size="small" 
                  onClick={() => setCurrentZ(prev => Math.min(dimensions.z - 1, prev + 1))}
                  disabled={currentZ === dimensions.z - 1}
                >
                  <NavigateNext fontSize="small" />
                </IconButton>
                <Slider
                  size="small"
                  value={currentZ}
                  onChange={(_, value) => setCurrentZ(value)}
                  min={0}
                  max={dimensions.z - 1}
                  step={1}
                  sx={{ width: 200, ml: 2 }}
                />
              </Stack>
            </Box>
          )}
          
          {/* Time Slider */}
          {dimensions.t > 1 && (
            <Box sx={{ mt: 2 }}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Typography variant="body2">Time:</Typography>
                <IconButton 
                  size="small" 
                  onClick={() => setCurrentT(prev => Math.max(0, prev - 1))}
                  disabled={currentT === 0}
                >
                  <NavigateBefore fontSize="small" />
                </IconButton>
                <Typography variant="body2" sx={{ minWidth: 60, textAlign: "center" }}>
                  {currentT + 1} / {dimensions.t}
                </Typography>
                <IconButton 
                  size="small" 
                  onClick={() => setCurrentT(prev => Math.min(dimensions.t - 1, prev + 1))}
                  disabled={currentT === dimensions.t - 1}
                >
                  <NavigateNext fontSize="small" />
                </IconButton>
                <Slider
                  size="small"
                  value={currentT}
                  onChange={(_, value) => setCurrentT(value)}
                  min={0}
                  max={dimensions.t - 1}
                  step={1}
                  sx={{ width: 200, ml: 2 }}
                />
              </Stack>
            </Box>
          )}
        </Paper>
      </Collapse>
      
      {/* Main Canvas Area */}
      <Box
        ref={containerRef}
        sx={{
          flex: 1,
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "#1a1a1a",
          cursor: isDragging ? "grabbing" : "grab",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <canvas
          ref={canvasRef}
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            imageRendering: zoom > 1 ? "pixelated" : "auto",
            maxWidth: "none",
            maxHeight: "none",
          }}
        />
      </Box>
      
      {/* Status Bar */}
      <Paper
        elevation={1}
        sx={{
          p: 0.5,
          px: 2,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderRadius: 0,
          bgcolor: "background.paper",
        }}
      >
        <Typography variant="caption" color="text.secondary">
          {dtype.toUpperCase()} • {dimensions.channels} channel{dimensions.channels !== 1 ? "s" : ""}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ wordBreak: "break-all", maxWidth: "60%" }}>
          {zarrUrl}
        </Typography>
      </Paper>
    </Box>
  );
};

export default VizarrViewer;
