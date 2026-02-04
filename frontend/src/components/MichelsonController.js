// src/components/MichelsonController.js
// Michelson Interferometer Time-Series Controller Component
// Provides camera-based ROI intensity time-series capture and visualization

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useTheme } from "@mui/material/styles";
import {
  Box,
  Button,
  Card,
  CardContent,
  Slider,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Chip,
  Stack,
  Paper,
  IconButton,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
} from "@mui/icons-material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";

// Redux slice
import * as michelsonSlice from "../state/slices/MichelsonSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";

// Components
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";

// API imports
import {
  apiMichelsonControllerGetParams,
  apiMichelsonControllerGetState,
  apiMichelsonControllerStartCapture,
  apiMichelsonControllerStopCapture,
  apiMichelsonControllerGetTimeseries,
  apiMichelsonControllerExportCsv,
  apiMichelsonControllerGetStatistics,
  apiMichelsonControllerClearBuffer,
  apiMichelsonControllerSetRoi,
  apiMichelsonControllerSetUpdateFreq,
  apiMichelsonControllerSetBufferDuration,
} from "../backendapi/apiMichelsonController";

const MichelsonController = () => {
  const dispatch = useDispatch();
  const theme = useTheme();

  // Redux state
  const michelsonState = useSelector(michelsonSlice.getMichelsonState);
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );

  // Local state for ROI selection
  const [roiSelection, setRoiSelection] = useState({
    centerX: 0,
    centerY: 0,
    size: 10,
  });

  // Local state for update settings
  const [updateFreq, setUpdateFreq] = useState(30);
  const [bufferDuration, setBufferDuration] = useState(60);

  // State for image dimensions
  const [imageSize, setImageSize] = useState({ width: 1920, height: 1080 });

  // Plot window duration (seconds shown in chart)
  const [plotWindowSeconds, setPlotWindowSeconds] = useState(10);

  // Polling interval ref
  const pollingRef = useRef(null);

  // Load initial parameters and state on mount
  useEffect(() => {
    loadParameters();
    loadState();

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  // Start/stop polling based on capture state
  useEffect(() => {
    if (michelsonState.isCapturing) {
      // Poll time-series data at a reasonable rate
      pollingRef.current = setInterval(async () => {
        await fetchTimeseries();
        await loadState();
      }, 100); // 10 Hz polling
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [michelsonState.isCapturing]);

  // Load parameters from backend
  const loadParameters = useCallback(async () => {
    try {
      const params = await apiMichelsonControllerGetParams();

      dispatch(michelsonSlice.setRoiCenter(params.roi_center || [0, 0]));
      dispatch(michelsonSlice.setRoiSize(params.roi_size || 10));
      dispatch(michelsonSlice.setUpdateFreq(params.update_freq || 30));
      dispatch(michelsonSlice.setBufferDuration(params.buffer_duration || 60));

      setRoiSelection({
        centerX: params.roi_center ? params.roi_center[0] : 0,
        centerY: params.roi_center ? params.roi_center[1] : 0,
        size: params.roi_size || 10,
      });
      setUpdateFreq(params.update_freq || 30);
      setBufferDuration(params.buffer_duration || 60);
    } catch (error) {
      console.error("Failed to load Michelson parameters:", error);
    }
  }, [dispatch]);

  // Load capture state from backend
  const loadState = useCallback(async () => {
    try {
      const state = await apiMichelsonControllerGetState();

      dispatch(michelsonSlice.setIsCapturing(state.is_capturing || false));
      dispatch(michelsonSlice.setSampleCount(state.sample_count || 0));
      dispatch(michelsonSlice.setActualFps(state.actual_fps || 0));
    } catch (error) {
      console.error("Failed to load Michelson state:", error);
    }
  }, [dispatch]);

  // Fetch time-series data
  const fetchTimeseries = useCallback(async () => {
    try {
      // Request last N samples based on plot window and update freq
      const maxSamples = Math.ceil(plotWindowSeconds * updateFreq * 1.5);
      const data = await apiMichelsonControllerGetTimeseries(maxSamples);

      dispatch(michelsonSlice.setTimestamps(data.timestamps || []));
      dispatch(michelsonSlice.setMeans(data.means || []));
      dispatch(michelsonSlice.setStds(data.stds || []));
    } catch (error) {
      console.error("Failed to fetch timeseries:", error);
    }
  }, [dispatch, plotWindowSeconds, updateFreq]);

  // Fetch statistics
  const fetchStatistics = useCallback(async () => {
    try {
      const stats = await apiMichelsonControllerGetStatistics();
      dispatch(michelsonSlice.setStatistics(stats));
    } catch (error) {
      console.error("Failed to fetch statistics:", error);
    }
  }, [dispatch]);

  // Start capture
  const handleStartCapture = useCallback(async () => {
    try {
      await apiMichelsonControllerStartCapture();
      dispatch(michelsonSlice.setIsCapturing(true));
      await loadState();
    } catch (error) {
      console.error("Failed to start capture:", error);
    }
  }, [dispatch, loadState]);

  // Stop capture
  const handleStopCapture = useCallback(async () => {
    try {
      await apiMichelsonControllerStopCapture();
      dispatch(michelsonSlice.setIsCapturing(false));
      await loadState();
      await fetchStatistics();
    } catch (error) {
      console.error("Failed to stop capture:", error);
    }
  }, [dispatch, loadState, fetchStatistics]);

  // Clear buffer
  const handleClearBuffer = useCallback(async () => {
    try {
      await apiMichelsonControllerClearBuffer();
      dispatch(michelsonSlice.setTimestamps([]));
      dispatch(michelsonSlice.setMeans([]));
      dispatch(michelsonSlice.setStds([]));
      dispatch(michelsonSlice.setStatistics(null));
      dispatch(michelsonSlice.setSampleCount(0));
    } catch (error) {
      console.error("Failed to clear buffer:", error);
    }
  }, [dispatch]);

  // Export CSV
  const handleExportCsv = useCallback(async () => {
    try {
      const blob = await apiMichelsonControllerExportCsv();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `michelson_timeseries_${Date.now()}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Failed to export CSV:", error);
    }
  }, []);

  // Update ROI
  const handleApplyRoi = useCallback(async () => {
    try {
      await apiMichelsonControllerSetRoi(
        roiSelection.centerX,
        roiSelection.centerY,
        roiSelection.size
      );
      dispatch(
        michelsonSlice.setRoiCenter([roiSelection.centerX, roiSelection.centerY])
      );
      dispatch(michelsonSlice.setRoiSize(roiSelection.size));
    } catch (error) {
      console.error("Failed to apply ROI:", error);
    }
  }, [dispatch, roiSelection]);

  // Update update frequency
  const handleApplyUpdateFreq = useCallback(async () => {
    try {
      await apiMichelsonControllerSetUpdateFreq(updateFreq);
      dispatch(michelsonSlice.setUpdateFreq(updateFreq));
    } catch (error) {
      console.error("Failed to apply update frequency:", error);
    }
  }, [dispatch, updateFreq]);

  // Update buffer duration
  const handleApplyBufferDuration = useCallback(async () => {
    try {
      await apiMichelsonControllerSetBufferDuration(bufferDuration);
      dispatch(michelsonSlice.setBufferDuration(bufferDuration));
    } catch (error) {
      console.error("Failed to apply buffer duration:", error);
    }
  }, [dispatch, bufferDuration]);

  // Handle image load from viewer
  const handleImageLoad = useCallback((width, height) => {
    setImageSize({ width, height });
  }, []);

  // Handle click on camera stream for ROI selection
  const handleLiveViewClick = useCallback(
    (pixelX, pixelY, imgWidth, imgHeight) => {
      // Convert to absolute pixel coordinates (not relative to center)
      const newCenterX = Math.round(pixelX);
      const newCenterY = Math.round(pixelY);

      setRoiSelection((prev) => ({
        ...prev,
        centerX: newCenterX,
        centerY: newCenterY,
      }));

      // Auto-apply ROI
      (async () => {
        try {
          await apiMichelsonControllerSetRoi(newCenterX, newCenterY, roiSelection.size);
        } catch (error) {
          console.error("Failed to auto-apply ROI:", error);
        }
      })();
    },
    [roiSelection.size]
  );

  // ROI overlay for camera stream
  const roiOverlay = useMemo(() => {
    if (!imageSize.width || !imageSize.height) return null;

    const centerXRel = roiSelection.centerX / imageSize.width;
    const centerYRel = roiSelection.centerY / imageSize.height;
    const sizeXRel = roiSelection.size / imageSize.width;
    const sizeYRel = roiSelection.size / imageSize.height;

    return (
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
      >
        {/* ROI rectangle */}
        <rect
          x={(centerXRel - sizeXRel / 2) * 100}
          y={(centerYRel - sizeYRel / 2) * 100}
          width={sizeXRel * 100}
          height={sizeYRel * 100}
          fill="rgba(0, 255, 0, 0.2)"
          stroke="lime"
          strokeWidth="0.5"
          opacity="0.9"
        />
        {/* Center crosshair */}
        <line
          x1={(centerXRel - 0.02) * 100}
          y1={centerYRel * 100}
          x2={(centerXRel + 0.02) * 100}
          y2={centerYRel * 100}
          stroke="lime"
          strokeWidth="0.3"
        />
        <line
          x1={centerXRel * 100}
          y1={(centerYRel - 0.02) * 100}
          x2={centerXRel * 100}
          y2={(centerYRel + 0.02) * 100}
          stroke="lime"
          strokeWidth="0.3"
        />
      </svg>
    );
  }, [imageSize, roiSelection]);

  // Prepare chart data - use relative time for x-axis
  const chartData = useMemo(() => {
    const { timestamps, means, stds } = michelsonState;
    if (!timestamps || timestamps.length === 0) return [];

    // Get the latest timestamp as reference
    const latestTime = timestamps[timestamps.length - 1];
    const windowStart = latestTime - plotWindowSeconds;

    // Filter to plot window and create data points
    return timestamps
      .map((t, i) => ({
        time: t - latestTime, // Relative time (negative values, 0 = now)
        absTime: t,
        mean: means[i],
        std: stds[i],
        upper: means[i] + stds[i],
        lower: means[i] - stds[i],
      }))
      .filter((d) => d.absTime >= windowStart);
  }, [michelsonState.timestamps, michelsonState.means, michelsonState.stds, plotWindowSeconds]);

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 }, maxWidth: "100%" }}>
      <Typography variant="h5" gutterBottom>
        Michelson Interferometer Time-Series
      </Typography>

      {/* Control Buttons */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
          <Button
            variant="contained"
            color="primary"
            startIcon={<PlayArrowIcon />}
            onClick={handleStartCapture}
            disabled={michelsonState.isCapturing}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Start Capture
          </Button>

          <Button
            variant="contained"
            color="error"
            startIcon={<StopIcon />}
            onClick={handleStopCapture}
            disabled={!michelsonState.isCapturing}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Stop Capture
          </Button>

          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadParameters}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Refresh
          </Button>

          <Button
            variant="outlined"
            color="warning"
            startIcon={<DeleteIcon />}
            onClick={handleClearBuffer}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Clear Buffer
          </Button>

          <Button
            variant="outlined"
            color="success"
            startIcon={<DownloadIcon />}
            onClick={handleExportCsv}
            disabled={michelsonState.sampleCount === 0}
            sx={{ minWidth: { xs: "100%", sm: "auto" } }}
          >
            Export CSV
          </Button>
        </Stack>

        {/* Status Chips */}
        <Stack direction="row" spacing={1} mt={2} flexWrap="wrap" useFlexGap>
          <Chip
            label={michelsonState.isCapturing ? "Capturing" : "Stopped"}
            color={michelsonState.isCapturing ? "success" : "default"}
            size="small"
          />
          <Chip
            label={`Samples: ${michelsonState.sampleCount}`}
            variant="outlined"
            size="small"
          />
          <Chip
            label={`FPS: ${michelsonState.actualFps?.toFixed(1) || 0}`}
            variant="outlined"
            size="small"
          />
          <Chip
            label={`ROI: ${roiSelection.size}×${roiSelection.size}px`}
            variant="outlined"
            size="small"
          />
        </Stack>
      </Paper>

      <Grid container spacing={2}>
        {/* Camera Stream with ROI */}
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Camera Stream (Click to set ROI)
              </Typography>
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  paddingTop: "75%",
                  backgroundColor: "#000",
                  borderRadius: 1,
                  overflow: "hidden",
                }}
              >
                <LiveViewControlWrapper
                  onClick={handleLiveViewClick}
                  onImageLoad={handleImageLoad}
                  overlayContent={roiOverlay}
                  enableStageMovement={false}
                />
              </Box>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 1, display: "block" }}
              >
                ROI Center: ({roiSelection.centerX}, {roiSelection.centerY}) |
                Size: {roiSelection.size}×{roiSelection.size}px
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Time-Series Plot */}
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Box
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                mb={1}
              >
                <Typography variant="h6">Intensity Time-Series</Typography>
                <Stack direction="row" spacing={1}>
                  <ToggleButtonGroup
                    value={plotWindowSeconds}
                    exclusive
                    onChange={(e, v) => v && setPlotWindowSeconds(v)}
                    size="small"
                  >
                    <ToggleButton value={5}>5s</ToggleButton>
                    <ToggleButton value={10}>10s</ToggleButton>
                    <ToggleButton value={30}>30s</ToggleButton>
                    <ToggleButton value={60}>60s</ToggleButton>
                  </ToggleButtonGroup>
                </Stack>
              </Box>
              <Box sx={{ width: "100%", height: 300 }}>
                <ResponsiveContainer>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="time"
                      tickFormatter={(v) => `${v.toFixed(1)}s`}
                      domain={[-plotWindowSeconds, 0]}
                      label={{
                        value: "Time (s)",
                        position: "insideBottom",
                        offset: -5,
                      }}
                    />
                    <YAxis
                      label={{
                        value: "Intensity",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <RechartsTooltip
                      formatter={(value, name) => [
                        typeof value === "number" ? value.toFixed(2) : value,
                        name,
                      ]}
                      labelFormatter={(label) => `t = ${label.toFixed(3)}s`}
                    />
                    <Legend />
                    {/* Standard deviation bands */}
                    <Line
                      type="monotone"
                      dataKey="upper"
                      stroke={theme.palette.info.light}
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      dot={false}
                      name="+σ"
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="lower"
                      stroke={theme.palette.info.light}
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      dot={false}
                      name="-σ"
                      isAnimationActive={false}
                    />
                    {/* Mean intensity line */}
                    <Line
                      type="monotone"
                      dataKey="mean"
                      stroke={theme.palette.primary.main}
                      strokeWidth={2}
                      dot={false}
                      name="Mean"
                      isAnimationActive={false}
                    />
                    {/* Reference line at current mean */}
                    {michelsonState.statistics?.mean_of_means && (
                      <ReferenceLine
                        y={michelsonState.statistics.mean_of_means}
                        stroke={theme.palette.warning.main}
                        strokeDasharray="5 5"
                        label="Avg"
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Statistics Display */}
      {michelsonState.statistics && (
        <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            Statistics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary">
                Mean Intensity
              </Typography>
              <Typography variant="h6">
                {michelsonState.statistics.mean_of_means?.toFixed(2) || "-"}
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary">
                Std of Mean
              </Typography>
              <Typography variant="h6">
                {michelsonState.statistics.std_of_means?.toFixed(2) || "-"}
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary">
                Min Intensity
              </Typography>
              <Typography variant="h6">
                {michelsonState.statistics.min_mean?.toFixed(2) || "-"}
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary">
                Max Intensity
              </Typography>
              <Typography variant="h6">
                {michelsonState.statistics.max_mean?.toFixed(2) || "-"}
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* ROI Settings */}
      <Accordion sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">ROI Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={4} sm={2}>
              <TextField
                label="Center X"
                type="number"
                value={roiSelection.centerX}
                onChange={(e) =>
                  setRoiSelection((prev) => ({
                    ...prev,
                    centerX: parseInt(e.target.value) || 0,
                  }))
                }
                size="small"
                fullWidth
              />
            </Grid>
            <Grid item xs={4} sm={2}>
              <TextField
                label="Center Y"
                type="number"
                value={roiSelection.centerY}
                onChange={(e) =>
                  setRoiSelection((prev) => ({
                    ...prev,
                    centerY: parseInt(e.target.value) || 0,
                  }))
                }
                size="small"
                fullWidth
              />
            </Grid>
            <Grid item xs={4} sm={2}>
              <FormControl fullWidth size="small">
                <InputLabel>ROI Size</InputLabel>
                <Select
                  value={roiSelection.size}
                  onChange={(e) =>
                    setRoiSelection((prev) => ({
                      ...prev,
                      size: e.target.value,
                    }))
                  }
                  label="ROI Size"
                >
                  <MenuItem value={5}>5×5</MenuItem>
                  <MenuItem value={10}>10×10</MenuItem>
                  <MenuItem value={20}>20×20</MenuItem>
                  <MenuItem value={50}>50×50</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Button variant="contained" onClick={handleApplyRoi} fullWidth>
                Apply ROI
              </Button>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Capture Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Capture Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={6} sm={3}>
              <TextField
                label="Update Freq (Hz)"
                type="number"
                value={updateFreq}
                onChange={(e) => setUpdateFreq(parseFloat(e.target.value) || 30)}
                size="small"
                fullWidth
                inputProps={{ min: 1, max: 100, step: 1 }}
              />
            </Grid>
            <Grid item xs={6} sm={3}>
              <Button
                variant="outlined"
                onClick={handleApplyUpdateFreq}
                fullWidth
              >
                Apply Freq
              </Button>
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField
                label="Buffer (sec)"
                type="number"
                value={bufferDuration}
                onChange={(e) =>
                  setBufferDuration(parseFloat(e.target.value) || 60)
                }
                size="small"
                fullWidth
                inputProps={{ min: 10, max: 600, step: 10 }}
              />
            </Grid>
            <Grid item xs={6} sm={3}>
              <Button
                variant="outlined"
                onClick={handleApplyBufferDuration}
                fullWidth
              >
                Apply Buffer
              </Button>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default MichelsonController;
