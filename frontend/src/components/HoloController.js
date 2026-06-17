// src/components/HoloController.js
// Inline Hologram Processing Controller Component
// Provides live camera view, processed hologram stream, and parameter controls

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
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Chip,
  Stack,
  Paper,
  IconButton,
  Tooltip,
  Alert,
  Divider,
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  CenterFocusStrong as CenterFocusStrongIcon,
  InfoOutlined as InfoOutlinedIcon,
  RestartAlt as RestartAltIcon,
} from "@mui/icons-material";

// Redux slice
import * as holoSlice from "../state/slices/HoloSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice";

// Components - use LiveViewControlWrapper for automatic format selection (JPEG/Binary/WebRTC)
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";

// API imports
import apiInLineHoloControllerGetParams from "../backendapi/apiInLineHoloControllerGetParams";
import apiInLineHoloControllerSetParams from "../backendapi/apiInLineHoloControllerSetParams";
import apiInLineHoloControllerGetState from "../backendapi/apiInLineHoloControllerGetState";
import apiInLineHoloControllerStartProcessing from "../backendapi/apiInLineHoloControllerStartProcessing";
import apiInLineHoloControllerStopProcessing from "../backendapi/apiInLineHoloControllerStopProcessing";
import apiInLineHoloControllerPauseProcessing from "../backendapi/apiInLineHoloControllerPauseProcessing";
import apiInLineHoloControllerResumeProcessing from "../backendapi/apiInLineHoloControllerResumeProcessing";
import apiInLineHoloControllerSetRoi from "../backendapi/apiInLineHoloControllerSetRoi";
import apiInLineHoloControllerSetDz from "../backendapi/apiInLineHoloControllerSetDz";
import apiInLineHoloControllerSetBinning from "../backendapi/apiInLineHoloControllerSetBinning";
import apiInLineHoloControllerGetCameraInfo from "../backendapi/apiInLineHoloControllerGetCameraInfo";
import apiInLineHoloControllerRestartStream from "../backendapi/apiInLineHoloControllerRestartStream";
import apiLiveViewControllerGetStreamParameters from "../backendapi/apiLiveViewControllerGetStreamParameters";

// How long without a server-side MJPEG emit before we declare the processed
// stream "stalled" and surface a warning + restart button to the user.
const PROCESSED_STREAM_STALL_MS = 5000;
const AUTO_ONCE_RESET_DELAY_MS = 1500;
const AUTO_ONCE_UI_HOLD_MS = AUTO_ONCE_RESET_DELAY_MS + 300;

// A free-typing numeric TextField: stores the literal string the user types
// (incl. empty / partially-typed values like "" or "0.") so the input never
// snaps back to a default mid-edit. Commits a parsed number on blur or Enter.
const FreeNumberField = ({
  label,
  value,
  onCommit,
  unitFactor = 1, // value-in-state = displayed-value * unitFactor
  fixedDecimals = null,
  helperText,
  tooltip,
  step,
  min,
  max,
  fullWidth = true,
  ...textFieldProps
}) => {
  const formatDisplay = useCallback(
    (v) => {
      if (v === null || v === undefined || Number.isNaN(v)) return "";
      const display = v / unitFactor;
      if (fixedDecimals !== null) {
        return Number(display).toFixed(fixedDecimals);
      }
      return String(display);
    },
    [unitFactor, fixedDecimals]
  );

  const [draft, setDraft] = useState(() => formatDisplay(value));
  const editingRef = useRef(false);

  // Sync from outside (e.g. backend pushed new value) ONLY while the user
  // is not editing — prevents the field from jumping while typing.
  useEffect(() => {
    if (!editingRef.current) {
      setDraft(formatDisplay(value));
    }
  }, [value, formatDisplay]);

  const handleChange = (e) => {
    editingRef.current = true;
    // Accept any string; don't reject empty / "-" / "." mid-edit.
    setDraft(e.target.value);
  };

  const commit = () => {
    editingRef.current = false;
    const trimmed = draft.trim();
    if (trimmed === "" || trimmed === "-" || trimmed === ".") {
      // Nothing meaningful entered → restore last known value.
      setDraft(formatDisplay(value));
      return;
    }
    const parsed = Number(trimmed);
    if (Number.isNaN(parsed)) {
      setDraft(formatDisplay(value));
      return;
    }
    let next = parsed * unitFactor;
    if (min !== undefined && next < min) next = min;
    if (max !== undefined && next > max) next = max;
    setDraft(formatDisplay(next));
    if (next !== value) onCommit(next);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.currentTarget.blur();
    }
  };

  const field = (
    <TextField
      label={label}
      type="text"
      inputProps={{ inputMode: "decimal", step, min, max }}
      value={draft}
      onFocus={() => {
        editingRef.current = true;
      }}
      onChange={handleChange}
      onBlur={commit}
      onKeyDown={handleKeyDown}
      helperText={helperText}
      fullWidth={fullWidth}
      size="small"
      {...textFieldProps}
    />
  );

  if (tooltip) {
    return (
      <Tooltip title={tooltip} arrow placement="top-start">
        <Box>{field}</Box>
      </Tooltip>
    );
  }
  return field;
};

const HoloController = () => {
  const dispatch = useDispatch();
  const theme = useTheme();

  // Redux state
  const holoState = useSelector(holoSlice.getHoloState);
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  // Local state for ROI selection (size is in BACKEND/sensor pixels)
  const [roiSelection, setRoiSelection] = useState({
    isSelecting: false,
    centerX: 0,
    centerY: 0,
    size: 256,
  });

  // State for image dimensions from the stream viewer
  const [imageSize, setImageSize] = useState({ width: 1920, height: 1080 });
  const [displayInfo, setDisplayInfo] = useState(null);

  // Focus sweep: step dz from start→end across N steps, one step per second.
  const [sweep, setSweep] = useState({
    startUm: 0,    // start dz in micrometers
    endUm: 1000,   // end dz in micrometers
    steps: 10,     // number of steps (must be < 20)
    running: false,
  });

  // Detector exposure/gain/AWB local UI state
  const [detectorParams, setDetectorParams] = useState({
    exposure: "",
    gain: "",
    mode: "manual",
    awb_mode: null,
    red_gain: null,
    blue_gain: null,
    isRGB: false,
  });
  const [autoOncePending, setAutoOncePending] = useState(false);
  const [awbOncePending, setAwbOncePending] = useState(false);

  // Stream stall detection: bump when we mount the <img> to force a new
  // browser connection on user-initiated restarts.
  const [streamNonce, setStreamNonce] = useState(0);
  const [streamStalled, setStreamStalled] = useState(false);

  // Resolve the subsampling factor of the *currently active* stream protocol.
  // Reading "jpeg" unconditionally returned the wrong factor (e.g. 1) while the
  // real stream was binary at factor 4, so the red ROI overlay box was wrong
  // until the user opened stream settings.
  const getActiveSubsamplingFactor = useCallback(() => {
    const ss = liveStreamState.streamSettings || {};
    const fmt = liveStreamState.imageFormat || ss.current_compression_algorithm || "jpeg";
    if (fmt === "binary") {
      return ss.binary?.subsampling?.factor || ss.binary?.subsampling_factor || 1;
    }
    if (fmt === "webrtc") {
      return ss.webrtc?.subsampling_factor || ss.webrtc?.subsampling?.factor || 1;
    }
    return ss.jpeg?.subsampling?.factor || ss.jpeg?.subsampling_factor || 1;
  }, [liveStreamState.streamSettings, liveStreamState.imageFormat]);

  // Total scaling: subsampling of the active stream × software binning
  const totalScalingFactor = useMemo(() => {
    return getActiveSubsamplingFactor() * (holoState.binning || 1);
  }, [getActiveSubsamplingFactor, holoState.binning]);

  // ROI size in preview pixels (for overlay display)
  const roiSizeInPreview = useMemo(() => {
    return roiSelection.size / totalScalingFactor;
  }, [roiSelection.size, totalScalingFactor]);

  // Ref for processed stream image
  const processedImageRef = useRef(null);

  // Refs for the focus-sweep timer and current step index
  const sweepTimerRef = useRef(null);
  const sweepIndexRef = useRef(0);

  // Build stream URLs - these need full URLs for video streaming
  const baseUrl = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api`;
  const processedStreamUrl = useMemo(
    () =>
      `${baseUrl}/InLineHoloController/mjpeg_stream_inlineholo?startStream=true&jpeg_quality=85&t=${streamNonce}`,
    [baseUrl, streamNonce]
  );

  // ---------------- Backend sync helpers ----------------

  const loadParameters = useCallback(async () => {
    try {
      const params = await apiInLineHoloControllerGetParams();

      dispatch(holoSlice.setPixelsize(params.pixelsize || 3.45e-6));
      dispatch(holoSlice.setWavelength(params.wavelength || 488e-9));
      dispatch(holoSlice.setNa(params.na || 0.3));
      dispatch(holoSlice.setDz(params.dz || 0.0));
      if (params.dz_max !== undefined)
        dispatch(holoSlice.setDzMax(params.dz_max));
      if (params.dz_step !== undefined)
        dispatch(holoSlice.setDzStep(params.dz_step));
      dispatch(holoSlice.setBinning(params.binning || 1));
      if (params.full_frame !== undefined)
        dispatch(holoSlice.setFullFrame(!!params.full_frame));
      dispatch(holoSlice.setRoiCenter(params.roi_center || [0, 0]));
      dispatch(holoSlice.setRoiSize(params.roi_size || 256));
      dispatch(holoSlice.setColorChannel(params.color_channel || "red"));
      dispatch(holoSlice.setFlipX(params.flip_x || false));
      dispatch(holoSlice.setFlipY(params.flip_y || false));
      dispatch(holoSlice.setRotation(params.rotation || 0));
      dispatch(holoSlice.setUpdateFreq(params.update_freq || 10.0));
      dispatch(holoSlice.setShowRaw(params.show_raw || false));

      setRoiSelection((prev) => ({
        ...prev,
        centerX: params.roi_center ? params.roi_center[0] : 0,
        centerY: params.roi_center ? params.roi_center[1] : 0,
        size: params.roi_size || 256,
      }));
    } catch (error) {
      console.error("Failed to load hologram parameters:", error);
    }
  }, [dispatch]);

  const loadState = useCallback(async () => {
    try {
      const state = await apiInLineHoloControllerGetState();
      dispatch(holoSlice.setIsProcessing(state.is_processing || false));
      dispatch(holoSlice.setIsPaused(state.is_paused || false));
      dispatch(holoSlice.setIsStreaming(state.is_streaming || false));
      dispatch(holoSlice.setLastProcessTime(state.last_process_time || 0.0));
      dispatch(holoSlice.setFrameCount(state.frame_count || 0));
      dispatch(holoSlice.setProcessedCount(state.processed_count || 0));
      if (state.last_mjpeg_emit_time !== undefined)
        dispatch(
          holoSlice.setLastMjpegEmitTime(state.last_mjpeg_emit_time || 0.0)
        );
      if (state.mjpeg_client_count !== undefined)
        dispatch(
          holoSlice.setMjpegClientCount(state.mjpeg_client_count || 0)
        );
    } catch (error) {
      console.error("Failed to load hologram state:", error);
    }
  }, [dispatch]);

  // Fetch the active stream parameters so the ROI overlay scaling is correct
  // on first open, before the user opens the stream-settings panel.
  const loadStreamParameters = useCallback(async () => {
    try {
      const response = await apiLiveViewControllerGetStreamParameters();
      const allParams = response.protocols || response;
      const currentProtocol =
        response.current_protocol || liveStreamState.imageFormat || "jpeg";
      const ss = liveStreamState.streamSettings || {};
      dispatch(
        liveStreamSlice.setStreamSettings({
          current_compression_algorithm: currentProtocol,
          binary: {
            ...ss.binary,
            subsampling: {
              factor:
                allParams.binary?.subsampling_factor ??
                ss.binary?.subsampling?.factor ??
                4,
            },
          },
          jpeg: {
            ...ss.jpeg,
            subsampling: {
              factor:
                allParams.jpeg?.subsampling_factor ??
                ss.jpeg?.subsampling?.factor ??
                1,
            },
          },
          webrtc: {
            ...ss.webrtc,
            subsampling_factor:
              allParams.webrtc?.subsampling_factor ??
              ss.webrtc?.subsampling_factor ??
              1,
          },
        })
      );
    } catch (error) {
      console.warn("Failed to load stream parameters for holo overlay:", error);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dispatch]);

  const loadCameraInfo = useCallback(async () => {
    try {
      const info = await apiInLineHoloControllerGetCameraInfo();
      if (info?.camera) {
        dispatch(holoSlice.setCameraName(info.camera));
        dispatch(holoSlice.setIsRGB(!!info.is_rgb));
        setDetectorParams((prev) => ({ ...prev, isRGB: !!info.is_rgb }));
      }
    } catch (error) {
      console.error("Failed to load camera info:", error);
    }
  }, [dispatch]);

  // Pull detector parameters (exposure/gain/mode/AWB) via SettingsController,
  // scoped to the detector that backs the hologram controller.
  const loadDetectorParameters = useCallback(async () => {
    const cam = holoState.cameraName;
    try {
      // Generic params (works without a detectorName: uses the current detector)
      const resp = await fetch(
        `${baseUrl}/SettingsController/getDetectorParameters`
      );
      if (!resp.ok) return;
      const data = await resp.json();
      setDetectorParams((prev) => ({
        ...prev,
        exposure: data.exposure ?? prev.exposure,
        gain: data.gain ?? prev.gain,
        mode: (data.mode ?? prev.mode ?? "manual").toString().toLowerCase(),
        awb_mode: data.awb_mode ?? prev.awb_mode,
        red_gain: data.red_gain ?? prev.red_gain,
        blue_gain: data.blue_gain ?? prev.blue_gain,
        isRGB: data.isRGB ? true : prev.isRGB,
      }));
      if (cam && data.isRGB) {
        try {
          const wbResp = await fetch(
            `${baseUrl}/SettingsController/getWhiteBalance?detectorName=${encodeURIComponent(cam)}`
          );
          if (wbResp.ok) {
            const wb = await wbResp.json();
            if (wb?.awb_supported) {
              setDetectorParams((prev) => ({
                ...prev,
                awb_mode: wb.awb_mode ?? prev.awb_mode,
                red_gain: wb.red_gain ?? prev.red_gain,
                blue_gain: wb.blue_gain ?? prev.blue_gain,
              }));
            }
          }
        } catch (err) {
          // Non-fatal: AWB section just won't show fresh values
          console.debug("getWhiteBalance failed:", err);
        }
      }
    } catch (error) {
      console.error("Failed to load detector parameters:", error);
    }
  }, [baseUrl, holoState.cameraName]);

  // ---------------- Mount / polling ----------------

  useEffect(() => {
    loadParameters();
    loadState();
    loadStreamParameters();
    loadCameraInfo();
    dispatch(holoSlice.setProcessedStreamUrl(processedStreamUrl));
    const stateInterval = setInterval(loadState, 3000);
    return () => {
      clearInterval(stateInterval);
      if (sweepTimerRef.current) {
        clearInterval(sweepTimerRef.current);
        sweepTimerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!holoState.cameraName) return;
    loadDetectorParameters();
    const detectorInterval = setInterval(loadDetectorParameters, 4000);
    return () => clearInterval(detectorInterval);
  }, [holoState.cameraName, loadDetectorParameters]);

  // Stall detection: server tracks last_mjpeg_emit_time. If processing is on
  // but no emit has happened for PROCESSED_STREAM_STALL_MS, surface a warning.
  useEffect(() => {
    if (!holoState.isProcessing) {
      setStreamStalled(false);
      return;
    }
    const now = Date.now() / 1000;
    const last = holoState.lastMjpegEmitTime || 0;
    if (last === 0) {
      // No frame yet — give it one polling cycle before flagging
      setStreamStalled(false);
      return;
    }
    setStreamStalled(now - last > PROCESSED_STREAM_STALL_MS / 1000);
  }, [holoState.lastMjpegEmitTime, holoState.isProcessing]);

  // ---------------- Processing control ----------------

  const handleStartProcessing = useCallback(async () => {
    try {
      await apiInLineHoloControllerStartProcessing();
      dispatch(holoSlice.setIsProcessing(true));
      dispatch(holoSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to start processing:", error);
    }
  }, [dispatch, loadState]);

  const handleStopProcessing = useCallback(async () => {
    try {
      await apiInLineHoloControllerStopProcessing();
      dispatch(holoSlice.setIsProcessing(false));
      dispatch(holoSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to stop processing:", error);
    }
  }, [dispatch, loadState]);

  const handlePauseProcessing = useCallback(async () => {
    try {
      dispatch(holoSlice.setPreviousBinning(holoState.binning));
      await apiInLineHoloControllerSetBinning(1);
      dispatch(holoSlice.setBinning(1));
      await apiInLineHoloControllerPauseProcessing();
      dispatch(holoSlice.setIsPaused(true));
      await loadState();
    } catch (error) {
      console.error("Failed to pause processing:", error);
    }
  }, [dispatch, holoState.binning, loadState]);

  const handleResumeProcessing = useCallback(async () => {
    try {
      const previousBinning = holoState.previousBinning || 1;
      await apiInLineHoloControllerSetBinning(previousBinning);
      dispatch(holoSlice.setBinning(previousBinning));
      await apiInLineHoloControllerResumeProcessing();
      dispatch(holoSlice.setIsPaused(false));
      await loadState();
    } catch (error) {
      console.error("Failed to resume processing:", error);
    }
  }, [dispatch, holoState.previousBinning, loadState]);

  const handleRestartProcessedStream = useCallback(async () => {
    try {
      await apiInLineHoloControllerRestartStream();
    } catch (err) {
      console.warn("restart_stream backend call failed, reloading <img> anyway:", err);
    }
    setStreamNonce((n) => n + 1);
    setStreamStalled(false);
    await loadState();
  }, [loadState]);

  // ---------------- dz slider ----------------

  const handleDzChange = useCallback(
    (event, value) => {
      dispatch(holoSlice.setDz(value));
    },
    [dispatch]
  );

  const handleDzCommit = useCallback(async (event, value) => {
    try {
      await apiInLineHoloControllerSetDz(value);
    } catch (error) {
      console.error("Failed to update dz:", error);
    }
  }, []);

  const commitDzMax = useCallback(
    async (newMax) => {
      try {
        const clamped = Math.max(1e-6, newMax); // ≥1 µm
        dispatch(holoSlice.setDzMax(clamped));
        await apiInLineHoloControllerSetParams({ dz_max: clamped });
        // If current dz is now above the new max, clamp it backend-side too
        if (holoState.dz > clamped) {
          dispatch(holoSlice.setDz(clamped));
          await apiInLineHoloControllerSetDz(clamped);
        }
      } catch (err) {
        console.error("Failed to set dz_max:", err);
      }
    },
    [dispatch, holoState.dz]
  );

  // Toggle between raw (dz=0) and the slider dz in the processed view
  const handleShowRawToggle = useCallback(
    async (event) => {
      const checked = event.target.checked;
      dispatch(holoSlice.setShowRaw(checked));
      try {
        await apiInLineHoloControllerSetParams({ show_raw: checked });
      } catch (error) {
        console.error("Failed to set show_raw:", error);
      }
    },
    [dispatch]
  );

  const commitDzStep = useCallback(
    async (newStep) => {
      try {
        const clamped = Math.max(1e-9, newStep); // ≥1 nm
        dispatch(holoSlice.setDzStep(clamped));
        await apiInLineHoloControllerSetParams({ dz_step: clamped });
      } catch (err) {
        console.error("Failed to set dz_step:", err);
      }
    },
    [dispatch]
  );

  // ----- Focus sweep -----
  const stopFocusSweep = useCallback(() => {
    if (sweepTimerRef.current) {
      clearInterval(sweepTimerRef.current);
      sweepTimerRef.current = null;
    }
    setSweep((prev) => ({ ...prev, running: false }));
  }, []);

  const startFocusSweep = useCallback(() => {
    const nSteps = Math.max(2, Math.min(19, Math.round(sweep.steps) || 2));
    const startUm = Number(sweep.startUm) || 0;
    const endUm = Number(sweep.endUm) || 0;

    if (sweepTimerRef.current) {
      clearInterval(sweepTimerRef.current);
      sweepTimerRef.current = null;
    }
    sweepIndexRef.current = 0;
    setSweep((prev) => ({ ...prev, steps: nSteps, running: true }));

    const applyStep = () => {
      const i = sweepIndexRef.current;
      const frac = nSteps <= 1 ? 0 : i / (nSteps - 1);
      const dzMeters = (startUm + (endUm - startUm) * frac) * 1e-6;
      dispatch(holoSlice.setDz(dzMeters));
      apiInLineHoloControllerSetDz(dzMeters).catch((error) =>
        console.error("Focus sweep: failed to set dz:", error)
      );
      sweepIndexRef.current = (i + 1) % nSteps;
    };

    applyStep();
    sweepTimerRef.current = setInterval(applyStep, 1000);
  }, [dispatch, sweep.startUm, sweep.endUm, sweep.steps]);

  // ---------------- ROI ----------------


  const handleRoiCenterXChange = useCallback((event, value) => {
    setRoiSelection((prev) => ({ ...prev, centerX: value }));
  }, []);
  const handleRoiCenterYChange = useCallback((event, value) => {
    setRoiSelection((prev) => ({ ...prev, centerY: value }));
  }, []);
  const handleRoiSizeChange = useCallback((event, value) => {
    setRoiSelection((prev) => ({ ...prev, size: value }));
  }, []);

  const handleApplyRoi = useCallback(async () => {
    try {
      const streamSubsampling = getActiveSubsamplingFactor();
      const streamedWidth = imageSize.width;
      const streamedHeight = imageSize.height;
      const streamedCenterX = streamedWidth / 2;
      const streamedCenterY = streamedHeight / 2;
      const absoluteXInStream = streamedCenterX + roiSelection.centerX;
      const absoluteYInStream = streamedCenterY + roiSelection.centerY;
      const absoluteXFullSensor = Math.round(absoluteXInStream * streamSubsampling);
      const absoluteYFullSensor = Math.round(absoluteYInStream * streamSubsampling);
      const finalSize = Math.min(roiSelection.size, 1024);

      await apiInLineHoloControllerSetRoi({
        center_x: absoluteXFullSensor,
        center_y: absoluteYFullSensor,
        size: finalSize,
      });
      dispatch(
        holoSlice.setRoiCenter([roiSelection.centerX, roiSelection.centerY])
      );
      dispatch(holoSlice.setRoiSize(roiSelection.size));
    } catch (error) {
      console.error("Failed to apply ROI:", error);
    }
  }, [dispatch, roiSelection, getActiveSubsamplingFactor, imageSize]);

  const handleResetRoi = useCallback(async () => {
    try {
      const streamSubsampling = getActiveSubsamplingFactor();
      const streamedWidth = imageSize.width;
      const streamedHeight = imageSize.height;
      const fullSensorCenterX = Math.round(
        (streamedWidth / 2) * streamSubsampling
      );
      const fullSensorCenterY = Math.round(
        (streamedHeight / 2) * streamSubsampling
      );
      const defaultSize = 256;

      await apiInLineHoloControllerSetRoi({
        center_x: fullSensorCenterX,
        center_y: fullSensorCenterY,
        size: defaultSize,
      });
      dispatch(holoSlice.setRoiCenter([0, 0]));
      dispatch(holoSlice.setRoiSize(defaultSize));
      setRoiSelection((prev) => ({
        ...prev,
        centerX: 0,
        centerY: 0,
        size: defaultSize,
      }));
    } catch (error) {
      console.error("Failed to reset ROI:", error);
    }
  }, [dispatch, getActiveSubsamplingFactor, imageSize]);

  // ---------------- Generic param updates (developer options) ----------------

  const commitParam = useCallback(
    async (paramName, value) => {
      try {
        switch (paramName) {
          case "pixelsize":
            dispatch(holoSlice.setPixelsize(value));
            await apiInLineHoloControllerSetParams({ pixelsize: value });
            break;
          case "wavelength":
            dispatch(holoSlice.setWavelength(value));
            await apiInLineHoloControllerSetParams({ wavelength: value });
            break;
          case "na":
            dispatch(holoSlice.setNa(value));
            await apiInLineHoloControllerSetParams({ na: value });
            break;
          case "update_freq":
            dispatch(holoSlice.setUpdateFreq(value));
            await apiInLineHoloControllerSetParams({ update_freq: value });
            break;
          case "binning":
            dispatch(holoSlice.setBinning(value));
            await apiInLineHoloControllerSetBinning(value);
            break;
          case "color_channel":
            dispatch(holoSlice.setColorChannel(value));
            await apiInLineHoloControllerSetParams({ color_channel: value });
            break;
          case "flip_x":
            dispatch(holoSlice.setFlipX(value));
            await apiInLineHoloControllerSetParams({ flip_x: value });
            break;
          case "flip_y":
            dispatch(holoSlice.setFlipY(value));
            await apiInLineHoloControllerSetParams({ flip_y: value });
            break;
          case "rotation":
            dispatch(holoSlice.setRotation(value));
            await apiInLineHoloControllerSetParams({ rotation: value });
            break;
          case "full_frame":
            dispatch(holoSlice.setFullFrame(value));
            await apiInLineHoloControllerSetParams({ full_frame: value });
            break;
          default:
            console.warn(`Unknown parameter: ${paramName}`);
        }
      } catch (error) {
        console.error(`Failed to update ${paramName}:`, error);
      }
    },
    [dispatch]
  );

  // ---------------- Detector exposure/gain/AWB ----------------

  const detectorQuery = useMemo(() => {
    return holoState.cameraName
      ? `&detectorName=${encodeURIComponent(holoState.cameraName)}`
      : "";
  }, [holoState.cameraName]);

  const commitExposure = useCallback(
    async (val) => {
      setDetectorParams((p) => ({ ...p, exposure: val }));
      try {
        await fetch(
          `${baseUrl}/SettingsController/setDetectorExposureTime?exposureTime=${val}${detectorQuery}`
        );
      } catch (err) {
        console.error("setDetectorExposureTime failed:", err);
      }
    },
    [baseUrl, detectorQuery]
  );

  const commitGain = useCallback(
    async (val) => {
      setDetectorParams((p) => ({ ...p, gain: val }));
      try {
        await fetch(
          `${baseUrl}/SettingsController/setDetectorGain?gain=${val}${detectorQuery}`
        );
      } catch (err) {
        console.error("setDetectorGain failed:", err);
      }
    },
    [baseUrl, detectorQuery]
  );

  const handleExposureModeChange = useCallback(
    async (mode) => {
      setDetectorParams((p) => ({ ...p, mode }));
      const isAuto = mode === "auto";
      try {
        await fetch(
          `${baseUrl}/SettingsController/setDetectorMode?isAuto=${isAuto}${detectorQuery}`
        );
      } catch (err) {
        console.error("setDetectorMode failed:", err);
      }
    },
    [baseUrl, detectorQuery]
  );

  const handleExposureAutoOnce = useCallback(async () => {
    setAutoOncePending(true);
    try {
      await fetch(
        `${baseUrl}/SettingsController/setDetectorExposureOnce?resetDelayMs=${AUTO_ONCE_RESET_DELAY_MS}${detectorQuery}`
      );
      await new Promise((r) => setTimeout(r, AUTO_ONCE_UI_HOLD_MS));
      await loadDetectorParameters();
    } catch (err) {
      console.error("setDetectorExposureOnce failed:", err);
    } finally {
      setAutoOncePending(false);
    }
  }, [baseUrl, detectorQuery, loadDetectorParameters]);

  const handleAwbModeChange = useCallback(
    async (mode) => {
      setDetectorParams((p) => ({ ...p, awb_mode: mode }));
      try {
        await fetch(
          `${baseUrl}/SettingsController/setWhiteBalance?mode=${mode}${detectorQuery}`
        );
      } catch (err) {
        console.error("setWhiteBalance failed:", err);
      }
    },
    [baseUrl, detectorQuery]
  );

  const handleAwbOnce = useCallback(async () => {
    setAwbOncePending(true);
    try {
      const resp = await fetch(
        `${baseUrl}/SettingsController/setWhiteBalance?mode=once${detectorQuery}`
      );
      if (resp.ok) {
        const data = await resp.json();
        setDetectorParams((p) => ({
          ...p,
          awb_mode: "manual",
          red_gain: data.red_gain ?? p.red_gain,
          blue_gain: data.blue_gain ?? p.blue_gain,
        }));
      }
      await loadDetectorParameters();
    } catch (err) {
      console.error("AWB once failed:", err);
    } finally {
      setAwbOncePending(false);
    }
  }, [baseUrl, detectorQuery, loadDetectorParameters]);

  const commitColourGain = useCallback(
    async (which, value) => {
      const red = which === "red" ? value : detectorParams.red_gain ?? 1.0;
      const blue = which === "blue" ? value : detectorParams.blue_gain ?? 1.0;
      setDetectorParams((p) => ({ ...p, red_gain: red, blue_gain: blue }));
      try {
        await fetch(
          `${baseUrl}/SettingsController/setColourGains?redGain=${red}&blueGain=${blue}${detectorQuery}`
        );
      } catch (err) {
        console.error("setColourGains failed:", err);
      }
    },
    [baseUrl, detectorQuery, detectorParams.red_gain, detectorParams.blue_gain]
  );

  // ---------------- Live-view click → ROI ----------------

  const handleImageLoad = useCallback((width, height) => {
    setImageSize({ width, height });
  }, []);

  const handleLiveViewClick = useCallback(
    (pixelX, pixelY, imgWidth, imgHeight, displayInfoData) => {
      let adjustedX = pixelX;
      let adjustedY = pixelY;
      const rotation = holoState.rotation || 0;
      if (rotation !== 0) {
        const centerX = imgWidth / 2;
        const centerY = imgHeight / 2;
        const rad = (-rotation * Math.PI) / 180;
        const dx = adjustedX - centerX;
        const dy = adjustedY - centerY;
        adjustedX = centerX + dx * Math.cos(rad) - dy * Math.sin(rad);
        adjustedY = centerY + dx * Math.sin(rad) + dy * Math.cos(rad);
      }
      if (holoState.flipX) adjustedX = imgWidth - adjustedX;
      if (holoState.flipY) adjustedY = imgHeight - adjustedY;

      const imageCenterX = imgWidth / 2;
      const imageCenterY = imgHeight / 2;
      const relativeX = adjustedX - imageCenterX;
      const relativeY = adjustedY - imageCenterY;

      if (displayInfoData) setDisplayInfo(displayInfoData);

      const newCenterX = Math.round(relativeX);
      const newCenterY = Math.round(relativeY);
      setRoiSelection((prev) => ({
        ...prev,
        centerX: newCenterX,
        centerY: newCenterY,
      }));

      (async () => {
        try {
          const streamSubsampling = getActiveSubsamplingFactor();
          const streamedCenterX = imgWidth / 2;
          const streamedCenterY = imgHeight / 2;
          const absoluteXInStream = streamedCenterX + newCenterX;
          const absoluteYInStream = streamedCenterY + newCenterY;
          const absoluteXFullSensor = Math.round(absoluteXInStream * streamSubsampling);
          const absoluteYFullSensor = Math.round(absoluteYInStream * streamSubsampling);
          const currentSize = roiSelection.size || 256;
          const finalSize = Math.min(currentSize, 1024);
          await apiInLineHoloControllerSetRoi({
            center_x: absoluteXFullSensor,
            center_y: absoluteYFullSensor,
            size: finalSize,
          });
        } catch (error) {
          console.error("Failed to auto-apply ROI:", error);
        }
      })();
    },
    [
      getActiveSubsamplingFactor,
      roiSelection.size,
      holoState.flipX,
      holoState.flipY,
      holoState.rotation,
    ]
  );

  // ROI overlay SVG
  const roiOverlay = useMemo(() => {
    if (!imageSize.width || !imageSize.height) return null;
    if (holoState.fullFrame) return null; // ROI doesn't apply
    const centerXRel = (roiSelection.centerX + imageSize.width / 2) / imageSize.width;
    const centerYRel = (roiSelection.centerY + imageSize.height / 2) / imageSize.height;
    const sizeXRel = roiSizeInPreview / imageSize.width;
    const sizeYRel = roiSizeInPreview / imageSize.height;
    return (
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
      >
        <rect
          x={(centerXRel - sizeXRel / 2) * 100}
          y={(centerYRel - sizeYRel / 2) * 100}
          width={sizeXRel * 100}
          height={sizeYRel * 100}
          fill="none"
          stroke="red"
          strokeWidth="0.5"
          opacity="0.8"
        />
        <line
          x1={(centerXRel - 0.02) * 100}
          y1={centerYRel * 100}
          x2={(centerXRel + 0.02) * 100}
          y2={centerYRel * 100}
          stroke="red"
          strokeWidth="0.3"
        />
        <line
          x1={centerXRel * 100}
          y1={(centerYRel - 0.02) * 100}
          x2={centerXRel * 100}
          y2={(centerYRel + 0.02) * 100}
          stroke="red"
          strokeWidth="0.3"
        />
      </svg>
    );
  }, [imageSize, roiSelection, roiSizeInPreview, holoState.fullFrame]);

  // Hint shown when dz=0: reconstruction is just the extracted channel
  const showZeroDzHint = useMemo(() => {
    return Math.abs(holoState.dz || 0) < 1e-9;
  }, [holoState.dz]);

  return (
    <Box sx={{ p: { xs: 1, sm: 2 }, width: "100%", maxWidth: "100%" }}>
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        sx={{ mb: 1 }}
      >
        <Typography variant="h5">Inline Hologram Processing</Typography>
        {holoState.cameraName && (
          <Chip
            size="small"
            label={`Detector: ${holoState.cameraName}${holoState.isRGB ? " · RGB" : ""}`}
            variant="outlined"
          />
        )}
      </Stack>

      {/* Control Buttons */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
          <Tooltip title="Start streaming frames from the detector into the hologram reconstruction pipeline.">
            <span>
              <Button
                variant="contained"
                color="primary"
                startIcon={<PlayArrowIcon />}
                onClick={handleStartProcessing}
                disabled={holoState.isProcessing && !holoState.isPaused}
              >
                Start
              </Button>
            </span>
          </Tooltip>

          {holoState.isPaused ? (
            <Tooltip title="Resume processing live frames (restores previous binning).">
              <span>
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<PlayArrowIcon />}
                  onClick={handleResumeProcessing}
                  disabled={!holoState.isProcessing}
                >
                  Resume
                </Button>
              </span>
            </Tooltip>
          ) : (
            <Tooltip title="Pause processing — re-reconstructs only the last frame (binning=1) so you can scrub dz/ROI cheaply.">
              <span>
                <Button
                  variant="contained"
                  color="warning"
                  startIcon={<PauseIcon />}
                  onClick={handlePauseProcessing}
                  disabled={!holoState.isProcessing || holoState.isPaused}
                >
                  Pause
                </Button>
              </span>
            </Tooltip>
          )}

          <Tooltip title="Stop processing entirely and close the worker.">
            <span>
              <Button
                variant="contained"
                color="error"
                startIcon={<StopIcon />}
                onClick={handleStopProcessing}
                disabled={!holoState.isProcessing}
              >
                Stop
              </Button>
            </span>
          </Tooltip>

          <Tooltip title="Re-read all hologram parameters from the backend.">
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={loadParameters}
            >
              Refresh
            </Button>
          </Tooltip>
        </Stack>

        <Stack direction="row" spacing={1} mt={2} flexWrap="wrap" useFlexGap>
          <Chip
            label={holoState.isProcessing ? "Processing" : "Stopped"}
            color={holoState.isProcessing ? "success" : "default"}
            size="small"
          />
          {holoState.isPaused && (
            <Chip label="Paused" color="warning" size="small" />
          )}
          <Chip label={`Frames: ${holoState.frameCount}`} variant="outlined" size="small" />
          <Chip label={`Processed: ${holoState.processedCount}`} variant="outlined" size="small" />
          {holoState.isStreaming && (
            <Chip
              label={`Stream clients: ${holoState.mjpegClientCount}`}
              variant="outlined"
              size="small"
            />
          )}
        </Stack>
      </Paper>

      {/* Video Streams — full viewport width, side-by-side from sm upward */}
      <Grid container spacing={2} mb={2} sx={{ alignItems: "stretch" }}>
        <Grid item xs={12} md={6} sx={{ display: "flex" }}>
          <Card sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <CardContent sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Typography variant="h6">Camera Stream</Typography>
                <Tooltip title="Live preview from the detector. Click to set the hologram ROI.">
                  <InfoOutlinedIcon fontSize="small" color="action" />
                </Tooltip>
              </Stack>
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  flex: 1,
                  minHeight: 300,
                  backgroundColor: "#000",
                  borderRadius: 1,
                  overflow: "hidden",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transform: `scaleX(${holoState.flipX ? -1 : 1}) scaleY(${holoState.flipY ? -1 : 1}) rotate(${holoState.rotation || 0}deg)`,
                }}
              >
                <LiveViewControlWrapper
                  onClick={handleLiveViewClick}
                  onImageLoad={handleImageLoad}
                  overlayContent={roiOverlay}
                  enableStageMovement={false}
                />
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                Click to set ROI center (auto-applies). Preview: {imageSize.width}×{imageSize.height}px
                {!holoState.fullFrame &&
                  ` | ROI: ${roiSelection.size}px → ${Math.round(roiSizeInPreview)}px preview`}
                {holoState.fullFrame && " | Full-frame mode (ROI disabled)"}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} sx={{ display: "flex" }}>
          <Card sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <CardContent sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                <Typography variant="h6">Processed Hologram</Typography>
                <Tooltip title="Reconstructed intensity at the current propagation distance dz. When dz=0 this is just the selected colour channel of the ROI.">
                  <InfoOutlinedIcon fontSize="small" color="action" />
                </Tooltip>
                <Box sx={{ flex: 1 }} />
                <Tooltip title="Show the raw, in-focus hologram (reconstruct at dz=0) instead of the dz set with the slider">
                  <FormControlLabel
                    control={
                      <Switch
                        size="small"
                        checked={holoState.showRaw}
                        onChange={handleShowRawToggle}
                      />
                    }
                    label="Raw (dz=0)"
                    sx={{ mr: 0 }}
                  />
                </Tooltip>
                <Tooltip title="Re-open the MJPEG stream (use if the processed view freezes).">
                  <IconButton size="small" onClick={handleRestartProcessedStream}>
                    <RestartAltIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Stack>
              {streamStalled && (
                <Alert
                  severity="warning"
                  sx={{ mt: 1, mb: 1 }}
                  action={
                    <Button
                      color="inherit"
                      size="small"
                      onClick={handleRestartProcessedStream}
                    >
                      Restart
                    </Button>
                  }
                >
                  Processed stream stalled — no frames for &gt;
                  {Math.round(PROCESSED_STREAM_STALL_MS / 1000)} s.
                </Alert>
              )}
              {showZeroDzHint && (
                <Alert severity="info" sx={{ mt: 1, mb: 1 }}>
                  dz = 0: showing the extracted{" "}
                  <strong>{holoState.colorChannel}</strong> channel of the ROI
                  (no propagation).
                </Alert>
              )}
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  flex: 1,
                  minHeight: 300,
                  backgroundColor: "#000",
                  borderRadius: 1,
                  overflow: "hidden",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {/* Scale the reconstruction up to fill the viewport (it is
                    typically a small ROI) while preserving aspect ratio */}
                <img
                  key={streamNonce}
                  ref={processedImageRef}
                  src={processedStreamUrl}
                  alt="Processed Hologram Stream"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* dz slider — with configurable max + step */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
          <Typography variant="h6">Propagation Distance (dz)</Typography>
          <Tooltip title="Distance from sensor to virtual image plane. Larger values reconstruct objects farther from the sensor.">
            <InfoOutlinedIcon fontSize="small" color="action" />
          </Tooltip>
        </Stack>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
            <Box sx={{ px: 2 }}>
              <Slider
                value={holoState.dz}
                onChange={handleDzChange}
                onChangeCommitted={handleDzCommit}
                min={0}
                max={holoState.dzMax}
                step={holoState.dzStep}
                valueLabelDisplay="on"
                valueLabelFormat={(v) => `${(v * 1e6).toFixed(1)} µm`}
                marks={[
                  { value: 0, label: "0" },
                  {
                    value: holoState.dzMax,
                    label: `${(holoState.dzMax * 1e3).toFixed(holoState.dzMax >= 1e-3 ? 1 : 3)} mm`,
                  },
                ]}
                sx={{
                  "& .MuiSlider-thumb": { width: 24, height: 24 },
                }}
              />
            </Box>
          </Grid>
          <Grid item xs={6} md={2}>
            <FreeNumberField
              label="Max dz (mm)"
              value={holoState.dzMax}
              onCommit={commitDzMax}
              unitFactor={1e-3}
              fixedDecimals={2}
              tooltip="Upper bound of the slider in millimeters."
              min={1e-6}
              size="small"
            />
          </Grid>
          <Grid item xs={6} md={2}>
            <FreeNumberField
              label="Step (µm)"
              value={holoState.dzStep}
              onCommit={commitDzStep}
              unitFactor={1e-6}
              fixedDecimals={2}
              tooltip="Slider step size in micrometers."
              min={1e-9}
              size="small"
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Detector exposure / gain / colour panel */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
          <Typography variant="h6">Detector</Typography>
          <Tooltip
            title={
              <Box sx={{ whiteSpace: "pre-line" }}>
                {[
                  "Exposure mode:",
                  "  · Manual — you set exposure directly.",
                  "  · Auto   — camera continuously adjusts exposure.",
                  "  · Auto once — single auto pass, then back to manual.",
                  "",
                  "White balance (RGB cameras only):",
                  "  · Auto — continuous AWB (NOT recommended under monochromatic illumination).",
                  "  · Manual — uses the red/blue gains below.",
                  "  · Once — measure now, then lock.",
                ].join("\n")}
              </Box>
            }
            arrow
          >
            <InfoOutlinedIcon fontSize="small" color="action" />
          </Tooltip>
        </Stack>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={3}>
            <FreeNumberField
              label="Exposure (ms)"
              value={
                detectorParams.exposure === ""
                  ? null
                  : Number(detectorParams.exposure)
              }
              onCommit={commitExposure}
              tooltip="Sensor integration time in milliseconds."
              disabled={detectorParams.mode === "auto"}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <FreeNumberField
              label="Gain"
              value={
                detectorParams.gain === ""
                  ? null
                  : Number(detectorParams.gain)
              }
              onCommit={commitGain}
              tooltip="Analog gain (sensor-dependent units)."
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Tooltip
              arrow
              title="Manual: fixed exposure. Auto: camera adapts exposure each frame."
              placement="top-start"
            >
              <FormControl size="small" fullWidth>
                <InputLabel id="exposure-mode-label">Exposure mode</InputLabel>
                <Select
                  labelId="exposure-mode-label"
                  value={detectorParams.mode}
                  label="Exposure mode"
                  onChange={(e) => handleExposureModeChange(e.target.value)}
                >
                  <MenuItem value="manual">Manual</MenuItem>
                  <MenuItem value="auto">Auto</MenuItem>
                </Select>
              </FormControl>
            </Tooltip>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Tooltip title="Run a single auto-exposure pass, then return to manual." arrow>
              <span>
                <Button
                  size="small"
                  variant="contained"
                  fullWidth
                  onClick={handleExposureAutoOnce}
                  disabled={detectorParams.mode !== "manual" || autoOncePending}
                >
                  Exposure Auto Once
                </Button>
              </span>
            </Tooltip>
          </Grid>
        </Grid>

        {detectorParams.isRGB && detectorParams.awb_mode !== null && (
          <>
            <Divider sx={{ my: 2 }} />
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
              <Typography variant="subtitle1">White balance</Typography>
              <Tooltip title="Tip: under a monochromatic laser, leave AWB on Manual with neutral (1.0/1.0) gains. AWB tries to balance the scene to white and pushes the opposite-channel gain way up under a single-colour source, which is what makes a red laser look blue.">
                <InfoOutlinedIcon fontSize="small" color="action" />
              </Tooltip>
            </Stack>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6} md={3}>
                <Tooltip
                  arrow
                  title="Auto: continuous (bad under laser). Manual: fixed gains. Once: measure now and lock."
                >
                  <FormControl size="small" fullWidth>
                    <InputLabel id="awb-mode-label">AWB mode</InputLabel>
                    <Select
                      labelId="awb-mode-label"
                      value={detectorParams.awb_mode || "manual"}
                      label="AWB mode"
                      onChange={(e) => handleAwbModeChange(e.target.value)}
                    >
                      <MenuItem value="manual">Manual</MenuItem>
                      <MenuItem value="auto">Auto</MenuItem>
                      <MenuItem value="once">Once (lock now)</MenuItem>
                    </Select>
                  </FormControl>
                </Tooltip>
              </Grid>
              <Grid item xs={6} sm={3} md={2}>
                <FreeNumberField
                  label="Red gain"
                  value={
                    detectorParams.red_gain === null
                      ? null
                      : Number(detectorParams.red_gain)
                  }
                  onCommit={(v) => commitColourGain("red", v)}
                  tooltip="Red channel gain. Neutral = 1.0."
                  disabled={detectorParams.awb_mode === "auto"}
                />
              </Grid>
              <Grid item xs={6} sm={3} md={2}>
                <FreeNumberField
                  label="Blue gain"
                  value={
                    detectorParams.blue_gain === null
                      ? null
                      : Number(detectorParams.blue_gain)
                  }
                  onCommit={(v) => commitColourGain("blue", v)}
                  tooltip="Blue channel gain. Neutral = 1.0."
                  disabled={detectorParams.awb_mode === "auto"}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={2}>
                <Tooltip
                  arrow
                  title="Run AWB once, lock the resulting gains. Point camera at a white target first."
                >
                  <span>
                    <Button
                      size="small"
                      variant="contained"
                      fullWidth
                      onClick={handleAwbOnce}
                      disabled={awbOncePending}
                    >
                      AWB Once
                    </Button>
                  </span>
                </Tooltip>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Tooltip
                  arrow
                  title="Reset both gains to 1.0 — neutral, no per-channel correction."
                >
                  <span>
                    <Button
                      size="small"
                      variant="outlined"
                      fullWidth
                      onClick={async () => {
                        await handleAwbModeChange("manual");
                        await commitColourGain("red", 1.0);
                        await commitColourGain("blue", 1.0);
                      }}
                    >
                      Neutral gains (1.0)
                    </Button>
                  </span>
                </Tooltip>
              </Grid>
            </Grid>
          </>
        )}
      </Paper>

      {/* Focus Sweep */}
      <Accordion sx={{ mb: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <CenterFocusStrongIcon sx={{ mr: 1 }} />
          <Typography>Focus Sweep (auto dz)</Typography>
          {sweep.running && (
            <Chip label="Running" color="success" size="small" sx={{ ml: 2 }} />
          )}
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={4}>
              <TextField
                label="Start dz (µm)"
                type="number"
                value={sweep.startUm}
                onChange={(e) =>
                  setSweep((prev) => ({ ...prev, startUm: parseFloat(e.target.value) || 0 }))
                }
                fullWidth
                disabled={sweep.running}
                inputProps={{ step: 10 }}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                label="End dz (µm)"
                type="number"
                value={sweep.endUm}
                onChange={(e) =>
                  setSweep((prev) => ({ ...prev, endUm: parseFloat(e.target.value) || 0 }))
                }
                fullWidth
                disabled={sweep.running}
                inputProps={{ step: 10 }}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                label="Steps (< 20)"
                type="number"
                value={sweep.steps}
                onChange={(e) =>
                  setSweep((prev) => ({ ...prev, steps: parseInt(e.target.value, 10) || 2 }))
                }
                fullWidth
                disabled={sweep.running}
                inputProps={{ step: 1, min: 2, max: 19 }}
              />
            </Grid>
          </Grid>

          <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<PlayArrowIcon />}
              onClick={startFocusSweep}
              disabled={sweep.running}
            >
              Start Sweep
            </Button>
            <Button
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
              onClick={stopFocusSweep}
              disabled={!sweep.running}
            >
              Stop
            </Button>
          </Stack>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ mt: 1, display: "block" }}
          >
            Steps dz from start to end (1 step/second) and loops until stopped.
            Stop keeps the currently active dz. Maximum 19 steps.
          </Typography>
        </AccordionDetails>
      </Accordion>

      {/* ROI Controls */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Typography variant="h6">ROI Selection</Typography>
            <Tooltip title="Square crop in sensor pixels that gets propagated. Set to full-frame to skip cropping entirely.">
              <InfoOutlinedIcon fontSize="small" color="action" />
            </Tooltip>
          </Stack>
          <Stack direction="row" spacing={1} alignItems="center">
            <Tooltip title="Bypass the ROI crop and reconstruct the full sensor (with software binning applied).">
              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={holoState.fullFrame}
                    onChange={(e) => commitParam("full_frame", e.target.checked)}
                  />
                }
                label="Full frame"
              />
            </Tooltip>
            <Tooltip title="Reset ROI to image center, size 256px.">
              <IconButton onClick={handleResetRoi} size="small">
                <CenterFocusStrongIcon />
              </IconButton>
            </Tooltip>
          </Stack>
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={12} sm={4}>
            <Typography gutterBottom>Center X (relative to center)</Typography>
            <Slider
              value={roiSelection.centerX}
              onChange={handleRoiCenterXChange}
              min={-Math.floor(imageSize.width / 2)}
              max={Math.floor(imageSize.width / 2)}
              step={1}
              valueLabelDisplay="auto"
              disabled={holoState.fullFrame}
              marks={[
                { value: -Math.floor(imageSize.width / 2), label: `${-Math.floor(imageSize.width / 2)}` },
                { value: 0, label: "0" },
                { value: Math.floor(imageSize.width / 2), label: `${Math.floor(imageSize.width / 2)}` },
              ]}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography gutterBottom>Center Y (relative to center)</Typography>
            <Slider
              value={roiSelection.centerY}
              onChange={handleRoiCenterYChange}
              min={-Math.floor(imageSize.height / 2)}
              max={Math.floor(imageSize.height / 2)}
              step={1}
              valueLabelDisplay="auto"
              disabled={holoState.fullFrame}
              marks={[
                { value: -Math.floor(imageSize.height / 2), label: `${-Math.floor(imageSize.height / 2)}` },
                { value: 0, label: "0" },
                { value: Math.floor(imageSize.height / 2), label: `${Math.floor(imageSize.height / 2)}` },
              ]}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography gutterBottom>
              ROI Size: {roiSelection.size}px (backend) /{" "}
              {Math.round(roiSizeInPreview)}px (preview)
            </Typography>
            <Slider
              value={roiSelection.size}
              onChange={handleRoiSizeChange}
              min={64}
              max={1024}
              step={64}
              valueLabelDisplay="auto"
              disabled={holoState.fullFrame}
              valueLabelFormat={(v) => `${v}px`}
              marks={[
                { value: 64, label: "64" },
                { value: 256, label: "256" },
                { value: 512, label: "512" },
                { value: 1024, label: "1024" },
              ]}
            />
            <Typography variant="caption" color="text.secondary">
              Scaling: {totalScalingFactor}× (subsampling: {getActiveSubsamplingFactor()}, binning: {holoState.binning || 1})
            </Typography>
          </Grid>
        </Grid>

        <Button
          variant="contained"
          color="primary"
          onClick={handleApplyRoi}
          fullWidth
          sx={{ mt: 2 }}
          disabled={holoState.fullFrame}
        >
          Apply ROI
        </Button>
      </Paper>

      {/* Developer Options */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <SettingsIcon sx={{ mr: 1 }} />
          <Typography>Developer Options</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <FreeNumberField
                label="Pixel Size (µm)"
                value={holoState.pixelsize}
                onCommit={(v) => commitParam("pixelsize", v)}
                unitFactor={1e-6}
                fixedDecimals={3}
                tooltip="Effective sensor pixel size before binning. Binning factor is applied automatically by the propagator."
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FreeNumberField
                label="Wavelength (nm)"
                value={holoState.wavelength}
                onCommit={(v) => commitParam("wavelength", v)}
                unitFactor={1e-9}
                fixedDecimals={1}
                tooltip="Illumination wavelength in nanometers. Common values: 405, 488, 532, 638, 660 nm."
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FreeNumberField
                label="Numerical Aperture (NA)"
                value={holoState.na}
                onCommit={(v) => commitParam("na", v)}
                tooltip="Reserved for future band-limiting; currently informational only."
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Tooltip
                arrow
                title="Software binning factor applied before propagation. Larger = faster, lower resolution."
              >
                <FormControl size="small" fullWidth>
                  <InputLabel>Binning</InputLabel>
                  <Select
                    value={holoState.binning}
                    label="Binning"
                    onChange={(e) => commitParam("binning", e.target.value)}
                  >
                    <MenuItem value={1}>1×1</MenuItem>
                    <MenuItem value={2}>2×2</MenuItem>
                    <MenuItem value={4}>4×4</MenuItem>
                    <MenuItem value={8}>8×8</MenuItem>
                  </Select>
                </FormControl>
              </Tooltip>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Tooltip
                arrow
                title="Which colour channel of the (RGB) raw frame to reconstruct. 'White' takes the mean of all channels (luminance-like)."
              >
                <FormControl size="small" fullWidth>
                  <InputLabel>Color Channel</InputLabel>
                  <Select
                    value={holoState.colorChannel}
                    label="Color Channel"
                    onChange={(e) => commitParam("color_channel", e.target.value)}
                  >
                    <MenuItem value="red">Red</MenuItem>
                    <MenuItem value="green">Green</MenuItem>
                    <MenuItem value="blue">Blue</MenuItem>
                    <MenuItem value="white">White (mean)</MenuItem>
                  </Select>
                </FormControl>
              </Tooltip>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FreeNumberField
                label="Update Frequency (Hz)"
                value={holoState.updateFreq}
                onCommit={(v) => commitParam("update_freq", v)}
                tooltip="Target processing rate. Higher = more CPU. The actual rate is bounded by camera fps and reconstruction cost."
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <Tooltip arrow title="Mirror image horizontally before reconstruction.">
                <FormControlLabel
                  control={
                    <Switch
                      checked={holoState.flipX}
                      onChange={(e) => commitParam("flip_x", e.target.checked)}
                    />
                  }
                  label="Flip X"
                />
              </Tooltip>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Tooltip arrow title="Mirror image vertically before reconstruction.">
                <FormControlLabel
                  control={
                    <Switch
                      checked={holoState.flipY}
                      onChange={(e) => commitParam("flip_y", e.target.checked)}
                    />
                  }
                  label="Flip Y"
                />
              </Tooltip>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Tooltip
                arrow
                title="Rotate image counter-clockwise before reconstruction (in degrees)."
              >
                <FormControl size="small" fullWidth>
                  <InputLabel>Rotation</InputLabel>
                  <Select
                    value={holoState.rotation}
                    label="Rotation"
                    onChange={(e) => commitParam("rotation", e.target.value)}
                  >
                    <MenuItem value={0}>0°</MenuItem>
                    <MenuItem value={90}>90°</MenuItem>
                    <MenuItem value={180}>180°</MenuItem>
                    <MenuItem value={270}>270°</MenuItem>
                  </Select>
                </FormControl>
              </Tooltip>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default HoloController;
