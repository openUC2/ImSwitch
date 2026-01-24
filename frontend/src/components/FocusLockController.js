import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Paper,
  Grid,
  Button,
  Typography,
  TextField,
  Box,
  Switch,
  FormControlLabel,
  Slider,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Alert,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  ButtonGroup,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import * as focusLockSlice from "../state/slices/FocusLockSlice.js";
import { useTheme } from "@mui/material/styles";
import { useWebSocket } from "../context/WebSocketContext";
import "chartjs-adapter-date-fns";
import { Line } from "react-chartjs-2";
import "../utils/chartSetup"; // Global Chart.js registration

// Import updated API functions
import apiFocusLockControllerFocusCalibrationStart from "../backendapi/apiFocusLockControllerFocusCalibrationStart.js";
import apiFocusLockControllerGetParamsAstigmatism from "../backendapi/apiFocusLockControllerGetParamsAstigmatism.js";
import apiFocusLockControllerReturnLastImage from "../backendapi/apiFocusLockControllerReturnLastImage.js";
import apiFocusLockControllerReturnLastCroppedImage from "../backendapi/apiFocusLockControllerReturnLastCroppedImage.js";
import apiFocusLockControllerSetCropFrameParameters from "../backendapi/apiFocusLockControllerSetCropFrameParameters.js";
import apiFocusLockControllerSetPIParameters from "../backendapi/apiFocusLockControllerSetPIParameters.js";
import apiFocusLockControllerSetPIControllerParams from "../backendapi/apiFocusLockControllerSetPIControllerParams.js";
import apiFocusLockControllerSetParamsAstigmatism from "../backendapi/apiFocusLockControllerSetParamsAstigmatism.js";
import apiFocusLockControllerToggleFocus from "../backendapi/apiFocusLockControllerToggleFocus.js";
import apiFocusLockControllerUnlockFocus from "../backendapi/apiFocusLockControllerUnlockFocus.js";
import apiFocusLockControllerStartFocusMeasurement from "../backendapi/apiFocusLockControllerStartFocusMeasurement.js";
import apiFocusLockControllerStopFocusMeasurement from "../backendapi/apiFocusLockControllerStopFocusMeasurement.js";
import apiFocusLockControllerEnableFocusLock from "../backendapi/apiFocusLockControllerEnableFocusLock.js";
import apiFocusLockControllerGetFocusLockState from "../backendapi/apiFocusLockControllerGetFocusLockState.js";
import apiFocusLockControllerGetPIParameters from "../backendapi/apiFocusLockControllerGetPIParameters.js";
import apiFocusLockControllerGetPIControllerParams from "../backendapi/apiFocusLockControllerGetPIControllerParams.js";
import apiFocusLockControllerRunFocusCalibrationDynamic from "../backendapi/apiFocusLockControllerRunFocusCalibrationDynamic.js";
import apiFocusLockControllerGetCalibrationStatus from "../backendapi/apiFocusLockControllerGetCalibrationStatus.js";
import apiFocusLockControllerGetCalibrationResults from "../backendapi/apiFocusLockControllerGetCalibrationResults.js";
import apiFocusLockControllerStopFocusCalibration from "../backendapi/apiFocusLockControllerStopFocusCalibration.js";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";
import apiFocusLockControllerSetExposureTime from "../backendapi/apiFocusLockControllerSetExposureTime.js";
import apiFocusLockControllerSetGain from "../backendapi/apiFocusLockControllerSetGain.js";
import apiFocusLockControllerGetExposureTime from "../backendapi/apiFocusLockControllerGetExposureTime.js";
import apiFocusLockControllerGetGain from "../backendapi/apiFocusLockControllerGetGain.js";
import apiFocusLockControllerGetCurrentFocusValue from "../backendapi/apiFocusLockControllerGetCurrentFocusValue.js";
import apiFocusLockControllerPerformOneStepAutofocus from "../backendapi/apiFocusLockControllerPerformOneStepAutofocus.js";

// Import LiveViewControlWrapper for camera livestream
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";



const FocusLockController = () => {
  // Calibration polling state
  const calibrationPollingRef = useRef(null);
  const calibrationTimeoutRef = useRef(null);
  const [isCalibrationPolling, setIsCalibrationPolling] = useState(false);
  const dispatch = useDispatch();
  const theme = useTheme();
  const socket = useWebSocket();
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const retryCountRef = useRef(0);
  const imgRef = useRef(null);
  const latestFocus = useRef(0);

  const maxRetries = 3;
  const [pollingError, setPollingError] = useState(false);
  const [continuousImageLoading, setContinuousImageLoading] = useState(false); // New state for image polling control

  // Calibration parameters exposed to user
  const [scanRangeUm, setScanRangeUm] = useState(200);
  const [numSteps, setNumSteps] = useState(10);
  const [settleTime, setSettleTime] = useState(0.5);

  // Manual target setpoint input
  const [manualTargetSetpoint, setManualTargetSetpoint] = useState("");

  // Calibration results display
  const [calibrationResults, setCalibrationResults] = useState(null);
  
  // Calibration curve display state
  const [showCalibrationCurve, setShowCalibrationCurve] = useState(false);
  const [calibrationCurveData, setCalibrationCurveData] = useState(null);
  
  // Main tab state for the controller
  const [mainTabValue, setMainTabValue] = useState(0); // 0: Calibration, 1: Continuous Lock, 2: One-Step Focus
  
  // Developer mode toggle
  const [devModeOpen, setDevModeOpen] = useState(false);

  // Access Redux state with specific selectors for better performance
  const isMeasuring = useSelector((state) => state.focusLockState.isMeasuring);
  // Split selectors to prevent unnecessary re-renders when only focus values change
  const focusLockUI = useSelector((state) => ({
    isFocusLocked: state.focusLockState.isFocusLocked,
    isCalibrating: state.focusLockState.isCalibrating,
    gaussian_sigma: state.focusLockState.gaussian_sigma,
    background_threshold: state.focusLockState.background_threshold,
    cropSize: state.focusLockState.cropSize,
    cropCenter: state.focusLockState.cropCenter,
    frameSize: state.focusLockState.frameSize,
    lastImage: state.focusLockState.lastImage,
    lastCroppedImage: state.focusLockState.lastCroppedImage,
    showImageSelector: state.focusLockState.showImageSelector,
    isLoadingImage: state.focusLockState.isLoadingImage,
    kp: state.focusLockState.kp,
    ki: state.focusLockState.ki,
    setPoint: state.focusLockState.setPoint,
    safetyDistanceLimit: state.focusLockState.safetyDistanceLimit,
    safetyMoveLimit: state.focusLockState.safetyMoveLimit,
    minStepThreshold: state.focusLockState.minStepThreshold,
    safetyMotionActive: state.focusLockState.safetyMotionActive,
    currentFocusValue: state.focusLockState.currentFocusValue,
    exposureTime: state.focusLockState.exposureTime,
    gain: state.focusLockState.gain,
    storedTargetSetpoint: state.focusLockState.storedTargetSetpoint,
    lastAutofocusResult: state.focusLockState.lastAutofocusResult,
    isPerformingAutofocus: state.focusLockState.isPerformingAutofocus,
    calibrationRange: state.focusLockState.calibrationRange,
  }));

  // Separate selector for chart data to minimize re-renders
  const chartDataSource = useSelector((state) => ({
    focusValues: state.focusLockState.focusValues,
    focusTimepoints: state.focusLockState.focusTimepoints,
    motorPositions: state.focusLockState.motorPositions,
    setPointSignals: state.focusLockState.setPointSignals,
    focusHistory: state.focusLockState.focusHistory,
  }));

  // Local state for image display and crop selection
  const [cropSelection, setCropSelection] = useState({
    isSelecting: false,
    startX: 0,
    startY: 0,
    endX: 0,
    endY: 0,
  });

  // Load last image from backend - memoized to prevent unnecessary re-renders
  const loadLastImage = useCallback(async () => {
    try {
      dispatch(focusLockSlice.setIsLoadingImage(true));
      setPollingError(false);

      const blob = await apiFocusLockControllerReturnLastImage();

      // Clean up previous blob URL to prevent memory leaks
      if (focusLockUI.lastImage && focusLockUI.lastImage.startsWith("blob:")) {
        URL.revokeObjectURL(focusLockUI.lastImage);
      }

      const dataUrl = URL.createObjectURL(blob);
      dispatch(focusLockSlice.setLastImage(dataUrl));
      dispatch(focusLockSlice.setShowImageSelector(true));

      // After image loads, set frameSize in Redux to natural image dimensions for consistency
      setTimeout(() => {
        const img = imgRef.current;
        if (img && img.naturalWidth && img.naturalHeight) {
          dispatch(
            focusLockSlice.setFrameSize([img.naturalWidth, img.naturalHeight])
          );
        }
      }, 100); // Reduce timeout from 1000ms to 100ms

      // Reset retry count on success
      retryCountRef.current = 0;
    } catch (error) {
      console.error("Failed to load last image:", error);
      retryCountRef.current += 1;

      // Stop polling after max retries to prevent CPU overload
      if (retryCountRef.current >= maxRetries) {
        setPollingError(true);
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
          dispatch(focusLockSlice.setIsMeasuring(false));
        }
      }
    } finally {
      dispatch(focusLockSlice.setIsLoadingImage(false));
    }
  }, [dispatch, maxRetries]); // Remove focusLockState dependency

  /** keep the latest value without causing re-renders */
  useEffect(() => {
    latestFocus.current = focusLockUI.currentFocusValue;
  }, [focusLockUI.currentFocusValue]); // Add dependency array

  /** dataset memoized to prevent unnecessary Chart.js re-renders */
  const chartData = useMemo(() => {
    // Memoize the data transformation to prevent recalculation on every render
    // Focus Value
    const transformedFocusData = chartDataSource.focusValues.map((v, i) => ({
      x: chartDataSource.focusTimepoints[i] || i,
      y: v ?? 0,
    }));
    // Motor Position
    const transformedMotorData =
      chartDataSource.motorPositions?.map((v, i) => ({
        x: chartDataSource.focusTimepoints[i] || i,
        y: v ?? 0,
      })) || [];
    // Set Point Signal
    const transformedSetPointData =
      chartDataSource.setPointSignals?.map((v, i) => ({
        x: chartDataSource.focusTimepoints[i] || i,
        y: v ?? 0,
      })) || [];

    // If setPointSignals is not present, fallback to extracting from focusHistory
    let setPointData = transformedSetPointData;
    if (
      setPointData.length === 0 &&
      Array.isArray(chartDataSource.focusHistory)
    ) {
      setPointData = chartDataSource.focusHistory.map((entry, i) => ({
        x: entry.timestamp || i,
        y: entry.setPointSignal ?? 0,
      }));
    }

    return {
      datasets: [
        {
          label: "Focus Value",
          borderColor: theme.palette.primary.main,
          backgroundColor: theme.palette.primary.light,
          data: transformedFocusData,
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y",
        },
        {
          label: "Set Point Signal",
          borderColor: "orange",
          backgroundColor: "rgba(255,165,0,0.2)",
          data: setPointData,
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y", // left axis
        },
        {
          label: "Motor Position",
          borderColor: theme.palette.secondary.main,
          backgroundColor: theme.palette.secondary.light,
          data: transformedMotorData,
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y1",
        },
      ],
    };
  }, [
    theme.palette.primary.main,
    theme.palette.primary.light,
    theme.palette.secondary.main,
    theme.palette.secondary.light,
    chartDataSource.focusValues,
    chartDataSource.motorPositions,
    chartDataSource.focusTimepoints,
    chartDataSource.setPointSignals,
    chartDataSource.focusHistory,
  ]);

  const chartOptions = useMemo(
    () => ({
      animation: false,
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Sample" },
          ticks: { autoSkip: true, maxTicksLimit: 10 },
        },
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Focus Value" },
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: { display: true, text: "Motor Position (µm)" },
          grid: {
            drawOnChartArea: false,
          },
        },
      },
      interaction: { intersect: false },
      plugins: { legend: { position: "top" } },
    }),
    []
  );

  // Polling for image updates with better error handling and cleanup
  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Only poll when measuring is active, continuous loading is enabled, and no polling errors
    if ( continuousImageLoading && !pollingError) {
      // Fix: Only poll when actually measuring AND continuous loading is enabled
      intervalRef.current = setInterval(loadLastImage, 500);
    }

    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isMeasuring, continuousImageLoading, pollingError, loadLastImage]);

  // Load initial parameters on mount
  useEffect(() => {
    loadAstigmatismParameters();
    loadPIParameters();
    loadFocusLockState();
    loadCameraParameters();

    // Cleanup blob URLs on unmount
    return () => {
      // Cleanup is handled automatically by React when state changes
      // No need to manually revoke URLs here as they're managed by Redux state
    };
  }, []);

  // WebSocket handler for laser value updates - optimized to reduce re-renders
  useEffect(() => {
    if (!socket) return;

    // Throttle mechanism to reduce Redux updates frequency
    let lastUpdateTime = 0;
    const UPDATE_THROTTLE_MS = 100; // Update at most every 100ms (10Hz)

    const handleSignal = (data) => {
      try {
        const parsedData = JSON.parse(data);

        // Handle one-step autofocus progress signals
        if (parsedData.sigOneStepAutofocusProgress) {
          const progressData = parsedData.sigOneStepAutofocusProgress;
          console.log("One-step autofocus progress:", progressData);

          if (progressData.event === "autofocus_started") {
            dispatch(focusLockSlice.setIsPerformingAutofocus(true));
          } else if (progressData.event === "autofocus_completed") {
            dispatch(focusLockSlice.setIsPerformingAutofocus(false));
            if (progressData.result) {
              dispatch(
                focusLockSlice.setLastAutofocusResult(progressData.result)
              );
            }
          }
          return;
        }

        // Handle calibration progress signals
        if (parsedData.sigCalibrationProgress) {
          const calibData = parsedData.sigCalibrationProgress;
          console.log("Calibration progress:", calibData);

          if (calibData.event === "calibration_completed") {
            dispatch(focusLockSlice.setIsCalibrating(false));
            if (
              calibData.calibration_data &&
              calibData.calibration_data.linear_range
            ) {
              dispatch(
                focusLockSlice.setCalibrationRange(
                  calibData.calibration_data.linear_range
                )
              );
            }
            // Store calibration results for display
            setCalibrationResults({
              r_squared: calibData.r_squared,
              sensitivity_nm_per_px: calibData.sensitivity_nm_per_px,
              monotone_failures: calibData.monotone_failures,
              total_points: calibData.total_points,
              coefficients: calibData.coefficients,
            });
          }
          return;
        }

        // Check if this is a laser value update
        // Format: {"p0":"('Laser', 'LED', 'Value')","p1":20812}
        if (parsedData.p0 && parsedData.p1 !== undefined) {
          // Parse the p0 string to extract laser information
          const p0String = parsedData.p0;
          const value = parsedData.p1;

          // Extract laser name and type from the tuple string
          // Expected format: "('Laser', 'LED', 'Value')"
          const tupleMatch = p0String.match(
            /\('([^']+)',\s*'([^']+)',\s*'([^']+)'\)/
          );

          if (
            tupleMatch &&
            tupleMatch[1] === "Laser" &&
            tupleMatch[3] === "Value"
          ) {
            const laserName = tupleMatch[2]; // e.g., 'LED'
            const now = Date.now();

            // Throttle updates to reduce CPU overhead
            if (now - lastUpdateTime < UPDATE_THROTTLE_MS) return;

            // Only update if the value actually changed significantly
            if (Math.abs(latestFocus.current - value) > 0.001) {
              // Add threshold to reduce noise
              lastUpdateTime = now;

              // Combine both actions into a single dispatch to reduce re-renders
              dispatch(
                focusLockSlice.addFocusValue({
                  focusValue: value,
                  timestamp: now,
                })
              );

              // Update the ref immediately for next comparison
              latestFocus.current = value;
            }
          }
        }
      } catch (error) {
        // Reduce error logging frequency to prevent console spam
        if (Math.random() < 0.01) {
          // Log only 1% of errors
          console.error("Error parsing WebSocket laser update:", error);
        }
      }
    };

    socket.on("signal", handleSignal);

    return () => {
      socket.off("signal", handleSignal);
    };
  }, [socket, dispatch]); // Remove latestFocus from dependencies to prevent recreation

  // Load astigmatism parameters from backend - memoized
  const loadAstigmatismParameters = useCallback(async () => {
    try {
      const params = await apiFocusLockControllerGetParamsAstigmatism();
      if (params) {
        dispatch(
          focusLockSlice.setgaussian_sigma(params.gaussian_sigma || 1.0)
        );
        dispatch(
          focusLockSlice.setBackgroundThreshold(
            params.background_threshold || 100.0
          )
        );
        dispatch(focusLockSlice.setCropSize(params.cropSize || 100));
        dispatch(focusLockSlice.setCropCenter(params.cropCenter || [0, 0]));
      }
    } catch (error) {
      console.error("Failed to load astigmatism parameters:", error);
    }
  }, [dispatch]);

  // Load PI parameters from backend - memoized
  const loadPIParameters = useCallback(async () => {
    try {
      // Try to get extended PI controller parameters first
      const params = await apiFocusLockControllerGetPIControllerParams();
      if (params) {
        dispatch(focusLockSlice.setKp(params.kp || 0.1));
        dispatch(focusLockSlice.setKi(params.ki || 0.0));
        dispatch(focusLockSlice.setSetPoint(params.setPoint || 0.0));
        dispatch(
          focusLockSlice.setSafetyDistanceLimit(
            params.safetyDistanceLimit || 500.0
          )
        );
        dispatch(
          focusLockSlice.setSafetyMoveLimit(params.safetyMoveLimit || 3.0)
        );
        dispatch(
          focusLockSlice.setMinStepThreshold(params.minStepThreshold || 0.002)
        );
        dispatch(
          focusLockSlice.setSafetyMotionActive(
            params.safetyMotionActive || false
          )
        );
      }
    } catch (error) {
      console.warn(
        "Extended PI parameters not available, trying legacy API:",
        error
      );
      // Fallback to legacy PI parameters API
      try {
        const params = await apiFocusLockControllerGetPIParameters();
        if (params && Array.isArray(params) && params.length >= 2) {
          dispatch(focusLockSlice.setKp(params[0] || 0.1));
          dispatch(focusLockSlice.setKi(params[1] || 0.01));
        }
      } catch (legacyError) {
        console.error("Failed to load PI parameters:", legacyError);
      }
    }
  }, [dispatch]);

  // Load focus lock state from backend - memoized
  const loadFocusLockState = useCallback(async () => {
    try {
      const state = await apiFocusLockControllerGetFocusLockState();
      if (state) {
        dispatch(focusLockSlice.setFocusLocked(state.is_locked || false));
        dispatch(
          focusLockSlice.setIsCalibrating(state.is_calibrating || false)
        );
        dispatch(focusLockSlice.setIsMeasuring(state.is_measuring || false));
      }
    } catch (error) {
      console.error("Failed to load focus lock state:", error);
    }
  }, [dispatch]);

  // Load camera parameters from backend - memoized
  const loadCameraParameters = useCallback(async () => {
    try {
      // Load exposure time
      try {
        const exposureTime = await apiFocusLockControllerGetExposureTime();
        if (exposureTime !== undefined && exposureTime !== null) {
          dispatch(focusLockSlice.setExposureTime(exposureTime));
        }
      } catch (error) {
        console.warn("Failed to load exposure time, using default:", error);
      }

      // Load gain
      try {
        const gain = await apiFocusLockControllerGetGain();
        if (gain !== undefined && gain !== null) {
          dispatch(focusLockSlice.setGain(gain));
        }
      } catch (error) {
        console.warn("Failed to load gain, using default:", error);
      }
    } catch (error) {
      console.error("Failed to load camera parameters:", error);
    }
  }, [dispatch]);

  // Start/stop focus measurement - memoized
  const toggleFocusMeasurement = useCallback(async () => {
    try {
      if (isMeasuring) {
        await apiFocusLockControllerStopFocusMeasurement();
        dispatch(focusLockSlice.setIsMeasuring(false));
        // Clear any polling errors when manually stopping
        setPollingError(false);
        retryCountRef.current = 0;
      } else {
        await apiFocusLockControllerStartFocusMeasurement();
        dispatch(focusLockSlice.setIsMeasuring(true));
        setPollingError(false);
        retryCountRef.current = 0;
      }
    } catch (error) {
      console.error("Failed to toggle focus measurement:", error);
    }
  }, [isMeasuring, dispatch]);

  // Handle PI parameter updates - memoized
  const updatePIParameters = useCallback(async () => {
    try {
      await apiFocusLockControllerSetPIParameters({
        kp: focusLockUI.kp,
        ki: focusLockUI.ki,
      });
    } catch (error) {
      console.error("Failed to update PI parameters:", error);
    }
  }, [focusLockUI.kp, focusLockUI.ki]);

  // Handle extended PI controller parameter updates - memoized
  const updatePIControllerParameters = useCallback(async () => {
    try {
      await apiFocusLockControllerSetPIControllerParams({
        kp: focusLockUI.kp,
        ki: focusLockUI.ki,
        setPoint: focusLockUI.setPoint,
        safetyDistanceLimit: focusLockUI.safetyDistanceLimit,
        safetyMoveLimit: focusLockUI.safetyMoveLimit,
        minStepThreshold: focusLockUI.minStepThreshold,
        safetyMotionActive: focusLockUI.safetyMotionActive,
      });
    } catch (error) {
      console.error("Failed to update PI controller parameters:", error);
    }
  }, [
    focusLockUI.kp,
    focusLockUI.ki,
    focusLockUI.setPoint,
    focusLockUI.safetyDistanceLimit,
    focusLockUI.safetyMoveLimit,
    focusLockUI.minStepThreshold,
    focusLockUI.safetyMotionActive,
  ]);

  // Handle astigmatism parameter updates - memoized
  const updateAstigmatismParameters = useCallback(async () => {
    try {
      await apiFocusLockControllerSetParamsAstigmatism({
        gaussian_sigma: focusLockUI.gaussian_sigma,
        background_threshold: focusLockUI.background_threshold,
        cropSize: focusLockUI.cropSize,
        cropCenter: focusLockUI.cropCenter,
      });
    } catch (error) {
      console.error("Failed to update astigmatism parameters:", error);
    }
  }, [
    focusLockUI.gaussian_sigma,
    focusLockUI.background_threshold,
    focusLockUI.cropSize,
    focusLockUI.cropCenter,
  ]);

  // Handle camera parameter updates - memoized with auto-save
  const updateExposureTime = useCallback(async (newValue) => {
    try {
      await apiFocusLockControllerSetExposureTime(newValue);
    } catch (error) {
      console.error("Failed to update exposure time:", error);
    }
  }, []);

  const updateGain = useCallback(async (newValue) => {
    try {
      await apiFocusLockControllerSetGain(newValue);
    } catch (error) {
      console.error("Failed to update gain:", error);
    }
  }, []);
  
  // Debounced handlers for camera parameters to avoid excessive API calls
  const exposureTimeoutRef = useRef(null);
  const gainTimeoutRef = useRef(null);
  
  const handleExposureChange = useCallback((newValue) => {
    dispatch(focusLockSlice.setExposureTime(newValue));
    // Clear existing timeout
    if (exposureTimeoutRef.current) {
      clearTimeout(exposureTimeoutRef.current);
    }
    // Set new timeout to update after 500ms
    exposureTimeoutRef.current = setTimeout(() => {
      updateExposureTime(newValue);
    }, 500);
  }, [dispatch, updateExposureTime]);
  
  const handleGainChange = useCallback((newValue) => {
    dispatch(focusLockSlice.setGain(newValue));
    // Clear existing timeout
    if (gainTimeoutRef.current) {
      clearTimeout(gainTimeoutRef.current);
    }
    // Set new timeout to update after 500ms
    gainTimeoutRef.current = setTimeout(() => {
      updateGain(newValue);
    }, 500);
  }, [dispatch, updateGain]);

  // Handle crop frame parameter updates - memoized
  // Always send cropSize as integer to match backend API signature
  // Always send frameSize as [width, height] in NATURAL image coordinates for consistency
  const updateCropFrameParameters = useCallback(
    async (overrideCropCenter = null, overrideCropSize = null) => {
      try {
        // Always use natural image dimensions for consistent coordinate system
        let frameSize = focusLockUI.frameSize;
        const img = imgRef.current;
        if (img && img.naturalWidth && img.naturalHeight) {
          // Use natural dimensions to ensure coordinate system consistency
          frameSize = [img.naturalWidth, img.naturalHeight];
          // Update Redux with natural dimensions
          dispatch(focusLockSlice.setFrameSize(frameSize));
        }

        await apiFocusLockControllerSetCropFrameParameters({
          cropSize: Math.round(overrideCropSize ?? focusLockUI.cropSize),
          cropCenter: overrideCropCenter ?? focusLockUI.cropCenter,
          frameSize: frameSize,
        });
      } catch (error) {
        console.error("Failed to update crop frame parameters:", error);
      }
    },
    [
      focusLockUI.cropSize,
      focusLockUI.cropCenter,
      focusLockUI.frameSize,
      dispatch,
    ]
  );

  // Reset crop coordinates - memoized
  const resetCropCoordinates = useCallback(() => {
    dispatch(focusLockSlice.resetCropCenter());
    updateCropFrameParameters();
  }, [dispatch, updateCropFrameParameters]);

  // Toggle focus lock - memoized
  const toggleFocusLock = useCallback(async () => {
    try {
      await apiFocusLockControllerEnableFocusLock({
        enable: !focusLockUI.isFocusLocked,
      });
      dispatch(focusLockSlice.setFocusLocked(!focusLockUI.isFocusLocked));
    } catch (error) {
      console.error("Failed to toggle focus lock:", error);
    }
  }, [focusLockUI.isFocusLocked, dispatch]);

  // Start focus calibration - memoized
  const startCalibration = useCallback(async () => {
    try {
      dispatch(focusLockSlice.setIsCalibrating(true));
      setIsCalibrationPolling(true);
      // Use user-provided values for calibration
      await apiFocusLockControllerRunFocusCalibrationDynamic({
        scan_range_um: scanRangeUm,
        num_steps: numSteps,
        settle_time: settleTime,
      });
      // Start polling calibration status
      if (calibrationPollingRef.current)
        clearInterval(calibrationPollingRef.current);
      if (calibrationTimeoutRef.current)
        clearTimeout(calibrationTimeoutRef.current);
      let elapsed = 0;
      calibrationPollingRef.current = setInterval(async () => {
        try {
          const status = await apiFocusLockControllerGetCalibrationStatus();
          if (status && status.calibration_active === false) {
            dispatch(focusLockSlice.setIsCalibrating(false));
            setIsCalibrationPolling(false);
            clearInterval(calibrationPollingRef.current);
            calibrationPollingRef.current = null;
          }
        } catch (err) {
          // Ignore polling errors
        }
      }, 1000);
      calibrationTimeoutRef.current = setTimeout(() => {
        if (calibrationPollingRef.current) {
          clearInterval(calibrationPollingRef.current);
          calibrationPollingRef.current = null;
        }
        dispatch(focusLockSlice.setIsCalibrating(false));
        setIsCalibrationPolling(false);
      }, 10000);
    } catch (error) {
      console.error("Failed to start calibration:", error);
      dispatch(focusLockSlice.setIsCalibrating(false));
      setIsCalibrationPolling(false);
    }
  }, [dispatch, scanRangeUm, numSteps, settleTime]);

  // Unlock focus - memoized
  const unlockFocus = useCallback(async () => {
    try {
      await apiFocusLockControllerUnlockFocus();
      dispatch(focusLockSlice.setFocusLocked(false));
    } catch (error) {
      console.error("Failed to unlock focus:", error);
    }
  }, [dispatch]);

  // Get and store current focus value as target setpoint - memoized
  const storeCurrentFocusAsSetpoint = useCallback(async () => {
    try {
      const result = await apiFocusLockControllerGetCurrentFocusValue();
      const focusValue = result.focus_value;
      dispatch(focusLockSlice.setStoredTargetSetpoint(focusValue));
      // Also update the setPoint in Redux to update the yellow curve in the live plot
      dispatch(focusLockSlice.setSetPoint(focusValue));
      console.log(`Stored focus value as setpoint: ${focusValue}`);
    } catch (error) {
      console.error("Failed to get current focus value:", error);
    }
  }, [dispatch]);

  // Stop focus calibration - memoized
  const stopCalibration = useCallback(async () => {
    try {
      await apiFocusLockControllerStopFocusCalibration();
      dispatch(focusLockSlice.setIsCalibrating(false));
      setIsCalibrationPolling(false);
      if (calibrationPollingRef.current) {
        clearInterval(calibrationPollingRef.current);
        calibrationPollingRef.current = null;
      }
      if (calibrationTimeoutRef.current) {
        clearTimeout(calibrationTimeoutRef.current);
        calibrationTimeoutRef.current = null;
      }
      console.log("Focus calibration stopped");
    } catch (error) {
      console.error("Failed to stop calibration:", error);
    }
  }, [dispatch]);

  // Fetch and display calibration results - memoized
  const fetchCalibrationResults = useCallback(async () => {
    try {
      const results = await apiFocusLockControllerGetCalibrationResults();
      setCalibrationCurveData(results);
      setShowCalibrationCurve(true);
      console.log("Calibration results fetched:", results);
    } catch (error) {
      console.error("Failed to fetch calibration results:", error);
    }
  }, []);

  // Perform one-step autofocus - memoized
  const performOneStepAutofocus = useCallback(
    async (useManualSetpoint = false) => {
      try {
        dispatch(focusLockSlice.setIsPerformingAutofocus(true));

        const params = {
          move_to_focus: true,
          max_attempts: 3,
          threshold_um: 0.5,
        };

        // Use manual setpoint if provided and requested, otherwise use stored setpoint
        if (useManualSetpoint && manualTargetSetpoint !== "") {
          const manualValue = parseFloat(manualTargetSetpoint);
          if (!isNaN(manualValue)) {
            params.target_focus_setpoint = manualValue;
          }
        } else if (focusLockUI.storedTargetSetpoint !== null) {
          params.target_focus_setpoint = focusLockUI.storedTargetSetpoint;
        }

        const result = await apiFocusLockControllerPerformOneStepAutofocus(
          params
        );

        dispatch(focusLockSlice.setLastAutofocusResult(result));
        if (result.calibration_range) {
          dispatch(
            focusLockSlice.setCalibrationRange(result.calibration_range)
          );
        }

        console.log("One-step autofocus result:", result);
      } catch (error) {
        console.error("Failed to perform one-step autofocus:", error);
        dispatch(
          focusLockSlice.setLastAutofocusResult({
            success: false,
            error: error.message || "Unknown error",
          })
        );
      } finally {
        dispatch(focusLockSlice.setIsPerformingAutofocus(false));
      }
    },
    [dispatch, focusLockUI.storedTargetSetpoint, manualTargetSetpoint]
  );

  // Z-axis movement functions - memoized
  const moveZAxis = useCallback(async (distance) => {
    try {
      await apiPositionerControllerMovePositioner({
        positionerName: "ESP32Stage", // Default positioner name, might need adjustment
        axis: "Z",
        dist: distance,
        isAbsolute: false, // Relative movement
        isBlocking: false, // Non-blocking movement
        speed: 1000, // Default speed, might need adjustment
      });
    } catch (error) {
      console.error(`Failed to move Z-axis by ${distance}:`, error);
    }
  }, []);

  const moveZUp10 = useCallback(() => moveZAxis(10), [moveZAxis]);
  const moveZDown10 = useCallback(() => moveZAxis(-10), [moveZAxis]);
  const moveZUp100 = useCallback(() => moveZAxis(100), [moveZAxis]);
  const moveZDown100 = useCallback(() => moveZAxis(-100), [moveZAxis]);

  // Handle mouse events for crop selection on image - improved to prevent dragging and memoized
  const handleImageMouseDown = useCallback((e) => {
    e.preventDefault(); // Prevent default drag behavior

    const currentImage = e.target;
    if (!currentImage) return;

    // Always use the image's natural size for coordinates
    const rect = currentImage.getBoundingClientRect();
    const x =
      ((e.clientX - rect.left) * currentImage.naturalWidth) / rect.width;
    const y =
      ((e.clientY - rect.top) * currentImage.naturalHeight) / rect.height;

    setCropSelection({
      isSelecting: true,
      startX: x,
      startY: y,
      endX: x,
      endY: y,
    });
  }, []);

  const handleImageMouseMove = useCallback(
    (e) => {
      if (!cropSelection.isSelecting) return;
      e.preventDefault();

      const currentImage = e.target;
      const rect = currentImage.getBoundingClientRect();
      const x =
        ((e.clientX - rect.left) * currentImage.naturalWidth) / rect.width;
      const y =
        ((e.clientY - rect.top) * currentImage.naturalHeight) / rect.height;

      setCropSelection((prev) => ({
        ...prev,
        endX: x,
        endY: y,
      }));
    },
    [cropSelection.isSelecting]
  );

  const handleImageMouseUp = useCallback(
    (e) => {
      if (!cropSelection.isSelecting) return;
      e.preventDefault();

      const currentImage = e.target;
      const rect = currentImage.getBoundingClientRect();
      const x =
        ((e.clientX - rect.left) * currentImage.naturalWidth) / rect.width;
      const y =
        ((e.clientY - rect.top) * currentImage.naturalHeight) / rect.height;

      // Use the last mouse position as endX/endY
      const newCrop = {
        ...cropSelection,
        endX: x,
        endY: y,
      };

      const centerX = (newCrop.startX + newCrop.endX) / 2;
      const centerY = (newCrop.startY + newCrop.endY) / 2;
      const newCropCenter = [Math.round(centerX), Math.round(centerY)];
      const newCropSize = Math.round(
        Math.max(
          Math.abs(newCrop.endX - newCrop.startX),
          Math.abs(newCrop.endY - newCrop.startY)
        )
      );

      dispatch(focusLockSlice.setCropCenter(newCropCenter));
      dispatch(focusLockSlice.setSelectedCropRegion(newCrop));
      // update cropSize in focusLockSlice, always as integer
      dispatch(focusLockSlice.setCropSize(newCropSize));

      setCropSelection((prev) => ({
        ...prev,
        isSelecting: false,
      }));

      // Update crop parameters in backend with the new values directly to avoid race condition
      updateCropFrameParameters(newCropCenter, newCropSize);
    },
    [
      cropSelection.isSelecting,
      cropSelection.startX,
      cropSelection.endX,
      cropSelection.startY,
      cropSelection.endY,
      dispatch,
      updateCropFrameParameters,
    ]
  );

  // Memoize current image to prevent unnecessary recalculations
  const currentImage = useMemo(() => {
    return focusLockUI.lastImage; // Simplified since pollImageUrl not in split state
  }, [focusLockUI.lastImage]);

  return (
    <Paper style={{ padding: "24px", maxWidth: "100%", margin: "0 auto" }}>
      <Grid container spacing={3}>
        {/* Title */}
        <Grid item xs={12}>
          <Typography variant="h5" gutterBottom>
            Focus Lock Controller
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Hardware-based autofocus using astigmatism detection
          </Typography>
        </Grid>
        
        {/* Always visible: Focus measurement toggle and live graph */}
        <Grid item xs={12}>
          <Card>
            <CardHeader 
              title="Live Focus Monitoring" 
              action={
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={isMeasuring}
                        onChange={toggleFocusMeasurement}
                        color="primary"
                      />
                    }
                    label={`Measurement ${isMeasuring ? "ON" : "OFF"}`}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={continuousImageLoading}
                        onChange={(e) => setContinuousImageLoading(e.target.checked)}
                        color="secondary"
                      />
                    }
                    label={`Image Polling ${continuousImageLoading ? "ON" : "OFF"}`}
                  />
                  <Typography variant="body2" sx={{ minWidth: 150 }}>
                    Focus: {focusLockUI.currentFocusValue.toFixed(3)}
                  </Typography>
                </Box>
              }
            />
            <CardContent>
              <Box sx={{ height: 300, position: "relative" }}>
                <Line data={chartData} options={chartOptions} />
                {chartDataSource.focusValues.length === 0 && (
                  <Box
                    sx={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "2px dashed #ccc",
                      borderRadius: 1,
                      flexDirection: "column",
                      gap: 1,
                      backgroundColor: "rgba(255,255,255,0.7)",
                      pointerEvents: "none",
                    }}
                  >
                    <Typography variant="h6" color="textSecondary">
                      Real-time Focus & Motor Position Chart
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Enable measurement to see live data
                    </Typography>
                  </Box>
                )}
              </Box>
              <Button
                size="small"
                onClick={() => dispatch(focusLockSlice.clearFocusHistory())}
                sx={{ mt: 1 }}
                variant="outlined"
                color="secondary"
              >
                Clear Graph Data
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Camera streams - always visible side by side */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Focus Lock Camera"
              subheader="Astigmatism detection camera"
            />
            <CardContent>
              <Box sx={{ position: "relative", minHeight: 300 }}>
                {currentImage ? (
                  <Box
                    sx={{
                      position: "relative",
                      display: "inline-block",
                      width: "100%",
                    }}
                  >
                    <img
                      ref={imgRef}
                      src={currentImage}
                      alt="Focus lock camera view"
                      style={{
                        width: "100%",
                        height: "auto",
                        maxHeight: "400px",
                        cursor: "crosshair",
                        border: "1px solid #ccc",
                        userSelect: "none",
                        objectFit: "contain",
                      }}
                      draggable={false}
                      onMouseDown={handleImageMouseDown}
                      onMouseMove={handleImageMouseMove}
                      onMouseUp={handleImageMouseUp}
                      onDragStart={(e) => e.preventDefault()}
                      onLoad={(e) => {
                        const img = e.target;
                        if (img.naturalWidth && img.naturalHeight) {
                          dispatch(
                            focusLockSlice.setFrameSize([
                              img.naturalWidth,
                              img.naturalHeight,
                            ])
                          );
                        }
                      }}
                    />
                    {/* Crop overlays */}
                    {(() => {
                      const img = imgRef.current;
                      if (!img) return null;
                      const dispW = img.clientWidth || img.width || img.naturalWidth;
                      const dispH = img.clientHeight || img.height || img.naturalHeight;
                      const natW = img.naturalWidth;
                      const natH = img.naturalHeight;
                      if (!natW || !natH) return null;
                      const scaleX = dispW / natW;
                      const scaleY = dispH / natH;

                      const toDisplayX = (x) => x * scaleX;
                      const toDisplayY = (y) => y * scaleY;

                      let cropOverlay = null;
                      if (cropSelection.isSelecting) {
                        const left = toDisplayX(Math.min(cropSelection.startX, cropSelection.endX));
                        const top = toDisplayY(Math.min(cropSelection.startY, cropSelection.endY));
                        const width = Math.abs(toDisplayX(cropSelection.endX) - toDisplayX(cropSelection.startX));
                        const height = Math.abs(toDisplayY(cropSelection.endY) - toDisplayY(cropSelection.startY));
                        cropOverlay = (
                          <div
                            style={{
                              position: "absolute",
                              left,
                              top,
                              width,
                              height,
                              border: "2px dashed red",
                              backgroundColor: "rgba(255, 0, 0, 0.1)",
                              pointerEvents: "none",
                            }}
                          />
                        );
                      }

                      const centerX = toDisplayX(focusLockUI.cropCenter[0]);
                      const centerY = toDisplayY(focusLockUI.cropCenter[1]);
                      const cropCenterOverlay = (
                        <div
                          style={{
                            position: "absolute",
                            left: centerX - 5,
                            top: centerY - 5,
                            width: 10,
                            height: 10,
                            backgroundColor: "red",
                            borderRadius: "50%",
                            pointerEvents: "none",
                          }}
                        />
                      );

                      const cropSize = focusLockUI.cropSize;
                      const cropSizeOverlay = (
                        <div
                          style={{
                            position: "absolute",
                            left: centerX - (cropSize * scaleX) / 2,
                            top: centerY - (cropSize * scaleY) / 2,
                            width: cropSize * scaleX,
                            height: cropSize * scaleY,
                            border: "2px solid blue",
                            backgroundColor: "rgba(0, 0, 255, 0.1)",
                            pointerEvents: "none",
                          }}
                        />
                      );
                      return (
                        <>
                          {cropOverlay}
                          {cropCenterOverlay}
                          {cropSizeOverlay}
                        </>
                      );
                    })()}
                  </Box>
                ) : (
                  <Box
                    sx={{
                      height: 300,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "2px dashed #ccc",
                      borderRadius: 1,
                      flexDirection: "column",
                      gap: 2,
                    }}
                  >
                    <Typography variant="body2" color="textSecondary">
                      No image loaded
                    </Typography>
                    <Button
                      variant="outlined"
                      onClick={loadLastImage}
                      disabled={focusLockUI.isLoadingImage}
                    >
                      {focusLockUI.isLoadingImage ? "Loading..." : "Load Image"}
                    </Button>
                  </Box>
                )}
              </Box>
              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={loadLastImage}
                  disabled={focusLockUI.isLoadingImage}
                  fullWidth
                >
                  {focusLockUI.isLoadingImage ? "Loading..." : "Refresh Image"}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Main Camera"
              subheader="Microscope imaging camera"
            />
            <CardContent>
              <Box sx={{ height: 300, position: "relative" }}>
                <LiveViewControlWrapper
                  useFastMode={true}
                  enableStageMovement={false}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Main Tabbed Controls */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Focus Control Modes" />
            <CardContent>
              <Tabs
                value={mainTabValue}
                onChange={(e, newValue) => setMainTabValue(newValue)}
                variant="fullWidth"
                sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}
              >
                <Tab label="Calibration" />
                <Tab label="Continuous Focus Lock" />
                <Tab label="One-Step Autofocus" />
              </Tabs>

              {/* Tab 0: Calibration */}
              {mainTabValue === 0 && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <Typography variant="body2" color="textSecondary">
                    Run a Z-scan to calibrate the focus signal vs. position relationship
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <TextField
                        label="Scan Range (µm)"
                        type="number"
                        value={scanRangeUm}
                        onChange={(e) => setScanRangeUm(Number(e.target.value))}
                        size="small"
                        fullWidth
                        inputProps={{ step: 1, min: 100, max: 10000 }}
                      />
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <TextField
                        label="Number of Steps"
                        type="number"
                        value={numSteps}
                        onChange={(e) => setNumSteps(Number(e.target.value))}
                        size="small"
                        fullWidth
                        inputProps={{ step: 1, min: 2, max: 100 }}
                      />
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <TextField
                        label="Settle Time (s)"
                        type="number"
                        value={settleTime}
                        onChange={(e) => setSettleTime(Number(e.target.value))}
                        size="small"
                        fullWidth
                        inputProps={{ step: 0.1, min: 0.1, max: 10 }}
                      />
                    </Grid>
                  </Grid>

                  <Box sx={{ display: "flex", gap: 2, mt: 2 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={startCalibration}
                      disabled={focusLockUI.isCalibrating || isCalibrationPolling}
                      sx={{ flex: 1 }}
                    >
                      {focusLockUI.isCalibrating || isCalibrationPolling
                        ? "Calibrating..."
                        : "Start Calibration"}
                    </Button>

                    <Button
                      variant="outlined"
                      color="error"
                      onClick={stopCalibration}
                      disabled={!focusLockUI.isCalibrating && !isCalibrationPolling}
                      sx={{ flex: 1 }}
                    >
                      Stop Calibration
                    </Button>

                    <Button
                      variant="outlined"
                      color="info"
                      onClick={fetchCalibrationResults}
                      sx={{ flex: 1 }}
                    >
                      Show Calibration Curve
                    </Button>
                  </Box>

                  {calibrationResults && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" fontWeight="bold">
                        Calibration Results
                      </Typography>
                      <Typography variant="body2">
                        R²: {calibrationResults.r_squared?.toFixed(4)} | 
                        Sensitivity: {calibrationResults.sensitivity_nm_per_px?.toFixed(1)} nm/unit | 
                        Monotone Failures: {calibrationResults.monotone_failures}/{calibrationResults.total_points}
                      </Typography>
                      {calibrationResults.coefficients && (
                        <Typography variant="caption">
                          Coefficients: [{calibrationResults.coefficients.map((c) => c.toFixed(4)).join(", ")}]
                        </Typography>
                      )}
                    </Alert>
                  )}
                </Box>
              )}

              {/* Tab 1: Continuous Focus Lock */}
              {mainTabValue === 1 && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  <Typography variant="body2" color="textSecondary">
                    Maintain focus continuously using PID controller
                  </Typography>

                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={focusLockUI.isFocusLocked}
                          onChange={toggleFocusLock}
                          color="primary"
                        />
                      }
                      label={`Focus Lock ${focusLockUI.isFocusLocked ? "ENABLED" : "DISABLED"}`}
                    />
                    
                    <Button
                      variant="outlined"
                      onClick={unlockFocus}
                      disabled={!focusLockUI.isFocusLocked}
                    >
                      Unlock Focus
                    </Button>
                  </Box>

                  <Divider />

                  {/* PID Parameters */}
                  <Typography variant="subtitle2" fontWeight="bold">
                    PID Controller Parameters
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        Kp (Proportional): {focusLockUI.kp.toFixed(2)}
                      </Typography>
                      <Slider
                        value={focusLockUI.kp}
                        onChange={(e, value) => dispatch(focusLockSlice.setKp(value))}
                        min={0}
                        max={2}
                        step={0.01}
                        valueLabelDisplay="auto"
                      />
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <Typography gutterBottom>
                        Ki (Integral): {focusLockUI.ki.toFixed(2)}
                      </Typography>
                      <Slider
                        value={focusLockUI.ki}
                        onChange={(e, value) => dispatch(focusLockSlice.setKi(value))}
                        min={0}
                        max={2}
                        step={0.01}
                        valueLabelDisplay="auto"
                      />
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Set Point"
                        type="number"
                        value={focusLockUI.setPoint}
                        onChange={(e) =>
                          dispatch(focusLockSlice.setSetPoint(parseFloat(e.target.value) || 0))
                        }
                        size="small"
                        fullWidth
                        inputProps={{ step: 0.1 }}
                      />
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Min Step Threshold"
                        type="number"
                        value={focusLockUI.minStepThreshold}
                        onChange={(e) =>
                          dispatch(focusLockSlice.setMinStepThreshold(parseFloat(e.target.value) || 0))
                        }
                        size="small"
                        fullWidth
                        inputProps={{ step: 0.001, min: 0 }}
                      />
                    </Grid>
                  </Grid>

                  <Button
                    variant="contained"
                    onClick={updatePIControllerParameters}
                    sx={{ mt: 2 }}
                  >
                    Update PID Parameters
                  </Button>
                </Box>
              )}

              {/* Tab 2: One-Step Autofocus */}
              {mainTabValue === 2 && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <Typography variant="body2" color="textSecondary">
                    Move to a stored or manual focus setpoint in one step
                  </Typography>

                  <Button
                    variant="contained"
                    color="secondary"
                    onClick={storeCurrentFocusAsSetpoint}
                    disabled={!isMeasuring}
                    startIcon={<span>💾</span>}
                  >
                    Store Current Focus Value as Setpoint
                  </Button>

                  {focusLockUI.storedTargetSetpoint !== null && (
                    <Alert severity="info">
                      <Typography variant="body2">
                        <strong>Stored Setpoint:</strong> {focusLockUI.storedTargetSetpoint.toFixed(3)}
                      </Typography>
                    </Alert>
                  )}

                  <TextField
                    label="Manual Target Setpoint"
                    type="number"
                    value={manualTargetSetpoint}
                    onChange={(e) => setManualTargetSetpoint(e.target.value)}
                    size="small"
                    fullWidth
                    helperText="Or enter a target focus value manually"
                  />

                  <Box sx={{ display: "flex", gap: 2, mt: 2 }}>
                    <Button
                      variant="contained"
                      color="success"
                      onClick={() => performOneStepAutofocus(false)}
                      disabled={
                        focusLockUI.isPerformingAutofocus ||
                        focusLockUI.storedTargetSetpoint === null
                      }
                      sx={{ flex: 1 }}
                    >
                      {focusLockUI.isPerformingAutofocus
                        ? "Focusing..."
                        : "Use Stored Setpoint"}
                    </Button>

                    <Button
                      variant="contained"
                      color="info"
                      onClick={() => performOneStepAutofocus(true)}
                      disabled={
                        focusLockUI.isPerformingAutofocus ||
                        manualTargetSetpoint === ""
                      }
                      sx={{ flex: 1 }}
                    >
                      {focusLockUI.isPerformingAutofocus
                        ? "Focusing..."
                        : "Use Manual Setpoint"}
                    </Button>
                  </Box>

                  {focusLockUI.lastAutofocusResult && (
                    <Alert
                      severity={
                        focusLockUI.lastAutofocusResult.success
                          ? "success"
                          : "error"
                      }
                      sx={{ mt: 2 }}
                    >
                      {focusLockUI.lastAutofocusResult.success ? (
                        <Box>
                          <Typography variant="body2" fontWeight="bold">
                            ✓ Autofocus Successful
                          </Typography>
                          <Typography variant="body2">
                            Attempts: {focusLockUI.lastAutofocusResult.num_attempts} | 
                            Error: {focusLockUI.lastAutofocusResult.final_error_um?.toFixed(2)} µm | 
                            Z Offset: {focusLockUI.lastAutofocusResult.z_offset?.toFixed(2)} µm
                          </Typography>
                        </Box>
                      ) : (
                        <Typography variant="body2">
                          {focusLockUI.lastAutofocusResult.error || "Autofocus failed"}
                        </Typography>
                      )}
                    </Alert>
                  )}

                  {focusLockUI.calibrationRange && (
                    <Typography variant="caption" color="textSecondary">
                      Calibration Range: [{focusLockUI.calibrationRange[0]?.toFixed(1)}, {focusLockUI.calibrationRange[1]?.toFixed(1)}] µm
                    </Typography>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>



        {/* Camera Parameters & Z-Axis Controls */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Camera Settings" />
            <CardContent>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <TextField
                  label="Exposure Time (ms)"
                  type="number"
                  value={focusLockUI.exposureTime}
                  onChange={(e) => handleExposureChange(parseFloat(e.target.value) || 0)}
                  size="small"
                  fullWidth
                  inputProps={{ step: 1, min: 1, max: 10000 }}
                  helperText="Auto-saves after typing"
                />

                <TextField
                  label="Gain"
                  type="number"
                  value={focusLockUI.gain}
                  onChange={(e) => handleGainChange(parseFloat(e.target.value) || 0)}
                  size="small"
                  fullWidth
                  inputProps={{ step: 0.1, min: 0, max: 100 }}
                  helperText="Auto-saves after typing"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Z-Axis Manual Control" />
            <CardContent>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Manual focus positioning
                </Typography>

                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Fine (±10µm)
                  </Typography>
                  <ButtonGroup fullWidth variant="contained">
                    <Button onClick={moveZDown10} color="primary">
                      ↓ 10µm
                    </Button>
                    <Button onClick={moveZUp10} color="primary">
                      ↑ 10µm
                    </Button>
                  </ButtonGroup>
                </Box>

                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Coarse (±100µm)
                  </Typography>
                  <ButtonGroup fullWidth variant="contained">
                    <Button onClick={moveZDown100} color="secondary">
                      ↓ 100µm
                    </Button>
                    <Button onClick={moveZUp100} color="secondary">
                      ↑ 100µm
                    </Button>
                  </ButtonGroup>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Developer Mode - Advanced Parameters */}
        <Grid item xs={12}>
          <Accordion expanded={devModeOpen} onChange={() => setDevModeOpen(!devModeOpen)}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">🔧 Developer Mode - Advanced Parameters</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                {/* Astigmatism Parameters */}
                <Grid item xs={12} md={6}>
                  <Card variant="outlined">
                    <CardHeader title="Astigmatism Detection Parameters" />
                    <CardContent>
                      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                        <TextField
                          label="Gaussian Sigma"
                          type="number"
                          value={focusLockUI.gaussian_sigma}
                          onChange={(e) =>
                            dispatch(focusLockSlice.setgaussian_sigma(parseFloat(e.target.value)))
                          }
                          size="small"
                          fullWidth
                          inputProps={{ step: 0.1, min: 0.1, max: 10 }}
                        />

                        <TextField
                          label="Background Threshold"
                          type="number"
                          value={focusLockUI.background_threshold}
                          onChange={(e) =>
                            dispatch(focusLockSlice.setBackgroundThreshold(parseFloat(e.target.value)))
                          }
                          size="small"
                          fullWidth
                          inputProps={{ step: 1, min: 0, max: 1000 }}
                        />

                        <TextField
                          label="Crop Size (pixels)"
                          type="number"
                          value={focusLockUI.cropSize}
                          onChange={(e) =>
                            dispatch(focusLockSlice.setCropSize(parseInt(e.target.value)))
                          }
                          size="small"
                          fullWidth
                          inputProps={{ step: 1, min: 10, max: 500 }}
                        />

                        <Typography variant="body2">
                          Crop Center: [{focusLockUI.cropCenter[0]}, {focusLockUI.cropCenter[1]}]
                        </Typography>

                        <Typography variant="body2" color="textSecondary">
                          Frame Size: [{focusLockUI.frameSize[0]}, {focusLockUI.frameSize[1]}]
                        </Typography>

                        <Box sx={{ display: "flex", gap: 1 }}>
                          <Button
                            variant="contained"
                            onClick={updateAstigmatismParameters}
                            sx={{ flex: 1 }}
                          >
                            Update Parameters
                          </Button>

                          <Button
                            variant="outlined"
                            onClick={resetCropCoordinates}
                            sx={{ flex: 1 }}
                          >
                            Reset Crop
                          </Button>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Extended PID Parameters */}
                <Grid item xs={12} md={6}>
                  <Card variant="outlined">
                    <CardHeader title="Extended PID Controller Parameters" />
                    <CardContent>
                      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                        <TextField
                          label="Safety Distance Limit"
                          type="number"
                          value={focusLockUI.safetyDistanceLimit}
                          onChange={(e) =>
                            dispatch(focusLockSlice.setSafetyDistanceLimit(parseFloat(e.target.value) || 0))
                          }
                          size="small"
                          fullWidth
                          inputProps={{ step: 1, min: 0 }}
                        />

                        <TextField
                          label="Safety Move Limit"
                          type="number"
                          value={focusLockUI.safetyMoveLimit}
                          onChange={(e) =>
                            dispatch(focusLockSlice.setSafetyMoveLimit(parseFloat(e.target.value) || 0))
                          }
                          size="small"
                          fullWidth
                          inputProps={{ step: 0.1, min: 0 }}
                        />

                        <FormControlLabel
                          control={
                            <Switch
                              checked={focusLockUI.safetyMotionActive}
                              onChange={(e) =>
                                dispatch(focusLockSlice.setSafetyMotionActive(e.target.checked))
                              }
                            />
                          }
                          label="Safety Motion Active"
                        />

                        <Button
                          variant="contained"
                          onClick={updatePIControllerParameters}
                        >
                          Update Extended PID Parameters
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Calibration Curve Display */}
        {showCalibrationCurve && calibrationCurveData && (
          <Grid item xs={12}>
            <Card>
              <CardHeader
                title="Calibration Curve (Position vs Focus)"
                subheader={`Sensitivity: ${calibrationCurveData.sensitivity_nm_per_px?.toFixed(1)} nm/unit | R²: ${calibrationCurveData.r_squared?.toFixed(4)}`}
                action={
                  <Button
                    size="small"
                    onClick={() => setShowCalibrationCurve(false)}
                    variant="outlined"
                  >
                    Close
                  </Button>
                }
              />
              <CardContent>
                <Box sx={{ height: 300 }}>
                  <Line
                    data={{
                      labels: calibrationCurveData.positionData || [],
                      datasets: [
                        {
                          label: "Focus Signal",
                          data: (calibrationCurveData.positionData || []).map((pos, idx) => ({
                            x: pos,
                            y: (calibrationCurveData.signalData || calibrationCurveData.calibration_data?.focus_data || [])[idx],
                          })),
                          borderColor: theme.palette.primary.main,
                          backgroundColor: theme.palette.primary.light,
                          pointRadius: 4,
                          pointHoverRadius: 6,
                          showLine: true,
                          tension: 0.1,
                        },
                        // Linear fit line
                        ...(calibrationCurveData.poly ? [{
                          label: `Linear Fit (slope: ${calibrationCurveData.poly[0]?.toFixed(4)})`,
                          data: (calibrationCurveData.positionData || []).map((pos) => ({
                            x: pos,
                            y: calibrationCurveData.poly[0] * pos + calibrationCurveData.poly[1],
                          })),
                          borderColor: theme.palette.secondary.main,
                          borderDash: [5, 5],
                          pointRadius: 0,
                          showLine: true,
                        }] : []),
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        x: {
                          type: "linear",
                          title: { display: true, text: "Z Position (µm)" },
                        },
                        y: {
                          type: "linear",
                          title: { display: true, text: "Focus Signal" },
                        },
                      },
                      plugins: {
                        legend: { position: "top" },
                        tooltip: {
                          callbacks: {
                            label: (ctx) => `Focus: ${ctx.parsed.y.toFixed(2)} @ Z: ${ctx.parsed.x.toFixed(1)} µm`,
                          },
                        },
                      },
                    }}
                  />
                </Box>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    <strong>Polynomial Coefficients:</strong> [{calibrationCurveData.poly?.map(c => c.toFixed(4)).join(", ")}]
                  </Typography>
                  <Typography variant="body2">
                    <strong>Linear Range:</strong> [{calibrationCurveData.calibration_data?.linear_range?.[0]?.toFixed(1)}, {calibrationCurveData.calibration_data?.linear_range?.[1]?.toFixed(1)}]
                  </Typography>
                  <Typography variant="body2">
                    <strong>PID Integration Active:</strong> {calibrationCurveData.pid_integration_active ? "Yes" : "No"}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}


      </Grid>
    </Paper>
  );
};

export default FocusLockController;
