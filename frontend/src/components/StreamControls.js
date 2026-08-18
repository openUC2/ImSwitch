import React, { useState, useEffect, useCallback } from "react";

import {
  Box,
  TextField,
  Typography,
  Button,
  FormControl,
  FormHelperText,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Tooltip,
  Switch,
  FormControlLabel,
  CircularProgress,
} from "@mui/material";
import {
  PlayArrow,
  Stop,
  CameraAlt,
  FiberManualRecord,
  Stop as StopIcon,
  Videocam,
  VideoLibrary,
  Settings,
} from "@mui/icons-material";
import StreamControlOverlay from "../components/StreamControlOverlay";
import StreamPresets from "./StreamPresets";
import apiViewControllerGetLiveViewActive from "../backendapi/apiViewControllerGetLiveViewActive";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner";
import { useSelector, useDispatch } from "react-redux";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";

export default function StreamControls({
  isStreamRunning, // This prop is kept for backwards compatibility but we prefer Redux state
  isLongExposure = false,
  exposureMs = 0,
  longExposureThresholdMs = 2000,
  onToggleStream,
  onSnap,
  onSnapAndDownload,
  isRecording,
  onStartRecord,
  onStopRecord,
  onStopRecordAndDownload,
  onRecordAndDownload,
  onCancelSnap,
  onGoToFolder,
  lastCapturePath,
}) {
  const dispatch = useDispatch();

  // Redux state
  const showHistogram = useSelector(
    (state) => state.liveStreamState.showHistogram,
  );

  // Internal state for file name
  // Default is empty - detector name is now automatically included in timestamp-based filename
  const [snapFileName, setSnapFileName] = useState("");
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [fijiInfoOpen, setFijiInfoOpen] = useState(false);
  // Track an in-flight snap so we can disable the buttons and show progress.
  // Snapping is allowed even when the live stream is off (e.g. long-exposure
  // experiments), in which case the request can take a while to return.
  const [isSnapping, setIsSnapping] = useState(false);
  // Timed record: clip length in seconds, and the countdown shown on the button
  // while the clip is being acquired. 0 = record until the user stops.
  const [clipSeconds, setClipSeconds] = useState(0);
  const [clipRemaining, setClipRemaining] = useState(0);
  // "Download when finished" replaces the separate *& Download* buttons: the
  // download is a property of the capture, not a different kind of capture, so
  // it belongs on a toggle rather than doubling the button count.
  const [downloadSnap, setDownloadSnap] = useState(false);
  const [downloadRecording, setDownloadRecording] = useState(false);

  // Wrap snap to track the in-flight state so the button shows progress and is
  // guarded against double-clicks. Works whether or not the live stream is
  // running (the backend arms the camera on demand for long-exposure snaps).
  const handleSnap = useCallback(
    async (fileName, format) => {
      if (isSnapping) return;
      setIsSnapping(true);
      try {
        await onSnap(fileName, format);
      } finally {
        setIsSnapping(false);
      }
    },
    [isSnapping, onSnap],
  );

  // Wrap snap & download to show a one-time Fiji hint for TIFF files, and to
  // track the in-flight state (see handleSnap).
  const handleSnapAndDownload = useCallback(
    async (fileName, format) => {
      if (isSnapping) return;
      // Show the hint only on the very first TIFF snap & download
      if (format === 1 && !localStorage.getItem("fijiHintShown")) {
        setFijiInfoOpen(true);
        localStorage.setItem("fijiHintShown", "1");
      }
      setIsSnapping(true);
      try {
        await onSnapAndDownload(fileName, format);
      } finally {
        setIsSnapping(false);
      }
    },
    [isSnapping, onSnapAndDownload],
  );

  // Separate format options for snap and record
  const snapFormatOptions = [
    { value: 1, label: "TIFF (Default)" },
    { value: 5, label: "PNG" },
    { value: 6, label: "JPG" },
  ];

  const recordFormatOptions = [
    { value: 4, label: "MP4 (Default)" },
    { value: 1, label: "TIFF" },
    { value: 3, label: "ZARR" },
  ];

  // Get stream stats from Redux (includes fps which indicates active frames)
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);

  const snapFormat = liveViewState.snapFormat || 1;
  const recordFormat = liveViewState.recordFormat || 4;

  // Use Redux state as source of truth for stream status
  const isLiveViewActive = liveViewState.isStreamRunning;
  // Set as soon as a cancel request goes out, so both Cancel buttons (here and
  // on the image overlay) disable together.
  const snapCancelling = liveViewState.snapCancelling;

  // One Snap button: the format select and the download toggle decide what
  // actually happens, so the user is not choosing between near-identical
  // buttons on every capture.
  const handleSnapClick = useCallback(() => {
    if (downloadSnap) {
      handleSnapAndDownload(snapFileName, snapFormat);
    } else {
      handleSnap(snapFileName, snapFormat);
    }
  }, [
    downloadSnap,
    handleSnap,
    handleSnapAndDownload,
    snapFileName,
    snapFormat,
  ]);

  // Record a fixed-length clip and download it in one click (the video
  // counterpart of Snap & Download). The countdown doubles as the busy state.
  const handleRecordAndDownload = useCallback(async () => {
    if (clipRemaining > 0 || !onRecordAndDownload) return;
    const seconds = Math.max(1, Number(clipSeconds) || 5);
    setClipRemaining(seconds);
    const ticker = setInterval(
      () => setClipRemaining((s) => (s > 1 ? s - 1 : 0)),
      1000,
    );
    try {
      await onRecordAndDownload(snapFileName, recordFormat, seconds);
    } finally {
      clearInterval(ticker);
      setClipRemaining(0);
    }
  }, [
    clipRemaining,
    clipSeconds,
    onRecordAndDownload,
    snapFileName,
    recordFormat,
  ]);

  // One Record button covering all four combinations that used to need three
  // buttons plus a clip-length field: manual vs. auto-stop, download vs. not.
  const handleRecordClick = useCallback(() => {
    if (isRecording) {
      if (downloadRecording) {
        onStopRecordAndDownload();
      } else {
        onStopRecord();
      }
      return;
    }
    const seconds = Number(clipSeconds) || 0;
    if (seconds > 0 && downloadRecording) {
      // Timed clip that downloads itself; the backend-side stop is handled by
      // the timed helper, which is the only path that also downloads.
      handleRecordAndDownload();
      return;
    }
    if (seconds > 0) {
      // Timed clip without download: start now, stop ourselves after `seconds`.
      onStartRecord(snapFileName, recordFormat);
      setClipRemaining(seconds);
      const ticker = setInterval(
        () => setClipRemaining((s) => (s > 1 ? s - 1 : 0)),
        1000,
      );
      setTimeout(() => {
        clearInterval(ticker);
        setClipRemaining(0);
        onStopRecord();
      }, seconds * 1000);
      return;
    }
    onStartRecord(snapFileName, recordFormat);
  }, [
    isRecording,
    downloadRecording,
    clipSeconds,
    handleRecordAndDownload,
    onStartRecord,
    onStopRecord,
    onStopRecordAndDownload,
    snapFileName,
    recordFormat,
  ]);

  // Ensure defaults are set on component mount
  useEffect(() => {
    if (!liveViewState.snapFormat) {
      dispatch(liveViewSlice.setSnapFormat(1));
    }
    if (!liveViewState.recordFormat) {
      dispatch(liveViewSlice.setRecordFormat(4));
    }
  }, [dispatch, liveViewState.snapFormat, liveViewState.recordFormat]);

  // State for HUD data for overlay display
  const [hudData, setHudData] = useState({
    stats: { fps: 0, bps: 0 },
    featureSupport: { webgl2: false, lz4: false },
    isWebGL: false,
    imageSize: { width: 0, height: 0 },
    viewTransform: { scale: 1, translateX: 0, translateY: 0 },
  });

  // Sync hudData stats with Redux stats for overlay display
  useEffect(() => {
    setHudData((prevData) => ({
      ...prevData,
      stats: {
        fps: liveStreamState.stats.fps || 0,
        bps: liveStreamState.stats.bps || 0,
      },
    }));
  }, [liveStreamState.stats.fps, liveStreamState.stats.bps]);

  // Periodic status check to keep Redux in sync with backend
  // This catches cases where backend state changes without frontend knowledge
  const checkLiveViewStatus = useCallback(async () => {
    try {
      const active = await apiViewControllerGetLiveViewActive();

      // Only update Redux if state differs from backend
      if (active !== liveViewState.isStreamRunning) {
        console.log(
          `[StreamControls] Backend status mismatch detected. Backend: ${active}, Frontend: ${liveViewState.isStreamRunning}`,
        );
        dispatch(liveViewSlice.setIsStreamRunning(active));
      }
    } catch (error) {
      console.warn("[StreamControls] Failed to check live view status:", error);
    }
  }, [liveViewState.isStreamRunning, dispatch]);

  // Periodic sync every 5 seconds to catch any state drift
  useEffect(() => {
    const interval = setInterval(checkLiveViewStatus, 5000);
    return () => clearInterval(interval);
  }, [checkLiveViewStatus]);

  // Move Z-axis handler
  const moveZAxis = useCallback((distance) => {
    apiPositionerControllerMovePositioner({
      axis: "Z",
      dist: distance,
      isAbsolute: false,
    })
      .then((response) => {
        console.log(`Moved Z-axis by ${distance}:`, response);
      })
      .catch((error) => {
        console.error(`Error moving Z-axis by ${distance}:`, error);
      });
  }, []);

  // Keyboard event handler for Z-axis control
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Only handle if not typing in an input field
      if (
        event.target.tagName === "INPUT" ||
        event.target.tagName === "TEXTAREA"
      ) {
        return;
      }

      // Don't intercept browser zoom shortcuts (Cmd/Ctrl + Plus/Minus)
      if (
        (event.metaKey || event.ctrlKey) &&
        (event.key === "+" || event.key === "-" || event.key === "=")
      ) {
        return; // Let browser handle zoom
      }

      switch (event.key) {
        case "+":
        case "=": // Also handle = key (same physical key as + without shift)
          event.preventDefault();
          moveZAxis(10); // Move up 10 steps
          break;
        case "-":
        case "_": // Also handle _ key (same physical key as - with shift)
          event.preventDefault();
          moveZAxis(-10); // Move down 10 steps
          break;
        case ".":
        case ">": // Also handle > key (same physical key as . with shift)
          event.preventDefault();
          moveZAxis(100); // Move up 100 steps
          break;
        case ",":
        case "<": // Also handle < key (same physical key as , with shift)
          event.preventDefault();
          moveZAxis(-100); // Move down 100 steps
          break;
        default:
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [moveZAxis]);

  // Handle start stream
  const handleStartStream = useCallback(async () => {
    if (!isLiveViewActive) {
      await onToggleStream();
      // Status will be updated by the toggleStream function in LiveView.js
    }
  }, [isLiveViewActive, onToggleStream]);

  // Handle stop stream
  const handleStopStream = useCallback(async () => {
    if (isLiveViewActive) {
      await onToggleStream();
      // Status will be updated by the toggleStream function in LiveView.js
    }
  }, [isLiveViewActive, onToggleStream]);

  // Render stream controls with editable image name and icon buttons
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        width: "100%",
      }}
    >
      {/* Stream Control Section */}
      <Box
        component="fieldset"
        data-tour="stream-controls"
        sx={{
          border: 1,
          borderColor: "divider",
          borderRadius: 1,
          p: 2,
          display: "flex",
          gap: 1,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <Box
          component="legend"
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.5,
            px: 1,
          }}
        >
          <VideoLibrary fontSize="small" sx={{ color: "text.secondary" }} />
          <Typography variant="subtitle1" sx={{ fontWeight: "medium" }}>
            Stream
          </Typography>
        </Box>

        <Tooltip
          arrow
          title={
            isLongExposure && !isLiveViewActive
              ? `Exposure is ${Math.round(exposureMs)} ms (> ${longExposureThresholdMs} ms). ` +
                `A live stream would deliver a frame slower than it times out — use Snap instead. ` +
                `Starting anyway is allowed but will be very slow.`
              : isLiveViewActive
                ? "Stop the live stream"
                : "Start the live stream"
          }
        >
          <span>
            <Button
              variant="contained"
              color={
                isLiveViewActive
                  ? "error"
                  : isLongExposure
                    ? "warning"
                    : "success"
              }
              size="small"
              onClick={isLiveViewActive ? handleStopStream : handleStartStream}
              startIcon={isLiveViewActive ? <Stop /> : <PlayArrow />}
              sx={{ width: 150, whiteSpace: "nowrap" }}
            >
              {isLiveViewActive
                ? "Stop stream"
                : isLongExposure
                  ? "Start anyway"
                  : "Start stream"}
            </Button>
          </span>
        </Tooltip>

        <FormControlLabel
          control={
            <Switch
              checked={showHistogram}
              onChange={(e) =>
                dispatch(liveStreamSlice.setShowHistogram(e.target.checked))
              }
              size="small"
              color="success"
            />
          }
          label="Histogram"
          sx={{ ml: 1 }}
        />

        <Button
          variant="outlined"
          size="small"
          onClick={() => setOverlayOpen(true)}
          sx={{ ml: "auto" }}
          startIcon={<Settings />}
          data-tour="stream-settings-button"
        >
          Settings
        </Button>
      </Box>

      {/* Stream presets / macros — recall named bundles of objective +
          exposure + gain + livestream parameters. Frontend-only. */}
      <StreamPresets />

      {/* Recording Controls Section */}
      <Box
        component="fieldset"
        data-tour="capture-controls"
        sx={{
          border: 1,
          borderColor: "divider",
          borderRadius: 1,
          p: 2,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 2,
        }}
      >
        <Box
          component="legend"
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.5,
            px: 1,
            gridColumn: "1 / -1",
          }}
        >
          <Videocam fontSize="small" sx={{ color: "text.secondary" }} />
          <Typography variant="subtitle1" sx={{ fontWeight: "medium" }}>
            Capture
          </Typography>
        </Box>
        {/* Common Description field - in the capture box, above Snap and Record */}
        <FormControl fullWidth size="small" sx={{ gridColumn: "1 / -1" }}>
          <TextField
            label="Description (Snap & Recording)"
            size="small"
            placeholder="e.g., sample-001, z-stack-start"
            value={snapFileName}
            onChange={(e) => {
              // Only allow alphanumeric, underscore, hyphen, and dot
              const sanitized = e.target.value.replace(/[^a-zA-Z0-9_.-]/g, "");
              setSnapFileName(sanitized);
            }}
            disabled={isSnapping}
            inputProps={{
              pattern: "[a-zA-Z0-9_.-]*",
              maxLength: 100,
            }}
            fullWidth
          />
          <FormHelperText>
            Optional. Applies to both snapshots and recordings. Format:
            ISO8601_DetectorName_Description
          </FormHelperText>
        </FormControl>

        {/* Snap row — one button. The format select and the "download when
            finished" toggle carry the variation that used to be spread across
            separate Snap / Snap & Download buttons. */}
        <Box
          sx={{
            gridColumn: "1 / -1",
            display: "flex",
            gap: 1,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <Typography
            variant="body2"
            sx={{ fontWeight: "medium", minWidth: 60 }}
          >
            Snap:
          </Typography>
          <FormControl size="small" sx={{ width: 160 }}>
            <InputLabel id="snap-format-label">Format</InputLabel>
            <Select
              labelId="snap-format-label"
              id="snap-format-select"
              value={snapFormat}
              label="Format"
              onChange={(e) =>
                dispatch(liveViewSlice.setSnapFormat(e.target.value))
              }
              disabled={isSnapping}
            >
              {snapFormatOptions.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Button
            variant="contained"
            color="primary"
            size="small"
            onClick={handleSnapClick}
            startIcon={
              isSnapping ? (
                <CircularProgress size={16} color="inherit" />
              ) : (
                <CameraAlt />
              )
            }
            disabled={isSnapping}
            sx={{ whiteSpace: "nowrap", height: 40, minHeight: 40, width: 150 }}
          >
            {isSnapping ? "Capturing…" : "Snap"}
          </Button>

          {/* Cancel appears only while a capture is running, so it never adds
              to the resting button count. Aborts the exposure on the camera
              rather than waiting it out. */}
          {isSnapping && onCancelSnap && (
            <Button
              variant="outlined"
              color="error"
              size="small"
              onClick={onCancelSnap}
              disabled={snapCancelling}
              sx={{ whiteSpace: "nowrap", height: 40, minHeight: 40 }}
            >
              {snapCancelling ? "Cancelling…" : "Cancel"}
            </Button>
          )}

          <FormControlLabel
            control={
              <Switch
                checked={downloadSnap}
                onChange={(e) => setDownloadSnap(e.target.checked)}
                size="small"
              />
            }
            label="Download when finished"
            sx={{ ml: 0.5 }}
          />
        </Box>

        {/* Record row — Record/Stop is one toggle; "Auto-stop after" turns it
            into a fixed-length clip instead of needing its own button. */}
        <Box
          sx={{
            gridColumn: "1 / -1",
            display: "flex",
            gap: 1,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <Typography
            variant="body2"
            sx={{ fontWeight: "medium", minWidth: 60 }}
          >
            Record:
          </Typography>
          <FormControl size="small" sx={{ width: 160 }}>
            <InputLabel id="record-format-label">Format</InputLabel>
            <Select
              labelId="record-format-label"
              id="record-format-select"
              value={recordFormat}
              label="Format"
              onChange={(e) =>
                dispatch(liveViewSlice.setRecordFormat(e.target.value))
              }
              disabled={!isLiveViewActive || isRecording}
            >
              {recordFormatOptions.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Tooltip
            arrow
            title={
              !isLiveViewActive
                ? "Recording needs a running live stream"
                : isRecording
                  ? "Stop the recording"
                  : Number(clipSeconds) > 0
                    ? `Record a ${Math.round(Number(clipSeconds))} s clip and stop automatically`
                    : "Record until you press Stop"
            }
          >
            <span>
              <Button
                variant="contained"
                color={isRecording ? "error" : "primary"}
                size="small"
                onClick={handleRecordClick}
                startIcon={
                  isRecording ? <StopIcon /> : <FiberManualRecord />
                }
                disabled={!isLiveViewActive || clipRemaining > 0}
                sx={{
                  whiteSpace: "nowrap",
                  height: 40,
                  minHeight: 40,
                  width: 150,
                  ...(isRecording && {
                    animation: "blinker 1.5s linear infinite",
                    "@keyframes blinker": { "50%": { opacity: 0.6 } },
                  }),
                }}
              >
                {clipRemaining > 0
                  ? `Recording ${clipRemaining}s`
                  : isRecording
                    ? "Stop"
                    : "Record"}
              </Button>
            </span>
          </Tooltip>

          <TextField
            label="Auto-stop (s)"
            type="number"
            size="small"
            value={clipSeconds}
            onChange={(e) => setClipSeconds(e.target.value)}
            inputProps={{ min: 0, max: 600, step: 1 }}
            disabled={isRecording || clipRemaining > 0}
            helperText="0 = manual"
            sx={{ width: 120 }}
          />

          <FormControlLabel
            control={
              <Switch
                checked={downloadRecording}
                onChange={(e) => setDownloadRecording(e.target.checked)}
                size="small"
              />
            }
            label="Download when finished"
            sx={{ ml: 0.5 }}
          />
        </Box>

        {/* Status line: one place for the long-exposure / stream-off hint and
            the link to the saved files, instead of a standalone button. */}
        <Box
          sx={{
            gridColumn: "1 / -1",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 2,
            flexWrap: "wrap",
          }}
        >
          <Typography
            variant="caption"
            color={
              isLongExposure && !isSnapping ? "warning.main" : "text.secondary"
            }
            sx={{ flex: 1, minWidth: 220 }}
          >
            {isSnapping
              ? "Capturing… the camera is armed on demand; progress and Cancel are shown on the image above."
              : isLongExposure
                ? `Long-exposure mode: exposure is ${Math.round(exposureMs)} ms (> ${longExposureThresholdMs} ms), so live streaming is off. Snap arms the camera, waits one full exposure and shows the frame above.`
                : !isLiveViewActive
                  ? "Stream is off — Snap still arms the camera on demand and captures a single frame."
                  : ""}
          </Typography>
          <Button
            variant="text"
            size="small"
            disabled={!lastCapturePath}
            onClick={onGoToFolder}
          >
            Open capture folder
          </Button>
        </Box>
      </Box>

      {/* Stream Control Overlay as Dialog */}
      {/* One-time Fiji info dialog shown on first TIFF snap & download */}
      <Dialog open={fijiInfoOpen} onClose={() => setFijiInfoOpen(false)}>
        <DialogTitle>Open TIFF files with Fiji / ImageJ</DialogTitle>
        <DialogContent>
          <DialogContentText>
            The downloaded TIFF image is 16-bit. Most operating system viewers
            will display it as a black or washed-out image because they cannot
            scale 16-bit data correctly.
            <br />
            <br />
            Please open the file with{" "}
            <strong>
              <a
                href="https://fiji.sc"
                target="_blank"
                rel="noopener noreferrer"
              >
                Fiji / ImageJ
              </a>
            </strong>{" "}
            — it handles 16-bit images properly and applies the correct
            brightness/contrast scaling automatically.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFijiInfoOpen(false)} variant="contained">
            Got it
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={overlayOpen}
        onClose={() => setOverlayOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogContent sx={{ pt: 2 }}>
          <StreamControlOverlay
            stats={hudData.stats}
            featureSupport={hudData.featureSupport}
            isWebGL={hudData.isWebGL}
            imageSize={hudData.imageSize}
            viewTransform={hudData.viewTransform}
            forceExpanded={true}
          />
        </DialogContent>
      </Dialog>
    </Box>
  );
}
