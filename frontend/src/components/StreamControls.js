import React, { useState, useEffect, useCallback } from "react";

import {
  Box,
  TextField,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  Switch,
  FormControlLabel,
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
import apiViewControllerGetLiveViewActive from "../backendapi/apiViewControllerGetLiveViewActive";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner";
import { useSelector, useDispatch } from "react-redux";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";

export default function StreamControls({
  isStreamRunning, // This prop is kept for backwards compatibility but we prefer Redux state
  onToggleStream,
  onSnap,
  isRecording,
  onStartRecord,
  onStopRecord,
  onGoToImage,
  lastSnapPath,
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

  // Separate format options for snap and record
  const snapFormatOptions = [
    { value: 1, label: "TIFF" },
    { value: 5, label: "PNG" },
    { value: 6, label: "JPG" },
  ];

  const recordFormatOptions = [
    { value: 4, label: "MP4" },
    { value: 1, label: "TIFF" },
    { value: 3, label: "ZARR" },
  ];

  // Get stream stats from Redux (includes fps which indicates active frames)
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);

  const snapFormat = liveViewState.snapFormat;
  const recordFormat = liveViewState.recordFormat;

  // Use Redux state as source of truth for stream status
  const isLiveViewActive = liveViewState.isStreamRunning;

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

        <Button
          variant="contained"
          color="success"
          size="small"
          onClick={handleStartStream}
          disabled={isLiveViewActive}
          startIcon={<PlayArrow />}
        >
          Start
        </Button>

        <Button
          variant="contained"
          color="error"
          size="small"
          onClick={handleStopStream}
          disabled={!isLiveViewActive}
          startIcon={<Stop />}
        >
          Stop
        </Button>

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
        >
          Settings
        </Button>
      </Box>

      {/* Recording Controls Section */}
      <Box
        component="fieldset"
        sx={{
          border: 1,
          borderColor: "divider",
          borderRadius: 1,
          p: 2,
          display: "flex",
          gap: 1.5,
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
          <Videocam fontSize="small" sx={{ color: "text.secondary" }} />
          <Typography variant="subtitle1" sx={{ fontWeight: "medium" }}>
            Record
          </Typography>
        </Box>

        {/* Format and Name inputs */}
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel id="snap-format-label">Snap Format</InputLabel>
          <Select
            labelId="snap-format-label"
            id="snap-format-select"
            value={snapFormat}
            label="Snap Format"
            onChange={(e) =>
              dispatch(liveViewSlice.setSnapFormat(e.target.value))
            }
            disabled={!isLiveViewActive}
          >
            {snapFormatOptions.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>
                {opt.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <TextField
          label="Description (optional)"
          size="small"
          placeholder="e.g., sample-001, z-stack-start"
          value={snapFileName}
          onChange={(e) => {
            // Only allow alphanumeric, underscore, hyphen, and dot
            const sanitized = e.target.value.replace(/[^a-zA-Z0-9_.-]/g, "");
            setSnapFileName(sanitized);
          }}
          sx={{ minWidth: 200, flex: 1 }}
          disabled={!isLiveViewActive}
          helperText="Optional description. Filename: ISO8601_DetectorName_Description"
          inputProps={{
            pattern: "[a-zA-Z0-9_.-]*",
            maxLength: 100,
          }}
        />

        {/* Action buttons */}
        <Button
          variant="contained"
          color="primary"
          size="small"
          onClick={() => onSnap(snapFileName, snapFormat)}
          startIcon={<CameraAlt />}
          disabled={!isLiveViewActive}
        >
          Snap
        </Button>

        <Button
          variant="outlined"
          size="small"
          disabled={!lastSnapPath}
          onClick={onGoToImage}
        >
          Go to image
        </Button>

        {/* Record Format Dropdown (separate section) */}
        <Box
          sx={{
            width: "100%",
            borderTop: 1,
            borderColor: "divider",
            pt: 1.5,
            mt: 1,
          }}
        />

        <FormControl size="small" sx={{ minWidth: 130 }}>
          <InputLabel id="record-format-label">Record Format</InputLabel>
          <Select
            labelId="record-format-label"
            id="record-format-select"
            value={recordFormat}
            label="Record Format"
            onChange={(e) =>
              dispatch(liveViewSlice.setRecordFormat(e.target.value))
            }
            disabled={!isLiveViewActive}
          >
            {recordFormatOptions.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>
                {opt.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {!isRecording ? (
          <Button
            variant="contained"
            color="secondary"
            size="small"
            onClick={() => onStartRecord(recordFormat)}
            startIcon={<FiberManualRecord />}
            disabled={!isLiveViewActive}
          >
            Record
          </Button>
        ) : (
          <Button
            variant="contained"
            color="error"
            size="small"
            onClick={onStopRecord}
            startIcon={<StopIcon />}
            sx={{
              animation: "blinker 1.5s linear infinite",
              "@keyframes blinker": {
                "50%": { opacity: 0.6 },
              },
            }}
          >
            Stop
          </Button>
        )}
      </Box>

      {/* Stream Control Overlay as Dialog */}
      <Dialog
        open={overlayOpen}
        onClose={() => setOverlayOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Stream Settings</DialogTitle>
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
