import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import {
  Box,
  Tabs,
  Tab,
  Typography,
  IconButton,
  Drawer,
  useMediaQuery,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import DetectorParameters from "./DetectorParameters";
import StreamControls from "./StreamControls";
import apiLiveViewControllerStartLiveView from "../backendapi/apiLiveViewControllerStartLiveView";
import apiLiveViewControllerStopLiveView from "../backendapi/apiLiveViewControllerStopLiveView";
import apiViewControllerGetLiveViewActive from "../backendapi/apiViewControllerGetLiveViewActive";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import {
  setNotification,
  clearNotification,
} from "../state/slices/NotificationSlice";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";
import LiveViewRightPanelContent from "./LiveViewRightPanelContent.jsx";

/*
<ImageViewport
            detectors={detectors}
            activeTab={activeTab}
            imageUrls={imageUrls}
            pollImageUrl={pollImageUrl}
            showHistogram={showHistogram}
            histogramActive={histogramActive}
            histogramX={histogramX}
            histogramY={histogramY}
            histogramData={histogramData}
            chartOptions={chartOptions}
            pixelSize={pixelSize}
            minVal={minVal}
            maxVal={maxVal}
            onRangeChange={handleRangeChange}
            onRangeCommit={handleRangeCommit}
            onMove={moveStage}
          />*/

export default function LiveView({ setFileManagerInitialPath }) {
  // Redux dispatcher
  const dispatch = useDispatch();

  // Get connection settings from Redux
  const { ip: hostIP, apiPort: hostPort } = useSelector(
    getConnectionSettingsState,
  );

  // Access global Redux state
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  // Track if auto-start has been attempted to prevent re-triggering on format changes
  const autoStartAttemptedRef = React.useRef(false);

  // Debug log to verify persisted stream format (only on mount)
  useEffect(() => {
    console.log("[LiveView] Mounted with stream state:", {
      imageFormat: liveStreamState.imageFormat,
      streamSettings: liveStreamState.streamSettings,
      isStreamRunning: liveViewState.isStreamRunning,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty dependency array = only run once on mount

  // Use Redux state instead of local state
  const detectors = liveViewState.detectors;
  const activeTab = liveViewState.activeTab;
  const lastCapturePath =
    liveViewState.lastCapturePath || liveViewState.lastSnapPath; // compatibility with persisted old key
  const isStreamRunning = liveViewState.isStreamRunning;

  // Keep some local state for now (these may need their own slices later)
  const [isRecording, setIsRecording] = useState(false);

  // Stage control tabs state
  const [stageControlTab, setStageControlTab] = useState(0); // 0 = Multiple Axis View, 1 = Joystick Control

  // Responsive design: collapsible right panel for mobile
  const isMobile = useMediaQuery("(max-width:600px)");
  const [rightPanelOpen, setRightPanelOpen] = useState(!isMobile);

  // Track previous activeTab to detect changes
  const prevActiveTabRef = React.useRef(activeTab);

  // Track notification timeout for proper cleanup
  const notificationTimeoutRef = React.useRef(null);

  const showNotification = (message, type = "success") => {
    // Clear any existing timeout to prevent premature clearing of new notifications
    if (notificationTimeoutRef.current) {
      clearTimeout(notificationTimeoutRef.current);
    }

    dispatch(setNotification({ message, type }));
    notificationTimeoutRef.current = setTimeout(() => {
      dispatch(clearNotification());
      notificationTimeoutRef.current = null;
    }, 3000);
  };

  // Cleanup notification timeout on unmount
  useEffect(() => {
    return () => {
      if (notificationTimeoutRef.current) {
        clearTimeout(notificationTimeoutRef.current);
      }
    };
  }, []);

  const formatLabels = {
    1: "TIFF",
    3: "ZARR",
    4: "MP4",
    5: "PNG",
    6: "JPG",
  };

  // Handle detector tab switching - restart stream with new detector
  useEffect(() => {
    const prevTab = prevActiveTabRef.current;
    prevActiveTabRef.current = activeTab;

    // If tab changed and stream is running, restart stream with new detector
    if (prevTab !== activeTab && isStreamRunning) {
      console.log(
        `[LiveView] Tab changed from ${prevTab} to ${activeTab}, restarting stream...`,
      );

      (async () => {
        try {
          // Stop current stream
          await apiLiveViewControllerStopLiveView();
          console.log("[LiveView] Stopped stream for previous detector");

          // Small delay to ensure clean shutdown
          await new Promise((resolve) => setTimeout(resolve, 200));

          // Start new stream with new detector
          const protocol = liveStreamState.imageFormat || "jpeg";
          const newDetectorName = detectors[activeTab] || null;
          await apiLiveViewControllerStartLiveView(newDetectorName, protocol);
          console.log(
            `[LiveView] Started ${protocol} stream for new detector: ${newDetectorName}`,
          );
        } catch (error) {
          console.error("[LiveView] Error switching detector stream:", error);
          // Ensure Redux state reflects actual state
          dispatch(liveViewSlice.setIsStreamRunning(false));
        }
      })();
    }
  }, [
    activeTab,
    isStreamRunning,
    detectors,
    liveStreamState.imageFormat,
    dispatch,
  ]);

  /* detectors */
  useEffect(() => {
    (async () => {
      try {
        // 'getDetectorNames' must return something array-like
        const r = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/SettingsController/getDetectorNames`,
        );
        const data = await r.json();
        // Check if data is an array before setting state
        if (Array.isArray(data)) {
          dispatch(liveViewSlice.setDetectors(data));
        } else {
          console.error("getDetectorNames returned non-array:", data);
          dispatch(liveViewSlice.setDetectors([]));
        }
      } catch (error) {
        console.error("Failed to fetch detectors:", error);
        dispatch(liveViewSlice.setDetectors([]));
      }
    })();
  }, [hostIP, hostPort, dispatch]);

  /* min/max - disabled auto-windowing to allow manual control via slider */
  // Commented out to prevent overriding manual slider settings
  /*
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/HistogrammController/minmaxvalues`
        );
        if (!r.ok) return;
        const d = await r.json();
        // Update Redux state instead of local state
        dispatch(liveStreamSlice.setMinVal(d.minVal || 0));
        dispatch(liveStreamSlice.setMaxVal(d.maxVal || 65535));
      } catch {}
    })();
  }, [hostIP, hostPort, dispatch]);
  */

  /* Check if stream is running and auto-start if not active (only on initial mount or connection change) */
  useEffect(() => {
    // Reset auto-start flag when connection changes
    autoStartAttemptedRef.current = false;
  }, [hostIP, hostPort]);

  useEffect(() => {
    // Skip if auto-start was already attempted
    if (autoStartAttemptedRef.current) return;

    (async () => {
      try {
        const isActive = await apiViewControllerGetLiveViewActive();
        dispatch(liveViewSlice.setIsStreamRunning(isActive));

        // Auto-start stream if not already running (improves first-time UX)
        if (!isActive) {
          console.log("[LiveView] Stream not active, auto-starting...");
          try {
            const protocol = liveStreamState.imageFormat || "jpeg";
            await apiLiveViewControllerStartLiveView(null, protocol);
            dispatch(liveViewSlice.setIsStreamRunning(true));
            console.log(`[LiveView] Auto-started ${protocol} stream`);
          } catch (error) {
            console.error("[LiveView] Failed to auto-start stream:", error);
          }
        }

        // Mark auto-start as attempted (prevents re-triggering on format changes)
        autoStartAttemptedRef.current = true;
      } catch (error) {
        console.error("[LiveView] Failed to check stream status:", error);
      }
    })();
  }, [hostIP, hostPort, activeTab, dispatch, liveStreamState.imageFormat]);

  /* handlers */
  // Note: Range handling now done directly in Redux dispatch - old handlers removed

  const toggleStream = async () => {
    const shouldStart = !isStreamRunning;

    try {
      if (shouldStart) {
        // Determine protocol from current stream settings
        // Use imageFormat from Redux state - supports binary, jpeg, and webrtc
        const protocol = liveStreamState.imageFormat || "jpeg"; // Default to JPEG

        console.log(
          `Starting ${protocol} stream (imageFormat: ${liveStreamState.imageFormat})`,
        );

        // Start stream with current protocol (binary, jpeg, or webrtc)
        // Get detector name from active tab
        const detectorName = detectors[activeTab] || null;
        await apiLiveViewControllerStartLiveView(detectorName, protocol);
        console.log(`Started ${protocol} stream for detector: ${detectorName}`);
      } else {
        // Stop stream
        await apiLiveViewControllerStopLiveView();
        console.log("Stopped stream");
      }

      // Update Redux state
      dispatch(liveViewSlice.setIsStreamRunning(shouldStart));
    } catch (error) {
      console.error("Error toggling stream:", error);
      // Fallback: try to update state anyway for UI consistency
      dispatch(liveViewSlice.setIsStreamRunning(shouldStart));
    }
  };
  function deriveCapturePaths(data) {
    const normalizedRelativeFilePath = data.relativeFilePath
      ? `/${String(data.relativeFilePath).replace(/^\/+/, "")}`
      : null;
    if (normalizedRelativeFilePath) {
      const fileName = normalizedRelativeFilePath.split("/").pop() || "";
      const folderPath = getFolderPath(normalizedRelativeFilePath);
      return { folderPath, filePath: normalizedRelativeFilePath, fileName };
    }

    const folderPath = `/${data.relativePath}`;
    const fullPath = String(data.fullPath || "").replace(/\\/g, "/");
    const fileName = fullPath ? fullPath.split("/").pop() : "";
    const filePath = fileName ? `${folderPath}/${fileName}` : folderPath;
    return { folderPath, filePath, fileName };
  }

  async function runSnap(fileName, format) {
    try {
      const response = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/RecordingController/snapImageToPath?fileName=${encodeURIComponent(fileName)}&saveFormat=${encodeURIComponent(format)}`,
      );
      if (!response.ok) {
        throw new Error(`Snap failed: ${response.status}`);
      }
      const data = await response.json();
      const { filePath } = deriveCapturePaths(data);
      dispatch(liveViewSlice.setLastCapturePath(filePath)); // Store latest capture path
      const label = formatLabels[format] || "Unknown";
      showNotification(`Image saved (${label})`, "success");
      return data;
    } catch (error) {
      console.error("Snap failed:", error);
      showNotification("Snap failed", "error");
      return null;
    }
  }

  async function snap(fileName, format) {
    await runSnap(fileName, format);
  }

  async function snapAndDownload(fileName, format) {
    const data = await runSnap(fileName, format);
    if (!data) return;

    const { filePath, fileName: downloadedFileName } = deriveCapturePaths(data);
    const downloadUrl = `${hostIP}:${hostPort}/imswitch/api/FileManager/download${encodeURI(filePath)}`;

    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = downloadedFileName || "capture";
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    const label = formatLabels[format] || "Unknown";
    showNotification(`Image downloaded (${label})`, "success");
  }

  function getFolderPath(path) {
    if (!path) return "/";
    const normalized = String(path).replace(/\\/g, "/");
    const looksLikeFile = /\/[^/]+\.[^/]+$/.test(normalized);
    if (!looksLikeFile) return normalized;
    const lastSlash = normalized.lastIndexOf("/");
    return lastSlash > 0 ? normalized.slice(0, lastSlash) : "/";
  }

  function handleGoToFolder() {
    if (lastCapturePath) {
      setFileManagerInitialPath(getFolderPath(lastCapturePath));
    }
  }
  const startRec = async (description, format) => {
    try {
      let url = `${hostIP}:${hostPort}/imswitch/api/RecordingController/startRecording?mSaveFormat=${format}`;
      // Add optional description if provided
      if (description && description.trim()) {
        url += `&fileName=${encodeURIComponent(description)}`;
      }
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Start recording failed: ${response.status}`);
      }
      setIsRecording(true);
      const label = formatLabels[format] || "Unknown";
      showNotification(`Recording started (${label})`, "info");
    } catch (error) {
      console.error("Start recording failed:", error);
      showNotification("Recording start failed", "error");
    }
  };
  const stopRec = async () => {
    try {
      const response = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/RecordingController/stopRecording`,
      );
      if (!response.ok) {
        throw new Error(`Stop recording failed: ${response.status}`);
      }
      setIsRecording(false);
      showNotification("Recording saved", "success");
    } catch (error) {
      console.error("Stop recording failed:", error);
      showNotification("Recording stop failed", "error");
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flex: 1,
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* LEFT - Main Content Area */}
      <Box
        sx={{
          width: isMobile ? "100%" : rightPanelOpen ? "60%" : "100%",
          height: "100%",
          display: "flex",
          paddingTop: 1,
          flexDirection: "column",
          boxSizing: "border-box",
          overflowY: "auto",
          transition: "width 0.3s ease",
          "&::-webkit-scrollbar": {
            width: "8px",
          },
          "&::-webkit-scrollbar-track": {
            background: "transparent",
          },
          "&::-webkit-scrollbar-thumb": {
            background: "rgba(128, 128, 128, 0.3)",
            borderRadius: "4px",
          },
          "&::-webkit-scrollbar-thumb:hover": {
            background: "rgba(128, 128, 128, 0.5)",
          },
        }}
      >
        {/* Header with toggle button for controls panel */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            px: 1,
          }}
        >
          <Tabs
            value={activeTab}
            onChange={(_, v) => dispatch(liveViewSlice.setActiveTab(v))}
            sx={{ mt: 1 }}
          >
            {detectors.map((d) => (
              <Tab key={d} label={d} />
            ))}
          </Tabs>

          {/* Toggle button to open/close right panel */}
          <IconButton
            onClick={() => setRightPanelOpen(!rightPanelOpen)}
            sx={{
              mt: 1,
              backgroundColor: "primary.main",
              color: "primary.contrastText",
              "&:hover": {
                backgroundColor: "primary.dark",
              },
            }}
            title={rightPanelOpen ? "Hide Controls" : "Show Controls"}
          >
            <MenuIcon />
          </IconButton>
        </Box>

        {/* Live View Container */}
        <Box
          sx={{
            flex: "0 0 auto",
            mb: 2,
            position: "relative",
            minHeight: 0,
          }}
        >
          <LiveViewControlWrapper />
        </Box>

        {/* Stream, Record and Detector Controls */}
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mb: 2 }}>
          <StreamControls
            isStreamRunning={isStreamRunning}
            onToggleStream={toggleStream}
            onSnap={snap}
            onSnapAndDownload={snapAndDownload}
            isRecording={isRecording}
            onStartRecord={startRec}
            onStopRecord={stopRec}
            onGoToFolder={handleGoToFolder}
            lastCapturePath={lastCapturePath}
          />

          <DetectorParameters hostIP={hostIP} hostPort={hostPort} />
        </Box>
      </Box>

      {/* RIGHT - Controls Panel (Drawer on mobile, inline on desktop) */}
      {isMobile ? (
        // Mobile: Use Drawer component for slide-in panel
        <Drawer
          anchor="right"
          open={rightPanelOpen}
          onClose={() => setRightPanelOpen(false)}
          PaperProps={{
            sx: {
              width: "85%",
              maxWidth: 360,
              backgroundColor: "background.paper",
            },
          }}
        >
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              p: 1,
              borderBottom: 1,
              borderColor: "divider",
            }}
          >
            <Typography variant="h6">Controls</Typography>
            <IconButton onClick={() => setRightPanelOpen(false)}>
              <ChevronRightIcon />
            </IconButton>
          </Box>
          <Box
            sx={{
              p: 2,
              overflowY: "auto",
              height: "100%",
            }}
          >
            <LiveViewRightPanelContent
              stageControlTab={stageControlTab}
              setStageControlTab={setStageControlTab}
              hostIP={hostIP}
              hostPort={hostPort}
            />
          </Box>
        </Drawer>
      ) : (
        // Desktop: Show inline panel when open
        rightPanelOpen && (
          <Box
            sx={{
              width: "40%",
              height: "100%",
              overflowY: "auto",
              p: 2,
              transition: "width 0.3s ease",
              "&::-webkit-scrollbar": {
                width: "8px",
              },
              "&::-webkit-scrollbar-track": {
                background: "transparent",
              },
              "&::-webkit-scrollbar-thumb": {
                background: "rgba(128, 128, 128, 0.3)",
                borderRadius: "4px",
              },
              "&::-webkit-scrollbar-thumb:hover": {
                background: "rgba(128, 128, 128, 0.5)",
              },
            }}
          >
            <LiveViewRightPanelContent
              stageControlTab={stageControlTab}
              setStageControlTab={setStageControlTab}
              hostIP={hostIP}
              hostPort={hostPort}
            />
          </Box>
        )
      )}
    </Box>
  );
}
