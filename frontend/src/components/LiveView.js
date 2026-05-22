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
import AxisControl from "./AxisControl.jsx";
import JoystickControl from "./JoystickControl.jsx";
import VirtualJoystickControl from "./VirtualJoystickControl.js";
import AutofocusController from "./AutofocusController";
import DetectorParameters from "./DetectorParameters";
import StreamControls from "./StreamControls";
import IlluminationController from "./IlluminationController";
import apiLiveViewControllerStartLiveView from "../backendapi/apiLiveViewControllerStartLiveView";
import apiLiveViewControllerStopLiveView from "../backendapi/apiLiveViewControllerStopLiveView";
import apiViewControllerGetLiveViewActive from "../backendapi/apiViewControllerGetLiveViewActive";
import apiLiveViewControllerGetActiveStreams from "../backendapi/apiLiveViewControllerGetActiveStreams";
import ObjectiveSwitcher from "./ObjectiveSwitcher";
import DetectorTriggerController from "./DetectorTriggerController";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import { setNotification } from "../state/slices/NotificationSlice";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";
import ExtendedLEDMatrixController from "./ExtendedLEDMatrixController.jsx";

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

  const showNotification = (message, type = "success") => {
    dispatch(setNotification({ message, type, autoHideDuration: 3000 }));
  };

  const formatLabels = {
    1: "TIFF",
    3: "ZARR",
    4: "MP4",
    5: "PNG",
    6: "JPG",
  };

  // Handle detector tab switching - restart stream with new detector, passing per-detector params
  useEffect(() => {
    const prevTab = prevActiveTabRef.current;
    prevActiveTabRef.current = activeTab;

    if (prevTab !== activeTab && isStreamRunning) {
      console.log(
        `[LiveView] Tab changed from ${prevTab} to ${activeTab}, restarting stream...`,
      );

      (async () => {
        try {
          await apiLiveViewControllerStopLiveView();
          await new Promise((resolve) => setTimeout(resolve, 200));

          const protocol = liveStreamState.imageFormat || "jpeg";
          const newDetectorName = detectors[activeTab] || null;

          // Look up saved per-detector params from Redux
          const savedParams =
            newDetectorName && liveStreamState.perDetectorSettings[newDetectorName];
          const overrideParams =
            savedParams && savedParams.protocol === protocol ? savedParams : null;

          const result = await apiLiveViewControllerStartLiveView(
            newDetectorName,
            protocol,
            overrideParams,
          );
          console.log(
            `[LiveView] Started ${protocol} stream for ${newDetectorName}`,
          );

          // Save effective params returned by backend into Redux
          if (result?.params && newDetectorName) {
            dispatch(
              liveStreamSlice.updateDetectorSettings({
                detectorName: newDetectorName,
                settings: result.params,
              }),
            );
          }
        } catch (error) {
          console.error("[LiveView] Error switching detector stream:", error);
          dispatch(liveViewSlice.setIsStreamRunning(false));
        }
      })();
    }
  }, [
    activeTab,
    isStreamRunning,
    detectors,
    liveStreamState.imageFormat,
    liveStreamState.perDetectorSettings,
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

  /* Sync activeTab with the backend's currently-streaming detector.
     Without this, the UI defaults to tab 0 (e.g. "Observationcamera")
     even when the backend is actually streaming a different detector
     (e.g. "WidefieldCamera"). */
  useEffect(() => {
    if (!detectors || detectors.length === 0) return;
    (async () => {
      try {
        const resp = await apiLiveViewControllerGetActiveStreams();
        const streams = resp?.active_streams || [];
        if (streams.length === 0) return;
        const activeDetectorName = streams[0]?.detector;
        if (!activeDetectorName) return;
        const idx = detectors.indexOf(activeDetectorName);
        if (idx >= 0 && idx !== activeTab) {
          console.log(
            `[LiveView] Backend streams '${activeDetectorName}' — syncing activeTab ${activeTab} → ${idx}`,
          );
          // Mark prevActiveTab so the stream-switch effect doesn't re-trigger.
          prevActiveTabRef.current = idx;
          dispatch(liveViewSlice.setActiveTab(idx));
        }
      } catch (error) {
        console.warn("[LiveView] Could not sync active detector tab:", error);
      }
    })();
    // Only re-run when the detector list itself changes (not on tab clicks).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectors]);

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
        const protocol = liveStreamState.imageFormat || "jpeg";

        console.log(
          `Starting ${protocol} stream (imageFormat: ${liveStreamState.imageFormat})`,
        );

        const detectorName = detectors[activeTab] || null;
        const savedParams =
          detectorName && liveStreamState.perDetectorSettings[detectorName];
        const overrideParams =
          savedParams && savedParams.protocol === protocol ? savedParams : null;

        const result = await apiLiveViewControllerStartLiveView(
          detectorName,
          protocol,
          overrideParams,
        );
        console.log(`Started ${protocol} stream for detector: ${detectorName}`);

        if (result?.params && detectorName) {
          dispatch(
            liveStreamSlice.updateDetectorSettings({
              detectorName,
              settings: result.params,
            }),
          );
        }
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

  function triggerDownload(filePath, fallbackFileName = "capture") {
    if (!filePath) return false;
    const normalizedFilePath = `/${String(filePath).replace(/^\/+/, "")}`;
    const downloadUrl = `${hostIP}:${hostPort}/imswitch/api/FileManager/download${encodeURI(normalizedFilePath)}`;
    const downloadedFileName =
      normalizedFilePath.split("/").pop() || fallbackFileName;

    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = downloadedFileName;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    return true;
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

      let data = null;
      try {
        data = await response.json();
      } catch {
        data = null;
      }

      setIsRecording(false);

      const recordedPath = data?.relativeFilePath || data?.fullPath || null;
      if (recordedPath) {
        const { filePath } = deriveCapturePaths(data);
        dispatch(liveViewSlice.setLastCapturePath(filePath));
      }

      showNotification("Recording saved", "success");
      return data;
    } catch (error) {
      console.error("Stop recording failed:", error);
      showNotification("Recording stop failed", "error");
      return null;
    }
  };

  const stopRecAndDownload = async () => {
    const data = await stopRec();
    if (!data) return;

    const filePath = data.relativeFilePath || null;
    if (!filePath) {
      showNotification(
        "Recording stopped, but no downloadable file was returned",
        "warning",
      );
      return;
    }

    const downloaded = triggerDownload(filePath, "recording.mp4");
    if (downloaded) {
      showNotification("Recording downloaded", "success");
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
            onStopRecordAndDownload={stopRecAndDownload}
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
            <RightPanelContent
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
            <RightPanelContent
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

// Extracted right panel content as a separate component for reuse
function RightPanelContent({
  stageControlTab,
  setStageControlTab,
  hostIP,
  hostPort,
}) {
  return (
    <>
      <Box mb={3}>
        <Typography variant="h6">Stage Control</Typography>

        {/* Stage Control Tabs */}
        <Tabs
          value={stageControlTab}
          onChange={(_, v) => setStageControlTab(v)}
          sx={{ mb: 2 }}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Multiple Axis View" />
          <Tab label="Joystick Control" />
          <Tab label="Virtual Joystick (speed mode)" />
        </Tabs>

        {/* Multiple Axis View */}
        {stageControlTab === 0 && (
          <AxisControl hostIP={hostIP} hostPort={hostPort} />
        )}

        {/* Joystick Control */}
        {stageControlTab === 1 && (
          <JoystickControl hostIP={hostIP} hostPort={hostPort} />
        )}

        {/* Virtual Joystick Speed Control */}
        {stageControlTab === 2 && (
          <VirtualJoystickControl hostIP={hostIP} hostPort={hostPort} />
        )}
      </Box>

      <Box mb={3}>
        <Typography variant="h6">Autofocus</Typography>
        <AutofocusController hostIP={hostIP} hostPort={hostPort} />
      </Box>

      <Box mb={3}>
        <Typography variant="h6">Illumination</Typography>
        <IlluminationController hostIP={hostIP} hostPort={hostPort} />
      </Box>

      <Box mb={3}>
        <Typography variant="h6">Objective</Typography>
        <ObjectiveSwitcher hostIP={hostIP} hostPort={hostPort} />
      </Box>

      <Box mb={3}>
        <Typography variant="h6">Extended LED Matrix</Typography>
        <ExtendedLEDMatrixController hostIP={hostIP} hostPort={hostPort} />
      </Box>

      <Box mb={3}>
        <Typography variant="h6">Detector Trigger</Typography>
        <DetectorTriggerController hostIP={hostIP} hostPort={hostPort} />
      </Box>
    </>
  );
}
