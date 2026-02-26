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
  useTheme,
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
import ObjectiveSwitcher from "./ObjectiveSwitcher";
import DetectorTriggerController from "./DetectorTriggerController";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
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
  const lastSnapPath = liveViewState.lastSnapPath; // Get from Redux
  const isStreamRunning = liveViewState.isStreamRunning;

  // Keep some local state for now (these may need their own slices later)
  const [isRecording, setIsRecording] = useState(false);

  // Stage control tabs state
  const [stageControlTab, setStageControlTab] = useState(0); // 0 = Multiple Axis View, 1 = Joystick Control

  // Responsive design: collapsible right panel for mobile
  const theme = useTheme();
  const isMobile = useMediaQuery("(max-width:600px)");
  const [rightPanelOpen, setRightPanelOpen] = useState(!isMobile);

  // Track previous activeTab to detect changes
  const prevActiveTabRef = React.useRef(activeTab);

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
  async function snap(fileName, format) {
    // English comment: Example fetch for snapping an image with editable fileName
    const response = await fetch(
      `${hostIP}:${hostPort}/imswitch/api/RecordingController/snapImageToPath?fileName=${fileName}&saveFormat=${format}`,
    );
    const data = await response.json();
    // data.relativePath might be "recordings/2025_05_20-11-12-44_PM"
    const snapPath = `/${data.relativePath}`;
    dispatch(liveViewSlice.setLastSnapPath(snapPath)); // Store in Redux
  }
  function handleGoToImage() {
    if (lastSnapPath) {
      setFileManagerInitialPath(lastSnapPath);
    }
  }
  const startRec = async (format) => {
    setIsRecording(true);
    fetch(
      `${hostIP}:${hostPort}/imswitch/api/RecordingController/startRecording?mSaveFormat=${format}`,
    ).catch(() => {});
  };
  const stopRec = async () => {
    setIsRecording(false);
    fetch(
      `${hostIP}:${hostPort}/imswitch/api/RecordingController/stopRecording`,
    ).catch(() => {});
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
            isRecording={isRecording}
            onStartRecord={startRec}
            onStopRecord={stopRec}
            onGoToImage={handleGoToImage}
            lastSnapPath={lastSnapPath}
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
