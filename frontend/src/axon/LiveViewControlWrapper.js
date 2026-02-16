import React, { useState, useCallback } from "react";
import {
  IconButton,
  Tooltip,
  Box,
  Chip,
  Typography,
  CircularProgress,
} from "@mui/material";
import { keyframes } from "@mui/system";
import {
  Gamepad,
  GamepadOutlined,
  FiberManualRecord,
  Videocam,
} from "@mui/icons-material";
import LiveViewComponent from "./LiveViewComponent";
import LiveViewerGL from "../components/LiveViewerGL";
import WebRTCViewer from "./WebRTCViewer";
import PositionControllerComponent from "./PositionControllerComponent";
import HistogramOverlay from "../components/HistogramOverlay";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner";
import { useSelector, useDispatch } from "react-redux";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";

// Pulsing animation for LIVE indicator
const pulse = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
`;

/**
 * LiveViewControlWrapper - Unified wrapper for different stream viewers
 * Automatically selects the appropriate viewer based on stream format (WebRTC, Binary/WebGL, JPEG)
 *
 * @param {boolean} useFastMode - Use optimized processing for better performance
 * @param {function} onClick - Callback for single click: (pixelX, pixelY, imageWidth, imageHeight, displayInfo)
 * @param {function} onImageLoad - Callback when image dimensions change: (width, height)
 * @param {React.ReactNode} overlayContent - Optional overlay content to render on top of the viewer
 * @param {boolean} enableStageMovement - Enable default double-click stage movement behavior (default: true)
 */
const LiveViewControlWrapper = ({
  useFastMode = true,
  onClick,
  onImageLoad,
  overlayContent,
  enableStageMovement = true,
}) => {
  const dispatch = useDispatch();
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);

  // Get persistent position controller visibility from Redux
  const showPositionController = liveViewState.showPositionController || false;
  const [isHovering, setIsHovering] = useState(false);

  // Determine which viewer to use based on stream format
  // - WebRTC: Use WebRTCViewer for real-time low-latency streaming
  // - Binary: Use LiveViewerGL for high-performance WebGL rendering
  // - JPEG: Use LiveViewComponent (legacy) for JPEG streaming
  const useWebRTC = liveStreamState.imageFormat === "webrtc";
  const useWebGL =
    !useWebRTC &&
    liveStreamState.backendCapabilities.webglSupported &&
    !liveStreamState.isLegacyBackend &&
    liveStreamState.imageFormat !== "jpeg";

  // Handle double-click for stage movement
  const handleImageDoubleClick = async (
    pixelX,
    pixelY,
    imageWidth,
    imageHeight,
  ) => {
    if (!enableStageMovement) return;

    try {
      // Calculate real-world position from pixel coordinates
      // Use the actual image dimensions and center coordinates properly
      const fovX = objectiveState.fovX || 1000; // fallback FOV in microns
      const fovY = objectiveState.fovY || (fovX * imageHeight) / imageWidth; // calculate FOV Y based on aspect ratio

      // Calculate the center of the image
      const centerX = imageWidth / 2;
      const centerY = imageHeight / 2;

      // Calculate relative movement from image center
      const relativeX = (pixelX - centerX) / imageWidth; // -0.5 to 0.5
      const relativeY = (pixelY - centerY) / imageHeight; // -0.5 to 0.5

      // Convert to microns
      const moveX = relativeX * fovX;
      const moveY = relativeY * fovY;

      console.log(
        `Image: ${imageWidth}x${imageHeight}, Click: (${pixelX}, ${pixelY}), Center: (${centerX}, ${centerY})`,
      );
      console.log(
        `Relative: (${relativeX.toFixed(3)}, ${relativeY.toFixed(
          3,
        )}), Moving stage by: X=${moveX.toFixed(2)}µm, Y=${moveY.toFixed(2)}µm`,
      );

      // Move stage to the clicked position (relative movement)
      await apiPositionerControllerMovePositioner({
        axis: "X",
        dist: moveX,
        isAbsolute: false,
        isBlocking: false,
      });

      await apiPositionerControllerMovePositioner({
        axis: "Y",
        dist: -moveY, // Invert Y as microscope Y often goes opposite to image Y
        isAbsolute: false,
        isBlocking: false,
      });
    } catch (error) {
      console.error("Failed to move stage:", error);
    }
  };

  // Handle image load - forward to parent if callback provided
  const handleImageLoadInternal = useCallback(
    (width, height) => {
      if (onImageLoad) {
        onImageLoad(width, height);
      }
    },
    [onImageLoad],
  );

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      {/* Toggle button for position controller (always visible) */}
      <Tooltip
        title={showPositionController ? "Hide controls" : "Show controls"}
      >
        <IconButton
          onClick={() =>
            dispatch(
              liveViewSlice.setShowPositionController(!showPositionController),
            )
          }
          sx={{
            position: "absolute",
            top: 10,
            left: 10,
            zIndex: 3,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            color: "white",
            "&:hover": {
              backgroundColor: "rgba(0, 0, 0, 0.7)",
            },
          }}
        >
          {showPositionController ? <Gamepad /> : <GamepadOutlined />}
        </IconButton>
      </Tooltip>

      {/* Live Stream Indicator */}
      <Box
        sx={{
          position: "absolute",
          top: 10,
          right: 10,
          zIndex: 3,
          display: "flex",
          gap: 1,
        }}
      >
        {liveViewState.isStreamRunning ? (
          <Chip
            icon={
              <FiberManualRecord
                sx={{ animation: `${pulse} 1.5s ease-in-out infinite` }}
              />
            }
            label={
              liveStreamState.stats?.fps > 0
                ? `LIVE • ${liveStreamState.stats.fps.toFixed(1)} FPS`
                : "LIVE"
            }
            size="small"
            sx={{
              backgroundColor: "error.main",
              color: "white",
              fontWeight: "bold",
              "& .MuiChip-icon": {
                color: "white",
              },
            }}
          />
        ) : (
          <Chip
            label="PAUSED"
            size="small"
            sx={{
              backgroundColor: "rgba(128, 128, 128, 0.8)",
              color: "white",
              fontWeight: "bold",
            }}
          />
        )}
      </Box>

      <Box
        sx={{
          position: "relative",
          flex: "1",
          width: "100%",
          minHeight: "480px", // Prevent collapse before stream loads
          maxHeight: "calc(100vh - 220px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "background.default",
          border: 1,
          borderColor: "divider",
          borderRadius: 1,
        }}
      >
        {/* Histogram overlay */}
        <HistogramOverlay
          active={true}
          visible={liveStreamState.showHistogram}
          x={liveStreamState.histogramX || []}
          y={liveStreamState.histogramY || []}
          dataObj={{
            labels: (liveStreamState.histogramX || []).map((v, i) =>
              // Show every 100th label for 16-bit, every 10th for 8-bit
              i % (liveStreamState.histogramX?.length > 500 ? 100 : 10) === 0
                ? v
                : "",
            ),
            datasets: [
              {
                label: "Histogram",
                data: liveStreamState.histogramY || [],
                backgroundColor: "rgba(75, 192, 192, 0.6)",
                barPercentage: 1.0,
                categoryPercentage: 1.0,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
              x: {
                display: true,
                grid: { display: false },
                ticks: {
                  maxRotation: 0,
                  autoSkip: true,
                  maxTicksLimit: 8,
                  color: "#fff",
                  font: { size: 9 },
                },
              },
              y: {
                beginAtZero: true,
                display: true,
                grid: { color: "rgba(255,255,255,0.1)" },
                ticks: {
                  color: "#fff",
                  font: { size: 9 },
                },
              },
            },
            plugins: {
              legend: { display: false },
              tooltip: { enabled: false },
            },
          }}
        />

        {/* Select appropriate viewer based on stream format */}
        {/* WebRTC: Only render when stream is running to force remount on start/stop */}
        {useWebRTC && liveViewState.isStreamRunning ? (
          <WebRTCViewer
            key="webrtc-viewer" // Force new instance on remount
            onClick={onClick}
            onDoubleClick={handleImageDoubleClick}
            onImageLoad={handleImageLoadInternal}
          />
        ) : !liveViewState.isStreamRunning ? (
          // Placeholder when stream is not running
          <Box
            sx={{
              width: "100%",
              height: "100%",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 2,
            }}
          >
            <Videocam
              sx={{
                fontSize: 80,
                color: "text.disabled",
              }}
            />
            <Typography
              variant="h6"
              sx={{
                color: "text.secondary",
              }}
            >
              Stream nicht aktiv
            </Typography>
            <CircularProgress
              size={24}
              sx={{
                color: "text.disabled",
              }}
            />
          </Box>
        ) : useWebGL ? (
          <LiveViewerGL
            onClick={onClick}
            onDoubleClick={handleImageDoubleClick}
            onImageLoad={handleImageLoadInternal}
            overlayContent={overlayContent}
          />
        ) : (
          <LiveViewComponent
            useFastMode={useFastMode}
            onClick={onClick}
            onDoubleClick={handleImageDoubleClick}
            onImageLoad={handleImageLoadInternal}
            overlayContent={overlayContent}
          />
        )}
      </Box>

      {/* Position controller - shown on hover OR when toggled on */}
      {(showPositionController ||
        (isHovering && window.matchMedia("(hover: hover)").matches)) && (
        <div
          style={{
            position: "absolute",
            bottom: "10px",
            left: "0px",
            zIndex: 2,
            opacity: isHovering ? 0.9 : 0.7,
            transition: "opacity 0.3s ease-in-out",
          }}
        >
          <PositionControllerComponent />
        </div>
      )}
    </div>
  );
};

export default LiveViewControlWrapper;
