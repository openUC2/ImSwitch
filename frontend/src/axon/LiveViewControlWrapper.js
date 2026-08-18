import React, { useState, useCallback, useEffect, useRef } from "react";
import {
  Button,
  IconButton,
  Tooltip,
  Box,
  Chip,
  Typography,
  CircularProgress,
  LinearProgress,
  Stack,
} from "@mui/material";
import { keyframes } from "@mui/system";
import {
  Gamepad,
  GamepadOutlined,
  FiberManualRecord,
  Videocam,
  ZoomIn,
  ZoomOut,
  RestartAlt,
} from "@mui/icons-material";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import LiveViewComponent from "./LiveViewComponent";
import LiveViewerGL from "../components/LiveViewerGL";
import WebRTCViewer from "./WebRTCViewer";
import MJPEGViewer from "./MJPEGViewer";
import PositionControllerComponent from "./PositionControllerComponent";
import HistogramOverlay from "../components/HistogramOverlay";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner";
import { useSelector, useDispatch } from "react-redux";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";
import { SNAP_PREVIEW_EVENT } from "../utils/snapPreview.js";
import { apiRecordingControllerCancelSnap } from "../backendapi/apiRecordingControllerSnapJob";

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
 * @param {boolean} enableZoomPan - Wrap the viewer in the zoom/pan shell (default: true).
 *   Set to false for precise single-click workflows (e.g. pixel calibration marking):
 *   the react-zoom-pan-pinch panning layer otherwise intercepts pointer events and
 *   swallows the click, so the viewer is rendered directly and clicks reach the canvas.
 */
const LiveViewControlWrapper = ({
  useFastMode = true,
  onClick,
  onImageLoad,
  overlayContent,
  enableStageMovement = true,
  enableZoomPan = true,
}) => {
  const dispatch = useDispatch();
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const transformFrameRef = useRef(null);
  const pendingTransformRef = useRef({
    scale: 1,
    positionX: 0,
    positionY: 0,
  });

  // Get persistent position controller visibility from Redux
  const showPositionController = liveViewState.showPositionController || false;
  const [isHovering, setIsHovering] = useState(false);
  const [transformState, setTransformState] = useState({
    scale: 1,
    positionX: 0,
    positionY: 0,
  });

  // Determine which viewer to use based on stream format
  // - WebRTC: Use WebRTCViewer for real-time low-latency streaming
  //   (currently hidden from the dropdown — the code path is kept
  //    intact for when we revisit the aiortc setup).
  // - MJPEG: Use MJPEGViewer (direct HTTP multipart, no socket.io).
  //          Recommended on Windows where the socket.io ws upgrade is
  //          flaky and the JPEG-over-ws path runs at 3–4× lower FPS.
  // - Binary: Use LiveViewerGL for high-performance WebGL rendering.
  // - JPEG: Use LiveViewComponent (legacy) for JPEG-over-socket.io.
  const useWebRTC = liveStreamState.imageFormat === "webrtc";
  const useMJPEG = liveStreamState.imageFormat === "mjpeg";
  const useWebGL =
    !useWebRTC &&
    !useMJPEG &&
    liveStreamState.backendCapabilities.webglSupported &&
    !liveStreamState.isLegacyBackend &&
    liveStreamState.imageFormat !== "jpeg";
  const canHover =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(hover: hover)").matches;
  const showInteractiveControls =
    showPositionController || (isHovering && canHover);
  const zoomPercent = Math.round(transformState.scale * 100);

  // ── Snap preview ────────────────────────────────────────────────────────
  // With the stream stopped (the long-exposure workflow) no frames ever reach
  // the viewers, so the viewport stays blank. LiveView pushes the captured
  // frame here as a PNG data URL and we show it in place of the live image.
  // Handled at wrapper level rather than inside one viewer so it works for
  // every protocol (binary/WebGL, JPEG, MJPEG, WebRTC). A restarted stream
  // clears it again.
  const [snapPreview, setSnapPreview] = useState(null);
  useEffect(() => {
    const onSnapPreview = (e) => {
      const dataUrl = e.detail && e.detail.dataUrl;
      if (dataUrl) setSnapPreview(dataUrl);
    };
    window.addEventListener(SNAP_PREVIEW_EVENT, onSnapPreview);
    return () => window.removeEventListener(SNAP_PREVIEW_EVENT, onSnapPreview);
  }, []);
  useEffect(() => {
    if (liveViewState.isStreamRunning) setSnapPreview(null);
  }, [liveViewState.isStreamRunning]);
  const showSnapPreviewImage = Boolean(snapPreview) && !liveViewState.isStreamRunning;

  // ── Snap countdown ──────────────────────────────────────────────────────
  // A long-exposure snap can run for a minute with no visible feedback. The
  // countdown ticks entirely client-side off the start time and the expected
  // duration (one exposure + overhead) — deliberately independent of the
  // backend, so it stays smooth even if a status poll is slow or dropped and it
  // starts the instant the user clicks. It is an estimate, so it stops at "any
  // moment now…" instead of hitting zero and pretending the frame is late.
  const { isSnapping, snapStartedAt, snapExpectedMs, snapJobId, snapCancelling } =
    liveViewState;
  const [snapElapsedMs, setSnapElapsedMs] = useState(0);
  useEffect(() => {
    if (!isSnapping || !snapStartedAt) {
      setSnapElapsedMs(0);
      return undefined;
    }
    const tick = () => setSnapElapsedMs(Date.now() - snapStartedAt);
    tick();
    const id = setInterval(tick, 100);
    return () => clearInterval(id);
  }, [isSnapping, snapStartedAt]);

  const snapRemainingMs = Math.max(0, snapExpectedMs - snapElapsedMs);
  const snapOverrun = snapExpectedMs > 0 && snapRemainingMs <= 0;
  const snapProgress =
    snapExpectedMs > 0
      ? Math.min(99, (snapElapsedMs / snapExpectedMs) * 100)
      : null;
  const snapLabel = snapCancelling
    ? "Cancelling…"
    : snapExpectedMs <= 0
      ? `Capturing… ${(snapElapsedMs / 1000).toFixed(1)} s`
      : snapOverrun
        ? `Capturing… any moment now (${(snapElapsedMs / 1000).toFixed(1)} s elapsed)`
        : `Capturing… ${(snapRemainingMs / 1000).toFixed(1)} s remaining`;

  // Cancel is offered here rather than only in the controls panel: when a
  // 60 s exposure is running this bar is where the user is looking.
  const handleCancelSnap = useCallback(async () => {
    dispatch(liveViewSlice.setSnapCancelling(true));
    try {
      await apiRecordingControllerCancelSnap(snapJobId);
    } catch (error) {
      console.error("Failed to cancel snap:", error);
      dispatch(liveViewSlice.setSnapCancelling(false));
    }
  }, [dispatch, snapJobId]);

  useEffect(() => {
    return () => {
      if (transformFrameRef.current !== null) {
        cancelAnimationFrame(transformFrameRef.current);
      }
    };
  }, []);

  const handleTransformed = useCallback((_, state) => {
    pendingTransformRef.current = {
      scale: state.scale,
      positionX: state.positionX,
      positionY: state.positionY,
    };

    if (transformFrameRef.current !== null) {
      return;
    }

    transformFrameRef.current = requestAnimationFrame(() => {
      transformFrameRef.current = null;

      setTransformState((prevState) => {
        const nextState = pendingTransformRef.current;

        if (
          Math.abs(prevState.scale - nextState.scale) < 0.001 &&
          Math.abs(prevState.positionX - nextState.positionX) < 0.001 &&
          Math.abs(prevState.positionY - nextState.positionY) < 0.001
        ) {
          return prevState;
        }

        return nextState;
      });
    });
  }, []);

  // Handle double-click for stage movement
  const handleImageDoubleClick = async (
    pixelX,
    pixelY,
    imageWidth,
    imageHeight,
  ) => {
    if (!enableStageMovement) return;

    try {
      // FOV (microns) for the full frame; derive FOV-Y from the aspect ratio
      // when only fovX is known.
      const fovX = objectiveState.fovX || 1000; // fallback FOV in microns
      const fovY = objectiveState.fovY || (fovX * imageHeight) / imageWidth;

      // Offset of the click from the image centre as a fraction of the frame
      // (-0.5 .. 0.5), then scaled by the FOV to get the µm offset of the
      // clicked feature from the current centre.
      const relativeX = (pixelX - imageWidth / 2) / imageWidth;
      const relativeY = (pixelY - imageHeight / 2) / imageHeight;

      // Sign mapping from image axes to stage axes. To bring the clicked point
      // to the centre, the stage must move opposite to the click offset, so
      // BOTH default to -1 (previously X was not inverted while Y was, which
      // made diagonal clicks move along the wrong direction). These signs are
      // hardware-dependent (camera mirroring / stage wiring): flip the relevant
      // constant if that axis goes the wrong way. The robust long-term source is
      // the affine calibration flip (SetupInfo.getFlipFromAffineMatrix).
      const IMAGE_TO_STAGE_SIGN_X = 1;
      const IMAGE_TO_STAGE_SIGN_Y = 1;

      const moveX = IMAGE_TO_STAGE_SIGN_X * relativeX * fovX;
      const moveY = IMAGE_TO_STAGE_SIGN_Y * relativeY * fovY;

      console.log(
        `Image: ${imageWidth}x${imageHeight}, Click: (${pixelX}, ${pixelY}), ` +
          `relative: (${relativeX.toFixed(3)}, ${relativeY.toFixed(3)}), ` +
          `moving stage by X=${moveX.toFixed(2)}µm, Y=${moveY.toFixed(2)}µm`,
      );

      // Move stage so the clicked feature ends up at the centre (relative move).
      await apiPositionerControllerMovePositioner({
        axis: "X",
        dist: moveX,
        isAbsolute: false,
        isBlocking: false,
      });

      await apiPositionerControllerMovePositioner({
        axis: "Y",
        dist: moveY,
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

  const renderViewer = () => {
    if (useWebRTC && liveViewState.isStreamRunning) {
      return (
        <WebRTCViewer
          key="webrtc-viewer"
          onClick={onClick}
          onDoubleClick={handleImageDoubleClick}
          onImageLoad={handleImageLoadInternal}
        />
      );
    }

    if (useMJPEG && liveViewState.isStreamRunning) {
      return (
        <MJPEGViewer
          key="mjpeg-viewer"
          onClick={onClick}
          onDoubleClick={handleImageDoubleClick}
          onImageLoad={handleImageLoadInternal}
          overlayContent={overlayContent}
        />
      );
    }

    if (!liveViewState.isStreamRunning) {
      return (
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
      );
    }

    if (useWebGL) {
      return (
        <LiveViewerGL
          onClick={onClick}
          onDoubleClick={handleImageDoubleClick}
          onImageLoad={handleImageLoadInternal}
          overlayContent={overlayContent}
          enableViewportControls={false}
        />
      );
    }

    return (
      <LiveViewComponent
        useFastMode={useFastMode}
        onClick={onClick}
        onDoubleClick={handleImageDoubleClick}
        onImageLoad={handleImageLoadInternal}
        overlayContent={overlayContent}
      />
    );
  };

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
        ) : liveViewState.isLongExposure ? (
          <Tooltip
            title={`Exposure is ${Math.round(liveViewState.exposureMs)} ms (> ${liveViewState.longExposureThresholdMs} ms). Live streaming is disabled at this exposure — use Snap to capture single frames.`}
            arrow
          >
            <Chip
              label={`LONG EXPOSURE • ${(liveViewState.exposureMs / 1000).toFixed(1)} s`}
              size="small"
              sx={{
                backgroundColor: "warning.main",
                color: "white",
                fontWeight: "bold",
              }}
            />
          </Tooltip>
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

        {/* Snap progress: a long exposure can run for a minute with no other
            feedback, so show how far along the capture is right over the
            viewport. Determinate against the exposure-based estimate, then
            indeterminate if the capture outlives it. */}
        {isSnapping && (
          <Box
            sx={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              zIndex: 5,
              px: 1.5,
              pt: 1,
            }}
          >
            <LinearProgress
              variant={
                snapProgress === null || snapOverrun
                  ? "indeterminate"
                  : "determinate"
              }
              value={snapProgress ?? 0}
              sx={{ height: 6, borderRadius: 3 }}
            />
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 1,
                mt: 0.5,
              }}
            >
              <Typography
                variant="caption"
                sx={{
                  color: "white",
                  textShadow: "0 1px 3px rgba(0,0,0,0.9)",
                  fontWeight: 600,
                }}
              >
                {snapLabel}
              </Typography>
              <Button
                size="small"
                variant="outlined"
                color="inherit"
                onClick={handleCancelSnap}
                disabled={snapCancelling}
                sx={{
                  color: "white",
                  borderColor: "rgba(255,255,255,0.6)",
                  backgroundColor: "rgba(0,0,0,0.45)",
                  minWidth: 0,
                  py: 0,
                }}
              >
                Cancel
              </Button>
            </Box>
          </Box>
        )}

        {showSnapPreviewImage ? (
          <Box
            sx={{
              width: "100%",
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              position: "relative",
            }}
          >
            <img
              src={snapPreview}
              alt="Last captured frame"
              style={{
                maxWidth: "100%",
                maxHeight: "100%",
                objectFit: "contain",
                imageRendering: "pixelated",
              }}
            />
            <Chip
              label="SNAPSHOT"
              size="small"
              sx={{
                position: "absolute",
                bottom: 10,
                left: 10,
                backgroundColor: "rgba(0, 0, 0, 0.6)",
                color: "white",
                fontWeight: "bold",
              }}
            />
          </Box>
        ) : liveViewState.isStreamRunning && enableZoomPan ? (
          <TransformWrapper
            key={`zoom-shell-${liveStreamState.imageFormat}-${liveViewState.isStreamRunning}`}
            initialScale={1}
            minScale={1}
            maxScale={8}
            centerOnInit
            limitToBounds={false}
            smooth
            doubleClick={{ disabled: true }}
            wheel={{
              step: 0.15,
              smoothStep: 0.01,
              wheelDisabled: true,
              touchPadDisabled: true,
            }}
            pinch={{ step: 5 }}
            panning={{
              velocityDisabled: true,
            }}
            onTransformed={handleTransformed}
          >
            {({ zoomIn, zoomOut, resetTransform }) => {
              const hasTransform =
                Math.abs(transformState.scale - 1) > 0.01 ||
                Math.abs(transformState.positionX) > 0.5 ||
                Math.abs(transformState.positionY) > 0.5;

              return (
                <>
                  <TransformComponent
                    wrapperStyle={{
                      width: "100%",
                      height: "100%",
                      touchAction: "none",
                    }}
                    contentStyle={{
                      width: "100%",
                      height: "100%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    {renderViewer()}
                  </TransformComponent>

                  {showInteractiveControls && (
                    <>
                      <Stack
                        direction="row"
                        spacing={1}
                        sx={{
                          position: "absolute",
                          right: 12,
                          bottom: 12,
                          zIndex: 4,
                          alignItems: "center",
                        }}
                      >
                        <Chip
                          label={`Zoom ${zoomPercent}%`}
                          size="small"
                          color={hasTransform ? "primary" : "default"}
                          sx={{
                            backgroundColor: hasTransform
                              ? "primary.main"
                              : "rgba(0, 0, 0, 0.55)",
                            color: "white",
                            fontWeight: 700,
                            backdropFilter: "blur(4px)",
                          }}
                        />
                        <Stack
                          direction="row"
                          spacing={0.5}
                          sx={{
                            p: 0.5,
                            borderRadius: 999,
                            backgroundColor: "rgba(0, 0, 0, 0.55)",
                            backdropFilter: "blur(4px)",
                          }}
                        >
                          <Tooltip title="Zoom out">
                            <IconButton
                              size="small"
                              onClick={() => zoomOut(0.2)}
                              sx={{ color: "white" }}
                            >
                              <ZoomOut fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Reset view">
                            <span>
                              <IconButton
                                size="small"
                                onClick={() => resetTransform(200)}
                                disabled={!hasTransform}
                                sx={{ color: "white" }}
                              >
                                <RestartAlt fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                          <Tooltip title="Zoom in">
                            <IconButton
                              size="small"
                              onClick={() => zoomIn(0.2)}
                              sx={{ color: "white" }}
                            >
                              <ZoomIn fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      </Stack>
                    </>
                  )}
                </>
              );
            }}
          </TransformWrapper>
        ) : (
          renderViewer()
        )}
      </Box>

      {/* Position controller - shown on hover OR when toggled on */}
      {showInteractiveControls && (
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
