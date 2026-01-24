// src/axon/CompositeStreamViewer.js
// MJPEG stream viewer component for composite acquisition
// Displays the live fused RGB image from multi-illumination acquisition

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Tooltip,
  Chip,
  CircularProgress,
} from "@mui/material";
import {
  Fullscreen,
  FullscreenExit,
  CameraAlt,
  FiberManualRecord,
} from "@mui/icons-material";
import { keyframes } from "@mui/system";

import * as compositeSlice from "../state/slices/CompositeAcquisitionSlice";

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
 * CompositeStreamViewer - MJPEG stream viewer for composite acquisition
 * 
 * Displays the live fused RGB composite image from the backend's MJPEG stream.
 * 
 * @param {string} baseUrl - Base URL for the backend API
 * @param {number} [width] - Optional fixed width
 * @param {number} [height] - Optional fixed height
 * @param {function} [onImageLoad] - Callback when image loads with (width, height)
 */
const CompositeStreamViewer = ({
  baseUrl,
  width,
  height,
  onImageLoad,
}) => {
  const dispatch = useDispatch();
  const compositeState = useSelector(compositeSlice.getCompositeState);
  
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  
  const containerRef = useRef(null);
  const imgRef = useRef(null);

  // Build stream URL
  const streamUrl = baseUrl
    ? `${baseUrl}/CompositeController/mjpeg_stream_composite?startStream=true`
    : null;

  // Handle image load
  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
    setImageError(false);
    
    if (imgRef.current) {
      const { naturalWidth, naturalHeight } = imgRef.current;
      setImageDimensions({ width: naturalWidth, height: naturalHeight });
      
      if (onImageLoad) {
        onImageLoad(naturalWidth, naturalHeight);
      }
    }
  }, [onImageLoad]);

  // Handle image error
  const handleImageError = useCallback(() => {
    setImageLoaded(false);
    setImageError(true);
  }, []);

  // Toggle fullscreen
  const handleToggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;
    
    if (!isFullscreen) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }, [isFullscreen]);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  // Reset image state when stream starts/stops
  useEffect(() => {
    if (!compositeState.isRunning) {
      setImageLoaded(false);
      setImageError(false);
    }
  }, [compositeState.isRunning]);

  const containerStyle = {
    position: "relative",
    width: width || "100%",
    height: height || "auto",
    minHeight: 200,
    backgroundColor: "#1a1a1a",
    borderRadius: isFullscreen ? 0 : 1,
    overflow: "hidden",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  };

  return (
    <Box ref={containerRef} sx={containerStyle}>
      {/* Live indicator and controls */}
      <Box
        sx={{
          position: "absolute",
          top: 8,
          left: 8,
          right: 8,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          zIndex: 2,
          pointerEvents: "none",
        }}
      >
        {/* Status chips */}
        <Box sx={{ display: "flex", gap: 1, pointerEvents: "auto" }}>
          {compositeState.isRunning && compositeState.isStreaming && imageLoaded ? (
            <Chip
              icon={<FiberManualRecord sx={{ animation: `${pulse} 1.5s ease-in-out infinite` }} />}
              label={`LIVE • ${compositeState.averageFps.toFixed(1)} FPS`}
              size="small"
              sx={{
                backgroundColor: "error.main",
                color: "white",
                fontWeight: "bold",
                "& .MuiChip-icon": { color: "white" },
              }}
            />
          ) : compositeState.isRunning ? (
            <Chip
              label="STARTING..."
              size="small"
              sx={{
                backgroundColor: "warning.main",
                color: "white",
              }}
            />
          ) : (
            <Chip
              label="STOPPED"
              size="small"
              sx={{
                backgroundColor: "rgba(128, 128, 128, 0.8)",
                color: "white",
              }}
            />
          )}
          
          {imageDimensions.width > 0 && (
            <Chip
              label={`${imageDimensions.width}×${imageDimensions.height}`}
              size="small"
              sx={{
                backgroundColor: "rgba(0, 0, 0, 0.6)",
                color: "white",
              }}
            />
          )}
        </Box>
        
        {/* Fullscreen button */}
        <Box sx={{ pointerEvents: "auto" }}>
          <Tooltip title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}>
            <IconButton
              onClick={handleToggleFullscreen}
              sx={{
                backgroundColor: "rgba(0, 0, 0, 0.5)",
                color: "white",
                "&:hover": { backgroundColor: "rgba(0, 0, 0, 0.7)" },
              }}
              size="small"
            >
              {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Stream image or placeholder */}
      {compositeState.isRunning && streamUrl ? (
        <img
          ref={imgRef}
          src={streamUrl}
          alt="Composite stream"
          onLoad={handleImageLoad}
          onError={handleImageError}
          style={{
            maxWidth: "100%",
            maxHeight: isFullscreen ? "100vh" : height || "calc(100vh - 300px)",
            objectFit: "contain",
            display: imageLoaded ? "block" : "none",
          }}
        />
      ) : null}

      {/* Loading state */}
      {compositeState.isRunning && !imageLoaded && !imageError && (
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
          <CircularProgress color="primary" />
          <Typography variant="body2" color="text.secondary">
            Connecting to stream...
          </Typography>
        </Box>
      )}

      {/* Error state */}
      {imageError && compositeState.isRunning && (
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
          <Typography variant="body1" color="error">
            Stream Error
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Could not connect to composite stream
          </Typography>
        </Box>
      )}

      {/* Idle state */}
      {!compositeState.isRunning && (
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
          <CameraAlt sx={{ fontSize: 48, color: "text.disabled" }} />
          <Typography variant="body1" color="text.secondary">
            Composite Acquisition
          </Typography>
          <Typography variant="body2" color="text.disabled">
            Start acquisition to view live composite stream
          </Typography>
        </Box>
      )}

      {/* Channel mapping indicator */}
      {compositeState.isRunning && imageLoaded && (
        <Box
          sx={{
            position: "absolute",
            bottom: 8,
            left: 8,
            display: "flex",
            gap: 0.5,
            zIndex: 2,
          }}
        >
          {["R", "G", "B"].map((channel) => {
            const source = compositeState.mapping[channel];
            const color = channel === "R" ? "#ff4444" : channel === "G" ? "#44ff44" : "#4444ff";
            return source ? (
              <Chip
                key={channel}
                label={`${channel}: ${source}`}
                size="small"
                sx={{
                  backgroundColor: "rgba(0, 0, 0, 0.7)",
                  color: color,
                  fontWeight: "bold",
                  fontSize: "0.7rem",
                }}
              />
            ) : null;
          })}
        </Box>
      )}
    </Box>
  );
};

export default CompositeStreamViewer;
