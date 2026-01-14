import React, { useRef, useEffect, useState, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box, Typography } from "@mui/material";
import * as liveViewSlice from "../state/slices/LiveStreamSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";

/**
 * LiveViewComponent - Unified image viewer with intensity scaling
 *
 * Uses optimized pixel-based intensity windowing for proper scientific image processing.
 * This provides linear intensity mapping: [minVal, maxVal] → [0, 255]
 *
 * @param {boolean} useFastMode - Use optimized processing for better performance (default: true)
 * @param {function} onClick - Callback for single click: (pixelX, pixelY, imageWidth, imageHeight, displayInfo)
 * @param {function} onDoubleClick - Callback for double click: (pixelX, pixelY, imageWidth, imageHeight)
 * @param {function} onImageLoad - Callback when image dimensions change: (width, height)
 * @param {React.ReactNode} overlayContent - Optional overlay content to render on top of the canvas
 */
const LiveViewComponent = ({
  useFastMode = true,
  onClick,
  onDoubleClick,
  onImageLoad,
  overlayContent,
}) => {
  // redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const liveStreamState = useSelector(liveViewSlice.getLiveStreamState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);

  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // FPS tracking for JPEG stream
  const fpsCounterRef = useRef({
    frames: 0,
    lastTime: performance.now(),
  });

  // Track FPS for JPEG stream (triggered on each new frame)
  useEffect(() => {
    if (!liveStreamState.liveViewImage) return;

    const counter = fpsCounterRef.current;
    counter.frames++;

    const now = performance.now();
    const elapsed = now - counter.lastTime;

    // Update FPS every second
    if (elapsed >= 1000) {
      const fps = Math.round((counter.frames * 1000) / elapsed);
      dispatch(liveViewSlice.setStats({ fps, bps: 0 })); // bps not available for JPEG

      console.log(`[LiveViewComponent] JPEG FPS: ${fps}`);

      // Reset counters
      counter.frames = 0;
      counter.lastTime = now;
    }
  }, [liveStreamState.liveViewImage, dispatch]);
  const prevDimensionsRef = useRef({ width: 0, height: 0 }); // Track dimensions to avoid redundant callbacks
  const histogramCounterRef = useRef(0); // Counter for throttling histogram computation
  const [imageLoaded, setImageLoaded] = useState(false);
  const [containerSize, setContainerSize] = useState({
    width: 800,
    height: 600,
  });
  const [canvasStyle, setCanvasStyle] = useState({});

  // Compute histogram from canvas (for JPEG streams)
  const computeHistogramFromCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !liveStreamState.showHistogram) return;

    // Throttle: only compute every 10th frame
    histogramCounterRef.current++;
    if (histogramCounterRef.current % 10 !== 0) return;

    try {
      const ctx = canvas.getContext("2d");
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      const binCount = 256; // 8-bit histogram for JPEG
      const histogram = new Array(binCount).fill(0);
      const histogramX = new Array(binCount);

      // Initialize x-axis values
      for (let i = 0; i < binCount; i++) {
        histogramX[i] = i;
      }

      // Count luminance values (RGB → Grayscale)
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        histogram[lum]++;
      }

      console.log("JPEG Histogram computed:", {
        bins: binCount,
        totalPixels: canvas.width * canvas.height,
      });

      // Update Redux
      dispatch(
        liveViewSlice.setHistogramData({
          x: histogramX,
          y: histogram,
        })
      );
    } catch (error) {
      console.warn("JPEG histogram computation failed:", error);
    }
  }, [dispatch, liveStreamState.showHistogram]);

  // Optimized intensity windowing - proper scientific image processing
  const applyIntensityWindowing = useCallback(
    (image, minVal, maxVal) => {
      const canvas = canvasRef.current;
      if (!canvas || !image) return;

      canvas.width = image.width;
      canvas.height = image.height;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Calculate the intensity range
      const range = Math.max(1, maxVal - minVal); // Avoid division by zero
      const scale = 255.0 / range;

      // Apply linear intensity windowing: [minVal, maxVal] → [0, 255]
      // Preserve color by mapping the pixel luminance and scaling RGB channels proportionally.
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        // Compute perceptual luminance from RGB
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;

        // Map luminance through the window [minVal, maxVal] -> [0,255]
        let mappedLum;
        if (lum <= minVal) {
          mappedLum = 0;
        } else if (lum >= maxVal) {
          mappedLum = 255;
        } else {
          mappedLum = (lum - minVal) * scale;
        }

        // If original luminance > 0, scale RGB channels proportionally to preserve colour
        if (lum > 0) {
          const factor = mappedLum / lum;
          data[i] = Math.min(255, Math.max(0, Math.round(r * factor)));
          data[i + 1] = Math.min(255, Math.max(0, Math.round(g * factor)));
          data[i + 2] = Math.min(255, Math.max(0, Math.round(b * factor)));
        } else {
          // Fallback: if luminance is zero, write mappedLum as grayscale
          const v = Math.round(mappedLum);
          data[i] = v;
          data[i + 1] = v;
          data[i + 2] = v;
        }
        // Alpha channel (i + 3) remains unchanged
      }

      ctx.putImageData(imageData, 0, 0);

      // Compute histogram after rendering (for JPEG streams)
      computeHistogramFromCanvas();
    },
    [computeHistogramFromCanvas]
  );

  // Monitor container size changes
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        setContainerSize({ width, height });
      }
    });

    resizeObserver.observe(container);

    // Initial size
    const rect = container.getBoundingClientRect();
    setContainerSize({ width: rect.width, height: rect.height });

    return () => resizeObserver.disconnect();
  }, []);

  // Calculate responsive canvas dimensions
  const getDisplayDimensions = useCallback(
    (imageWidth, imageHeight) => {
      if (!imageWidth || !imageHeight) {
        console.log("Missing image dimensions, using fallback");
        return { width: 400, height: 300 };
      }

      if (!containerSize.width || !containerSize.height) {
        console.log("Missing container size, using fallback");
        return { width: 400, height: 300 };
      }

      const imageAspectRatio = imageWidth / imageHeight;

      // Match container width exactly as requested
      // Always use the full container width and scale height accordingly
      const displayWidth = Math.floor(containerSize.width);
      const displayHeight = Math.floor(containerSize.width / imageAspectRatio);

      return { width: displayWidth, height: displayHeight };
    },
    [containerSize]
  );

  // Legacy pixel-perfect intensity scaling (for compatibility)
  const applyPixelIntensityScaling = useCallback(
    (image, minVal, maxVal) => {
      // This method is now deprecated - use applyIntensityWindowing instead
      return applyIntensityWindowing(image, minVal, maxVal);
    },
    [applyIntensityWindowing]
  );
  // Apply responsive sizing to the image
  const applyResponsiveSizing = useCallback(
    (image) => {
      if (!image) return;

      // Get display dimensions based on current container size
      const displayDimensions = getDisplayDimensions(image.width, image.height);

      // Calculate display scale factor for scale bar based on actual display size
      // Apply proper intensity windowing (scientific image processing)
      applyIntensityWindowing(
        image,
        liveStreamState.minVal,
        liveStreamState.maxVal
      );

      setCanvasStyle({
        width: `${displayDimensions.width}px`,
        height: `${displayDimensions.height}px`,
        display: "block",
        margin: "20px auto", // Center the canvas horizontally
        objectFit: "contain",
      });

      // Notify parent of image dimensions only if they changed
      if (
        onImageLoad &&
        (image.width !== prevDimensionsRef.current.width ||
          image.height !== prevDimensionsRef.current.height)
      ) {
        prevDimensionsRef.current = {
          width: image.width,
          height: image.height,
        };
        onImageLoad(image.width, image.height);
      }
    },
    [
      applyIntensityWindowing,
      getDisplayDimensions,
      liveStreamState.minVal,
      liveStreamState.maxVal,
      containerSize,
      onImageLoad,
    ]
  );

  // Load and process image when it changes
  useEffect(() => {
    if (liveStreamState.liveViewImage) {
      const img = new Image();
      img.onload = () => {
        try {
          // Set canvas to image's natural size
          const canvas = canvasRef.current;
          if (canvas) {
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
          }
          applyResponsiveSizing(img);
          setImageLoaded(true);
        } catch (error) {
          console.error("Error processing image:", error);
          setImageLoaded(false);
          setCanvasStyle({}); // Reset canvas style on error
        }
      };
      img.onerror = () => {
        console.error("Error loading image");
        setImageLoaded(false);
        setCanvasStyle({}); // Reset canvas style on error
      };
      img.src = `data:image/jpeg;base64,${liveStreamState.liveViewImage}`; // TODO: We could consider rendering the fraame into a canvas
    } else {
      setImageLoaded(false);
      setCanvasStyle({}); // Reset canvas style when no image
    }
  }, [liveStreamState.liveViewImage, applyResponsiveSizing]); // TODO: @gokugiant => let's check if this re-renders/repaints the whole page with every frame

  // Reapply sizing when container size changes
  useEffect(() => {
    if (imageLoaded && liveStreamState.liveViewImage) {
      const img = new Image();
      img.onload = () => applyResponsiveSizing(img);
      img.src = `data:image/jpeg;base64,${liveStreamState.liveViewImage}`;
    }
  }, [
    containerSize,
    imageLoaded,
    liveStreamState.liveViewImage,
    applyResponsiveSizing,
  ]);

  // Update intensity windowing when min/max values change
  useEffect(() => {
    if (imageLoaded && liveStreamState.liveViewImage) {
      // Re-apply intensity windowing when intensity values change
      const img = new Image();
      img.onload = () => {
        applyIntensityWindowing(
          img,
          liveStreamState.minVal,
          liveStreamState.maxVal
        );
      };
      img.src = `data:image/jpeg;base64,${liveStreamState.liveViewImage}`;
    }
  }, [
    liveStreamState.minVal,
    liveStreamState.maxVal,
    imageLoaded,
    liveStreamState.liveViewImage,
    applyIntensityWindowing,
  ]);

  // Handle container resize to maintain aspect ratio
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver(() => {
      // Reprocess the image when container size changes
      if (liveStreamState.liveViewImage && imageLoaded) {
        const img = new Image();
        img.onload = () => {
          applyResponsiveSizing(img);
        };
        img.src = `data:image/jpeg;base64,${liveStreamState.liveViewImage}`;
      }
    });

    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, [liveStreamState.liveViewImage, imageLoaded, applyResponsiveSizing]);

  // Handle intensity range change
  const handleRangeChange = (event, newValue) => {
    dispatch(liveViewSlice.setMinVal(newValue[0]));
    dispatch(liveViewSlice.setMaxVal(newValue[1]));
  };

  // Move positioner to specified real-world coordinates
  const moveToPosition = useCallback(async (x, y) => {
    try {
      console.log(
        `Moving to position X: ${x.toFixed(2)}, Y: ${y.toFixed(2)} µm`
      );

      // Move X axis
      await apiPositionerControllerMovePositioner({
        axis: "X",
        dist: x,
        isAbsolute: false,
        isBlocking: false,
      });

      // Move Y axis
      await apiPositionerControllerMovePositioner({
        axis: "Y",
        dist: y,
        isAbsolute: false,
        isBlocking: false,
      });

      console.log(
        `Successfully moved to position X: ${x.toFixed(2)}, Y: ${y.toFixed(
          2
        )} µm`
      );
    } catch (error) {
      console.error("Error moving to position:", error);
    }
  }, []);

  // Calculate adaptive pixel size based on field of view and canvas dimensions
  const getAdaptivePixelSize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !objectiveState.fovX || canvas.width === 0) {
      return null;
    }
    return objectiveState.fovX / canvas.width;
  }, [objectiveState.fovX]);

  // Handle single click for ROI selection etc.
  const handleCanvasClick = useCallback(
    (event) => {
      const canvas = canvasRef.current;

      if (!canvas || !onClick) return;

      // Get click position relative to canvas
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const clickX = (event.clientX - rect.left) * scaleX;
      const clickY = (event.clientY - rect.top) * scaleY;

      // Provide display info
      const displayInfo = {
        displayWidth: rect.width,
        displayHeight: rect.height,
        scale: 1, // No zoom/pan in LiveViewComponent
        translateX: 0,
        translateY: 0,
      };

      onClick(clickX, clickY, canvas.width, canvas.height, displayInfo);
    },
    [onClick]
  );

  // Handle double-click to move to position
  const handleCanvasDoubleClick = useCallback(
    (event) => {
      const canvas = canvasRef.current;

      if (!canvas) {
        console.warn("Canvas not available for position calculation");
        return;
      }

      // Get click position relative to canvas
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const clickX = (event.clientX - rect.left) * scaleX;
      const clickY = (event.clientY - rect.top) * scaleY;

      // If external handler is provided, use it with image dimensions
      if (onDoubleClick) {
        onDoubleClick(clickX, clickY, canvas.width, canvas.height);
        return;
      }

      // Fallback to original logic if no external handler
      const adaptivePixelSize = getAdaptivePixelSize();

      if (!adaptivePixelSize) {
        console.warn(
          "Field of view (fovX) not available for position calculation"
        );
        return;
      }

      // Convert pixel coordinates to real-world coordinates
      // Center of image is (0,0) in real coordinates
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;

      // Calculate real-world distances from center using adaptive pixel size
      const realX = (clickX - centerX) * adaptivePixelSize;
      const realY = (clickY - centerY) * adaptivePixelSize;

      console.log(
        `Double-click at pixel (${clickX.toFixed(1)}, ${clickY.toFixed(
          1
        )}) -> real coordinates (${realX.toFixed(2)}, ${realY.toFixed(2)}) µm`
      );
      console.log(
        `Adaptive pixel size: ${adaptivePixelSize.toFixed(4)} µm/pixel (fovX: ${
          objectiveState.fovX
        }, canvas width: ${canvas.width})`
      );

      // Move to the calculated position
      // Note: Y direction might need to be inverted depending on stage orientation
      moveToPosition(-realX, -realY); // Inverting Y as microscope Y often goes opposite to image Y
    },
    [onDoubleClick, getAdaptivePixelSize, moveToPosition, objectiveState.fovX]
  );

  // Calculate scale bar dimensions - using adaptive pixel size
  const scaleBarPx = 50;
  const adaptivePixelSize = getAdaptivePixelSize();
  const scaleBarMicrons = adaptivePixelSize
    ? (scaleBarPx * adaptivePixelSize).toFixed(2)
    : null;

  return (
    <Box
      ref={containerRef}
      sx={{
        position: "relative",
        width: "80%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "black",
        overflow: "hidden",
      }}
    >
      {/* Canvas for intensity-scaled image */}
      {liveStreamState.liveViewImage ? (
        <canvas
          ref={canvasRef}
          style={{
            ...canvasStyle,
            display: imageLoaded ? canvasStyle.display : "none",
            cursor: adaptivePixelSize ? "crosshair" : "default",
          }}
          onClick={handleCanvasClick}
          onDoubleClick={handleCanvasDoubleClick}
        />
      ) : (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
          }}
        >
          <Typography>Loading image...</Typography>
        </Box>
      )}

      {/* External overlay content (e.g., ROI overlays) */}
      {overlayContent}

      {/* Scale bar */}
      {scaleBarMicrons && (
        <Box
          sx={{
            position: "absolute",
            bottom: 50,
            transform: "translateX(-10%)",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            zIndex: 4,
          }}
        >
          <Box
            sx={{
              width: `${scaleBarPx}px`,
              height: "10px",
              backgroundColor: "white",
              mr: 2,
            }}
          />
          <Typography variant="body2">{scaleBarMicrons} µm</Typography>
        </Box>
      )}
    </Box>
  );
};

export default LiveViewComponent;
