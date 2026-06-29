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

  // Track FPS for the JPEG stream. Frames are counted in the uc2:jpeg-frame
  // listener below (Redux now updates only ~3 Hz, so we can't infer FPS from it).
  useEffect(() => {
    const id = setInterval(() => {
      const counter = fpsCounterRef.current;
      const now = performance.now();
      const elapsed = now - counter.lastTime;
      if (elapsed >= 1000) {
        const fps = Math.round((counter.frames * 1000) / elapsed);
        dispatch(liveViewSlice.setStats({ fps, bps: 0 })); // bps not available for JPEG
        counter.frames = 0;
        counter.lastTime = now;
      }
    }, 1000);
    return () => clearInterval(id);
  }, [dispatch]);
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
        }),
      );
    } catch (error) {
      console.warn("JPEG histogram computation failed:", error);
    }
  }, [dispatch, liveStreamState.showHistogram]);

  // Linear intensity windowing applied IN PLACE to the already-drawn canvas.
  // Only invoked when the window is non-default ([0,255] is an identity map),
  // so the common live-preview case skips this full-frame pass entirely.
  const windowCanvasInPlace = useCallback((minVal, maxVal) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const range = Math.max(1, maxVal - minVal); // avoid division by zero
    const scale = 255.0 / range;
    // Map luminance through [minVal, maxVal] -> [0,255], scaling RGB to keep colour.
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      let mappedLum;
      if (lum <= minVal) mappedLum = 0;
      else if (lum >= maxVal) mappedLum = 255;
      else mappedLum = (lum - minVal) * scale;
      if (lum > 0) {
        const factor = mappedLum / lum;
        data[i] = Math.min(255, Math.max(0, Math.round(r * factor)));
        data[i + 1] = Math.min(255, Math.max(0, Math.round(g * factor)));
        data[i + 2] = Math.min(255, Math.max(0, Math.round(b * factor)));
      } else {
        const v = Math.round(mappedLum);
        data[i] = v;
        data[i + 1] = v;
        data[i + 2] = v;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, []);

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
    [containerSize],
  );

  // Apply responsive sizing (canvas CSS box) for the given frame dimensions.
  // Pure layout — no pixel work, no decoding.
  const applyResponsiveSizing = useCallback(
    (imgWidth, imgHeight) => {
      if (!imgWidth || !imgHeight) return;

      const displayDimensions = getDisplayDimensions(imgWidth, imgHeight);

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
        (imgWidth !== prevDimensionsRef.current.width ||
          imgHeight !== prevDimensionsRef.current.height)
      ) {
        prevDimensionsRef.current = { width: imgWidth, height: imgHeight };
        onImageLoad(imgWidth, imgHeight);
      }
    },
    [getDisplayDimensions, onImageLoad],
  );

  // ── Efficient JPEG render path ──────────────────────────────────────────
  // Decode each base64 JPEG ONCE via createImageBitmap (off the main thread)
  // and blit it with drawImage — replacing the old path that built several
  // <img src="data:image/jpeg;base64,…"> per frame and ran an always-on
  // per-pixel intensity loop, which pegged the CPU. Frames are coalesced: while
  // one frame is decoding we keep only the most recent incoming frame, so no
  // backlog builds up. This is render-side only — the socket frame_ack
  // backpressure is untouched.
  const latestB64Ref = useRef(null);
  const decodingRef = useRef(false);
  const lastDimsRef = useRef({ width: 0, height: 0 });

  const decodeAndDraw = useCallback(() => {
    if (decodingRef.current) return; // a decode is in flight; it picks up the latest
    const b64 = latestB64Ref.current;
    if (!b64) return;
    decodingRef.current = true;

    let blob;
    try {
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      blob = new Blob([bytes], { type: "image/jpeg" });
    } catch (e) {
      decodingRef.current = false;
      return;
    }

    // Promise.resolve().then(...) so a missing createImageBitmap (old browsers /
    // jsdom test env) becomes a rejected promise handled below, not a throw.
    Promise.resolve()
      .then(() => createImageBitmap(blob))
      .then((bitmap) => {
        try {
          const canvas = canvasRef.current;
          if (canvas) {
            if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
              canvas.width = bitmap.width;
              canvas.height = bitmap.height;
            }
            const ctx = canvas.getContext("2d");
            ctx.drawImage(bitmap, 0, 0);

            // Intensity windowing only when a non-default window is set
            // (a [0,255] window is identity → skip the full-frame pass).
            if (liveStreamState.minVal > 0 || liveStreamState.maxVal < 255) {
              windowCanvasInPlace(liveStreamState.minVal, liveStreamState.maxVal);
            }
            // Histogram (throttled internally; only when the panel is open).
            computeHistogramFromCanvas();

            if (
              bitmap.width !== lastDimsRef.current.width ||
              bitmap.height !== lastDimsRef.current.height
            ) {
              lastDimsRef.current = { width: bitmap.width, height: bitmap.height };
              applyResponsiveSizing(bitmap.width, bitmap.height);
            }
            if (!imageLoaded) setImageLoaded(true);
          }
        } finally {
          bitmap.close();
        }
      })
      .catch((err) => console.error("createImageBitmap failed:", err))
      .finally(() => {
        decodingRef.current = false;
        // A newer frame arrived during decode → render it now (coalesced).
        if (latestB64Ref.current && latestB64Ref.current !== b64) {
          decodeAndDraw();
        }
      });
  }, [
    liveStreamState.minVal,
    liveStreamState.maxVal,
    windowCanvasInPlace,
    computeHistogramFromCanvas,
    applyResponsiveSizing,
    imageLoaded,
  ]);

  // Keep a ref to the latest decodeAndDraw so the mount-once frame listener
  // always uses the current one (which closes over min/max) without re-binding.
  const decodeRef = useRef(decodeAndDraw);
  useEffect(() => {
    decodeRef.current = decodeAndDraw;
  }, [decodeAndDraw]);

  // Live render path: each JPEG frame arrives as an imperative CustomEvent
  // (dispatched by WebSocketHandler), NOT through Redux — so the canvas runs at
  // full FPS while the React tree does not re-render per frame.
  useEffect(() => {
    const onFrame = (e) => {
      const image = e.detail && e.detail.image;
      if (!image) return;
      latestB64Ref.current = image;
      fpsCounterRef.current.frames++;
      decodeRef.current();
    };
    window.addEventListener("uc2:jpeg-frame", onFrame);
    return () => window.removeEventListener("uc2:jpeg-frame", onFrame);
  }, []);

  // Reset the canvas when the stream stops (Redux liveViewImage cleared). The
  // Redux mirror is throttled, but this only needs to fire on stop.
  useEffect(() => {
    if (!liveStreamState.liveViewImage) {
      latestB64Ref.current = null;
      setImageLoaded(false);
      setCanvasStyle({});
    }
  }, [liveStreamState.liveViewImage]);

  // Re-apply responsive sizing when the container resizes — no re-decode needed.
  useEffect(() => {
    if (lastDimsRef.current.width && lastDimsRef.current.height) {
      applyResponsiveSizing(lastDimsRef.current.width, lastDimsRef.current.height);
    }
  }, [containerSize, applyResponsiveSizing]);

  // Handle intensity range change
  const handleRangeChange = (event, newValue) => {
    dispatch(liveViewSlice.setMinVal(newValue[0]));
    dispatch(liveViewSlice.setMaxVal(newValue[1]));
  };

  // Move positioner to specified real-world coordinates
  const moveToPosition = useCallback(async (x, y) => {
    try {
      console.log(
        `Moving to position X: ${x.toFixed(2)}, Y: ${y.toFixed(2)} µm`,
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
          2,
        )} µm`,
      );
    } catch (error) {
      console.error("Error moving to position:", error);
    }
  }, []);

  // Calculate adaptive pixel size based on field of view and canvas dimensions.
  // Primary: fovX (pixelsize * fullSensorWidth) / canvas.width (subsampled frame width)
  //   = pixelsize * subsamplingFactor  →  physical µm per displayed pixel. Correct for
  //   any subsampling level without needing to know the factor explicitly.
  // Fallback: objectiveState.pixelsize * subsamplingFactor when fovX is not yet populated.
  const getAdaptivePixelSize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvas.width === 0) return null;

    // Primary path: fovX already encodes full-sensor FOV; dividing by the
    // subsampled canvas width gives µm/displayed-pixel automatically.
    if (objectiveState.fovX) {
      return objectiveState.fovX / canvas.width;
    }

    // Fallback: compute from pixelsize + per-format subsampling factor.
    // This covers the startup window where fovX has not been fetched yet.
    const pixelsize = objectiveState.pixelsize;
    if (!pixelsize) return null;

    const settings = liveStreamState.streamSettings;
    const currentFormat = liveStreamState.imageFormat || "jpeg";
    let subsamplingFactor = 1;
    if (currentFormat === "binary") {
      subsamplingFactor = settings?.binary?.subsampling?.factor ?? 4;
    } else if (currentFormat === "jpeg") {
      subsamplingFactor = settings?.jpeg?.subsampling?.factor ?? 1;
    } else if (currentFormat === "webrtc") {
      subsamplingFactor = settings?.webrtc?.subsampling_factor ?? 1;
    }

    return pixelsize * subsamplingFactor;
  }, [objectiveState.fovX, objectiveState.pixelsize, liveStreamState.streamSettings, liveStreamState.imageFormat]);

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
    [onClick],
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
          "Field of view (fovX) not available for position calculation",
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
          1,
        )}) -> real coordinates (${realX.toFixed(2)}, ${realY.toFixed(2)}) µm`,
      );
      console.log(
        `Adaptive pixel size: ${adaptivePixelSize.toFixed(4)} µm/pixel (fovX: ${
          objectiveState.fovX
        }, canvas width: ${canvas.width})`,
      );

      // Sign mapping from image axes to stage axes (same convention as
      // LiveViewControlWrapper): both inverted so the clicked feature moves to
      // the centre. Flip the relevant constant per hardware if an axis goes the
      // wrong way.
      const IMAGE_TO_STAGE_SIGN_X = 1;
      const IMAGE_TO_STAGE_SIGN_Y = 1;
      moveToPosition(
        IMAGE_TO_STAGE_SIGN_X * realX,
        IMAGE_TO_STAGE_SIGN_Y * realY,
      );
    },
    [onDoubleClick, getAdaptivePixelSize, moveToPosition, objectiveState.fovX],
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
      {/* Canvas is always mounted so the imperative uc2:jpeg-frame listener can
          draw into it immediately; it stays hidden until the first frame is
          decoded (then imageLoaded flips it visible). */}
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
      {!imageLoaded && (
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
