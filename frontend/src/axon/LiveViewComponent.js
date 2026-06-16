import React, { useRef, useEffect, useState, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box, Typography } from "@mui/material";
import * as liveViewSlice from "../state/slices/LiveStreamSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";
import { frameBus } from "../stream/frameBus.js";

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

  // FPS is now counted in the imperative paint path (paintFrameUrl)
  // since the hot path no longer flows through Redux/liveViewImage.
  const prevDimensionsRef = useRef({ width: 0, height: 0 }); // Track dimensions to avoid redundant callbacks
  // Monotonic counters that gate the JPEG-frame paint pipeline. Each
  // arriving frame gets a unique ``decodeIdRef`` value; that frame's
  // async <img> onload only writes the canvas if no newer frame has
  // already painted (``lastPaintedIdRef``). This keeps a slow decode
  // from painting over a fresher frame, without cancelling decodes
  // (cancel-on-next-frame starved every paint on slow platforms and
  // showed a solid black canvas).
  const decodeIdRef = useRef(0);
  const lastPaintedIdRef = useRef(0);
  // Cache of the most recently decoded frame (an HTMLImageElement).
  // Resize / intensity-window changes re-render FROM this cache instead
  // of decoding the base64 frame again — one decode per frame, not
  // three.
  const lastImageRef = useRef(null);
  // Track mount state so async callbacks don't poke setState on an
  // unmounted component.
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [containerSize, setContainerSize] = useState({
    width: 800,
    height: 600,
  });
  const [canvasStyle, setCanvasStyle] = useState({});

  // JPEG histogram from the canvas is disabled: getImageData + a
  // full-image luminance walk per frame was a meaningful chunk of the
  // per-frame budget, and the live histogram isn't worth that cost on
  // the streaming hot path. Kept as a no-op so call sites stay simple;
  // re-enable behind a throttle + an explicit "show histogram" gate if
  // it's ever needed again.
  const computeHistogramFromCanvas = useCallback(() => {}, []);

  // Optimized intensity windowing - proper scientific image processing
  const applyIntensityWindowing = useCallback(
    (image, minVal, maxVal) => {
      const canvas = canvasRef.current;
      if (!canvas || !image) return;

      canvas.width = image.width;
      canvas.height = image.height;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0);

      // Fast path: an identity window (the JPEG default, 0..255) maps
      // every pixel to itself. The per-pixel loop below would scan the
      // whole image — millions of array ops — to reproduce a
      // bit-identical result. For a live JPEG stream that ran several
      // times PER FRAME and was the single biggest cause of jank.
      // Skip it whenever the window is the full 8-bit range; only pay
      // for the pixel walk once the user actually narrows the slider.
      if (minVal <= 0 && maxVal >= 255) {
        computeHistogramFromCanvas();
        return;
      }

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
    [computeHistogramFromCanvas],
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
    [containerSize],
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
        liveStreamState.maxVal,
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
    ],
  );

  // Keep a ref to the latest applyResponsiveSizing so the frameBus
  // subscriber (registered once) always calls the current closure
  // (which captures the up-to-date min/max/containerSize).
  const applyResponsiveSizingRef = useRef(applyResponsiveSizing);
  applyResponsiveSizingRef.current = applyResponsiveSizing;
  // Set the "has image" state only once (avoids a setState per frame).
  const imageLoadedRef = useRef(false);

  // Paint a frame URL onto the canvas IMPERATIVELY — no React render,
  // no Redux dispatch. This is the streaming hot path: it runs at full
  // frame rate driven by frameBus, completely off the React commit
  // cycle. That is the whole point — the previous design dispatched
  // every frame into Redux, which re-rendered ~20 components (incl. the
  // react-zoom-pan-pinch wrapper) 16×/sec and blocked the main thread,
  // delaying the <img> onload to ~180 ms. MJPEG is smooth precisely
  // because it never touches React per frame; this restores that.
  //
  // A monotonic id guards against a slow decode painting over a newer
  // frame that already landed.
  const paintFrameUrl = useCallback(
    (src) => {
      if (!src || !mountedRef.current) return;
      const myId = ++decodeIdRef.current;
      const img = new Image();
      img.onload = () => {
        if (!mountedRef.current || myId < lastPaintedIdRef.current) return;
        lastPaintedIdRef.current = myId;
        lastImageRef.current = img;
        try {
          // Draws the image, applies the intensity window (identity
          // fast-path for the JPEG default), sizes the canvas.
          applyResponsiveSizingRef.current(img);
          if (!imageLoadedRef.current) {
            imageLoadedRef.current = true;
            setImageLoaded(true);
          }
        } catch (error) {
          console.error("Error processing image:", error);
        }

        // FPS (real paint rate) — one Redux dispatch per second.
        const fc = fpsCounterRef.current;
        fc.frames++;
        const tnow = performance.now();
        const fElapsed = tnow - fc.lastTime;
        if (fElapsed >= 1000) {
          dispatch(
            liveViewSlice.setStats({
              fps: Math.round((fc.frames * 1000) / fElapsed),
              bps: 0,
            }),
          );
          fc.frames = 0;
          fc.lastTime = tnow;
        }
      };
      img.onerror = () => {
        // A blob URL that already expired (e.g. on remount) fails
        // silently; the next published frame paints.
      };
      img.src = src;
    },
    [dispatch],
  );

  // Subscribe to the off-Redux frame stream ONCE. Paints every frame
  // imperatively (paintFrameUrl). Also paints whatever frame is already
  // buffered so a freshly-mounted viewer shows the last image instead
  // of a blank canvas.
  useEffect(() => {
    paintFrameUrl(frameBus.getLatest());
    return frameBus.subscribe(paintFrameUrl);
  }, [paintFrameUrl]);

  // Re-apply sizing when the container resizes — reuse the cached
  // decoded frame, do NOT re-decode. (containerSize does not change
  // per frame, so this never runs on the streaming hot path.)
  useEffect(() => {
    if (lastImageRef.current) {
      applyResponsiveSizing(lastImageRef.current);
    }
  }, [containerSize, applyResponsiveSizing]);

  // Re-apply intensity windowing when the user moves the min/max
  // slider — reuse the cached decoded frame, do NOT re-decode.
  // (min/max do not change per frame.)
  useEffect(() => {
    if (lastImageRef.current) {
      applyIntensityWindowing(
        lastImageRef.current,
        liveStreamState.minVal,
        liveStreamState.maxVal,
      );
    }
  }, [
    liveStreamState.minVal,
    liveStreamState.maxVal,
    applyIntensityWindowing,
  ]);

  // Handle container resize to maintain aspect ratio. Reuses the
  // cached decoded frame — no per-frame re-decode, and the observer
  // is created once (deps no longer include liveViewImage, so it
  // isn't torn down and rebuilt on every frame).
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver(() => {
      if (lastImageRef.current) {
        applyResponsiveSizing(lastImageRef.current);
      }
    });

    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, [applyResponsiveSizing]);

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

      // Move to the calculated position
      // Note: Y direction might need to be inverted depending on stage orientation
      moveToPosition(-realX, -realY); // Inverting Y as microscope Y often goes opposite to image Y
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
      {/* Canvas for intensity-scaled image */}
      {imageLoaded || liveStreamState.liveViewImage ? (
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
