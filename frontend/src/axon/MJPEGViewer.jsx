/**
 * MJPEGViewer — direct HTTP MJPEG viewer.
 *
 * Bypasses the socket.io/MessagePack path entirely. The browser opens a
 * long-lived ``multipart/x-mixed-replace`` connection to
 * ``/imswitch/api/LiveViewController/mjpeg_stream`` and the platform's
 * built-in MJPEG decoder paints each part as soon as it arrives — so
 * we don't pay any per-frame Python serialisation cost on the server
 * and we don't pay any JS decode/canvas cost on the client.
 *
 * Trade-offs vs. socket.io JPEG:
 *   - No per-frame ack/backpressure (the TCP socket itself handles
 *     flow control). On a slow client the kernel will pause the
 *     server's writes, which on Pi 5 just slows the FPS rather than
 *     building a queue.
 *   - No metadata sidecar (frame_id, server_timestamp, pixel_size).
 *     Anything that needs that should use the JPEG protocol instead.
 *   - Works particularly well on Windows where the socket.io transport
 *     upgrade from long-poll → websocket is unreliable; the MJPEG
 *     stream is a plain HTTP GET with no upgrade dance.
 */

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Box, Typography } from "@mui/material";
import { useSelector } from "react-redux";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";

const MJPEGViewer = ({ onClick, onDoubleClick, onImageLoad, overlayContent }) => {
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState,
  );
  const liveStreamState = useSelector(
    liveStreamSlice.getLiveStreamState,
  );
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const imgRef = useRef(null);

  // A counter we append as a query param so the browser re-opens a
  // fresh connection if the user toggles the stream, switches
  // detectors, or switches protocols. Without this, the <img> can
  // hang on a stale, closed multipart response when the backend has
  // just torn the previous worker down.
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    setNonce((n) => n + 1);
  }, [
    liveViewState.isStreamRunning,
    liveViewState.activeTab,
    liveStreamState.imageFormat,
  ]);

  const detectorName = useMemo(() => {
    const detectors = liveViewState.detectors || [];
    if (!detectors.length) return null;
    return detectors[liveViewState.activeTab] || detectors[0] || null;
  }, [liveViewState.detectors, liveViewState.activeTab]);

  const src = useMemo(() => {
    if (!liveViewState.isStreamRunning) return null;
    const base = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api/LiveViewController/mjpeg_stream`;
    const params = new URLSearchParams({ startStream: "true" });
    if (detectorName) params.set("detectorName", detectorName);
    // The nonce defeats the HTTP cache for the very first GET after a
    // restart; subsequent frames keep flowing on the same connection.
    params.set("t", String(nonce));
    return `${base}?${params.toString()}`;
  }, [
    connectionSettings.ip,
    connectionSettings.apiPort,
    detectorName,
    nonce,
    liveViewState.isStreamRunning,
  ]);

  // Pixel-coordinate translation for parent callbacks. The browser
  // gives us mouse coordinates relative to the displayed <img>; we
  // multiply by ``natural / displayed`` to get sensor-pixel
  // coordinates. Matches what LiveViewComponent does for the canvas.
  const computePixelCoords = (event) => {
    const img = imgRef.current;
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const naturalW = img.naturalWidth || img.width || 1;
    const naturalH = img.naturalHeight || img.height || 1;
    const sx = naturalW / (rect.width || 1);
    const sy = naturalH / (rect.height || 1);
    return {
      pixelX: Math.round(localX * sx),
      pixelY: Math.round(localY * sy),
      imageWidth: naturalW,
      imageHeight: naturalH,
    };
  };

  const handleClick = (event) => {
    if (!onClick) return;
    const c = computePixelCoords(event);
    if (!c) return;
    onClick(c.pixelX, c.pixelY, c.imageWidth, c.imageHeight, {});
  };

  const handleDoubleClick = (event) => {
    if (!onDoubleClick) return;
    const c = computePixelCoords(event);
    if (!c) return;
    onDoubleClick(c.pixelX, c.pixelY, c.imageWidth, c.imageHeight);
  };

  const handleLoad = () => {
    if (!onImageLoad || !imgRef.current) return;
    const w = imgRef.current.naturalWidth;
    const h = imgRef.current.naturalHeight;
    if (w && h) onImageLoad(w, h);
  };

  if (!src) {
    return (
      <Box
        sx={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "text.disabled",
        }}
      >
        <Typography variant="body2">MJPEG stream not active</Typography>
      </Box>
    );
  }

  // Parent layout note: ``LiveViewControlWrapper`` renders viewers
  // inside a ``flex-direction: column`` div that has no fixed height.
  // ``height: 100%`` on this Box collapses unless the parent
  // explicitly allocates space — which it doesn't. So we let the img
  // dictate the box height via ``width: 100%; height: auto``, with a
  // ``maxHeight: 80vh`` cap so portrait-orientation sensors don't push
  // the rest of the page off-screen. ``objectFit: contain`` keeps the
  // aspect ratio crisp inside that frame.
  return (
    <Box
      sx={{
        position: "relative",
        width: "100%",
        bgcolor: "#000",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
      }}
    >
      <img
        ref={imgRef}
        src={src}
        alt="MJPEG live stream"
        onLoad={handleLoad}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        style={{
          display: "block",
          width: "100%",
          height: "auto",
          maxHeight: "80vh",
          objectFit: "contain",
          WebkitUserSelect: "none",
        }}
      />
      {overlayContent}
    </Box>
  );
};

export default MJPEGViewer;
