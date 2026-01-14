import React, { useEffect, useRef, useState, useCallback } from "react";
import { useSelector, useDispatch } from "react-redux";
import { Box, Alert, CircularProgress, Typography } from "@mui/material";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";

/**
 * WebRTC viewer component for real-time low-latency video streaming.
 * Based on webrtc_stream.html implementation.
 */
const WebRTCViewer = ({ detectorName, onDoubleClick }) => {
  const dispatch = useDispatch();
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const statsIntervalRef = useRef(null);
  const statsCounterRef = useRef({
    frames: 0,
    bytes: 0,
    lastTime: performance.now(),
  });

  const [connectionState, setConnectionState] = useState("disconnected");
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);

  // Get WebRTC settings from Redux
  const streamSettings = useSelector(
    (state) => state.liveStreamState?.streamSettings
  );
  const webrtcSettings = streamSettings?.webrtc || {};

  // Get connection settings from Redux
  const connectionSettings = useSelector(
    (state) => state.connectionSettingsState
  );
  // Build base URL for WebRTC streaming - requires full URL
  const baseUrl = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api`;

  // Detect if connection is local
  const isLocalConnection = useCallback(() => {
    const url = window.location.hostname;
    return (
      url === "localhost" ||
      url === "127.0.0.1" ||
      url.startsWith("192.168.") ||
      url.startsWith("10.0.") ||
      url.startsWith("100.") || // tailscale
      url.startsWith("172.16.")
    );
  }, []);

  // Create peer connection
  const createPeerConnection = useCallback(() => {
    console.log("Creating WebRTC peer connection...");

    // Configure ICE servers based on connection type
    const isLocal = isLocalConnection();
    let config = {};

    if (!isLocal) {
      // For remote connections, use STUN servers
      config.iceServers = [
        { urls: ["stun:stun.l.google.com:19302"] },
        { urls: ["stun:stun1.l.google.com:19302"] },
      ];
      console.log("Remote connection detected, using STUN servers");
    } else {
      // For local connections, no ICE servers needed
      config.iceServers = [];
      console.log("Local connection detected, no STUN servers needed");
    }

    config.iceCandidatePoolSize = isLocal ? 0 : 4;

    const pc = new RTCPeerConnection(config);

    // Handle incoming tracks
    pc.addEventListener("track", (evt) => {
      console.log("Received video track:", evt.track.kind);
      if (videoRef.current && evt.streams && evt.streams[0]) {
        videoRef.current.srcObject = evt.streams[0];
        console.log("Video stream attached to video element");
      }
    });

    // Handle connection state changes
    pc.addEventListener("connectionstatechange", () => {
      console.log("Connection state changed:", pc.connectionState);
      setConnectionState(pc.connectionState);

      if (pc.connectionState === "connected") {
        console.log("WebRTC connection established!");
        setError(null);
        startStatsMonitoring();
      } else if (pc.connectionState === "failed") {
        console.error("WebRTC connection failed");
        setError("Connection failed");
        stopStatsMonitoring();
      } else if (
        pc.connectionState === "disconnected" ||
        pc.connectionState === "closed"
      ) {
        console.log("WebRTC connection closed");
        stopStatsMonitoring();
      }
    });

    // Handle ICE connection state
    pc.addEventListener("iceconnectionstatechange", () => {
      console.log("ICE connection state:", pc.iceConnectionState);
    });

    // Handle ICE candidate errors
    pc.addEventListener("icecandidateerror", (event) => {
      console.warn("ICE candidate error:", event);
    });

    pcRef.current = pc;
    return pc;
  }, [isLocalConnection]);

  // Start stats monitoring
  const startStatsMonitoring = useCallback(() => {
    if (statsIntervalRef.current) return;

    // Reset counter on start
    statsCounterRef.current = {
      frames: 0,
      bytes: 0,
      lastTime: performance.now(),
    };

    statsIntervalRef.current = setInterval(async () => {
      if (!pcRef.current) return;

      try {
        const stats = await pcRef.current.getStats(null);
        let inboundStats = null;

        stats.forEach((report) => {
          if (report.type === "inbound-rtp" && report.kind === "video") {
            inboundStats = {
              bytesReceived: report.bytesReceived || 0,
              packetsReceived: report.packetsReceived || 0,
              packetsLost: report.packetsLost || 0,
              framesDecoded: report.framesDecoded || 0,
              timestamp: report.timestamp || 0,
            };
          }
        });

        if (inboundStats) {
          setStats(inboundStats);

          // Calculate FPS and BPS for Redux
          const counter = statsCounterRef.current;
          const now = performance.now();
          const elapsed = now - counter.lastTime;

          if (elapsed >= 1000) {
            const framesDelta = inboundStats.framesDecoded - counter.frames;
            const bytesDelta = inboundStats.bytesReceived - counter.bytes;

            const fps = Math.round((framesDelta * 1000) / elapsed);
            const bps = Math.round((bytesDelta * 8) / (elapsed / 1000)); // bits per second

            // Update Redux stats
            dispatch(liveStreamSlice.setStats({ fps, bps }));
            console.log(`[WebRTCViewer] FPS: ${fps}, BPS: ${bps}`);

            // Reset counters
            counter.frames = inboundStats.framesDecoded;
            counter.bytes = inboundStats.bytesReceived;
            counter.lastTime = now;
          } else if (counter.frames === 0) {
            // Initialize on first stats
            counter.frames = inboundStats.framesDecoded;
            counter.bytes = inboundStats.bytesReceived;
          }
        }
      } catch (err) {
        console.warn("Failed to get stats:", err);
      }
    }, 1000);
  }, [dispatch]);

  // Stop stats monitoring
  const stopStatsMonitoring = useCallback(() => {
    if (statsIntervalRef.current) {
      clearInterval(statsIntervalRef.current);
      statsIntervalRef.current = null;
    }
    setStats(null);
    // Reset Redux stats
    dispatch(liveStreamSlice.setStats({ fps: 0, bps: 0 }));
  }, [dispatch]);

  // Negotiate WebRTC connection
  const negotiate = useCallback(async () => {
    if (!pcRef.current) return;

    const pc = pcRef.current;
    const timings = {
      start: performance.now(),
      offerCreated: 0,
      localDescSet: 0,
      iceComplete: 0,
      requestSent: 0,
      responseReceived: 0,
      remoteDescSet: 0,
    };

    try {
      console.log("Starting WebRTC negotiation...");
      setConnectionState("connecting");

      // Add transceiver to receive video
      const transceiver = pc.addTransceiver("video", {
        direction: "recvonly",
        streams: [],
      });

      // Prefer VP8 for local connections (lower latency)
      const isLocal = isLocalConnection();
      const codecs = RTCRtpReceiver.getCapabilities("video").codecs;
      const preferredCodecs = [];

      if (isLocal) {
        preferredCodecs.push(
          ...codecs.filter((c) => c.mimeType.includes("VP8"))
        );
        preferredCodecs.push(
          ...codecs.filter((c) => !c.mimeType.includes("VP8"))
        );
      } else {
        preferredCodecs.push(
          ...codecs.filter((c) => c.mimeType.includes("H264"))
        );
        preferredCodecs.push(
          ...codecs.filter((c) => !c.mimeType.includes("H264"))
        );
      }

      if (preferredCodecs.length > 0) {
        await transceiver.setCodecPreferences(preferredCodecs);
      }

      // Create offer
      console.log("⚡ Creating offer...");
      const offer = await pc.createOffer({
        offerToReceiveVideo: true,
        offerToReceiveAudio: false,
        iceRestart: false,
      });
      timings.offerCreated = performance.now();
      console.log(
        `Offer created in ${(timings.offerCreated - timings.start).toFixed(
          1
        )}ms`
      );

      // Set local description
      await pc.setLocalDescription(offer);
      timings.localDescSet = performance.now();
      console.log(
        `Local description set in ${(
          timings.localDescSet - timings.offerCreated
        ).toFixed(1)}ms`
      );

      // Wait for ICE gathering to complete
      const gatheringTimeout = isLocal ? 500 : 2000;
      console.log("Waiting for ICE gathering...");

      await new Promise((resolve) => {
        let timeoutId = null;

        const checkState = () => {
          if (pc.iceGatheringState === "complete") {
            if (timeoutId) clearTimeout(timeoutId);
            pc.removeEventListener("icegatheringstatechange", checkState);
            resolve();
          }
        };

        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          pc.addEventListener("icegatheringstatechange", checkState);
          timeoutId = setTimeout(() => {
            pc.removeEventListener("icegatheringstatechange", checkState);
            console.log("⚠️ ICE gathering timeout, continuing anyway");
            resolve();
          }, gatheringTimeout);
        }
      });

      timings.iceComplete = performance.now();
      console.log(
        `ICE gathering completed in ${(
          timings.iceComplete - timings.localDescSet
        ).toFixed(1)}ms`
      );

      // Prepare request payload
      const payload = {
        sdp: pc.localDescription.sdp,
        sdp_type: pc.localDescription.type,
        detectorName: detectorName,
        params: {
          throttle_ms: webrtcSettings.throttle_ms || 33,
          subsampling_factor: webrtcSettings.subsampling_factor || 1,
          max_width: webrtcSettings.max_width || 1280,
        },
      };

      console.log("📡 Sending offer to server...", payload.params);
      timings.requestSent = performance.now();

      // Send offer to backend
      const response = await fetch(
        `${baseUrl}/LiveViewController/webrtc_offer`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      timings.responseReceived = performance.now();
      console.log(
        `⏱️  Server response received in ${(
          timings.responseReceived - timings.requestSent
        ).toFixed(1)}ms`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const answer = await response.json();
      console.log("✅ Received answer from server");

      if (answer.status === "error") {
        throw new Error(answer.message || "Server error");
      }

      // Set remote description
      console.log("🔄 Setting remote description...");
      await pc.setRemoteDescription(
        new RTCSessionDescription({
          sdp: answer.sdp,
          type: answer.type,
        })
      );

      timings.remoteDescSet = performance.now();
      console.log(
        `⏱️  Remote description set in ${(
          timings.remoteDescSet - timings.responseReceived
        ).toFixed(1)}ms`
      );

      // Print complete timing breakdown
      const totalTime = timings.remoteDescSet - timings.start;
      console.log("🕐 Complete timing breakdown:");
      console.log(`   Total time: ${totalTime.toFixed(1)}ms`);
      console.log(
        `   Offer creation: ${(timings.offerCreated - timings.start).toFixed(
          1
        )}ms`
      );
      console.log(
        `   Local desc set: ${(
          timings.localDescSet - timings.offerCreated
        ).toFixed(1)}ms`
      );
      console.log(
        `   ICE gathering: ${(
          timings.iceComplete - timings.localDescSet
        ).toFixed(1)}ms`
      );
      console.log(
        `   Server response: ${(
          timings.responseReceived - timings.requestSent
        ).toFixed(1)}ms`
      );
      console.log(
        `   Remote desc set: ${(
          timings.remoteDescSet - timings.responseReceived
        ).toFixed(1)}ms`
      );

      console.log("✅ WebRTC negotiation completed!");
    } catch (err) {
      console.error("❌ Negotiation error:", err);
      setError(err.message);
      setConnectionState("failed");
    }
  }, [detectorName, webrtcSettings, isLocalConnection]);

  // Start WebRTC connection on mount ONLY
  useEffect(() => {
    console.log("🎬 WebRTCViewer mounted, starting connection...");

    // Create peer connection and start negotiation
    createPeerConnection();
    negotiate();

    // Cleanup on unmount
    return () => {
      console.log("🛑 WebRTCViewer unmounting, cleaning up...");
      stopStatsMonitoring();

      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }

      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps = mount/unmount only

  // DO NOT reconnect on settings change - settings are applied on mount only
  // WebRTC backend handles parameter updates via setStreamParameters
  // Reconnecting on every settings change causes endless loop because
  // the connection closes immediately after connecting
  //
  // If settings need to be updated, user should:
  // 1. Change settings in UI
  // 2. Click Submit (which calls setStreamParameters)
  // 3. Manually restart stream (stop -> start) if needed

  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#000",
        position: "relative",
      }}
    >
      {/* Connection status */}
      {connectionState === "connecting" && (
        <Box sx={{ position: "absolute", top: 16, left: 16, zIndex: 10 }}>
          <Alert severity="info" icon={<CircularProgress size={20} />}>
            Connecting to WebRTC stream...
          </Alert>
        </Box>
      )}

      {/* Error message */}
      {error && (
        <Box sx={{ position: "absolute", top: 16, left: 16, zIndex: 10 }}>
          <Alert severity="error">{error}</Alert>
        </Box>
      )}

      {/* Stats overlay */}
      {stats && connectionState === "connected" && (
        <Box
          sx={{
            position: "absolute",
            top: 16,
            right: 16,
            zIndex: 10,
            backgroundColor: "rgba(0, 0, 0, 0.7)",
            color: "#fff",
            padding: 1,
            borderRadius: 1,
            fontFamily: "monospace",
            fontSize: 12,
          }}
        >
          <Typography variant="caption" component="div">
            Packets: {stats.packetsReceived} (Lost: {stats.packetsLost})
          </Typography>
          <Typography variant="caption" component="div">
            Frames: {stats.framesDecoded}
          </Typography>
          <Typography variant="caption" component="div">
            Data: {(stats.bytesReceived / 1024 / 1024).toFixed(2)} MB
          </Typography>
        </Box>
      )}

      {/* Video element */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        controls
        onDoubleClick={onDoubleClick}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "contain", // Preserve aspect ratio and fill available space
          display: "block",
        }}
      />

      {/* Connection indicator */}
      {connectionState === "connected" && (
        <Box
          sx={{
            position: "absolute",
            bottom: 16,
            left: 16,
            zIndex: 10,
            backgroundColor: "rgba(40, 167, 69, 0.8)",
            color: "#fff",
            padding: "4px 8px",
            borderRadius: 1,
            fontSize: 12,
            fontWeight: "bold",
          }}
        >
          ● LIVE (WebRTC)
        </Box>
      )}
    </Box>
  );
};

export default WebRTCViewer;
