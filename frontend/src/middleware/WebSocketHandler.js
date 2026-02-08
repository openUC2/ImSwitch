import { useEffect, useRef, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import store from "../state/store.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as webSocketSlice from "../state/slices/WebSocketSlice.js";
import * as experimentStateSlice from "../state/slices/ExperimentStateSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import * as tileStreamSlice from "../state/slices/TileStreamSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import * as omeZarrSlice from "../state/slices/OmeZarrTileStreamSlice.js";
import * as focusLockSlice from "../state/slices/FocusLockSlice.js";
import * as mazeGameSlice from "../state/slices/MazeGameSlice.js";
import * as autofocusSlice from "../state/slices/AutofocusSlice.js";
import * as socketDebugSlice from "../state/slices/SocketDebugSlice.js";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";
import * as canOtaSlice from "../state/slices/canOtaSlice.js";
import * as usbFlashSlice from "../state/slices/usbFlashSlice.js";
import * as laserSlice from "../state/slices/LaserSlice.js";
import * as lightsheetSlice from "../state/slices/LightsheetSlice";


import { io } from "socket.io-client";

// MessagePack for efficient binary serialization
import { decode as msgpackDecode } from "@msgpack/msgpack";

// Import API to check livestream status
import apiViewControllerGetLiveViewActive from "../backendapi/apiViewControllerGetLiveViewActive.js";
import { fetchWithTimeout } from "../utils/fetchWithTimeout";

//##################################################################################
const WebSocketHandler = () => {
  const dispatch = useDispatch();
  const connectionCheckRef = useRef(null);

  // Per-instance server capabilities (each tab has its own)
  const serverCapabilitiesRef = useRef({
    messagepack: false,
    binary_streaming: false,
    protocol_version: "unknown",
  });

  // Access global Redux state
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const hostIP = connectionSettingsState.ip;
  const hostPort = connectionSettingsState.apiPort;

  // Memoized connection check function
  const checkUc2Connection = useCallback(
    async (ip = hostIP, port = hostPort) => {
      // Skip monitoring if backend connection is not configured
      if (!ip || !port) {
        dispatch(uc2Slice.setUc2Connected(false));
        return;
      }

      try {
        console.debug(`Checking UC2 connection to ${ip}:${port}/imswitch`);

        // Use cross-browser compatible fetch with timeout
        const response = await fetchWithTimeout(
          `${ip}:${port}/imswitch/api/UC2ConfigController/is_connected`,
          { method: "GET" },
          10000
        );

        if (response.ok) {
          const data = await response.json();
          const hardwareConnected = data === true;

          console.debug(
            `Backend API: Connected, Hardware: ${
              hardwareConnected ? "Connected" : "Disconnected"
            }`
          );

          // Update BOTH statuses
          dispatch(uc2Slice.setBackendConnected(true)); // API is reachable
          dispatch(uc2Slice.setUc2Connected(hardwareConnected)); // Hardware status

          return hardwareConnected; // Return hardware status for compatibility
        } else {
          console.debug(`UC2 connection check: HTTP ${response.status}`);
          dispatch(uc2Slice.setBackendConnected(false));
          dispatch(uc2Slice.setUc2Connected(false));
          return false;
        }
      } catch (error) {
        if (error.name === "AbortError") {
          console.debug("UC2 connection check: Request timeout");
        } else {
          console.debug("UC2 connection check: Network error", error.message);
        }
        dispatch(uc2Slice.setBackendConnected(false));
        dispatch(uc2Slice.setUc2Connected(false));
        return false;
      }
    },
    [hostIP, hostPort, dispatch] // Dependencies for useCallback
  );

  // Sync livestream status with backend
  const syncLivestreamStatus = useCallback(async () => {
    try {
      console.log("[WebSocket] Syncing livestream status with backend...");
      const isActive = await apiViewControllerGetLiveViewActive();
      console.log(`[WebSocket] Backend livestream status: ${isActive}`);

      // Update Redux state to match backend
      dispatch(liveViewSlice.setIsStreamRunning(isActive));

      return isActive;
    } catch (error) {
      console.error("[WebSocket] Failed to sync livestream status:", error);
      // On error, assume stream is not running to prevent incorrect state
      dispatch(liveViewSlice.setIsStreamRunning(false));
      return false;
    }
  }, [dispatch]);

  // WebSocket connection test
  const testWebSocketConnection = useCallback(
    async (ip, port) => {
      const testIP = ip || hostIP;
      const testPort = port || connectionSettingsState.websocketPort;

      if (!testIP || !testPort) {
        dispatch(webSocketSlice.setTestStatus("failed"));
        return false;
      }

      // Extract protocol from IP settings (following Copilot Instructions)
      let protocol = "http"; // Default fallback
      let cleanIP;

      if (testIP.startsWith("https://")) {
        protocol = "https";
        cleanIP = testIP.replace(/^https?:\/\//, "");
      } else if (testIP.startsWith("http://")) {
        protocol = "http";
        cleanIP = testIP.replace(/^https?:\/\//, "");
      } else {
        // No protocol specified, use HTTP as default
        cleanIP = testIP;
      }

      const socketIOUrl = `${protocol}://${cleanIP}:${testPort}`;

      console.log(
        `Testing Socket.IO connection to: ${socketIOUrl} (protocol: ${protocol})`
      );
      dispatch(webSocketSlice.setTestStatus("testing"));

      return new Promise((resolve) => {
        try {
          const testSocket = io(socketIOUrl, {
            path: '/imswitch/socket.io/',
            transports: ["websocket"], // Force WebSocket transport
            timeout: 5000,
            forceNew: true, // Create new connection for test
            autoConnect: false, // Manual connection control
            secure: protocol === "https", // Enable secure connection for HTTPS
          });

          const timeout = setTimeout(() => {
            testSocket.disconnect();
            dispatch(webSocketSlice.setTestStatus("timeout"));
            console.log("Socket.IO test: Timeout after 5 seconds");
            resolve(false);
          }, 5000);

          testSocket.on("connect", () => {
            clearTimeout(timeout);
            dispatch(webSocketSlice.setTestStatus("success"));
            console.log(
              `Socket.IO test: Connection successful via ${protocol}`
            );
            testSocket.disconnect(); // Disconnect test connection immediately
            resolve(true);
          });

          testSocket.on("connect_error", (error) => {
            clearTimeout(timeout);
            dispatch(webSocketSlice.setTestStatus("failed"));
            console.log(
              `Socket.IO test: Connection failed via ${protocol}`,
              error.message
            );
            testSocket.disconnect();
            resolve(false);
          });

          // Start the connection test
          testSocket.connect();
        } catch (error) {
          dispatch(webSocketSlice.setTestStatus("failed"));
          console.error("Socket.IO test: Exception during connection", error);
          resolve(false);
        }
      });
    },
    [hostIP, connectionSettingsState.websocketPort, dispatch]
  );

  // Listen for manual connection check requests
  useEffect(() => {
    const handleManualConnectionCheck = async (event) => {
      console.log("Manual connection check triggered (HTTP + WebSocket)");
      const { ip, port, websocketPort } = event.detail || {};

      // Test HTTP connection first
      const httpResult = await checkUc2Connection(ip, port);

      // Test WebSocket connection if websocketPort is provided
      if (websocketPort) {
        const wsResult = await testWebSocketConnection(ip, websocketPort);
        console.log(
          `Connection test results - HTTP: ${httpResult}, WebSocket: ${wsResult}`
        );
      }
    };

    // Add event listener for manual connection checks
    window.addEventListener(
      "imswitch:checkConnection",
      handleManualConnectionCheck
    );

    return () => {
      window.removeEventListener(
        "imswitch:checkConnection",
        handleManualConnectionCheck
      );
    };
  }, [checkUc2Connection, testWebSocketConnection]);

  // Listen for WebSocket test requests
  useEffect(() => {
    const handleWebSocketTest = async (event) => {
      console.log("WebSocket connection test triggered");
      const { ip, websocketPort } = event.detail || {};

      const result = await testWebSocketConnection(ip, websocketPort);
      console.log(`WebSocket test result: ${result}`);
    };

    window.addEventListener("imswitch:testWebSocket", handleWebSocketTest);

    return () => {
      window.removeEventListener("imswitch:testWebSocket", handleWebSocketTest);
    };
  }, [testWebSocketConnection]);

  //##################################################################################
  useEffect(() => {
    // Extract protocol from IP settings (following ImSwitch-ReactAPP patterns)
    let protocol = "http"; // Default fallback
    let cleanIP;

    if (connectionSettingsState.ip.startsWith("https://")) {
      protocol = "https";
      cleanIP = connectionSettingsState.ip.replace(/^https?:\/\//, "");
    } else if (connectionSettingsState.ip.startsWith("http://")) {
      protocol = "http";
      cleanIP = connectionSettingsState.ip.replace(/^https?:\/\//, "");
    } else {
      // No protocol specified, use HTTP as default
      cleanIP = connectionSettingsState.ip;
    }

    // Create the socket address with proper protocol
    const address = `${protocol}://${cleanIP}:${connectionSettingsState.websocketPort}`;
    console.log("WebSocket: Creating new connection to", address);

    // Create new socket instance for this component/tab
    const socket = io(address, {
      path: '/imswitch/socket.io/',
      transports: ["websocket"],
      secure: protocol === "https", // Enable secure connection for HTTPS
    });

    //listen to all
    socket.on("*", (packet) => {
      const [eventName, data] = packet.data;
      console.log(`WebSocket received event: ${eventName}`, data);
    });

    // Listen for a connection confirmation
    socket.on("connect", () => {
      console.log(`WebSocket connected with socket id: ${socket.id}`);
      //update redux state
      dispatch(webSocketSlice.setConnected(true));

      // Sync livestream status with backend on connect/reconnect
      // This ensures frontend state matches backend state after backend restart
      syncLivestreamStatus().then((isActive) => {
        console.log(`[WebSocket] Livestream status synced: ${isActive}`);
      });
    });

    // Listen for server capabilities
    socket.on("server_capabilities", (capabilities) => {
      console.log("Received server capabilities:", capabilities);
      serverCapabilitiesRef.current = capabilities;

      // Update Redux with backend capabilities
      dispatch(
        liveStreamSlice.setBackendCapabilities({
          binaryStreaming: capabilities.binary_streaming,
          messagepack: capabilities.messagepack,
          protocolVersion: capabilities.protocol_version,
        })
      );
    });

    // Listen for binary frame events (UC2F packets)
    // UNIFIED HANDLING: Both binary and JPEG frames use 'frame' event with complete MessagePack encoding
    socket.on("frame", (payload, ack) => {
      let metadata = null;
      let frameData = null;

      try {
        // Decode complete MessagePack payload
        if (payload instanceof ArrayBuffer || payload instanceof Uint8Array) {
          // MessagePack-encoded frame (new format)
          const decoded = msgpackDecode(payload);
          metadata = decoded.metadata;

          // Handle different frame types
          if (decoded.data) {
            // Binary frame - ensure it's an ArrayBuffer for UC2F parser
            if (decoded.data instanceof ArrayBuffer) {
              frameData = decoded.data;
            } else if (decoded.data instanceof Uint8Array) {
              // Extract the underlying ArrayBuffer
              frameData = decoded.data.buffer.slice(
                decoded.data.byteOffset,
                decoded.data.byteOffset + decoded.data.byteLength
              );
            } else if (Array.isArray(decoded.data)) {
              // MessagePack may decode binary as array - convert to ArrayBuffer
              const uint8 = new Uint8Array(decoded.data);
              frameData = uint8.buffer;
            } else {
              console.error(
                "Unexpected binary data format:",
                typeof decoded.data,
                decoded.data
              );
              return;
            }
          } else if (decoded.image) {
            // JPEG frame
            frameData = decoded.image;
          }
        } else if (Array.isArray(payload) && payload.length === 2) {
          // Legacy format: [packed_metadata, frameData]
          const metadataRaw = payload[0];
          frameData = payload[1];

          // Decode metadata
          if (
            metadataRaw instanceof ArrayBuffer ||
            metadataRaw instanceof Uint8Array
          ) {
            metadata = msgpackDecode(metadataRaw);
          } else if (typeof metadataRaw === "object") {
            metadata = metadataRaw;
          } else if (typeof metadataRaw === "string") {
            metadata = JSON.parse(metadataRaw);
          }
        } else {
          // Very old legacy format: just binary data
          frameData = payload;
        }

        // Update Redux with current frame ID (unified field name)
        if (metadata && metadata.frame_id !== undefined) {
          dispatch(liveStreamSlice.setCurrentFrameId(metadata.frame_id));
        }

        // Handle based on protocol type in metadata
        if (
          metadata &&
          (metadata.protocol === "binary" || metadata.format === "binary")
        ) {
          // Binary frame - dispatch to UC2F parser
          window.dispatchEvent(
            new CustomEvent("uc2:frame", {
              detail: {
                buffer: frameData,
                metadata: metadata,
              },
            })
          );
        } else if (
          metadata &&
          (metadata.protocol === "jpeg" || metadata.format === "jpeg")
        ) {
          // JPEG frame - update Redux (base64 image)
          dispatch(liveStreamSlice.setLiveViewImage(frameData));
          dispatch(liveStreamSlice.setImageFormat("jpeg"));

          // Update pixel size if available
          if (metadata.pixel_size) {
            dispatch(liveStreamSlice.setPixelSize(metadata.pixel_size));
          }

          // Track latency if server timestamp is available
          if (metadata.server_timestamp) {
            const latency =
              (Date.now() / 1000 - metadata.server_timestamp) * 1000; // Convert to ms
            dispatch(liveStreamSlice.updateLatency(latency));
          }
        } else if (frameData) {
          // No metadata - fallback to binary frame
          window.dispatchEvent(
            new CustomEvent("uc2:frame", {
              detail: {
                buffer: frameData,
                metadata: null,
              },
            })
          );
        }
      } catch (error) {
        console.error("Error decoding frame:", error);
        return;
      }

      // Send acknowledgement to enable flow control
      // This tells the server we're ready for the next frame
      if (ack && typeof ack === "function") {
        ack();
      } else {
        // Fallback: emit explicit acknowledgement event with unified field name
        socket.emit("frame_ack", { frame_id: metadata?.frame_id });
      }
    });

    // Listen to signals (JSON format - legacy/fallback)
    socket.on("signal", (data, ack) => {
      //console.log('WebSocket signal', data);
      //update redux state
      dispatch(webSocketSlice.incrementSignalCount());

      //handle signal
      const dataJson = JSON.parse(data);
      //console.log(dataJson);

      // Process signal data
      processSignalData(dataJson, ack);
    });

    // Listen to signals (MessagePack format - preferred for efficiency)
    socket.on("signal_msgpack", (data, ack) => {
      //console.log('WebSocket signal_msgpack received');
      //update redux state
      dispatch(webSocketSlice.incrementSignalCount());

      try {
        // Decode MessagePack data
        const dataJson = msgpackDecode(data);
        //console.log(dataJson);

        // Process signal data
        processSignalData(dataJson, ack);
      } catch (error) {
        console.error("Error decoding MessagePack signal:", error);
      }
    });

    // Common signal processing function for both JSON and MessagePack
    const processSignalData = (dataJson, ack) => {
      // Dispatch to debug slice for SocketView (do this first for all signals)
      dispatch(socketDebugSlice.addMessage(dataJson));

      // REMOVED: frame_meta handler - metadata is now sent together with binary frame
      // Store frame metadata for UC2F parsing
      // if (dataJson.name === "frame_meta" && dataJson.metadata) { ... }

      //----------------------------------------------
      if (dataJson.name === "sigUpdateImage") {
        //console.log("sigUpdateImage", dataJson);
        //update redux state - LEGACY JPEG PATH (deprecated - use 'frame' event instead)
        if (dataJson.detectorname || dataJson.detector_name) {
          // Note: Legacy JPEG image handling - kept for backward compatibility
          // The new unified "frame" event handler above should be preferred
          dispatch(liveStreamSlice.setLiveViewImage(dataJson.image));

          // Track image format and set appropriate defaults based on streaming capability
          if (dataJson.format === "jpeg") {
            dispatch(liveStreamSlice.setImageFormat("jpeg"));
            // Only set defaults on first load or if values are at default
            const currentState = store.getState().liveStreamState;
            if (
              currentState.minVal === 0 &&
              (currentState.maxVal === 65535 || currentState.maxVal === 32768)
            ) {
              // Set initial defaults for JPEG
              dispatch(liveStreamSlice.setMinVal(0));
              dispatch(liveStreamSlice.setMaxVal(255));
            }
            // Support both legacy image_id and new frame_id
            const frameId = dataJson.frame_id ?? dataJson.image_id;
            if (frameId !== undefined) {
              dispatch(liveStreamSlice.setCurrentFrameId(frameId));
            }
          } else {
            // Binary streaming - use 16-bit range
            dispatch(liveStreamSlice.setImageFormat(dataJson.format || "raw"));
            const currentState = store.getState().liveStreamState;
            if (
              currentState.minVal === 0 &&
              currentState.maxVal === 65535 &&
              currentState.backendCapabilities.binaryStreaming
            ) {
              // Set default range for binary streaming (common 16-bit range)
              dispatch(liveStreamSlice.setMinVal(0));
              dispatch(liveStreamSlice.setMaxVal(32768));
            }
          }

          // Update pixel size if available (support both field names)
          const pixelSize = dataJson.pixel_size ?? dataJson.pixelsize;
          if (pixelSize) {
            dispatch(liveStreamSlice.setPixelSize(pixelSize));
          }

          // Track latency if server timestamp is available
          if (dataJson.server_timestamp) {
            const latency =
              (Date.now() / 1000 - dataJson.server_timestamp) * 1000; // Convert to ms
            dispatch(liveStreamSlice.updateLatency(latency));
            // Log every 30th frame to avoid spam
            const currentState = store.getState().liveStreamState;
            if (currentState.stats.frameCount % 30 === 0) {
              console.log(
                `Frame latency: ${latency.toFixed(
                  1
                )}ms (avg: ${currentState.stats.avg_latency_ms.toFixed(1)}ms)`
              );
            }
          }

          // Send acknowledgement for JPEG frames to enable flow control
          // Support both legacy image_id and new frame_id
          const frameId = dataJson.frame_id ?? dataJson.image_id;
          if (ack && typeof ack === "function") {
            console.log("Acknowledging JPEG image frame");
            ack();
          } else {
            console.log("Emitting frame_ack for JPEG image");
            socket.emit("frame_ack", { frame_id: frameId });
          }

          /*
        sigUpdateImage: 
        Object { 
            name: "sigUpdateImage", 
            detectorname: "WidefieldCamera", 
            pixelsize: 0.2, 
            format: "jpeg", 
            image: "................Base64.encodede.image......." 
        }
        */
          //----------------------------------------------
        }
      } else if (dataJson.name === "sigHistogramComputed") {
        //console.log("sigHistogramComputed", dataJson);
        // Handle histogram data similar to image updates
        if (dataJson.args && dataJson.args.p0 && dataJson.args.p1) {
          dispatch(
            liveStreamSlice.setHistogramData({
              x: dataJson.args.p0, // units
              y: dataJson.args.p1, // hist
            })
          );
        }
        //----------------------------------------------
      } else if (dataJson.name === "sigExperimentWorkflowUpdate") {
        //Args: {"arg0":{"status":"completed","step_id":0,"name":"Move to point 0","total_step_number":2424}}
        console.log("sigExperimentWorkflowUpdate", dataJson);

        dispatch(experimentStateSlice.setStatus(dataJson.args.arg0.status));
        dispatch(experimentStateSlice.setStepID(dataJson.args.arg0.step_id));
        dispatch(experimentStateSlice.setStepName(dataJson.args.arg0.name));
        dispatch(
          experimentStateSlice.setTotalSteps(
            dataJson.args.arg0.total_step_number
          )
        );
      } else if (dataJson.name === "sigExperimentImageUpdate") {
        console.log("sigExperimentImageUpdate", dataJson);

        // update from tiled view
        dispatch(tileStreamSlice.setTileViewImage(dataJson.image));
      } else if (dataJson.name === "sigObjectiveChanged") {
        console.log("sigObjectiveChanged", dataJson);
        //update redux state
        // TODO add check if parameter exists
        // TODO check if this works
        dispatch(objectiveSlice.setPixelSize(dataJson.args.p0.pixelsize));
        dispatch(objectiveSlice.setNA(dataJson.args.p0.NA));
        dispatch(
          objectiveSlice.setMagnification(dataJson.args.p0.magnification)
        );
        dispatch(
          objectiveSlice.setObjectiveName(dataJson.args.p0.objectiveName)
        );
        dispatch(objectiveSlice.setFovX(dataJson.args.p0.FOV[0]));
        dispatch(objectiveSlice.setFovY(dataJson.args.p0.FOV[1]));

        /*  data:
        Args: {"p0":0.2,"p1":0.5,"p2":10,"p3":"10x","p4":100,"p5":100}
        sigObjectiveChanged = Signal(float, float, float, str, float, float) 
              # pixelsize, NA, magnification, objectiveName, FOVx, FOVy
        */
        //----------------------------------------------
      } else if (dataJson.name == "sigUpdateMotorPosition") {
        console.log("sigUpdateMotorPosition received:", dataJson);
        //parse
        try {
          const parsedArgs = dataJson.args.p0;
          const positionerKeys = Object.keys(parsedArgs);

          if (positionerKeys.length > 0) {
            const key = positionerKeys[0];
            const correctedPositions = parsedArgs[key];

            console.log("Parsed positions:", correctedPositions);

            // Build position update object, filtering out undefined values
            const positionUpdate = Object.fromEntries(
              Object.entries({
                x: correctedPositions.X,
                y: correctedPositions.Y,
                z: correctedPositions.Z,
                a: correctedPositions.A,
              }).filter(([_, value]) => value !== undefined)
            );

            console.log("Position update to dispatch:", positionUpdate);

            //update redux state
            dispatch(positionSlice.setPosition(positionUpdate));
          }
        } catch (error) {
          console.error("Error in sigUpdateMotorPosition handler:", error);
        }
        //----------------------------------------------
      } else if (dataJson.name === "sigUpdateLaserPower") {
        // Handle laser power/enabled state updates from backend
        // Signal format: { "p0": { "635": { "power": 10000, "enabled": true }, ... } }
        console.log("sigUpdateLaserPower received:", dataJson);
        try {
          const laserData = dataJson.args?.p0;
          
          if (laserData && typeof laserData === 'object') {
            // Batch update all lasers in the signal
            dispatch(laserSlice.setLasersState(laserData));
          }
        } catch (error) {
          console.error("Error in sigUpdateLaserPower handler:", error);
        }
        //----------------------------------------------
      } else if (dataJson.name === "sigUC2SerialReadMessage") {
        // Handle serial read messages from UC2 board
        console.log("sigUC2SerialReadMessage received:", dataJson);
        try {
          const message = dataJson.args?.p0 || "";
          dispatch(uc2Slice.addSerialLogEntry(`Read: ${message}`));
        } catch (error) {
          console.error("Error in sigUC2SerialReadMessage handler:", error);
        }
        //----------------------------------------------
      } else if (dataJson.name === "sigUC2SerialWriteMessage") {
        // Handle serial write messages from UC2 board
        console.log("sigUC2SerialWriteMessage received:", dataJson);
        try {
          const message = dataJson.args?.p0 || "";
          dispatch(uc2Slice.addSerialLogEntry(`Write: ${message}`));
        } catch (error) {
          console.error("Error in sigUC2SerialWriteMessage handler:", error);
        }
        //----------------------------------------------
      } else if (dataJson.name === "sigScanStatusUpdate") {
        // Handle lightsheet scan status updates
        console.log("sigScanStatusUpdate received:", dataJson);
        try {
          // Expected format: dataJson.args.p0 = { isRunning, scanMode, currentPosition, totalPositions, currentFrame, progress, zarrPath, tiffPath, errorMessage }
          const scanStatus = dataJson.args?.p0;
          
          if (scanStatus) {
            // Update lightsheet scan status in Redux
            dispatch(lightsheetSlice.setScanStatus(scanStatus));
          }
        } catch (error) {
          console.error("Error in sigScanStatusUpdate handler:", error);
        }
        //----------------------------------------------
      } else if (dataJson.name === "sigOTAStatusUpdate") {
        // Handle CAN OTA status updates (both WiFi OTA and CAN streaming formats)
        console.log("sigOTAStatusUpdate received:", dataJson);
        try {
          const otaStatus = dataJson.args?.p0;

          if (otaStatus && otaStatus.canId !== undefined) {
            const { canId, status, message, progress, method } = otaStatus;

            // Determine status string and progress.
            // CAN streaming sends status as a string ("uploading", "success", "error", "initializing")
            // WiFi OTA sends status as a number (0=completed, 1=wifi_failed, 2=ota_failed)
            let statusString;
            let progressValue;

            if (typeof status === "number") {
              // Legacy WiFi OTA format
              if (status === 0) {
                statusString = "completed";
                progressValue = 100;
              } else if (status === 1) {
                statusString = "wifi_failed";
                progressValue = 0;
              } else if (status === 2) {
                statusString = "ota_failed";
                progressValue = 50;
              } else {
                statusString = otaStatus.success ? "completed" : "failed";
                progressValue = otaStatus.success ? 100 : 0;
              }
            } else {
              // CAN streaming format – status is a descriptive string
              statusString = status || "in_progress";
              progressValue = progress ?? 0;
            }

            const displayMessage =
              message || otaStatus.statusMsg || "Status update received";

            // Update Redux state with OTA progress
            dispatch(
              canOtaSlice.setUpdateProgress({
                canId: canId,
                status: statusString,
                message: displayMessage,
                progress: progressValue,
                timestamp: new Date().toISOString(),
              })
            );

            // If update is completed or failed, check if all updates are done
            const terminalStates = [
              "completed", "success", "failed", "error",
              "wifi_failed", "ota_failed",
            ];
            if (terminalStates.includes(statusString)) {
              const state = store.getState();
              const canOtaState = state.canOtaState;
              const totalDevices = canOtaState.selectedDeviceIds.length;
              const completedCount = canOtaState.completedUpdateCount;
              const failedCount = canOtaState.failedUpdateCount;

              // If all devices are done, stop updating state
              if (completedCount + failedCount >= totalDevices) {
                dispatch(canOtaSlice.setIsUpdating(false));
              }
            }

            console.log(
              `OTA update for device ${canId}: ${statusString} (${progressValue}%) - ${displayMessage}`
            );
          }
        } catch (error) {
          console.error("Error in sigOTAStatusUpdate handler:", error);
        }
        //----------------------------------------------
      } else if (dataJson.name === "sigUSBFlashStatusUpdate") {
        // Handle USB flash status updates from esptool flashing process
        console.log("sigUSBFlashStatusUpdate received:", dataJson);
        try {
          const flashStatus = dataJson.args?.p0;
          if (flashStatus) {
            dispatch(usbFlashSlice.updateFlashProgress({
              status: flashStatus.status,
              progress: flashStatus.progress,
              message: flashStatus.message,
              details: flashStatus.details,
            }));

            // Auto-advance to completion step on terminal states
            const terminalStates = ["success", "failed", "warning"];
            if (terminalStates.includes(flashStatus.status)) {
              if (flashStatus.status === "success" || flashStatus.status === "warning") {
                dispatch(usbFlashSlice.setFlashResult({
                  status: flashStatus.status,
                  message: flashStatus.message,
                }));
              }
              dispatch(usbFlashSlice.setIsFlashing(false));
            }
          }
        } catch (error) {
          console.error("Error in sigUSBFlashStatusUpdate handler:", error);
        }
        //----------------------------------------------
      } else if (dataJson.name === "sigUpdateOMEZarrStore") {
        console.log("sigUpdateOMEZarrStore", dataJson);
        //update redux state
        dispatch(omeZarrSlice.setZarrUrl(dataJson.args.p0));
        dispatch(omeZarrSlice.tileArrived());
      } else if (dataJson.name === "sigNSTORMImageAcquired") {
        //console.log("sigNSTORMImageAcquired", dataJson);
        // Update STORM frame count - expected p0 to be frame number
        if (dataJson.args && dataJson.args.p0 !== undefined) {
          dispatch({
            type: "storm/setCurrentFrameNumber",
            payload: dataJson.args.p0,
          });
        }
      } else if (dataJson.name === "sigSTORMReconstructionUpdated") {
        //console.log("sigSTORMReconstructionUpdated", dataJson);
        // Update STORM reconstructed image
        if (dataJson.args && dataJson.args.p0) {
          dispatch({
            type: "storm/setReconstructedImage",
            payload: dataJson.args.p0,
          });
        }
      } else if (dataJson.name === "sigSTORMReconstructionStarted") {
        //console.log("sigSTORMReconstructionStarted", dataJson);
        dispatch({ type: "storm/setIsReconstructing", payload: true });
      } else if (dataJson.name === "sigSTORMReconstructionStopped") {
        //console.log("sigSTORMReconstructionStopped", dataJson);
        dispatch({ type: "storm/setIsReconstructing", payload: false });
      } else if (dataJson.name === "sigUpdatedSTORMReconstruction") {
        //console.log("sigUpdatedSTORMReconstruction", dataJson);
        // Handle localization data - expected p0 to be an object with x, y, and intensity arrays
        if (dataJson.args && dataJson.args.p0) {
          try {
            let localizationsData = dataJson.args.p0;

            // If p0 is a string, parse it first
            if (typeof localizationsData === "string") {
              localizationsData = JSON.parse(localizationsData);
            }

            // Check if it has the new format with separate x, y, intensity arrays
            if (
              localizationsData.x &&
              localizationsData.y &&
              Array.isArray(localizationsData.x) &&
              Array.isArray(localizationsData.y)
            ) {
              const localizations = [];
              const minLength = Math.min(
                localizationsData.x.length,
                localizationsData.y.length
              );

              for (let i = 0; i < minLength; i++) {
                localizations.push({
                  x: localizationsData.x[i],
                  y: localizationsData.y[i],
                  intensity: localizationsData.intensity
                    ? localizationsData.intensity[i]
                    : 0,
                });
              }

              dispatch({
                type: "storm/addLocalizations",
                payload: localizations,
              });
            } else if (Array.isArray(localizationsData)) {
              // Fallback for old format - array of objects
              const localizations = localizationsData.map((loc) => ({
                x: loc.x || loc[0],
                y: loc.y || loc[1],
                intensity: loc.intensity || 0,
              }));
              dispatch({
                type: "storm/addLocalizations",
                payload: localizations,
              });
            }
          } catch (e) {
            console.warn("Failed to parse STORM localization data:", e);
          }
        }
      } else if (dataJson.name === "sigFocusValueUpdate") {
        console.log("sigFocusValueUpdate received:", dataJson); // Debug log
        // Handle focus value updates - updated for new library format
        try {
          // Try the new format first (direct object)
          let focusData = dataJson.args || {};

          // If args is available but not an object, try args.p0
          if (dataJson.args.p0) {
            focusData = dataJson.args.p0 || {};
          }

          console.log("Parsed focus data:", focusData); // Debug log

          // Support both old and new formats
          const focusValue = focusData.focus_value || focusData.focusValue || 0;
          const currentFocusMotorPosition = focusData.current_position || 0;
          const setPointSignal =
            focusData.focus_setpoint || focusData.setPointSignal || 0;
          const timestamp = focusData.timestamp || Date.now();

          console.log("Dispatching focus values:", {
            focusValue,
            setPointSignal,
            currentFocusMotorPosition,
            timestamp,
          }); // Debug log

          dispatch(
            focusLockSlice.addFocusValue({
              focusValue,
              setPointSignal,
              currentFocusMotorPosition,
              timestamp,
            })
          );
        } catch (error) {
          console.error("Error parsing focus value signal:", error);
        }
      } else if (dataJson.name === "sigFocusLockStateChanged") {
        //console.log("sigFocusLockStateChanged", dataJson);
        // Handle focus lock state changes - updated for new library format
        try {
          // Try to get state data from args directly or args.p0
          let stateData = dataJson.args || {};
          if (dataJson.args && dataJson.args.p0) {
            stateData = dataJson.args.p0;
          }

          if (typeof stateData === "object") {
            if (stateData.is_locked !== undefined) {
              dispatch(focusLockSlice.setFocusLocked(stateData.is_locked));
            }
            if (stateData.is_calibrating !== undefined) {
              dispatch(
                focusLockSlice.setIsCalibrating(stateData.is_calibrating)
              );
            }
            if (stateData.is_measuring !== undefined) {
              dispatch(focusLockSlice.setIsMeasuring(stateData.is_measuring));
            }
          }
        } catch (error) {
          console.error("Error parsing focus lock state signal:", error);
        }
      } else if (dataJson.name === "sigCalibrationProgress") {
        //console.log("sigCalibrationProgress", dataJson);
        // Handle calibration progress updates - updated for new library format
        try {
          // Try to get progress data from args directly or args.p0
          let progressData = dataJson.args || {};
          if (dataJson.args && dataJson.args.p0) {
            progressData = dataJson.args.p0;
          }

          if (typeof progressData === "object") {
            // Add calibration progress state to Redux if needed
            console.log("Calibration progress:", progressData);
            // TODO: Add calibration progress to Redux state if needed
          }
        } catch (error) {
          console.error("Error parsing calibration progress signal:", error);
        }
      } else if (dataJson.name === "sigUpdateFocusPlot") {
        // Handle autofocus plot data updates
        try {
          if (dataJson.args && dataJson.args.p0 && dataJson.args.p1) {
            // Store focus positions in p0, contrast values in p1
            dispatch(
              autofocusSlice.setPlotData({
                x: dataJson.args.p0,
                y: dataJson.args.p1,
              })
            );
          }
        } catch (error) {
          console.error("Error parsing autofocus plot signal:", error);
        }
      } else if (dataJson.name === "sigAutoFocusLiveValue") {
        // Handle live focus value updates during monitoring mode
        try {
          if (dataJson.args && dataJson.args.p0) {
            // p0 contains {focus_value, timestamp, method}
            dispatch(autofocusSlice.setLiveFocusValue(dataJson.args.p0));
          }
        } catch (error) {
          console.error("Error parsing autofocus live value signal:", error);
        }
      } else if (dataJson.name === "sigGameState") {
        // Handle maze game state updates
        try {
          const gameState = dataJson.args?.p0 || dataJson.args || {};
          dispatch(mazeGameSlice.setGameState(gameState));
        } catch (error) {
          console.error("Error parsing game state signal:", error);
        }
      } else if (dataJson.name === "sigCounterUpdated") {
        // Handle maze game counter updates
        try {
          const counter =
            dataJson.args?.p0 ?? dataJson.counter ?? dataJson.value ?? 0;
          dispatch(mazeGameSlice.setCounter(counter));
        } catch (error) {
          console.error("Error parsing counter signal:", error);
        }
      } else if (dataJson.name === "sigPreviewUpdated") {
        // Handle maze game preview updates
        try {
          if (dataJson.args.p0) {
            let rawImage = dataJson.args.p0.jpeg_b64;

            // Remove the b'...' wrapper if present
            if (
              typeof rawImage === "string" &&
              rawImage.startsWith("b'") &&
              rawImage.endsWith("'")
            ) {
              rawImage = rawImage.slice(2, -1);

              // Convert escaped hex sequences (\xHH) to actual bytes
              const bytes = [];
              let i = 0;
              while (i < rawImage.length) {
                if (rawImage[i] === "\\" && rawImage[i + 1] === "x") {
                  // Parse hex escape sequence
                  const hex = rawImage.slice(i + 2, i + 4);
                  bytes.push(parseInt(hex, 16));
                  i += 4;
                } else if (rawImage[i] === "\\" && rawImage[i + 1] === "r") {
                  bytes.push(13); // \r
                  i += 2;
                } else if (rawImage[i] === "\\" && rawImage[i + 1] === "n") {
                  bytes.push(10); // \n
                  i += 2;
                } else {
                  bytes.push(rawImage.charCodeAt(i));
                  i += 1;
                }
              }

              // Convert bytes to Base64
              const uint8Array = new Uint8Array(bytes);
              let binaryString = "";
              for (let i = 0; i < uint8Array.length; i++) {
                binaryString += String.fromCharCode(uint8Array[i]);
              }
              const base64 = btoa(binaryString);

              dispatch(
                mazeGameSlice.setPreviewImage(`data:image/png;base64,${base64}`)
              );
            } else {
              // If it's already a base64 string, use it directly
              dispatch(
                mazeGameSlice.setPreviewImage(
                  `data:image/png;base64,${rawImage}`
                )
              );
            }
          }
        } catch (error) {
          console.error("Error processing preview image:", error);
        }
      }
      // Name: sigUpdatedSTORMReconstruction => Args: {"p0":[[252.2014923095703,298.37579345703125,2814.840087890625,206508.3125,1.037859320640564]}
      else {
        //console.warn("WebSocket: Unhandled signal from socket:", dataJson.name);
        //console.warn(dataJson);
      }
    }; // End of processSignalData function

    socket.on("broadcast", (data) => {
      console.log("WebSocket broadcast");
    });

    socket.onerror = (error) => {
      console.error("WebSocket Error: ", error);
    };

    socket.onclose = () => {
      console.log("WebSocket closed");
      //update redux state
      dispatch(webSocketSlice.resetState());
    };

    // Listen for disconnect events
    socket.on("disconnect", (reason) => {
      console.log(`[WebSocket] Disconnected. Reason: ${reason}`);
      dispatch(webSocketSlice.setConnected(false));

      // Note: We don't reset isStreamRunning here because the backend might still be running
      // The stream status will be re-synced on reconnect
    });

    //##################################################################################
    return () => {
      console.log("WebSocket: Cleaning up connection", socket.id);
      socket.off("signal");
      socket.off("signal_msgpack");
      socket.off("frame");
      socket.off("broadcast");
      socket.off("connect");
      socket.off("disconnect");
      socket.off("server_capabilities");
      socket.disconnect();
      socket.close();
    };
  }, [dispatch, connectionSettingsState]);

  // Global UC2 connection monitoring (periodic checks with pause functionality)
  useEffect(() => {
    let isPaused = false;

    // Clear any existing interval
    if (connectionCheckRef.current) {
      clearInterval(connectionCheckRef.current);
      connectionCheckRef.current = null;
    }

    // Function to start/stop periodic checks
    const startPeriodicChecks = () => {
      if (!isPaused && !connectionCheckRef.current) {
        // Initial connection check
        checkUc2Connection();

        // Set up periodic monitoring
        connectionCheckRef.current = setInterval(() => {
          if (!isPaused) {
            checkUc2Connection();
          }
        }, 10000); // Every 10 seconds
      }
    };

    const stopPeriodicChecks = () => {
      if (connectionCheckRef.current) {
        clearInterval(connectionCheckRef.current);
        connectionCheckRef.current = null;
      }
    };

    // Listen for pause/resume events from ConnectionSettings
    const handlePausePeriodicTests = (event) => {
      isPaused = event.detail.pause;
      console.log(
        `Periodic connection tests ${isPaused ? "paused" : "resumed"}`
      );

      if (isPaused) {
        stopPeriodicChecks();
      } else {
        startPeriodicChecks();
      }
    };

    // Add event listener
    window.addEventListener(
      "imswitch:pausePeriodicTests",
      handlePausePeriodicTests
    );

    // Start periodic checks initially
    startPeriodicChecks();

    return () => {
      // Cleanup
      window.removeEventListener(
        "imswitch:pausePeriodicTests",
        handlePausePeriodicTests
      );
      stopPeriodicChecks();
    };
  }, [checkUc2Connection]); // Now using the memoized function

  return null; // This component does not render anything, just manages the WebSocket
};

export default WebSocketHandler;
