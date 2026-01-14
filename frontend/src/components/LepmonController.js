import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Paper,
  Button,
  Typography,
  Switch,
  TextField,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Card,
  CardContent,
  CardMedia,
  Box,
  Chip,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
// ggf. weitere MUI-Komponenten wie Slider, IconButton etc.

import { useWebSocket } from "../context/WebSocketContext"; // Beispiel: eigener WebSocket-Context
import * as lepmonSlice from "../state/slices/LepmonSlice.js";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

export default function LepmonController() {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  // Redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const lepmonState = useSelector(lepmonSlice.getLepmonState);

  // Use Redux state instead of local useState
  const isRunning = lepmonState.isRunning;
  const currentImageCount = lepmonState.currentImageCount;
  const serverTime = lepmonState.serverTime;
  // const deviceTime = lepmonState.deviceTime; // Currently unused
  const insideTemp = lepmonState.insideTemp;
  const outsideTemp = lepmonState.outsideTemp;
  const humidity = lepmonState.humidity;
  const freeSpace = lepmonState.freeSpace;
  const storagePath = lepmonState.storagePath;
  const sharpnessValue = lepmonState.sharpnessValue;
  const exposure = lepmonState.exposure;
  const gain = lepmonState.gain;
  const timelapsePeriod = lepmonState.timelapsePeriod;
  const timelapseLocked = lepmonState.timelapseLocked;
  const rebootLocked = lepmonState.rebootLocked;
  const lat = lepmonState.lat;
  const lng = lepmonState.lng;
  const time = lepmonState.time;
  const date = lepmonState.date;

  // New state variables
  const lightStates = lepmonState.lightStates;
  const availableLights = lepmonState.availableLights;
  const hardwareStatus = lepmonState.hardwareStatus;
  const latestImage = lepmonState.latestImage;
  const lcdDisplay = lepmonState.lcdDisplay;
  // const buttonStates = lepmonState.buttonStates; // Currently unused in UI
  // const availableButtons = lepmonState.availableButtons; // Currently unused in UI
  const timingConfig = lepmonState.timingConfig;
  // const sensorData = lepmonState.sensorData; // Currently unused in UI
  // const availableSensors = lepmonState.availableSensors; // Currently unused in UI

  // WebSocket
  const socket = useWebSocket(); // Context, der `socket` zur Verfügung stellt

  // Beim ersten Laden: Initial-Daten per REST holen
  useEffect(() => {
    async function fetchInitialData() {
      try {
        // Build base URL from Redux state
        const urlStatus = `${hostIP}:${hostPort}/imswitch/api/LepmonController/getStatus`;
        const resStatus = await fetch(urlStatus);
        const statusData = await resStatus.json();
        dispatch(lepmonSlice.setInitialStatus(statusData));

        // 2. Hole Kamera-/Timelapse-Einstellungen
        const urlInitialParams = `${hostIP}:${hostPort}/imswitch/api/LepmonController/getInitialParams`;
        const resParams = await fetch(urlInitialParams);
        const paramsData = await resParams.json();
        dispatch(lepmonSlice.setInitialParams(paramsData));

        // 3. Hole Hardware-Status (new)
        const urlHardwareStatus = `${hostIP}:${hostPort}/imswitch/api/LepmonController/getHardwareStatus`;
        const resHardware = await fetch(urlHardwareStatus);
        const hardwareData = await resHardware.json();
        dispatch(lepmonSlice.setHardwareStatus(hardwareData.hardwareStatus));
        dispatch(lepmonSlice.setLightStates(hardwareData.lightStates));
        dispatch(lepmonSlice.setAvailableLights(hardwareData.availableLEDs));
        dispatch(lepmonSlice.setButtonStates(hardwareData.buttonStates));
        dispatch(
          lepmonSlice.setAvailableButtons(hardwareData.availableButtons)
        );
        dispatch(lepmonSlice.setLcdDisplay(hardwareData.lcdDisplay));

        // 4. Hole Timing-Konfiguration (new)
        const urlTimingConfig = `${hostIP}:${hostPort}/imswitch/api/LepmonController/getTimingConfig`;
        const resTiming = await fetch(urlTimingConfig);
        const timingData = await resTiming.json();
        dispatch(lepmonSlice.setTimingConfig(timingData));
      } catch (err) {
        console.error("Fehler beim Laden der Initialdaten:", err);
      }
    }
    fetchInitialData();

    // Versuche, Client-GPS zu ermitteln (optional)
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          dispatch(lepmonSlice.setLat(pos.coords.latitude));
          dispatch(lepmonSlice.setLng(pos.coords.longitude));
        },
        (err) => {
          console.warn("GPS nicht verfügbar:", err);
          // set default values of Jena
          dispatch(lepmonSlice.setLat(50.928014));
          dispatch(lepmonSlice.setLng(11.589237));
        },
        { enableHighAccuracy: true }
      );
    }

    // Setze die aktuelle Zeit und das Datum
    const now = new Date();
    dispatch(lepmonSlice.setTime(now.toTimeString().slice(0, 8))); // HH:MM:SS
    dispatch(lepmonSlice.setDate(now.toISOString().slice(0, 10))); // YYYY-MM-DD
  }, [dispatch, hostIP, hostPort]);

  // WebSocket-Ereignisse
  useEffect(() => {
    if (!socket) return;

    socket.on("signal", (msg) => {
      // msg = JSON.stringify(...) vom Server => parsen
      const data = JSON.parse(msg);
      switch (data.name) {
        case "sigImagesTaken":
          dispatch(lepmonSlice.setCurrentImageCount(data.args.p0));
          break;
        case "temperatureUpdate":
          const _args = data.args.p0;
          dispatch(
            lepmonSlice.setTemperatureData({
              innerTemp: _args.innerTemp,
              outerTemp: _args.outerTemp,
              humidity: _args.humidity,
            })
          );
          break;
        case "freeSpaceUpdate": {
          let _argsFreeSpace;
          if (typeof data.args.p0 === "string") {
            _argsFreeSpace = JSON.parse(data.args.p0.replace(/'/g, '"'));
          } else {
            _argsFreeSpace = data.args.p0;
          }
          dispatch(lepmonSlice.setFreeSpace(_argsFreeSpace.freeSpace));
          break;
        }
        case "sigFocusSharpness":
          console.log("Fokus-Wert:", data.args.p0);
          dispatch(lepmonSlice.setSharpnessValue(data.args.p0));
          break;
        case "serverTime":
          dispatch(lepmonSlice.setServerTime(data.value));
          break;
        case "sigUpdateImage":
          // Handle lepmon images specifically
          if (
            data.detectorname === "LepmonCamera" ||
            data.detectorname === "lepmonCam"
          ) {
            dispatch(lepmonSlice.setLatestImage(data.image));
            if (data.format) {
              dispatch(lepmonSlice.setImageFormat(data.format));
            }
          }
          break;
        case "sigLightStateChanged":
          // Handle light state changes
          if (data.args && data.args.p0) {
            let lightData;
            if (Array.isArray(data.args.p0)) {
              // If p0 is an array, handle each entry as a light state update
              data.args.p0.forEach((ld) => {
                dispatch(
                  lepmonSlice.setLightState({
                    lightName: ld.lightName,
                    isOn: ld.state,
                  })
                );
              });
              break;
            } else if (typeof data.args.p0 === "string") {
              lightData = JSON.parse(data.args.p0.replace(/'/g, '"'));
              dispatch(
                lepmonSlice.setLightState({
                  lightName: lightData.lightName,
                  isOn: lightData.state,
                })
              );
            } else if (typeof data.args.p0 === "object") {
              lightData = data.args.p0;
              dispatch(
                lepmonSlice.setLightState({
                  lightName: lightData.lightName,
                  isOn: lightData.state,
                })
              );
            } else {
              // fallback: do nothing
            }
          }
          break;
        case "sigLCDDisplayUpdate":
          // Handle LCD display updates
          if (data.args && data.args.p0) {
            let displayLines;
            if (typeof data.args.p0 === "string") {
              displayLines = data.args.p0.split("\n");
            } else if (Array.isArray(data.args.p0)) {
              displayLines = data.args.p0;
            } else {
              displayLines = ["", "", "", ""];
            }
            dispatch(
              lepmonSlice.setLcdDisplay({
                line1: displayLines[0] || "",
                line2: displayLines[1] || "",
                line3: displayLines[2] || "",
                line4: displayLines[3] || "",
              })
            );
          }
          break;
        case "sigButtonPressed":
          // Handle button press events
          if (data.args && data.args.p0) {
            let buttonData;
            if (typeof data.args.p0 === "string") {
              buttonData = JSON.parse(data.args.p0.replace(/'/g, '"'));
            } else if (typeof data.args.p0 === "object") {
              buttonData = data.args.p0;
            } else {
              buttonData = {};
            }
            dispatch(
              lepmonSlice.setButtonState({
                buttonName: buttonData.buttonName,
                isPressed: buttonData.state,
              })
            );
          }
          break;
        default:
          console.log("Unbekannter msgType:", data.msgType);
          break;
      }
    });

    // Cleanup if needed
    return () => {
      socket.off("message");
    };
  }, [socket, dispatch]);

  // Experiment Start / Stop
  const handleStart = async () => {
    // ggf. Client-Zeit und GPS hochladen
    const userTime = new Date().toISOString();
    dispatch(lepmonSlice.setDeviceTime(userTime));

    const params = new URLSearchParams({
      deviceTime: userTime,
      deviceLat: lat,
      deviceLng: lng,
      exposureTime: exposure,
      gain: gain,
      timelapsePeriod: timelapsePeriod,
      time: time,
      date: date,
    });

    try {
      const urlStart = `${hostIP}:${hostPort}/imswitch/api/LepmonController/startExperiment?${params.toString()}`;
      const res = await fetch(urlStart, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json();
      if (data.success) {
        dispatch(lepmonSlice.setIsRunning(true));
      }
    } catch (err) {
      console.error("Fehler beim Starten:", err);
    }
  };

  const handleStop = async () => {
    try {
      const urlStop = `${hostIP}:${hostPort}/imswitch/api/LepmonController/stopLepmonExperiment`;
      const res = await fetch(urlStop, { method: "POST" });
      const data = await res.json();
      if (data.success) {
        dispatch(lepmonSlice.setIsRunning(false));
      }
    } catch (err) {
      console.error("Fehler beim Stoppen:", err);
    }
  };

  // Fokusmodus
  const handleFocus = async () => {
    try {
      const urlFocus = `${hostIP}:${hostPort}/imswitch/api/LepmonController/focusMode`;
      await fetch(urlFocus, { method: "POST" });
      // Backend startet 15s-Fokusmodus => sendet "focusSharpness" als WS
    } catch (err) {
      console.error("Fehler beim Fokus:", err);
    }
  };

  // Reboot
  const handleReboot = async () => {
    // z. B. mit Confirm-Dialog
    try {
      const urlReboot = `${hostIP}:${hostPort}/imswitch/api/LepmonController/reboot`;
      await fetch(urlReboot, { method: "POST" });
    } catch (err) {
      console.error("Reboot-Error:", err);
    }
  };

  // Light Control Functions (new)
  // Sends lightName and state as query parameters, as required by OpenAPI spec
  const handleLightToggle = async (lightName, newState) => {
    try {
      // Build query string
      const params = new URLSearchParams({
        lightName: lightName,
        state: newState,
      });
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/setLightState?${params.toString()}`;
      const response = await fetch(url, {
        method: "POST",
      });
      const data = await response.json();
      if (data.success) {
        dispatch(lepmonSlice.setLightState({ lightName, isOn: newState }));
      }
    } catch (err) {
      console.error("Error controlling light:", err);
    }
  };

  const handleAllLightsToggle = async (newState) => {
    try {
      // Build query string
      const params = new URLSearchParams({
        state: newState,
      });
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/setAllLightsState?${params.toString()}`;
      const response = await fetch(url, {
        method: "POST",
      });
      const data = await response.json();
      if (data.success) {
        // Update all light states
        const newLightStates = {};
        availableLights.forEach((light) => {
          newLightStates[light] = newState;
        });
        dispatch(lepmonSlice.setLightStates(newLightStates));
      }
    } catch (err) {
      console.error("Error controlling all lights:", err);
    }
  };

  // Snap Image Function (new)
  const handleSnapImage = async () => {
    try {
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/lepmonSnapImage`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ format: "jpeg", exposure: exposure }),
      });
      const data = await response.json();
      if (data.success) {
        console.log("Image captured:", data.filename);
      }
    } catch (err) {
      console.error("Error capturing image:", err);
    }
  };

  // Update LCD Display Function (new)
  const handleUpdateDisplay = async (line1, line2, line3, line4) => {
    try {
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/updateLCDDisplay`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ line1, line2, line3, line4 }),
      });
      const data = await response.json();
      if (data.success) {
        dispatch(lepmonSlice.setLcdDisplay({ line1, line2, line3, line4 }));
      }
    } catch (err) {
      console.error("Error updating display:", err);
    }
  };

  // Timing Configuration Update Function (new)
  const handleTimingConfigUpdate = async (newConfig) => {
    try {
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/setTimingConfig`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: newConfig }),
      });
      const data = await response.json();
      if (data.success) {
        dispatch(lepmonSlice.setTimingConfig(data.updated_config));
      }
    } catch (err) {
      console.error("Error updating timing config:", err);
    }
  };

  // LepmonOS Control Functions (new)
  const handleLepmonStartup = async () => {
    try {
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/lepmonStartup`;
      const response = await fetch(url, { method: "POST" });
      const data = await response.json();
      if (data.success) {
        console.log("LepmonOS startup completed");
      }
    } catch (err) {
      console.error("Error starting LepmonOS:", err);
    }
  };

  const handleLepmonShutdown = async () => {
    try {
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/lepmonShutdown`;
      const response = await fetch(url, { method: "POST" });
      const data = await response.json();
      if (data.success) {
        console.log("LepmonOS shutdown completed");
      }
    } catch (err) {
      console.error("Error shutting down LepmonOS:", err);
    }
  };

  // Sends action as query parameter, as required by OpenAPI spec
  const handleUVLedControl = async (action) => {
    try {
      const params = new URLSearchParams({ action });
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/lepmonUVLed?${params.toString()}`;
      const response = await fetch(url, {
        method: "POST",
      });
      const data = await response.json();
      if (data.success) {
        dispatch(
          lepmonSlice.setLightState({ lightName: "UV_LED", isOn: data.state })
        );
      }
    } catch (err) {
      console.error("Error controlling UV LED:", err);
    }
  };

  // Sends action as query parameter, as required by OpenAPI spec
  const handleVisibleLedControl = async (action) => {
    try {
      const params = new URLSearchParams({ action });
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/lepmonVisibleLed?${params.toString()}`;
      const response = await fetch(url, {
        method: "POST",
      });
      const data = await response.json();
      if (data.success) {
        dispatch(
          lepmonSlice.setLightState({
            lightName: "Visible_LED",
            isOn: data.state,
          })
        );
      }
    } catch (err) {
      console.error("Error controlling Visible LED:", err);
    }
  };

  // Live Sensor Data Fetch (new)
  const fetchLiveSensorData = async () => {
    try {
      const url = `${hostIP}:${hostPort}/imswitch/api/LepmonController/getSensorDataLive`;
      const response = await fetch(url);
      const data = await response.json();
      if (data.success) {
        dispatch(lepmonSlice.setSensorData(data.sensorData));
        dispatch(lepmonSlice.setAvailableSensors(data.availableSensors));
      }
    } catch (err) {
      console.error("Error fetching live sensor data:", err);
    }
  };

  return (
    <Paper style={{ padding: 20 }}>
      <Typography variant="h5">LepMon Controller</Typography>

      {/* Main Status Section */}
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Typography>
            Aktueller Status: {isRunning ? "Läuft" : "Gestoppt"}
          </Typography>
          <Typography>Aufgenommene Bilder: {currentImageCount}</Typography>
          <Typography>Freier Speicher: {freeSpace}</Typography>
          <Typography>Speicherpfad: {storagePath}</Typography>
          <Typography>Serverzeit: {serverTime}</Typography>
          <Typography>
            Inside Temp: {insideTemp} °C / Outside Temp: {outsideTemp} °C /
            Humidity: {humidity} %
          </Typography>
        </Grid>

        {/* Hardware Status Section */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Hardware Status</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography>
                    GPIO Available:
                    <Chip
                      label={hardwareStatus.gpio_available ? "Yes" : "No"}
                      color={
                        hardwareStatus.gpio_available ? "success" : "error"
                      }
                      size="small"
                      style={{ marginLeft: 8 }}
                    />
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography>
                    OLED Available:
                    <Chip
                      label={hardwareStatus.oled_available ? "Yes" : "No"}
                      color={
                        hardwareStatus.oled_available ? "success" : "error"
                      }
                      size="small"
                      style={{ marginLeft: 8 }}
                    />
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography>
                    I2C Available:
                    <Chip
                      label={hardwareStatus.i2c_available ? "Yes" : "No"}
                      color={hardwareStatus.i2c_available ? "success" : "error"}
                      size="small"
                      style={{ marginLeft: 8 }}
                    />
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography>
                    Simulation Mode:
                    <Chip
                      label={hardwareStatus.simulation_mode ? "On" : "Off"}
                      color={
                        hardwareStatus.simulation_mode ? "warning" : "success"
                      }
                      size="small"
                      style={{ marginLeft: 8 }}
                    />
                  </Typography>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Light Control Section */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Light Controls</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={() => handleAllLightsToggle(true)}
                    style={{ marginRight: 10 }}
                  >
                    All Lights On
                  </Button>
                  <Button
                    variant="contained"
                    color="secondary"
                    onClick={() => handleAllLightsToggle(false)}
                    style={{ marginRight: 10 }}
                  >
                    All Lights Off
                  </Button>
                </Grid>

                {/* Specialized LED Controls */}
                <Grid item xs={12}>
                  <Typography
                    variant="subtitle2"
                    style={{ marginTop: 16, marginBottom: 8 }}
                  >
                    Specialized LEDs
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Button
                    variant="outlined"
                    color="secondary"
                    onClick={() => handleUVLedControl("on")}
                    style={{ marginRight: 8 }}
                  >
                    UV LED On
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => handleUVLedControl("off")}
                  >
                    UV LED Off
                  </Button>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Button
                    variant="outlined"
                    color="primary"
                    onClick={() => handleVisibleLedControl("on")}
                    style={{ marginRight: 8 }}
                  >
                    Visible LED On
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => handleVisibleLedControl("off")}
                  >
                    Visible LED Off
                  </Button>
                </Grid>

                {/* Individual Light Controls */}
                <Grid item xs={12}>
                  <Typography
                    variant="subtitle2"
                    style={{ marginTop: 16, marginBottom: 8 }}
                  >
                    Individual LED Controls
                  </Typography>
                </Grid>
                {availableLights.map((lightName) => (
                  <Grid item xs={6} md={4} key={lightName}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={lightStates[lightName] || false}
                          onChange={(e) =>
                            handleLightToggle(lightName, e.target.checked)
                          }
                          color="primary"
                        />
                      }
                      label={`${lightName} LED`}
                    />
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* LepmonOS Controls Section */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">LepmonOS Controls</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Button
                    variant="outlined"
                    color="primary"
                    onClick={handleLepmonStartup}
                    fullWidth
                  >
                    LepmonOS Startup
                  </Button>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Button
                    variant="outlined"
                    color="secondary"
                    onClick={handleLepmonShutdown}
                    fullWidth
                  >
                    LepmonOS Shutdown
                  </Button>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Button
                    variant="outlined"
                    onClick={fetchLiveSensorData}
                    fullWidth
                  >
                    Update Sensors
                  </Button>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Button
                    variant="outlined"
                    onClick={() =>
                      handleUpdateDisplay(
                        "Status Update",
                        `Images: ${currentImageCount}`,
                        `Time: ${new Date().toLocaleTimeString()}`,
                        "LepMon Ready"
                      )
                    }
                    fullWidth
                  >
                    Update Display
                  </Button>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Image Display Section */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Latest Image</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={8}>
                  {latestImage ? (
                    <Card>
                      <CardMedia
                        component="img"
                        style={{ maxHeight: 400, objectFit: "contain" }}
                        image={`data:image/jpeg;base64,${latestImage}`}
                        alt="Latest captured image"
                      />
                      <CardContent>
                        <Typography variant="body2">
                          Latest image from LepMon camera
                        </Typography>
                      </CardContent>
                    </Card>
                  ) : (
                    <Box
                      style={{
                        height: 200,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        border: "2px dashed #ccc",
                        borderRadius: 8,
                      }}
                    >
                      <Typography color="textSecondary">
                        No image available. Start experiment or capture image.
                      </Typography>
                    </Box>
                  )}
                </Grid>
                <Grid item xs={12} md={4}>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSnapImage}
                    fullWidth
                  >
                    Capture Image
                  </Button>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* LCD Display Section */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">OLED Display</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Card
                    style={{
                      backgroundColor: "#000",
                      color: "#0f0",
                      fontFamily: "monospace",
                    }}
                  >
                    <CardContent>
                      <Typography style={{ color: "#0f0", fontSize: "12px" }}>
                        {lcdDisplay.line1 || "Line 1"}
                      </Typography>
                      <Typography style={{ color: "#0f0", fontSize: "12px" }}>
                        {lcdDisplay.line2 || "Line 2"}
                      </Typography>
                      <Typography style={{ color: "#0f0", fontSize: "12px" }}>
                        {lcdDisplay.line3 || "Line 3"}
                      </Typography>
                      <Typography style={{ color: "#0f0", fontSize: "12px" }}>
                        {lcdDisplay.line4 || "Line 4"}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Camera and Experiment Settings */}
        <Grid item xs={6}>
          <TextField
            label="Exposure"
            value={exposure}
            onChange={(e) => dispatch(lepmonSlice.setExposure(e.target.value))}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            label="Gain"
            value={gain}
            onChange={(e) => dispatch(lepmonSlice.setGain(e.target.value))}
          />
        </Grid>

        <Grid item xs={6}>
          <Typography>Timelapse Period (Sekunden)</Typography>
          <Switch
            checked={!timelapseLocked}
            onChange={() =>
              dispatch(lepmonSlice.setTimelapseLocked(!timelapseLocked))
            }
          />
          <TextField
            disabled={timelapseLocked}
            value={timelapsePeriod}
            onChange={(e) =>
              dispatch(lepmonSlice.setTimelapsePeriod(e.target.value))
            }
          />
        </Grid>

        <Grid item xs={6}>
          <Typography>Reboot Control</Typography>
          <Switch
            checked={!rebootLocked}
            onChange={() =>
              dispatch(lepmonSlice.setRebootLocked(!rebootLocked))
            }
          />
          <Button
            disabled={rebootLocked}
            variant="contained"
            onClick={handleReboot}
            style={{ marginLeft: 10 }}
          >
            Reboot
          </Button>
        </Grid>

        {/* Timing Configuration Section */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Timing Configuration</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Acquisition Interval (s)"
                    type="number"
                    value={timingConfig.acquisitionInterval || 60}
                    onChange={(e) =>
                      handleTimingConfigUpdate({
                        acquisitionInterval: parseInt(e.target.value),
                      })
                    }
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Stabilization Time (s)"
                    type="number"
                    value={timingConfig.stabilizationTime || 5}
                    onChange={(e) =>
                      handleTimingConfigUpdate({
                        stabilizationTime: parseInt(e.target.value),
                      })
                    }
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Pre-Acquisition Delay (s)"
                    type="number"
                    value={timingConfig.preAcquisitionDelay || 2}
                    onChange={(e) =>
                      handleTimingConfigUpdate({
                        preAcquisitionDelay: parseInt(e.target.value),
                      })
                    }
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Post-Acquisition Delay (s)"
                    type="number"
                    value={timingConfig.postAcquisitionDelay || 1}
                    onChange={(e) =>
                      handleTimingConfigUpdate({
                        postAcquisitionDelay: parseInt(e.target.value),
                      })
                    }
                    fullWidth
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        <Grid item xs={6}>
          <TextField
            label="Latitude"
            value={lat || ""}
            onChange={(e) => dispatch(lepmonSlice.setLat(e.target.value))}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            label="Longitude"
            value={lng || ""}
            onChange={(e) => dispatch(lepmonSlice.setLng(e.target.value))}
          />
        </Grid>

        <Grid item xs={6}>
          <TextField
            label="Time (HH:MM:SS)"
            value={time}
            onChange={(e) => dispatch(lepmonSlice.setTime(e.target.value))}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            label="Date (YYYY-MM-DD)"
            value={date}
            onChange={(e) => dispatch(lepmonSlice.setDate(e.target.value))}
          />
        </Grid>

        {/* Focus Control */}
        <Grid item xs={12}>
          <Button variant="contained" color="primary" onClick={handleFocus}>
            Focus
          </Button>
          {sharpnessValue && (
            <Typography style={{ marginLeft: 16, display: "inline" }}>
              Sharpness: {sharpnessValue}
            </Typography>
          )}
        </Grid>

        {/* Experiment Control */}
        <Grid item xs={12} style={{ marginTop: 20 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleStart}
            disabled={isRunning}
          >
            Start Experiment
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={handleStop}
            disabled={!isRunning}
            style={{ marginLeft: 10 }}
          >
            Stop Experiment
          </Button>
        </Grid>
      </Grid>
    </Paper>
  );
}
