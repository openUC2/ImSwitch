import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { setNotification } from "../state/slices/NotificationSlice.js";
import apiUC2ConfigControllerListSerialPorts from "../backendapi/apiUC2ConfigControllerListSerialPorts";
import apiUC2ConfigControllerSetSerialConfig from "../backendapi/apiUC2ConfigControllerSetSerialConfig";
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Paper,
  Tooltip,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  TextField,
  CircularProgress,
  Divider,
} from "@mui/material";
import {
  Memory,
  Warning,
  CheckCircle,
  Refresh,
  Build,
  AutoFixHigh as WizardIcon,
  Usb as UsbIcon,
  Bluetooth as BluetoothIcon,
  LightbulbOutlined as LedIcon,
} from "@mui/icons-material";

import CanOtaWizard from "./CanOtaWizard";
import UsbFlashWizard from "./UsbFlashWizard";

// Redux state management
import * as uc2Slice from "../state/slices/UC2Slice.js";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

/**
 * System Update Controller
 * Handles firmware flashing as well as related UC2 device control and
 * status actions, including hardware control and LED matrix status updates.
 */
const SystemUpdateController = () => {
  const dispatch = useDispatch();
  const uc2State = useSelector(uc2Slice.getUc2State);
  const uc2Connected = uc2State.uc2Connected; // Hardware connected
  const isBackendConnected = uc2State.backendConnected; // API reachable

  // Connection settings for direct API calls
  const { ip: hostIP, apiPort: hostPort } = useSelector(
    getConnectionSettingsState,
  );
  const base = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController`;
  const experimentBase = `${hostIP}:${hostPort}/imswitch/api/ExperimentController`;

  // UC2 Hardware Control toggle
  const [enableHardwareControl, setEnableHardwareControl] = useState(false);

  // --- USB serial override state (lives inside UC2 Hardware Control card) ---
  const [serialPorts, setSerialPorts] = useState([]);
  const [overridePort, setOverridePort] = useState("");
  const [overrideBaudrate, setOverrideBaudrate] = useState(115200);
  const [isLoadingPorts, setIsLoadingPorts] = useState(false);
  const [isApplyingSerial, setIsApplyingSerial] = useState(false);

  const loadSerialPorts = async () => {
    if (!isBackendConnected) return;
    try {
      setIsLoadingPorts(true);
      const ports = await apiUC2ConfigControllerListSerialPorts();
      setSerialPorts(Array.isArray(ports) ? ports : []);
    } catch (e) {
      dispatch(
        setNotification({
          message: "Failed to list serial ports: " + (e.message || e),
          type: "error",
        }),
      );
    } finally {
      setIsLoadingPorts(false);
    }
  };

  // Auto-load the port list the first time hardware control gets enabled,
  // so the user doesn't have to hunt for a refresh button.
  useEffect(() => {
    if (enableHardwareControl && isBackendConnected && serialPorts.length === 0) {
      loadSerialPorts();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enableHardwareControl, isBackendConnected]);

  const reconnectWithOverrides = async (persist) => {
    try {
      setIsApplyingSerial(true);
      const result = await apiUC2ConfigControllerSetSerialConfig(
        overridePort || null,
        Number(overrideBaudrate) || null,
        persist,
      );
      const ok = result?.status === "success";
      dispatch(
        setNotification({
          message: ok
            ? `ESP32 ${persist ? "saved & " : ""}reconnected (${result?.port || "auto-detected port"} @ ${result?.baudrate || "default"} baud)`
            : `Reconnect finished with status=${result?.status}, connected=${String(result?.connected)} — check backend logs.`,
          type: ok ? "success" : "warning",
        }),
      );
    } catch (e) {
      dispatch(
        setNotification({
          message: "setSerialConfig failed: " + (e.message || e),
          type: "error",
        }),
      );
    } finally {
      setIsApplyingSerial(false);
    }
  };

  // LED status control
  const [ledStatus, setLedStatus] = useState("idle");
  const [isSettingLed, setIsSettingLed] = useState(false);

  const handleSetLedStatus = async (status) => {
    setIsSettingLed(true);
    try {
      const url = `${experimentBase}/set_led_status?status=${encodeURIComponent(status)}`;
      await callEndpoint(url);
      setLedStatus(status);
    } finally {
      setIsSettingLed(false);
    }
  };

  // Direct API call helper
  const callEndpoint = async (url) => {
    try {
      const response = await fetch(url, { method: "GET" });
      if (!response.ok) {
        console.error(
          `API call failed: ${response.status} ${response.statusText}`,
        );
      } else {
        const data = await response.json();
        console.log("API response:", data);
      }
    } catch (e) {
      console.error("API call error:", e);
    }
  };

  // Wizard state
  const [showCanOtaWizard, setShowCanOtaWizard] = React.useState(false);
  const [showUsbFlashWizard, setShowUsbFlashWizard] = React.useState(false);

  // Mock firmware flash (future API integration)
  const handleFirmwareFlash = async () => {
    window.open("https://youseetoo.github.io", "_blank");
    console.log("Flashing firmware...");
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: "auto" }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Firmware updates
        </Typography>
      </Box>

      {/* Connection Status Alert */}
      {!isBackendConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body2">
            Backend connection required for system updates. Please configure
            connection in Settings.
          </Typography>
        </Alert>
      )}

      {/* Firmware Update Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <Memory color="secondary" />
            <Typography variant="h6">Device Firmware</Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Flash new firmware to connected microscopy devices.
          </Typography>

          {/* Firmware Status */}
          <Paper sx={{ p: 2, bgcolor: "background.default", mb: 2 }}>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <Memory fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Device Firmware"
                  secondary="UC2 ESP32 - Version detection pending"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  {isBackendConnected ? (
                    <CheckCircle fontSize="small" color="success" />
                  ) : (
                    <Warning fontSize="small" color="warning" />
                  )}
                </ListItemIcon>
                <ListItemText
                  primary="Connection Status"
                  secondary={
                    isBackendConnected
                      ? "Ready for firmware operations"
                      : "Backend connection required"
                  }
                />
              </ListItem>
            </List>
          </Paper>
        </CardContent>

        <CardActions>
          <Button
            startIcon={<Memory />}
            onClick={handleFirmwareFlash}
            disabled={!isBackendConnected}
          >
            Flash New Firmware
          </Button>
          <Tooltip title="Coming soon: Automatic firmware detection">
            <span>
              <Button startIcon={<Refresh />} disabled={true}>
                Detect Firmware
              </Button>
            </span>
          </Tooltip>
        </CardActions>
      </Card>

      {/* UC2 Hardware Control Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Memory color="primary" />
            <Typography variant="h6">UC2 Hardware Control</Typography>
            <Chip
              label={uc2Connected ? "ESP32 Connected" : "ESP32 Disconnected"}
              color={uc2Connected ? "success" : "error"}
              size="small"
              variant="outlined"
            />
          </Box>

          <FormControlLabel
            control={
              <Switch
                checked={enableHardwareControl}
                onChange={(e) => setEnableHardwareControl(e.target.checked)}
              />
            }
            label="Enable UC2 hardware control"
            sx={{ mb: 2 }}
          />

          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
            <Button
              variant="contained"
              disabled={!enableHardwareControl || !isBackendConnected}
              onClick={() => callEndpoint(`${base}/reconnect`)}
            >
              Reconnect UC2 Board
            </Button>

            <Button
              variant="contained"
              color="warning"
              disabled={!enableHardwareControl || !isBackendConnected}
              onClick={() => callEndpoint(`${base}/espRestart`)}
            >
              Force Restart ESP
            </Button>

            <Button
              variant="contained"
              color="info"
              startIcon={<BluetoothIcon />}
              disabled={!enableHardwareControl || !isBackendConnected}
              onClick={() => callEndpoint(`${base}/btpairing`)}
            >
              Bluetooth Pairing
            </Button>
          </Box>

          {/* USB connection override */}
          <Divider sx={{ my: 3 }} />
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
            <UsbIcon color="action" fontSize="small" />
            <Typography variant="subtitle2">
              USB Connection Override
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Override the serial port and/or baudrate the backend uses to talk
            to the ESP32. Leave the port empty to keep the value from the
            setup JSON. <strong>Reconnect &amp; Save</strong> persists the
            change; <strong>Reconnect (session only)</strong> applies it
            without writing to disk.
          </Typography>

          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", sm: "2fr 1fr" },
              gap: 2,
              alignItems: "start",
              mb: 2,
            }}
          >
            <TextField
              select
              size="small"
              label="Serial Port"
              value={overridePort}
              onChange={(e) => setOverridePort(e.target.value)}
              disabled={!enableHardwareControl || !isBackendConnected}
              helperText={
                serialPorts.length === 0
                  ? "Click 'Refresh Ports' to populate"
                  : `${serialPorts.length} port(s) detected`
              }
              fullWidth
            >
              <MenuItem value="">(keep current)</MenuItem>
              {serialPorts.map((p) => (
                <MenuItem key={p.device} value={p.device}>
                  {p.device}
                  {p.description ? ` — ${p.description}` : ""}
                </MenuItem>
              ))}
            </TextField>

            <TextField
              select
              size="small"
              label="Baudrate"
              value={overrideBaudrate}
              onChange={(e) =>
                setOverrideBaudrate(Number(e.target.value) || 115200)
              }
              disabled={!enableHardwareControl || !isBackendConnected}
              fullWidth
            >
              <MenuItem value={115200}>115200 (default)</MenuItem>
              <MenuItem value={230400}>230400</MenuItem>
              <MenuItem value={460800}>460800</MenuItem>
              <MenuItem value={921600}>921600</MenuItem>
            </TextField>
          </Box>

          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
            <Button
              variant="outlined"
              size="small"
              startIcon={
                isLoadingPorts ? <CircularProgress size={16} /> : <Refresh />
              }
              onClick={loadSerialPorts}
              disabled={
                !enableHardwareControl ||
                !isBackendConnected ||
                isLoadingPorts
              }
            >
              Refresh Ports
            </Button>
            <Button
              variant="contained"
              size="small"
              onClick={() => reconnectWithOverrides(true)}
              disabled={
                !enableHardwareControl ||
                !isBackendConnected ||
                isApplyingSerial
              }
              startIcon={
                isApplyingSerial ? <CircularProgress size={16} /> : undefined
              }
            >
              Reconnect &amp; Save
            </Button>
            <Button
              variant="text"
              size="small"
              onClick={() => reconnectWithOverrides(false)}
              disabled={
                !enableHardwareControl ||
                !isBackendConnected ||
                isApplyingSerial
              }
            >
              Reconnect (session only)
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* LED Matrix Status Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <LedIcon color="primary" />
            <Typography variant="h6">LED Matrix Status</Typography>
            <Chip
              label={isBackendConnected ? "Available" : "Disconnected"}
              color={isBackendConnected ? "success" : "error"}
              size="small"
              variant="outlined"
            />
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Override the LED matrix status indicator on the connected device.
          </Typography>

          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 2,
              flexWrap: "wrap",
            }}
          >
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel id="led-status-label">Status</InputLabel>
              <Select
                labelId="led-status-label"
                value={ledStatus}
                label="Status"
                onChange={(e) => setLedStatus(e.target.value)}
                disabled={!isBackendConnected}
              >
                <MenuItem value="idle">Idle</MenuItem>
                <MenuItem value="rainbow">Rainbow (Busy)</MenuItem>
                <MenuItem value="error">Error</MenuItem>
                <MenuItem value="scanning">Scanning</MenuItem>
                <MenuItem value="done">Done</MenuItem>
                <MenuItem value="on">On</MenuItem>
                <MenuItem value="off">Off</MenuItem>
              </Select>
            </FormControl>

            <Button
              variant="contained"
              startIcon={<LedIcon />}
              onClick={() => handleSetLedStatus(ledStatus)}
              disabled={!isBackendConnected || isSettingLed}
            >
              {isSettingLed ? "Setting..." : "Apply"}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* CAN OTA Update Card */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <Build color="primary" />
            <Typography variant="h6">Device Firmware Update</Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Update firmware on connected devices (motors, lasers, LEDs) via CAN
            or via Over-The-Air WIFI (OTA) updates
          </Typography>

          <Button
            variant="contained"
            color="secondary"
            onClick={() => setShowCanOtaWizard(true)}
            startIcon={<WizardIcon />}
            size="large"
            fullWidth
            disabled={!uc2Connected}
          >
            Launch CAN OTA Wizard
          </Button>

          {!uc2Connected && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              UC2 device must be connected to use CAN OTA updates
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* USB Master Flash Card */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <UsbIcon color="primary" />
            <Typography variant="h6">Master CAN HAT Firmware (USB)</Typography>
            <Chip
              label="esptool"
              color="info"
              size="small"
              variant="outlined"
            />
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Flash firmware to the master CAN HAT controller via USB connection.
            This device coordinates all CAN slave devices and cannot be updated
            via WiFi OTA.
          </Typography>

          <Button
            variant="contained"
            color="primary"
            onClick={() => setShowUsbFlashWizard(true)}
            startIcon={<UsbIcon />}
            size="large"
            fullWidth
          >
            Launch USB Flash Wizard
          </Button>

          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>Note:</strong> The ESP32 will be disconnected temporarily
              during flashing. Make sure the device is connected via USB before
              starting.
            </Typography>
          </Alert>
        </CardContent>
      </Card>

      {/* CAN OTA Wizard */}
      <CanOtaWizard
        open={showCanOtaWizard}
        onClose={() => setShowCanOtaWizard(false)}
      />

      {/* USB Flash Wizard */}
      <UsbFlashWizard
        open={showUsbFlashWizard}
        onClose={() => setShowUsbFlashWizard(false)}
      />
    </Box>
  );
};

export default SystemUpdateController;
