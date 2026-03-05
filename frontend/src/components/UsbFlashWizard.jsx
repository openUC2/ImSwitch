import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Stepper,
  Step,
  StepLabel,
  Box,
  Typography,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
  Paper,
  FormControl,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  InputLabel,
  TextField,
  Radio,
  RadioGroup,
  Divider,
  IconButton,
  Tooltip,
  Chip,
} from "@mui/material";
import {
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Usb as UsbIcon,
  Memory as MemoryIcon,
  CloudDownload as DownloadIcon,
  Settings as SettingsIcon,
  DeleteForever as EraseIcon,
  Router as CanIcon,
} from "@mui/icons-material";

// Redux slice
import * as usbFlashSlice from "../state/slices/usbFlashSlice";

// API functions
import apiUC2ConfigControllerListSerialPorts from "../backendapi/apiUC2ConfigControllerListSerialPorts";
import apiUC2ConfigControllerFlashMasterFirmwareUSB from "../backendapi/apiUC2ConfigControllerFlashMasterFirmwareUSB";
import apiUC2ConfigControllerGetOTAFirmwareServer from "../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer";
import apiUC2ConfigControllerSetOTAFirmwareServer from "../backendapi/apiUC2ConfigControllerSetOTAFirmwareServer";
import apiUC2ConfigControllerListAllFirmwareFiles from "../backendapi/apiUC2ConfigControllerListAllFirmwareFiles";
import apiUC2ConfigControllerSendCanAddress from "../backendapi/apiUC2ConfigControllerSendCanAddress";

const steps = [
  "Firmware Server",
  "Select Firmware",
  "Port & Options",
  "Flash Device",
  "CAN Address",
  "Complete",
];

// VID:PID → device type hint and auto-chip mapping
const VID_PID_HINTS = {
  "10c4:ea60": { label: "UC2 CAN HAT", chip: "esp32", color: "primary" },
  "303a:1001": { label: "UC2 XIAO (S3)", chip: "esp32s3", color: "secondary" },
  "1a86:55d4": { label: "UC2 XIAO (CH9102)", chip: "esp32s3", color: "secondary" },
  "303a:0002": { label: "ESP32-S2", chip: "esp32s2", color: "default" },
};

// Known CAN bus addresses
const CAN_ADDRESS_OPTIONS = [
  { value: null, label: "Skip (no CAN assignment)" },
  { value: 1, label: "Master (1)" },
  { value: 10, label: "A axis (10)" },
  { value: 11, label: "X axis (11)" },
  { value: 12, label: "Y axis (12)" },
  { value: 13, label: "Z axis (13)" },
  { value: 30, label: "LED (30)" },
];

/**
 * Helper: get device type hint from VID:PID
 */
const getDeviceHint = (port) => {
  if (port?.vid != null && port?.pid != null) {
    const key = port.vid.toString(16).padStart(4, "0") + ":" + port.pid.toString(16).padStart(4, "0");
    return VID_PID_HINTS[key] || null;
  }
  return null;
};

/**
 * USB Flash Wizard Component
 * Wizard for flashing ESP32 firmware via USB/esptool.
 * The user configures the firmware server, picks any .bin file, selects a port, then flashes.
 */
const UsbFlashWizard = ({ open, onClose }) => {
  const dispatch = useDispatch();
  const usbFlashState = useSelector(usbFlashSlice.getUsbFlashState);

  // Load initial data when wizard opens
  useEffect(() => {
    if (open) {
      loadInitialData();
    }
  }, [open]);

  const loadInitialData = async () => {
    try {
      // Load firmware server URL from backend
      const firmwareServer = await apiUC2ConfigControllerGetOTAFirmwareServer();
      dispatch(usbFlashSlice.setFirmwareServerUrl(firmwareServer.firmware_server_url || ""));
      dispatch(usbFlashSlice.setDefaultFirmwareServerUrl(firmwareServer.firmware_server_url || ""));
    } catch (error) {
      console.error("Error loading initial data:", error);
      dispatch(usbFlashSlice.setError("Failed to load firmware server configuration"));
    }
  };

  // --- Firmware server ---
  const handleLoadFirmwareFiles = async () => {
    try {
      dispatch(usbFlashSlice.setIsLoadingFirmware(true));
      dispatch(usbFlashSlice.clearMessages());

      // Save server URL to backend first
      if (usbFlashState.firmwareServerUrl) {
        try {
          await apiUC2ConfigControllerSetOTAFirmwareServer(usbFlashState.firmwareServerUrl);
        } catch (_) {
          // Non-fatal - the URL might already be set
        }
      }

      const result = await apiUC2ConfigControllerListAllFirmwareFiles();
      if (result.status === "success") {
        dispatch(usbFlashSlice.setFirmwareFiles(result.files || []));
        if (!result.files || result.files.length === 0) {
          dispatch(usbFlashSlice.setError("No .bin firmware files found on the server"));
        }
      } else {
        dispatch(usbFlashSlice.setError(result.message || "Failed to load firmware list"));
      }
    } catch (error) {
      console.error("Error loading firmware files:", error);
      dispatch(usbFlashSlice.setError("Failed to load firmware list from server"));
    } finally {
      dispatch(usbFlashSlice.setIsLoadingFirmware(false));
    }
  };

  // --- Serial ports ---
  const loadSerialPorts = async () => {
    try {
      dispatch(usbFlashSlice.setIsLoadingPorts(true));
      const ports = await apiUC2ConfigControllerListSerialPorts();
      dispatch(usbFlashSlice.setAvailablePorts(ports || []));
    } catch (error) {
      console.error("Error loading serial ports:", error);
      dispatch(usbFlashSlice.setError("Failed to load serial ports"));
    } finally {
      dispatch(usbFlashSlice.setIsLoadingPorts(false));
    }
  };

  // --- Navigation ---
  const handleNext = async () => {
    const step = usbFlashState.currentStep;
    dispatch(usbFlashSlice.clearMessages());

    if (step === 0) {
      // Firmware server - validate URL and load files
      if (!usbFlashState.firmwareServerUrl) {
        dispatch(usbFlashSlice.setError("Please enter a firmware server URL"));
        return;
      }
      await handleLoadFirmwareFiles();
    } else if (step === 1) {
      // Select firmware - validate selection
      if (!usbFlashState.selectedFirmware) {
        dispatch(usbFlashSlice.setError("Please select a firmware file"));
        return;
      }
      // Pre-load serial ports for next step
      loadSerialPorts();
    } else if (step === 3) {
      // Start flashing
      await startFlashing();
      return; // Don't auto-advance; signal-driven progress does it
    } else if (step === 4) {
      // CAN address step - send if configured
      if (usbFlashState.canAddress != null) {
        await sendCanAddress();
        return; // Will advance on success
      }
      // If skipped, just advance
    }

    dispatch(usbFlashSlice.nextStep());
  };

  const handleBack = () => {
    dispatch(usbFlashSlice.clearMessages());
    dispatch(usbFlashSlice.previousStep());
  };

  const handleClose = () => {
    dispatch(usbFlashSlice.resetWizard());
    onClose();
  };

  // Start over from scratch (used in error recovery)
  const handleStartOver = () => {
    dispatch(usbFlashSlice.resetWizard());
  };

  // Auto-detect chip type when a port is selected
  const handlePortSelect = (portDevice) => {
    dispatch(usbFlashSlice.setSelectedPort(portDevice));
    if (portDevice && portDevice !== "auto") {
      const portObj = usbFlashState.availablePorts.find((p) => p.device === portDevice);
      const hint = portObj ? getDeviceHint(portObj) : null;
      if (hint) {
        dispatch(usbFlashSlice.setChipType(hint.chip));
        // For XIAO, auto-enable skip disconnect
        if (hint.chip === "esp32s3" || hint.chip === "esp32s2") {
          dispatch(usbFlashSlice.setSkipDisconnect(true));
        } else {
          dispatch(usbFlashSlice.setSkipDisconnect(false));
        }
      } else {
        dispatch(usbFlashSlice.setChipType("auto"));
        dispatch(usbFlashSlice.setSkipDisconnect(false));
      }
    } else {
      dispatch(usbFlashSlice.setChipType("auto"));
      dispatch(usbFlashSlice.setSkipDisconnect(false));
    }
  };

  // --- Flashing ---
  const startFlashing = async () => {
    try {
      dispatch(usbFlashSlice.setIsFlashing(true));
      dispatch(usbFlashSlice.setFlashStatus("disconnecting"));
      dispatch(usbFlashSlice.setFlashProgress(0));
      dispatch(usbFlashSlice.setFlashMessage("Starting flash process..."));
      dispatch(usbFlashSlice.clearMessages());

      // Call the flash API with all parameters
      const result = await apiUC2ConfigControllerFlashMasterFirmwareUSB(
        usbFlashState.selectedPort,
        usbFlashState.portMatch,
        usbFlashState.baudRate,
        usbFlashState.selectedFirmware?.filename || null,
        usbFlashState.reconnectAfter,
        usbFlashState.chipType,
        usbFlashState.eraseFlash,
        usbFlashState.skipDisconnect
      );

      dispatch(usbFlashSlice.setFlashResult(result));

      if (result.status === "success") {
        dispatch(usbFlashSlice.setFlashStatus("success"));
        dispatch(usbFlashSlice.setFlashProgress(100));
        dispatch(usbFlashSlice.setFlashMessage("Firmware flashed successfully!"));
        dispatch(usbFlashSlice.setSuccessMessage("Firmware has been updated successfully"));
        dispatch(usbFlashSlice.nextStep()); // Move to CAN address step
      } else if (result.status === "warning") {
        dispatch(usbFlashSlice.setFlashStatus("success"));
        dispatch(usbFlashSlice.setFlashProgress(100));
        dispatch(usbFlashSlice.setFlashMessage("Firmware flashed, but reconnection failed"));
        dispatch(usbFlashSlice.setError(result.message || "Reconnection failed"));
        dispatch(usbFlashSlice.nextStep());
      } else {
        dispatch(usbFlashSlice.setFlashStatus("failed"));
        dispatch(usbFlashSlice.setFlashMessage("Flashing failed"));
        dispatch(usbFlashSlice.setError(result.message || "Unknown error"));
        dispatch(usbFlashSlice.setFlashDetails(result.details));
      }
    } catch (error) {
      console.error("Error during flashing:", error);
      dispatch(usbFlashSlice.setFlashStatus("failed"));
      dispatch(usbFlashSlice.setFlashMessage("Flashing failed"));
      dispatch(usbFlashSlice.setError("Failed to flash firmware: " + error.message));
    } finally {
      dispatch(usbFlashSlice.setIsFlashing(false));
    }
  };

  // --- CAN address assignment ---
  const sendCanAddress = async () => {
    try {
      dispatch(usbFlashSlice.setIsFlashing(true));
      dispatch(usbFlashSlice.setFlashMessage("Assigning CAN address..."));

      const port = usbFlashState.selectedPort || usbFlashState.flashResult?.port;
      if (!port) {
        dispatch(usbFlashSlice.setError("No port available for CAN address assignment"));
        dispatch(usbFlashSlice.setIsFlashing(false));
        return;
      }

      const result = await apiUC2ConfigControllerSendCanAddress(
        port,
        usbFlashState.canAddress,
        usbFlashState.canBaudRate
      );

      if (result.status === "success") {
        dispatch(usbFlashSlice.setSuccessMessage(`CAN address ${usbFlashState.canAddress} assigned successfully`));
        dispatch(usbFlashSlice.nextStep()); // Move to completion
      } else {
        dispatch(usbFlashSlice.setError(result.message || "CAN address assignment failed"));
      }
    } catch (error) {
      console.error("Error sending CAN address:", error);
      dispatch(usbFlashSlice.setError("Failed to assign CAN address: " + error.message));
    } finally {
      dispatch(usbFlashSlice.setIsFlashing(false));
    }
  };

  // ======================================
  // Step renderers
  // ======================================
  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return renderFirmwareServer();
      case 1:
        return renderFirmwareSelection();
      case 2:
        return renderPortSelection();
      case 3:
        return renderFlashProgress();
      case 4:
        return renderCanAddress();
      case 5:
        return renderCompletion();
      default:
        return null;
    }
  };

  // --- Step 0: Firmware Server ---
  const renderFirmwareServer = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        <DownloadIcon sx={{ mr: 1, verticalAlign: "middle" }} />
        Firmware Server Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Enter the URL of the firmware server that hosts the .bin files.
      </Typography>

      <Box sx={{ mt: 3 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <TextField
            fullWidth
            label="Firmware Server URL"
            value={usbFlashState.firmwareServerUrl}
            onChange={(e) => dispatch(usbFlashSlice.setFirmwareServerUrl(e.target.value))}
            margin="normal"
            placeholder="https://example.com/firmware/latest"
            helperText={
              usbFlashState.defaultFirmwareServerUrl
                ? "Default: " + usbFlashState.defaultFirmwareServerUrl
                : "Set the URL where firmware .bin files are hosted"
            }
          />
          <Tooltip title="Reload firmware files from server">
            <IconButton
              onClick={handleLoadFirmwareFiles}
              disabled={usbFlashState.isLoadingFirmware || !usbFlashState.firmwareServerUrl}
              sx={{ mt: 1 }}
            >
              {usbFlashState.isLoadingFirmware ? (
                <CircularProgress size={24} />
              ) : (
                <RefreshIcon />
              )}
            </IconButton>
          </Tooltip>
        </Box>

        {usbFlashState.firmwareFiles.length > 0 && (
          <Paper sx={{ mt: 2, p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              {usbFlashState.firmwareFiles.length} firmware file(s) found on server
            </Typography>
          </Paper>
        )}
      </Box>
    </Box>
  );

  // --- Step 1: Select Firmware ---
  const renderFirmwareSelection = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        <MemoryIcon sx={{ mr: 1, verticalAlign: "middle" }} />
        Select Firmware
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Choose the firmware file you want to flash to the device.
      </Typography>

      {usbFlashState.isLoadingFirmware ? (
        <Box sx={{ display: "flex", justifyContent: "center", my: 3 }}>
          <CircularProgress />
        </Box>
      ) : usbFlashState.firmwareFiles.length > 0 ? (
        <Box sx={{ mt: 2 }}>
          <FormControl component="fieldset" sx={{ width: "100%" }}>
            <RadioGroup
              value={usbFlashState.selectedFirmware?.filename || ""}
              onChange={(e) => {
                const selected = usbFlashState.firmwareFiles.find(
                  (f) => f.filename === e.target.value
                );
                dispatch(usbFlashSlice.setSelectedFirmware(selected || null));
              }}
            >
              {usbFlashState.firmwareFiles.map((fw) => (
                <Paper
                  key={fw.filename}
                  sx={{
                    p: 1.5,
                    mb: 1,
                    border: usbFlashState.selectedFirmware?.filename === fw.filename ? 2 : 1,
                    borderColor:
                      usbFlashState.selectedFirmware?.filename === fw.filename
                        ? "primary.main"
                        : "divider",
                    cursor: "pointer",
                  }}
                  onClick={() => dispatch(usbFlashSlice.setSelectedFirmware(fw))}
                >
                  <FormControlLabel
                    value={fw.filename}
                    control={<Radio />}
                    label={
                      <Box>
                        <Typography variant="body1" sx={{ fontFamily: "monospace" }}>
                          {fw.filename}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {(fw.size / 1024).toFixed(1)} KB
                          {fw.mod_time && (" \u00b7 " + fw.mod_time)}
                        </Typography>
                      </Box>
                    }
                    sx={{ width: "100%", m: 0 }}
                  />
                </Paper>
              ))}
            </RadioGroup>
          </FormControl>

          <Button
            size="small"
            startIcon={<RefreshIcon />}
            onClick={handleLoadFirmwareFiles}
            sx={{ mt: 1 }}
          >
            Refresh
          </Button>
        </Box>
      ) : (
        <Alert severity="warning" sx={{ mt: 2 }}>
          No firmware files found. Go back and check the firmware server URL.
        </Alert>
      )}
    </Box>
  );

  // --- Step 2: Port & Options ---
  const renderPortSelection = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        <UsbIcon sx={{ mr: 1, verticalAlign: "middle" }} />
        USB Port &amp; Flash Options
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Select the serial port connected to the ESP32 device and configure flash options.
      </Typography>

      {usbFlashState.isLoadingPorts ? (
        <Box sx={{ display: "flex", justifyContent: "center", my: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ mt: 3 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
            <Typography variant="subtitle2">Port Selection:</Typography>
            <Tooltip title="Refresh port list">
              <IconButton onClick={loadSerialPorts} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>

          <FormControl component="fieldset">
            <RadioGroup
              value={usbFlashState.selectedPort || "auto"}
              onChange={(e) => {
                const value = e.target.value === "auto" ? null : e.target.value;
                handlePortSelect(value);
              }}
            >
              <FormControlLabel
                value="auto"
                control={<Radio />}
                label={
                  <Box>
                    <Typography variant="body1">Auto-detect</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Automatically find the device by matching "{usbFlashState.portMatch}" in port metadata
                    </Typography>
                  </Box>
                }
              />
              <Divider sx={{ my: 1 }} />
              {usbFlashState.availablePorts.length > 0 ? (
                usbFlashState.availablePorts.map((port) => {
                  const hint = getDeviceHint(port);
                  return (
                    <FormControlLabel
                      key={port.device}
                      value={port.device}
                      control={<Radio />}
                      label={
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <Box>
                            <Typography variant="body1">
                              {port.device}
                              {hint && (
                                <Chip
                                  label={hint.label}
                                  color={hint.color}
                                  size="small"
                                  sx={{ ml: 1 }}
                                />
                              )}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {port.description || port.manufacturer || port.product || "Unknown device"}
                              {port.vid != null && port.pid != null && (
                                " (VID:" + port.vid.toString(16).padStart(4, "0") +
                                " PID:" + port.pid.toString(16).padStart(4, "0") + ")"
                              )}
                              {hint && ` → auto: ${hint.chip}`}
                            </Typography>
                          </Box>
                        </Box>
                      }
                    />
                  );
                })
              ) : (
                <Alert severity="warning" sx={{ mt: 1 }}>
                  No serial ports detected. Make sure the device is connected.
                </Alert>
              )}
            </RadioGroup>
          </FormControl>

          <TextField
            fullWidth
            label="Port Match Pattern (for auto-detect)"
            value={usbFlashState.portMatch}
            onChange={(e) => dispatch(usbFlashSlice.setPortMatch(e.target.value))}
            margin="normal"
            helperText="Substring to search for in port metadata when auto-detecting"
            size="small"
          />

          {/* Flash options */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              <SettingsIcon sx={{ mr: 1, verticalAlign: "middle", fontSize: 18 }} />
              Flash Options
            </Typography>

            <FormControl fullWidth margin="normal" size="small">
              <InputLabel>Baud Rate</InputLabel>
              <Select
                value={usbFlashState.baudRate}
                onChange={(e) => dispatch(usbFlashSlice.setBaudRate(e.target.value))}
                label="Baud Rate"
              >
                <MenuItem value={115200}>115200</MenuItem>
                <MenuItem value={230400}>230400</MenuItem>
                <MenuItem value={460800}>460800</MenuItem>
                <MenuItem value={921600}>921600 (recommended)</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth margin="normal" size="small">
              <InputLabel>Chip Type</InputLabel>
              <Select
                value={usbFlashState.chipType}
                onChange={(e) => dispatch(usbFlashSlice.setChipType(e.target.value))}
                label="Chip Type"
              >
                <MenuItem value="auto">Auto-detect (from VID:PID)</MenuItem>
                <MenuItem value="esp32">ESP32 (HAT / standard)</MenuItem>
                <MenuItem value="esp32s3">ESP32-S3 (XIAO)</MenuItem>
                <MenuItem value="esp32s2">ESP32-S2</MenuItem>
                <MenuItem value="esp32c3">ESP32-C3</MenuItem>
              </Select>
            </FormControl>

            <FormControlLabel
              control={
                <Checkbox
                  checked={usbFlashState.eraseFlash}
                  onChange={(e) => dispatch(usbFlashSlice.setEraseFlash(e.target.checked))}
                />
              }
              label={
                <Box>
                  <Typography variant="body2">Erase flash before writing</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Full erase – useful for clean installs or switching firmware type
                  </Typography>
                </Box>
              }
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={usbFlashState.skipDisconnect}
                  onChange={(e) => dispatch(usbFlashSlice.setSkipDisconnect(e.target.checked))}
                />
              }
              label={
                <Box>
                  <Typography variant="body2">Skip ImSwitch disconnect</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Enable for XIAO and other non-HAT devices that don't share the ImSwitch serial port
                  </Typography>
                </Box>
              }
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={usbFlashState.reconnectAfter}
                  onChange={(e) => dispatch(usbFlashSlice.setReconnectAfter(e.target.checked))}
                />
              }
              label="Reconnect ImSwitch to device after flashing"
            />
          </Paper>

          {/* Selected firmware summary */}
          {usbFlashState.selectedFirmware && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <strong>Firmware:</strong> {usbFlashState.selectedFirmware.filename} ({(usbFlashState.selectedFirmware.size / 1024).toFixed(1)} KB)
              {usbFlashState.chipType !== "auto" && <> &bull; <strong>Chip:</strong> {usbFlashState.chipType}</>}
              {usbFlashState.eraseFlash && <> &bull; <Chip label="Erase first" size="small" color="warning" sx={{ ml: 0.5 }} /></>}
            </Alert>
          )}
        </Box>
      )}
    </Box>
  );

  // --- Step 3: Flash Progress ---
  const renderFlashProgress = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        <MemoryIcon sx={{ mr: 1, verticalAlign: "middle" }} />
        Flashing Progress
      </Typography>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
          {usbFlashState.flashStatus === "success" ? (
            <CheckCircleIcon color="success" sx={{ fontSize: 40, mr: 2 }} />
          ) : usbFlashState.flashStatus === "failed" ? (
            <ErrorIcon color="error" sx={{ fontSize: 40, mr: 2 }} />
          ) : (
            <CircularProgress size={40} sx={{ mr: 2 }} />
          )}
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6">
              {usbFlashState.flashMessage || "Preparing..."}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {getStatusDescription(usbFlashState.flashStatus)}
            </Typography>
          </Box>
        </Box>

        <LinearProgress
          variant="determinate"
          value={usbFlashState.flashProgress}
          sx={{ height: 10, borderRadius: 5, mb: 2 }}
        />

        <Typography variant="body2" align="center" color="text.secondary">
          {usbFlashState.flashProgress}% Complete
        </Typography>

        {usbFlashState.flashDetails && (
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography
              variant="body2"
              component="pre"
              sx={{ whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: "0.8rem" }}
            >
              {usbFlashState.flashDetails}
            </Typography>
          </Alert>
        )}

        {usbFlashState.flashStatus === "failed" && (
          <Box sx={{ mt: 2, textAlign: "center", display: "flex", gap: 2, justifyContent: "center" }}>
            <Button variant="outlined" color="primary" onClick={startFlashing}>
              Retry
            </Button>
            <Button variant="outlined" color="secondary" onClick={handleStartOver}>
              Start Over
            </Button>
          </Box>
        )}
      </Paper>

      <Alert severity="warning" sx={{ mt: 2 }}>
        <Typography variant="body2">
          <strong>Important:</strong> Do not disconnect the device or close this dialog during flashing.
          The ESP32 will be disconnected temporarily during the flash process.
        </Typography>
      </Alert>
    </Box>
  );

  // --- Step 4: CAN Address Assignment ---
  const renderCanAddress = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        <CanIcon sx={{ mr: 1, verticalAlign: "middle" }} />
        CAN Bus Address Assignment
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Optionally assign a CAN bus address to the freshly-flashed device.
        This tells the device which axis/role it serves on the CAN bus.
      </Typography>

      <Paper sx={{ p: 3, mt: 3 }}>
        <FormControl component="fieldset" sx={{ width: "100%" }}>
          <RadioGroup
            value={usbFlashState.canAddress === null ? "skip" : String(usbFlashState.canAddress)}
            onChange={(e) => {
              const val = e.target.value === "skip" ? null : parseInt(e.target.value, 10);
              dispatch(usbFlashSlice.setCanAddress(val));
            }}
          >
            {CAN_ADDRESS_OPTIONS.map((opt) => (
              <FormControlLabel
                key={opt.value === null ? "skip" : opt.value}
                value={opt.value === null ? "skip" : String(opt.value)}
                control={<Radio />}
                label={opt.label}
              />
            ))}
          </RadioGroup>
        </FormControl>

        {usbFlashState.canAddress != null && (
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Serial Baud Rate</InputLabel>
              <Select
                value={usbFlashState.canBaudRate}
                onChange={(e) => dispatch(usbFlashSlice.setCanBaudRate(e.target.value))}
                label="Serial Baud Rate"
              >
                <MenuItem value={9600}>9600</MenuItem>
                <MenuItem value={115200}>115200 (default)</MenuItem>
                <MenuItem value={500000}>500000</MenuItem>
              </Select>
            </FormControl>

            <Alert severity="info" sx={{ mt: 2 }}>
              Will send <code>{`{"task":"/can_act","address":${usbFlashState.canAddress}}`}</code> to{" "}
              {usbFlashState.selectedPort || usbFlashState.flashResult?.port || "auto-detected port"}
            </Alert>
          </Box>
        )}
      </Paper>

      {usbFlashState.flashResult && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Firmware was flashed successfully to <strong>{usbFlashState.flashResult.port}</strong>
          {usbFlashState.flashResult.chip && <> (chip: {usbFlashState.flashResult.chip})</>}
        </Alert>
      )}
    </Box>
  );

  // --- Step 5: Completion ---
  const renderCompletion = () => (
    <Box sx={{ mt: 2, textAlign: "center" }}>
      {usbFlashState.flashResult?.status === "success" ? (
        <>
          <CheckCircleIcon color="success" sx={{ fontSize: 80, mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Firmware Update Complete!
          </Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            The firmware has been successfully flashed to the device.
          </Typography>
        </>
      ) : usbFlashState.flashResult?.status === "warning" ? (
        <>
          <WarningIcon color="warning" sx={{ fontSize: 80, mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Firmware Flashed with Warning
          </Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            The firmware was flashed, but there was an issue reconnecting.
          </Typography>
        </>
      ) : (
        <>
          <ErrorIcon color="error" sx={{ fontSize: 80, mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Firmware Update Failed
          </Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            {usbFlashState.flashResult?.message || "An error occurred during flashing."}
          </Typography>
        </>
      )}

      {usbFlashState.flashResult && (
        <Paper sx={{ p: 2, mt: 3, textAlign: "left" }}>
          <Typography variant="subtitle2" gutterBottom>
            Flash Details:
          </Typography>
          <List dense>
            {usbFlashState.flashResult.port && (
              <ListItem>
                <ListItemText primary="Port" secondary={usbFlashState.flashResult.port} />
              </ListItem>
            )}
            {usbFlashState.flashResult.firmware && (
              <ListItem>
                <ListItemText primary="Firmware" secondary={usbFlashState.flashResult.firmware} />
              </ListItem>
            )}
            {usbFlashState.flashResult.baud && (
              <ListItem>
                <ListItemText primary="Baud Rate" secondary={usbFlashState.flashResult.baud} />
              </ListItem>
            )}
          </List>
        </Paper>
      )}
    </Box>
  );

  const getStatusDescription = (status) => {
    switch (status) {
      case "disconnecting":
        return "Disconnecting ImSwitch from the ESP32...";
      case "downloading":
        return "Downloading firmware from the server...";
      case "flashing":
        return "Writing firmware to the device via USB (esptool)...";
      case "reconnecting":
        return "Reconnecting to the device...";
      case "success":
        return "Firmware update completed successfully!";
      case "failed":
        return "An error occurred during the update process.";
      default:
        return "Initializing...";
    }
  };

  const isNextDisabled = () => {
    const step = usbFlashState.currentStep;
    if (step === 0 && !usbFlashState.firmwareServerUrl) return true;
    if (step === 1 && !usbFlashState.selectedFirmware) return true;
    if (step === 3 && usbFlashState.isFlashing) return true;
    if (step === 4 && usbFlashState.isFlashing) return true;
    return false;
  };

  // Label for "Next" button depending on step
  const getNextLabel = () => {
    const step = usbFlashState.currentStep;
    if (step === 3) return "Start Flashing";
    if (step === 4 && usbFlashState.canAddress != null) return "Assign CAN Address";
    if (step === 4) return "Skip & Finish";
    return "Next";
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
      disableEscapeKeyDown={usbFlashState.isFlashing}
    >
      <DialogTitle>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <UsbIcon color="primary" />
          <Typography variant="h6">USB Firmware Flash Wizard</Typography>
        </Box>
      </DialogTitle>

      <DialogContent>
        {/* Stepper */}
        <Stepper activeStep={usbFlashState.currentStep} sx={{ mb: 3 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {/* Error/Success messages */}
        {usbFlashState.error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => dispatch(usbFlashSlice.clearMessages())}>
            {usbFlashState.error}
          </Alert>
        )}
        {usbFlashState.successMessage && (
          <Alert severity="success" sx={{ mb: 2 }} onClose={() => dispatch(usbFlashSlice.clearMessages())}>
            {usbFlashState.successMessage}
          </Alert>
        )}

        {/* Step Content */}
        {renderStepContent(usbFlashState.currentStep)}
      </DialogContent>

      <DialogActions>
        <Button onClick={handleClose} disabled={usbFlashState.isFlashing}>
          {usbFlashState.currentStep === 5 ? "Close" : "Cancel"}
        </Button>
        {usbFlashState.currentStep > 0 && usbFlashState.currentStep < 5 && (
          <Button onClick={handleBack} disabled={usbFlashState.isFlashing}>
            Back
          </Button>
        )}
        {usbFlashState.currentStep < 5 && (
          <Button
            variant="contained"
            onClick={handleNext}
            disabled={isNextDisabled() || usbFlashState.isFlashing}
          >
            {getNextLabel()}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default UsbFlashWizard;
