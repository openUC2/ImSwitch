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
  InputAdornment,
  Switch,
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
  Search as SearchIcon,
  FlashOn as FlashOnIcon,
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
import apiUC2ConfigControllerProbeDeviceState from "../backendapi/apiUC2ConfigControllerProbeDeviceState";
import apiUC2ConfigControllerTestDeviceAction from "../backendapi/apiUC2ConfigControllerTestDeviceAction";
import apiUC2ConfigControllerCancelUSBFlash from "../backendapi/apiUC2ConfigControllerCancelUSBFlash";

// Firmware filenames that ship the *master* (CAN HAT) image. When the user
// picks one of these we should disconnect ImSwitch first and reconnect after.
// Slave images skip both.
const MASTER_FIRMWARE_REGEX = /master|hat|^id_1_/i;
const isMasterFirmwareName = (filename) =>
  !!filename && MASTER_FIRMWARE_REGEX.test(filename);

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

  // Local UI state for the device validation panel (completion step)
  const [testBaud, setTestBaud] = React.useState(115200);
  const [testBusy, setTestBusy] = React.useState(false);
  const [testResult, setTestResult] = React.useState(null); // { kind, status, response, command }
  const [motorPosition, setMotorPosition] = React.useState(1000);
  const [motorSpeed, setMotorSpeed] = React.useState(2000);
  const [motorStepperId, setMotorStepperId] = React.useState(1);
  const [ledRgb, setLedRgb] = React.useState(25);
  const [laserVal, setLaserVal] = React.useState(118);

  // Keep testBaud in sync with the wizard's canBaudRate when entering the completion step
  useEffect(() => {
    if (usbFlashState.canBaudRate) {
      setTestBaud(usbFlashState.canBaudRate);
    }
  }, [usbFlashState.canBaudRate]);

  // Drive skipDisconnect / reconnectAfter defaults from the selected firmware
  // filename. Master firmware needs ImSwitch to release the serial first and
  // reopen it after; slave firmware doesn't share the port so both are off.
  // The user can still override the checkboxes manually after this fires.
  useEffect(() => {
    const filename = usbFlashState.selectedFirmware?.filename;
    if (!filename) return;
    const isMaster = isMasterFirmwareName(filename);
    dispatch(usbFlashSlice.setSkipDisconnect(!isMaster));
    dispatch(usbFlashSlice.setReconnectAfter(isMaster));
  }, [usbFlashState.selectedFirmware?.filename, dispatch]);

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

  // Skip directly to CAN address step (no flashing)
  const handleSkipToCanAddress = () => {
    dispatch(usbFlashSlice.clearMessages());
    // Load serial ports for the CAN step
    loadSerialPorts();
    // Jump directly to the CAN address step (step 4)
    dispatch(usbFlashSlice.setCurrentStep(4));
    dispatch(usbFlashSlice.setFlashResult({ status: "skipped", message: "Flashing skipped" }));
  };

  // Auto-detect chip type when a port is selected. skipDisconnect is *not*
  // touched here — it's driven from the firmware filename above so the
  // master/slave choice stays the single source of truth.
  const handlePortSelect = (portDevice) => {
    dispatch(usbFlashSlice.setSelectedPort(portDevice));
    if (portDevice && portDevice !== "auto") {
      const portObj = usbFlashState.availablePorts.find((p) => p.device === portDevice);
      const hint = portObj ? getDeviceHint(portObj) : null;
      dispatch(usbFlashSlice.setChipType(hint ? hint.chip : "auto"));
    } else {
      dispatch(usbFlashSlice.setChipType("auto"));
    }
  };

  // Cancel a flash that's currently in progress. Updates UI optimistically
  // so the user gets immediate feedback even before the backend emits
  // sigUSBFlashStatusUpdate{status:"cancelled"}.
  const handleCancelFlashing = async () => {
    try {
      dispatch(usbFlashSlice.setFlashMessage("Cancelling..."));
      await apiUC2ConfigControllerCancelUSBFlash();
    } catch (error) {
      console.error("Cancel flash failed:", error);
      // Still unblock the UI — the user wants out.
      dispatch(usbFlashSlice.setError("Cancel request failed: " + error.message));
    } finally {
      dispatch(usbFlashSlice.setIsFlashing(false));
      dispatch(usbFlashSlice.setFlashStatus("cancelled"));
      dispatch(
        usbFlashSlice.setFlashResult({
          status: "cancelled",
          message: "Flashing cancelled by user",
        })
      );
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

      if (result.boot_loop_detected) {
        dispatch(usbFlashSlice.setError(
          "Boot-loop detected! The device is showing 'invalid header: 0xffffffff'. " +
          "The firmware was not flashed correctly. Try using a merged firmware (_merged.bin) " +
          "with erase flash enabled."
        )); 
        dispatch(usbFlashSlice.setIsFlashing(false));
        return;
      }

      if (result.status === "success") {
        // Store state_response for display in the completion step
        if (result.state_response !== undefined) {
          dispatch(usbFlashSlice.setCanStateResponse(result.state_response));
        }
        dispatch(usbFlashSlice.setSuccessMessage(
          `CAN address ${usbFlashState.canAddress} assigned successfully` +
          (result.firmware_verified ? " (firmware verified ✓)" : "")
        ));
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

  // --- Hardware validation: motor / ledarray / laser test commands ---
  const runDeviceTest = async (kind, extraParams = {}) => {
    const port = usbFlashState.selectedPort || usbFlashState.flashResult?.port;
    if (!port) {
      dispatch(usbFlashSlice.setError("No port available for device test"));
      return;
    }
    try {
      setTestBusy(true);
      setTestResult(null);
      dispatch(usbFlashSlice.clearMessages());
      const result = await apiUC2ConfigControllerTestDeviceAction({
        port,
        deviceType: kind,
        baud: testBaud,
        ...extraParams,
      });
      setTestResult({ kind, ...result });
      if (result.status === "error") {
        dispatch(usbFlashSlice.setError(result.message || `${kind} test failed`));
      } else if (result.status === "warning") {
        dispatch(usbFlashSlice.setError(`${kind} test returned a warning - check response below.`));
      } else {
        dispatch(usbFlashSlice.setSuccessMessage(`${kind} test command sent successfully✓`));
      }
    } catch (error) {
      console.error(`Device test (${kind}) failed:`, error);
      dispatch(usbFlashSlice.setError(`${kind} test failed: ` + error.message));
    } finally {
      setTestBusy(false);
    }
  };

  // --- Standalone device state probe (used in completion step) ---
  const probeState = async () => {
    const port = usbFlashState.selectedPort || usbFlashState.flashResult?.port;
    if (!port) {
      dispatch(usbFlashSlice.setError("No port available for probing"));
      return;
    }
    try {
      dispatch(usbFlashSlice.setIsFlashing(true));
      const result = await apiUC2ConfigControllerProbeDeviceState(port, usbFlashState.canBaudRate);
      dispatch(usbFlashSlice.setCanStateResponse(result.state_response || result.message || "No response"));
      if (!result.firmware_ok) {
        dispatch(usbFlashSlice.setError("Device may not be running firmware correctly (invalid header detected)"));
      }
    } catch (error) {
      console.error("Error probing device state:", error);
      dispatch(usbFlashSlice.setError("Probe failed: " + error.message));
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
  const renderFirmwareSelection = () => {
    // Filter by merged-firmware toggle first, then by search query
    const query = (usbFlashState.firmwareSearchQuery || "").toLowerCase();
    const filteredFiles = usbFlashState.firmwareFiles
      .filter((fw) => usbFlashState.showMergedFirmware || !fw.filename.includes("_merged"))
      .filter((fw) => !query || fw.filename.toLowerCase().includes(query));

    return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        <MemoryIcon sx={{ mr: 1, verticalAlign: "middle" }} />
        Select Firmware
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Choose the firmware file you want to flash to the device.
        Use <strong>_merged</strong> binaries when flashing with "erase flash"
        or when flashing a new device for the first time.
      </Typography>

      {usbFlashState.isLoadingFirmware ? (
        <Box sx={{ display: "flex", justifyContent: "center", my: 3 }}>
          <CircularProgress />
        </Box>
      ) : usbFlashState.firmwareFiles.length > 0 ? (
        <Box sx={{ mt: 2 }}>
          {/* Search bar */}
          <TextField
            fullWidth
            size="small"
            placeholder="Search firmware files..."
            value={usbFlashState.firmwareSearchQuery || ""}
            onChange={(e) => dispatch(usbFlashSlice.setFirmwareSearchQuery(e.target.value))}
            sx={{ mb: 1 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon color="action" />
                </InputAdornment>
              ),
            }}
          />

          {/* Merged firmware opt-in toggle (off by default to avoid BOOT_ADDR errors) */}
          <FormControlLabel
            control={
              <Switch
                checked={usbFlashState.showMergedFirmware}
                onChange={(e) => dispatch(usbFlashSlice.setShowMergedFirmware(e.target.checked))}
                size="small"
                color="warning"
              />
            }
            label={<Typography variant="body2">Show merged firmware (advanced)</Typography>}
            sx={{ mb: 1 }}
          />
          {usbFlashState.showMergedFirmware && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              Merged binaries include the bootloader and partition table. Only use them when
              flashing a brand-new device or when "erase flash" is required. If you see{" "}
              <code>invalid header: 0xffffffff</code> after flashing, the binary was not built
              with <code>BOOT_ADDR=0x0000</code> and cannot be used as a merged image.
            </Alert>
          )}

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
              {filteredFiles.length > 0 ? filteredFiles.map((fw) => (
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
                          {fw.filename.includes("_merged") && (
                            <Chip label="merged" size="small" color="success" sx={{ ml: 1 }} />
                          )}
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
              )) : (
                <Alert severity="info" sx={{ mt: 1 }}>
                  No firmware files match "{usbFlashState.firmwareSearchQuery}".
                </Alert>
              )}
            </RadioGroup>
          </FormControl>

          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
            Showing {filteredFiles.length} of {usbFlashState.firmwareFiles.length} files
          </Typography>

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
  };

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
                    Full erase – useful for clean installs or switching firmware type.
                    Requires a <strong>merged</strong> firmware binary.
                  </Typography>
                </Box>
              }
            />

            {/* Warning: erase + non-merged */}
            {usbFlashState.eraseFlash &&
              usbFlashState.selectedFirmware &&
              !usbFlashState.selectedFirmware.filename.includes("_merged") && (
              <Alert severity="warning" sx={{ mt: 1, mb: 1 }}>
                <strong>Warning:</strong> Erase flash is enabled but the selected firmware
                (<code>{usbFlashState.selectedFirmware.filename}</code>) is not a merged binary.
                The backend will automatically disable erase to protect the existing
                bootloader. Use a <code>_merged.bin</code> firmware for erasing.
              </Alert>
            )}

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

          {/* 12V power warning for XIAO boards */}
          <Alert severity="warning" icon={<FlashOnIcon />} sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>⚡ XIAO boards:</strong> If flashing fails or the device doesn't enter
              download mode, try turning off the 12V power supply (press the emergency stop
              button) before flashing. The XIAO USB boot pin may not trigger correctly
              with 12V power connected.
            </Typography>
          </Alert>

          {/* Selected firmware summary */}
          {usbFlashState.selectedFirmware && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <strong>Firmware:</strong> {usbFlashState.selectedFirmware.filename} ({(usbFlashState.selectedFirmware.size / 1024).toFixed(1)} KB)
              {usbFlashState.selectedFirmware.filename.includes("_merged")
                ? <Chip label="merged" size="small" color="success" sx={{ ml: 0.5 }} />
                : <Chip label="app-only" size="small" color="default" sx={{ ml: 0.5 }} />
              }
              {usbFlashState.chipType !== "auto" && <> &bull; <strong>Chip:</strong> {usbFlashState.chipType}</>}
              {usbFlashState.eraseFlash && <> &bull; <Chip label="Erase first" size="small" color="warning" sx={{ ml: 0.5 }} /></>}
            </Alert>
          )}

          {/* Skip flash – just assign CAN address */}
          <Divider sx={{ my: 2 }} />
          <Button
            variant="outlined"
            color="secondary"
            startIcon={<CanIcon />}
            onClick={handleSkipToCanAddress}
            fullWidth
          >
            Skip Flashing – Just Assign CAN Address
          </Button>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block", textAlign: "center" }}>
            Use this if the device already has the correct firmware and you only need to set its CAN bus address.
          </Typography>
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
          ) : usbFlashState.flashStatus === "cancelled" ? (
            <WarningIcon color="warning" sx={{ fontSize: 40, mr: 2 }} />
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

        {usbFlashState.isFlashing && (
          <Box sx={{ mt: 2, textAlign: "center" }}>
            <Button
              variant="outlined"
              color="warning"
              onClick={handleCancelFlashing}
            >
              Cancel Flashing
            </Button>
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
              Aborting mid-write can leave the chip in an inconsistent state — only cancel if the device is unreachable.
            </Typography>
          </Box>
        )}

        {(usbFlashState.flashStatus === "failed" ||
          usbFlashState.flashStatus === "cancelled") && (
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
        Assign a CAN bus address to the device.
        This tells the device which axis/role it serves on the CAN bus.
      </Typography>

      {/* Port selection (when flash was skipped) */}
      {usbFlashState.flashResult?.status === "skipped" && (
        <Paper sx={{ p: 2, mt: 2, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            <UsbIcon sx={{ mr: 1, verticalAlign: "middle", fontSize: 18 }} />
            Select Serial Port
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
            <Tooltip title="Refresh port list">
              <IconButton onClick={loadSerialPorts} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
          <FormControl component="fieldset" sx={{ width: "100%" }}>
            <RadioGroup
              value={usbFlashState.selectedPort || ""}
              onChange={(e) => dispatch(usbFlashSlice.setSelectedPort(e.target.value))}
            >
              {usbFlashState.availablePorts.map((port) => {
                const hint = getDeviceHint(port);
                return (
                  <FormControlLabel
                    key={port.device}
                    value={port.device}
                    control={<Radio />}
                    label={
                      <Typography variant="body2">
                        {port.device}
                        {hint && (
                          <Chip label={hint.label} color={hint.color} size="small" sx={{ ml: 1 }} />
                        )}
                      </Typography>
                    }
                  />
                );
              })}
            </RadioGroup>
          </FormControl>
          {usbFlashState.availablePorts.length === 0 && (
            <Alert severity="warning" sx={{ mt: 1 }}>
              No serial ports detected. Connect the device and click refresh.
            </Alert>
          )}
        </Paper>
      )}

      <Paper sx={{ p: 3, mt: usbFlashState.flashResult?.status === "skipped" ? 0 : 3 }}>
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
                <MenuItem value={115200}>115200 (default)</MenuItem>
                <MenuItem value={921600}>921600 (high-speed)</MenuItem>
              </Select>
            </FormControl>

            <Alert severity="info" sx={{ mt: 2 }}>
              Will send <code>{`{"task":"/can_act","address":${usbFlashState.canAddress}, "nodeId":${usbFlashState.canAddress}, "canMotorAxis":1}}`}</code> to{" "}
              {usbFlashState.selectedPort || usbFlashState.flashResult?.port || "auto-detected port"}
            </Alert>
          </Box>
        )}
      </Paper>

      {usbFlashState.flashResult && usbFlashState.flashResult.status === "success" && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Firmware was flashed successfully to <strong>{usbFlashState.flashResult.port}</strong>
          {usbFlashState.flashResult.chip && <> (chip: {usbFlashState.flashResult.chip})</>}
        </Alert>
      )}

      {usbFlashState.flashResult?.status === "skipped" && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Flashing was skipped. Only CAN address assignment will be performed.
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

      {/* Device state verification output from /state_get probe */}
      {usbFlashState.canStateResponse && (
        <Paper sx={{ p: 2, mt: 2, textAlign: "left" }}>
          <Typography variant="subtitle2" gutterBottom>
            <CheckCircleIcon color="success" sx={{ mr: 1, verticalAlign: "middle", fontSize: 18 }} />
            Device State Verification:
          </Typography>
          <Typography
            variant="body2"
            component="pre"
            sx={{
              whiteSpace: "pre-wrap",
              fontFamily: "monospace",
              fontSize: "0.8rem",
              mt: 1,
              overflowX: "auto",
            }}
          >
            {typeof usbFlashState.canStateResponse === "string"
              ? usbFlashState.canStateResponse
              : JSON.stringify(usbFlashState.canStateResponse, null, 2)}
          </Typography>
        </Paper>
      )}

      {/* Re-probe button + baud selector */}
      <Paper sx={{ p: 2, mt: 2, textAlign: "left" }}>
        <Typography variant="subtitle2" gutterBottom>
          Probe Device State
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, flexWrap: "wrap" }}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Baud Rate</InputLabel>
            <Select
              value={usbFlashState.canBaudRate}
              onChange={(e) => dispatch(usbFlashSlice.setCanBaudRate(e.target.value))}
              label="Baud Rate"
            >
              <MenuItem value={115200}>115200 (default)</MenuItem>
              <MenuItem value={921600}>921600 (high-speed)</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            color="primary"
            onClick={probeState}
            disabled={usbFlashState.isFlashing}
            startIcon={usbFlashState.isFlashing ? <CircularProgress size={16} /> : <CheckCircleIcon />}
          >
            {usbFlashState.isFlashing ? "Probing..." : "Probe Device"}
          </Button>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
          Sends {`{"task":"/state_get"}`} to the device and shows the firmware response.
        </Typography>
      </Paper>

      {/* Hardware validation: send device-specific test commands */}
      <Paper sx={{ p: 2, mt: 2, textAlign: "left" }}>
        <Typography variant="subtitle2" gutterBottom>
          <FlashOnIcon sx={{ mr: 1, verticalAlign: "middle", fontSize: 18 }} />
          Hardware Validation
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 2 }}>
          Send a real action to the slave to verify motor / LED array / laser firmware works.
        </Typography>

        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2, flexWrap: "wrap" }}>
          <TextField
            label="Baud Rate"
            type="number"
            size="small"
            value={testBaud}
            onChange={(e) => setTestBaud(parseInt(e.target.value, 10) || 115200)}
            sx={{ width: 140 }}
            helperText="Default 115200"
          />
        </Box>

        {/* Motor test */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
            Motor (/motor_act)
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 1 }}>
            <TextField
              label="Stepper ID"
              type="number"
              size="small"
              value={motorStepperId}
              onChange={(e) => setMotorStepperId(parseInt(e.target.value, 10) || 1)}
              sx={{ width: 100 }}
            />
            <TextField
              label="Speed"
              type="number"
              size="small"
              value={motorSpeed}
              onChange={(e) => setMotorSpeed(parseInt(e.target.value, 10) || 0)}
              sx={{ width: 110 }}
            />
            <TextField
              label="Position (steps)"
              type="number"
              size="small"
              value={motorPosition}
              onChange={(e) => setMotorPosition(parseInt(e.target.value, 10) || 0)}
              sx={{ width: 140 }}
            />
            <Button
              variant="outlined"
              size="small"
              disabled={testBusy}
              onClick={() => runDeviceTest("motor", {
                stepperid: motorStepperId,
                speed: motorSpeed,
                position: motorPosition,
                isabs: 0,
              })}
            >
              Move +
            </Button>
            <Button
              variant="outlined"
              size="small"
              disabled={testBusy}
              onClick={() => runDeviceTest("motor", {
                stepperid: motorStepperId,
                speed: motorSpeed,
                position: -Math.abs(motorPosition),
                isabs: 0,
              })}
            >
              Move -
            </Button>
          </Box>
        </Box>

        <Divider sx={{ my: 1 }} />

        {/* LED array test */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
            LED Array (/ledarr_act)
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
            <TextField
              label="R=G=B"
              type="number"
              size="small"
              value={ledRgb}
              onChange={(e) => setLedRgb(Math.max(0, Math.min(255, parseInt(e.target.value, 10) || 0)))}
              sx={{ width: 100 }}
              inputProps={{ min: 0, max: 255 }}
            />
            <Button
              variant="outlined"
              size="small"
              disabled={testBusy}
              onClick={() => runDeviceTest("ledarray", {
                r: ledRgb, g: ledRgb, b: ledRgb, ledAction: "fill",
              })}
            >
              Fill LEDs
            </Button>
            <Button
              variant="outlined"
              size="small"
              disabled={testBusy}
              onClick={() => runDeviceTest("ledarray", {
                r: 0, g: 0, b: 0, ledAction: "fill",
              })}
            >
              All Off
            </Button>
          </Box>
        </Box>

        <Divider sx={{ my: 1 }} />

        {/* Laser test */}
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
            Laser (/laser_act)
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 1 }}>
            <TextField
              label="LASERval"
              type="number"
              size="small"
              value={laserVal}
              onChange={(e) => setLaserVal(parseInt(e.target.value, 10) || 0)}
              sx={{ width: 120 }}
            />
            {[0, 1, 2, 3, 4].map((lid) => (
              <Button
                key={lid}
                variant="outlined"
                size="small"
                disabled={testBusy}
                onClick={() => runDeviceTest("laser", { laserid: lid, laserval: laserVal })}
              >
                On L{lid}
              </Button>
            ))}
            {[0, 1, 2, 3, 4].map((lid) => (
              <Button
                key={`off${lid}`}
                variant="text"
                size="small"
                color="inherit"
                disabled={testBusy}
                onClick={() => runDeviceTest("laser", { laserid: lid, laserval: 0 })}
              >
                Off L{lid}
              </Button>
            ))}
          </Box>
        </Box>

        {testBusy && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1 }}>
            <CircularProgress size={16} />
            <Typography variant="caption">Sending command...</Typography>
          </Box>
        )}

        {testResult && (
          <Paper variant="outlined" sx={{ p: 1.5, mt: 2 }}>
            <Typography variant="caption" sx={{ fontWeight: 500 }}>
              Last {testResult.kind} test — status:{" "}
              <span style={{
                color: testResult.status === "success" ? "green"
                  : testResult.status === "warning" ? "orange" : "red",
              }}>
                {testResult.status}
              </span>
            </Typography>
            {testResult.command && (
              <Typography
                variant="body2"
                component="pre"
                sx={{
                  whiteSpace: "pre-wrap",
                  fontFamily: "monospace",
                  fontSize: "0.75rem",
                  mt: 0.5,
                  color: "text.secondary",
                }}
              >
                {">> "}{JSON.stringify(testResult.command)}
              </Typography>
            )}
            {testResult.response !== undefined && (
              <Typography
                variant="body2"
                component="pre"
                sx={{
                  whiteSpace: "pre-wrap",
                  fontFamily: "monospace",
                  fontSize: "0.75rem",
                  mt: 0.5,
                  overflowX: "auto",
                }}
              >
                {testResult.response || "(no response)"}
              </Typography>
            )}
            {testResult.message && (
              <Typography variant="caption" color="error" sx={{ display: "block", mt: 0.5 }}>
                {testResult.message}
              </Typography>
            )}
          </Paper>
        )}
      </Paper>

      {/* Start Over button to flash another device */}
      <Box sx={{ mt: 3 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleStartOver}
          startIcon={<RefreshIcon />}
          size="large"
        >
          Flash Another Device
        </Button>
      </Box>
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
      case "cancelled":
        return "Flashing was cancelled.";
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
