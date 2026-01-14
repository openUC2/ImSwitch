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
  ListItemIcon,
  LinearProgress,
  Paper,
  FormControl,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  InputLabel,
  TextField,
  Chip,
  Radio,
  RadioGroup,
  Divider,
  IconButton,
  Tooltip,
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
} from "@mui/icons-material";

// Redux slice
import * as usbFlashSlice from "../state/slices/usbFlashSlice";

// API functions
import apiUC2ConfigControllerListSerialPorts from "../backendapi/apiUC2ConfigControllerListSerialPorts";
import apiUC2ConfigControllerFlashMasterFirmwareUSB from "../backendapi/apiUC2ConfigControllerFlashMasterFirmwareUSB";
import apiUC2ConfigControllerGetOTAFirmwareServer from "../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer";
import apiUC2ConfigControllerListAvailableFirmware from "../backendapi/apiUC2ConfigControllerListAvailableFirmware";

const steps = [
  "Port Selection",
  "Firmware Info",
  "Flash Device",
  "Complete",
];

/**
 * USB Flash Wizard Component
 * Wizard for flashing the master CAN HAT firmware via USB/esptool
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
      // Load serial ports
      await loadSerialPorts();

      // Load firmware server URL
      const firmwareServer = await apiUC2ConfigControllerGetOTAFirmwareServer();
      dispatch(usbFlashSlice.setFirmwareServerUrl(firmwareServer.firmware_server_url));

      // Load available firmware to find master firmware
      await loadFirmwareInfo();
    } catch (error) {
      console.error("Error loading initial data:", error);
      dispatch(usbFlashSlice.setError("Failed to load initial configuration"));
    }
  };

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

  const loadFirmwareInfo = async () => {
    try {
      dispatch(usbFlashSlice.setIsLoadingFirmware(true));
      const firmwareList = await apiUC2ConfigControllerListAvailableFirmware();
      
      // Find master firmware (CAN ID 1)
      const masterFirmware = firmwareList.firmware?.["1"] || null;
      dispatch(usbFlashSlice.setMasterFirmwareInfo(masterFirmware));
      
      if (!masterFirmware) {
        dispatch(usbFlashSlice.setError("Master firmware not found on server"));
      }
    } catch (error) {
      console.error("Error loading firmware info:", error);
      dispatch(usbFlashSlice.setError("Failed to load firmware information"));
    } finally {
      dispatch(usbFlashSlice.setIsLoadingFirmware(false));
    }
  };

  const handleNext = async () => {
    const currentStep = usbFlashState.currentStep;

    // Step-specific validation and actions
    if (currentStep === 0) {
      // Port selection - no specific validation needed (auto-detect is allowed)
      dispatch(usbFlashSlice.clearMessages());
    } else if (currentStep === 1) {
      // Firmware info - check if firmware is available
      if (!usbFlashState.masterFirmwareInfo) {
        dispatch(usbFlashSlice.setError("Master firmware not available. Please check firmware server."));
        return;
      }
      dispatch(usbFlashSlice.clearMessages());
    } else if (currentStep === 2) {
      // Start flashing
      await startFlashing();
      return; // Don't advance step automatically, wait for flash completion
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

  const startFlashing = async () => {
    try {
      dispatch(usbFlashSlice.setIsFlashing(true));
      dispatch(usbFlashSlice.setFlashStatus("disconnecting"));
      dispatch(usbFlashSlice.setFlashProgress(0));
      dispatch(usbFlashSlice.setFlashMessage("Disconnecting from ESP32..."));
      dispatch(usbFlashSlice.clearMessages());

      // Simulate progress updates (actual progress comes from backend via socket)
      setTimeout(() => {
        dispatch(usbFlashSlice.setFlashStatus("downloading"));
        dispatch(usbFlashSlice.setFlashProgress(10));
        dispatch(usbFlashSlice.setFlashMessage("Downloading firmware from server..."));
      }, 1000);

      setTimeout(() => {
        dispatch(usbFlashSlice.setFlashStatus("flashing"));
        dispatch(usbFlashSlice.setFlashProgress(20));
        dispatch(usbFlashSlice.setFlashMessage("Flashing firmware via USB..."));
      }, 3000);

      // Call the flash API
      const result = await apiUC2ConfigControllerFlashMasterFirmwareUSB(
        usbFlashState.selectedPort,
        usbFlashState.portMatch,
        usbFlashState.baudRate,
        usbFlashState.flashOffset,
        usbFlashState.eraseFlash,
        usbFlashState.reconnectAfter
      );

      dispatch(usbFlashSlice.setFlashResult(result));

      if (result.status === "success") {
        dispatch(usbFlashSlice.setFlashStatus("success"));
        dispatch(usbFlashSlice.setFlashProgress(100));
        dispatch(usbFlashSlice.setFlashMessage("✅ Firmware flashed successfully!"));
        dispatch(usbFlashSlice.setSuccessMessage("Master firmware has been updated successfully"));
        dispatch(usbFlashSlice.nextStep()); // Move to completion step
      } else if (result.status === "warning") {
        dispatch(usbFlashSlice.setFlashStatus("success"));
        dispatch(usbFlashSlice.setFlashProgress(100));
        dispatch(usbFlashSlice.setFlashMessage("⚠️ Firmware flashed, but reconnection failed"));
        dispatch(usbFlashSlice.setError(result.message || "Reconnection failed"));
        dispatch(usbFlashSlice.nextStep());
      } else {
        dispatch(usbFlashSlice.setFlashStatus("failed"));
        dispatch(usbFlashSlice.setFlashMessage("❌ Flashing failed"));
        dispatch(usbFlashSlice.setError(result.message || "Unknown error"));
        dispatch(usbFlashSlice.setFlashDetails(result.details));
      }
    } catch (error) {
      console.error("Error during flashing:", error);
      dispatch(usbFlashSlice.setFlashStatus("failed"));
      dispatch(usbFlashSlice.setFlashMessage("❌ Flashing failed"));
      dispatch(usbFlashSlice.setError(`Failed to flash firmware: ${error.message}`));
    } finally {
      dispatch(usbFlashSlice.setIsFlashing(false));
    }
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return renderPortSelection();
      case 1:
        return renderFirmwareInfo();
      case 2:
        return renderFlashProgress();
      case 3:
        return renderCompletion();
      default:
        return null;
    }
  };

  const renderPortSelection = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        <UsbIcon sx={{ mr: 1, verticalAlign: "middle" }} />
        USB Port Selection
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Select the serial port connected to the master CAN HAT, or use auto-detection.
      </Typography>

      {usbFlashState.isLoadingPorts ? (
        <Box sx={{ display: "flex", justifyContent: "center", my: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ mt: 3 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
            <Typography variant="subtitle2">Port Selection Mode:</Typography>
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
                dispatch(usbFlashSlice.setSelectedPort(value));
              }}
            >
              <FormControlLabel
                value="auto"
                control={<Radio />}
                label={
                  <Box>
                    <Typography variant="body1">Auto-detect</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Automatically find the CAN HAT by matching "{usbFlashState.portMatch}" in port metadata
                    </Typography>
                  </Box>
                }
              />
              <Divider sx={{ my: 1 }} />
              {usbFlashState.availablePorts.length > 0 ? (
                usbFlashState.availablePorts.map((port) => (
                  <FormControlLabel
                    key={port.device}
                    value={port.device}
                    control={<Radio />}
                    label={
                      <Box>
                        <Typography variant="body1">{port.device}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {port.description || port.manufacturer || port.product || "Unknown device"}
                          {port.vid && port.pid && ` (VID:${port.vid.toString(16)} PID:${port.pid.toString(16)})`}
                        </Typography>
                      </Box>
                    }
                  />
                ))
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
        </Box>
      )}
    </Box>
  );

  const renderFirmwareInfo = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        <MemoryIcon sx={{ mr: 1, verticalAlign: "middle" }} />
        Firmware Information
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Review firmware and flash options before proceeding.
      </Typography>

      {usbFlashState.isLoadingFirmware ? (
        <Box sx={{ display: "flex", justifyContent: "center", my: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ mt: 3 }}>
          {/* Firmware Info */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              <DownloadIcon sx={{ mr: 1, verticalAlign: "middle", fontSize: 20 }} />
              Master CAN HAT Firmware
            </Typography>
            {usbFlashState.masterFirmwareInfo ? (
              <List dense>
                <ListItem>
                  <ListItemIcon><MemoryIcon fontSize="small" /></ListItemIcon>
                  <ListItemText
                    primary="Filename"
                    secondary={usbFlashState.masterFirmwareInfo.filename}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><DownloadIcon fontSize="small" /></ListItemIcon>
                  <ListItemText
                    primary="Size"
                    secondary={`${(usbFlashState.masterFirmwareInfo.size / 1024).toFixed(2)} KB`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Server URL"
                    secondary={usbFlashState.firmwareServerUrl}
                  />
                </ListItem>
              </List>
            ) : (
              <Alert severity="error">
                Master firmware not found on server. Please configure the firmware server.
              </Alert>
            )}
            <Button
              size="small"
              startIcon={<RefreshIcon />}
              onClick={loadFirmwareInfo}
              sx={{ mt: 1 }}
            >
              Refresh
            </Button>
          </Paper>

          {/* Flash Options */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              <SettingsIcon sx={{ mr: 1, verticalAlign: "middle", fontSize: 20 }} />
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
              <InputLabel>Flash Offset</InputLabel>
              <Select
                value={usbFlashState.flashOffset}
                onChange={(e) => dispatch(usbFlashSlice.setFlashOffset(e.target.value))}
                label="Flash Offset"
              >
                <MenuItem value={0x10000}>0x10000 (app-only image)</MenuItem>
                {/*<MenuItem value={0x0}>0x0 (merged/factory image)</MenuItem>*/}
              </Select>
            </FormControl>

            <FormControlLabel
              control={
                <Checkbox
                  checked={usbFlashState.eraseFlash}
                  onChange={(e) => dispatch(usbFlashSlice.setEraseFlash(e.target.checked))}
                />
              }
              label="Erase flash before writing (clears all settings)"
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={usbFlashState.reconnectAfter}
                  onChange={(e) => dispatch(usbFlashSlice.setReconnectAfter(e.target.checked))}
                />
              }
              label="Reconnect to device after flashing"
            />
          </Paper>
        </Box>
      )}
    </Box>
  );

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
            <Typography variant="body2" component="pre" sx={{ whiteSpace: "pre-wrap", fontFamily: "monospace" }}>
              {usbFlashState.flashDetails}
            </Typography>
          </Alert>
        )}

        {usbFlashState.flashStatus === "failed" && (
          <Box sx={{ mt: 2, textAlign: "center" }}>
            <Button
              variant="outlined"
              color="primary"
              onClick={startFlashing}
            >
              Retry
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

  const renderCompletion = () => (
    <Box sx={{ mt: 2, textAlign: "center" }}>
      {usbFlashState.flashResult?.status === "success" ? (
        <>
          <CheckCircleIcon color="success" sx={{ fontSize: 80, mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Firmware Update Complete!
          </Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            The master CAN HAT firmware has been successfully updated.
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
        return "Writing firmware to the device via USB...";
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
    if (step === 1 && !usbFlashState.masterFirmwareInfo) return true;
    if (step === 2 && usbFlashState.isFlashing) return true;
    return false;
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
          {usbFlashState.currentStep === 3 ? "Close" : "Cancel"}
        </Button>
        {usbFlashState.currentStep > 0 && usbFlashState.currentStep < 3 && (
          <Button onClick={handleBack} disabled={usbFlashState.isFlashing}>
            Back
          </Button>
        )}
        {usbFlashState.currentStep < 3 && (
          <Button
            variant="contained"
            onClick={handleNext}
            disabled={isNextDisabled() || usbFlashState.isFlashing}
          >
            {usbFlashState.currentStep === 2 ? "Start Flashing" : "Next"}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default UsbFlashWizard;
