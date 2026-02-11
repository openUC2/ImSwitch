import React, { useEffect, useCallback } from "react";
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
  TextField,
  Box,
  Typography,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  Checkbox,
  LinearProgress,
  Paper,
  Grid,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  RadioGroup,
  Radio,
  FormControlLabel,
  FormControl,
  FormLabel,
} from "@mui/material";
import {
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Wifi as WifiIcon,
  Cable as CableIcon,
  AddCircleOutline as AddCircleOutlineIcon,
} from "@mui/icons-material";

// Redux slice
import * as canOtaSlice from "../state/slices/canOtaSlice";

// API functions
import apiUC2ConfigControllerGetOTAWiFiCredentials from "../backendapi/apiUC2ConfigControllerGetOTAWiFiCredentials";
import apiUC2ConfigControllerSetOTAWiFiCredentials from "../backendapi/apiUC2ConfigControllerSetOTAWiFiCredentials";
import apiUC2ConfigControllerGetOTAFirmwareServer from "../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer";
import apiUC2ConfigControllerSetOTAFirmwareServer from "../backendapi/apiUC2ConfigControllerSetOTAFirmwareServer";
import apiUC2ConfigControllerListAvailableFirmware from "../backendapi/apiUC2ConfigControllerListAvailableFirmware";
import apiUC2ConfigControllerScanCanbus from "../backendapi/apiUC2ConfigControllerScanCanbus";
import apiUC2ConfigControllerStartSingleDeviceOTA from "../backendapi/apiUC2ConfigControllerStartSingleDeviceOTA";
import apiUC2ConfigControllerStartMultipleDeviceOTA from "../backendapi/apiUC2ConfigControllerStartMultipleDeviceOTA";
import apiUC2ConfigControllerGetOTADeviceMapping from "../backendapi/apiUC2ConfigControllerGetOTADeviceMapping";
import apiUC2ConfigControllerStartCANStreamingOTA from "../backendapi/apiUC2ConfigControllerStartCANStreamingOTA";
import apiUC2ConfigControllerStartMultipleCANStreamingOTA from "../backendapi/apiUC2ConfigControllerStartMultipleCANStreamingOTA";

const steps = [
  "OTA Method",
  "WiFi Setup",
  "Firmware Server",
  "Scan Devices",
  "Select Devices",
  "Update Progress",
  "Complete",
];

const CanOtaWizard = ({ open, onClose }) => {
  const dispatch = useDispatch();
  const canOtaState = useSelector(canOtaSlice.getCanOtaState);

  // Device type mapping for display
  const [deviceMapping, setDeviceMapping] = React.useState({});

  // Manual CAN address entry
  const [manualCanId, setManualCanId] = React.useState("");

  // Load initial data when wizard opens
  useEffect(() => {
    if (open) {
      loadInitialData();
      loadDeviceMapping();
    }
  }, [open]);

  const loadInitialData = async () => {
    try {
      // Load WiFi credentials
      dispatch(canOtaSlice.setIsLoadingWifiCredentials(true));
      const wifiCreds = await apiUC2ConfigControllerGetOTAWiFiCredentials();
      dispatch(canOtaSlice.setDefaultWifiCredentials(wifiCreds));
      dispatch(canOtaSlice.setIsLoadingWifiCredentials(false));

      // Load firmware server
      dispatch(canOtaSlice.setIsLoadingFirmwareServer(true));
      const firmwareServer = await apiUC2ConfigControllerGetOTAFirmwareServer();
      dispatch(canOtaSlice.setDefaultFirmwareServerUrl(firmwareServer.firmware_server_url));
      dispatch(canOtaSlice.setIsLoadingFirmwareServer(false));
    } catch (error) {
      console.error("Error loading initial data:", error);
      dispatch(canOtaSlice.setError("Failed to load initial configuration"));
      dispatch(canOtaSlice.setIsLoadingWifiCredentials(false));
      dispatch(canOtaSlice.setIsLoadingFirmwareServer(false));
    }
  };

  const loadDeviceMapping = async () => {
    try {
      const mapping = await apiUC2ConfigControllerGetOTADeviceMapping();
      setDeviceMapping(mapping);
    } catch (error) {
      console.error("Error loading device mapping:", error);
    }
  };

  const handleReloadFirmware = async () => {
    if (!canOtaState.firmwareServerUrl) {
      dispatch(canOtaSlice.setError("Please provide a firmware server URL"));
      return;
    }
    try {
      dispatch(canOtaSlice.setIsLoadingFirmwareList(true));
      dispatch(canOtaSlice.clearMessages());

      // Save the URL first to ensure the server knows which URL to check
      await apiUC2ConfigControllerSetOTAFirmwareServer(canOtaState.firmwareServerUrl);

      // Load available firmware
      const firmwareList = await apiUC2ConfigControllerListAvailableFirmware();
      dispatch(canOtaSlice.setAvailableFirmware(firmwareList.firmware || {}));
      dispatch(canOtaSlice.setSuccessMessage("Firmware list reloaded"));
    } catch (error) {
      console.error("Error reloading firmware:", error);
      dispatch(canOtaSlice.setError("Failed to reload firmware list"));
    } finally {
      dispatch(canOtaSlice.setIsLoadingFirmwareList(false));
    }
  };

  const handleNext = async () => {
    const currentStep = canOtaState.currentStep;
    const isCAN = canOtaState.otaMethod === "can_streaming";

    // Step-specific validation and actions before proceeding
    if (currentStep === 0) {
      // OTA Method Selection - no validation needed, just proceed
      // For CAN streaming, we can skip WiFi setup (step 1)
      if (isCAN) {
        dispatch(canOtaSlice.clearMessages());
        dispatch(canOtaSlice.setCurrentStep(2)); // Skip to Firmware Server
        return;
      }
    } else if (currentStep === 1) {
      // WiFi Setup - save credentials (only for WiFi OTA)
      if (!canOtaState.wifiSsid || !canOtaState.wifiPassword) {
        dispatch(canOtaSlice.setError("Please provide both WiFi SSID and password"));
        return;
      }
      try {
        await apiUC2ConfigControllerSetOTAWiFiCredentials(
          canOtaState.wifiSsid,
          canOtaState.wifiPassword
        );
        dispatch(canOtaSlice.setSuccessMessage("WiFi credentials saved"));
      } catch (error) {
        dispatch(canOtaSlice.setError("Failed to save WiFi credentials"));
        return;
      }
    } else if (currentStep === 2) {
      // Firmware Server - save and validate
      if (!canOtaState.firmwareServerUrl) {
        dispatch(canOtaSlice.setError("Please provide a firmware server URL"));
        return;
      }
      try {
        dispatch(canOtaSlice.setIsLoadingFirmwareList(true));
        const result = await apiUC2ConfigControllerSetOTAFirmwareServer(
          canOtaState.firmwareServerUrl
        );
        if (result.status === "success") {
          dispatch(canOtaSlice.setSuccessMessage("Firmware server configured successfully"));
          // Load available firmware
          const firmwareList = await apiUC2ConfigControllerListAvailableFirmware();
          dispatch(canOtaSlice.setAvailableFirmware(firmwareList.firmware || {}));
        } else {
          dispatch(canOtaSlice.setError(result.message || "Failed to configure firmware server"));
          dispatch(canOtaSlice.setIsLoadingFirmwareList(false));
          return;
        }
        dispatch(canOtaSlice.setIsLoadingFirmwareList(false));
      } catch (error) {
        dispatch(canOtaSlice.setError("Failed to configure firmware server"));
        dispatch(canOtaSlice.setIsLoadingFirmwareList(false));
        return;
      }
    } else if (currentStep === 3) {
      // Moving from Scan Devices to Select Devices
      console.log("[CanOtaWizard] Step 3->4 transition. Current state:", {
        scannedDevices: canOtaState.scannedDevices,
        selectedDeviceIds: canOtaState.selectedDeviceIds,
      });
      // Auto-select any manual devices that aren't already selected
      const manualDeviceIds = canOtaState.scannedDevices
        .filter(d => d.manual && !canOtaState.selectedDeviceIds.includes(d.canId))
        .map(d => d.canId);
      if (manualDeviceIds.length > 0) {
        console.log("[CanOtaWizard] Auto-selecting manual devices:", manualDeviceIds);
        dispatch(canOtaSlice.setSelectedDeviceIds([
          ...canOtaState.selectedDeviceIds,
          ...manualDeviceIds,
        ]));
      }
    } else if (currentStep === 4) {
      // Device Selection - check if at least one device is selected
      console.log("[CanOtaWizard] Step 4->5 validation. State:", {
        scannedDevices: canOtaState.scannedDevices,
        selectedDeviceIds: canOtaState.selectedDeviceIds,
      });
      if (canOtaState.selectedDeviceIds.length === 0) {
        dispatch(canOtaSlice.setError("Please select at least one device to update"));
        return;
      }
      // Start the OTA update process
      startOtaUpdate();
    }

    dispatch(canOtaSlice.clearMessages());
    dispatch(canOtaSlice.nextStep());
  };

  const handleBack = () => {
    const currentStep = canOtaState.currentStep;
    const isCAN = canOtaState.otaMethod === "can_streaming";
    
    // For CAN streaming, skip WiFi setup when going back from Firmware Server
    if (isCAN && currentStep === 2) {
      dispatch(canOtaSlice.clearMessages());
      dispatch(canOtaSlice.setCurrentStep(0)); // Go back to method selection
      return;
    }
    
    dispatch(canOtaSlice.clearMessages());
    dispatch(canOtaSlice.previousStep());
  };

  const handleClose = () => {
    dispatch(canOtaSlice.resetWizard());
    onClose();
  };

  const handleScanDevices = async () => {
    try {
      dispatch(canOtaSlice.setIsScanningDevices(true));
      dispatch(canOtaSlice.setScanError(null));
      const devices = await apiUC2ConfigControllerScanCanbus(5);
      const scannedList = devices.scan || [];
      // Preserve manually added devices that are not in the scan results
      const manualDevices = canOtaState.scannedDevices.filter(
        (d) => d.manual && !scannedList.some((s) => s.canId === d.canId)
      );
      dispatch(canOtaSlice.setScannedDevices([...scannedList, ...manualDevices]));
      if (!devices || devices.scan.length === 0) {
        dispatch(canOtaSlice.setScanError("No CAN devices found"));
      }
      dispatch(canOtaSlice.setIsScanningDevices(false));
    } catch (error) {
      console.error("Error scanning CAN bus:", error);
      dispatch(canOtaSlice.setScanError("Failed to scan CAN bus: " + error.message));
      dispatch(canOtaSlice.setIsScanningDevices(false));
    }
  };

  const startOtaUpdate = async () => {
    const isCAN = canOtaState.otaMethod === "can_streaming";
    
    try {
      dispatch(canOtaSlice.setIsUpdating(true));
      dispatch(canOtaSlice.clearUpdateProgress());
      dispatch(canOtaSlice.setActiveUpdateCount(canOtaState.selectedDeviceIds.length));

      // Initialize all devices with "initiating" status
      canOtaState.selectedDeviceIds.forEach(canId => {
        dispatch(canOtaSlice.setUpdateProgress({
          canId: canId,
          status: "initiating",
          message: isCAN 
            ? "Starting CAN streaming OTA update..." 
            : "Starting WiFi OTA update...",
          progress: 0,
          timestamp: new Date().toISOString(),
        }));
      });

      let result;
      
      if (isCAN) {
        // CAN Streaming OTA - firmware is transferred over CAN bus
        console.log("Starting CAN Streaming OTA for devices:", canOtaState.selectedDeviceIds);
        result = await apiUC2ConfigControllerStartMultipleCANStreamingOTA(
          canOtaState.selectedDeviceIds,
          5 // 5 seconds delay between devices for reboot
        );
        console.log("CAN Streaming OTA initiated:", result);
        
        // Update status to "initiated" for all devices
        canOtaState.selectedDeviceIds.forEach(canId => {
          dispatch(canOtaSlice.setUpdateProgress({
            canId: canId,
            status: "initiated",
            message: "CAN streaming started, transferring firmware chunks...",
            progress: 5,
            timestamp: new Date().toISOString(),
          }));
        });
      } else {
        // WiFi OTA - devices download firmware from server
        result = await apiUC2ConfigControllerStartMultipleDeviceOTA(
          canOtaState.selectedDeviceIds,
          canOtaState.wifiSsid,
          canOtaState.wifiPassword,
          300000, // 5 minutes timeout
          2 // 2 seconds delay between devices
        );
        console.log("WiFi OTA update initiated:", result);
        
        // Update status to "initiated" for all devices
        canOtaState.selectedDeviceIds.forEach(canId => {
          dispatch(canOtaSlice.setUpdateProgress({
            canId: canId,
            status: "initiated",
            message: "OTA command sent, waiting for device response...",
            progress: 5,
            timestamp: new Date().toISOString(),
          }));
        });
      }
      
      // Note: Further progress updates will come via WebSocket (sigOTAStatusUpdate)
      // The backend uploads firmware in the background and sends status via socket
      
    } catch (error) {
      console.error("Error starting OTA update:", error);
      
      // Even if API call fails/times out, the backend might still be processing
      // So we show a warning instead of complete failure
      const methodName = isCAN ? "CAN streaming" : "WiFi";
      dispatch(canOtaSlice.setError(
        `${methodName} OTA update may be starting in background. Check progress below. ` +
        "If no progress appears within 30 seconds, the update may have failed. Error: " + 
        error.message
      ));
      
      // Don't set isUpdating to false - let WebSocket updates control the state
      // If truly failed, user can manually stop or restart
    }
  };

  const getDeviceTypeString = (canId) => {
    for (const [key, value] of Object.entries(deviceMapping)) {
      if (value === canId) {
        return key.toUpperCase();
      }
    }
    return `Device ${canId}`;
  };

  const getDeviceStatusColor = (status) => {
    switch (status) {
      case "completed":
      case "success":
        return "success";
      case "failed":
      case "error":
        return "error";
      case "initiated":
      case "in_progress":
        return "primary";
      default:
        return "default";
    }
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return renderOtaMethodSelection();
      case 1:
        return renderWifiSetup();
      case 2:
        return renderFirmwareServer();
      case 3:
        return renderDeviceScan();
      case 4:
        return renderDeviceSelection();
      case 5:
        return renderUpdateProgress();
      case 6:
        return renderCompletion();
      default:
        return null;
    }
  };

  const renderOtaMethodSelection = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Select OTA Update Method
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Choose how firmware will be transferred to the CAN devices.
      </Typography>

      <FormControl component="fieldset" sx={{ mt: 3, width: "100%" }}>
        <RadioGroup
          value={canOtaState.otaMethod}
          onChange={(e) => dispatch(canOtaSlice.setOtaMethod(e.target.value))}
        >
          <Paper 
            sx={{ 
              p: 2, 
              mb: 2, 
              border: canOtaState.otaMethod === "wifi" ? 2 : 1,
              borderColor: canOtaState.otaMethod === "wifi" ? "primary.main" : "divider",
              cursor: "pointer"
            }}
            onClick={() => dispatch(canOtaSlice.setOtaMethod("wifi"))}
          >
            <FormControlLabel
              value="wifi"
              control={<Radio />}
              label={
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <WifiIcon color={canOtaState.otaMethod === "wifi" ? "primary" : "action"} />
                  <Typography variant="subtitle1">WiFi OTA</Typography>
                </Box>
              }
            />
            <Typography variant="body2" color="text.secondary" sx={{ ml: 4 }}>
              Devices download firmware directly from a server via WiFi.
              Requires WiFi credentials and a firmware server URL.
              Best for devices with stable WiFi connection.
            </Typography>
          </Paper>

          <Paper 
            sx={{ 
              p: 2, 
              border: canOtaState.otaMethod === "can_streaming" ? 2 : 1,
              borderColor: canOtaState.otaMethod === "can_streaming" ? "primary.main" : "divider",
              cursor: "pointer"
            }}
            onClick={() => dispatch(canOtaSlice.setOtaMethod("can_streaming"))}
          >
            <FormControlLabel
              value="can_streaming"
              control={<Radio />}
              label={
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <CableIcon color={canOtaState.otaMethod === "can_streaming" ? "primary" : "action"} />
                  <Typography variant="subtitle1">CAN Streaming OTA</Typography>
                </Box>
              }
            />
            <Typography variant="body2" color="text.secondary" sx={{ ml: 4 }}>
              Firmware is streamed directly over the CAN bus from this computer.
              No WiFi required on target devices. Slower but more reliable for
              devices without WiFi access.
            </Typography>
          </Paper>
        </RadioGroup>
      </FormControl>

      <Alert severity="info" sx={{ mt: 3 }}>
        {canOtaState.otaMethod === "wifi" 
          ? "WiFi OTA: Devices will connect to WiFi and download firmware from the server."
          : "CAN Streaming: Firmware will be transferred chunk by chunk over the CAN bus."}
      </Alert>
    </Box>
  );

  const renderWifiSetup = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        WiFi Configuration for OTA Updates
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        These credentials will be used by CAN devices to connect to WiFi during the OTA update process.
      </Typography>

      {canOtaState.isLoadingWifiCredentials ? (
        <Box sx={{ display: "flex", justifyContent: "center", my: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ mt: 3 }}>
          <TextField
            fullWidth
            label="WiFi SSID"
            value={canOtaState.wifiSsid}
            onChange={(e) => dispatch(canOtaSlice.setWifiSsid(e.target.value))}
            margin="normal"
            helperText={`Default: ${canOtaState.defaultWifiSsid || "Not set"}`}
          />
          <TextField
            fullWidth
            label="WiFi Password"
            type="password"
            value={canOtaState.wifiPassword}
            onChange={(e) => dispatch(canOtaSlice.setWifiPassword(e.target.value))}
            margin="normal"
            helperText="Password for the WiFi network"
          />
        </Box>
      )}
    </Box>
  );

  const renderFirmwareServer = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Firmware Server Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        The server should serve firmware files at &lt;server_url&gt;/latest/ with files named like: id_10_*.bin, id_11_*.bin, etc.
      </Typography>

      {canOtaState.isLoadingFirmwareServer ? (
        <Box sx={{ display: "flex", justifyContent: "center", my: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ mt: 3 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <TextField
              fullWidth
              label="Firmware Server URL"
              value={canOtaState.firmwareServerUrl}
              onChange={(e) => dispatch(canOtaSlice.setFirmwareServerUrl(e.target.value))}
              margin="normal"
              placeholder="http://localhost/firmware"
              helperText={`Default: ${canOtaState.defaultFirmwareServerUrl || "http://localhost/firmware"}`}
            />
            <Tooltip title="Reload firmware files from server">
              <IconButton
                onClick={handleReloadFirmware}
                disabled={canOtaState.isLoadingFirmwareList}
                sx={{ mt: 1 }}
              >
                {canOtaState.isLoadingFirmwareList ? (
                  <CircularProgress size={24} />
                ) : (
                  <RefreshIcon />
                )}
              </IconButton>
            </Tooltip>
          </Box>

          {canOtaState.isLoadingFirmwareList && (
            <Box sx={{ mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2" sx={{ ml: 2, display: "inline" }}>
                Checking firmware availability...
              </Typography>
            </Box>
          )}

          {Object.keys(canOtaState.availableFirmware).length > 0 && (
            <Paper sx={{ mt: 2, p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Available Firmware Files:
              </Typography>
              <List dense>
                {Object.entries(canOtaState.availableFirmware).map(([canId, firmware]) => (
                  <ListItem key={canId}>
                    <ListItemText
                      primary={`CAN ID ${canId}: ${firmware.filename}`}
                      secondary={`${(firmware.size / 1024).toFixed(2)} KB`}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          )}
        </Box>
      )}
    </Box>
  );

  const renderDeviceScan = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Scan for CAN Devices
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Click the button below to scan the CAN bus for connected devices.
      </Typography>

      <Box sx={{ mt: 3, mb: 2 }}>
        <Button
          variant="contained"
          onClick={handleScanDevices}
          //disabled={canOtaState.isScanningDevices}
          startIcon={canOtaState.isScanningDevices ? <CircularProgress size={20} /> : <RefreshIcon />}
        >
          {canOtaState.isScanningDevices ? "Scanning..." : "Scan CAN Bus"}
        </Button>
      </Box>

      {canOtaState.scanError && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          {canOtaState.scanError}
        </Alert>
      )}

      {/* Manual CAN address entry */}
      <Paper sx={{ mt: 3, p: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Manual CAN Address Entry
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          If a device was not found by the scan, you can manually add it by entering its CAN ID below.
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1 }}>
          <TextField
            label="CAN ID"
            type="number"
            size="small"
            value={manualCanId}
            onChange={(e) => setManualCanId(e.target.value)}
            placeholder="e.g. 11"
            sx={{ width: 140 }}
          />
          <Button
            variant="outlined"
            size="small"
            startIcon={<AddCircleOutlineIcon />}
            disabled={!manualCanId}
            onClick={() => {
              const id = parseInt(manualCanId, 10);
              if (isNaN(id) || id < 1 || id > 255) {
                dispatch(canOtaSlice.setScanError("CAN ID must be between 1 and 255"));
                return;
              }
              // Check if already in list
              const exists = canOtaState.scannedDevices.some((d) => d.canId === id);
              if (exists) {
                dispatch(canOtaSlice.setScanError(`Device with CAN ID ${id} is already in the list`));
                return;
              }
              const newDevice = { canId: id, deviceTypeStr: "Manual", deviceType: -1, statusStr: "Unknown", manual: true };
              console.log("[CanOtaWizard] Adding manual device:", newDevice);
              console.log("[CanOtaWizard] Current scannedDevices before add:", canOtaState.scannedDevices);
              dispatch(canOtaSlice.setScannedDevices([
                ...canOtaState.scannedDevices,
                newDevice,
              ]));
              // Auto-select the manually added device
              dispatch(canOtaSlice.setSelectedDeviceIds([
                ...canOtaState.selectedDeviceIds,
                id,
              ]));
              console.log("[CanOtaWizard] Manual device added, new selectedDeviceIds:", [...canOtaState.selectedDeviceIds, id]);
              setManualCanId("");
              dispatch(canOtaSlice.setScanError(null));
            }}
          >
            Add Device
          </Button>
        </Box>
        <Alert severity="warning" sx={{ mt: 2 }} icon={<WarningIcon />}>
          <strong>Use at your own risk:</strong> Manually adding a CAN address bypasses
          device detection. Make sure the CAN ID is correct – sending firmware to
          the wrong device may cause it to become unresponsive.
        </Alert>
      </Paper>

      {canOtaState.scannedDevices.length > 0 && (
        <Paper sx={{ mt: 2 }}>
          <List>
            {canOtaState.scannedDevices.map((device, index) => (
              <React.Fragment key={device.canId}>
                {index > 0 && <Divider />}
                <ListItem>
                  <ListItemText
                    primary={
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <Typography variant="subtitle1">
                          {getDeviceTypeString(device.canId)} (CAN ID: {device.canId})
                        </Typography>
                        <Chip
                          label={device.deviceTypeStr || "Unknown"}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      </Box>
                    }
                    secondary={
                      <Typography variant="body2" color="text.secondary">
                        Status: {device.statusStr || "Unknown"} | Type Code: {device.deviceType}
                      </Typography>
                    }
                  />
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        </Paper>
      )}
    </Box>
  );

  const renderDeviceSelection = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Select Devices to Update
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Choose which devices you want to update with new firmware.
      </Typography>

      <Box sx={{ mt: 2, mb: 2 }}>
        <Button
          variant="outlined"
          size="small"
          onClick={() => dispatch(canOtaSlice.selectAllDevices())}
          sx={{ mr: 1 }}
        >
          Select All
        </Button>
        <Button
          variant="outlined"
          size="small"
          onClick={() => dispatch(canOtaSlice.deselectAllDevices())}
        >
          Deselect All
        </Button>
      </Box>

      {canOtaState.scannedDevices.length > 0 ? (
        <Paper sx={{ mt: 2 }}>
          <List>
            {canOtaState.scannedDevices.map((device, index) => {
              const isSelected = canOtaState.selectedDeviceIds.includes(device.canId);
              const hasFirmware = canOtaState.availableFirmware[device.canId];

              return (
                <React.Fragment key={device.canId}>
                  {index > 0 && <Divider />}
                  <ListItem
                    button
                    onClick={() => dispatch(canOtaSlice.toggleDeviceSelection(device.canId))}
                  >
                    <Checkbox checked={isSelected} />
                    <ListItemText
                      primary={
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <Typography variant="subtitle1">
                            {getDeviceTypeString(device.canId)} (CAN ID: {device.canId})
                          </Typography>
                          {hasFirmware ? (
                            <Chip label="Firmware Available" size="small" color="success" />
                          ) : (
                            <Chip label="No Firmware" size="small" color="warning" />
                          )}
                        </Box>
                      }
                      secondary={
                        hasFirmware ? (
                          <Typography variant="body2" color="text.secondary">
                            {canOtaState.availableFirmware[device.canId].filename}
                          </Typography>
                        ) : (
                          <Typography variant="body2" color="warning.main">
                            No matching firmware file found on server
                          </Typography>
                        )
                      }
                    />
                  </ListItem>
                </React.Fragment>
              );
            })}
          </List>
        </Paper>
      ) : canOtaState.selectedDeviceIds.length > 0 ? (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Manually entered CAN IDs: {canOtaState.selectedDeviceIds.join(", ")}
          <br />
          <Typography variant="body2" sx={{ mt: 1 }}>
            Note: Device information not available. Click NEXT to proceed with update.
          </Typography>
        </Alert>
      ) : (
        <Alert severity="info" sx={{ mt: 2 }}>
          No devices found. Please go back and scan for devices first.
        </Alert>
      )}

      {canOtaState.selectedDeviceIds.length > 0 && (
        <>
          <Alert severity="info" sx={{ mt: 2 }}>
            {canOtaState.selectedDeviceIds.length} device(s) selected for update
          </Alert>
          <Alert severity="warning" sx={{ mt: 1 }} icon={<WarningIcon />}>
            <strong>Important:</strong> Do not turn off the microscope or disconnect
            USB during the update. The process may take several minutes per device.
          </Alert>
        </>
      )}
    </Box>
  );

  const renderUpdateProgress = () => {
    const totalDevices = canOtaState.selectedDeviceIds.length;
    const completedDevices = canOtaState.completedUpdateCount;
    const failedDevices = canOtaState.failedUpdateCount;

    // For CAN streaming, compute overall progress from per-device progress values
    let overallProgress = 0;
    if (totalDevices > 0) {
      const progressSum = canOtaState.selectedDeviceIds.reduce((sum, canId) => {
        const p = canOtaState.updateProgress[canId];
        return sum + (p ? p.progress || 0 : 0);
      }, 0);
      overallProgress = progressSum / totalDevices;
    }

    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Update Progress
        </Typography>

        {/* Safety warnings */}
        <Alert severity="error" sx={{ mb: 2 }} icon={<WarningIcon />}>
          <strong>Do NOT turn off the microscope or disconnect USB</strong> while
          the firmware update is in progress. Interrupting the update can leave
          devices in an unrecoverable state.
        </Alert>
        <Alert severity="info" sx={{ mb: 2 }}>
          CAN streaming updates transfer firmware at ~2-4 KB/s.
          A typical update takes <strong>2–5 minutes per device</strong>.
          Please be patient – the progress bar below shows real-time page-by-page upload status.
        </Alert>

        <Paper sx={{ p: 2, mt: 2, mb: 2 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12}>
              <LinearProgress
                variant="determinate"
                value={overallProgress}
                sx={{ height: 10, borderRadius: 5 }}
              />
              <Typography variant="caption" align="center" display="block" sx={{ mt: 0.5 }}>
                Overall: {Math.round(overallProgress)}%
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2" align="center">
                Total: {totalDevices}
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2" align="center" color="success.main">
                Completed: {completedDevices}
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2" align="center" color="error.main">
                Failed: {failedDevices}
              </Typography>
            </Grid>
          </Grid>
        </Paper>

        <Paper sx={{ mt: 2 }}>
          <List>
            {canOtaState.selectedDeviceIds.map((canId, index) => {
              const progress = canOtaState.updateProgress[canId];
              const statusColor = progress ? getDeviceStatusColor(progress.status) : "default";
              const deviceTypeName = deviceMapping[canId] || getDeviceTypeString(canId);

              return (
                <React.Fragment key={canId}>
                  {index > 0 && <Divider />}
                  <ListItem>
                    <ListItemText
                      primary={
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <Typography variant="subtitle1">
                            Device {canId} (CAN ID: {canId})
                          </Typography>
                          {deviceTypeName && deviceTypeName !== `Device ${canId}` && (
                            <Chip
                              label={deviceTypeName}
                              size="small"
                              color="info"
                              variant="outlined"
                            />
                          )}
                          {progress && (
                            <Chip
                              label={progress.status}
                              size="small"
                              color={statusColor}
                              icon={
                                progress.status === "completed" ? (
                                  <CheckCircleIcon />
                                ) : progress.status === "failed" || progress.status === "error" ? (
                                  <ErrorIcon />
                                ) : (
                                  <CircularProgress size={16} />
                                )
                              }
                            />
                          )}
                        </Box>
                      }
                      secondary={
                        progress ? (
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              {progress.message}
                            </Typography>
                            {progress.progress > 0 && (
                              <LinearProgress
                                variant="determinate"
                                value={progress.progress}
                                sx={{ mt: 1 }}
                              />
                            )}
                          </Box>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            Waiting to start...
                          </Typography>
                        )
                      }
                    />
                  </ListItem>
                </React.Fragment>
              );
            })}
          </List>
        </Paper>

        {canOtaState.isUpdating && (
          <Box sx={{ display: "flex", alignItems: "center", mt: 2 }}>
            <CircularProgress size={24} />
            <Typography variant="body2" sx={{ ml: 2 }}>
              Update in progress... Please wait.
            </Typography>
          </Box>
        )}

        {canOtaState.isUpdating && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>Warning:</strong> Canceling during an active update may leave devices in an unstable state.
              Only cancel if absolutely necessary.
            </Typography>
          </Alert>
        )}
      </Box>
    );
  };

  const renderCompletion = () => {
    const totalDevices = canOtaState.selectedDeviceIds.length;
    const completedDevices = canOtaState.completedUpdateCount;
    const failedDevices = canOtaState.failedUpdateCount;
    const allSuccessful = completedDevices === totalDevices && failedDevices === 0;

    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Update Complete
        </Typography>

        {allSuccessful ? (
          <Alert severity="success" sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              All devices updated successfully! 🎉
            </Typography>
            <Typography variant="body2">
              {completedDevices} device(s) were updated without errors.
            </Typography>
          </Alert>
        ) : (
          <Alert severity={failedDevices > 0 ? "error" : "warning"} sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Update completed with some issues
            </Typography>
            <Typography variant="body2">
              Completed: {completedDevices}/{totalDevices} | Failed: {failedDevices}
            </Typography>
          </Alert>
        )}

        <Paper sx={{ mt: 3, p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Summary:
          </Typography>
          <List dense>
            {canOtaState.selectedDeviceIds.map((canId) => {
              const progress = canOtaState.updateProgress[canId];
              const isSuccess = progress && progress.status === "completed";

              return (
                <ListItem key={canId}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        {isSuccess ? (
                          <CheckCircleIcon color="success" fontSize="small" />
                        ) : (
                          <ErrorIcon color="error" fontSize="small" />
                        )}
                        <Typography variant="body2">
                          {getDeviceTypeString(canId)} (CAN ID: {canId})
                        </Typography>
                      </Box>
                    }
                    secondary={progress ? progress.message : "No status available"}
                  />
                </ListItem>
              );
            })}
          </List>
        </Paper>

        <Box sx={{ mt: 3, display: "flex", gap: 2 }}>
          <Button
            variant="outlined"
            onClick={() => {
              dispatch(canOtaSlice.resetWizard());
              dispatch(canOtaSlice.setCurrentStep(0));
            }}
          >
            Update More Devices
          </Button>
          <Button variant="contained" onClick={handleClose}>
            Finish
          </Button>
        </Box>
      </Box>
    );
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>CAN OTA Update Wizard</DialogTitle>
      <DialogContent>
        <Stepper activeStep={canOtaState.currentStep} sx={{ mt: 2, mb: 3 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {canOtaState.error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => dispatch(canOtaSlice.clearMessages())}>
            {canOtaState.error}
          </Alert>
        )}

        {canOtaState.successMessage && (
          <Alert severity="success" sx={{ mb: 2 }} onClose={() => dispatch(canOtaSlice.clearMessages())}>
            {canOtaState.successMessage}
          </Alert>
        )}

        {renderStepContent(canOtaState.currentStep)}
      </DialogContent>
      <DialogActions>
        {canOtaState.currentStep !== 6 && (
          <>
            <Button 
              onClick={handleClose}
              color={canOtaState.currentStep === 5 && canOtaState.isUpdating ? "error" : "inherit"}
            >
              {canOtaState.currentStep === 5 && canOtaState.isUpdating ? "Cancel Update" : "Cancel"}
            </Button>
            <Box sx={{ flex: "1 1 auto" }} />
            {canOtaState.currentStep !== 5 && (
              <Button
                onClick={handleBack}
              >
                Back
              </Button>
            )}
            {canOtaState.currentStep !== 5 && (
              <Button
                variant="contained"
                onClick={handleNext}
              >
                {canOtaState.currentStep === steps.length - 2 ? "Start Update" : "Next"}
              </Button>
            )}
          </>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default CanOtaWizard;
