import { useState, useEffect } from "react";
import { setNotification } from "../state/slices/NotificationSlice.js";
import { useDispatch, useSelector } from "react-redux";
import {
  setIp,
  setWebsocketPort,
  setApiPort,
} from "../state/slices/ConnectionSettingsSlice.js";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import * as webSocketSlice from "../state/slices/WebSocketSlice.js";
import {
  Box,
  TextField,
  Button,
  MenuItem,
  Typography,
  Paper,
  Card,
  CardContent,
  CardActions,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Chip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import {
  Cable,
  Wifi,
  Computer,
  CheckCircle,
  Warning,
  Info,
  Save,
  Settings,
  ExpandMore,
  Tune,
} from "@mui/icons-material";

/**
 * ImSwitch Connection Settings Component
 */
function ConnectionSettings() {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(
    (state) => state.connectionSettingsState
  );

  // Get connection status from Redux state
  const uc2State = useSelector(uc2Slice.getUc2State);
  const isBackendConnected = uc2State.backendConnected; // Backend API reachable
  const isHardwareConnected = uc2State.uc2Connected; // UC2 hardware connected

  // Get WebSocket connection status from Redux state
  const webSocketState = useSelector(webSocketSlice.getWebSocketState);
  const websocketTestStatus = webSocketState.testStatus;
  const isWebSocketConnected = webSocketState.connected;

  // Smart fallbacks for empty values
  const getSmartDefaults = () => {
    const location = window.location;
    const isHttps = location.protocol === "https:";
    return {
      protocol: isHttps ? "https://" : "http://",
      hostname: location.hostname,
      // Use current port if available. If empty on HTTPS, it implies 443.
      // Default to 80 only on non-standard HTTP setups.
      port: location.port || (isHttps ? "443" : "80"),
    };
  };

  const smartDefaults = getSmartDefaults();

  // Local state, initialized from Redux with smart fallbacks
  const [hostProtocol, setHostProtocol] = useState(
    connectionSettings.ip?.startsWith("http://")
      ? "http://"
      : connectionSettings.ip?.startsWith("https://")
      ? "https://"
      : smartDefaults.protocol
  );
  const [hostIP, setHostIP] = useState(
    connectionSettings.ip?.replace(/^https?:\/\//, "") || smartDefaults.hostname
  );
  const [websocketPort, setWebsocketPortState] = useState(
    connectionSettings.websocketPort || smartDefaults.port
  );
  const [apiPort, setApiPortState] = useState(
    connectionSettings.apiPort || smartDefaults.port
  );

  // Connection test state
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Check if connection settings are configured
  const hasConnectionSettings = hostIP && apiPort;

  // Auto-test connection on component mount if settings are already configured
  const [hasAutoTested, setHasAutoTested] = useState(false);

  // Pause periodic connection tests while user is in connection settings
  useEffect(() => {
    // Signal that user is actively configuring connection settings
    // This will pause periodic background tests to avoid confusion
    window.dispatchEvent(
      new CustomEvent("imswitch:pausePeriodicTests", {
        detail: { pause: true },
      })
    );

    // Only auto-test once on mount if we have saved connection settings from Redux
    if (connectionSettings.ip && connectionSettings.apiPort && !hasAutoTested) {
      setHasAutoTested(true);

      const timer = setTimeout(() => {
        // Trigger connection test with saved settings
        window.dispatchEvent(
          new CustomEvent("imswitch:checkConnection", {
            detail: {
              ip: connectionSettings.ip,
              port: connectionSettings.apiPort,
              websocketPort: connectionSettings.websocketPort,
            },
          })
        );
      }, 1500); // Delay to allow component to settle

      return () => clearTimeout(timer);
    }

    // Cleanup: Resume periodic tests when component unmounts
    return () => {
      window.dispatchEvent(
        new CustomEvent("imswitch:pausePeriodicTests", {
          detail: { pause: false },
        })
      );
    };
  }, [
    connectionSettings.ip,
    connectionSettings.apiPort,
    connectionSettings.websocketPort,
    hasAutoTested,
  ]);

  // Handler for testing connection (separate from saving)
  const handleTestConnection = async () => {
    if (!hasConnectionSettings) {
      dispatch(
        setNotification({
          message: "Please configure IP and API port first",
          type: "warning",
        })
      );
      return;
    }

    try {
      setIsTestingConnection(true);

      const fullIP = `${hostProtocol}${hostIP}`;

      // Dispatch custom event to trigger connection check in WebSocketHandler
      window.dispatchEvent(
        new CustomEvent("imswitch:checkConnection", {
          detail: {
            ip: fullIP,
            port: apiPort,
            websocketPort: websocketPort,
          },
        })
      );

      // Give it a moment to complete both tests
      setTimeout(() => {
        setIsTestingConnection(false);
        dispatch(
          setNotification({
            message: "Connection tests completed - check status indicators",
            type: "info",
          })
        );
      }, 5000);
    } catch (e) {
      dispatch(
        setNotification({
          message: "Error testing connection!",
          type: "error",
        })
      );
      setIsTestingConnection(false);
    }
  };

  // Handler for saving settings only (no automatic test)
  const handleSave = async () => {
    try {
      setIsSaving(true);
      const fullIP = `${hostProtocol}${hostIP}`;

      // Save to Redux
      dispatch(setIp(fullIP));
      dispatch(setWebsocketPort(websocketPort));
      dispatch(setApiPort(apiPort));

      dispatch(
        setNotification({
          message: "Settings saved successfully!",
          type: "success",
        })
      );
    } catch (e) {
      dispatch(
        setNotification({
          message: "Error saving settings!",
          type: "error",
        })
      );
    } finally {
      setIsSaving(false);
    }
  };

  const isDirty =
    `${hostProtocol}${hostIP}` !== (connectionSettings.ip || "") ||
    websocketPort !== (connectionSettings.websocketPort || "") ||
    apiPort !== (connectionSettings.apiPort || "");

  const getWebSocketStatusColor = (status) => {
    switch (status) {
      case "success":
        return "success";
      case "testing":
        return "info";
      case "failed":
      case "timeout":
        return "error";
      default:
        return "default";
    }
  };

  const getWebSocketStatusLabel = (status) => {
    switch (status) {
      case "success":
        return "Connected";
      case "testing":
        return "Testing...";
      case "failed":
        return "Failed";
      case "timeout":
        return "Timeout";
      case "idle":
        return "Not Tested";
      default:
        return "Unknown";
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: "auto" }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Backend Connection
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure connection settings for ImSwitch backend communication
        </Typography>
      </Box>

      {/* Periodic Tests Paused Info */}
      <Alert severity="info" sx={{ mb: 2 }}>
        <Typography variant="body2">
          <strong>Info:</strong> Periodic connection tests are paused while you
          configure settings to avoid confusion between test results and your
          current changes.
        </Typography>
      </Alert>

      {/* Connection Status Alert */}
      <Alert
        severity={
          isBackendConnected &&
          (websocketTestStatus === "success" || isWebSocketConnected)
            ? "success"
            : hasConnectionSettings
            ? "error"
            : "warning"
        }
        sx={{ mb: 3 }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          {isBackendConnected ? (
            <CheckCircle fontSize="small" />
          ) : (
            <Warning fontSize="small" />
          )}
          <Typography variant="body2">
            {isBackendConnected ? (
              <>
                <strong>Backend API Connected:</strong> Settings are accessible.
                {isHardwareConnected ? (
                  <span>
                    {" "}
                    Hardware also connected - full control available.
                  </span>
                ) : (
                  <span>
                    {" "}
                    Hardware not connected - check UC2 board connection.
                  </span>
                )}
                {websocketTestStatus === "success" || isWebSocketConnected ? (
                  <span> WebSocket ready for live streaming.</span>
                ) : (
                  <span> Configure WebSocket port for live streaming.</span>
                )}
              </>
            ) : hasConnectionSettings ? (
              <>
                <strong>Backend API Failed:</strong> Please check your settings
                and ensure the backend is running.
              </>
            ) : (
              <>
                <strong>Not Configured:</strong> Please configure connection
                settings below.
              </>
            )}
          </Typography>
        </Box>
      </Alert>

      {/* Connection Configuration Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Cable color="primary" />
            <Typography variant="h6">Backend Configuration</Typography>
            <Chip
              label={isBackendConnected ? "API Connected" : "API Disconnected"}
              color={isBackendConnected ? "success" : "error"}
              size="small"
              variant="outlined"
            />
            {isBackendConnected && (
              <Chip
                label={
                  isHardwareConnected
                    ? "Hardware Connected"
                    : "Hardware Disconnected"
                }
                color={isHardwareConnected ? "success" : "warning"}
                size="small"
                variant="outlined"
              />
            )}
            {websocketPort && (
              <Chip
                label={`WebSocket ${getWebSocketStatusLabel(
                  websocketTestStatus
                )}`}
                color={getWebSocketStatusColor(websocketTestStatus)}
                size="small"
                variant="outlined"
                icon={
                  websocketTestStatus === "testing" ? <Settings /> : undefined
                }
              />
            )}
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Configure the IP address and ports for connecting to the ImSwitch
            backend server.
          </Typography>

          {/* Basic Configuration Form */}
          <Box
            component="form"
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr 2fr",
              gap: 3,
              alignItems: "start",
              mb: 3,
            }}
            autoComplete="off"
          >
            {/* Protocol Selection */}
            <TextField
              select
              id="protocol"
              label="Protocol"
              value={hostProtocol}
              onChange={(e) => setHostProtocol(e.target.value)}
              fullWidth
            >
              <MenuItem value="https://">https://</MenuItem>
              <MenuItem value="http://">http://</MenuItem>
            </TextField>

            {/* IP Address */}
            <TextField
              id="ip-address"
              label="IP Address / Hostname"
              type="text"
              value={hostIP}
              onChange={(e) =>
                setHostIP(e.target.value.trim().replace(/^https?:\/\//, ""))
              }
              fullWidth
              placeholder="e.g., 192.168.1.100 or localhost"
              helperText="Usually same as frontend URL"
            />
          </Box>

          {/* Advanced Settings Accordion */}
          <Accordion sx={{ mb: 3 }}>
            <AccordionSummary
              expandIcon={<ExpandMore />}
              aria-controls="advanced-settings-content"
              id="advanced-settings-header"
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Tune color="action" />
                <Typography variant="subtitle1">Advanced Settings</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Both services typically run on port 80. Only change these if
                you have a custom setup.
              </Typography>

              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: "1fr 2fr",
                  gap: 3,
                  alignItems: "start",
                }}
              >
                {/* WebSocket Port */}
                <TextField
                  id="port-websocket"
                  label="WebSocket Port"
                  type="text"
                  value={websocketPort}
                  onChange={(e) => setWebsocketPortState(e.target.value.trim())}
                  fullWidth
                  placeholder="e.g., 80"
                />

                {/* API Port */}
                <TextField
                  id="port-api"
                  label="API Port"
                  type="text"
                  value={apiPort}
                  onChange={(e) => setApiPortState(e.target.value.trim())}
                  fullWidth
                  placeholder="e.g., 80"
                />
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Current Configuration Preview */}
          {hasConnectionSettings && (
            <>
              <Divider sx={{ my: 3 }} />
              <Paper sx={{ p: 2, bgcolor: "background.default" }}>
                <Typography
                  variant="subtitle2"
                  gutterBottom
                  color="text.secondary"
                >
                  Current Configuration Preview:
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <Computer fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Backend API"
                      secondary={`${hostProtocol}${hostIP}${
                        apiPort &&
                        !(
                          (hostProtocol === "https://" && apiPort === "443") ||
                          (hostProtocol === "http://" && apiPort === "80")
                        )
                          ? `:${apiPort}`
                          : ""
                      }/imswitch/api`}
                    />
                    <Chip
                      label={isBackendConnected ? "Connected" : "Disconnected"}
                      color={isBackendConnected ? "success" : "error"}
                      size="small"
                      variant="outlined"
                    />
                  </ListItem>
                  {websocketPort && (
                    <ListItem>
                      <ListItemIcon>
                        <Wifi fontSize="small" />
                      </ListItemIcon>
                      <ListItemText
                        primary="WebSocket Connection"
                        secondary={`${
                          hostProtocol === "https://" ? "wss" : "ws"
                        }://${hostIP}${
                          websocketPort &&
                          !(
                            (hostProtocol === "https://" &&
                              websocketPort === "443") ||
                            (hostProtocol === "http://" &&
                              websocketPort === "80")
                          )
                            ? `:${websocketPort}`
                            : ""
                        }/imswitch/socket.io`}
                      />
                      <Chip
                        label={getWebSocketStatusLabel(websocketTestStatus)}
                        color={getWebSocketStatusColor(websocketTestStatus)}
                        size="small"
                        variant="outlined"
                      />
                    </ListItem>
                  )}
                </List>
              </Paper>
            </>
          )}
        </CardContent>

        <CardActions sx={{ p: 2, pt: 0 }}>
          {/* Save Settings Button */}
          <Button
            variant="contained"
            color="primary"
            onClick={handleSave}
            disabled={!isDirty || isSaving || isTestingConnection}
            startIcon={isSaving ? <CircularProgress size={16} /> : <Save />}
          >
            {isSaving ? "Saving..." : "Save Settings"}
          </Button>

          {/* Test Connection Button */}
          <Button
            variant="outlined"
            color="secondary"
            onClick={handleTestConnection}
            disabled={!hasConnectionSettings || isTestingConnection || isSaving}
            startIcon={
              isTestingConnection ? (
                <CircularProgress size={16} />
              ) : (
                <Settings />
              )
            }
          >
            {isTestingConnection ? "Testing..." : "Test Connection"}
          </Button>

          {/* Reset to Smart Defaults Button */}
          <Button
            variant="text"
            color="secondary"
            onClick={() => {
              const defaults = getSmartDefaults();
              setHostProtocol(defaults.protocol);
              setHostIP(defaults.hostname);
              setWebsocketPortState(defaults.port);
              setApiPortState(defaults.port);
            }}
            disabled={isTestingConnection || isSaving}
          >
            Reset to Defaults
          </Button>

          {/* Status Text */}
          {(isTestingConnection || isSaving) && (
            <Typography variant="body2" color="text.secondary" sx={{ ml: 2 }}>
              {isTestingConnection
                ? "Testing backend and WebSocket connection..."
                : "Saving settings..."}
            </Typography>
          )}

          {/* Auto-test notification */}
          {!hasAutoTested && hasConnectionSettings && (
            <Typography variant="caption" color="info.main" sx={{ ml: 2 }}>
              Auto-testing connection...
            </Typography>
          )}

          {/* Periodic tests paused indicator */}
          <Chip
            label="Periodic Tests Paused"
            size="small"
            variant="outlined"
            color="info"
            sx={{ ml: "auto" }}
          />
        </CardActions>
      </Card>

      {/* Connection Help Card */}
      <Card>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <Info color="info" />
            <Typography variant="h6">Connection Help</Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Common configuration examples and troubleshooting tips. Use "Test
            Connection" to verify your settings work without saving, or "Save
            Settings" to store your configuration.
          </Typography>

          <List dense>
            <ListItem>
              <ListItemIcon>
                <Computer fontSize="small" />
              </ListItemIcon>
              <ListItemText
                primary="Local Development"
                secondary="IP: localhost, API Port: 8000, WebSocket Port: 80"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <Wifi fontSize="small" />
              </ListItemIcon>
              <ListItemText
                primary="Raspberry Pi"
                secondary="IP: 192.168.1.x (check your router), same ports as above"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <Warning fontSize="small" />
              </ListItemIcon>
              <ListItemText
                primary="Troubleshooting"
                secondary="Ensure backend server is running and firewall allows connections"
              />
            </ListItem>
          </List>
        </CardContent>
      </Card>
    </Box>
  );
}

export default ConnectionSettings;
