import { useState, useEffect } from "react";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import { getStorageState } from "../state/slices/StorageSlice";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import {
  Box,
  Typography,
  Button,
  Switch,
  FormControlLabel,
  Card,
  CardContent,
  Alert,
  Chip,
  LinearProgress,
  Link,
} from "@mui/material";
import {
  Computer,
  Storage,
  Warning,
  CheckCircle,
  ErrorOutline,
  OpenInNew,
} from "@mui/icons-material";

export default function SystemSettings() {
  // Get connection settings from Redux
  const { ip: hostIP, apiPort: hostPort } = useSelector(
    getConnectionSettingsState,
  );

  // Get backend connection status
  const uc2State = useSelector(uc2Slice.getUc2State);
  const storageState = useSelector(getStorageState);
  const isBackendConnected = uc2State.backendConnected; // API reachable (enables UI)

  // Safety toggles
  const [enableImSwitch, setEnableImSwitch] = useState(false);
  const [isImSwitchRunning, setIsImSwitchRunning] = useState(false);
  const [deviceAdminUrl] = useState(
    () => `http://${hostIP}/admin/panel/boot/?mode=minimal&nav=hidden`,
  );
  const [deviceAdminLoaded, setDeviceAdminLoaded] = useState(false);

  const base = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController`;
  const activeUsage = storageState.status.active_device?.usage || null;
  const diskUsagePercent =
    typeof activeUsage?.percent_used === "number"
      ? activeUsage.percent_used
      : null;
  const diskUsage =
    typeof diskUsagePercent === "number"
      ? `${diskUsagePercent.toFixed(1)}%`
      : null;

  // API communication
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

  // Poll every 10 seconds to check if ImSwitch is running (less frequent)
  useEffect(() => {
    // Only poll if backend is connected
    if (!isBackendConnected) {
      setIsImSwitchRunning(false);
      return;
    }

    const checkImSwitchStatus = async () => {
      try {
        const res = await fetch(`${base}/isImSwitchRunning`);
        if (res.ok) {
          const data = await res.json();
          console.log("Full ImSwitch status response:", data);

          // Handle the actual API response format - direct boolean
          let isRunning;
          if (typeof data === "boolean") {
            // API returns direct boolean (true/false)
            isRunning = data;
          } else if (data && typeof data === "object") {
            // Fallback: check for various possible property names
            isRunning = data.running || data.isRunning || data.status || false;
          } else {
            isRunning = false;
          }

          setIsImSwitchRunning(isRunning);
          console.log("Processed ImSwitch status:", isRunning);
        }
      } catch (error) {
        console.error("Error checking ImSwitch status:", error);
        setIsImSwitchRunning(false);
      }
    };

    // Initial check
    checkImSwitchStatus();

    const intervalId = setInterval(checkImSwitchStatus, 10000);
    return () => clearInterval(intervalId);
  }, [base, isBackendConnected]);

  // Helper function to get disk usage color based on percentage
  const getDiskUsageColor = (usage) => {
    if (!usage || usage === "Loading..." || usage.includes("Error"))
      return "default";
    const percentage = parseFloat(usage.replace("%", ""));
    if (percentage > 90) return "error";
    if (percentage > 75) return "warning";
    return "success";
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: "auto" }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          System Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Control ImSwitch services and system operations
        </Typography>
      </Box>

      {/* Backend Connection Status */}
      <Alert
        severity={isBackendConnected ? "success" : "error"}
        sx={{ mb: 3 }}
        icon={isBackendConnected ? <CheckCircle /> : <ErrorOutline />}
      >
        <Typography variant="body2">
          {isBackendConnected ? (
            <>
              <strong>Backend Connected:</strong> System controls are available.
            </>
          ) : (
            <>
              <strong>Backend Disconnected:</strong> Please configure connection
              settings first.
            </>
          )}
        </Typography>
      </Alert>

      {/* System Status Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Computer color="primary" />
            <Typography variant="h6">System Status</Typography>
          </Box>

          <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 3 }}>
            {/* ImSwitch Status */}
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                ImSwitch Service
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Chip
                  label={isImSwitchRunning ? "Running" : "Not Running"}
                  color={isImSwitchRunning ? "success" : "default"}
                  size="small"
                  variant="outlined"
                />
              </Box>
            </Box>

            {/* Disk Usage */}
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Disk Usage
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Storage fontSize="small" />
                <Chip
                  label={
                    isBackendConnected
                      ? storageState.hasReceivedSnapshot
                        ? (diskUsage ?? "Unavailable")
                        : "Loading..."
                      : "Backend disconnected"
                  }
                  color={getDiskUsageColor(diskUsage)}
                  size="small"
                  variant="outlined"
                />
              </Box>
              {diskUsage &&
                !diskUsage.includes("Error") &&
                !diskUsage.includes("Loading") && (
                  <LinearProgress
                    variant="determinate"
                    value={parseFloat(diskUsage.replace("%", ""))}
                    color={getDiskUsageColor(diskUsage)}
                    sx={{ mt: 1, height: 6, borderRadius: 3 }}
                  />
                )}
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* ImSwitch Control Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Computer color="secondary" />
            <Typography variant="h6">ImSwitch Control</Typography>
          </Box>

          <FormControlLabel
            control={
              <Switch
                checked={enableImSwitch}
                onChange={(e) => setEnableImSwitch(e.target.checked)}
              />
            }
            label="Enable ImSwitch control"
            sx={{ mb: 2 }}
          />

          <Box sx={{ display: "flex", gap: 2 }}>
            <Button
              variant="contained"
              disabled={!enableImSwitch || !isBackendConnected}
              onClick={() => callEndpoint(`${base}/restartImSwitch`)}
            >
              Restart ImSwitch
            </Button>

            <Button
              variant="outlined"
              color="secondary"
              disabled={!enableImSwitch || !isBackendConnected}
              onClick={() => callEndpoint(`${base}/stopImSwitch`)}
            >
              Stop ImSwitch
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Raspberry Pi Control Card */}
      <Card>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Warning color="error" />
            <Typography variant="h6">Raspberry Pi Control</Typography>
          </Box>

          <Alert severity="warning" sx={{ mb: 3 }}>
            <Typography variant="body2">
              <strong>Warning:</strong> These operations will affect the entire
              system.
            </Typography>
          </Alert>

          {/* Device-Admin iframe for reboot/shutdown */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              System Management
            </Typography>
            {deviceAdminLoaded ? (
              <Box
                sx={{
                  border: "1px solid #e0e0e0",
                  borderRadius: 1,
                  overflow: "hidden",
                  mb: 2,
                }}
              >
                <iframe
                  src={deviceAdminUrl}
                  style={{
                    width: "100%",
                    height: "300px",
                    border: "none",
                    borderRadius: "4px",
                  }}
                  title="Device Admin Panel - Reboot/Shutdown"
                  onLoad={() => setDeviceAdminLoaded(true)}
                  onError={() => setDeviceAdminLoaded(false)}
                  sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                />
              </Box>
            ) : (
              <Box
                sx={{
                  border: "1px solid #424242",
                  borderRadius: 1,
                  p: 2,
                  mb: 2,
                  backgroundColor: "#2a2a2a",
                  display: "flex",
                  flexDirection: "column",
                  gap: 1,
                }}
              >
                <Typography variant="body2" sx={{ color: "#b0b0b0", mb: 1 }}>
                  Device admin panel not available. Please use the direct link:
                </Typography>
                <Link
                  href={deviceAdminUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 0.5,
                    color: "#64b5f6",
                    "&:hover": { color: "#90caf9" },
                  }}
                >
                  Reboot/Shutdown in device-admin <OpenInNew fontSize="small" />
                </Link>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
