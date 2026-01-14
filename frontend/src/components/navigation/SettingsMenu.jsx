import { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../../backendapi/createAxiosInstance';
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Badge,
  Tooltip,
  Typography,
  Box,
  Switch,
  LinearProgress,
  Chip,
} from "@mui/material";
import {
  Settings,
  DarkMode,
  LightMode,
  Wifi,
  Info,
  Memory,
  Cable,
  Tune,
  SystemUpdate,
  Storage,
  Build,
  Article,
} from "@mui/icons-material";
import { formatDiskUsage } from "../../utils/formatUtils";
import { useDeveloperMode } from "../../utils/useDeveloperMode";

// Redux state management following Copilot Instructions
import { toggleTheme, getThemeState } from "../../state/slices/ThemeSlice.js";
import * as connectionSettingsSlice from "../../state/slices/ConnectionSettingsSlice.js";
import * as uc2Slice from "../../state/slices/UC2Slice.js";

/**
 * ImSwitch Settings Menu Component
 * Following Copilot Instructions for API communication patterns
 * All backend-dependent features disabled when connection unavailable
 */
const SettingsMenu = ({ onNavigate }) => {
  const dispatch = useDispatch();
  const [anchorEl, setAnchorEl] = useState(null);
  const [diskUsage, setDiskUsage] = useState(null);
  const open = Boolean(anchorEl);

  // Developer Mode Hook - enables backdoor access when backend is offline
  const { isDeveloperMode, deactivateDeveloperMode } = useDeveloperMode();

  const { isDarkMode } = useSelector(getThemeState);
  const connectionSettings = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );

  // Get actual backend connection status from UC2 slice - following Copilot Instructions
  const uc2State = useSelector(uc2Slice.getUc2State);
  const isBackendConnected = uc2State.backendConnected; // API reachable (enables UI)
  const isHardwareConnected = uc2State.uc2Connected; // Hardware connected

  // Developer Override: Allow access to all features when developer mode is active
  const allowAccess = isBackendConnected || isDeveloperMode;

  // Fetch disk usage when backend is connected - following Copilot Instructions
  useEffect(() => {
    // Only fetch if backend is connected and menu is open
    if (!isBackendConnected || !open) {
      setDiskUsage(null);
      return;
    }

    const fetchDiskUsage = async () => {
      try {
        const api = createAxiosInstance();
        const response = await api.get('/UC2ConfigController/getDiskUsage');
        const data = response.data;

        // Handle the actual API response format - direct number
        const usage = formatDiskUsage(data);

        setDiskUsage(usage);
      } catch (error) {
        console.error("Error fetching disk usage:", error);
        setDiskUsage("Error");
        setDiskUsage("Error");
      }
    };

    // Initial fetch when menu opens
    fetchDiskUsage();

    // Refresh every 30 seconds while menu is open
    const intervalId = setInterval(fetchDiskUsage, 30000);
    return () => clearInterval(intervalId);
  }, [isBackendConnected, open]);

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  // Handle navigation items (close menu)
  const handleNavigationClick = (pluginName) => {
    handleClose();
    if (onNavigate) {
      onNavigate(pluginName);
    }
  };

  // Handle theme toggle (keep menu open)
  const handleThemeToggle = (event) => {
    event.stopPropagation(); // Prevent menu close
    dispatch(toggleTheme());
  };

  // Check if connection settings are configured
  const hasConnectionSettings =
    connectionSettings.ip && connectionSettings.apiPort;

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
    <>
      <Tooltip
        title={
          isDeveloperMode
            ? "Settings & Configuration (🔧 Developer Mode)"
            : "Settings & Configuration"
        }
      >
        <IconButton
          color="inherit"
          onClick={handleClick}
          sx={{
            ml: 1,
            // Developer mode glow effect
            ...(isDeveloperMode && {
              boxShadow: "0 0 8px rgba(255, 152, 0, 0.5)",
              backgroundColor: "rgba(255, 152, 0, 0.1)",
            }),
          }}
          aria-label="settings menu"
        >
          <Badge
            color={
              isDeveloperMode
                ? "warning" // 🔧 Developer mode overrides status
                : isBackendConnected
                ? "success"
                : hasConnectionSettings
                ? "error"
                : "warning"
            }
            variant="dot"
            sx={{
              "& .MuiBadge-badge": {
                right: 2,
                top: 2,
                width: 8,
                height: 8,
              },
            }}
          >
            <Settings />
          </Badge>
        </IconButton>
      </Tooltip>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        PaperProps={{
          elevation: 3,
          sx: {
            mt: 1.5,
            minWidth: 280, // Slightly wider to accommodate disk usage
            "& .MuiMenuItem-root": {
              px: 2,
              py: 1,
            },
          },
        }}
        transformOrigin={{ horizontal: "right", vertical: "top" }}
        anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
      >
        {/* Enhanced Connection Status Header */}
        <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: "divider" }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            ImSwitch Backend
          </Typography>

          {/* Connection Status */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
            <Chip
              label={
                isBackendConnected
                  ? isHardwareConnected
                    ? "Connected"
                    : "API Connected"
                  : hasConnectionSettings
                  ? "Connection Failed"
                  : "Not Configured"
              }
              color={
                isBackendConnected
                  ? isHardwareConnected
                    ? "success" // ✅ Full connection
                    : "warning" // 🟡 API only, no hardware
                  : hasConnectionSettings
                  ? "error" // ❌ Connection failed
                  : "warning" // ⚠️ Not configured
              }
              size="small"
              variant="outlined"
              sx={{
                height: 20,
                fontSize: "0.75rem",
                "& .MuiChip-label": { px: 1 },
              }}
            />
          </Box>

          {/* Connection Details */}
          {hasConnectionSettings && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: "block", mb: 1 }}
            >
              {connectionSettings.ip}:{connectionSettings.apiPort}/imswitch
            </Typography>
          )}
          {!hasConnectionSettings && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: "block", mb: 1 }}
            >
              Click "Backend Connection" to configure
            </Typography>
          )}
        </Box>

        {/* Theme Toggle - Always available (local frontend state) */}
        <MenuItem onClick={handleThemeToggle}>
          <ListItemIcon>
            {isDarkMode ? (
              <LightMode fontSize="small" />
            ) : (
              <DarkMode fontSize="small" />
            )}
          </ListItemIcon>
          <ListItemText primary="Dark Mode" />
          <Switch
            checked={isDarkMode}
            onChange={handleThemeToggle}
            size="small"
            onClick={(e) => e.stopPropagation()}
          />
        </MenuItem>

        <Divider />

        {/* Backend Connection - Always available (needed to establish connection) */}
        <MenuItem onClick={() => handleNavigationClick("Connections")}>
          <ListItemIcon>
            <Cable
              fontSize="small"
              color={
                isBackendConnected
                  ? "success"
                  : hasConnectionSettings
                  ? "error"
                  : "warning"
              }
            />
          </ListItemIcon>
          <ListItemText
            primary="Backend Connection"
            secondary={
              isBackendConnected
                ? "Connected - Configure settings"
                : hasConnectionSettings
                ? "Connection failed - Check settings"
                : "Setup required"
            }
          />
        </MenuItem>

        {/* Backend-Dependent Items - Following Copilot Instructions for API communication */}

        {/* System Settings - Requires backend API calls */}
        <MenuItem
          onClick={() => handleNavigationClick("SystemSettings")}
          disabled={!allowAccess}
          sx={{
            opacity: allowAccess ? 1 : 0.5,
            "&.Mui-disabled": {
              opacity: 0.5,
            },
          }}
        >
          <ListItemIcon>
            <Storage
              fontSize="small"
              color={allowAccess ? "inherit" : "disabled"}
            />
          </ListItemIcon>
          <ListItemText
            primary="System Settings"
            secondary={
              allowAccess
                ? isDeveloperMode && !isBackendConnected
                  ? "Hardware configuration (Developer Mode)"
                  : "Hardware configuration"
                : "Requires backend connection"
            }
          />
        </MenuItem>

        {/* Motor Settings - Motor configuration per axis */}
        <MenuItem
          onClick={() => handleNavigationClick("MotorSettings")}
          disabled={!allowAccess}
          sx={{
            opacity: allowAccess ? 1 : 0.5,
            "&.Mui-disabled": {
              opacity: 0.5,
            },
          }}
        >
          <ListItemIcon>
            <Tune
              fontSize="small"
              color={allowAccess ? "inherit" : "disabled"}
            />
          </ListItemIcon>
          <ListItemText
            primary="Motor Settings"
            secondary={
              allowAccess
                ? isDeveloperMode && !isBackendConnected
                  ? "Stepsize, homing, TMC config (Developer Mode)"
                  : "Stepsize, homing, TMC config"
                : "Requires backend connection"
            }
          />
        </MenuItem>

        {/* ImSwitch Backend Settings */}
        <MenuItem
          onClick={() => handleNavigationClick("UC2")}
          disabled={!allowAccess}
          sx={{
            opacity: allowAccess ? 1 : 0.5,
            "&.Mui-disabled": {
              opacity: 0.5,
            },
          }}
        >
          <ListItemIcon>
            <Memory
              fontSize="small"
              color={allowAccess ? "inherit" : "disabled"}
            />
          </ListItemIcon>
          <ListItemText
            primary="ImSwitch Backend Settings"
            secondary={
              allowAccess
                ? isDeveloperMode && !isBackendConnected
                  ? "Microscope configuration (Developer Mode)"
                  : "Microscope configuration"
                : "Requires backend connection"
            }
          />
        </MenuItem>

        {/* Logging Interface */}
        <MenuItem
          onClick={() => handleNavigationClick("Logging")}
          disabled={!allowAccess}
          sx={{
            opacity: allowAccess ? 1 : 0.5,
            "&.Mui-disabled": {
              opacity: 0.5,
            },
          }}
        >
          <ListItemIcon>
            <Article
              fontSize="small"
              color={allowAccess ? "inherit" : "disabled"}
            />
          </ListItemIcon>
          <ListItemText
            primary="Logging"
            secondary={
              allowAccess
                ? isDeveloperMode && !isBackendConnected
                  ? "View & download log files (Developer Mode)"
                  : "View & download log files"
                : "Requires backend connection"
            }
          />
        </MenuItem>


        {/* WiFi Configuration - Requires backend API calls for network management */}
        <MenuItem
          onClick={() => handleNavigationClick("WiFi")}
          disabled={!allowAccess}
          sx={{
            opacity: allowAccess ? 1 : 0.5,
            "&.Mui-disabled": {
              opacity: 0.5,
            },
          }}
        >
          <ListItemIcon>
            <Wifi
              fontSize="small"
              color={allowAccess ? "inherit" : "disabled"}
            />
          </ListItemIcon>
          <ListItemText
            primary="WiFi Configuration"
            secondary={
              allowAccess
                ? isDeveloperMode && !isBackendConnected
                  ? "Network setup (Developer Mode)"
                  : "Network setup"
                : "Requires backend connection"
            }
          />
        </MenuItem>

        <Divider />

        {/* System Updates */}
        <MenuItem
          onClick={() => handleNavigationClick("SystemUpdate")}
          disabled={!allowAccess}
          sx={{
            opacity: allowAccess ? 1 : 0.5,
            "&.Mui-disabled": {
              opacity: 0.5,
            },
          }}
        >
          <ListItemIcon>
            <SystemUpdate
              fontSize="small"
              color={allowAccess ? "inherit" : "disabled"}
            />
          </ListItemIcon>
          <ListItemText
            primary="System Updates"
            secondary={
              allowAccess
                ? isDeveloperMode && !isBackendConnected
                  ? "Update system & firmware (Developer Mode)"
                  : "Update system & firmware"
                : "Requires backend connection"
            }
          />
        </MenuItem>

        <Divider />

        {/* Developer Mode Toggle - Only visible when active */}
        {isDeveloperMode && (
          <MenuItem onClick={deactivateDeveloperMode}>
            <ListItemIcon>
              <Build fontSize="small" color="warning" />
            </ListItemIcon>
            <ListItemText
              primary="Developer Mode"
              secondary="Click to deactivate (🔧 Active)"
            />
          </MenuItem>
        )}

        {isDeveloperMode && <Divider />}

        {/* About - Always available (static information) */}
        <MenuItem onClick={() => handleNavigationClick("About")}>
          <ListItemIcon>
            <Info fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="About ImSwitch"
            secondary="Version & information"
          />
        </MenuItem>
      </Menu>
    </>
  );
};

export default SettingsMenu;
