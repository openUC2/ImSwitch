import React, { useState, useEffect, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  IconButton,
  Badge,
  Popover,
  Box,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  Button,
  Chip,
  CircularProgress,
  Divider,
  LinearProgress,
} from "@mui/material";
import {
  SdStorage as SdStorageIcon,
  CheckCircle as CheckCircleIcon,
  Refresh as RefreshIcon,
  Close as CloseIcon,
  Eject as EjectIcon,
  OpenInNew as OpenInNewIcon,
  Folder as FolderIcon,
} from "@mui/icons-material";
import { setNotification } from "../state/slices/NotificationSlice";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import apiStorageControllerListExternalDrives from "../backendapi/apiStorageControllerListExternalDrives";
import apiStorageControllerGetStorageStatus from "../backendapi/apiStorageControllerGetStorageStatus";
import apiStorageControllerSetActivePath from "../backendapi/apiStorageControllerSetActivePath";
import apiStorageControllerGetConfigPaths from "../backendapi/apiStorageControllerGetConfigPaths";
import apiUC2ConfigControllerGetDiskUsage from "../backendapi/apiUC2ConfigControllerGetDiskUsage";

/**
 * StorageButton Component
 *
 * Top bar button with auto-detection of external drives.
 * Shows badge with number of available drives and allows switching between them.
 * Note: Drives are automatically mounted by the OS. Use the admin panel for unmounting.
 *
 * @param {function} onStorageChange - Callback when storage location changes
 * @param {function} onFileManagerRefresh - Callback to refresh FileManager
 * @param {number} scanInterval - Interval for background scanning in ms (default: 10000)
 * @param {boolean} disabled - Disable button when backend is not connected
 */
const StorageButton = ({
  onStorageChange,
  onFileManagerRefresh,
  scanInterval = 10000,
  disabled = false,
}) => {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const [anchorEl, setAnchorEl] = useState(null);
  const [externalDrives, setExternalDrives] = useState([]);
  const [storageStatus, setStorageStatus] = useState(null);
  const [defaultPath, setDefaultPath] = useState(null);
  const [internalDiskUsage, setInternalDiskUsage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [switching, setSwitching] = useState(null);
  const [previousDriveCount, setPreviousDriveCount] = useState(0);

  const open = Boolean(anchorEl);

  // Format size in human-readable format
  const formatSize = (bytes) => {
    if (!bytes) return "Unknown";
    const gb = bytes / 1024 ** 3;
    if (gb >= 1) return `${gb.toFixed(2)} GB`;
    const mb = bytes / 1024 ** 2;
    return `${mb.toFixed(2)} MB`;
  };

  // Fetch storage status and external drives
  const fetchStorageInfo = useCallback(
    async (showLoading = true) => {
      console.log(
        "StorageButton: fetchStorageInfo called, showLoading:",
        showLoading
      );
      if (showLoading) setLoading(true);
      setError(null);

      try {
        // Get current storage status
        console.log("StorageButton: Fetching storage status...");
        const status = await apiStorageControllerGetStorageStatus();
        console.log("StorageButton: Storage status:", status);
        setStorageStatus(status);

        // Get default path if not already loaded
        if (!defaultPath) {
          const configPaths = await apiStorageControllerGetConfigPaths();
          console.log("StorageButton: Config paths from backend:", configPaths);

          // Use /home/pi/Datasets as the default local storage path
          const localPath = "/home/pi/Datasets";
          console.log("StorageButton: Using local storage path:", localPath);
          setDefaultPath(localPath);
        }

        // Get disk usage for internal storage from UC2ConfigController
        try {
          const diskUsageData = await apiUC2ConfigControllerGetDiskUsage();
          console.log("Internal disk usage:", diskUsageData);

          // API now returns {raw, formatted, percent}
          setInternalDiskUsage({
            percent: diskUsageData.percent,
            formatted: diskUsageData.formatted,
          });
        } catch (err) {
          console.error("Failed to fetch internal disk usage:", err);
          // Keep old value or set null
        }

        // Get list of external drives
        console.log("StorageButton: Fetching external drives...");
        const drives = await apiStorageControllerListExternalDrives();
        console.log("StorageButton: Raw API response:", drives);

        // Filter out EFI system partitions and other non-user drives
        const allDrives = drives.drives || [];
        const newDrives = allDrives.filter((drive) => {
          const name = (drive.name || "").toLowerCase();
          const label = (drive.label || "").toLowerCase();
          const path = (drive.path || "").toLowerCase();

          // Filter out EFI partitions, boot partitions, and system partitions
          const isSystemPartition =
            name.includes("efi") ||
            label.includes("efi") ||
            path.includes("/boot") ||
            path.includes("efi") ||
            name.includes("boot") ||
            label.includes("boot");

          return !isSystemPartition;
        });

        console.log(
          "StorageButton: Scanned drives (after filtering):",
          newDrives.length,
          newDrives
        );

        // Check for new drives
        if (newDrives.length > previousDriveCount && previousDriveCount > 0) {
          const newDrive = newDrives[newDrives.length - 1];
          dispatch(
            setNotification({
              message: `New drive detected: ${newDrive.label || newDrive.path}`,
              type: "info",
            })
          );
        }

        setExternalDrives(newDrives);
        setPreviousDriveCount(newDrives.length);
      } catch (err) {
        console.error("Failed to fetch storage info:", err);
        if (showLoading) {
          setError("Failed to load storage information.");
        }
      } finally {
        if (showLoading) setLoading(false);
      }
    },
    [previousDriveCount, dispatch, defaultPath]
  );

  // Select/switch to a drive
  const handleSelectDrive = async (drivePath, persist = true) => {
    console.log("StorageButton: Switching to path:", drivePath);
    setSwitching(drivePath);
    setError(null);

    try {
      // Check if this is local storage (not in /media/)
      const isLocal = !drivePath.includes("/media/");

      if (isLocal) {
        // For local storage: Backend always creates ImSwitchData subfolder
        // So we need to pass the parent directory, not the full path with ImSwitchData
        console.log("StorageButton: Switching to local storage");

        // Use the local path that was clicked (should be /home/pi/Datasets)
        // Backend will create /home/pi/Datasets/ImSwitchData automatically
        console.log("StorageButton: Using local base path:", drivePath);

        const result = await apiStorageControllerSetActivePath(
          drivePath,
          persist
        );

        console.log("StorageButton: Backend response:", result);
        console.log("StorageButton: Backend created path:", result.active_path);

        await fetchStorageInfo(false);

        // Use the active_path returned by backend (will be drivePath/ImSwitchData)
        const newActivePath = result.active_path || drivePath;
        console.log(
          "StorageButton: Active path after local switch:",
          newActivePath
        );

        if (onStorageChange) {
          onStorageChange(newActivePath);
        }

        dispatch(
          setNotification({
            message: result.message || "Switched to local storage",
            type: "success",
          })
        );
      } else {
        // For external drives, use set_active_path
        const result = await apiStorageControllerSetActivePath(
          drivePath,
          persist
        );

        console.log("StorageButton: Backend response:", result);
        console.log("StorageButton: result.active_path:", result.active_path);

        // Backend returns success message directly or in result object
        await fetchStorageInfo(false);

        // Notify parent about storage change (this updates FileManager's initialPath)
        const newActivePath = result.active_path || drivePath;
        console.log(
          "StorageButton: Notifying storage change to:",
          newActivePath
        );

        if (onStorageChange) {
          onStorageChange(newActivePath);
        }

        dispatch(
          setNotification({
            message: result.message || "Switched to external storage",
            type: "success",
          })
        );
      }

      // Don't call onFileManagerRefresh here - App.jsx handles it after path change
    } catch (err) {
      console.error("Failed to switch drive:", err);
      setError(`Failed to switch drive: ${err.message}`);
    } finally {
      setSwitching(null);
    }
  };

  // Handle button click
  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
    fetchStorageInfo();
  };

  const handleClose = () => {
    setAnchorEl(null);
    setError(null);
  };

  // Initial load
  useEffect(() => {
    console.log("StorageButton: Initial mount, starting scan...");
    fetchStorageInfo(false);
  }, [fetchStorageInfo]);

  // Background scanning
  useEffect(() => {
    console.log(
      "StorageButton: Setting up background scan every",
      scanInterval / 1000,
      "seconds"
    );
    const interval = setInterval(() => {
      fetchStorageInfo(false);
    }, scanInterval);

    return () => clearInterval(interval);
  }, [scanInterval, fetchStorageInfo]);

  return (
    <>
      <IconButton
        onClick={handleClick}
        title={
          disabled
            ? "Storage Management (Backend not connected)"
            : "Storage Management"
        }
        size="small"
        disabled={disabled}
        sx={{ color: "inherit" }}
      >
        <Badge badgeContent={externalDrives.length} color="primary">
          <SdStorageIcon />
        </Badge>
      </IconButton>

      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "right",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
      >
        <Box sx={{ width: 400, maxHeight: 500, overflow: "auto" }}>
          {/* Header */}
          <Box
            sx={{
              p: 2,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              borderBottom: 1,
              borderColor: "divider",
            }}
          >
            <Typography
              variant="h6"
              sx={{ display: "flex", alignItems: "center", gap: 1 }}
            >
              <SdStorageIcon /> Select Storage
            </Typography>
            <Box>
              <IconButton
                size="small"
                onClick={() => fetchStorageInfo()}
                disabled={loading}
              >
                <RefreshIcon />
              </IconButton>
              <IconButton size="small" onClick={handleClose}>
                <CloseIcon />
              </IconButton>
            </Box>
          </Box>

          {error && (
            <Box
              sx={{
                m: 2,
                p: 2,
                bgcolor: "error.main",
                color: "error.contrastText",
                borderRadius: 1,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <Typography variant="body2">{error}</Typography>
              <IconButton
                size="small"
                onClick={() => setError(null)}
                sx={{ color: "inherit" }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Box>
          )}

          {/* Current Active Storage */}
          {storageStatus && (
            <Box
              sx={{
                p: 2,
                bgcolor: "primary.main",
                color: "primary.contrastText",
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <CheckCircleIcon fontSize="small" />
                <Typography variant="subtitle2" component="div">
                  FileManager Base Directory
                </Typography>
              </Box>
              <Typography
                variant="body2"
                component="div"
                sx={{ mt: 1, fontFamily: "monospace", fontSize: "0.85rem" }}
              >
                {storageStatus.active_path || "Not set"}
              </Typography>
              {storageStatus.disk_usage && (
                <Box sx={{ mt: 1, display: "flex", gap: 1 }}>
                  <Chip
                    label={`Free: ${formatSize(storageStatus.disk_usage.free)}`}
                    size="small"
                    sx={{
                      bgcolor: "rgba(255, 255, 255, 0.2)",
                      color: "inherit",
                    }}
                  />
                  <Chip
                    label={`Total: ${formatSize(
                      storageStatus.disk_usage.total
                    )}`}
                    size="small"
                    sx={{
                      bgcolor: "rgba(255, 255, 255, 0.2)",
                      color: "inherit",
                    }}
                  />
                </Box>
              )}
            </Box>
          )}

          <Divider />

          {/* External Drives */}
          <Box sx={{ p: 2 }}>
            {/* Unmount Warning */}
            <Box
              sx={{
                mb: 2,
                p: 1,
                bgcolor: (theme) =>
                  theme.palette.mode === "dark"
                    ? "rgba(255, 152, 0, 0.15)"
                    : "#fff3e0",
                borderRadius: 1,
                border: 1,
                borderColor: "warning.main",
                display: "flex",
                alignItems: "center",
                gap: 1,
              }}
            >
              <EjectIcon sx={{ color: "warning.main", fontSize: 20 }} />
              <Box sx={{ flex: 1 }}>
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: "bold",
                    color: (theme) =>
                      theme.palette.mode === "dark" ? "#ffa726" : "#e65100",
                    display: "block",
                    mb: 0.25,
                  }}
                >
                  Before removing a drive:
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    color: (theme) =>
                      theme.palette.mode === "dark"
                        ? "text.secondary"
                        : "#bf360c",
                    fontSize: "0.7rem",
                  }}
                >
                  Unmount it in the Admin Panel first
                </Typography>
              </Box>
              <Button
                size="small"
                variant="contained"
                color="warning"
                onClick={() => {
                  const ip = connectionSettings.ip || "localhost";
                  const cleanIp = ip.replace(/^https?:\/\//, "");
                  window.open(
                    `http://${cleanIp}/admin/panel/storage/`,
                    "_blank"
                  );
                }}
                startIcon={<OpenInNewIcon fontSize="small" />}
                sx={{
                  whiteSpace: "nowrap",
                  fontSize: "0.75rem",
                  py: 0.5,
                  px: 1,
                }}
              >
                Open Panel
              </Button>
            </Box>

            {/* Internal Storage */}
            <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: "bold" }}>
              Internal Storage
            </Typography>

            {loading ? (
              <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
                <CircularProgress size={24} />
              </Box>
            ) : defaultPath ? (
              <List dense>
                <ListItem
                  sx={{
                    mb: 1,
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                    p: 1,
                    bgcolor: !storageStatus?.active_path?.includes("/media/")
                      ? "action.selected"
                      : "transparent",
                    borderRadius: 1,
                    border: 1,
                    borderColor: !storageStatus?.active_path?.includes(
                      "/media/"
                    )
                      ? "success.main"
                      : "divider",
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    <FolderIcon
                      fontSize="small"
                      color={
                        !storageStatus?.active_path?.includes("/media/")
                          ? "success"
                          : "action"
                      }
                    />
                  </ListItemIcon>
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        mb: 0.5,
                      }}
                    >
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: !storageStatus?.active_path?.includes(
                            "/media/"
                          )
                            ? "bold"
                            : "normal",
                        }}
                        noWrap
                      >
                        Local Storage
                      </Typography>
                      {internalDiskUsage?.percent !== undefined && (
                        <Typography variant="caption" color="text.secondary">
                          {internalDiskUsage.percent.toFixed(1)}% used
                        </Typography>
                      )}
                    </Box>
                    {internalDiskUsage?.percent !== undefined && (
                      <LinearProgress
                        variant="determinate"
                        value={internalDiskUsage.percent}
                        sx={{
                          height: 6,
                          borderRadius: 1,
                          bgcolor: "action.hover",
                          "& .MuiLinearProgress-bar": {
                            bgcolor: (theme) => {
                              const usage = internalDiskUsage.percent;
                              if (usage > 90) return "error.main";
                              if (usage > 75) return "warning.main";
                              return "success.main";
                            },
                          },
                        }}
                      />
                    )}
                    {Boolean(defaultPath) && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ mt: 0.25, display: "block" }}
                      >
                        {defaultPath}
                      </Typography>
                    )}
                  </Box>
                  {!storageStatus?.active_path?.includes("/media/") ? (
                    <Chip
                      label="ACTIVE"
                      color="success"
                      size="small"
                      icon={<CheckCircleIcon fontSize="small" />}
                      sx={{ fontWeight: "bold" }}
                    />
                  ) : (
                    <Button
                      variant="contained"
                      color="primary"
                      size="small"
                      onClick={() => handleSelectDrive(defaultPath, true)}
                      disabled={switching === defaultPath}
                      startIcon={
                        switching === defaultPath ? (
                          <CircularProgress size={12} />
                        ) : null
                      }
                    >
                      {switching === defaultPath ? "Switching..." : "SELECT"}
                    </Button>
                  )}
                </ListItem>
              </List>
            ) : null}

            <Divider sx={{ my: 2 }} />

            {/* External Drives */}
            <Typography
              variant="subtitle2"
              sx={{ mb: 1.5, fontWeight: "bold" }}
            >
              External Drives ({externalDrives.length})
            </Typography>

            {loading ? (
              <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
                <CircularProgress size={24} />
              </Box>
            ) : externalDrives.length === 0 ? (
              <Box sx={{ mt: 1, p: 2, textAlign: "center" }}>
                <Typography variant="body2" color="text.secondary">
                  No external drives detected.
                </Typography>
              </Box>
            ) : (
              <List dense>
                {externalDrives.map((drive, index) => {
                  const isActive =
                    storageStatus?.active_path?.startsWith(drive.path) ||
                    drive.is_active;
                  const isSwitching = switching === drive.path;

                  return (
                    <ListItem
                      key={index}
                      sx={{
                        mb: 0.5,
                        display: "flex",
                        alignItems: "center",
                        gap: 1,
                        p: 1,
                        bgcolor: isActive ? "action.selected" : "transparent",
                        borderRadius: 1,
                        border: 1,
                        borderColor: isActive ? "success.main" : "divider",
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        <SdStorageIcon
                          fontSize="small"
                          color={isActive ? "success" : "action"}
                        />
                      </ListItemIcon>
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            mb: 0.5,
                          }}
                        >
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: isActive ? "bold" : "normal" }}
                            noWrap
                          >
                            {drive.label || drive.path.split("/").pop()}
                          </Typography>
                          {drive.free_space_gb && drive.total_space_gb && (
                            <Typography
                              variant="caption"
                              color="text.secondary"
                            >
                              {(
                                drive.total_space_gb - drive.free_space_gb
                              ).toFixed(1)}{" "}
                              GB / {drive.total_space_gb.toFixed(1)} GB
                            </Typography>
                          )}
                        </Box>
                        {drive.free_space_gb && drive.total_space_gb && (
                          <LinearProgress
                            variant="determinate"
                            value={
                              (1 - drive.free_space_gb / drive.total_space_gb) *
                              100
                            }
                            sx={{
                              height: 6,
                              borderRadius: 1,
                              bgcolor: "action.hover",
                              "& .MuiLinearProgress-bar": {
                                bgcolor: (theme) => {
                                  const usage =
                                    (1 -
                                      drive.free_space_gb /
                                        drive.total_space_gb) *
                                    100;
                                  if (usage > 90) return "error.main";
                                  if (usage > 75) return "warning.main";
                                  return "success.main";
                                },
                              },
                            }}
                          />
                        )}
                        {drive.filesystem && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ mt: 0.25, display: "block" }}
                          >
                            {drive.filesystem}
                          </Typography>
                        )}
                      </Box>
                      {isActive ? (
                        <Chip
                          label="ACTIVE"
                          color="success"
                          size="small"
                          icon={<CheckCircleIcon fontSize="small" />}
                          sx={{ fontWeight: "bold" }}
                        />
                      ) : (
                        <Button
                          variant="contained"
                          color="primary"
                          size="small"
                          onClick={() => handleSelectDrive(drive.path, true)}
                          disabled={isSwitching}
                          startIcon={
                            isSwitching ? <CircularProgress size={12} /> : null
                          }
                        >
                          {isSwitching ? "Switching..." : "SELECT"}
                        </Button>
                      )}
                    </ListItem>
                  );
                })}
              </List>
            )}
          </Box>
        </Box>
      </Popover>
    </>
  );
};

export default StorageButton;
