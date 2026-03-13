import React, { useState, useEffect, useMemo, useRef } from "react";
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
  Close as CloseIcon,
  Eject as EjectIcon,
  OpenInNew as OpenInNewIcon,
  Folder as FolderIcon,
} from "@mui/icons-material";
import { setNotification } from "../state/slices/NotificationSlice";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import { getStorageState } from "../state/slices/StorageSlice";
import apiStorageControllerSetActivePath from "../backendapi/apiStorageControllerSetActivePath";

/**
 * StorageButton Component
 *
 * Top bar button with auto-detection of external drives.
 * Shows badge with number of available drives and allows switching between them.
 * Note: Drives are automatically mounted by the OS. Use the admin panel for unmounting.
 *
 * @param {function} onStorageChange - Callback when storage location changes
 * @param {boolean} disabled - Disable button when backend is not connected
 */
const StorageButton = ({ onStorageChange, disabled = false }) => {
  const dispatch = useDispatch();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const storageState = useSelector(getStorageState);
  const [anchorEl, setAnchorEl] = useState(null);
  const [error, setError] = useState(null);
  const [switching, setSwitching] = useState(null);
  const [previousDriveCount, setPreviousDriveCount] = useState(0);
  const storageUsageAlertLevelRef = useRef(null);

  const open = Boolean(anchorEl);
  const storageStatus = storageState.status;
  const storageDevices = useMemo(
    () => storageStatus.storage_devices || [],
    [storageStatus.storage_devices],
  );
  const externalDrives = useMemo(
    () => storageStatus.external_devices || [],
    [storageStatus.external_devices],
  );
  const defaultDevice = storageStatus.default_device || null;
  const activeDevice = storageStatus.active_device || null;
  const isLoadingStorage = !storageState.hasReceivedSnapshot;

  const formatSize = (bytes) => {
    if (typeof bytes !== "number") return "Unknown";
    const gb = bytes / 1024 ** 3;
    if (gb >= 1) return `${gb.toFixed(2)} GB`;
    const mb = bytes / 1024 ** 2;
    return `${mb.toFixed(2)} MB`;
  };

  const getUsagePercent = (usage) => {
    if (!usage) return null;
    if (typeof usage.percent_used === "number") return usage.percent_used;
    if (
      typeof usage.used === "number" &&
      typeof usage.total === "number" &&
      usage.total > 0
    ) {
      return (usage.used / usage.total) * 100;
    }
    return null;
  };

  const getUsageLevel = (percent) => {
    if (typeof percent !== "number") return "normal";
    if (percent >= 90) return "critical";
    if (percent >= 75) return "warning";
    return "normal";
  };

  const handleSelectDrive = async (drivePath, persist = true) => {
    const targetDevice = storageDevices.find(
      (device) => device.path === drivePath,
    );
    setSwitching(drivePath);
    setError(null);

    try {
      const result = await apiStorageControllerSetActivePath(
        drivePath,
        persist,
      );
      const newActivePath = result.active_path || drivePath;

      if (onStorageChange) {
        onStorageChange(newActivePath);
      }

      dispatch(
        setNotification({
          message:
            result.message ||
            (targetDevice?.is_internal
              ? "Switched to internal storage"
              : "Switched to external storage"),
          type: "success",
        }),
      );
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
  };

  const handleClose = () => {
    setAnchorEl(null);
    setError(null);
  };

  useEffect(() => {
    if (externalDrives.length > previousDriveCount && previousDriveCount > 0) {
      const newDrive = externalDrives[externalDrives.length - 1];
      dispatch(
        setNotification({
          message: `New drive detected: ${newDrive.label || newDrive.path}`,
          type: "info",
        }),
      );
    }

    setPreviousDriveCount(externalDrives.length);
  }, [dispatch, externalDrives, previousDriveCount]);

  useEffect(() => {
    const usagePercent = getUsagePercent(activeDevice?.usage);
    const currentLevel = getUsageLevel(usagePercent);
    const previousLevel = storageUsageAlertLevelRef.current;

    if (previousLevel === currentLevel) {
      return;
    }

    storageUsageAlertLevelRef.current = currentLevel;

    if (currentLevel === "normal") {
      return;
    }

    const levelText = currentLevel === "critical" ? "critical" : "high";
    const severity = currentLevel === "critical" ? "warning" : "info";
    const storageLabel = activeDevice?.label || "active storage";
    const usageText =
      typeof usagePercent === "number"
        ? `${usagePercent.toFixed(1)}%`
        : "unknown";

    dispatch(
      setNotification({
        message: `Storage usage ${levelText} on ${storageLabel} (${usageText} used).`,
        type: severity,
      }),
    );
  }, [activeDevice, dispatch]);

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
          {storageState.hasReceivedSnapshot && (
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
              {activeDevice?.usage && (
                <Box sx={{ mt: 1, display: "flex", gap: 1 }}>
                  <Chip
                    label={`Free: ${formatSize(activeDevice.usage.free)}`}
                    size="small"
                    sx={{
                      bgcolor: "rgba(255, 255, 255, 0.2)",
                      color: "inherit",
                    }}
                  />
                  <Chip
                    label={`Total: ${formatSize(activeDevice.usage.total)}`}
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
                    "_blank",
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

            {isLoadingStorage ? (
              <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
                <CircularProgress size={24} />
              </Box>
            ) : defaultDevice ? (
              <List dense>
                <ListItem
                  sx={{
                    mb: 1,
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                    p: 1,
                    bgcolor: defaultDevice.is_active
                      ? "action.selected"
                      : "transparent",
                    borderRadius: 1,
                    border: 1,
                    borderColor: defaultDevice.is_active
                      ? "success.main"
                      : "divider",
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    <FolderIcon
                      fontSize="small"
                      color={defaultDevice.is_active ? "success" : "action"}
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
                          fontWeight: defaultDevice.is_active
                            ? "bold"
                            : "normal",
                        }}
                        noWrap
                      >
                        {defaultDevice.label}
                      </Typography>
                      {typeof getUsagePercent(defaultDevice.usage) ===
                        "number" && (
                        <Typography variant="caption" color="text.secondary">
                          {getUsagePercent(defaultDevice.usage).toFixed(1)}%
                          used
                        </Typography>
                      )}
                    </Box>
                    {typeof getUsagePercent(defaultDevice.usage) ===
                      "number" && (
                      <LinearProgress
                        variant="determinate"
                        value={getUsagePercent(defaultDevice.usage)}
                        sx={{
                          height: 6,
                          borderRadius: 1,
                          bgcolor: "action.hover",
                          "& .MuiLinearProgress-bar": {
                            bgcolor: (theme) => {
                              const usage = getUsagePercent(
                                defaultDevice.usage,
                              );
                              if (usage > 90) return "error.main";
                              if (usage > 75) return "warning.main";
                              return "success.main";
                            },
                          },
                        }}
                      />
                    )}
                    {defaultDevice.path && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ mt: 0.25, display: "block" }}
                      >
                        {defaultDevice.path}
                      </Typography>
                    )}
                  </Box>
                  {defaultDevice.is_active ? (
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
                      onClick={() =>
                        handleSelectDrive(defaultDevice.path, true)
                      }
                      disabled={switching === defaultDevice.path}
                      startIcon={
                        switching === defaultDevice.path ? (
                          <CircularProgress size={12} />
                        ) : null
                      }
                    >
                      {switching === defaultDevice.path
                        ? "Switching..."
                        : "SELECT"}
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

            {isLoadingStorage ? (
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
                            {drive.label}
                          </Typography>
                          {drive.usage && (
                            <Typography
                              variant="caption"
                              color="text.secondary"
                            >
                              {formatSize(drive.usage.used)} /{" "}
                              {formatSize(drive.usage.total)}
                            </Typography>
                          )}
                        </Box>
                        {typeof getUsagePercent(drive.usage) === "number" && (
                          <LinearProgress
                            variant="determinate"
                            value={getUsagePercent(drive.usage)}
                            sx={{
                              height: 6,
                              borderRadius: 1,
                              bgcolor: "action.hover",
                              "& .MuiLinearProgress-bar": {
                                bgcolor: (theme) => {
                                  const usage = getUsagePercent(drive.usage);
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
