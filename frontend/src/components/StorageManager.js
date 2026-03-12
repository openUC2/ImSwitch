import React, { useState } from "react";
import { useSelector } from "react-redux";
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Button,
  Chip,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
} from "@mui/material";
import {
  Usb as UsbIcon,
  Folder as FolderIcon,
  CheckCircle as CheckCircleIcon,
  Storage as StorageIcon,
  HardDrive as HardDriveIcon,
} from "@mui/icons-material";
import { getStorageState } from "../../state/slices/StorageSlice";
import apiStorageControllerSetActivePath from "../../backendapi/apiStorageControllerSetActivePath";

/**
 * StorageManager Component
 *
 * Manages external storage drives (USB, SD cards) for the FileManager.
 * Allows users to:
 * - Detect connected external drives
 * - Mount/activate a drive for data storage
 * - View current active storage location
 * - Switch between drives
 *
 * @param {function} onStorageChange - Callback when storage location changes
 */
const StorageManager = ({ onStorageChange }) => {
  const storageState = useSelector(getStorageState);
  const [error, setError] = useState(null);
  const [mounting, setMounting] = useState(null); // Track which drive is being mounted
  const storageStatus = storageState.status;
  const externalDrives = storageStatus.available_drives || [];
  const loading = !storageState.hasReceivedSnapshot;

  // Mount/activate a drive
  const handleMountDrive = async (drivePath, persist = false) => {
    setMounting(drivePath);
    setError(null);

    try {
      const result = await apiStorageControllerSetActivePath(
        drivePath,
        persist,
      );

      if (result.status === "success") {
        // Notify parent component
        if (onStorageChange) {
          onStorageChange(result.active_path);
        }
      } else {
        setError(result.message || "Failed to mount drive");
      }
    } catch (err) {
      console.error("Failed to mount drive:", err);
      setError(`Failed to mount drive: ${err.message}`);
    } finally {
      setMounting(null);
    }
  };

  // Format size in human-readable format
  const formatSize = (bytes) => {
    if (!bytes) return "Unknown";
    const gb = bytes / 1024 ** 3;
    if (gb >= 1) return `${gb.toFixed(2)} GB`;
    const mb = bytes / 1024 ** 2;
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <Box sx={{ p: 2 }}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 2,
        }}
      >
        <Typography
          variant="h6"
          sx={{ display: "flex", alignItems: "center", gap: 1 }}
        >
          <StorageIcon /> Storage Management
        </Typography>
        <Tooltip title="Storage status updates automatically">
          <span>
            <IconButton disabled>
              <StorageIcon />
            </IconButton>
          </span>
        </Tooltip>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Current Active Storage */}
      {storageStatus && (
        <Card
          sx={{
            mb: 2,
            bgcolor: "primary.light",
            color: "primary.contrastText",
          }}
        >
          <CardContent>
            <Typography
              variant="subtitle2"
              sx={{ display: "flex", alignItems: "center", gap: 1 }}
            >
              <CheckCircleIcon fontSize="small" /> Active Storage Location
            </Typography>
            <Typography variant="body2" sx={{ mt: 1, fontFamily: "monospace" }}>
              {storageStatus.active_data_path || "Not set"}
            </Typography>
            {storageStatus.disk_usage && (
              <Box sx={{ mt: 1, display: "flex", gap: 2 }}>
                <Chip
                  label={`Free: ${formatSize(storageStatus.disk_usage.free)}`}
                  size="small"
                  sx={{ bgcolor: "rgba(255, 255, 255, 0.2)" }}
                />
                <Chip
                  label={`Total: ${formatSize(storageStatus.disk_usage.total)}`}
                  size="small"
                  sx={{ bgcolor: "rgba(255, 255, 255, 0.2)" }}
                />
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* External Drives */}
      <Card>
        <CardContent>
          <Typography
            variant="subtitle1"
            sx={{ mb: 2, display: "flex", alignItems: "center", gap: 1 }}
          >
            <UsbIcon /> External Drives
          </Typography>

          {loading ? (
            <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
              <CircularProgress />
            </Box>
          ) : externalDrives.length === 0 ? (
            <Alert severity="info">
              No external drives detected. Please connect a USB drive or SD
              card.
            </Alert>
          ) : (
            <List>
              {externalDrives.map((drive, index) => {
                const drivePath = drive.path || drive.mount_point;
                const isActive =
                  storageStatus?.active_data_path?.startsWith(drivePath);
                const isMounting = mounting === drivePath;

                return (
                  <ListItem
                    key={index}
                    disablePadding
                    secondaryAction={
                      <Button
                        variant={isActive ? "outlined" : "contained"}
                        color={isActive ? "success" : "primary"}
                        size="small"
                        onClick={() => handleMountDrive(drivePath, true)}
                        disabled={isActive || isMounting}
                        startIcon={
                          isMounting ? <CircularProgress size={16} /> : null
                        }
                      >
                        {isActive
                          ? "Active"
                          : isMounting
                            ? "Mounting..."
                            : "Mount"}
                      </Button>
                    }
                  >
                    <ListItemButton disabled={isActive}>
                      <ListItemIcon>
                        <HardDriveIcon
                          color={isActive ? "success" : "action"}
                        />
                      </ListItemIcon>
                      <ListItemText
                        primary={drive.device || drivePath}
                        secondary={
                          <Box>
                            <Typography variant="body2" component="span">
                              {drivePath}
                            </Typography>
                            {drive.size && (
                              <Typography
                                variant="body2"
                                component="span"
                                sx={{ ml: 2 }}
                              >
                                {formatSize(drive.size)}
                              </Typography>
                            )}
                            {drive.filesystem && (
                              <Chip
                                label={drive.filesystem}
                                size="small"
                                sx={{ ml: 1 }}
                              />
                            )}
                          </Box>
                        }
                      />
                    </ListItemButton>
                  </ListItem>
                );
              })}
            </List>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default StorageManager;
