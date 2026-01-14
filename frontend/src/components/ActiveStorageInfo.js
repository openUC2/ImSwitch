import React, { useState, useEffect, useCallback } from "react";
import { useDispatch } from "react-redux";
import { Box, Typography, Chip, Button, CircularProgress } from "@mui/material";
import {
  SdStorage as SdStorageIcon,
  Folder as FolderIcon,
} from "@mui/icons-material";
import { setNotification } from "../state/slices/NotificationSlice";
import apiStorageControllerGetStorageStatus from "../backendapi/apiStorageControllerGetStorageStatus";
import apiStorageControllerSetActivePath from "../backendapi/apiStorageControllerSetActivePath";

/**
 * ActiveStorageInfo Component
 * Displays the currently active storage location in the FileManager
 * Provides quick access to switch back to local storage
 */
const ActiveStorageInfo = ({ onRefresh, onStorageChange }) => {
  const dispatch = useDispatch();
  const [storageStatus, setStorageStatus] = useState(null);
  const [defaultPath, setDefaultPath] = useState(null);
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const status = await apiStorageControllerGetStorageStatus();
      setStorageStatus(status);

      // Get default local path
      if (!defaultPath) {
        // Use /home/pi/Datasets as the default local storage path
        const localPath = "/home/pi/Datasets";
        setDefaultPath(localPath);
      }
    } catch (error) {
      console.error("Failed to fetch storage status:", error);
    } finally {
      setLoading(false);
    }
  }, [defaultPath]);

  useEffect(() => {
    fetchStatus();

    // Refresh every 10 seconds
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleSwitchToLocal = async () => {
    if (!defaultPath) return;

    setSwitching(true);
    try {
      const result = await apiStorageControllerSetActivePath(defaultPath, true);
      console.log("ActiveStorageInfo: Switch to local result:", result);

      await fetchStatus();

      // Notify parent about storage change (updates FileManager's initialPath)
      const newActivePath = result.active_path || defaultPath;
      console.log(
        "ActiveStorageInfo: Notifying storage change to:",
        newActivePath
      );

      if (onStorageChange) {
        onStorageChange(newActivePath);
      }

      dispatch(
        setNotification({
          message: "Switched to local storage",
          type: "success",
        })
      );

      // Trigger FileManager refresh if callback provided
      if (onRefresh) {
        onRefresh();
      }
    } catch (error) {
      console.error("Failed to switch to local storage:", error);
      dispatch(
        setNotification({
          message: `Failed to switch: ${error.message}`,
          type: "error",
        })
      );
    } finally {
      setSwitching(false);
    }
  };

  if (loading || !storageStatus?.active_path) {
    return null;
  }

  // Check if it's an external drive (contains /media/)
  const isExternalDrive = storageStatus.active_path.includes("/media/");

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1,
        p: 1.5,
        mb: 1,
        bgcolor: isExternalDrive ? "success.light" : "info.light",
        color: isExternalDrive ? "success.contrastText" : "info.contrastText",
        borderRadius: 1,
      }}
    >
      {isExternalDrive ? (
        <SdStorageIcon fontSize="small" />
      ) : (
        <FolderIcon fontSize="small" />
      )}
      <Box sx={{ flex: 1 }}>
        <Typography
          variant="caption"
          sx={{ fontWeight: "bold", display: "block" }}
        >
          {isExternalDrive ? "External Storage Active" : "Local Storage Active"}
        </Typography>
        <Typography
          variant="caption"
          sx={{ fontFamily: "monospace", fontSize: "0.75rem" }}
        >
          {storageStatus.active_path}
        </Typography>
      </Box>
      {storageStatus.disk_usage && (
        <Chip
          label={`${(storageStatus.disk_usage.free / 1024 ** 3).toFixed(
            1
          )} GB free`}
          size="small"
          sx={{
            bgcolor: "rgba(255, 255, 255, 0.3)",
            color: "inherit",
            fontWeight: "bold",
          }}
        />
      )}
      {isExternalDrive && (
        <Button
          variant="outlined"
          size="small"
          onClick={handleSwitchToLocal}
          disabled={switching}
          startIcon={
            switching ? (
              <CircularProgress size={12} />
            ) : (
              <FolderIcon fontSize="small" />
            )
          }
          sx={{
            color: "inherit",
            borderColor: "rgba(255, 255, 255, 0.5)",
            "&:hover": {
              borderColor: "rgba(255, 255, 255, 0.8)",
              bgcolor: "rgba(255, 255, 255, 0.1)",
            },
          }}
        >
          {switching ? "Switching..." : "Switch to Local"}
        </Button>
      )}
    </Box>
  );
};

export default ActiveStorageInfo;
