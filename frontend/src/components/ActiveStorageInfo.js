import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box, Typography, Chip, Button, CircularProgress } from "@mui/material";
import {
  SdStorage as SdStorageIcon,
  Folder as FolderIcon,
} from "@mui/icons-material";
import { setNotification } from "../state/slices/NotificationSlice";
import { getStorageState } from "../state/slices/StorageSlice";
import apiStorageControllerSetActivePath from "../backendapi/apiStorageControllerSetActivePath";

/**
 * ActiveStorageInfo Component
 * Displays the currently active storage location in the FileManager
 * Provides quick access to switch back to the default internal storage.
 */
const ActiveStorageInfo = ({ onStorageChange }) => {
  const dispatch = useDispatch();
  const storageState = useSelector(getStorageState);
  const [switching, setSwitching] = useState(false);
  const storageStatus = storageState.status;
  const activeDevice = storageStatus.active_device;
  const defaultDevice = storageStatus.default_device;

  const handleSwitchToDefault = async () => {
    if (!defaultDevice?.path) return;

    setSwitching(true);
    try {
      const result = await apiStorageControllerSetActivePath(
        defaultDevice.path,
        true,
      );
      const newActivePath = result.active_path || defaultDevice.path;

      if (onStorageChange) {
        onStorageChange(newActivePath);
      }

      dispatch(
        setNotification({
          message: "Switched to internal storage",
          type: "success",
        }),
      );
    } catch (error) {
      console.error("Failed to switch to internal storage:", error);
      dispatch(
        setNotification({
          message: `Failed to switch: ${error.message}`,
          type: "error",
        }),
      );
    } finally {
      setSwitching(false);
    }
  };

  if (!storageState.hasReceivedSnapshot || !storageStatus?.active_path) {
    return null;
  }

  const isExternalDrive = Boolean(activeDevice && !activeDevice.is_internal);
  const activeUsage = activeDevice?.usage;

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
          {isExternalDrive
            ? "External Storage Active"
            : "Internal Storage Active"}
        </Typography>
        <Typography
          variant="caption"
          sx={{ fontFamily: "monospace", fontSize: "0.75rem" }}
        >
          {storageStatus.active_path}
        </Typography>
      </Box>
      {activeUsage && (
        <Chip
          label={`${(activeUsage.free / 1024 ** 3).toFixed(1)} GB free`}
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
          onClick={handleSwitchToDefault}
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
          {switching ? "Switching..." : "Switch to Internal"}
        </Button>
      )}
    </Box>
  );
};

export default ActiveStorageInfo;
