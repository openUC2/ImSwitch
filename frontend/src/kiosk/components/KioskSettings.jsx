/**
 * KioskSettings.jsx
 * 
 * Minimal Settings for Kiosk Mode
 * - Device Manager access
 * - Pi Reboot button
 * - Basic system info
 */

import { useState } from "react";
import { useSelector } from "react-redux";
import {
  Box,
  Button,
  Paper,
  Typography,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Divider,
} from "@mui/material";
import {
  Devices as DevicesIcon,
  RestartAlt as RebootIcon,
  Info as InfoIcon,
} from "@mui/icons-material";
import { getConnectionSettingsState } from "../../state/slices/ConnectionSettingsSlice";

const KioskSettings = () => {
  const [rebootDialogOpen, setRebootDialogOpen] = useState(false);
  const [deviceManagerOpen, setDeviceManagerOpen] = useState(false);
  const connectionSettings = useSelector(getConnectionSettingsState);
  const { ip, apiPort } = connectionSettings;

  const handleRebootConfirm = async () => {
    try {
      // TODO: Implement reboot API call when available
      // await fetch(`${ip}:${apiPort}/SystemUpdateController/reboot`, { method: "POST" });
      console.log("Reboot requested for:", ip);
      setRebootDialogOpen(false);
      // Show confirmation
      alert("Reboot command would be sent here (API integration pending)");
    } catch (error) {
      console.error("Reboot failed:", error);
    }
  };

  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        p: 2,
      }}
    >
      <Stack spacing={2}>
        {/* System Info */}
        <Paper elevation={2} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            <InfoIcon sx={{ mr: 1, verticalAlign: "middle" }} />
            System Information
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Backend: {ip}:{apiPort}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Mode: Kiosk (800x480)
          </Typography>
        </Paper>

        <Divider />

        {/* Device Manager Button */}
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<DevicesIcon />}
          onClick={() => setDeviceManagerOpen(true)}
          fullWidth
          sx={{
            py: 2,
            fontSize: "1.1rem",
          }}
        >
          Device Manager
        </Button>

        {/* Reboot Button */}
        <Button
          variant="contained"
          color="warning"
          size="large"
          startIcon={<RebootIcon />}
          onClick={() => setRebootDialogOpen(true)}
          fullWidth
          sx={{
            py: 2,
            fontSize: "1.1rem",
          }}
        >
          Reboot Raspberry Pi
        </Button>

        {/* Info Alert */}
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="body2">
            Exit Kiosk Mode: Press <strong>Ctrl + Alt + F2</strong> to access terminal
          </Typography>
        </Alert>
      </Stack>

      {/* Reboot Confirmation Dialog */}
      <Dialog
        open={rebootDialogOpen}
        onClose={() => setRebootDialogOpen(false)}
      >
        <DialogTitle>Confirm Reboot</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to reboot the Raspberry Pi?
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            The system will restart and the kiosk mode will reload automatically.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRebootDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleRebootConfirm}
            color="warning"
            variant="contained"
          >
            Reboot Now
          </Button>
        </DialogActions>
      </Dialog>

      {/* Device Manager Dialog - Placeholder */}
      <Dialog
        open={deviceManagerOpen}
        onClose={() => setDeviceManagerOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Device Manager</DialogTitle>
        <DialogContent>
          <Alert severity="info">
            Device Manager integration coming soon.
            For now, access it from the main web interface.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeviceManagerOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default KioskSettings;
