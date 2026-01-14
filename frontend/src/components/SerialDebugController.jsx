import React from "react";
import { useSelector, useDispatch } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Alert,
  Chip,
  TextField,
  Paper,
} from "@mui/material";
import {
  Terminal,
  CheckCircle,
  ErrorOutline,
  Memory,
  Send,
  Clear,
} from "@mui/icons-material";

const SerialDebugController = () => {
  const dispatch = useDispatch();

  // Get connection settings from Redux
  const { ip: hostIP, apiPort: hostPort } = useSelector(
    getConnectionSettingsState
  );

  // Get UC2 state from Redux
  const uc2State = useSelector(uc2Slice.getUc2State);
  const isBackendConnected = uc2State.backendConnected; // API reachable
  const serialPayload = uc2State.serialPayload;
  const serialLog = uc2State.serialLog;

  // Serial communication handler
  const handleSendSerial = () => {
    if (!serialPayload?.trim()) return;

    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/writeSerial?payload=${encodeURIComponent(
      serialPayload
    )}`;

    fetch(url, { method: "GET" })
      .then((response) => response.json())
      .then(() => {
        dispatch(uc2Slice.addSerialLogEntry(`Sent: ${serialPayload}`));
        // Clear the input after sending
        dispatch(uc2Slice.setSerialPayload(""));
      })
      .catch((error) => {
        console.error("Error sending serial:", error);
        dispatch(uc2Slice.addSerialLogEntry(`Error: Failed to send command`));
      });
  };

  // Clear serial log
  const handleClearLog = () => {
    dispatch(uc2Slice.clearSerialLog());
  };

  // Handle Enter key press in text field
  const handleKeyPress = (event) => {
    if (event.key === "Enter" && event.ctrlKey) {
      handleSendSerial();
    }
  };

  // Note: WebSocket signals (sigUC2SerialReadMessage, sigUC2SerialWriteMessage)
  // are now handled centrally by WebSocketHandler.js and dispatched to Redux

  return (
    <Box sx={{ p: 3, maxWidth: 1000, mx: "auto" }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Serial Debug Interface
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Send raw serial commands directly to the UC2 board for debugging and
          advanced control
        </Typography>
      </Box>

      {/* Connection Status */}
      <Alert
        severity={isBackendConnected ? "success" : "error"}
        sx={{ mb: 3 }}
        icon={isBackendConnected ? <CheckCircle /> : <ErrorOutline />}
      >
        <Typography variant="body2">
          {isBackendConnected ? (
            <>
              <strong>UC2 Board Connected:</strong> Serial commands can be sent.
            </>
          ) : (
            <>
              <strong>UC2 Board Disconnected:</strong> Please check your
              connection settings first.
            </>
          )}
        </Typography>
      </Alert>

      {/* Serial Command Input Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Terminal color="primary" />
            <Typography variant="h6">Command Input</Typography>
            <Chip
              label={isBackendConnected ? "Ready" : "Disconnected"}
              color={isBackendConnected ? "success" : "error"}
              size="small"
              variant="outlined"
            />
          </Box>

          <Box sx={{ mb: 3 }}>
            <TextField
              label="Serial Command Payload"
              variant="outlined"
              fullWidth
              multiline
              rows={4}
              value={serialPayload || ""}
              onChange={(e) =>
                dispatch(uc2Slice.setSerialPayload(e.target.value))
              }
              onKeyDown={handleKeyPress}
              disabled={!isBackendConnected}
              placeholder={`Enter serial commands here...

Examples:
- {"task":"/state_get"}
- {"task":"/motor_get"}
- {"task":"/led_set", "led": {"LEDArray_left": 100}}

Press Ctrl+Enter to send`}
              helperText={
                isBackendConnected
                  ? "Enter JSON commands for the UC2 board. Use Ctrl+Enter to send quickly."
                  : "Connect to UC2 board first in Connection Settings"
              }
              sx={{
                "& .MuiInputBase-input": {
                  fontFamily: "monospace",
                  fontSize: "0.9rem",
                },
              }}
            />
          </Box>

          <Box sx={{ display: "flex", gap: 2 }}>
            <Button
              variant="contained"
              onClick={handleSendSerial}
              disabled={!isBackendConnected || !serialPayload?.trim()}
              startIcon={<Send />}
              size="large"
            >
              Send Command
            </Button>

            <Button
              variant="outlined"
              onClick={() => dispatch(uc2Slice.setSerialPayload(""))}
              disabled={!serialPayload?.trim()}
              startIcon={<Clear />}
            >
              Clear Input
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Serial Communication Log Card */}
      <Card>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Memory color="secondary" />
            <Typography variant="h6">Communication Log</Typography>
            <Chip
              label={`${serialLog?.length || 0} entries`}
              color="info"
              size="small"
              variant="outlined"
            />
            <Box sx={{ flexGrow: 1 }} />
            <Button
              variant="outlined"
              size="small"
              onClick={handleClearLog}
              disabled={!serialLog?.length}
              startIcon={<Clear />}
            >
              Clear Log
            </Button>
          </Box>

          <Paper
            elevation={2}
            sx={{
              height: 400,
              overflow: "auto",
              p: 2,
              bgcolor: "grey.900",
              color: "grey.100",
              border: "1px solid",
              borderColor: "grey.700",
              borderRadius: 2,
            }}
          >
            {serialLog?.length > 0 ? (
              serialLog.map((entry, index) => (
                <Typography
                  key={index}
                  variant="body2"
                  component="div"
                  sx={{
                    fontFamily: "monospace",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    mb: 0.5,
                    py: 0.5,
                    px: 1,
                    bgcolor: index % 2 === 0 ? "transparent" : "grey.800",
                    borderRadius: 1,
                    fontSize: "0.85rem",
                    lineHeight: 1.4,
                    // Color coding for different types of messages
                    color: entry.startsWith("Sent:")
                      ? "#4caf50"
                      : entry.startsWith("Read:")
                      ? "#2196f3"
                      : entry.startsWith("Write:")
                      ? "#ff9800"
                      : entry.startsWith("Error:")
                      ? "#f44336"
                      : "inherit",
                  }}
                >
                  <Box component="span" sx={{ opacity: 0.7, mr: 1 }}>
                    [{new Date().toLocaleTimeString()}]
                  </Box>
                  {entry}
                </Typography>
              ))
            ) : (
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  height: "100%",
                  color: "grey.500",
                }}
              >
                <Terminal sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                <Typography variant="body1" sx={{ textAlign: "center" }}>
                  No serial communication yet
                </Typography>
                <Typography variant="body2" sx={{ textAlign: "center", mt: 1 }}>
                  Send a command to see the communication log here
                </Typography>
              </Box>
            )}
          </Paper>

          {/* Quick Command Examples */}
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Quick Commands:
            </Typography>
            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
              {[
                '{"task":"/state_get"}',
                '{"task":"/motor_get"}',
                '{"task":"/led_get"}',
                '{"task":"/home_xyz"}',
              ].map((cmd, index) => (
                <Button
                  key={index}
                  variant="outlined"
                  size="small"
                  onClick={() => dispatch(uc2Slice.setSerialPayload(cmd))}
                  disabled={!isBackendConnected}
                  sx={{
                    fontFamily: "monospace",
                    fontSize: "0.75rem",
                    textTransform: "none",
                  }}
                >
                  {cmd}
                </Button>
              ))}
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SerialDebugController;
