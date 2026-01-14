// src/components/WiFiController.js
import { Alert, Box, Button, Paper, Typography } from "@mui/material";
import { OpenInNew as OpenInNewIcon } from "@mui/icons-material";
import { useState } from "react";
import { useSelector } from "react-redux";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";

const WiFiController = () => {
  const [iframeError, setIframeError] = useState(false);

  // Access ImSwitch backend connection settings from Redux - following Copilot Instructions
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  const hostIP = connectionSettingsState.ip;

  // Construct Internet Access URL from backend connection settings
  // According to developer: if ImSwitch URL is {protocol}://{hostname}/imswitch/ui/index.html
  // then device-admin Internet Access page is {protocol}://{hostname}/admin/panel/internet
  const internetAccessUrl = `${hostIP}/admin/panel/internet?nav=hidden&theme=dark`;

  const handleIframeError = () => {
    setIframeError(true);
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Internet Access Configuration
      </Typography>
      <Typography
        variant="body2"
        color="textSecondary"
        gutterBottom
        sx={{ mb: 2 }}
      >
        Configure internet access and network settings for your device.
      </Typography>

      <Alert severity="info" sx={{ mb: 2 }}>
        If the page doesn't load below due to security restrictions,
        <Button
          size="small"
          startIcon={<OpenInNewIcon />}
          onClick={() => window.open(internetAccessUrl, "_blank")}
          sx={{ ml: 1 }}
        >
          open it in a new tab
        </Button>
      </Alert>

      {iframeError && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          The admin panel could not be loaded in the iframe. This may be due to
          network issues or browser security restrictions.
          <Button
            size="small"
            startIcon={<OpenInNewIcon />}
            onClick={() => window.open(internetAccessUrl, "_blank")}
            sx={{ ml: 1 }}
          >
            Open in new tab
          </Button>
        </Alert>
      )}

      <Box
        sx={{
          width: "100%",
          height: "700px",
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 1,
          overflow: "hidden",
          position: "relative",
        }}
      >
        <iframe
          src={internetAccessUrl}
          title="Internet Access Configuration"
          onError={handleIframeError}
          style={{
            width: "100%",
            height: "100%",
            border: "none",
          }}
        />
        {iframeError && (
          <Box
            sx={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              bgcolor: "background.default",
              flexDirection: "column",
              gap: 2,
              p: 4,
            }}
          >
            <Typography variant="h6" color="text.secondary">
              Unable to load admin panel
            </Typography>
            <Typography variant="body2" color="text.secondary" align="center">
              The page at <code>{internetAccessUrl}</code> could not be loaded.
            </Typography>
            <Button
              variant="contained"
              startIcon={<OpenInNewIcon />}
              onClick={() => window.open(internetAccessUrl, "_blank")}
            >
              Open in New Tab
            </Button>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default WiFiController;
