import { useDispatch, useSelector } from "react-redux";
import { usePWA } from "../context/PWAContext.js";
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Alert,
  Chip,
} from "@mui/material";
import { Download, DesktopWindows, DeleteOutline } from "@mui/icons-material";
import { setNotification } from "../state/slices/NotificationSlice.js";

function detectBrowser() {
  const ua = navigator.userAgent;
  if (ua.includes("Edg/")) return "edge";
  if (ua.includes("Firefox")) return "firefox";
  if (ua.includes("Chrome") || ua.includes("Chromium")) return "chrome";
  if (ua.includes("Safari")) return "safari";
  return "other";
}

function getBrowserInstallInstructions() {
  const browser = detectBrowser();
  const instructions = {
    chrome: {
      title: "Google Chrome",
      installSteps: [
        "Open ImSwitch in Chrome.",
        "At the top right, click More (⋯).",
        "Go to Cast, save, and share > Install page as app...",
        "On some pages, you can click Install in the address bar instead.",
        "Follow the on-screen instructions.",
      ],
      uninstallSteps: [
        "Open the installed ImSwitch app.",
        "At the top right, click More (⋯).",
        "Select Uninstall ImSwitch...",
        "Confirm with Remove.",
        "Optional: Manage installed web apps at chrome://apps.",
      ],
    },
    edge: {
      title: "Microsoft Edge",
      installSteps: [
        "Open ImSwitch in Edge.",
        "At the top right, click More (⋯).",
        "Select Apps > Install this site as an app.",
        "Confirm the installation dialog.",
      ],
      uninstallSteps: [
        "Open the installed ImSwitch app.",
        "At the top right, click More (⋯).",
        "Select App settings > Uninstall this app.",
        "Confirm uninstall.",
      ],
    },
    safari: {
      title: "Safari",
      installSteps: [
        "Open ImSwitch in Safari.",
        "Use Share > Add to Dock.",
        "Confirm to create the app shortcut.",
      ],
      uninstallSteps: [
        "Open Finder > Applications.",
        "Locate the installed web app.",
        "Move it to Trash.",
      ],
    },
    firefox: {
      title: "Firefox",
      installSteps: [
        "Firefox has limited PWA support.",
        "Use a bookmark as fallback or use Chrome/Edge for app install.",
      ],
      uninstallSteps: [
        "If you created a bookmark shortcut, remove the bookmark.",
      ],
    },
    other: {
      title: "Your Browser",
      installSteps: [
        "Look for an Install/Add to Home Screen option in the browser menu.",
      ],
      uninstallSteps: [
        "Remove the installed web app from your system app list.",
      ],
    },
  };

  return instructions[browser] || instructions.other;
}

function DesktopAppSettings() {
  const dispatch = useDispatch();
  const connectionSettings = useSelector((state) => state.connectionSettingsState);
  const { installPromptEvent, setInstallPromptEvent } = usePWA();

  const hasConnectionSettings =
    Boolean(connectionSettings?.ip) && Boolean(connectionSettings?.apiPort);

  const instructions = getBrowserInstallInstructions();

  const handleInstallDesktopApp = async () => {
    if (!hasConnectionSettings) {
      dispatch(
        setNotification({
          message: "Please configure backend connection first.",
          type: "warning",
        }),
      );
      return;
    }

    if (installPromptEvent) {
      try {
        installPromptEvent.prompt();
        const choiceResult = await installPromptEvent.userChoice;
        setInstallPromptEvent(null);

        dispatch(
          setNotification({
            message:
              choiceResult?.outcome === "accepted"
                ? "App installation started."
                : "Installation cancelled.",
            type: choiceResult?.outcome === "accepted" ? "success" : "info",
          }),
        );
        return;
      } catch (error) {
        console.error("PWA install prompt failed:", error);
      }
    }

    dispatch(
      setNotification({
        message: `${instructions.title}:\n${instructions.installSteps.join("\n")}`,
        type: "info",
      }),
    );
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: "auto" }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Desktop App
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Install or uninstall ImSwitch as a desktop web app.
        </Typography>
      </Box>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <DesktopWindows color="primary" />
            <Typography variant="h6">Install</Typography>
            <Chip
              size="small"
              label={
                installPromptEvent ? "Install prompt available" : "Manual steps"
              }
              color={installPromptEvent ? "success" : "default"}
              variant="outlined"
            />
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Browser detected: {instructions.title}
          </Typography>

          <Button
            variant="contained"
            startIcon={<Download />}
            onClick={handleInstallDesktopApp}
            sx={{ mb: 2 }}
          >
            Install Desktop App
          </Button>

          <Alert severity="info">
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
              Installation steps
            </Typography>
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {instructions.installSteps.map((step, idx) => (
                <li key={idx}>
                  <Typography variant="caption">{step}</Typography>
                </li>
              ))}
            </ol>
          </Alert>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <DeleteOutline color="warning" />
            <Typography variant="h6">Uninstall</Typography>
          </Box>

          <Alert severity="warning">
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
              Uninstallation steps ({instructions.title})
            </Typography>
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {instructions.uninstallSteps.map((step, idx) => (
                <li key={idx}>
                  <Typography variant="caption">{step}</Typography>
                </li>
              ))}
            </ol>
          </Alert>
        </CardContent>
      </Card>
    </Box>
  );
}

export default DesktopAppSettings;
