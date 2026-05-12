import { Box, Typography, Tooltip } from "@mui/material";
import { Computer, Storage, Lan, Memory } from "@mui/icons-material";

function StepLabel({ step, label }) {
  return (
    <Typography
      variant="caption"
      sx={{ fontWeight: 600, color: "#888", mb: 0.5 }}
    >
      {step}. {label}
    </Typography>
  );
}

export default function ConnectionGraphHorizontal({
  isBackendConnected,
  websocketTestStatus,
  isHardwareConnected,
  hasRunConnectionTest,
  hasWebsocketPort,
}) {
  // Status für Farben
  const wsSuccess = websocketTestStatus === "success";
  const wsFailed =
    websocketTestStatus === "failed" || websocketTestStatus === "timeout";

  const neutralColor = "#bdbdbd";
  const successColor = "#2e7d32";
  const failedColor = "#d32f2f";

  const apiColor = !hasRunConnectionTest
    ? neutralColor
    : isBackendConnected
      ? successColor
      : failedColor;

  const wsColor =
    !hasRunConnectionTest || !hasWebsocketPort
      ? neutralColor
      : wsSuccess
        ? successColor
        : wsFailed
          ? failedColor
          : neutralColor;

  const espColor = !hasRunConnectionTest
    ? neutralColor
    : isHardwareConnected
      ? successColor
      : failedColor;

  const apiArrowColor = apiColor;
  const wsArrowColor = wsColor;
  const espArrowColor = !hasRunConnectionTest
    ? neutralColor
    : !isBackendConnected
      ? failedColor
      : espColor;

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        mb: 3,
        justifyContent: "center",
        gap: 4,
      }}
    >
      {/* Step 1: Frontend */}
      <Box
        sx={{ display: "flex", flexDirection: "column", alignItems: "center" }}
      >
        <StepLabel step={1} label="Frontend" />
        <Tooltip title="The ImSwitch-Frontend (Browser)">
          <Computer sx={{ fontSize: 36, color: "#1976d2" }} />
        </Tooltip>
      </Box>

      {/* Branches from Frontend */}
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          gap: 2,
          ml: 2,
        }}
      >
        {/* Upper branch: Frontend -> API -> ESP */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span
            style={{
              display: "flex",
              alignItems: "center",
              color: apiArrowColor,
              fontSize: 24,
            }}
          >
            &#8599;
          </span>

          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <StepLabel step={2} label="API" />
            <Tooltip title="Backend-Server (API)">
              <Storage sx={{ fontSize: 36, color: apiColor }} />
            </Tooltip>
          </Box>

          <span
            style={{
              display: "flex",
              alignItems: "center",
              color: espArrowColor,
              fontSize: 28,
            }}
          >
            &#8594;
          </span>

          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <StepLabel step={3} label="ESP32" />
            <Tooltip title="ESP32 Hardware (Controller)">
              <Memory sx={{ fontSize: 32, color: espColor }} />
            </Tooltip>
          </Box>
        </Box>

        {/* Lower branch: Frontend -> WebSocket */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span
            style={{
              display: "flex",
              alignItems: "center",
              color: wsArrowColor,
              fontSize: 24,
            }}
          >
            &#8600;
          </span>

          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <StepLabel step={4} label="WebSocket" />
            <Tooltip title="WebSocket-connection (Live communication)">
              <Lan sx={{ fontSize: 32, color: wsColor }} />
            </Tooltip>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
