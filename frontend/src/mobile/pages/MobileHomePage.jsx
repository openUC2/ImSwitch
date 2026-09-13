// Kiosk home: live 3D render of the FRAME (static, driven by real stage
// positions) plus a Formlabs-style status column and quick-nav shortcuts.
import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Box, Chip, LinearProgress, Paper, Typography, Divider } from "@mui/material";
import OpenWithRoundedIcon from "@mui/icons-material/OpenWithRounded";
import FlareRoundedIcon from "@mui/icons-material/FlareRounded";
import VideocamRoundedIcon from "@mui/icons-material/VideocamRounded";
import CenterFocusStrongRoundedIcon from "@mui/icons-material/CenterFocusStrongRounded";
import SdStorageRoundedIcon from "@mui/icons-material/SdStorageRounded";
import ThermostatRoundedIcon from "@mui/icons-material/ThermostatRounded";

import Frame3DViewer from "../../components/Frame3DViewer";
import * as positionSlice from "../../state/slices/PositionSlice";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice";
import * as frame3DViewerSlice from "../../state/slices/Frame3DViewerSlice";
import { getUc2State } from "../../state/slices/UC2Slice";
import { getStorageState } from "../../state/slices/StorageSlice";
import { getHomingState } from "../../state/slices/HomingSlice";
import apiUC2ConfigControllerGetMicroscopeStandName from "../../backendapi/apiUC2ConfigControllerGetMicroscopeStandName";
import apiUC2ConfigControllerGetBoardTemperature from "../../backendapi/apiUC2ConfigControllerGetBoardTemperature";
import ConnectionDot from "../components/ConnectionDot";
import TouchButton from "../components/TouchButton";
import { kioskColors } from "../mobileTheme";
import { MOBILE_PAGES } from "../mobileRoutes";

const formatUm = (v) =>
  typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(0)}` : "—";

const StatusRow = ({ label, children }) => (
  <Box sx={{ py: 1.25 }}>
    <Typography
      variant="caption"
      sx={{ color: "text.secondary", fontWeight: 700, letterSpacing: "0.5px" }}
    >
      {label}
    </Typography>
    <Box sx={{ mt: 0.25 }}>{children}</Box>
  </Box>
);

const MobileHomePage = ({ navigate }) => {
  const positionState = useSelector(positionSlice.getPositionState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const frame3DState = useSelector(frame3DViewerSlice.getFrame3DViewerState);
  const uc2State = useSelector(getUc2State);
  const storageState = useSelector(getStorageState);
  const homingState = useSelector(getHomingState);

  const [standName, setStandName] = useState("openUC2 FRAME");
  const [boardTemp, setBoardTemp] = useState(null);

  useEffect(() => {
    apiUC2ConfigControllerGetMicroscopeStandName()
      .then((data) => {
        if (data?.name) setStandName(data.name);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      apiUC2ConfigControllerGetBoardTemperature()
        .then((data) => {
          if (!cancelled) setBoardTemp(data);
        })
        .catch(() => {});
    };
    poll();
    const timer = setInterval(poll, 15000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // One status word, Formlabs-style, most-severe-first.
  let statusLabel = "IDLE";
  let statusColor = "primary";
  if (!uc2State.backendConnected) {
    statusLabel = "OFFLINE";
    statusColor = "error";
  } else if (uc2State.emergencyActive) {
    statusLabel = "E-STOP";
    statusColor = "error";
  } else if (homingState.active) {
    statusLabel = "HOMING";
    statusColor = "warning";
  }

  const usage = storageState.status?.active_device?.usage || null;
  const freeGb =
    usage && typeof usage.free === "number" ? usage.free / 1024 ** 3 : null;
  const percentUsed =
    usage && typeof usage.percent_used === "number" ? usage.percent_used : null;

  const objectiveLabel =
    objectiveState.currentObjective != null
      ? objectiveState.availableObjectivesNames?.[
          objectiveState.currentObjective
        ] || `Slot ${objectiveState.currentObjective + 1}`
      : "Unknown";

  const pcbTemp =
    boardTemp && typeof boardTemp.pcb === "number" ? boardTemp.pcb : null;

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column", minWidth: 0 }}>
      {/* Header: stand name + one-word state */}
      <Box
        sx={{
          px: 3,
          py: 2,
          display: "flex",
          alignItems: "center",
          gap: 2,
          flexShrink: 0,
        }}
      >
        <Typography variant="h4" sx={{ flex: 1, textTransform: "uppercase" }} noWrap>
          {standName}
        </Typography>
        <Chip
          label={statusLabel}
          color={statusColor}
          variant={statusLabel === "IDLE" ? "outlined" : "filled"}
          sx={{ fontSize: "1rem", px: 1, height: 40 }}
        />
      </Box>

      <Box
        sx={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          gap: 2.5,
          px: 3,
          pb: 2,
        }}
      >
        {/* Live 3D render — static (not draggable), tracks real positions */}
        <Paper
          variant="outlined"
          sx={{
            flex: 1.5,
            minWidth: 0,
            borderRadius: 3,
            overflow: "hidden",
            position: "relative",
          }}
        >
          <Frame3DViewer
            positions={{
              x: positionState.x,
              y: positionState.y,
              z: positionState.z,
              a: positionState.a,
            }}
            axisConfig={frame3DState.axisConfig}
            visibility={frame3DState.visibility}
            width="100%"
            height="100%"
            interactive={false}
            showAxes={false}
            background={kioskColors.surface}
            sx={{ border: "none", borderRadius: 0 }}
          />
          <Typography
            variant="caption"
            sx={{
              position: "absolute",
              left: 12,
              bottom: 8,
              color: "text.disabled",
            }}
          >
            Live digital twin — positions update in real time
          </Typography>
        </Paper>

        {/* Status column */}
        <Paper
          variant="outlined"
          sx={{
            width: 300,
            flexShrink: 0,
            borderRadius: 3,
            px: 2.5,
            py: 1.5,
            overflowY: "auto",
          }}
        >
          <StatusRow label="CONNECTION">
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
              <ConnectionDot ok={uc2State.backendConnected} label="Backend" />
              <ConnectionDot ok={uc2State.uc2Connected} label="ESP32" />
            </Box>
          </StatusRow>
          <Divider />
          <StatusRow label="OBJECTIVE">
            <Typography variant="body1" sx={{ fontWeight: 700 }}>
              {objectiveLabel}
              {objectiveState.magnification
                ? ` · ${objectiveState.magnification}×`
                : ""}
            </Typography>
          </StatusRow>
          <Divider />
          <StatusRow label="STAGE POSITION (µM)">
            <Typography variant="body1" sx={{ fontVariantNumeric: "tabular-nums" }}>
              X {formatUm(positionState.x)} · Y {formatUm(positionState.y)} · Z{" "}
              {formatUm(positionState.z)}
            </Typography>
          </StatusRow>
          <Divider />
          <StatusRow label="STORAGE">
            {percentUsed != null ? (
              <>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, percentUsed)}
                  color={percentUsed > 90 ? "error" : "primary"}
                  sx={{ height: 10, borderRadius: 5, mb: 0.5 }}
                />
                <Typography variant="body2" color="text.secondary">
                  {freeGb != null ? `${freeGb.toFixed(1)} GB free` : `${percentUsed.toFixed(0)}% used`}
                </Typography>
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No storage info yet
              </Typography>
            )}
          </StatusRow>
          <Divider />
          <StatusRow label="BOARD TEMPERATURE">
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <ThermostatRoundedIcon fontSize="small" sx={{ color: "text.secondary" }} />
              <Typography variant="body1" sx={{ fontVariantNumeric: "tabular-nums" }}>
                {pcbTemp != null ? `${pcbTemp.toFixed(1)} °C` : "—"}
              </Typography>
            </Box>
          </StatusRow>
        </Paper>
      </Box>

      {/* Quick actions */}
      <Box
        sx={{
          display: "flex",
          gap: 2,
          px: 3,
          pb: 2.5,
          flexShrink: 0,
        }}
      >
        <TouchButton
          icon={<OpenWithRoundedIcon />}
          label="Stage"
          onClick={() => navigate(MOBILE_PAGES.STAGE)}
          sx={{ flex: 1, minHeight: 76 }}
        />
        <TouchButton
          icon={<FlareRoundedIcon />}
          label="Lasers"
          onClick={() => navigate(MOBILE_PAGES.LASERS)}
          sx={{ flex: 1, minHeight: 76 }}
        />
        <TouchButton
          icon={<VideocamRoundedIcon />}
          label="Camera"
          onClick={() => navigate(MOBILE_PAGES.CAMERA)}
          sx={{ flex: 1, minHeight: 76 }}
        />
        <TouchButton
          icon={<CenterFocusStrongRoundedIcon />}
          label="Objective"
          onClick={() => navigate(MOBILE_PAGES.OBJECTIVE)}
          sx={{ flex: 1, minHeight: 76 }}
        />
        <TouchButton
          icon={<SdStorageRoundedIcon />}
          label="System"
          onClick={() => navigate(MOBILE_PAGES.SYSTEM)}
          sx={{ flex: 1, minHeight: 76 }}
        />
      </Box>
    </Box>
  );
};

export default MobileHomePage;
