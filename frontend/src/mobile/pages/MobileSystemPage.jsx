// Kiosk system page: ESP32 controller state + basic actions, a reduced
// one-button master firmware update (USB, auto-detect defaults; the full
// wizard stays in the desktop UI), storage overview and version info.
import { useCallback, useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  LinearProgress,
  Paper,
  Switch,
  Typography,
} from "@mui/material";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import UsbRoundedIcon from "@mui/icons-material/UsbRounded";
import SystemUpdateAltRoundedIcon from "@mui/icons-material/SystemUpdateAltRounded";
import PowerRoundedIcon from "@mui/icons-material/PowerRounded";
import OpenInNewRoundedIcon from "@mui/icons-material/OpenInNewRounded";
import { enqueueSnackbar } from "notistack";

import MobilePage from "../components/MobilePage";
import SectionLabel from "../components/SectionLabel";
import ConnectionDot from "../components/ConnectionDot";
import ConfirmDialog from "../components/ConfirmDialog";
import PlaceholderImage from "../components/PlaceholderImage";
import { getUc2State } from "../../state/slices/UC2Slice";
import { getStorageState, setStorageSnapshot } from "../../state/slices/StorageSlice";
import { getUsbFlashState } from "../../state/slices/usbFlashSlice";
import { APP_VERSION } from "../../version";
import { exitMobileUI } from "../mobileRoutes";
import apiUC2ConfigControllerGetFirmwareInfo from "../../backendapi/apiUC2ConfigControllerGetFirmwareInfo";
import apiUC2ConfigControllerGetBoardTemperature from "../../backendapi/apiUC2ConfigControllerGetBoardTemperature";
import apiUC2ConfigControllerSetBusPower from "../../backendapi/apiUC2ConfigControllerSetBusPower";
import apiUC2ConfigControllerReconnect from "../../backendapi/apiUC2ConfigControllerReconnect";
import apiUC2ConfigControllerEspRestart from "../../backendapi/apiUC2ConfigControllerEspRestart";
import apiUC2ConfigControllerRestartImSwitch from "../../backendapi/apiUC2ConfigControllerRestartImSwitch";
import apiUC2ConfigControllerFlashMasterFirmwareUSB from "../../backendapi/apiUC2ConfigControllerFlashMasterFirmwareUSB";
import apiUC2ConfigControllerCancelUSBFlash from "../../backendapi/apiUC2ConfigControllerCancelUSBFlash";
import apiStorageControllerGetStorageStatus from "../../backendapi/apiStorageControllerGetStorageStatus";
import apiVersion from "../../backendapi/apiVersion";

const InfoRow = ({ label, value }) => (
  <Box sx={{ display: "flex", justifyContent: "space-between", gap: 2, py: 0.5 }}>
    <Typography variant="body2" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="body2" sx={{ fontWeight: 700, textAlign: "right" }} noWrap>
      {value ?? "—"}
    </Typography>
  </Box>
);

const FLASH_ACTIVE_STATES = ["disconnecting", "downloading", "flashing", "reconnecting"];

const MobileSystemPage = () => {
  const dispatch = useDispatch();
  const uc2State = useSelector(getUc2State);
  const storageState = useSelector(getStorageState);
  const usbFlashState = useSelector(getUsbFlashState);

  const [firmwareInfo, setFirmwareInfo] = useState(null);
  const [boardTemp, setBoardTemp] = useState(null);
  const [backendVersion, setBackendVersion] = useState(null);
  const [confirm, setConfirm] = useState(null); // "espRestart" | "flash" | "restartImSwitch" | "busOff"
  const [flashRequested, setFlashRequested] = useState(false);

  const refreshFirmwareInfo = useCallback(() => {
    apiUC2ConfigControllerGetFirmwareInfo()
      .then(setFirmwareInfo)
      .catch(() => setFirmwareInfo(null));
  }, []);

  useEffect(() => {
    refreshFirmwareInfo();
    apiVersion()
      .then((data) => setBackendVersion(data?.version || String(data)))
      .catch(() => {});
    if (!storageState.hasReceivedSnapshot) {
      apiStorageControllerGetStorageStatus()
        .then((snapshot) => dispatch(setStorageSnapshot(snapshot)))
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const poll = () => {
      apiUC2ConfigControllerGetBoardTemperature().then(setBoardTemp).catch(() => {});
    };
    poll();
    const timer = setInterval(poll, 10000);
    return () => clearInterval(timer);
  }, []);

  const flashActive =
    flashRequested || FLASH_ACTIVE_STATES.includes(usbFlashState.flashStatus);

  // Refresh the firmware identity once a flash cycle ends. Only react to
  // *transitions* — the slice may still hold "success" from an earlier flash
  // when this page mounts.
  const prevFlashStatusRef = useRef(usbFlashState.flashStatus);
  useEffect(() => {
    if (prevFlashStatusRef.current === usbFlashState.flashStatus) return;
    prevFlashStatusRef.current = usbFlashState.flashStatus;
    if (usbFlashState.flashStatus === "success") {
      setFlashRequested(false);
      refreshFirmwareInfo();
      enqueueSnackbar("Firmware updated", { variant: "success" });
    } else if (usbFlashState.flashStatus === "failed") {
      setFlashRequested(false);
      enqueueSnackbar("Firmware update failed", { variant: "error" });
    }
  }, [usbFlashState.flashStatus, refreshFirmwareInfo]);

  const busPowerOn = uc2State.busPower === 1 || uc2State.busPower === true;

  const handleBusPowerChange = (event) => {
    const enable = event.target.checked;
    if (!enable) {
      setConfirm("busOff");
      return;
    }
    apiUC2ConfigControllerSetBusPower(true).catch(() =>
      enqueueSnackbar("Could not switch bus power", { variant: "error" }),
    );
  };

  const actions = {
    espRestart: () => {
      apiUC2ConfigControllerEspRestart()
        .then(() => enqueueSnackbar("ESP32 restarting…", { variant: "info" }))
        .catch(() => enqueueSnackbar("Restart failed", { variant: "error" }));
    },
    busOff: () => {
      apiUC2ConfigControllerSetBusPower(false).catch(() =>
        enqueueSnackbar("Could not switch bus power", { variant: "error" }),
      );
    },
    flash: () => {
      setFlashRequested(true);
      // Defaults: auto-detect port + newest master firmware on the server.
      apiUC2ConfigControllerFlashMasterFirmwareUSB().catch(() => {
        setFlashRequested(false);
        enqueueSnackbar("Could not start the firmware update", { variant: "error" });
      });
    },
    restartImSwitch: () => {
      apiUC2ConfigControllerRestartImSwitch()
        .then(() =>
          enqueueSnackbar("Backend restarting — this screen will reconnect", {
            variant: "info",
          }),
        )
        .catch(() => enqueueSnackbar("Restart request failed", { variant: "error" }));
    },
  };

  const confirmProps = {
    espRestart: {
      title: "Restart ESP32?",
      text: "The controller board reboots (~3 s). Any running motion stops.",
      confirmLabel: "Restart",
      danger: true,
    },
    busOff: {
      title: "Turn bus power off?",
      text: "All CAN-bus modules (motors, lasers) lose power until it is switched back on.",
      confirmLabel: "Turn off",
      danger: true,
    },
    flash: {
      title: "Update master firmware?",
      text: "Flashes the newest master firmware over USB (2–3 min). Do not power off the microscope while the update runs.",
      confirmLabel: "Update",
      danger: true,
    },
    restartImSwitch: {
      title: "Restart the backend?",
      text: "ImSwitch restarts and every stream disconnects for about half a minute.",
      confirmLabel: "Restart backend",
      danger: true,
    },
  };

  const devices = storageState.status?.storage_devices || [];
  const pcbTemp = boardTemp && typeof boardTemp.pcb === "number" ? boardTemp.pcb : null;
  const airTemp = boardTemp && typeof boardTemp.air === "number" ? boardTemp.air : null;

  return (
    <MobilePage title="System" subtitle="Controller, firmware, storage and versions">
      {uc2State.emergencyActive && (
        <Alert severity="error" sx={{ mb: 2.5 }}>
          Emergency stop is active — motion is blocked until it is released.
        </Alert>
      )}

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: 2.5,
          alignItems: "start",
        }}
      >
        {/* ESP32 controller */}
        <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3 }}>
          <SectionLabel>Controller (ESP32)</SectionLabel>
          <Box sx={{ mb: 1.5 }}>
            <ConnectionDot ok={uc2State.uc2Connected} labelOn="Connected" labelOff="Not connected" />
          </Box>
          <InfoRow label="Firmware" value={firmwareInfo?.version} />
          <InfoRow label="Build date" value={firmwareInfo?.date} />
          <InfoRow label="Serial port" value={firmwareInfo?.serialport} />
          <InfoRow
            label="Board temperature"
            value={pcbTemp != null ? `${pcbTemp.toFixed(1)} °C${airTemp != null ? ` / air ${airTemp.toFixed(1)} °C` : ""}` : null}
          />
          <Divider sx={{ my: 1.5 }} />
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
            <PowerRoundedIcon sx={{ color: "text.secondary" }} />
            <Typography sx={{ flex: 1 }}>Bus power</Typography>
            {uc2State.busPower === null && <Chip size="small" label="unknown" variant="outlined" />}
            <Switch checked={busPowerOn} onChange={handleBusPowerChange} />
          </Box>
          <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
            <Button
              variant="outlined"
              startIcon={<UsbRoundedIcon />}
              onClick={() =>
                apiUC2ConfigControllerReconnect()
                  .then(() => enqueueSnackbar("Reconnecting to ESP32…", { variant: "info" }))
                  .catch(() => enqueueSnackbar("Reconnect failed", { variant: "error" }))
              }
              sx={{ flex: 1 }}
            >
              Reconnect
            </Button>
            <Button
              variant="outlined"
              color="warning"
              startIcon={<RestartAltRoundedIcon />}
              onClick={() => setConfirm("espRestart")}
              sx={{ flex: 1 }}
            >
              Restart ESP32
            </Button>
          </Box>
        </Paper>

        {/* Firmware update */}
        <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3 }}>
          <SectionLabel>Firmware update</SectionLabel>
          {flashActive ? (
            <>
              <Typography variant="body2" sx={{ mb: 1 }}>
                {usbFlashState.flashMessage || `Status: ${usbFlashState.flashStatus}`}
              </Typography>
              <LinearProgress
                variant={usbFlashState.flashProgress > 0 ? "determinate" : "indeterminate"}
                value={usbFlashState.flashProgress || 0}
                sx={{ height: 12, borderRadius: 6, mb: 1.5 }}
              />
              <Button
                color="warning"
                variant="outlined"
                fullWidth
                onClick={() =>
                  apiUC2ConfigControllerCancelUSBFlash()
                    .then(() => setFlashRequested(false))
                    .catch(() => enqueueSnackbar("Cancel failed", { variant: "error" }))
                }
              >
                Cancel update
              </Button>
            </>
          ) : (
            <>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                Installs the newest master firmware over the built-in USB
                connection. Advanced options (specific files, CAN modules) are
                in the full interface.
              </Typography>
              <PlaceholderImage label="Illustration: firmware update" height={90} sx={{ mb: 1.5 }} />
              <Button
                variant="contained"
                fullWidth
                size="large"
                startIcon={<SystemUpdateAltRoundedIcon />}
                onClick={() => setConfirm("flash")}
              >
                Update master firmware
              </Button>
            </>
          )}
        </Paper>

        {/* Storage */}
        <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3 }}>
          <SectionLabel>Storage</SectionLabel>
          {devices.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No storage information available.
            </Typography>
          )}
          {devices.map((device) => {
            const usage = device.usage || {};
            const percent = typeof usage.percent_used === "number" ? usage.percent_used : null;
            const freeGb = typeof usage.free === "number" ? usage.free / 1024 ** 3 : null;
            return (
              <Box key={device.path || device.label} sx={{ mb: 1.75 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                  <Typography variant="body2" sx={{ flex: 1, fontWeight: 700 }} noWrap>
                    {device.label}
                  </Typography>
                  {device.is_active && <Chip size="small" color="primary" label="active" />}
                  {!device.is_available && <Chip size="small" variant="outlined" label="missing" />}
                </Box>
                {percent != null && (
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, percent)}
                    color={percent > 90 ? "error" : "primary"}
                    sx={{ height: 10, borderRadius: 5, mb: 0.5 }}
                  />
                )}
                <Typography variant="caption" color="text.secondary">
                  {freeGb != null ? `${freeGb.toFixed(1)} GB free` : "usage unknown"}
                </Typography>
              </Box>
            );
          })}
        </Paper>

        {/* About */}
        <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3 }}>
          <SectionLabel>About</SectionLabel>
          <InfoRow label="Backend" value={backendVersion} />
          <InfoRow label="Frontend" value={APP_VERSION} />
          <InfoRow label="Controller firmware" value={firmwareInfo?.name} />
          <Divider sx={{ my: 1.5 }} />
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
            <Button
              variant="outlined"
              startIcon={<OpenInNewRoundedIcon />}
              onClick={exitMobileUI}
            >
              Open full interface
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<RestartAltRoundedIcon />}
              onClick={() => setConfirm("restartImSwitch")}
            >
              Restart backend
            </Button>
          </Box>
        </Paper>
      </Box>

      <ConfirmDialog
        open={confirm !== null}
        {...(confirmProps[confirm] || { title: "", text: "" })}
        onCancel={() => setConfirm(null)}
        onConfirm={() => {
          const action = actions[confirm];
          setConfirm(null);
          if (action) action();
        }}
      />
    </MobilePage>
  );
};

export default MobileSystemPage;
