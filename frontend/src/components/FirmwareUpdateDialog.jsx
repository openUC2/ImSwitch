import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import apiUC2ConfigControllerCheckFirmwareUpdates from "../backendapi/apiUC2ConfigControllerCheckFirmwareUpdates";
import apiUC2ConfigControllerStartFirmwareUpdate from "../backendapi/apiUC2ConfigControllerStartFirmwareUpdate";
import apiUC2ConfigControllerGetFirmwareUpdateStatus from "../backendapi/apiUC2ConfigControllerGetFirmwareUpdateStatus";
import apiUC2ConfigControllerCancelFirmwareUpdate from "../backendapi/apiUC2ConfigControllerCancelFirmwareUpdate";
import { FIRMWARE_STATUS } from "./firmwareStatus";
import { getCanOtaState } from "../state/slices/canOtaSlice";
import { getUsbFlashState } from "../state/slices/usbFlashSlice";

const mono = { fontFamily: "monospace", fontSize: "0.8rem", wordBreak: "break-all" };
const POLL_MS = 1500;

const rowKey = (d) => (d.connection === "usb" ? "usb" : `can-${d.canId}`);
const selectable = (d) => Boolean(d.available_version) && d.update_status !== "unreachable";
const boardLabel = (d) =>
  d.connection === "usb" ? `${d.deviceTypeStr} (USB master)` : `${d.deviceTypeStr} ${d.canId}`;

const STEP = {
  pending: { label: "Waiting", color: "text.secondary" },
  running: { label: "Updating", color: "text.primary" },
  done: { label: "Done", color: "success.main" },
  failed: { label: "Failed", color: "error.main" },
  skipped: { label: "Not updated", color: "text.secondary" },
};
const FINAL = {
  success: "success",
  failed: "error",
  cancelled: "warning",
};

// Update outdated boards in one go: CAN nodes one at a time, the USB master
// last. The backend refuses while anything runs or moves and counts a board
// as done only when it reports the new version after the reboot.
const FirmwareUpdateDialog = ({ open, onClose, initialCheck = null }) => {
  const [check, setCheck] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [reasons, setReasons] = useState([]);
  const [run, setRun] = useState(null); // getFirmwareUpdateStatus once started
  const canOta = useSelector(getCanOtaState);
  const usbFlash = useSelector(getUsbFlashState);

  const applyCheck = (result) => {
    setCheck(result);
    setSelected(
      new Set(
        (result?.devices || [])
          .filter((d) => selectable(d) && d.update_status === "update_available")
          .map(rowKey),
      ),
    );
  };

  const loadCheck = async () => {
    setLoading(true);
    setError(null);
    try {
      applyCheck(await apiUC2ConfigControllerCheckFirmwareUpdates());
    } catch (e) {
      setError(`Version check failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // On open: show a running update, else the given (fresh) check, else check now.
  useEffect(() => {
    if (!open) return;
    setReasons([]);
    setError(null);
    apiUC2ConfigControllerGetFirmwareUpdateStatus()
      .then((status) => {
        if (status?.state === "running") {
          setRun(status);
        } else {
          setRun(null);
          if (initialCheck) applyCheck(initialCheck);
          else loadCheck();
        }
      })
      .catch(() => loadCheck());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const running = run?.state === "running";
  useEffect(() => {
    if (!open || !running) return undefined;
    const id = setInterval(async () => {
      try {
        setRun(await apiUC2ConfigControllerGetFirmwareUpdateStatus());
      } catch {
        // the backend stays reachable during a USB flash; retry next tick
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [open, running]);

  const toggle = (key) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const start = async () => {
    const chosen = (check?.devices || []).filter((d) => selected.has(rowKey(d)));
    const canIds = chosen.filter((d) => d.connection === "can").map((d) => d.canId);
    const includeMaster = chosen.some((d) => d.connection === "usb");
    setReasons([]);
    setError(null);
    try {
      const result = await apiUC2ConfigControllerStartFirmwareUpdate(canIds, includeMaster);
      if (result.status === "started") {
        setRun({ state: "running", steps: result.steps, message: "Starting…" });
      } else {
        setReasons(result.reasons || [result.message || "The update was refused."]);
      }
    } catch (e) {
      setError(`Could not start the update: ${e.message}`);
    }
  };

  const liveProgress = (step) => {
    if (step.status !== "running") return null;
    if (step.connection === "usb") {
      return { progress: usbFlash.flashProgress, message: usbFlash.flashMessage };
    }
    return canOta.updateProgress?.[step.canId] || null;
  };

  const renderSelection = () => (
    <>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        Server firmware:{" "}
        <Box component="span" sx={mono}>
          {check?.server_version || "no version.json"}
        </Box>
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell padding="checkbox" />
            <TableCell>Board</TableCell>
            <TableCell>Installed</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {(check?.devices || []).map((d) => {
            const status = FIRMWARE_STATUS[d.update_status] || FIRMWARE_STATUS.unknown;
            return (
              <TableRow key={rowKey(d)}>
                <TableCell padding="checkbox">
                  <Checkbox
                    size="small"
                    checked={selected.has(rowKey(d))}
                    disabled={!selectable(d)}
                    onChange={() => toggle(rowKey(d))}
                    inputProps={{ "aria-label": `update ${boardLabel(d)}` }}
                  />
                </TableCell>
                <TableCell>
                  {boardLabel(d)}
                  <Typography variant="caption" color="text.secondary" sx={{ display: "block", ...mono }}>
                    {d.filename || "no image"}
                  </Typography>
                </TableCell>
                <TableCell sx={mono}>{d.installed_version || "—"}</TableCell>
                <TableCell sx={{ color: status.color, whiteSpace: "nowrap" }}>{status.label}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      <Alert severity="info" sx={{ mt: 2 }}>
        Lasers are switched off first. CAN boards are updated one at a time, the USB master last
        (ImSwitch reconnects afterwards). Updated boards restart; motors lose their position and
        need homing. The update stops at the first board that fails. Keep power and USB connected.
      </Alert>
      {reasons.length > 0 && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Not started:
          <Box component="ul" sx={{ m: 0, pl: 2 }}>
            {reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </Box>
        </Alert>
      )}
    </>
  );

  const renderRun = () => (
    <>
      {running && <LinearProgress sx={{ mb: 2 }} />}
      {!running && run?.state && FINAL[run.state] && (
        <Alert severity={FINAL[run.state]} sx={{ mb: 2 }}>
          {run.message}
        </Alert>
      )}
      {running && (
        <Typography variant="body2" sx={{ mb: 1 }}>
          {run.message}
        </Typography>
      )}
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Board</TableCell>
            <TableCell>From → to</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {(run?.steps || []).map((step) => {
            const live = liveProgress(step);
            const look = STEP[step.status] || STEP.pending;
            return (
              <TableRow key={rowKey(step)}>
                <TableCell>{boardLabel(step)}</TableCell>
                <TableCell sx={mono}>
                  {step.from_version || "—"} → {step.to_version}
                </TableCell>
                <TableCell sx={{ minWidth: 180 }}>
                  <Typography variant="body2" sx={{ color: look.color }}>
                    {look.label}
                  </Typography>
                  {live && (
                    <>
                      <LinearProgress variant="determinate" value={live.progress || 0} sx={{ my: 0.5 }} />
                      <Typography variant="caption" color="text.secondary">
                        {live.message}
                      </Typography>
                    </>
                  )}
                  {!live && step.message && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                      {step.message}
                    </Typography>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      {run?.homing_required && !running && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Motor boards restarted and lost their position. Home the stage before moving it.
        </Alert>
      )}
    </>
  );

  const nSelected = selected.size;
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Firmware update</DialogTitle>
      <DialogContent dividers>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        {loading ? (
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, py: 3 }}>
            <CircularProgress size={20} />
            <Typography variant="body2">Reading the boards' firmware versions…</Typography>
          </Box>
        ) : run ? (
          renderRun()
        ) : (
          check && renderSelection()
        )}
      </DialogContent>
      <DialogActions>
        {running && (
          <Button color="warning" onClick={() => apiUC2ConfigControllerCancelFirmwareUpdate()}>
            Stop after current board
          </Button>
        )}
        {run && !running && (
          <Button
            onClick={() => {
              setRun(null);
              loadCheck();
            }}
          >
            Check again
          </Button>
        )}
        <Button onClick={onClose}>{running ? "Hide (keeps running)" : "Close"}</Button>
        {!run && (
          <Button variant="contained" onClick={start} disabled={loading || nSelected === 0}>
            Update {nSelected} board{nSelected === 1 ? "" : "s"}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default FirmwareUpdateDialog;
