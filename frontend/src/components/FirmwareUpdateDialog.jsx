import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  ButtonBase,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { Hub as CanIcon, Usb as UsbIcon } from "@mui/icons-material";
import apiUC2ConfigControllerCheckFirmwareUpdates from "../backendapi/apiUC2ConfigControllerCheckFirmwareUpdates";
import apiUC2ConfigControllerStartFirmwareUpdate from "../backendapi/apiUC2ConfigControllerStartFirmwareUpdate";
import apiUC2ConfigControllerGetFirmwareUpdateStatus from "../backendapi/apiUC2ConfigControllerGetFirmwareUpdateStatus";
import apiUC2ConfigControllerCancelFirmwareUpdate from "../backendapi/apiUC2ConfigControllerCancelFirmwareUpdate";
import apiUC2ConfigControllerSetOTAFirmwareServer from "../backendapi/apiUC2ConfigControllerSetOTAFirmwareServer";
import apiUC2ConfigControllerReassignCANId from "../backendapi/apiUC2ConfigControllerReassignCANId";
import { FIRMWARE_STATUS } from "./firmwareStatus";
import UsbFlashWizard from "./UsbFlashWizard";
import { getCanOtaState } from "../state/slices/canOtaSlice";
import { getUsbFlashState } from "../state/slices/usbFlashSlice";

const mono = { fontFamily: "monospace", fontSize: "0.8rem", wordBreak: "break-all" };
const POLL_MS = 1500;
const METHOD_TITLE = { can: "over the CAN bus", usb: "over a USB cable" };

const rowKey = (d) => (d.connection === "usb" ? "usb" : `can-${d.canId}`);
const selectable = (d) =>
  Boolean(d.filename) && !["unreachable", "no_firmware"].includes(d.update_status);
const boardLabel = (d) =>
  d.connection === "usb" ? `${d.deviceTypeStr} (USB master)` : `${d.deviceTypeStr} ${d.canId}`;

const STEP = {
  pending: { label: "Waiting", color: "text.secondary" },
  running: { label: "Updating", color: "text.primary" },
  done: { label: "Done", color: "success.main" },
  failed: { label: "Failed", color: "error.main" },
  skipped: { label: "Not updated", color: "text.secondary" },
};
const FINAL = { success: "success", failed: "error", cancelled: "warning" };

// "Update firmware": over the CAN bus (every board, through the master
// ImSwitch is connected to; the backend compares versions and verifies the
// result) or over a USB cable (one board on its own port: flash, CAN
// address, test).
const FirmwareUpdateDialog = ({ open, onClose, initialCheck = null, initialMethod = null,
                                canAvailable = true }) => {
  const [method, setMethod] = useState(initialMethod);
  const flashing = useSelector(getUsbFlashState).isFlashing;

  useEffect(() => {
    if (!open) return;
    setMethod(initialMethod);
    // A CAN update keeps running in the backend while the dialog is closed.
    apiUC2ConfigControllerGetFirmwareUpdateStatus()
      .then((status) => status?.state === "running" && setMethod("can"))
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  return (
    <Dialog open={open} onClose={flashing ? undefined : onClose} maxWidth="md" fullWidth
            disableEscapeKeyDown={flashing}>
      <DialogTitle>
        Update firmware{METHOD_TITLE[method] ? ` ${METHOD_TITLE[method]}` : ""}
      </DialogTitle>
      {method === null && (
        <MethodChoice onChoose={setMethod} onClose={onClose} canAvailable={canAvailable} />
      )}
      {method === "can" && (
        <CanUpdate initialCheck={initialCheck} onBack={() => setMethod(null)} onClose={onClose} />
      )}
      {method === "usb" && (
        <DialogContent dividers>
          {!flashing && (
            <Button size="small" sx={{ mb: 1 }} onClick={() => setMethod(null)}>
              ← Other method
            </Button>
          )}
          <UsbFlashWizard open onClose={onClose} />
        </DialogContent>
      )}
    </Dialog>
  );
};

const METHODS = [
  {
    key: "can",
    icon: <CanIcon />,
    title: "Over the CAN bus",
    text: "Every board on the bus, through the master ImSwitch is connected to — the master " +
          "itself last, over its USB link. Versions are compared with the firmware server and " +
          "checked after the update. About a minute per board.",
  },
  {
    key: "usb",
    icon: <UsbIcon />,
    title: "Over a USB cable",
    text: "One board plugged in by its own USB cable: flash any image, assign its CAN address, " +
          "test it. For new boards, or boards that no longer answer on the bus.",
  },
];

const MethodChoice = ({ onChoose, onClose, canAvailable }) => (
  <>
    <DialogContent dividers>
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" }, gap: 2 }}>
        {METHODS.map((m) => (
          <ButtonBase
            key={m.key}
            onClick={() => onChoose(m.key)}
            disabled={m.key === "can" && !canAvailable}
            sx={{
              display: "block", textAlign: "left", p: 2, borderRadius: 1,
              border: 1, borderColor: "divider", "&:hover": { borderColor: "primary.main" },
              "&.Mui-disabled": { opacity: 0.5 },
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
              {m.icon}
              <Typography variant="subtitle1">{m.title}</Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              {m.key === "can" && !canAvailable ? "Needs a connected master board." : m.text}
            </Typography>
          </ButtonBase>
        ))}
      </Box>
    </DialogContent>
    <DialogActions>
      <Button onClick={onClose}>Close</Button>
    </DialogActions>
  </>
);

// Over the CAN bus: check every board, pick, run the backend's update
// (startFirmwareUpdate) and follow getFirmwareUpdateStatus. The state lives in
// the backend, so closing/reopening the dialog shows a running update again.
const CanUpdate = ({ initialCheck, onBack, onClose }) => {
  const [check, setCheck] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [loading, setLoading] = useState(false);
  const [deepScan, setDeepScan] = useState(false);
  const [error, setError] = useState(null);
  const [reasons, setReasons] = useState([]);
  const [run, setRun] = useState(null); // getFirmwareUpdateStatus once started
  const [serverEdit, setServerEdit] = useState(null); // URL being edited, or null
  const [reassign, setReassign] = useState(null); // { device, newId, busy, error }
  const canOta = useSelector(getCanOtaState);
  const usbFlash = useSelector(getUsbFlashState);

  const applyCheck = (result) => {
    setCheck(result);
    setSelected(new Set((result?.devices || [])
      .filter((d) => selectable(d) && d.update_status === "update_available").map(rowKey)));
  };

  const loadCheck = async (probe = deepScan) => {
    setLoading(true);
    setError(null);
    try {
      applyCheck(await apiUC2ConfigControllerCheckFirmwareUpdates(probe ? 10 : 5, probe));
    } catch (e) {
      setError(`Version check failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Show a running update, else the given (fresh) check, else check now.
  useEffect(() => {
    apiUC2ConfigControllerGetFirmwareUpdateStatus()
      .then((status) => {
        if (status?.state === "running") setRun(status);
        else if (initialCheck) applyCheck(initialCheck);
        else loadCheck();
      })
      .catch(() => loadCheck());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const running = run?.state === "running";
  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(async () => {
      try {
        setRun(await apiUC2ConfigControllerGetFirmwareUpdateStatus());
      } catch {
        // the backend stays reachable during a USB flash; retry next tick
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [running]);

  const toggle = (key) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const start = async () => {
    const chosen = (check?.devices || []).filter((d) => selected.has(rowKey(d)));
    setReasons([]);
    setError(null);
    try {
      const result = await apiUC2ConfigControllerStartFirmwareUpdate(
        chosen.filter((d) => d.connection === "can").map((d) => d.canId),
        chosen.some((d) => d.connection === "usb"),
      );
      if (result.status === "started") {
        setRun({ state: "running", steps: result.steps, message: "Starting…" });
      } else {
        setReasons(result.reasons || [result.message || "The update was refused."]);
      }
    } catch (e) {
      setError(`Could not start the update: ${e.message}`);
    }
  };

  const applyServer = async () => {
    try {
      const result = await apiUC2ConfigControllerSetOTAFirmwareServer(serverEdit);
      if (result.status !== "success") {
        setError(result.message || "The firmware server did not answer.");
        return;
      }
      setServerEdit(null);
      loadCheck();
    } catch (e) {
      setError(`Could not set the firmware server: ${e.message}`);
    }
  };

  const assignId = async () => {
    const { device, newId } = reassign;
    const id = parseInt(newId, 10);
    if (Number.isNaN(id) || id < 1 || id > 127) {
      setReassign({ ...reassign, error: "The CAN ID must be between 1 and 127." });
      return;
    }
    setReassign({ ...reassign, busy: true, error: null });
    try {
      const result = await apiUC2ConfigControllerReassignCANId(
        id, device.mac || null, device.mac ? null : device.canId);
      if (result?.status === "ok" || result?.status === "success") {
        setReassign(null);
        setTimeout(() => loadCheck(), 1500); // the node reappears after ~0.3 s
      } else {
        setReassign({ ...reassign, busy: false,
                      error: result?.error || result?.message || "Reassigning failed." });
      }
    } catch (e) {
      setReassign({ ...reassign, busy: false, error: `Reassigning failed: ${e.message}` });
    }
  };

  const liveProgress = (step) => {
    if (step.status !== "running") return null;
    if (step.connection === "usb") {
      return { progress: usbFlash.flashProgress, message: usbFlash.flashMessage };
    }
    return canOta.updateProgress?.[step.canId] || null;
  };

  const renderServer = () => (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 1 }}>
      <Typography variant="body2" color="text.secondary">Firmware server</Typography>
      {serverEdit === null ? (
        <>
          <Box component="span" sx={mono}>{check?.firmware_server}</Box>
          <Typography variant="body2" color="text.secondary">
            · <Box component="span" sx={mono}>{check?.server_version || "no version.json"}</Box>
          </Typography>
          <Button size="small" onClick={() => setServerEdit(check?.firmware_server || "")}>
            Change
          </Button>
        </>
      ) : (
        <>
          <TextField size="small" value={serverEdit} sx={{ minWidth: 320 }}
                     onChange={(e) => setServerEdit(e.target.value)}
                     placeholder="http://host.docker.internal/firmware" />
          <Button size="small" variant="outlined" onClick={applyServer}>Use</Button>
          <Button size="small" onClick={() => setServerEdit(null)}>Cancel</Button>
        </>
      )}
    </Box>
  );

  const renderSelection = () => (
    <>
      {renderServer()}
      {check?.server_reachable === false && (
        <Alert severity="error" sx={{ mb: 1 }}>
          The firmware server is not reachable. Change it above, or start the
          firmware-image-server container.
        </Alert>
      )}
      {check?.server_reachable && !check?.server_version && (
        <Alert severity="info" sx={{ mb: 1 }}>
          This firmware server has no version.json (an older image), so versions cannot be
          compared and updated boards cannot be verified. Select the boards to flash.
        </Alert>
      )}
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell padding="checkbox" />
            <TableCell>Board</TableCell>
            <TableCell>Installed</TableCell>
            <TableCell>Status</TableCell>
            <TableCell />
          </TableRow>
        </TableHead>
        <TableBody>
          {(check?.devices || []).map((d) => {
            const status = FIRMWARE_STATUS[d.update_status] || FIRMWARE_STATUS.unknown;
            return (
              <TableRow key={rowKey(d)}>
                <TableCell padding="checkbox">
                  <Checkbox size="small" checked={selected.has(rowKey(d))} disabled={!selectable(d)}
                            onChange={() => toggle(rowKey(d))}
                            inputProps={{ "aria-label": `update ${boardLabel(d)}` }} />
                </TableCell>
                <TableCell>
                  {boardLabel(d)}
                  <Typography variant="caption" color="text.secondary" sx={{ display: "block", ...mono }}>
                    {d.filename || "no image"}
                  </Typography>
                </TableCell>
                <TableCell sx={mono}>{d.installed_version || "—"}</TableCell>
                <TableCell sx={{ color: status.color, whiteSpace: "nowrap" }}>{status.label}</TableCell>
                <TableCell align="right">
                  {d.connection === "can" && d.update_status !== "unreachable" && (
                    <Button size="small" onClick={() => setReassign({ device: d, newId: "" })}>
                      Change ID
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      <FormControlLabel
        sx={{ mt: 1 }}
        control={<Checkbox size="small" checked={deepScan}
                           onChange={(e) => { setDeepScan(e.target.checked); loadCheck(e.target.checked); }} />}
        label={<Typography variant="body2">Also find boards without a CAN route (probes ids 1–127, slower)</Typography>}
      />
      <Alert severity="info" sx={{ mt: 1 }}>
        Lasers are switched off first. CAN boards are updated one at a time, the USB master last
        (ImSwitch reconnects afterwards). Updated boards restart; motors lose their position and
        need homing. The update stops at the first board that fails. Keep power and USB connected.
      </Alert>
      {reasons.length > 0 && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Not started:
          <Box component="ul" sx={{ m: 0, pl: 2 }}>
            {reasons.map((r) => <li key={r}>{r}</li>)}
          </Box>
        </Alert>
      )}
    </>
  );

  const renderRun = () => (
    <>
      {running && <LinearProgress sx={{ mb: 2 }} />}
      {!running && FINAL[run?.state] && (
        <Alert severity={FINAL[run.state]} sx={{ mb: 2 }}>{run.message}</Alert>
      )}
      {running && <Typography variant="body2" sx={{ mb: 1 }}>{run.message}</Typography>}
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
                <TableCell sx={mono}>{step.from_version || "—"} → {step.to_version || "?"}</TableCell>
                <TableCell sx={{ minWidth: 180 }}>
                  <Typography variant="body2" sx={{ color: look.color }}>{look.label}</Typography>
                  {live && (
                    <>
                      <LinearProgress variant="determinate" value={live.progress || 0} sx={{ my: 0.5 }} />
                      <Typography variant="caption" color="text.secondary">{live.message}</Typography>
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
    <>
      <DialogContent dividers>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        {loading ? (
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, py: 3 }}>
            <CircularProgress size={20} />
            <Typography variant="body2">Reading the boards' firmware versions…</Typography>
          </Box>
        ) : run ? renderRun() : check && renderSelection()}
      </DialogContent>
      <DialogActions>
        {!run && <Button onClick={onBack} sx={{ mr: "auto" }}>← Other method</Button>}
        {running && (
          <Button color="warning" onClick={() => apiUC2ConfigControllerCancelFirmwareUpdate()}>
            Stop after current board
          </Button>
        )}
        {run && !running && (
          <Button onClick={() => { setRun(null); loadCheck(); }}>Check again</Button>
        )}
        <Button onClick={onClose}>{running ? "Hide (keeps running)" : "Close"}</Button>
        {!run && check && (
          <Button variant="contained" onClick={start} disabled={loading || nSelected === 0}>
            Update {nSelected} board{nSelected === 1 ? "" : "s"}
          </Button>
        )}
      </DialogActions>

      <Dialog open={Boolean(reassign)} onClose={() => !reassign?.busy && setReassign(null)}
              maxWidth="xs" fullWidth>
        <DialogTitle>Change CAN ID</DialogTitle>
        <DialogContent>
          {reassign && (
            <>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {reassign.device.mac
                  ? `The board with MAC ${reassign.device.mac} (now CAN ID ${reassign.device.canId}) ` +
                    "keeps the new ID; no reflash needed."
                  : `The board now at CAN ID ${reassign.device.canId} (no MAC reported).`}
              </Typography>
              <TextField autoFocus fullWidth margin="normal" type="number" label="New CAN ID (1–127)"
                         value={reassign.newId} disabled={reassign.busy}
                         onChange={(e) => setReassign({ ...reassign, newId: e.target.value })} />
              {reassign.error && <Alert severity="error">{reassign.error}</Alert>}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReassign(null)} disabled={reassign?.busy}>Cancel</Button>
          <Button variant="contained" onClick={assignId} disabled={reassign?.busy || !reassign?.newId}>
            {reassign?.busy ? "Assigning…" : "Assign"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default FirmwareUpdateDialog;
