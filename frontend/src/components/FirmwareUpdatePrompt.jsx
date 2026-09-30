import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
} from "@mui/material";
import apiUC2ConfigControllerGetFirmwareUpdatePrompt from "../backendapi/apiUC2ConfigControllerGetFirmwareUpdatePrompt";
import { getFirmwareUpdateState, setFirmwarePrompt } from "../state/slices/FirmwareUpdateSlice";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import FirmwareUpdateDialog from "./FirmwareUpdateDialog";

// Per browser: the server version the user chose to skip.
const SKIP_KEY = "uc2.firmwareUpdate.skippedVersion";
const readSkipped = () => {
  try {
    return localStorage.getItem(SKIP_KEY);
  } catch {
    return null;
  }
};

// Asks once when the backend's opt-in check after startup found outdated
// boards (uc2Config.checkFirmwareOnConnect). The result is pulled on connect —
// the push signal is lost if no browser was open when the check ran.
const FirmwareUpdatePrompt = () => {
  const dispatch = useDispatch();
  const { prompt } = useSelector(getFirmwareUpdateState);
  const { ip, apiPort } = useSelector(getConnectionSettingsState);
  const [closed, setClosed] = useState(false);
  const [reviewOpen, setReviewOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    apiUC2ConfigControllerGetFirmwareUpdatePrompt()
      .then((result) => {
        if (!cancelled && result?.updates_available > 0) dispatch(setFirmwarePrompt(result));
      })
      .catch(() => {}); // no UC2 board / older backend: nothing to offer
    return () => {
      cancelled = true;
    };
  }, [ip, apiPort, dispatch]);

  const outdated = (prompt?.devices || []).filter((d) => d.update_status === "update_available");
  const show =
    !closed && outdated.length > 0 && prompt.server_version !== readSkipped();

  const skipVersion = () => {
    try {
      localStorage.setItem(SKIP_KEY, prompt.server_version);
    } catch {
      // private window / blocked storage: behaves like "Not now"
    }
    setClosed(true);
  };

  return (
    <>
      <Dialog open={show} onClose={() => setClosed(true)} maxWidth="xs" fullWidth>
        <DialogTitle>Firmware update available</DialogTitle>
        <DialogContent>
          <Typography variant="body2" gutterBottom>
            {outdated.length} board{outdated.length === 1 ? "" : "s"} run older firmware than the
            firmware server offers:
          </Typography>
          {outdated.map((d) => (
            <Typography key={`${d.connection}-${d.canId}`} variant="body2" color="text.secondary">
              {d.connection === "usb" ? `${d.deviceTypeStr} (USB master)` : `${d.deviceTypeStr} ${d.canId}`}
            </Typography>
          ))}
          <Typography variant="body2" sx={{ mt: 1.5, fontFamily: "monospace", fontSize: "0.8rem" }}>
            {prompt?.server_version}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={skipVersion}>Skip this version</Button>
          <Button onClick={() => setClosed(true)}>Not now</Button>
          <Button
            variant="contained"
            onClick={() => {
              setClosed(true);
              setReviewOpen(true);
            }}
          >
            Review…
          </Button>
        </DialogActions>
      </Dialog>
      {/* Re-checks on open: the startup result may be hours old. */}
      {reviewOpen && <FirmwareUpdateDialog open onClose={() => setReviewOpen(false)} />}
    </>
  );
};

export default FirmwareUpdatePrompt;
