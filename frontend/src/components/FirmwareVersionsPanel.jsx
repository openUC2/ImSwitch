import React, { useEffect, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  CircularProgress,
  FormControlLabel,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import apiUC2ConfigControllerCheckFirmwareUpdates from "../backendapi/apiUC2ConfigControllerCheckFirmwareUpdates";
import apiUC2ConfigControllerGetFirmwareCheckOnConnect from "../backendapi/apiUC2ConfigControllerGetFirmwareCheckOnConnect";
import apiUC2ConfigControllerSetFirmwareCheckOnConnect from "../backendapi/apiUC2ConfigControllerSetFirmwareCheckOnConnect";
import { FIRMWARE_STATUS } from "./firmwareStatus";
import FirmwareUpdateDialog from "./FirmwareUpdateDialog";

const mono = { fontFamily: "monospace", fontSize: "0.8rem", wordBreak: "break-all" };

// Installed vs. available firmware for every connected board (USB board + CAN
// nodes). Checking is read-only; "Update outdated boards" opens the update dialog.
const FirmwareVersionsPanel = ({ disabled }) => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [updateOpen, setUpdateOpen] = useState(false);
  const [checkOnStart, setCheckOnStart] = useState(null); // null = unknown (older backend)

  useEffect(() => {
    apiUC2ConfigControllerGetFirmwareCheckOnConnect()
      .then((r) => setCheckOnStart(Boolean(r?.enabled)))
      .catch(() => setCheckOnStart(null));
  }, []);

  const toggleCheckOnStart = async (enabled) => {
    try {
      const r = await apiUC2ConfigControllerSetFirmwareCheckOnConnect(enabled);
      setCheckOnStart(Boolean(r?.enabled));
      if (r?.status === "error") setError(r.message);
    } catch (e) {
      setError(`Could not save the setting: ${e.message}`);
    }
  };

  const check = async () => {
    setLoading(true);
    setError(null);
    try {
      setResult(await apiUC2ConfigControllerCheckFirmwareUpdates());
    } catch (e) {
      setError(`Version check failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const devices = result?.devices || [];

  return (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, flexWrap: "wrap" }}>
        <Button
          variant="outlined"
          onClick={check}
          disabled={disabled || loading}
          startIcon={loading ? <CircularProgress size={16} /> : null}
        >
          {loading ? "Reading versions…" : "Check firmware versions"}
        </Button>
        {result?.updates_available > 0 && (
          <Button variant="contained" onClick={() => setUpdateOpen(true)} disabled={disabled}>
            Update outdated boards…
          </Button>
        )}
        {result && (
          <Typography variant="body2" color="text.secondary">
            Server: <Box component="span" sx={mono}>{result.server_version || "no version.json"}</Box>
          </Typography>
        )}
      </Box>
      {checkOnStart !== null && (
        <FormControlLabel
          sx={{ mt: 1 }}
          control={
            <Checkbox
              size="small"
              checked={checkOnStart}
              onChange={(e) => toggleCheckOnStart(e.target.checked)}
            />
          }
          label={
            <Typography variant="body2">
              Check for firmware updates when ImSwitch starts (applies from the next start)
            </Typography>
          }
        />
      )}
      {updateOpen && (
        <FirmwareUpdateDialog
          open
          initialCheck={result}
          initialMethod="can"
          onClose={() => {
            setUpdateOpen(false);
            check(); // show what the boards run now
          }}
        />
      )}

      {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}

      {result && devices.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>No boards answered.</Alert>
      )}

      {devices.length > 0 && (
        <Table size="small" sx={{ mt: 2 }}>
          <TableHead>
            <TableRow>
              <TableCell>Board</TableCell>
              <TableCell>Installed</TableCell>
              <TableCell>Available</TableCell>
              <TableCell>Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {devices.map((d) => {
              const status = FIRMWARE_STATUS[d.update_status] || FIRMWARE_STATUS.unknown;
              return (
                <TableRow key={`${d.connection}-${d.canId}`}>
                  <TableCell>
                    {d.deviceTypeStr}
                    <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                      {d.connection === "usb" ? "USB" : "CAN"}
                      {d.canId != null ? ` · ID ${d.canId}` : ""}
                    </Typography>
                  </TableCell>
                  <TableCell sx={mono}>
                    {d.installed_version || "—"}
                    {d.build && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                        built {d.build}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell sx={mono}>{d.available_version || "—"}</TableCell>
                  <TableCell sx={{ color: status.color, whiteSpace: "nowrap" }}>
                    {status.label}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}

      {result?.updates_available > 0 && (
        <Typography variant="body2" sx={{ mt: 1.5 }}>
          {result.updates_available} board(s) run a different firmware than the server offers.
        </Typography>
      )}
    </Box>
  );
};

export default FirmwareVersionsPanel;
