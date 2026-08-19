// src/components/CameraSettingsDialog.jsx
//
// Advanced camera settings dialog, opened from the live view.
//
// Renders the complete parameter dictionary the detector manager exposes
// (/SettingsController/getDetectorParameterTree) grouped the way the manager
// declared it, with the right widget per parameter type. Every edit is pushed
// to the backend immediately and the dialog re-renders from the tree the
// backend echoes back, so hardware-side clamping is visible.
//
// Read-only entries (sensor temperature, frame counter, image size, ...) are
// shown as disabled fields and refreshed by a slow poll while the dialog is open.

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  Grid,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import RefreshIcon from "@mui/icons-material/Refresh";

import apiSettingsControllerGetDetectorParameterTree from "../backendapi/apiSettingsControllerGetDetectorParameterTree";
import apiSettingsControllerSetDetectorParameterValue from "../backendapi/apiSettingsControllerSetDetectorParameterValue";
import apiSettingsControllerSetDetectorBinning from "../backendapi/apiSettingsControllerSetDetectorBinning";
import apiSettingsControllerGetDetectorNames from "../backendapi/apiSettingsControllerGetDetectorNames";

// Values that the camera reports back on its own (temperature, frame counter);
// a slow poll keeps them current without fighting the user's typing.
const REFRESH_INTERVAL_MS = 5000;

function formatLimits(min, max, units) {
  const unit = units ? ` ${units}` : "";
  const hasMin = min !== null && min !== undefined;
  const hasMax = max !== null && max !== undefined;
  if (hasMin && hasMax) return `range ${min} – ${max}${unit}`;
  if (hasMin) return `min ${min}${unit}`;
  if (hasMax) return `max ${max}${unit}`;
  return "";
}

function formatReadonlyValue(value) {
  if (value === null || value === undefined) return "–";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}

const ParameterField = React.memo(function ParameterField({
  param,
  onChange,
  disabled,
}) {
  const { name, type, value, editable, options, units, min, max } = param;
  const label = units ? `${name} (${units})` : name;

  // `dirty` = user is typing, `pending` = commit in flight. While either is set
  // the polled server value must not overwrite what is on screen.
  const [draft, setDraft] = useState(value ?? "");
  const [dirty, setDirty] = useState(false);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    if (!dirty && !pending) setDraft(value ?? "");
  }, [value, dirty, pending]);

  const commitValue = async (toSend) => {
    setDraft(toSend);
    setDirty(false);
    setPending(true);
    try {
      await onChange(name, toSend);
    } finally {
      setPending(false);
    }
  };

  if (!editable) {
    return (
      <TextField
        size="small"
        fullWidth
        label={label}
        value={formatReadonlyValue(value)}
        disabled
        helperText="read-only"
        FormHelperTextProps={{ sx: { minHeight: "1.2em", m: 0 } }}
      />
    );
  }

  if (type === "list") {
    return (
      <FormControl size="small" fullWidth disabled={disabled}>
        <InputLabel id={`camera-param-${name}-label`}>{name}</InputLabel>
        <Select
          labelId={`camera-param-${name}-label`}
          id={`camera-param-${name}`}
          label={name}
          value={pending || dirty ? draft : value ?? ""}
          onChange={(e) => commitValue(e.target.value)}
        >
          {(options || []).map((opt) => (
            <MenuItem key={opt} value={opt}>
              {opt}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    );
  }

  if (type === "boolean") {
    const boolStr = (pending || dirty ? draft : value) ? "true" : "false";
    return (
      <FormControl size="small" fullWidth disabled={disabled}>
        <InputLabel id={`camera-param-${name}-label`}>{name}</InputLabel>
        <Select
          labelId={`camera-param-${name}-label`}
          id={`camera-param-${name}`}
          label={name}
          value={boolStr}
          onChange={(e) => commitValue(e.target.value === "true")}
        >
          <MenuItem value="true">True</MenuItem>
          <MenuItem value="false">False</MenuItem>
        </Select>
      </FormControl>
    );
  }

  const commit = async () => {
    if (!dirty) return;
    let toSend = draft;
    if (type === "number") {
      const num = parseFloat(draft);
      if (!Number.isFinite(num)) {
        setDraft(value ?? "");
        setDirty(false);
        return;
      }
      toSend = num;
    }
    await commitValue(toSend);
  };

  const limitsText = formatLimits(min, max, units);
  const numberInputProps = {};
  if (min !== null && min !== undefined) numberInputProps.min = min;
  if (max !== null && max !== undefined) numberInputProps.max = max;

  return (
    <TextField
      size="small"
      fullWidth
      label={label}
      type={type === "number" ? "number" : "text"}
      value={draft}
      disabled={disabled}
      inputProps={type === "number" ? numberInputProps : undefined}
      helperText={
        dirty ? "Press Enter or click outside to apply" : limitsText || " "
      }
      FormHelperTextProps={{ sx: { minHeight: "1.2em", m: 0 } }}
      onChange={(e) => {
        setDraft(e.target.value);
        setDirty(true);
      }}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          commit();
        } else if (e.key === "Escape") {
          setDraft(value ?? "");
          setDirty(false);
        }
      }}
    />
  );
});

export default function CameraSettingsDialog({ open, onClose }) {
  const [detectorNames, setDetectorNames] = useState([]);
  // Only set by the detector picker; null means "whatever the backend
  // considers the current detector". The tree carries the resolved name.
  const [selectedDetector, setSelectedDetector] = useState(null);
  const [tree, setTree] = useState(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  // Guards the background poll: never replace the tree while a write is in
  // flight, otherwise a stale response can undo what was just applied.
  const busyRef = useRef(false);
  busyRef.current = busy;

  const loadTree = useCallback(
    async (name, { silent = false } = {}) => {
      if (!silent) setLoading(true);
      try {
        const data = await apiSettingsControllerGetDetectorParameterTree(name);
        if (data && data.error && !data.groups) {
          setError(data.error);
        } else {
          setTree(data);
          setError(null);
        }
      } catch (e) {
        setError(`Could not load camera parameters: ${e?.message || e}`);
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [],
  );

  // Detector list (only used to offer a picker when there is more than one).
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    apiSettingsControllerGetDetectorNames()
      .then((names) => {
        if (!cancelled && Array.isArray(names)) setDetectorNames(names);
      })
      .catch(() => {
        /* the picker is optional – the current detector still works */
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    loadTree(selectedDetector);
  }, [open, selectedDetector, loadTree]);

  // Slow refresh for the values the camera updates on its own.
  useEffect(() => {
    if (!open) return undefined;
    const id = setInterval(() => {
      if (!busyRef.current) loadTree(selectedDetector, { silent: true });
    }, REFRESH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [open, selectedDetector, loadTree]);

  const targetDetector = tree?.detectorName ?? selectedDetector;

  const handleParamChange = useCallback(
    async (name, value) => {
      setBusy(true);
      try {
        const updated = await apiSettingsControllerSetDetectorParameterValue({
          detectorName: targetDetector,
          name,
          value,
        });
        if (updated?.error) setError(`Failed to set ${name}: ${updated.error}`);
        else setError(null);
        if (updated?.groups) setTree(updated);
      } catch (e) {
        setError(`Failed to set ${name}: ${e?.message || e}`);
        await loadTree(selectedDetector, { silent: true });
      } finally {
        setBusy(false);
      }
    },
    [targetDetector, selectedDetector, loadTree],
  );

  const handleBinningChange = useCallback(
    async (value) => {
      setBusy(true);
      try {
        const result = await apiSettingsControllerSetDetectorBinning({
          binning: value,
          detectorName: targetDetector,
        });
        if (result?.status === "error") {
          setError(`Failed to set binning: ${result.error}`);
        } else {
          setError(null);
        }
      } catch (e) {
        setError(`Failed to set binning: ${e?.message || e}`);
      } finally {
        setBusy(false);
        await loadTree(selectedDetector, { silent: true });
      }
    },
    [targetDetector, selectedDetector, loadTree],
  );

  const supportedBinnings = tree?.supportedBinnings || [];
  const groups = tree?.groups || [];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      scroll="paper"
      aria-labelledby="camera-settings-dialog-title"
    >
      <DialogTitle
        id="camera-settings-dialog-title"
        sx={{ display: "flex", alignItems: "center", gap: 1 }}
      >
        <Box sx={{ flexGrow: 1 }}>
          Camera settings
          {tree?.model ? (
            <Typography variant="body2" color="text.secondary">
              {tree.detectorName} — {tree.model}
            </Typography>
          ) : null}
        </Box>
        <Tooltip title="Reload from camera">
          <span>
            <IconButton
              onClick={() => loadTree(selectedDetector)}
              disabled={loading || busy}
            >
              <RefreshIcon />
            </IconButton>
          </span>
        </Tooltip>
        <IconButton onClick={onClose} aria-label="Close">
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers>
        {error ? (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        ) : null}

        {loading && !tree ? (
          <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
            <CircularProgress />
          </Box>
        ) : null}

        {tree ? (
          <>
            {/* Hardware summary */}
            <Stack
              direction="row"
              spacing={1}
              sx={{ flexWrap: "wrap", gap: 1, mb: 2 }}
            >
              {tree.sensorWidth ? (
                <Chip
                  size="small"
                  label={`Sensor ${tree.sensorWidth} × ${tree.sensorHeight}`}
                />
              ) : null}
              {tree.currentWidth ? (
                <Chip
                  size="small"
                  label={`Frame ${tree.currentWidth} × ${tree.currentHeight}`}
                />
              ) : null}
              {Array.isArray(tree.pixelSizeUm) ? (
                <Chip
                  size="small"
                  label={`Pixel ${Number(
                    tree.pixelSizeUm[tree.pixelSizeUm.length - 1],
                  ).toFixed(3)} µm`}
                />
              ) : null}
              <Chip size="small" label={tree.isRGB ? "RGB" : "Mono"} />
              {tree.cameraType ? (
                <Chip size="small" label={tree.cameraType} />
              ) : null}
              <Chip
                size="small"
                color={tree.isConnected ? "success" : "default"}
                label={tree.isConnected ? "Connected" : "Not connected"}
              />
              {tree.isMock ? (
                <Chip size="small" color="warning" label="Mock camera" />
              ) : null}
            </Stack>

            {/* Detector picker + binning */}
            <Grid container spacing={2} sx={{ mb: 1 }}>
              {detectorNames.length > 1 ? (
                <Grid item xs={12} sm={6} md={4}>
                  <FormControl size="small" fullWidth>
                    <InputLabel id="camera-detector-label">Detector</InputLabel>
                    <Select
                      labelId="camera-detector-label"
                      id="camera-detector-select"
                      label="Detector"
                      value={tree.detectorName || ""}
                      onChange={(e) => setSelectedDetector(e.target.value)}
                      disabled={busy}
                    >
                      {detectorNames.map((n) => (
                        <MenuItem key={n} value={n}>
                          {n}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              ) : null}

              <Grid item xs={12} sm={6} md={4}>
                <Tooltip
                  arrow
                  title="Hardware/digital binning. Higher binning gives a smaller, brighter and less noisy image."
                >
                  <FormControl
                    size="small"
                    fullWidth
                    disabled={busy || supportedBinnings.length < 2}
                  >
                    <InputLabel id="camera-binning-label">Binning</InputLabel>
                    <Select
                      labelId="camera-binning-label"
                      id="camera-binning-select"
                      label="Binning"
                      value={tree.binning ?? 1}
                      onChange={(e) => handleBinningChange(e.target.value)}
                    >
                      {(supportedBinnings.length
                        ? supportedBinnings
                        : [tree.binning ?? 1]
                      ).map((b) => (
                        <MenuItem key={b} value={b}>
                          {`${b} × ${b}`}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Tooltip>
              </Grid>
            </Grid>

            {/* Parameter groups */}
            {groups.map((group) => (
              <Box key={group.name} sx={{ mb: 2 }}>
                <Divider textAlign="left" sx={{ mb: 1.5, mt: 1 }}>
                  <Typography variant="overline" color="text.secondary">
                    {group.name}
                  </Typography>
                </Divider>
                <Grid container spacing={2}>
                  {(group.parameters || []).map((param) => (
                    <Grid item xs={12} sm={6} md={4} key={param.name}>
                      <ParameterField
                        param={param}
                        onChange={handleParamChange}
                        disabled={busy}
                      />
                    </Grid>
                  ))}
                </Grid>
              </Box>
            ))}

            {groups.length === 0 && !loading ? (
              <Typography variant="body2" color="text.secondary">
                This detector does not expose any parameters.
              </Typography>
            ) : null}
          </>
        ) : null}
      </DialogContent>

      <DialogActions>
        {busy ? <CircularProgress size={20} sx={{ mr: 1 }} /> : null}
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
