// src/components/MMCoreController.jsx
// Generic MMCore (Micro-Manager) parameter editor + long-exposure snap-to-disk widget.
//
// Renders the full property tree returned by /MMCoreController/getMMCoreParameters
// (grouped, with the right widget per parameter type), exposes Exposure in
// seconds for human-friendly long exposures, and runs snaps in the background
// via /MMCoreController/snapMMCoreToDisk with a polled progress chip.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
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
import RefreshIcon from "@mui/icons-material/Refresh";
import PhotoCameraIcon from "@mui/icons-material/PhotoCamera";

import apiMMCoreControllerGetMMCoreDetectors from "../backendapi/apiMMCoreControllerGetMMCoreDetectors";
import apiMMCoreControllerGetMMCoreParameters from "../backendapi/apiMMCoreControllerGetMMCoreParameters";
import apiMMCoreControllerSetMMCoreParameter from "../backendapi/apiMMCoreControllerSetMMCoreParameter";
import apiMMCoreControllerSnapMMCoreToDisk from "../backendapi/apiMMCoreControllerSnapMMCoreToDisk";
import apiMMCoreControllerGetMMCoreSnapStatus from "../backendapi/apiMMCoreControllerGetMMCoreSnapStatus";

// Names that mean the same as "Exposure" so we hide the duplicate field in the
// generic editor (we render a friendlier Exposure-in-seconds box separately).
const EXPOSURE_KEY_RE = /posure/i;

function formatElapsed(ms) {
  if (!ms || ms < 0) return "0s";
  const totalSeconds = Math.round(ms / 1000);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

const ParameterField = React.memo(function ParameterField({ param, onChange, disabled }) {
  const { name, type, value, editable, options, units } = param;
  const isDisabled = disabled || !editable;

  if (type === "list") {
    return (
      <FormControl size="small" fullWidth disabled={isDisabled}>
        <InputLabel>{name}</InputLabel>
        <Select
          label={name}
          value={value ?? ""}
          onChange={(e) => onChange(name, e.target.value)}
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
    return (
      <FormControl size="small" fullWidth disabled={isDisabled}>
        <InputLabel>{name}</InputLabel>
        <Select
          label={name}
          value={value ? "true" : "false"}
          onChange={(e) => onChange(name, e.target.value === "true")}
        >
          <MenuItem value="true">True</MenuItem>
          <MenuItem value="false">False</MenuItem>
        </Select>
      </FormControl>
    );
  }

  // number / unknown — render as text field
  return (
    <TextField
      size="small"
      fullWidth
      label={name + (units ? ` (${units})` : "")}
      type={type === "number" ? "number" : "text"}
      value={value ?? ""}
      disabled={isDisabled}
      onChange={(e) => {
        const v = type === "number" ? parseFloat(e.target.value) : e.target.value;
        onChange(name, Number.isNaN(v) ? e.target.value : v);
      }}
    />
  );
});

const MMCoreController = () => {
  const [detectors, setDetectors] = useState([]);
  const [detectorName, setDetectorName] = useState("");
  const [tree, setTree] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);

  // Exposure in seconds, decoupled from the parameter tree so the user can
  // type freely without it being clobbered by polling responses.
  const [exposureSec, setExposureSec] = useState("");
  const [snapName, setSnapName] = useState("");
  const [activeJob, setActiveJob] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);
  const pollRef = useRef(null);

  // ----------------------------------------------------------------------
  // Data loading
  // ----------------------------------------------------------------------
  const refreshDetectors = useCallback(async () => {
    try {
      const list = await apiMMCoreControllerGetMMCoreDetectors();
      setDetectors(list || []);
      setLoadError(null);
      if (list && list.length > 0) {
        setDetectorName((prev) => (list.includes(prev) ? prev : list[0]));
      } else {
        setDetectorName("");
        setTree(null);
      }
    } catch (e) {
      setLoadError(
        "Could not contact the MMCore controller. " +
          "Is MMCoreController included in your setup's availableWidgets?",
      );
    }
  }, []);

  const refreshParameters = useCallback(async () => {
    if (!detectorName) return;
    setLoading(true);
    try {
      const data = await apiMMCoreControllerGetMMCoreParameters(detectorName);
      setTree(data);
      // Seed Exposure-in-seconds from the device the first time only — don't
      // clobber an in-flight edit by the user.
      const exposureParam = findExposureParam(data);
      if (exposureParam && exposureSec === "") {
        setExposureSec(String(Number(exposureParam.value) / 1000));
      }
      setLoadError(null);
    } catch (e) {
      setLoadError(`Failed to load parameters: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  }, [detectorName, exposureSec]);

  useEffect(() => {
    refreshDetectors();
  }, [refreshDetectors]);

  useEffect(() => {
    if (detectorName) refreshParameters();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectorName]);

  // ----------------------------------------------------------------------
  // Parameter mutation
  // ----------------------------------------------------------------------
  const handleParamChange = useCallback(
    async (name, value) => {
      if (!detectorName) return;
      try {
        const updated = await apiMMCoreControllerSetMMCoreParameter({
          detectorName,
          name,
          value,
        });
        setTree(updated);
      } catch (e) {
        setLoadError(`Failed to set ${name}: ${e?.message || e}`);
      }
    },
    [detectorName],
  );

  // ----------------------------------------------------------------------
  // Snap-to-disk + polling
  // ----------------------------------------------------------------------
  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPolling = (jobId) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const status = await apiMMCoreControllerGetMMCoreSnapStatus(jobId);
        setActiveJob(status);
        if (status.state === "done" || status.state === "error") {
          stopPolling();
          setRecentJobs((prev) => [status, ...prev].slice(0, 5));
          // Refresh device parameters so any device-side clamping is shown.
          refreshParameters();
        }
      } catch (e) {
        stopPolling();
        setActiveJob((prev) =>
          prev
            ? { ...prev, state: "error", error: e?.message || String(e) }
            : prev,
        );
      }
    }, 1000);
  };

  useEffect(() => () => stopPolling(), []);

  const onSnap = async () => {
    if (!detectorName) return;
    const expSecParsed = parseFloat(exposureSec);
    const exposureMs =
      Number.isFinite(expSecParsed) && expSecParsed > 0
        ? expSecParsed * 1000
        : undefined;
    try {
      const job = await apiMMCoreControllerSnapMMCoreToDisk({
        detectorName,
        exposureMs,
        fileName: snapName || undefined,
        saveFormat: "tiff",
      });
      setActiveJob(job);
      if (job?.jobId && job.state !== "done" && job.state !== "error") {
        startPolling(job.jobId);
      }
    } catch (e) {
      setLoadError(`Failed to start snap: ${e?.message || e}`);
    }
  };

  // ----------------------------------------------------------------------
  // Derived data for rendering
  // ----------------------------------------------------------------------
  const groupedParams = useMemo(() => {
    if (!tree?.groups) return [];
    return tree.groups
      .map((group) => ({
        name: group.name,
        parameters: (group.parameters || []).filter(
          (p) => !EXPOSURE_KEY_RE.test(p.name),
        ),
      }))
      .filter((g) => g.parameters.length > 0);
  }, [tree]);

  const isJobRunning =
    activeJob && (activeJob.state === "running" || activeJob.state === "pending");

  return (
    <Box sx={{ width: "100%", p: 1 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
        <Typography variant="h5">MMCore</Typography>
        <Tooltip title="Reload detector list">
          <IconButton size="small" onClick={refreshDetectors}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Stack>

      {loadError && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setLoadError(null)}>
          {loadError}
        </Alert>
      )}

      {detectors.length === 0 ? (
        <Alert severity="info">
          No MMCore detectors are registered in this setup. Add an
          MMCoreDetectorManager entry to your setup JSON and reload.
        </Alert>
      ) : (
        <Stack spacing={2}>
          <Card>
            <CardContent>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} md={4}>
                  <FormControl size="small" fullWidth>
                    <InputLabel>Detector</InputLabel>
                    <Select
                      label="Detector"
                      value={detectorName}
                      onChange={(e) => setDetectorName(e.target.value)}
                    >
                      {detectors.map((d) => (
                        <MenuItem key={d} value={d}>
                          {d}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Sensor
                  </Typography>
                  <Typography variant="body2">
                    {tree?.sensorWidth ?? "?"} × {tree?.sensorHeight ?? "?"}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Model
                  </Typography>
                  <Typography variant="body2" noWrap>
                    {tree?.model ?? "—"}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={2} sx={{ textAlign: "right" }}>
                  <Tooltip title="Reload parameters from device">
                    <span>
                      <IconButton
                        size="small"
                        onClick={refreshParameters}
                        disabled={loading || !detectorName}
                      >
                        <RefreshIcon />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Long-exposure snap panel */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Snap to disk
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Runs a single software-triggered exposure on the device and
                writes the resulting frame to the recordings folder. Stop the
                live view first for long exposures (&gt; ~10 s).
              </Typography>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={4}>
                  <TextField
                    size="small"
                    fullWidth
                    type="number"
                    label="Exposure (s)"
                    value={exposureSec}
                    onChange={(e) => setExposureSec(e.target.value)}
                    inputProps={{ min: 0, step: 0.1 }}
                    helperText="Leave blank to use current device setting"
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    size="small"
                    fullWidth
                    label="File name suffix (optional)"
                    value={snapName}
                    onChange={(e) => setSnapName(e.target.value)}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<PhotoCameraIcon />}
                    onClick={onSnap}
                    disabled={isJobRunning || !detectorName}
                  >
                    Snap to disk
                  </Button>
                </Grid>
              </Grid>

              {activeJob && (
                <Box sx={{ mt: 2 }}>
                  <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
                    <Chip
                      label={activeJob.state}
                      color={
                        activeJob.state === "done"
                          ? "success"
                          : activeJob.state === "error"
                            ? "error"
                            : "warning"
                      }
                      icon={isJobRunning ? <CircularProgress size={14} /> : undefined}
                    />
                    <Typography variant="body2">
                      Exposure: {activeJob.exposureMs?.toFixed?.(0) ?? "?"} ms
                    </Typography>
                    <Typography variant="body2">
                      Elapsed: {formatElapsed(activeJob.elapsedMs)}
                    </Typography>
                    {activeJob.filePath && (
                      <Typography variant="body2" noWrap sx={{ flex: 1 }}>
                        Saved: {activeJob.relativeFilePath || activeJob.filePath}
                      </Typography>
                    )}
                    {activeJob.error && (
                      <Typography variant="body2" color="error" noWrap sx={{ flex: 1 }}>
                        {activeJob.error}
                      </Typography>
                    )}
                  </Stack>
                </Box>
              )}

              {recentJobs.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Divider sx={{ mb: 1 }} />
                  <Typography variant="caption" color="text.secondary">
                    Recent
                  </Typography>
                  {recentJobs.map((j) => (
                    <Typography
                      key={j.jobId}
                      variant="caption"
                      display="block"
                      noWrap
                      color={j.state === "error" ? "error" : "text.secondary"}
                    >
                      [{j.state}] {formatElapsed(j.elapsedMs)} •{" "}
                      {j.relativeFilePath || j.error || j.filePath || ""}
                    </Typography>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Parameter editor grouped by MMCore property group */}
          {groupedParams.map((group) => (
            <Card key={group.name}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {group.name}
                </Typography>
                <Grid container spacing={2}>
                  {group.parameters.map((param) => (
                    <Grid item xs={12} sm={6} md={4} key={param.name}>
                      <ParameterField
                        param={param}
                        onChange={handleParamChange}
                        disabled={loading || isJobRunning}
                      />
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          ))}
        </Stack>
      )}
    </Box>
  );
};

function findExposureParam(tree) {
  if (!tree?.groups) return null;
  for (const g of tree.groups) {
    for (const p of g.parameters || []) {
      if (EXPOSURE_KEY_RE.test(p.name)) return p;
    }
  }
  return null;
}

export default MMCoreController;
