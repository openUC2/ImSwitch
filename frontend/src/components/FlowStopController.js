// src/components/FlowStopController.js
// FlowStop: PlanktoScope/FairScope-style flow-cell imaging app.
// Preview & manual control -> sample metadata -> acquisition -> gallery.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Divider,
  FormControlLabel,
  Grid,
  IconButton,
  LinearProgress,
  MenuItem,
  Paper,
  Slider,
  Switch,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import FastForwardIcon from "@mui/icons-material/FastForward";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import StopIcon from "@mui/icons-material/Stop";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import RefreshIcon from "@mui/icons-material/Refresh";

import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import FreeNumberField from "./FreeNumberField";
import { useWebSocket } from "../context/WebSocketContext";
import { useT } from "../i18n";
import * as flowStopSlice from "../state/slices/FlowStopSlice.js";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import {
  apiFlowStopGetHardware,
  apiFlowStopGetStatus,
  apiFlowStopGetParameters,
  apiFlowStopSetParameters,
  apiFlowStopGetMetadata,
  apiFlowStopSetMetadata,
  apiFlowStopStart,
  apiFlowStopStop,
  apiFlowStopMovePump,
  apiFlowStopStopPump,
  apiFlowStopMoveFocus,
  apiFlowStopStopFocus,
  apiFlowStopSetIllumination,
  apiFlowStopSetAutoExposure,
  apiFlowStopSetExposureTime,
  apiFlowStopListFiles,
} from "../backendapi/apiFlowStopController";

const PARAM_KEYS = [
  "experimentName",
  "experimentDescription",
  "uniqueId",
  "numImages",
  "volumePerImage",
  "timeToStabilize",
  "pumpSpeed",
  "frameRate",
  "fileFormat",
  "isRecordVideo",
  "wasRunning",
];

// EcoTaxa-style sample fields, in the order PlanktoScope presents them.
const METADATA_FIELDS = [
  { key: "sample_project", label: "Project", type: "text" },
  { key: "sample_id", label: "Sample ID", type: "text" },
  { key: "sample_ship", label: "Ship / Platform", type: "text" },
  { key: "sample_operator", label: "Operator", type: "text" },
  { key: "sample_date", label: "Sampling date", type: "date" },
  { key: "sample_time", label: "Sampling time", type: "time" },
  { key: "sample_latitude", label: "Latitude (°N)", type: "number" },
  { key: "sample_longitude", label: "Longitude (°E)", type: "number" },
  { key: "sample_depth_min_m", label: "Depth min (m)", type: "number" },
  { key: "sample_depth_max_m", label: "Depth max (m)", type: "number" },
  { key: "sample_gear", label: "Sampling gear / net", type: "text" },
  { key: "sample_mesh_size_um", label: "Net mesh size (µm)", type: "number" },
  { key: "sample_total_volume_ml", label: "Sampled volume (ml)", type: "number" },
  { key: "acq_instrument", label: "Instrument", type: "text" },
  { key: "acq_celltype_ul", label: "Flow cell volume (µl)", type: "number" },
];

const formatDuration = (seconds) => {
  if (seconds === null || seconds === undefined || seconds < 0) return "--:--";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
};

const TabPanel = ({ children, value, index }) =>
  value === index ? <Box sx={{ p: 2 }}>{children}</Box> : null;

const FlowStopController = () => {
  const dispatch = useDispatch();
  const t = useT();
  const socket = useWebSocket();
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  const fileManagerBase = `${hostIP}:${hostPort}/imswitch/api/FileManager`;

  const state = useSelector(flowStopSlice.getFlowStopState);
  const {
    tabIndex,
    metadata,
    hardware,
    isRunning,
    currentImageCount,
    progress,
    etaSeconds,
    elapsedSeconds,
    relativePath,
    lastError,
    galleryFiles,
  } = state;

  const [notice, setNotice] = useState(null);
  const illuRange = hardware?.illumination || {};

  const set = useCallback((patch) => dispatch(flowStopSlice.setField(patch)), [dispatch]);

  // ---------------------------------------------------------------- bootstrap
  useEffect(() => {
    apiFlowStopGetHardware()
      .then((hw) => dispatch(flowStopSlice.setHardware(hw)))
      .catch(() => {});
    apiFlowStopGetParameters()
      .then((p) => {
        const patch = {};
        PARAM_KEYS.forEach((k) => {
          if (p[k] !== undefined && p[k] !== null) patch[k] = p[k];
        });
        dispatch(flowStopSlice.setField(patch));
      })
      .catch(() => {});
    apiFlowStopGetMetadata()
      .then((m) => dispatch(flowStopSlice.setMetadata(m)))
      .catch(() => {});
  }, [dispatch, hostIP, hostPort]);

  // Poll status: fast while running, slowly while idle.
  useEffect(() => {
    let cancelled = false;
    const tick = () =>
      apiFlowStopGetStatus()
        .then((s) => {
          if (!cancelled) dispatch(flowStopSlice.setStatus(s));
        })
        .catch(() => {});
    tick();
    const handle = setInterval(tick, isRunning ? 1000 : 5000);
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
  }, [dispatch, isRunning, hostIP, hostPort]);

  // Live counter updates between polls.
  useEffect(() => {
    if (!socket) return undefined;
    const handleSignal = (data) => {
      let jdata;
      try {
        jdata = typeof data === "string" ? JSON.parse(data) : data;
      } catch {
        return;
      }
      if (jdata?.name === "sigImagesTaken") {
        dispatch(flowStopSlice.setCurrentImageCount(jdata.args.p0));
      } else if (jdata?.name === "sigIsRunning") {
        dispatch(flowStopSlice.setIsRunning(jdata.args.p0));
      }
    };
    socket.on("signal", handleSignal);
    return () => socket.off("signal", handleSignal);
  }, [socket, dispatch]);

  // ------------------------------------------------------------------ actions
  const saveParameters = useCallback(() => {
    const params = {};
    PARAM_KEYS.forEach((k) => {
      params[k] = state[k];
    });
    return apiFlowStopSetParameters(params);
  }, [state]);

  const saveMetadata = useCallback(
    () =>
      apiFlowStopSetMetadata(metadata)
        .then(() => setNotice({ severity: "success", text: t("Sample metadata saved.") }))
        .catch((e) => setNotice({ severity: "error", text: String(e) })),
    [metadata, t]
  );

  const startExperiment = async () => {
    try {
      await apiFlowStopSetMetadata(metadata);
      await saveParameters();
      await apiFlowStopStart({});
      dispatch(flowStopSlice.setIsRunning(true));
      setNotice(null);
    } catch (e) {
      setNotice({ severity: "error", text: String(e) });
    }
  };

  const stopExperiment = () =>
    apiFlowStopStop()
      .then((s) => dispatch(flowStopSlice.setStatus(s)))
      .catch((e) => setNotice({ severity: "error", text: String(e) }));

  const useBrowserGPS = () => {
    if (!navigator.geolocation) {
      setNotice({ severity: "warning", text: t("This browser has no geolocation.") });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        dispatch(
          flowStopSlice.setMetadataField({
            key: "sample_latitude",
            value: Number(pos.coords.latitude.toFixed(6)),
          })
        );
        dispatch(
          flowStopSlice.setMetadataField({
            key: "sample_longitude",
            value: Number(pos.coords.longitude.toFixed(6)),
          })
        );
      },
      (err) => setNotice({ severity: "warning", text: String(err.message) })
    );
  };

  // ------------------------------------------------------------------ gallery
  const galleryPathRef = useRef("");
  const refreshGallery = useCallback(() => {
    const path = (relativePath || "").replace(/\\/g, "/");
    if (!path) return;
    galleryPathRef.current = path;
    apiFlowStopListFiles(path)
      .then((items) =>
        dispatch(
          flowStopSlice.setGalleryFiles(
            (items || []).filter((i) => !i.isDirectory && i.isImage).slice(0, 60)
          )
        )
      )
      .catch(() => {});
  }, [relativePath, dispatch]);

  useEffect(() => {
    if (tabIndex === 3) refreshGallery();
  }, [tabIndex, refreshGallery]);

  // ---------------------------------------------------------------- rendering
  const renderPreviewTab = () => (
    <Grid container spacing={2}>
      <Grid item xs={12} md={7}>
        <Card sx={{ height: "100%" }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              {t("Live view")}
            </Typography>
            <Box sx={{ minHeight: 320 }}>
              <LiveViewControlWrapper enableStageMovement={false} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={5}>
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              {t("Focus")} ({hardware?.focusAxis || "Z"})
            </Typography>
            <Grid container spacing={1} alignItems="center">
              <Grid item xs={6}>
                <FreeNumberField
                  label={t("Step (motor steps)")}
                  value={state.focusStep}
                  onCommit={(v) => set({ focusStep: v })}
                />
              </Grid>
              <Grid item xs={6}>
                <FreeNumberField
                  label={t("Speed")}
                  value={state.focusSpeed}
                  onCommit={(v) => set({ focusSpeed: v })}
                />
              </Grid>
              <Grid item xs={12}>
                <Tooltip title={t("Move focus up")}>
                  <IconButton
                    onClick={() => apiFlowStopMoveFocus(state.focusStep, state.focusSpeed)}
                  >
                    <ArrowUpwardIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title={t("Move focus down")}>
                  <IconButton
                    onClick={() => apiFlowStopMoveFocus(-state.focusStep, state.focusSpeed)}
                  >
                    <ArrowDownwardIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title={t("Stop focus")}>
                  <IconButton color="error" onClick={() => apiFlowStopStopFocus()}>
                    <StopIcon />
                  </IconButton>
                </Tooltip>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              {t("Pump")} ({hardware?.pumpAxis || "X"})
            </Typography>
            <Grid container spacing={1} alignItems="center">
              <Grid item xs={6}>
                <FreeNumberField
                  label={t("Step (motor steps)")}
                  value={state.pumpStep}
                  onCommit={(v) => set({ pumpStep: v })}
                />
              </Grid>
              <Grid item xs={6}>
                <FreeNumberField
                  label={t("Speed")}
                  value={state.pumpJogSpeed}
                  onCommit={(v) => set({ pumpJogSpeed: v })}
                />
              </Grid>
              <Grid item xs={12}>
                <Tooltip title={t("Pump backward")}>
                  <IconButton
                    onClick={() => apiFlowStopMovePump(-state.pumpStep, state.pumpJogSpeed)}
                  >
                    <FastRewindIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title={t("Pump forward")}>
                  <IconButton
                    onClick={() => apiFlowStopMovePump(state.pumpStep, state.pumpJogSpeed)}
                  >
                    <FastForwardIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title={t("Stop pump")}>
                  <IconButton color="error" onClick={() => apiFlowStopStopPump()}>
                    <StopIcon />
                  </IconButton>
                </Tooltip>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              {t("Illumination")} {illuRange.name ? `(${illuRange.name})` : ""}
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={state.illuminationOn}
                  onChange={(e) => {
                    set({ illuminationOn: e.target.checked });
                    apiFlowStopSetIllumination(state.illuminationValue, e.target.checked);
                  }}
                />
              }
              label={state.illuminationOn ? t("On") : t("Off")}
            />
            <Slider
              value={state.illuminationValue}
              min={illuRange.min ?? 0}
              max={illuRange.max ?? 1023}
              step={illuRange.step || 1}
              valueLabelDisplay="auto"
              onChange={(e, v) => set({ illuminationValue: v })}
              onChangeCommitted={(e, v) => apiFlowStopSetIllumination(v, state.illuminationOn)}
            />
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              {t("Camera")}
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={state.autoExposure}
                  onChange={(e) => {
                    set({ autoExposure: e.target.checked });
                    apiFlowStopSetAutoExposure(e.target.checked ? "auto" : "manual");
                  }}
                />
              }
              label={t("Auto exposure")}
            />
            {!state.autoExposure && (
              <FreeNumberField
                label={t("Exposure time (ms)")}
                value={state.exposureTime}
                onCommit={(v) => {
                  set({ exposureTime: v });
                  apiFlowStopSetExposureTime(v);
                }}
              />
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderSampleTab = () => (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          {t("Sample metadata")}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {t("Written as metadata.json next to the acquired frames (EcoTaxa field names).")}
        </Typography>
        <Grid container spacing={2} sx={{ mt: 0 }}>
          {METADATA_FIELDS.map((f) => (
            <Grid item xs={12} sm={6} md={4} key={f.key}>
              <TextField
                fullWidth
                label={t(f.label)}
                type={f.type}
                value={metadata[f.key] ?? ""}
                InputLabelProps={
                  f.type === "date" || f.type === "time" ? { shrink: true } : undefined
                }
                onChange={(e) =>
                  dispatch(
                    flowStopSlice.setMetadataField({
                      key: f.key,
                      value:
                        f.type === "number"
                          ? parseFloat(e.target.value) || 0
                          : e.target.value,
                    })
                  )
                }
              />
            </Grid>
          ))}
          <Grid item xs={12}>
            <TextField
              fullWidth
              multiline
              minRows={2}
              label={t("Notes")}
              value={metadata.object_notes ?? ""}
              onChange={(e) =>
                dispatch(
                  flowStopSlice.setMetadataField({
                    key: "object_notes",
                    value: e.target.value,
                  })
                )
              }
            />
          </Grid>
          <Grid item xs={12}>
            <Button startIcon={<MyLocationIcon />} onClick={useBrowserGPS} sx={{ mr: 1 }}>
              {t("Use browser GPS")}
            </Button>
            <Button variant="contained" onClick={saveMetadata}>
              {t("Save metadata")}
            </Button>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const renderAcquisitionTab = () => (
    <Grid container spacing={2}>
      <Grid item xs={12} md={7}>
        <Card>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              {t("Acquisition")}
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label={t("Experiment name")}
                  value={state.experimentName}
                  onChange={(e) => set({ experimentName: e.target.value })}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label={t("Acquisition ID")}
                  value={state.uniqueId}
                  helperText={t("Empty = generated")}
                  onChange={(e) => set({ uniqueId: e.target.value })}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label={t("Description")}
                  value={state.experimentDescription}
                  onChange={(e) => set({ experimentDescription: e.target.value })}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FreeNumberField
                  label={t("Volume per image (motor steps)")}
                  value={state.volumePerImage}
                  onCommit={(v) => set({ volumePerImage: v })}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FreeNumberField
                  label={t("Number of images")}
                  value={state.numImages}
                  helperText={t("-1 = until stopped")}
                  onCommit={(v) => set({ numImages: v })}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FreeNumberField
                  label={t("Stabilization time (s)")}
                  value={state.timeToStabilize}
                  onCommit={(v) => set({ timeToStabilize: v })}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FreeNumberField
                  label={t("Pump speed")}
                  value={state.pumpSpeed}
                  onCommit={(v) => set({ pumpSpeed: v })}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FreeNumberField
                  label={t("Max frame rate (Hz)")}
                  value={state.frameRate}
                  onCommit={(v) => set({ frameRate: v })}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  select
                  fullWidth
                  label={t("File format")}
                  value={state.fileFormat}
                  onChange={(e) => set({ fileFormat: e.target.value })}
                >
                  {["JPG", "PNG", "TIF"].map((f) => (
                    <MenuItem key={f} value={f}>
                      {f}
                    </MenuItem>
                  ))}
                </TextField>
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={state.isRecordVideo}
                      onChange={(e) => set({ isRecordVideo: e.target.checked })}
                    />
                  }
                  label={t("Also record video")}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={!!state.wasRunning}
                      onChange={(e) => set({ wasRunning: e.target.checked })}
                    />
                  }
                  label={t("Resume automatically after a restart")}
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<PlayArrowIcon />}
                  onClick={startExperiment}
                  disabled={isRunning}
                  sx={{ mr: 1 }}
                >
                  {t("Start")}
                </Button>
                <Button
                  variant="contained"
                  color="error"
                  startIcon={<StopIcon />}
                  onClick={stopExperiment}
                  disabled={!isRunning}
                >
                  {t("Stop")}
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={5}>
        <Card>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              {t("Progress")}
            </Typography>
            <LinearProgress
              variant={progress >= 0 ? "determinate" : isRunning ? "indeterminate" : "determinate"}
              value={progress >= 0 ? progress * 100 : 0}
              sx={{ height: 10, borderRadius: 5, mb: 2 }}
            />
            <Typography variant="body2">
              {t("Images")}: {currentImageCount}
              {state.numImages > 0 ? ` / ${state.numImages}` : ""}
            </Typography>
            <Typography variant="body2">
              {t("Elapsed")}: {formatDuration(elapsedSeconds)}
            </Typography>
            <Typography variant="body2">
              {t("Remaining")}: {formatDuration(etaSeconds)}
            </Typography>
            <Divider sx={{ my: 1 }} />
            <Typography variant="caption" color="text.secondary">
              {relativePath || t("No acquisition yet")}
            </Typography>
            {lastError ? (
              <Alert severity="error" sx={{ mt: 2 }}>
                {lastError}
              </Alert>
            ) : null}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderGalleryTab = () => (
    <Card>
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
            {t("Gallery")} — {relativePath || t("No acquisition yet")}
          </Typography>
          <IconButton onClick={refreshGallery}>
            <RefreshIcon />
          </IconButton>
        </Box>
        {galleryFiles.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            {t("No images found for the last acquisition.")}
          </Typography>
        ) : (
          <Grid container spacing={1}>
            {galleryFiles.map((f) => (
              <Grid item xs={6} sm={4} md={3} lg={2} key={f.path}>
                <Box
                  component="a"
                  href={`${fileManagerBase}${f.filePreviewPath}`}
                  target="_blank"
                  rel="noreferrer"
                  sx={{ display: "block", textDecoration: "none" }}
                >
                  <Box
                    component="img"
                    alt={f.name}
                    src={`${fileManagerBase}${f.thumbnailPath || f.filePreviewPath}`}
                    sx={{ width: "100%", borderRadius: 1, display: "block" }}
                  />
                  <Typography variant="caption" noWrap display="block">
                    {f.name}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        )}
      </CardContent>
    </Card>
  );

  return (
    <Paper sx={{ p: 1 }}>
      <Box sx={{ display: "flex", alignItems: "center", px: 2, pt: 1, gap: 1 }}>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          {t("Flow Stop")}
        </Typography>
        <Chip
          size="small"
          color={isRunning ? "success" : "default"}
          label={isRunning ? t("Running") : t("Idle")}
        />
        <Chip size="small" variant="outlined" label={`${currentImageCount} ${t("images")}`} />
        {isRunning && (
          <Button size="small" color="error" startIcon={<StopIcon />} onClick={stopExperiment}>
            {t("Stop")}
          </Button>
        )}
      </Box>

      {notice ? (
        <Alert severity={notice.severity} onClose={() => setNotice(null)} sx={{ mx: 2, mt: 1 }}>
          {notice.text}
        </Alert>
      ) : null}

      <Tabs
        value={tabIndex}
        onChange={(e, v) => dispatch(flowStopSlice.setTabIndex(v))}
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab label={t("Preview & Control")} />
        <Tab label={t("Sample")} />
        <Tab label={t("Acquisition")} />
        <Tab label={t("Gallery")} />
      </Tabs>

      <TabPanel value={tabIndex} index={0}>
        {renderPreviewTab()}
      </TabPanel>
      <TabPanel value={tabIndex} index={1}>
        {renderSampleTab()}
      </TabPanel>
      <TabPanel value={tabIndex} index={2}>
        {renderAcquisitionTab()}
      </TabPanel>
      <TabPanel value={tabIndex} index={3}>
        {renderGalleryTab()}
      </TabPanel>
    </Paper>
  );
};

export default FlowStopController;
