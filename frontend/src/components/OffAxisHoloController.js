// src/components/OffAxisHoloController.js
// Off-axis holography app: live sideband reconstruction (camera, spectrum,
// amplitude/phase), still-image reconstruction, interferometer fringe trace,
// browser simulation and the build guides.
import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  FormControl,
  FormControlLabel,
  Grid,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Slider,
  Stack,
  Switch,
  Tab,
  Tabs,
  Tooltip,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  AutoFixHigh as AutoFixHighIcon,
  PhotoCamera as PhotoCameraIcon,
  Timeline as TimelineIcon,
  Science as ScienceIcon,
  School as SchoolIcon,
} from "@mui/icons-material";

import * as offAxisHoloSlice from "../state/slices/OffAxisHoloSlice";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import FreeNumberField from "./FreeNumberField";
import HoloSnapshot from "./holo/HoloSnapshot";
import HoloSimulator from "./holo/HoloSimulator";
import HoloTutorial from "./holo/HoloTutorial";
import FringeTrace from "./holo/FringeTrace";
import { useT } from "../i18n";
import apiLiveViewControllerGetStreamParameters from "../backendapi/apiLiveViewControllerGetStreamParameters";
import {
  apiOffAxisHoloControllerGetParams,
  apiOffAxisHoloControllerSetParams,
  apiOffAxisHoloControllerGetState,
  apiOffAxisHoloControllerStartProcessing,
  apiOffAxisHoloControllerStopProcessing,
  apiOffAxisHoloControllerPauseProcessing,
  apiOffAxisHoloControllerResumeProcessing,
  apiOffAxisHoloControllerSetDz,
  apiOffAxisHoloControllerSetRoi,
  apiOffAxisHoloControllerSetCcRoi,
  apiOffAxisHoloControllerSetApodization,
  apiOffAxisHoloControllerSetBinning,
  apiOffAxisHoloControllerSetPixelsize,
  apiOffAxisHoloControllerSetWavelength,
  apiOffAxisHoloControllerSetUnwrapPhase,
  apiOffAxisHoloControllerSetShowFftSpace,
  apiOffAxisHoloControllerFindSideband,
} from "../backendapi/apiOffAxisHoloController";

const streamImgStyle = (visible = true) => ({
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "contain",
  imageRendering: "pixelated",
  display: visible ? "block" : "none",
});

const OffAxisHoloController = () => {
  const dispatch = useDispatch();
  const t = useT();
  const s = useSelector(offAxisHoloSlice.getOffAxisHoloState);
  const connectionSettings = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

  const [tab, setTab] = useState(0);
  const [cc, setCc] = useState({ centerX: 100, centerY: 100, sizeX: 50, sizeY: 50 });
  const [roi, setRoi] = useState({ centerX: 0, centerY: 0, size: 512 }); // centre relative to image centre, preview px
  const [imageSize, setImageSize] = useState({ width: 1920, height: 1080 });
  const [showPhase, setShowPhase] = useState(false);
  const [sidebandInfo, setSidebandInfo] = useState(null);
  const [error, setError] = useState(null);
  const fftImageRef = useRef(null);

  const baseUrl = `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api`;
  const docsUrl = `${connectionSettings.ip}/docs/`;
  const stream = (kind) => `${baseUrl}/OffAxisHoloController/mjpeg_stream_offaxisholo_${kind}?startStream=true&jpeg_quality=85`;
  const fftShape = s.fftShape || [512, 512]; // [h, w] of the processed array

  // Subsampling of the stream protocol that is actually active (reading jpeg
  // unconditionally gave the wrong ROI box while the binary stream was used).
  const [serverSubsampling, setServerSubsampling] = useState(null);
  useEffect(() => {
    apiLiveViewControllerGetStreamParameters()
      .then((r) => {
        const all = r.protocols || r;
        const f = all[r.current_protocol || "jpeg"]?.subsampling_factor;
        if (f) setServerSubsampling(f);
      })
      .catch(() => {});
  }, []);
  const subsampling = useMemo(() => {
    const ss = liveStreamState.streamSettings || {};
    if (!ss.current_compression_algorithm && serverSubsampling) return serverSubsampling;
    const fmt = liveStreamState.imageFormat || ss.current_compression_algorithm || "jpeg";
    if (fmt === "binary") return ss.binary?.subsampling?.factor || ss.binary?.subsampling_factor || 1;
    if (fmt === "mjpeg") return ss.mjpeg?.subsampling_factor || ss.mjpeg?.subsampling?.factor || 1;
    return ss.jpeg?.subsampling?.factor || ss.jpeg?.subsampling_factor || 1;
  }, [liveStreamState.streamSettings, liveStreamState.imageFormat, serverSubsampling]);
  const roiInPreview = roi.size / (subsampling * (s.binning || 1));

  // ---------------------------------------------------------------- backend sync
  const loadParameters = useCallback(async () => {
    try {
      const p = await apiOffAxisHoloControllerGetParams();
      dispatch(offAxisHoloSlice.setPixelsize(p.pixelsize || 3.45e-6));
      dispatch(offAxisHoloSlice.setWavelength(p.wavelength || 633e-9));
      dispatch(offAxisHoloSlice.setDz(p.dz || 0));
      dispatch(offAxisHoloSlice.setBinning(p.binning || 1));
      dispatch(offAxisHoloSlice.setColorChannel(p.color_channel || "green"));
      dispatch(offAxisHoloSlice.setApodizationEnabled(!!p.apodization_enabled));
      dispatch(offAxisHoloSlice.setApodizationType(p.apodization_type || "tukey"));
      dispatch(offAxisHoloSlice.setApodizationAlpha(p.apodization_alpha ?? 0.1));
      dispatch(offAxisHoloSlice.setPhaseUnwrapEnabled(p.unwrap_phase ?? true));
      dispatch(offAxisHoloSlice.setShowFftSpace(!!p.show_fft_space));
      setRoi((prev) => ({ ...prev, size: p.roi_size || 512 }));
      if (p.cc_center) {
        setCc({
          centerX: p.cc_center[0],
          centerY: p.cc_center[1],
          sizeX: p.cc_size ? p.cc_size[0] : 50,
          sizeY: p.cc_size ? p.cc_size[1] : 50,
        });
      }
    } catch (e) {
      setError(t("Off-axis controller not reachable: {msg}", { msg: e.message }));
    }
  }, [dispatch, t]);

  const loadState = useCallback(async () => {
    try {
      const st = await apiOffAxisHoloControllerGetState();
      dispatch(offAxisHoloSlice.setIsProcessing(!!st.is_processing));
      dispatch(offAxisHoloSlice.setIsPaused(!!st.is_paused));
      if (st.fft_shape) dispatch(offAxisHoloSlice.setFftShape(st.fft_shape));
    } catch (e) {
      console.error("Failed to load off-axis holo state:", e);
    }
  }, [dispatch]);

  useEffect(() => {
    loadParameters();
    loadState();
    const id = setInterval(loadState, 3000);
    return () => clearInterval(id);
  }, [loadParameters, loadState]);

  const run = (fn) => async (...args) => {
    try {
      await fn(...args);
    } catch (e) {
      setError(e.message);
    }
    loadState();
  };

  const sendCc = (u) => {
    setCc(u);
    apiOffAxisHoloControllerSetCcRoi({
      center_x: Math.round(u.centerX), center_y: Math.round(u.centerY),
      size_x: Math.round(u.sizeX), size_y: Math.round(u.sizeY),
    }).catch((e) => setError(e.message));
  };

  const findSideband = run(async () => {
    const r = await apiOffAxisHoloControllerFindSideband();
    if (!r.success) throw new Error(r.error || "no frame");
    setCc({ centerX: r.cc_center[0], centerY: r.cc_center[1], sizeX: r.cc_size[0], sizeY: r.cc_size[1] });
    dispatch(offAxisHoloSlice.setFftShape(r.fft_shape));
    setSidebandInfo(r.fringe_period_px);
  });

  // Live image click -> sensor ROI centre (sensor pixels for the backend).
  const handleLiveViewClick = (px, py, w, h) => {
    const rel = { centerX: Math.round(px - w / 2), centerY: Math.round(py - h / 2) };
    setRoi((prev) => ({ ...prev, ...rel }));
    apiOffAxisHoloControllerSetRoi({
      center_x: Math.round(px * subsampling),
      center_y: Math.round(py * subsampling),
      size: roi.size,
    }).catch((e) => setError(e.message));
  };

  const applyRoiSize = (size) => {
    setRoi((prev) => ({ ...prev, size }));
    apiOffAxisHoloControllerSetRoi({
      center_x: Math.round((imageSize.width / 2 + roi.centerX) * subsampling),
      center_y: Math.round((imageSize.height / 2 + roi.centerY) * subsampling),
      size,
    }).catch((e) => setError(e.message));
  };

  const handleFftClick = (e) => {
    const r = fftImageRef.current?.getBoundingClientRect();
    if (!r) return;
    // the <img> is object-fit: contain - map through the drawn image area
    const [h, w] = fftShape;
    const scale = Math.min(r.width / w, r.height / h);
    const ox = (r.width - w * scale) / 2;
    const oy = (r.height - h * scale) / 2;
    const x = (e.clientX - r.left - ox) / scale;
    const y = (e.clientY - r.top - oy) / scale;
    if (x < 0 || y < 0 || x > w || y > h) return;
    sendCc({ ...cc, centerX: Math.round(x), centerY: Math.round(y) });
    setSidebandInfo(null);
  };

  const setApodization = (enabled, type, alpha) => {
    dispatch(offAxisHoloSlice.setApodizationEnabled(enabled));
    dispatch(offAxisHoloSlice.setApodizationType(type));
    dispatch(offAxisHoloSlice.setApodizationAlpha(alpha));
    apiOffAxisHoloControllerSetApodization(enabled, type, alpha).catch((e) => setError(e.message));
  };

  // ---------------------------------------------------------------- overlays
  const roiOverlay = (() => {
    const cx = (roi.centerX + imageSize.width / 2) / imageSize.width;
    const cy = (roi.centerY + imageSize.height / 2) / imageSize.height;
    const sx = roiInPreview / imageSize.width;
    const sy = roiInPreview / imageSize.height;
    return (
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none"
        style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}>
        <rect x={(cx - sx / 2) * 100} y={(cy - sy / 2) * 100} width={sx * 100} height={sy * 100}
          fill="none" stroke="red" strokeWidth="0.5" />
      </svg>
    );
  })();

  const ccOverlay = (
    <svg viewBox={`0 0 ${fftShape[1]} ${fftShape[0]}`} preserveAspectRatio="xMidYMid meet"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
      <rect x={cc.centerX - cc.sizeX / 2} y={cc.centerY - cc.sizeY / 2} width={cc.sizeX} height={cc.sizeY}
        fill="none" stroke="cyan" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      <circle cx={fftShape[1] / 2} cy={fftShape[0] / 2} r={2} fill="white" />
    </svg>
  );

  // ---------------------------------------------------------------- render
  return (
    <Box sx={{ p: 1, maxWidth: "100%" }}>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, mr: 1 }}>{t("Off-Axis Holography")}</Typography>
        <Chip size="small" label={s.isProcessing ? t("Processing") : t("Stopped")} color={s.isProcessing ? "success" : "default"} />
        {s.isPaused && <Chip size="small" label={t("Paused")} color="warning" />}
      </Stack>

      <Tabs value={tab} onChange={(e, v) => setTab(v)} variant="scrollable" scrollButtons="auto"
        sx={{ mb: 1.5, borderBottom: 1, borderColor: "divider" }}>
        <Tab icon={<PlayArrowIcon />} iconPosition="start" label={t("Live")} />
        <Tab icon={<PhotoCameraIcon />} iconPosition="start" label={t("Snapshot")} />
        <Tab icon={<TimelineIcon />} iconPosition="start" label={t("Fringe trace")} />
        <Tab icon={<ScienceIcon />} iconPosition="start" label={t("Simulate")} />
        <Tab icon={<SchoolIcon />} iconPosition="start" label={t("Learn")} />
      </Tabs>

      {error && <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 1 }}>{error}</Alert>}

      {tab === 0 && (
        <>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
            <Button size="small" variant="contained" startIcon={<PlayArrowIcon />}
              onClick={run(apiOffAxisHoloControllerStartProcessing)} disabled={s.isProcessing && !s.isPaused}>
              {t("Start")}
            </Button>
            {s.isPaused ? (
              <Button size="small" variant="contained" color="success" startIcon={<PlayArrowIcon />}
                onClick={run(apiOffAxisHoloControllerResumeProcessing)}>
                {t("Resume")}
              </Button>
            ) : (
              <Tooltip title={t("Freeze the current frame; dz and sideband changes still apply to it.")}>
                <span>
                  <Button size="small" variant="contained" color="warning" startIcon={<PauseIcon />}
                    onClick={run(apiOffAxisHoloControllerPauseProcessing)} disabled={!s.isProcessing}>
                    {t("Pause")}
                  </Button>
                </span>
              </Tooltip>
            )}
            <Button size="small" variant="contained" color="error" startIcon={<StopIcon />}
              onClick={run(apiOffAxisHoloControllerStopProcessing)} disabled={!s.isProcessing}>
              {t("Stop")}
            </Button>
            <Tooltip title={t("Find the brightest side peak in the spectrum and select it.")}>
              <Button size="small" variant="outlined" startIcon={<AutoFixHighIcon />} onClick={findSideband}>
                {t("Find sideband")}
              </Button>
            </Tooltip>
            <Tooltip title={t("Re-read all parameters from the backend.")}>
              <IconButton size="small" onClick={loadParameters}><RefreshIcon fontSize="small" /></IconButton>
            </Tooltip>
            {sidebandInfo && (
              <Chip size="small" variant="outlined"
                color={sidebandInfo < 2.5 ? "warning" : "default"}
                label={t("Fringe period {p} px", { p: sidebandInfo })} />
            )}
          </Stack>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {t("1. Click the camera image to place the square on the fringes. 2. Click one of the two side peaks in the spectrum (or Find sideband). 3. Drag dz to refocus.")}
          </Typography>

          <Box sx={{ display: "flex", gap: 1, mb: 1, alignItems: "stretch", flexWrap: { xs: "wrap", md: "nowrap" } }}>
            <Box sx={{ flex: "2 1 0", minWidth: 260, display: "flex", flexDirection: "column", gap: 0.25 }}>
              <Typography variant="caption" color="text.secondary">
                {t("Camera - click to place the square")} ({roi.size} px)
              </Typography>
              <Box sx={{ flex: 1, position: "relative", bgcolor: "#000", borderRadius: 1, overflow: "hidden", minHeight: 200, lineHeight: 0 }}>
                <LiveViewControlWrapper onClick={handleLiveViewClick} onImageLoad={(w, h) => setImageSize({ width: w, height: h })}
                  overlayContent={roiOverlay} enableStageMovement={false} />
              </Box>
            </Box>
            <Box sx={{ flex: "1 1 0", minWidth: 200, display: "flex", flexDirection: "column", gap: 0.25 }}>
              <Typography variant="caption" color="text.secondary">
                {t("Spectrum - click a side peak")} ({cc.centerX}, {cc.centerY})
              </Typography>
              <Box ref={fftImageRef} onClick={handleFftClick}
                sx={{ flex: 1, position: "relative", bgcolor: "#000", borderRadius: 1, overflow: "hidden", cursor: "crosshair", minHeight: 200 }}>
                <img src={stream("fft")} alt="FFT" style={streamImgStyle()} />
                {ccOverlay}
              </Box>
            </Box>
            <Box sx={{ flex: "1 1 0", minWidth: 200, display: "flex", flexDirection: "column", gap: 0.25 }}>
              <Stack direction="row" spacing={0.5} alignItems="center">
                <Chip size="small" label={t("Amplitude")} clickable onClick={() => setShowPhase(false)}
                  color={!showPhase ? "primary" : "default"} variant={!showPhase ? "filled" : "outlined"} />
                <Chip size="small" label={t("Phase")} clickable onClick={() => setShowPhase(true)}
                  color={showPhase ? "secondary" : "default"} variant={showPhase ? "filled" : "outlined"} />
              </Stack>
              <Box sx={{ flex: 1, position: "relative", bgcolor: "#000", borderRadius: 1, overflow: "hidden", minHeight: 200 }}>
                {/* both kept mounted so switching does not reconnect the streams */}
                <img src={stream("mag")} alt="Amplitude" style={streamImgStyle(!showPhase)} />
                <img src={stream("phase")} alt="Phase" style={streamImgStyle(showPhase)} />
              </Box>
            </Box>
          </Box>

          <Paper variant="outlined" sx={{ p: 1.5, mb: 1 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={7}>
                <Typography variant="caption" color="text.secondary">
                  {t("Refocus dz")}: <strong>{(s.dz * 1e6).toFixed(0)} µm</strong>
                </Typography>
                <Slider size="small" value={s.dz} min={-15e-3} max={15e-3} step={10e-6}
                  onChange={(e, v) => dispatch(offAxisHoloSlice.setDz(v))}
                  onChangeCommitted={(e, v) => apiOffAxisHoloControllerSetDz(v).catch((er) => setError(er.message))}
                  valueLabelDisplay="auto" valueLabelFormat={(v) => `${(v * 1e6).toFixed(0)} µm`}
                  marks={[{ value: -15e-3, label: "-15 mm" }, { value: 0, label: "0" }, { value: 15e-3, label: "+15 mm" }]} />
              </Grid>
              <Grid item xs={6} md={2}>
                <FreeNumberField label={t("dz (µm)")} value={s.dz} unitFactor={1e-6} fixedDecimals={0} size="small"
                  onCommit={(v) => { dispatch(offAxisHoloSlice.setDz(v)); apiOffAxisHoloControllerSetDz(v).catch((er) => setError(er.message)); }} />
              </Grid>
              <Grid item xs={6} md={3}>
                <FormControlLabel sx={{ m: 0 }} label={<Typography variant="body2">{t("Unwrap phase")}</Typography>}
                  control={<Switch size="small" checked={s.phaseUnwrapEnabled}
                    onChange={(e) => {
                      dispatch(offAxisHoloSlice.setPhaseUnwrapEnabled(e.target.checked));
                      apiOffAxisHoloControllerSetUnwrapPhase(e.target.checked).catch((er) => setError(er.message));
                    }} />} />
                <FormControlLabel sx={{ m: 0 }} label={<Typography variant="body2">{t("Show the cropped spectrum")}</Typography>}
                  control={<Switch size="small" checked={s.showFftSpace}
                    onChange={(e) => {
                      dispatch(offAxisHoloSlice.setShowFftSpace(e.target.checked));
                      apiOffAxisHoloControllerSetShowFftSpace(e.target.checked).catch((er) => setError(er.message));
                    }} />} />
              </Grid>
            </Grid>
          </Paper>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2">
                <SettingsIcon sx={{ fontSize: 15, mr: 0.5, verticalAlign: "middle" }} />
                {t("Settings - optics, region, sideband, apodization")}
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={1.5} alignItems="center">
                <Grid item xs={6} sm={3}>
                  <FreeNumberField label={t("Wavelength (nm)")} value={s.wavelength} unitFactor={1e-9} fixedDecimals={0} size="small"
                    onCommit={(v) => v > 0 && apiOffAxisHoloControllerSetWavelength(v).then(() => dispatch(offAxisHoloSlice.setWavelength(v)))} />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <FreeNumberField label={t("Pixel size (µm)")} value={s.pixelsize} unitFactor={1e-6} fixedDecimals={2} size="small"
                    onCommit={(v) => v > 0 && apiOffAxisHoloControllerSetPixelsize(v).then(() => dispatch(offAxisHoloSlice.setPixelsize(v)))} />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>{t("Colour channel")}</InputLabel>
                    <Select label={t("Colour channel")} value={s.colorChannel}
                      onChange={(e) => {
                        dispatch(offAxisHoloSlice.setColorChannel(e.target.value));
                        apiOffAxisHoloControllerSetParams({ color_channel: e.target.value }).catch((er) => setError(er.message));
                      }}>
                      <MenuItem value="red">{t("Red")}</MenuItem>
                      <MenuItem value="green">{t("Green")}</MenuItem>
                      <MenuItem value="blue">{t("Blue")}</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>{t("Binning")}</InputLabel>
                    <Select label={t("Binning")} value={s.binning}
                      onChange={(e) => apiOffAxisHoloControllerSetBinning(e.target.value).then(() => dispatch(offAxisHoloSlice.setBinning(e.target.value)))}>
                      {[1, 2, 4].map((b) => <MenuItem key={b} value={b}>{b}×{b}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}><Divider /></Grid>
                <Grid item xs={6} sm={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>{t("Square size (px)")}</InputLabel>
                    <Select label={t("Square size (px)")} value={roi.size} onChange={(e) => applyRoiSize(e.target.value)}>
                      {[256, 512, 1024, 2048].map((v) => <MenuItem key={v} value={v}>{v}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>
                {[["centerX", t("Sideband X")], ["centerY", t("Sideband Y")], ["sizeX", t("Crop width")], ["sizeY", t("Crop height")]].map(([k, label]) => (
                  <Grid item xs={6} sm={2} key={k}>
                    <FreeNumberField label={label} value={cc[k]} fixedDecimals={0} size="small"
                      onCommit={(v) => sendCc({ ...cc, [k]: v })} />
                  </Grid>
                ))}
                <Grid item xs={12} sm={1}>
                  <Typography variant="caption" color="text.secondary">
                    {t("FFT")} {fftShape[1]}×{fftShape[0]}
                  </Typography>
                </Grid>

                <Grid item xs={12}><Divider /></Grid>
                <Grid item xs={12} sm={4}>
                  <FormControlLabel label={<Typography variant="body2">{t("Apodization (soft crop edge)")}</Typography>}
                    control={<Switch size="small" checked={s.apodizationEnabled}
                      onChange={(e) => setApodization(e.target.checked, s.apodizationType, s.apodizationAlpha)} />} />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <FormControl fullWidth size="small" disabled={!s.apodizationEnabled}>
                    <InputLabel>{t("Window")}</InputLabel>
                    <Select label={t("Window")} value={s.apodizationType}
                      onChange={(e) => setApodization(s.apodizationEnabled, e.target.value, s.apodizationAlpha)}>
                      {["tukey", "hann", "hamming", "blackman"].map((w) => <MenuItem key={w} value={w}>{w}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6} sm={5}>
                  <Typography variant="caption" color="text.secondary">Tukey α: {s.apodizationAlpha.toFixed(2)}</Typography>
                  <Slider size="small" min={0} max={1} step={0.01} value={s.apodizationAlpha}
                    disabled={!s.apodizationEnabled || s.apodizationType !== "tukey"}
                    onChange={(e, v) => dispatch(offAxisHoloSlice.setApodizationAlpha(v))}
                    onChangeCommitted={(e, v) => setApodization(s.apodizationEnabled, s.apodizationType, v)} />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </>
      )}

      {tab === 1 && (
        <HoloSnapshot mode="offaxis" baseUrl={baseUrl} defaultWavelength={s.wavelength}
          defaultPixelsize={s.pixelsize} defaultDz={s.dz} />
      )}
      {tab === 2 && <FringeTrace defaultWavelength={s.wavelength} />}
      {tab === 3 && (
        <HoloSimulator initialMode="offaxis" defaultPixelUm={Math.round(s.pixelsize * 1e8) / 100}
          defaultWavelengthNm={Math.round(s.wavelength * 1e9)} />
      )}
      {tab === 4 && <HoloTutorial initialTopic="machzehnder" docsUrl={docsUrl} />}
    </Box>
  );
};

export default OffAxisHoloController;
