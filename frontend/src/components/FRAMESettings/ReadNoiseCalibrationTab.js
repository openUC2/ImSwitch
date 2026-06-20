// src/components/FRAMESettings/ReadNoiseCalibrationTab.js
//
// Guided wizard for camera read-noise / gain calibration. Ports the analysis of
// NanoImagingPack's cal_readnoise (a photon-transfer-curve fit) into ImSwitch:
//
//   1. Setup        -- name the session, pick a detector, choose frame counts.
//   2. Bright frames-- tune exposure/gain against a live histogram so the bright
//                      frames span the gray range, then capture a bright stack.
//   3. Dark frames  -- switch off all illumination, confirm darkness, capture a
//                      dark stack, then restore the lights.
//   4. Compute      -- run cal_readnoise; show gain/read-noise/offset + charts.
//   5. Notes        -- store a free-text comment.
//
// Images, results and the comment are stored per session under
// recordings/readnoise_calibration/<id>/ and older sessions can be browsed below.
import React, { useCallback, useEffect, useState } from "react";
import { useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  Collapse,
  Divider,
  FormControl,
  FormControlLabel,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Step,
  StepLabel,
  Stepper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import Plot from "react-plotly.js";

import LiveViewControlWrapper from "../../axon/LiveViewControlWrapper";
import * as connectionSettingsSlice from "../../state/slices/ConnectionSettingsSlice";

import apiGetStatus from "../../backendapi/apiReadNoiseCalibrationControllerGetStatus";
import apiGetProgress from "../../backendapi/apiReadNoiseCalibrationControllerGetProgress";
import apiGetDetectorSettings from "../../backendapi/apiReadNoiseCalibrationControllerGetDetectorSettings";
import apiSetDetectorSetting from "../../backendapi/apiReadNoiseCalibrationControllerSetDetectorSetting";
import apiGetHistogram from "../../backendapi/apiReadNoiseCalibrationControllerGetHistogram";
import apiSetIllumination from "../../backendapi/apiReadNoiseCalibrationControllerSetIllumination";
import apiStartSession from "../../backendapi/apiReadNoiseCalibrationControllerStartSession";
import apiAcquireFrames from "../../backendapi/apiReadNoiseCalibrationControllerAcquireFrames";
import apiStopAcquisition from "../../backendapi/apiReadNoiseCalibrationControllerStopAcquisition";
import apiComputeCalibration from "../../backendapi/apiReadNoiseCalibrationControllerComputeCalibration";
import apiListSessions from "../../backendapi/apiReadNoiseCalibrationControllerListSessions";
import apiGetSession from "../../backendapi/apiReadNoiseCalibrationControllerGetSession";
import apiSaveComment from "../../backendapi/apiReadNoiseCalibrationControllerSaveComment";
import apiDeleteSession from "../../backendapi/apiReadNoiseCalibrationControllerDeleteSession";

const STEPS = ["Setup", "Bright frames", "Dark frames", "Compute", "Notes"];

// midpoints of histogram bin edges (edges length = counts length + 1)
const binMids = (edges) =>
  Array.isArray(edges) && edges.length > 1
    ? edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2)
    : [];

// -------------------------------------------------------------------------- //
// Interactive result charts (shared by the Compute step and the browse view) //
// -------------------------------------------------------------------------- //
const CalibrationCharts = ({ charts }) => {
  if (!charts) return null;
  const ptc = charts.ptc || {};
  const lin = charts.linearity || {};
  const hist = charts.histograms || {};

  const ptcTraces = [];
  if (ptc.meanADU && ptc.varADU) {
    ptcTraces.push({
      x: ptc.meanADU,
      y: ptc.varADU,
      type: "scattergl",
      mode: "markers",
      marker: { color: "#1f77b4", size: 5 },
      name: "Brightness bins",
    });
  }
  if (ptc.fitX && ptc.fitY) {
    ptcTraces.push({
      x: ptc.fitX,
      y: ptc.fitY,
      type: "scatter",
      mode: "lines",
      line: { color: "#d62728", width: 2 },
      name: "Gain fit",
    });
  }
  if (ptc.fitX && ptc.errUpper) {
    ptcTraces.push({
      x: ptc.fitX,
      y: ptc.errUpper,
      type: "scatter",
      mode: "lines",
      line: { color: "#d62728", width: 1, dash: "dash" },
      name: "Error",
    });
  }
  if (ptc.fitX && ptc.errLower) {
    ptcTraces.push({
      x: ptc.fitX,
      y: ptc.errLower,
      type: "scatter",
      mode: "lines",
      line: { color: "#d62728", width: 1, dash: "dash" },
      name: "Error (lower)",
      showlegend: false,
    });
  }

  const histTraces = [];
  if (hist.darkEdges && hist.darkCounts) {
    histTraces.push({
      x: binMids(hist.darkEdges),
      y: hist.darkCounts,
      type: "scatter",
      mode: "lines",
      fill: "tozeroy",
      line: { color: "#7f7f7f" },
      name: "Dark",
    });
  }
  if (hist.brightEdges && hist.brightCounts) {
    histTraces.push({
      x: binMids(hist.brightEdges),
      y: hist.brightCounts,
      type: "scatter",
      mode: "lines",
      fill: "tozeroy",
      line: { color: "#2ca02c" },
      name: "Bright",
    });
  }

  const common = { config: { displayModeBar: false }, style: { width: "100%" }, useResizeHandler: true };

  return (
    <Grid container spacing={2}>
      {ptcTraces.length > 0 && (
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>
            Photon transfer curve
          </Typography>
          <Plot
            data={ptcTraces}
            layout={{
              margin: { t: 10, r: 10, b: 45, l: 60 },
              xaxis: { title: "Pixel brightness / ADU" },
              yaxis: { title: "Pixel variance / ADU²" },
              height: 300,
              legend: { orientation: "h", y: -0.25 },
            }}
            {...common}
          />
        </Grid>
      )}
      {lin.x && lin.devPercent && (
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>
            Deviation from linearity
          </Typography>
          <Plot
            data={[
              {
                x: lin.x,
                y: lin.devPercent,
                type: "scatter",
                mode: "lines",
                line: { color: "#9467bd" },
                name: "Deviation",
              },
            ]}
            layout={{
              margin: { t: 10, r: 10, b: 45, l: 60 },
              xaxis: { title: "Pixel brightness / ADU" },
              yaxis: { title: "Deviation / %" },
              height: 300,
            }}
            {...common}
          />
        </Grid>
      )}
      {histTraces.length > 0 && (
        <Grid item xs={12}>
          <Typography variant="subtitle2" gutterBottom>
            Dark &amp; bright histograms
          </Typography>
          <Plot
            data={histTraces}
            layout={{
              margin: { t: 10, r: 10, b: 45, l: 60 },
              xaxis: { title: "Signal intensity / ADU" },
              yaxis: { title: "Counts / frame", type: "log" },
              height: 300,
              legend: { orientation: "h", y: -0.25 },
            }}
            {...common}
          />
        </Grid>
      )}
    </Grid>
  );
};

// -------------------------------------------------------------------------- //
// Metrics table (key/value with definition tooltips)                          //
// -------------------------------------------------------------------------- //
const MetricsTable = ({ metrics, definitions }) => {
  if (!metrics || Object.keys(metrics).length === 0) return null;
  return (
    <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
      <Table size="small">
        <TableBody>
          {Object.entries(metrics).map(([key, value]) => (
            <TableRow key={key}>
              <TableCell sx={{ fontWeight: 500, width: "55%" }}>
                {definitions && definitions[key] ? (
                  <Tooltip title={definitions[key]} placement="top-start">
                    <span style={{ borderBottom: "1px dotted #999", cursor: "help" }}>{key}</span>
                  </Tooltip>
                ) : (
                  key
                )}
              </TableCell>
              <TableCell>{String(value)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// -------------------------------------------------------------------------- //
// Live tuning histogram                                                       //
// -------------------------------------------------------------------------- //
const LiveHistogram = ({ histogram, title }) => {
  if (!histogram || histogram.status !== "ok") {
    return (
      <Typography variant="body2" color="textSecondary">
        Waiting for camera frames&hellip;
      </Typography>
    );
  }
  const x = binMids(histogram.binEdges);
  const clipPct = (histogram.clipFraction || 0) * 100;
  const satPct = (histogram.saturationFraction || 0) * 100;
  return (
    <Box>
      <Plot
        data={[
          {
            x,
            y: histogram.counts,
            type: "bar",
            marker: { color: "#1f77b4" },
            name: "histogram",
          },
        ]}
        layout={{
          margin: { t: 10, r: 10, b: 35, l: 50 },
          xaxis: { title: title || "Pixel value / ADU" },
          yaxis: { title: "Count" },
          height: 220,
          bargap: 0,
        }}
        config={{ displayModeBar: false }}
        style={{ width: "100%" }}
        useResizeHandler
      />
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 1 }}>
        <Chip size="small" label={`min ${histogram.min?.toFixed(0)}`} />
        <Chip size="small" label={`mean ${histogram.mean?.toFixed(1)}`} />
        <Chip size="small" label={`max ${histogram.max?.toFixed(0)}`} />
        <Chip size="small" label={`std ${histogram.std?.toFixed(1)}`} />
        <Chip
          size="small"
          color={satPct > 0.5 ? "error" : "default"}
          label={`saturated ${satPct.toFixed(2)}%`}
        />
      </Box>
      {clipPct > 1 && satPct > 0.5 && (
        <Alert severity="warning" sx={{ mt: 1 }}>
          {satPct.toFixed(1)}% of pixels are saturated &mdash; reduce exposure or gain so the
          bright peak stays below the detector maximum.
        </Alert>
      )}
    </Box>
  );
};

// -------------------------------------------------------------------------- //
// Main wizard                                                                 //
// -------------------------------------------------------------------------- //
const ReadNoiseCalibrationTab = () => {
  const conn = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  const dataUrl = useCallback(
    (relPath, fname) => `${conn.ip}:${conn.apiPort}/imswitch/data/${relPath}/${fname}`,
    [conn.ip, conn.apiPort],
  );

  const [available, setAvailable] = useState(true);
  const [activeStep, setActiveStep] = useState(0);
  const [detectors, setDetectors] = useState([]);
  const [illumination, setIllumination] = useState([]);
  const [session, setSession] = useState(null);

  // setup form
  const [form, setForm] = useState({
    name: "",
    detectorName: "",
    nBright: 20,
    nDark: 20,
    numBins: 100,
  });

  // tuning
  const [settings, setSettings] = useState({});
  const [settingDrafts, setSettingDrafts] = useState({});
  const [histogram, setHistogram] = useState(null);
  const [illuminationOff, setIlluminationOff] = useState(false);

  // acquisition
  const [acquiring, setAcquiring] = useState(false);
  const [acqProgress, setAcqProgress] = useState({ running: false, step: 0, total: 0, message: "" });
  const [actionError, setActionError] = useState(null);

  // compute
  const [advNumBins, setAdvNumBins] = useState("");
  const [advLow, setAdvLow] = useState("");
  const [advHigh, setAdvHigh] = useState("");
  const [advSat, setAdvSat] = useState(false);
  const [computing, setComputing] = useState(false);
  const [computeError, setComputeError] = useState(null);
  const [result, setResult] = useState(null);

  // notes
  const [comment, setComment] = useState("");
  const [noteSaved, setNoteSaved] = useState(false);

  // browse
  const [sessions, setSessions] = useState([]);
  const [expanded, setExpanded] = useState(null);
  const [detail, setDetail] = useState(null);

  const brightDone = (session?.brightCount || 0) > 0;
  const darkDone = (session?.darkCount || 0) > 0;

  const refreshSessions = useCallback(async () => {
    try {
      const r = await apiListSessions();
      setSessions(r?.sessions || []);
    } catch (e) {
      // controller may not be loaded
    }
  }, []);

  // ----------------------- mount: status + resume ------------------------ //
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const st = await apiGetStatus();
        if (cancelled) return;
        setAvailable(true);
        setDetectors(st.detectors || []);
        setIllumination(st.illumination || []);
        setForm((f) => ({
          ...f,
          detectorName: f.detectorName || st.currentDetector || (st.detectors || [])[0] || "",
        }));
        if (st.session) {
          setSession(st.session);
          // resume at the earliest unfinished step
          const s = st.session;
          if ((s.brightCount || 0) === 0) setActiveStep(1);
          else if ((s.darkCount || 0) === 0) setActiveStep(2);
          else setActiveStep(3);
        }
      } catch (e) {
        if (!cancelled) setAvailable(false);
      }
    })();
    refreshSessions();
    return () => {
      cancelled = true;
    };
  }, [refreshSessions]);

  // ----------------------- live histogram polling ------------------------ //
  useEffect(() => {
    const shouldPoll = (activeStep === 1 || activeStep === 2) && available && !acquiring;
    if (!shouldPoll) return undefined;
    let cancelled = false;
    let inflight = false;
    const tick = async () => {
      if (inflight) return;
      inflight = true;
      try {
        const h = await apiGetHistogram(form.detectorName || null, 160);
        if (!cancelled) setHistogram(h);
      } catch (e) {
        /* transient */
      } finally {
        inflight = false;
      }
    };
    tick();
    const id = setInterval(tick, 1200);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [activeStep, available, acquiring, form.detectorName]);

  // ----------------------- detector settings load ------------------------ //
  useEffect(() => {
    if (activeStep !== 1 || !available) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await apiGetDetectorSettings(form.detectorName || null);
        if (cancelled) return;
        setSettings(r.settings || {});
        setSettingDrafts(
          Object.fromEntries(Object.entries(r.settings || {}).map(([k, v]) => [k, v])),
        );
      } catch (e) {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeStep, available, form.detectorName]);

  // ----------------------- acquisition progress -------------------------- //
  useEffect(() => {
    if (!acquiring) return undefined;
    let cancelled = false;
    let finished = false; // process completion exactly once
    const id = setInterval(async () => {
      if (finished) return;
      try {
        const p = await apiGetProgress();
        if (cancelled || finished) return;
        setAcqProgress(p);
        if (!p.running) {
          finished = true;
          if (p.error) {
            setActionError(p.message);
          } else if (p.done && p.phase) {
            // Optimistic local sync straight from the progress payload, so the
            // wizard can advance even if the authoritative getStatus below blips.
            setSession((s) =>
              s ? { ...s, [`${p.phase}Count`]: p.total || s[`${p.phase}Count`] || 1 } : s,
            );
          }
          // Sync the active session (brightCount/darkCount) BEFORE flipping
          // `acquiring`. Flipping it first triggers this effect's cleanup, which
          // sets cancelled=true; the `await apiGetStatus()` below yields long
          // enough for React to run that cleanup, so the terminal setSession was
          // being dropped -- leaving brightDone/darkDone false and the Next
          // button stuck disabled. Refreshing first keeps cancelled=false here.
          try {
            const st = await apiGetStatus();
            if (!cancelled && st && st.session) setSession(st.session);
          } catch (e) {
            /* keep the optimistic value */
          }
          clearInterval(id);
          setAcquiring(false);
        }
      } catch (e) {
        /* transient */
      }
    }, 600);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [acquiring]);

  // ------------------------------ actions -------------------------------- //
  const applySetting = async (name, value) => {
    try {
      const r = await apiSetDetectorSetting({ detectorName: form.detectorName || null, name, value });
      if (r.status === "ok") {
        setSettings(r.settings || {});
        setSettingDrafts((d) => ({ ...d, ...(r.settings || {}) }));
      }
    } catch (e) {
      setActionError(`Could not set ${name}: ${e.message}`);
    }
  };

  const startSession = async () => {
    setActionError(null);
    try {
      const r = await apiStartSession({
        name: form.name,
        detectorName: form.detectorName || null,
        nBright: form.nBright,
        nDark: form.nDark,
        numBins: form.numBins,
      });
      if (r.status === "ok") {
        setSession(r.session);
        setResult(null);
        setComment("");
        setNoteSaved(false);
        setActiveStep(1);
      } else {
        setActionError(r.message || "Could not start session");
      }
    } catch (e) {
      setActionError(`Could not start session: ${e.message}`);
    }
  };

  const capture = async (kind) => {
    setActionError(null);
    try {
      const r = await apiAcquireFrames(kind);
      if (r.status === "started") {
        setAcquiring(true);
        setAcqProgress({ running: true, phase: kind, step: 0, total: r.count, message: "Starting…" });
      } else {
        setActionError(r.message || "Capture failed");
      }
    } catch (e) {
      setActionError(`Capture failed: ${e.message}`);
    }
  };

  const stopCapture = async () => {
    try {
      await apiStopAcquisition();
    } catch (e) {
      /* ignore */
    }
  };

  const toggleIllumination = async (state) => {
    setActionError(null);
    try {
      await apiSetIllumination(state);
      setIlluminationOff(state === "off");
      const st = await apiGetStatus();
      setIllumination(st.illumination || []);
    } catch (e) {
      setActionError(`Illumination ${state} failed: ${e.message}`);
    }
  };

  const runCompute = async () => {
    setComputing(true);
    setComputeError(null);
    try {
      const r = await apiComputeCalibration({
        sessionId: session?.sessionId,
        numBins: advNumBins !== "" ? Number(advNumBins) : null,
        validRangeLow: advLow,
        validRangeHigh: advHigh,
        saturationImage: advSat,
      });
      if (r.status === "ok") {
        setResult(r.result);
      } else {
        setComputeError(r.message || "Computation failed");
      }
    } catch (e) {
      setComputeError(`Computation failed: ${e.message}`);
    } finally {
      setComputing(false);
    }
  };

  const saveNote = async () => {
    if (!session) return;
    try {
      await apiSaveComment(session.sessionId, comment);
      setNoteSaved(true);
      refreshSessions();
    } catch (e) {
      setActionError(`Could not save note: ${e.message}`);
    }
  };

  const finishWizard = async () => {
    if (illuminationOff) await toggleIllumination("restore");
    await refreshSessions();
    setSession(null);
    setResult(null);
    setHistogram(null);
    setActiveStep(0);
    setForm((f) => ({ ...f, name: "" }));
  };

  const openDetail = async (sessionId) => {
    if (expanded === sessionId) {
      setExpanded(null);
      setDetail(null);
      return;
    }
    setExpanded(sessionId);
    setDetail(null);
    try {
      const r = await apiGetSession(sessionId);
      setDetail(r);
    } catch (e) {
      setDetail({ status: "error", message: e.message });
    }
  };

  const removeSession = async (sessionId) => {
    try {
      await apiDeleteSession(sessionId);
      if (expanded === sessionId) {
        setExpanded(null);
        setDetail(null);
      }
      refreshSessions();
    } catch (e) {
      setActionError(`Could not delete: ${e.message}`);
    }
  };

  // ------------------------------ render --------------------------------- //
  if (!available) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Read Noise Calibration
        </Typography>
        <Alert severity="warning">
          The ReadNoiseCalibrationController is not available on this backend. Make sure the
          server was started with this controller registered.
        </Alert>
      </Paper>
    );
  }

  const progressPct =
    acqProgress.total > 0 ? Math.round((acqProgress.step / acqProgress.total) * 100) : 0;

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Read Noise Calibration
      </Typography>
      <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
        Characterise the camera by fitting a photon-transfer curve: capture a series of bright
        frames spanning the gray range plus a series of dark frames, then derive the{" "}
        <strong>gain (e&#8315;/ADU)</strong>, <strong>read noise (e&#8315; RMS)</strong> and{" "}
        <strong>offset (ADU)</strong>. To convert images to photoelectrons later:
        (image &minus; offset) &times; gain.
      </Typography>

      <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 3 }}>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {actionError && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setActionError(null)}>
          {actionError}
        </Alert>
      )}

      {/* ---------------------------- Step 0: setup ---------------------------- */}
      {activeStep === 0 && (
        <Box>
          <Alert severity="info" sx={{ mb: 2 }}>
            Pick the camera to calibrate and how many frames to average. ~20 bright and ~20 dark
            frames are recommended. All images and results are stored under{" "}
            <code>recordings/readnoise_calibration/</code>.
          </Alert>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Session name (optional)"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                fullWidth
                helperText="Added to the folder name, e.g. tritium_1s_gain20"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Detector</InputLabel>
                <Select
                  value={form.detectorName}
                  label="Detector"
                  onChange={(e) => setForm({ ...form, detectorName: e.target.value })}
                >
                  {detectors.map((d) => (
                    <MenuItem key={d} value={d}>
                      {d}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={4}>
              <TextField
                label="Bright frames"
                type="number"
                value={form.nBright}
                onChange={(e) => setForm({ ...form, nBright: parseInt(e.target.value, 10) || 1 })}
                inputProps={{ min: 2, step: 1 }}
                fullWidth
              />
            </Grid>
            <Grid item xs={6} sm={4}>
              <TextField
                label="Dark frames"
                type="number"
                value={form.nDark}
                onChange={(e) => setForm({ ...form, nDark: parseInt(e.target.value, 10) || 1 })}
                inputProps={{ min: 2, step: 1 }}
                fullWidth
              />
            </Grid>
            <Grid item xs={6} sm={4}>
              <TextField
                label="Histogram bins"
                type="number"
                value={form.numBins}
                onChange={(e) => setForm({ ...form, numBins: parseInt(e.target.value, 10) || 10 })}
                inputProps={{ min: 10, step: 10 }}
                fullWidth
              />
            </Grid>
          </Grid>
          <Box sx={{ mt: 2 }}>
            <Button variant="contained" onClick={startSession} disabled={!form.detectorName}>
              Start session
            </Button>
          </Box>
        </Box>
      )}

      {/* ------------------------- Step 1: bright frames ----------------------- */}
      {activeStep === 1 && (
        <Box>
          <Alert severity="info" sx={{ mb: 2 }}>
            Adjust exposure / gain so the live histogram spreads across the range (ideally many
            gray levels populated) without saturating the bright peak. Use a smooth, slightly
            out-of-focus bright scene. Then capture the bright stack.
          </Alert>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, overflow: "hidden", maxHeight: 360 }}>
                <LiveViewControlWrapper useFastMode={true} />
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <LiveHistogram histogram={histogram} title="Bright pixel value / ADU" />
              <Grid container spacing={1} sx={{ mt: 1 }}>
                {["exposure", "gain"].map((key) =>
                  settings[key] !== undefined ? (
                    <Grid item xs={6} key={key}>
                      <TextField
                        label={key}
                        type="number"
                        size="small"
                        value={settingDrafts[key] ?? ""}
                        onChange={(e) => setSettingDrafts((d) => ({ ...d, [key]: e.target.value }))}
                        onBlur={(e) => applySetting(key, e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") applySetting(key, e.target.value);
                        }}
                        fullWidth
                      />
                    </Grid>
                  ) : null,
                )}
              </Grid>
            </Grid>
          </Grid>

          {acquiring && acqProgress.phase === "bright" ? (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2">{acqProgress.message}</Typography>
              <LinearProgress variant="determinate" value={progressPct} sx={{ mt: 0.5 }} />
              <Button variant="outlined" color="warning" onClick={stopCapture} sx={{ mt: 1 }}>
                Stop
              </Button>
            </Box>
          ) : (
            <Box sx={{ mt: 2, display: "flex", gap: 1, alignItems: "center" }}>
              <Button variant="contained" onClick={() => capture("bright")}>
                Capture {form.nBright} bright frames
              </Button>
              {brightDone && <Chip color="success" label={`${session.brightCount} bright frames captured`} />}
            </Box>
          )}

          <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
            <Button onClick={() => setActiveStep(0)}>Back</Button>
            <Button variant="contained" onClick={() => setActiveStep(2)} disabled={!brightDone}>
              Next: dark frames
            </Button>
          </Box>
        </Box>
      )}

      {/* -------------------------- Step 2: dark frames ------------------------ */}
      {activeStep === 2 && (
        <Box>
          <Alert severity="info" sx={{ mb: 2 }}>
            Switch off all illumination and confirm the histogram collapses to a narrow peak near
            the offset, then capture the dark stack with the <strong>same exposure/gain</strong>.
          </Alert>
          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 2, alignItems: "center" }}>
            <Button
              variant={illuminationOff ? "outlined" : "contained"}
              color="primary"
              onClick={() => toggleIllumination("off")}
            >
              Turn off illumination
            </Button>
            <Button variant="outlined" onClick={() => toggleIllumination("restore")} disabled={!illuminationOff}>
              Restore illumination
            </Button>
            {illumination.map((s) => (
              <Chip
                key={`${s.type}:${s.name}`}
                size="small"
                color={s.enabled ? "warning" : "default"}
                label={`${s.name}: ${s.enabled ? "on" : "off"}`}
              />
            ))}
          </Box>
          <LiveHistogram histogram={histogram} title="Dark pixel value / ADU" />
          {histogram && histogram.status === "ok" && histogram.std > 30 && (
            <Alert severity="warning" sx={{ mt: 1 }}>
              The dark frame still shows a wide spread (std {histogram.std?.toFixed(1)}). Make sure
              every light source is off and the path is covered.
            </Alert>
          )}

          {acquiring && acqProgress.phase === "dark" ? (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2">{acqProgress.message}</Typography>
              <LinearProgress variant="determinate" value={progressPct} sx={{ mt: 0.5 }} />
              <Button variant="outlined" color="warning" onClick={stopCapture} sx={{ mt: 1 }}>
                Stop
              </Button>
            </Box>
          ) : (
            <Box sx={{ mt: 2, display: "flex", gap: 1, alignItems: "center" }}>
              <Button variant="contained" onClick={() => capture("dark")}>
                Capture {form.nDark} dark frames
              </Button>
              {darkDone && <Chip color="success" label={`${session.darkCount} dark frames captured`} />}
            </Box>
          )}

          <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
            <Button onClick={() => setActiveStep(1)}>Back</Button>
            <Button variant="contained" onClick={() => setActiveStep(3)} disabled={!darkDone}>
              Next: compute
            </Button>
          </Box>
        </Box>
      )}

      {/* --------------------------- Step 3: compute -------------------------- */}
      {activeStep === 3 && (
        <Box>
          <Alert severity="info" sx={{ mb: 2 }}>
            Run the photon-transfer analysis on the captured stacks. Optional advanced settings let
            you override the histogram bins or restrict the fit range (in ADU).
          </Alert>
          <Grid container spacing={2} sx={{ mb: 1 }}>
            <Grid item xs={6} sm={3}>
              <TextField label="Bins (override)" type="number" size="small" value={advNumBins}
                onChange={(e) => setAdvNumBins(e.target.value)} fullWidth />
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField label="Fit from (ADU)" type="number" size="small" value={advLow}
                onChange={(e) => setAdvLow(e.target.value)} fullWidth />
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField label="Fit to (ADU)" type="number" size="small" value={advHigh}
                onChange={(e) => setAdvHigh(e.target.value)} fullWidth />
            </Grid>
            <Grid item xs={6} sm={3}>
              <FormControlLabel
                control={<Checkbox checked={advSat} onChange={(e) => setAdvSat(e.target.checked)} />}
                label="Saturation metrics"
              />
            </Grid>
          </Grid>
          <Button variant="contained" onClick={runCompute} disabled={computing}>
            {computing ? "Computing…" : "Compute calibration"}
          </Button>
          {computeError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {computeError}
            </Alert>
          )}

          {result && (
            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1 }}>
                <Chip color="primary" label={`Gain ${result.gain?.toFixed(3)} e−/ADU`} />
                <Chip color="primary" label={`Read noise ${result.readnoise?.toFixed(2)} e−`} />
                <Chip color="primary" label={`Offset ${result.offset?.toFixed(1)} ADU`} />
              </Box>
              <CalibrationCharts charts={result.charts} />
              <MetricsTable metrics={result.metrics} definitions={result.definitions} />
            </Box>
          )}

          <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
            <Button onClick={() => setActiveStep(2)}>Back</Button>
            <Button variant="contained" onClick={() => setActiveStep(4)} disabled={!result}>
              Next: notes
            </Button>
          </Box>
        </Box>
      )}

      {/* ---------------------------- Step 4: notes --------------------------- */}
      {activeStep === 4 && (
        <Box>
          <Alert severity="info" sx={{ mb: 2 }}>
            Add any notes about this calibration (conditions, sample, light source). They are saved
            to <code>comment.txt</code> in the session folder.
          </Alert>
          <TextField
            label="Notes"
            multiline
            minRows={4}
            value={comment}
            onChange={(e) => {
              setComment(e.target.value);
              setNoteSaved(false);
            }}
            fullWidth
          />
          <Box sx={{ mt: 2, display: "flex", gap: 1, alignItems: "center" }}>
            <Button variant="outlined" onClick={saveNote} disabled={!session}>
              Save note
            </Button>
            {noteSaved && <Chip color="success" label="Saved" />}
            <Button variant="contained" onClick={finishWizard}>
              Finish
            </Button>
          </Box>
        </Box>
      )}

      {/* --------------------------- Browse section --------------------------- */}
      <Divider sx={{ my: 3 }} />
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
        <Typography variant="subtitle1">Previous calibrations</Typography>
        <Button size="small" onClick={refreshSessions}>
          Refresh
        </Button>
      </Box>
      {sessions.length === 0 ? (
        <Typography variant="body2" color="textSecondary">
          No stored calibrations yet.
        </Typography>
      ) : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Session</TableCell>
                <TableCell>Detector</TableCell>
                <TableCell>Gain</TableCell>
                <TableCell>Read noise</TableCell>
                <TableCell>Offset</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sessions.map((s) => (
                <React.Fragment key={s.sessionId}>
                  <TableRow hover>
                    <TableCell>
                      <Typography variant="body2">{s.name || s.sessionId}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {s.created} &middot; {s.brightCount}/{s.darkCount} frames
                      </Typography>
                    </TableCell>
                    <TableCell>{s.detector}</TableCell>
                    <TableCell>{s.gain != null ? Number(s.gain).toFixed(3) : "—"}</TableCell>
                    <TableCell>{s.readnoise != null ? Number(s.readnoise).toFixed(2) : "—"}</TableCell>
                    <TableCell>{s.offset != null ? Number(s.offset).toFixed(1) : "—"}</TableCell>
                    <TableCell align="right">
                      <Button size="small" onClick={() => openDetail(s.sessionId)}>
                        {expanded === s.sessionId ? "Hide" : "Details"}
                      </Button>
                      <Button size="small" color="error" onClick={() => removeSession(s.sessionId)}>
                        Delete
                      </Button>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={6} sx={{ py: 0, borderBottom: expanded === s.sessionId ? undefined : "none" }}>
                      <Collapse in={expanded === s.sessionId} timeout="auto" unmountOnExit>
                        <Box sx={{ py: 2 }}>
                          {!detail ? (
                            <Typography variant="body2" color="textSecondary">
                              Loading&hellip;
                            </Typography>
                          ) : detail.status === "error" ? (
                            <Alert severity="error">{detail.message}</Alert>
                          ) : (
                            <Box>
                              {detail.comment && (
                                <Alert severity="info" sx={{ mb: 2, whiteSpace: "pre-wrap" }}>
                                  {detail.comment}
                                </Alert>
                              )}
                              {detail.result ? (
                                <>
                                  <CalibrationCharts charts={detail.result.charts} />
                                  <MetricsTable
                                    metrics={detail.result.metrics}
                                    definitions={detail.result.definitions}
                                  />
                                  {Array.isArray(detail.result.figures) &&
                                    detail.result.figures.length > 0 && (
                                      <Box sx={{ mt: 1, display: "flex", gap: 1, flexWrap: "wrap" }}>
                                        {detail.result.figures.map((f) => (
                                          <Button
                                            key={f}
                                            size="small"
                                            href={dataUrl(detail.dataRelPath, `figures/${f}`)}
                                            target="_blank"
                                            rel="noreferrer"
                                          >
                                            {f}
                                          </Button>
                                        ))}
                                        <Button
                                          size="small"
                                          href={dataUrl(detail.dataRelPath, "calibration_results.txt")}
                                          target="_blank"
                                          rel="noreferrer"
                                        >
                                          results.txt
                                        </Button>
                                      </Box>
                                    )}
                                </>
                              ) : (
                                <Typography variant="body2" color="textSecondary">
                                  This session has no computed result.
                                </Typography>
                              )}
                            </Box>
                          )}
                        </Box>
                      </Collapse>
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
};

export default ReadNoiseCalibrationTab;
