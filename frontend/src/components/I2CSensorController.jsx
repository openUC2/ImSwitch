import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Paper,
  Grid,
  Button,
  Typography,
  TextField,
  Box,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Chip,
  Stack,
} from "@mui/material";
import { decode as msgpackDecode } from "@msgpack/msgpack";
import { Line } from "react-chartjs-2";
import "../utils/chartSetup"; // Global Chart.js registration
import { useWebSocket } from "../context/WebSocketContext";

import apiI2CSensorControllerStartPolling from "../backendapi/apiI2CSensorControllerStartPolling.js";
import apiI2CSensorControllerStopPolling from "../backendapi/apiI2CSensorControllerStopPolling.js";
import apiI2CSensorControllerSetPollPeriod from "../backendapi/apiI2CSensorControllerSetPollPeriod.js";
import apiI2CSensorControllerGetStatus from "../backendapi/apiI2CSensorControllerGetStatus.js";
import apiI2CSensorControllerGetBuffer from "../backendapi/apiI2CSensorControllerGetBuffer.js";
import apiI2CSensorControllerGetLatest from "../backendapi/apiI2CSensorControllerGetLatest.js";

const MAX_POINTS = 50; // rolling window length (matches backend bufferSize)
const SIGNAL_NAME = "sigI2CSensorUpdate";

export default function I2CSensorController() {
  const socket = useWebSocket();

  const [samples, setSamples] = useState([]); // rolling array of records
  const [running, setRunning] = useState(false);
  const [available, setAvailable] = useState(true);
  const [periodInput, setPeriodInput] = useState(10);
  const [csvPath, setCsvPath] = useState("");
  const [error, setError] = useState(null);

  // Keep a ref mirror so the socket callback can append without stale closure.
  const samplesRef = useRef([]);
  const appendSample = useCallback((record) => {
    if (!record || typeof record !== "object") return;
    const next = [...samplesRef.current, record].slice(-MAX_POINTS);
    samplesRef.current = next;
    setSamples(next);
  }, []);

  // ── Initial load: status + existing buffer ────────────────────────────
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const status = await apiI2CSensorControllerGetStatus();
        if (mounted && status) {
          setRunning(Boolean(status.running));
          setAvailable(status.available !== false);
          if (status.period) setPeriodInput(status.period);
          if (status.csvPath) setCsvPath(status.csvPath);
        }
        const buf = await apiI2CSensorControllerGetBuffer();
        if (mounted && buf && Array.isArray(buf.buffer)) {
          const seeded = buf.buffer.slice(-MAX_POINTS);
          samplesRef.current = seeded;
          setSamples(seeded);
        }
      } catch (e) {
        if (mounted) setError(e.message || "Failed to load sensor state");
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // ── Live push: subscribe to the backend signal over Socket.IO ─────────
  useEffect(() => {
    if (!socket) return;
    const handler = (raw) => {
      try {
        const msg = msgpackDecode(raw);
        if (!msg || msg.name !== SIGNAL_NAME) return;
        // args = { <paramName>: record }; take the single record dict.
        const args = msg.args || {};
        const record = Array.isArray(args) ? args[0] : Object.values(args)[0];
        appendSample(record);
      } catch (e) {
        // ignore non-msgpack / unrelated frames
      }
    };
    socket.on("signal_msgpack", handler);
    return () => socket.off("signal_msgpack", handler);
  }, [socket, appendSample]);

  // ── Controls ──────────────────────────────────────────────────────────
  const handleStart = async () => {
    setError(null);
    try {
      const r = await apiI2CSensorControllerStartPolling(Number(periodInput));
      setRunning(Boolean(r?.running ?? true));
    } catch (e) {
      setError(e.message || "Failed to start");
    }
  };

  const handleStop = async () => {
    setError(null);
    try {
      await apiI2CSensorControllerStopPolling();
      setRunning(false);
    } catch (e) {
      setError(e.message || "Failed to stop");
    }
  };

  const handleApplyPeriod = async () => {
    setError(null);
    try {
      await apiI2CSensorControllerSetPollPeriod(Number(periodInput));
    } catch (e) {
      setError(e.message || "Failed to set period");
    }
  };

  const handleReadOnce = async () => {
    setError(null);
    try {
      const rec = await apiI2CSensorControllerGetLatest();
      appendSample(rec);
    } catch (e) {
      setError(e.message || "Failed to read");
    }
  };

  // ── Derived chart data ────────────────────────────────────────────────
  const labels = samples.map((s) =>
    s?.timestamp ? new Date(s.timestamp * 1000).toLocaleTimeString() : ""
  );
  const latest = samples.length ? samples[samples.length - 1] : null;

  const chartData = {
    labels,
    datasets: [
      {
        label: "Temperature (°C)",
        data: samples.map((s) => (s?.temperature_c ?? null)),
        borderColor: "#e53935",
        backgroundColor: "rgba(229,57,53,0.15)",
        yAxisID: "yTH",
        tension: 0.25,
        spanGaps: true,
      },
      {
        label: "Humidity (%)",
        data: samples.map((s) => (s?.humidity_pct ?? null)),
        borderColor: "#1e88e5",
        backgroundColor: "rgba(30,136,229,0.15)",
        yAxisID: "yTH",
        tension: 0.25,
        spanGaps: true,
      },
      {
        label: "Light (lux)",
        data: samples.map((s) => (s?.lux ?? null)),
        borderColor: "#fb8c00",
        backgroundColor: "rgba(251,140,0,0.15)",
        yAxisID: "yLux",
        tension: 0.25,
        spanGaps: true,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    stacked: false,
    plugins: { legend: { position: "top" } },
    scales: {
      yTH: {
        type: "linear",
        position: "left",
        title: { display: true, text: "°C / %RH" },
      },
      yLux: {
        type: "linear",
        position: "right",
        title: { display: true, text: "lux" },
        grid: { drawOnChartArea: false },
      },
    },
  };

  const fmt = (v, unit) =>
    v === null || v === undefined ? "—" : `${v} ${unit}`;

  return (
    <Paper sx={{ p: 2, m: 1 }}>
      <Typography variant="h5" gutterBottom>
        Environmental Sensors (I2C over CAN)
      </Typography>

      {!available && (
        <Chip
          color="warning"
          label="ESP32 I2C interface not available — check the CAN connection"
          sx={{ mb: 2 }}
        />
      )}
      {error && (
        <Chip color="error" label={error} sx={{ mb: 2 }} onDelete={() => setError(null)} />
      )}

      <Grid container spacing={2}>
        {/* Live values */}
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardHeader title="Current" subheader={running ? "Polling" : "Idle"} />
            <CardContent>
              <Stack spacing={1}>
                <Typography variant="h6">
                  🌡 {fmt(latest?.temperature_c, "°C")}
                </Typography>
                <Typography variant="h6">
                  💧 {fmt(latest?.humidity_pct, "%")}
                </Typography>
                <Typography variant="h6">
                  💡 {fmt(latest?.lux, "lux")}
                </Typography>
                {latest && (
                  <Typography variant="caption" color="text.secondary">
                    ch0={latest.ch0_full ?? "—"} · ch1={latest.ch1_ir ?? "—"} ·{" "}
                    {latest.datetime || ""}
                  </Typography>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Controls */}
        <Grid item xs={12} md={8}>
          <Card variant="outlined">
            <CardHeader title="Controls" />
            <CardContent>
              <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
                <Button
                  variant="contained"
                  color="success"
                  onClick={handleStart}
                  disabled={running}
                >
                  Start
                </Button>
                <Button
                  variant="contained"
                  color="error"
                  onClick={handleStop}
                  disabled={!running}
                >
                  Stop
                </Button>
                <Button variant="outlined" onClick={handleReadOnce}>
                  Read once
                </Button>
                <Divider orientation="vertical" flexItem />
                <TextField
                  label="Poll period (s)"
                  type="number"
                  size="small"
                  value={periodInput}
                  onChange={(e) => setPeriodInput(e.target.value)}
                  inputProps={{ min: 0.2, step: 0.5 }}
                  sx={{ width: 140 }}
                />
                <Button variant="outlined" onClick={handleApplyPeriod}>
                  Apply period
                </Button>
              </Stack>
              {csvPath && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                  Logging to: {csvPath}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Chart */}
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardHeader
              title={`Rolling window (last ${MAX_POINTS} samples)`}
              subheader={`${samples.length} points`}
            />
            <CardContent>
              <Box sx={{ height: 360 }}>
                <Line data={chartData} options={chartOptions} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Paper>
  );
}
