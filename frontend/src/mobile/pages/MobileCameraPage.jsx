// Kiosk camera view: a deliberately basic MJPEG live stream (plain <img>,
// no socket/decode cost — ideal for the Pi's kiosk browser) with start/stop,
// a detector picker and a snap button.
import { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import StopRoundedIcon from "@mui/icons-material/StopRounded";
import CameraAltRoundedIcon from "@mui/icons-material/CameraAltRounded";
import VideocamOffRoundedIcon from "@mui/icons-material/VideocamOffRounded";
import { enqueueSnackbar } from "notistack";

import MobilePage from "../components/MobilePage";
import { getConnectionSettingsState } from "../../state/slices/ConnectionSettingsSlice";
import apiSettingsControllerGetDetectorNames from "../../backendapi/apiSettingsControllerGetDetectorNames";
import apiLiveViewControllerStartLiveView from "../../backendapi/apiLiveViewControllerStartLiveView";
import apiLiveViewControllerStopLiveView from "../../backendapi/apiLiveViewControllerStopLiveView";
import apiLiveViewControllerGetActiveStreams from "../../backendapi/apiLiveViewControllerGetActiveStreams";
import {
  apiRecordingControllerStartSnap,
  apiRecordingControllerGetSnapStatus,
} from "../../backendapi/apiRecordingControllerSnapJob";

const SNAP_POLL_MS = 800;
const SNAP_POLL_MAX = 40;

const MobileCameraPage = () => {
  const connectionSettings = useSelector(getConnectionSettingsState);

  const [detectors, setDetectors] = useState([]);
  const [selectedDetector, setSelectedDetector] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [starting, setStarting] = useState(false);
  const [longExposure, setLongExposure] = useState(false);
  const [snapBusy, setSnapBusy] = useState(false);
  const [nonce, setNonce] = useState(0);

  // Only tear the stream down on unmount if this page started it — a stream
  // that a laptop user started over WiFi must survive the kiosk navigating away.
  const startedByUsRef = useRef(false);
  const streamingRef = useRef(false);
  const detectorRef = useRef(null);
  useEffect(() => {
    streamingRef.current = streaming;
    detectorRef.current = selectedDetector;
  }, [streaming, selectedDetector]);

  useEffect(() => {
    let cancelled = false;
    apiSettingsControllerGetDetectorNames()
      .then((names) => {
        if (cancelled || !Array.isArray(names) || !names.length) return;
        setDetectors(names);
        setSelectedDetector((current) => current ?? names[0]);
        return apiLiveViewControllerGetActiveStreams().then((active) => {
          if (cancelled || !active || typeof active !== "object") return;
          // Adopt an already-running stream for the default detector.
          if (active[names[0]]) setStreaming(true);
        });
      })
      .catch(() => {
        if (!cancelled) enqueueSnackbar("Could not list cameras", { variant: "error" });
      });
    return () => {
      cancelled = true;
      if (startedByUsRef.current && streamingRef.current) {
        apiLiveViewControllerStopLiveView(detectorRef.current).catch(() => {});
      }
    };
  }, []);

  const startStream = (force = false) => {
    setStarting(true);
    setLongExposure(false);
    apiLiveViewControllerStartLiveView(selectedDetector, "mjpeg", null, force)
      .then((result) => {
        if (result?.status === "long_exposure") {
          setLongExposure(true);
          return;
        }
        startedByUsRef.current = true;
        setNonce((n) => n + 1);
        setStreaming(true);
      })
      .catch(() => enqueueSnackbar("Could not start the stream", { variant: "error" }))
      .finally(() => setStarting(false));
  };

  const stopStream = () => {
    apiLiveViewControllerStopLiveView(selectedDetector).catch(() => {});
    startedByUsRef.current = false;
    setStreaming(false);
  };

  const switchDetector = (name) => {
    if (name === selectedDetector) return;
    if (streaming && startedByUsRef.current) {
      apiLiveViewControllerStopLiveView(selectedDetector).catch(() => {});
    }
    setStreaming(false);
    startedByUsRef.current = false;
    setSelectedDetector(name);
  };

  const handleSnap = async () => {
    setSnapBusy(true);
    try {
      const job = await apiRecordingControllerStartSnap({});
      let tries = 0;
      let status = job?.status;
      while (["pending", "running"].includes(status) && tries < SNAP_POLL_MAX) {
        // eslint-disable-next-line no-await-in-loop
        await new Promise((resolve) => setTimeout(resolve, SNAP_POLL_MS));
        // eslint-disable-next-line no-await-in-loop
        const info = await apiRecordingControllerGetSnapStatus(job?.jobId);
        status = info?.status;
        tries += 1;
      }
      if (status === "done") {
        enqueueSnackbar("Image saved", { variant: "success" });
      } else {
        enqueueSnackbar(`Snap ${status || "did not finish"}`, { variant: "warning" });
      }
    } catch (error) {
      enqueueSnackbar("Snap failed", { variant: "error" });
    } finally {
      setSnapBusy(false);
    }
  };

  const streamUrl = selectedDetector
    ? `${connectionSettings.ip}:${connectionSettings.apiPort}/imswitch/api/LiveViewController/mjpeg_stream?startStream=true&detectorName=${encodeURIComponent(selectedDetector)}&t=${nonce}`
    : null;

  return (
    <MobilePage
      title="Camera"
      subtitle={selectedDetector || "No camera found"}
      action={
        <Box sx={{ display: "flex", gap: 1.5 }}>
          <Button
            variant="outlined"
            startIcon={snapBusy ? <CircularProgress size={18} /> : <CameraAltRoundedIcon />}
            disabled={snapBusy || !selectedDetector}
            onClick={handleSnap}
          >
            Snap
          </Button>
          {streaming ? (
            <Button
              color="error"
              variant="contained"
              startIcon={<StopRoundedIcon />}
              onClick={stopStream}
            >
              Stop
            </Button>
          ) : (
            <Button
              variant="contained"
              startIcon={starting ? <CircularProgress size={18} /> : <PlayArrowRoundedIcon />}
              disabled={starting || !selectedDetector}
              onClick={() => startStream(false)}
            >
              Start
            </Button>
          )}
        </Box>
      }
      disablePadding
    >
      <Box sx={{ height: "100%", display: "flex", flexDirection: "column", p: 2.5, gap: 2 }}>
        {detectors.length > 1 && (
          <ToggleButtonGroup
            exclusive
            value={selectedDetector}
            onChange={(e, v) => v !== null && switchDetector(v)}
          >
            {detectors.map((d) => (
              <ToggleButton key={d} value={d} sx={{ px: 2.5 }}>
                {d}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        )}

        {longExposure && (
          <Alert
            severity="warning"
            action={
              <Button color="inherit" size="small" onClick={() => startStream(true)}>
                Start anyway
              </Button>
            }
          >
            The camera exposure is very long — the live view would update slowly.
          </Alert>
        )}

        <Paper
          variant="outlined"
          sx={{
            flex: 1,
            minHeight: 0,
            borderRadius: 3,
            bgcolor: "#000",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            overflow: "hidden",
          }}
        >
          {streaming && streamUrl ? (
            <img
              src={streamUrl}
              alt="Live camera stream"
              style={{
                maxWidth: "100%",
                maxHeight: "100%",
                objectFit: "contain",
                display: "block",
              }}
            />
          ) : (
            <Box sx={{ textAlign: "center", color: "text.disabled" }}>
              <VideocamOffRoundedIcon sx={{ fontSize: 56, mb: 1 }} />
              <Typography>Stream stopped</Typography>
            </Box>
          )}
        </Paper>
      </Box>
    </MobilePage>
  );
};

export default MobileCameraPage;
