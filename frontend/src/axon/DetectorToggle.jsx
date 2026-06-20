import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  Box,
} from "@mui/material";
import { Videocam as VideocamIcon } from "@mui/icons-material";

import * as liveViewSlice from "../state/slices/LiveViewSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import apiLiveViewControllerStartLiveView from "../backendapi/apiLiveViewControllerStartLiveView";
import apiLiveViewControllerStopLiveView from "../backendapi/apiLiveViewControllerStopLiveView";
import apiLiveViewControllerGetActiveStreams from "../backendapi/apiLiveViewControllerGetActiveStreams";

/**
 * DetectorToggle - Camera selector for switching between detectors
 * (e.g. overview vs widefield) within the WellPlate app.
 *
 * Extracted from AxonTabComponent so it can be shared between the legacy
 * two-tab-bar layout and the new WellPlateWorkspace viewport strip.
 */
const DetectorToggle = () => {
  const dispatch = useDispatch();
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const detectors = liveViewState.detectors || [];
  const activeTab = liveViewState.activeTab || 0;

  // Reflect the backend's actually-streaming detector in the UI on mount,
  // so we don't show e.g. "Observationcamera" selected while the backend
  // is streaming "WidefieldCamera".
  useEffect(() => {
    if (!detectors || detectors.length === 0) return;
    let cancelled = false;
    (async () => {
      try {
        const resp = await apiLiveViewControllerGetActiveStreams();
        if (cancelled) return;
        const streams = resp?.active_streams || [];
        if (streams.length === 0) return;
        const activeDetectorName = streams[0]?.detector;
        const idx = detectors.indexOf(activeDetectorName);
        if (idx >= 0 && idx !== activeTab) {
          dispatch(liveViewSlice.setActiveTab(idx));
        }
      } catch (_err) {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectors.length]);

  if (detectors.length < 2) return null; // No toggle needed for single camera

  const handleSwitch = async (_event, newIdx) => {
    if (newIdx === null || newIdx === activeTab) return;
    dispatch(liveViewSlice.setActiveTab(newIdx));

    // Restart stream on new detector (mirrors LiveView.js behaviour)
    if (liveViewState.isStreamRunning) {
      try {
        await apiLiveViewControllerStopLiveView();
        await new Promise((r) => setTimeout(r, 200));
        const protocol = liveStreamState.imageFormat || "jpeg";
        const newDetectorName = detectors[newIdx] || null;
        const savedParams =
          newDetectorName &&
          liveStreamState.perDetectorSettings?.[newDetectorName];
        const overrideParams =
          savedParams && savedParams.protocol === protocol ? savedParams : null;
        const result = await apiLiveViewControllerStartLiveView(
          newDetectorName,
          protocol,
          overrideParams,
        );
        if (result?.params && newDetectorName) {
          dispatch(
            liveStreamSlice.updateDetectorSettings({
              detectorName: newDetectorName,
              settings: result.params,
            }),
          );
        }
      } catch (err) {
        console.error("[DetectorToggle] Switch failed:", err);
      }
    }
  };

  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
      <VideocamIcon fontSize="small" color="action" />
      <Typography variant="caption" color="text.secondary">
        Camera:
      </Typography>
      <ToggleButtonGroup
        value={activeTab}
        exclusive
        onChange={handleSwitch}
        size="small"
        sx={{
          "& .MuiToggleButton-root": {
            textTransform: "none",
            py: 0.25,
            px: 1.5,
          },
        }}
      >
        {detectors.map((name, idx) => (
          <ToggleButton key={name} value={idx}>
            {name}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </Box>
  );
};

export default DetectorToggle;
