import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";

import {
  Box,
  Button,
  Typography,
  Slider,
  TextField,
  LinearProgress,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Paper,
  FormControlLabel,
  Switch,
  Stack,
  Divider,
  Alert,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import SaveIcon from "@mui/icons-material/Save";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";

import * as overviewRegSlice from "../state/slices/OverviewRegistrationSlice.js";

import apiGetOverviewOverlayData from "../backendapi/apiGetOverviewOverlayData.js";
import apiGetOverviewRegistrationConfigData from "../backendapi/apiGetOverviewRegistrationConfigData.js";
import apiUpdateOverviewRegistrationConfig from "../backendapi/apiUpdateOverviewRegistrationConfig.js";
import apiRunAutonomousOverviewScan from "../backendapi/apiRunAutonomousOverviewScan.js";

/**
 * Tab for the autonomous overview scan + overlay controls.
 *
 * Replaces the previous Accordion. Provides:
 *  - Overlay enable / opacity controls
 *  - Editable XYZ table for the registered slots
 *  - Save / reload buttons for the config
 *  - "Run Autonomous Scan" button with progress
 *  - Launch button for the manual registration wizard
 */
const OverviewScanTab = () => {
  const dispatch = useDispatch();
  const overviewRegState = useSelector(
    overviewRegSlice.getOverviewRegistrationState
  );

  const [localError, setLocalError] = useState(null);
  const [localInfo, setLocalInfo] = useState(null);

  // ── helpers ──────────────────────────────────────────────────────
  const layoutName = overviewRegState.layoutName || "Heidstar 4x Histosample";
  const cameraName = overviewRegState.cameraName || "";

  const loadConfig = async () => {
    setLocalError(null);
    try {
      const data = await apiGetOverviewRegistrationConfigData(
        cameraName,
        layoutName
      );
      if (data && data.exists) {
        dispatch(overviewRegSlice.setRegistrationConfig(data.config));
      } else {
        dispatch(overviewRegSlice.setRegistrationConfig(null));
      }
    } catch (e) {
      setLocalError("Failed to load config: " + e.message);
    }
  };

  const loadOverlay = async () => {
    try {
      const data = await apiGetOverviewOverlayData(cameraName, layoutName);
      dispatch(overviewRegSlice.setOverlayData(data));
    } catch (e) {
      console.warn("Failed to load overlay data:", e);
    }
  };

  // Initial load
  useEffect(() => {
    loadConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraName, layoutName]);

  // ── handlers ─────────────────────────────────────────────────────
  const handleOverlayToggle = (e) => {
    const checked = e.target.checked;
    dispatch(overviewRegSlice.setOverlayEnabled(checked));
    if (
      checked &&
      (!overviewRegState.overlayData ||
        !overviewRegState.overlayData.slides ||
        Object.keys(overviewRegState.overlayData.slides || {}).length === 0)
    ) {
      loadOverlay();
    }
  };

  const handleOpacityChange = (_e, v) => {
    dispatch(overviewRegSlice.setOverlayOpacity(v));
  };

  const handleOpenWizard = () => {
    dispatch(overviewRegSlice.setWizardOpen(true));
  };

  const handlePositionEdit = (slotId, axis, value) => {
    const num = parseFloat(value);
    dispatch(
      overviewRegSlice.updateRegistrationConfigSlotPosition({
        slotId,
        axis,
        value: isNaN(num) ? 0 : num,
      })
    );
  };

  const handleSaveConfig = async () => {
    if (!overviewRegState.registrationConfig) return;
    setLocalError(null);
    setLocalInfo(null);
    try {
      await apiUpdateOverviewRegistrationConfig(
        overviewRegState.registrationConfig
      );
      setLocalInfo("Configuration saved.");
    } catch (e) {
      setLocalError("Save failed: " + e.message);
    }
  };

  const handleRunScan = async () => {
    setLocalError(null);
    setLocalInfo(null);

    const cfg = overviewRegState.registrationConfig;
    const slotIds = cfg && cfg.slots ? Object.keys(cfg.slots) : [];
    if (slotIds.length === 0) {
      setLocalError(
        "No registered slots found. Run the registration wizard first."
      );
      return;
    }

    dispatch(overviewRegSlice.setAutonomousScanRunning(true));
    dispatch(
      overviewRegSlice.setAutonomousScanProgress({
        current: 0,
        total: slotIds.length,
        slotId: "",
      })
    );

    try {
      const result = await apiRunAutonomousOverviewScan(
        cameraName,
        layoutName,
        0.5
      );
      if (result && result.overlayData) {
        dispatch(overviewRegSlice.setOverlayData(result.overlayData));
        dispatch(overviewRegSlice.setOverlayEnabled(true));
      }
      dispatch(
        overviewRegSlice.setAutonomousScanProgress({
          current: slotIds.length,
          total: slotIds.length,
          slotId: "",
        })
      );
      const failures = Object.entries(result?.scanResults || {})
        .filter(([, r]) => !r.success)
        .map(([sid, r]) => `${sid}: ${r.error}`);
      if (failures.length > 0) {
        setLocalError("Some slots failed:\n" + failures.join("\n"));
      } else {
        setLocalInfo("Autonomous scan finished.");
      }
    } catch (e) {
      setLocalError("Scan failed: " + e.message);
    } finally {
      dispatch(overviewRegSlice.setAutonomousScanRunning(false));
    }
  };

  // ── render ───────────────────────────────────────────────────────
  const config = overviewRegState.registrationConfig;
  const slotEntries = config && config.slots ? Object.entries(config.slots) : [];

  return (
    <Box sx={{ p: 2 }}>
      <Stack spacing={2}>
        {/* Overlay controls */}
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            Overview Overlay
          </Typography>
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
            <FormControlLabel
              control={
                <Switch
                  checked={overviewRegState.overlayEnabled}
                  onChange={handleOverlayToggle}
                  size="small"
                />
              }
              label="Show overlay"
            />
            <Button
              size="small"
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={loadOverlay}
            >
              Reload overlay
            </Button>
            <Button
              size="small"
              variant="outlined"
              startIcon={<OpenInNewIcon />}
              onClick={handleOpenWizard}
            >
              Manual registration wizard
            </Button>
          </Stack>
          {overviewRegState.overlayEnabled && (
            <Box sx={{ mt: 2, display: "flex", alignItems: "center", gap: 2 }}>
              <Typography variant="caption" sx={{ minWidth: 60 }}>
                Opacity:
              </Typography>
              <Slider
                value={overviewRegState.overlayOpacity}
                onChange={handleOpacityChange}
                min={0}
                max={1}
                step={0.05}
                size="small"
                sx={{ maxWidth: 240 }}
              />
              <Typography variant="caption">
                {Math.round(overviewRegState.overlayOpacity * 100)}%
              </Typography>
            </Box>
          )}
        </Paper>

        {/* Autonomous scan */}
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            Autonomous Overview Scan
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }} color="text.secondary">
            Visits each registered slot using its stored XYZ position, snaps a
            new image and refreshes the overlay automatically.
          </Typography>
          <Stack direction="row" spacing={2} alignItems="center">
            <Button
              variant="contained"
              startIcon={<PlayArrowIcon />}
              onClick={handleRunScan}
              disabled={
                overviewRegState.autonomousScanRunning ||
                slotEntries.length === 0
              }
            >
              Run autonomous scan
            </Button>
            <Button
              size="small"
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={loadConfig}
            >
              Reload config
            </Button>
          </Stack>
          {overviewRegState.autonomousScanRunning && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress
                variant={
                  overviewRegState.autonomousScanProgress.total > 0
                    ? "determinate"
                    : "indeterminate"
                }
                value={
                  overviewRegState.autonomousScanProgress.total > 0
                    ? (overviewRegState.autonomousScanProgress.current /
                        overviewRegState.autonomousScanProgress.total) *
                      100
                    : undefined
                }
              />
              <Typography variant="caption">
                Scanning… {overviewRegState.autonomousScanProgress.current} /{" "}
                {overviewRegState.autonomousScanProgress.total}
              </Typography>
            </Box>
          )}
        </Paper>

        {/* Slot XYZ table */}
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
            sx={{ mb: 1 }}
          >
            <Typography variant="subtitle1">
              Registered slots ({slotEntries.length})
            </Typography>
            <Button
              size="small"
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={handleSaveConfig}
              disabled={!config}
            >
              Save changes
            </Button>
          </Stack>
          <Divider sx={{ mb: 1 }} />
          {slotEntries.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No registrations yet. Use the manual wizard above to register
              slots first.
            </Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Slot</TableCell>
                  <TableCell>Name</TableCell>
                  <TableCell align="right">X (µm)</TableCell>
                  <TableCell align="right">Y (µm)</TableCell>
                  <TableCell align="right">Z (µm)</TableCell>
                  <TableCell align="right">Reproj. err.</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {slotEntries.map(([slotId, slot]) => {
                  const pos = slot.stagePosition || { x: 0, y: 0, z: 0 };
                  return (
                    <TableRow key={slotId}>
                      <TableCell>{slotId}</TableCell>
                      <TableCell>{slot.slotName || `Slide ${slotId}`}</TableCell>
                      {["x", "y", "z"].map((axis) => (
                        <TableCell key={axis} align="right">
                          <TextField
                            type="number"
                            size="small"
                            value={pos[axis] ?? 0}
                            onChange={(e) =>
                              handlePositionEdit(slotId, axis, e.target.value)
                            }
                            inputProps={{
                              step: 100,
                              style: { textAlign: "right", width: 90 },
                            }}
                            variant="standard"
                          />
                        </TableCell>
                      ))}
                      <TableCell align="right">
                        {(slot.reprojectionError || 0).toFixed(2)}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </Paper>

        {localError && (
          <Alert severity="error" onClose={() => setLocalError(null)}>
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{localError}</pre>
          </Alert>
        )}
        {localInfo && (
          <Alert severity="success" onClose={() => setLocalInfo(null)}>
            {localInfo}
          </Alert>
        )}
      </Stack>
    </Box>
  );
};

export default OverviewScanTab;
