import React, { useState, useEffect, useRef, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  TextField,
  MenuItem,
  Box,
  Typography,
  Tooltip,
  IconButton,
  Button,
  FormControl,
  InputLabel,
  Select,
} from "@mui/material";
import { Camera, InfoOutlined, Tune } from "@mui/icons-material";
import * as detectorParametersSlice from "../state/slices/DetectorParametersSlice.js";
import { normalizeGainValue } from "../constants/cameraGainValues.js";
import CameraSettingsDialog from "./CameraSettingsDialog.jsx";

const AUTO_ONCE_RESET_DELAY_MS = 1500;
const AUTO_ONCE_UI_HOLD_MS = AUTO_ONCE_RESET_DELAY_MS + 300;

/**
 * Detector Parameters Component - Now with WebSocket support
 *
 * This component displays and updates detector parameters.
 * It now receives parameter updates via WebSocket when the backend changes them
 * (e.g., in Auto Mode), without requiring polling.
 *
 * Usage:
 *   <DetectorParameters hostIP={hostIP} hostPort={hostPort} />
 */
export default function DetectorParameters({ hostIP, hostPort }) {
  const dispatch = useDispatch();

  // Get detector parameters from Redux (updated via WebSocket)
  const detectorParams = useSelector(
    detectorParametersSlice.getDetectorParameters,
  );

  // Local string values for text fields to avoid race conditions.
  // These are what the user sees while typing – they are NOT sent to
  // the backend until the user commits (blur / Enter).
  const [localExposure, setLocalExposure] = useState("");
  const [localGain, setLocalGain] = useState("");
  const [autoOncePending, setAutoOncePending] = useState(false);
  const [cameraDialogOpen, setCameraDialogOpen] = useState(false);

  // Track whether a field is currently being edited so we don't
  // overwrite the user's in-progress typing with a WebSocket update.
  const editingRef = useRef({
    exposure: false,
    gain: false,
  });

  // Gain limits come from the camera when the manager knows them; otherwise the
  // generic clamp in normalizeGainValue applies.
  const gainMin = detectorParams.gainMin;
  const gainMax = detectorParams.gainMax;
  const clampGain = useCallback(
    (value) => {
      const num = Number(value);
      if (!Number.isFinite(num)) return null;
      const hasMin = gainMin !== null && gainMin !== undefined;
      const hasMax = gainMax !== null && gainMax !== undefined;
      if (!hasMin && !hasMax) return normalizeGainValue(num);
      let result = num;
      if (hasMin) result = Math.max(Number(gainMin), result);
      if (hasMax) result = Math.min(Number(gainMax), result);
      return result;
    },
    [gainMin, gainMax],
  );

  const supportedBinnings = Array.isArray(detectorParams.supportedBinnings)
    ? detectorParams.supportedBinnings
    : [];

  // Fetch existing detector parameters on mount and when connection changes
  useEffect(() => {
    let cancelled = false;
    async function fetchParams() {
      try {
        const resp = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/SettingsController/getDetectorParameters`,
        );
        if (!resp.ok || cancelled) return;
        const data = await resp.json();
        // Keep every key the backend reports (supportedBinnings, gain limits,
        // ...) so new camera capabilities show up without touching this list.
        const newParams = {
          ...data,
          exposure: data.exposure ?? "",
          gain: data.gain ?? "",
          pixelSize: data.pixelSize ?? "",
          binning: data.binning ?? "",
          blacklevel: data.blacklevel ?? "",
          isRGB: data.isRGB === 1,
          mode: (data.mode ?? "manual").toLowerCase(),
        };
        dispatch(detectorParametersSlice.setParameters(newParams));
        // Sync local text fields only if user is not currently editing
        if (!editingRef.current.exposure)
          setLocalExposure(String(newParams.exposure));
        if (!editingRef.current.gain) setLocalGain(String(newParams.gain));
      } catch (error) {
        console.error("Error fetching detector parameters:", error);
      }
    }
    fetchParams();
    return () => {
      cancelled = true;
    };
  }, [hostIP, hostPort, dispatch]);

  // Sync local text fields when Redux state changes (from WebSocket)
  // but only if user is not currently editing those fields
  useEffect(() => {
    if (
      !editingRef.current.exposure &&
      detectorParams.exposure !== undefined &&
      detectorParams.exposure !== null
    ) {
      const exposureValue =
        typeof detectorParams.exposure === "string" &&
        detectorParams.exposure.trim() === ""
          ? Number.NaN
          : Number(detectorParams.exposure);
      setLocalExposure(
        Number.isFinite(exposureValue) ? exposureValue.toFixed(3) : "",
      );
    }
    if (
      !editingRef.current.gain &&
      detectorParams.gain !== undefined &&
      detectorParams.gain !== null
    ) {
      const gainValue = Number(detectorParams.gain);
      setLocalGain(Number.isFinite(gainValue) ? String(gainValue) : "");
    }
  }, [detectorParams.exposure, detectorParams.gain]);

  // Update numeric field immediately on change
  const handleImmediateFieldChange = useCallback(
    async (field, rawValue) => {
      let value = Number(rawValue);
      if (rawValue === "" || isNaN(value)) return;

      if (field === "gain") {
        const clampedGain = clampGain(value);
        if (clampedGain === null) return;
        value = clampedGain;
      }

      dispatch(detectorParametersSlice.updateParameter({ key: field, value }));
      try {
        switch (field) {
          case "exposure":
            await fetch(
              `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorExposureTime?exposureTime=${value}`,
            );
            break;
          case "gain":
            await fetch(
              `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorGain?gain=${value}`,
            );
            break;
          case "blacklevel":
            await fetch(
              `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorBlackLevel?blackLevel=${value}`,
            );
            break;
          default:
            break;
        }
      } catch (error) {
        console.error(`Error updating '${field}' to '${value}':`, error);
      }
    },
    [hostIP, hostPort, dispatch, clampGain],
  );

  // Handle non-numeric fields (binning, isRGB, mode) immediately on change
  const handleParamChange = async (field, value) => {
    dispatch(detectorParametersSlice.updateParameter({ key: field, value }));
    try {
      switch (field) {
        case "binning":
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorBinning?binning=${value}`,
          );
          break;
        case "isRGB": {
          const intVal = value ? 1 : 0;
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorIsRGB?isRGB=${intVal}`,
          );
          break;
        }
        case "mode": {
          const isAuto = value === "auto";
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorMode?isAuto=${isAuto}`,
          );
          break;
        }
        default:
          break;
      }
    } catch (error) {
      console.error(`Error updating '${field}' to '${value}':`, error);
    }
  };

  const refreshDetectorParameters = useCallback(async () => {
    try {
      const resp = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/SettingsController/getDetectorParameters`,
      );
      if (!resp.ok) return;
      const data = await resp.json();
      dispatch(
        detectorParametersSlice.setParameters({
          ...data,
          exposure: data.exposure ?? "",
          gain: data.gain ?? "",
          pixelSize: data.pixelSize ?? "",
          binning: data.binning ?? "",
          blacklevel: data.blacklevel ?? "",
          isRGB: data.isRGB === 1,
          mode: (data.mode ?? "manual").toLowerCase(),
        }),
      );
    } catch (error) {
      console.error("Error refreshing detector parameters:", error);
    }
  }, [hostIP, hostPort, dispatch]);

  const handleAutoExposureOnce = useCallback(async () => {
    setAutoOncePending(true);
    try {
      const resp = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorExposureOnce?resetDelayMs=${AUTO_ONCE_RESET_DELAY_MS}`,
      );
      if (!resp.ok) {
        throw new Error(`Auto once request failed with status ${resp.status}`);
      }
      // Keep pending true until backend's once->manual reset window should be complete.
      await new Promise((resolve) => setTimeout(resolve, AUTO_ONCE_UI_HOLD_MS));
      await refreshDetectorParameters();
    } catch (error) {
      console.error("Error running one-shot exposure auto:", error);
    } finally {
      setAutoOncePending(false);
    }
  }, [hostIP, hostPort, refreshDetectorParameters]);

  const beginEditing = (field) => {
    editingRef.current[field] = true;
  };

  const endEditing = (field) => {
    editingRef.current[field] = false;
  };

  const isValidNumericInput = (field, value) => {
    if (value === "") return true;
    // Decimals are allowed while typing (e.g. "10.").
    return /^\d*\.?\d*$/.test(value);
  };

  const handleNumericFieldChange = (field, setValue) => (e) => {
    beginEditing(field);
    const raw = e.target.value;
    if (!isValidNumericInput(field, raw)) {
      return;
    }
    setValue(raw);
    handleImmediateFieldChange(field, raw);
  };

  const handleNumericFieldKeyDown = (e) => {
    if (e.key === "Enter") {
      e.currentTarget.blur();
    }
  };

  // Binning changes the frame size, so re-read the parameters afterwards.
  const handleBinningChange = async (event) => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) return;
    await handleParamChange("binning", value);
    await refreshDetectorParameters();
  };

  const gainHelperText =
    gainMin !== null &&
    gainMin !== undefined &&
    gainMax !== null &&
    gainMax !== undefined
      ? `${gainMin} – ${gainMax}`
      : " ";

  return (
    <Box
      component="fieldset"
      sx={{
        border: 1,
        borderColor: "divider",
        borderRadius: 1,
        p: 2,
        display: "flex",
        flexDirection: "column",
        gap: 1,
      }}
    >
      <Box
        component="legend"
        sx={{
          display: "inline-flex",
          alignItems: "center",
          gap: 0.5,
          px: 1,
        }}
      >
        <Camera fontSize="small" sx={{ color: "text.secondary" }} />
        <Typography variant="subtitle1" sx={{ fontWeight: "medium" }}>
          Detector Parameters
        </Typography>
        <Tooltip
          arrow
          title={
            <Box sx={{ whiteSpace: "pre-line" }}>
              {
                "Exposure mode controls exposure only on this camera.\n\nManual: you set exposure directly.\nAuto: the camera continuously adapts exposure time.\nAuto once: the camera makes a single exposure adjustment and then returns to manual."
              }
            </Box>
          }
        >
          <IconButton size="small" sx={{ p: 0.25, color: "text.disabled" }}>
            <InfoOutlined fontSize="inherit" />
          </IconButton>
        </Tooltip>
        <Tooltip
          arrow
          title="All camera settings (black level, trigger, cooling, temperature, ...)"
        >
          <Button
            size="small"
            variant="outlined"
            startIcon={<Tune fontSize="small" />}
            onClick={() => setCameraDialogOpen(true)}
            sx={{ ml: 1, py: 0.1 }}
          >
            Camera settings
          </Button>
        </Tooltip>
      </Box>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: {
            xs: "1fr",
            md: "130px 190px minmax(120px, 1fr)",
          },
          gap: 1,
          alignItems: "end",
        }}
      >
        <Box>
          <TextField
            label="Exposure"
            type="text"
            inputProps={{ inputMode: "decimal" }}
            value={localExposure}
            onFocus={() => beginEditing("exposure")}
            onChange={handleNumericFieldChange("exposure", setLocalExposure)}
            onBlur={() => endEditing("exposure")}
            onKeyDown={handleNumericFieldKeyDown}
            size="small"
            disabled={detectorParams.mode === "auto"}
            sx={{
              width: 130,
              "& .MuiInputBase-root": {
                height: 40,
              },
            }}
            InputProps={{
              endAdornment: (
                <Box sx={{ display: "flex", flexDirection: "column", ml: 0.5 }}>
                  <IconButton
                    size="small"
                    sx={{ p: 0, height: 18 }}
                    aria-label="Increment exposure"
                    onClick={() => {
                      if (detectorParams.mode === "auto") return;
                      const next = Number(localExposure || 0) + 1;
                      setLocalExposure(String(next));
                      handleImmediateFieldChange("exposure", next);
                    }}
                    disabled={detectorParams.mode === "auto"}
                  >
                    <span style={{ fontSize: 14, lineHeight: 1 }}>▲</span>
                  </IconButton>
                  <IconButton
                    size="small"
                    sx={{ p: 0, height: 18 }}
                    aria-label="Decrement exposure"
                    onClick={() => {
                      if (detectorParams.mode === "auto") return;
                      const next = Number(localExposure || 0) - 1;
                      setLocalExposure(String(next));
                      handleImmediateFieldChange("exposure", next);
                    }}
                    disabled={detectorParams.mode === "auto"}
                  >
                    <span style={{ fontSize: 14, lineHeight: 1 }}>▼</span>
                  </IconButton>
                </Box>
              ),
            }}
          />
        </Box>

        <Box>
          <Tooltip
            arrow
            placement="top-start"
            title="Manual: set exposure directly. Auto: camera continuously adjusts exposure. Gain is not auto-adjusted here."
          >
            <FormControl size="small" sx={{ width: 180, height: 40 }}>
              <InputLabel id="detector-mode-label">Mode</InputLabel>
              <Select
                labelId="detector-mode-label"
                id="detector-mode-select"
                value={detectorParams.mode}
                label="Mode"
                onChange={(e) => handleParamChange("mode", e.target.value)}
              >
                <MenuItem value="manual">Manual</MenuItem>
                <MenuItem value="auto">Auto</MenuItem>
              </Select>
            </FormControl>
          </Tooltip>
        </Box>

        <Box>
          <Tooltip
            title="Run a single auto-exposure pass and then return to manual mode."
            arrow
          >
            <Button
              size="small"
              variant="contained"
              onClick={handleAutoExposureOnce}
              disabled={detectorParams.mode !== "manual" || autoOncePending}
              sx={{
                whiteSpace: "nowrap",
                height: 40,
                minHeight: 40,
                width: 130,
              }}
            >
              Auto once
            </Button>
          </Tooltip>
        </Box>
      </Box>

      <Box sx={{ pt: 0.5 }}>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontWeight: 500 }}
        >
          Gain and binning
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "160px 160px" },
            gap: 1,
            alignItems: "start",
            mt: 0.25,
          }}
        >
          <TextField
            label="Gain"
            type="text"
            inputProps={{ inputMode: "decimal" }}
            value={localGain}
            onFocus={() => beginEditing("gain")}
            onChange={handleNumericFieldChange("gain", setLocalGain)}
            onBlur={() => endEditing("gain")}
            onKeyDown={handleNumericFieldKeyDown}
            size="small"
            helperText={gainHelperText}
            FormHelperTextProps={{ sx: { minHeight: "1.2em", m: 0, mt: 0.25 } }}
            sx={{
              "& .MuiInputBase-root": {
                height: 40,
              },
            }}
          />

          <Tooltip
            arrow
            title="Combines neighbouring pixels: smaller, brighter and less noisy images at lower resolution."
          >
            <FormControl
              size="small"
              sx={{ minWidth: 160 }}
              disabled={supportedBinnings.length < 2}
            >
              <InputLabel id="detector-binning-label">Binning</InputLabel>
              <Select
                labelId="detector-binning-label"
                id="detector-binning-select"
                label="Binning"
                value={
                  supportedBinnings.length
                    ? Number(detectorParams.binning) || supportedBinnings[0]
                    : Number(detectorParams.binning) || 1
                }
                onChange={handleBinningChange}
                sx={{ height: 40 }}
              >
                {(supportedBinnings.length
                  ? supportedBinnings
                  : [Number(detectorParams.binning) || 1]
                ).map((val) => (
                  <MenuItem key={val} value={val}>
                    {`${val} × ${val}`}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Tooltip>
        </Box>
      </Box>

      <CameraSettingsDialog
        open={cameraDialogOpen}
        onClose={() => {
          setCameraDialogOpen(false);
          refreshDetectorParameters();
        }}
      />
    </Box>
  );
}
