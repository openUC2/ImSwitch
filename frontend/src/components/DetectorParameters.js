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
import { Camera, InfoOutlined } from "@mui/icons-material";
import * as detectorParametersSlice from "../state/slices/DetectorParametersSlice.js";

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
  const [localBlacklevel, setLocalBlacklevel] = useState("");
  const [autoOncePending, setAutoOncePending] = useState(false);

  // Track whether a field is currently being edited so we don't
  // overwrite the user's in-progress typing with a WebSocket update.
  const editingRef = useRef({
    exposure: false,
    gain: false,
    blacklevel: false,
  });

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
        const newParams = {
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
        if (!editingRef.current.blacklevel)
          setLocalBlacklevel(String(newParams.blacklevel));
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
    if (!editingRef.current.exposure && detectorParams.exposure !== undefined && detectorParams.exposure !== null) {
      setLocalExposure(Number(detectorParams.exposure).toFixed(1));
    }
    if (!editingRef.current.gain && detectorParams.gain !== undefined && detectorParams.gain !== null) {
      setLocalGain(String(Math.round(Number(detectorParams.gain))));
    }
    if (!editingRef.current.blacklevel && detectorParams.blacklevel !== undefined && detectorParams.blacklevel !== null) {
      setLocalBlacklevel(String(Math.round(Number(detectorParams.blacklevel))));
    }
  }, [detectorParams.exposure, detectorParams.gain, detectorParams.blacklevel]);


  // Update numeric field immediately on change
  const handleImmediateFieldChange = useCallback(
    async (field, rawValue) => {
      const value = Number(rawValue);
      if (rawValue === "" || isNaN(value)) return;
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
              `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorBlackLevel?blacklevel=${value}`,
            );
            break;
          default:
            break;
        }
      } catch (error) {
        console.error(`Error updating '${field}' to '${value}':`, error);
      }
    },
    [hostIP, hostPort, dispatch],
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

  const handleAutoExposureOnce = useCallback(async () => {
    setAutoOncePending(true);
    try {
      await fetch(
        `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorExposureOnce`,
      );
    } catch (error) {
      console.error("Error running one-shot exposure auto:", error);
    } finally {
      setAutoOncePending(false);
    }
  }, [hostIP, hostPort]);

  // Helper: handle Enter key to commit
  const handleKeyDown = (field, localValue) => (e) => {
    if (e.key === "Enter") {
      e.target.blur(); // triggers onBlur → commitField
    }
  };

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
            onChange={(e) => {
              setLocalExposure(e.target.value);
              handleImmediateFieldChange("exposure", e.target.value);
            }}
            size="small"
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
                      const next = Number(localExposure || 0) + 1;
                      setLocalExposure(String(next));
                      handleImmediateFieldChange("exposure", next);
                    }}
                  >
                    <span style={{ fontSize: 14, lineHeight: 1 }}>▲</span>
                  </IconButton>
                  <IconButton
                    size="small"
                    sx={{ p: 0, height: 18 }}
                    aria-label="Decrement exposure"
                    onClick={() => {
                      const next = Number(localExposure || 0) - 1;
                      setLocalExposure(String(next));
                      handleImmediateFieldChange("exposure", next);
                    }}
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
          Advanced parameters
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "160px 160px" },
            gap: 1,
            alignItems: "end",
            mt: 0.25,
          }}
        >
          <TextField
            label="Gain"
            type="text"
            inputProps={{ inputMode: "decimal" }}
            value={localGain}
            onChange={(e) => {
              setLocalGain(e.target.value);
              handleImmediateFieldChange("gain", e.target.value);
            }}
            size="small"
            sx={{
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
                    aria-label="Increment gain"
                    onClick={() => {
                      const next = Number(localGain || 0) + 1;
                      setLocalGain(String(next));
                      handleImmediateFieldChange("gain", next);
                    }}
                  >
                    <span style={{ fontSize: 14, lineHeight: 1 }}>▲</span>
                  </IconButton>
                  <IconButton
                    size="small"
                    sx={{ p: 0, height: 18 }}
                    aria-label="Decrement gain"
                    onClick={() => {
                      const next = Number(localGain || 0) - 1;
                      setLocalGain(String(next));
                      handleImmediateFieldChange("gain", next);
                    }}
                  >
                    <span style={{ fontSize: 14, lineHeight: 1 }}>▼</span>
                  </IconButton>
                </Box>
              ),
            }}
          />

          <TextField
            label="Black Level"
            type="text"
            inputProps={{ inputMode: "decimal" }}
            value={localBlacklevel}
            onChange={(e) => {
              setLocalBlacklevel(e.target.value);
              handleImmediateFieldChange("blacklevel", e.target.value);
            }}
            size="small"
            sx={{
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
                    aria-label="Increment black level"
                    onClick={() => {
                      const next = Number(localBlacklevel || 0) + 1;
                      setLocalBlacklevel(String(next));
                      handleImmediateFieldChange("blacklevel", next);
                    }}
                  >
                    <span style={{ fontSize: 14, lineHeight: 1 }}>▲</span>
                  </IconButton>
                  <IconButton
                    size="small"
                    sx={{ p: 0, height: 18 }}
                    aria-label="Decrement black level"
                    onClick={() => {
                      const next = Number(localBlacklevel || 0) - 1;
                      setLocalBlacklevel(String(next));
                      handleImmediateFieldChange("blacklevel", next);
                    }}
                  >
                    <span style={{ fontSize: 14, lineHeight: 1 }}>▼</span>
                  </IconButton>
                </Box>
              ),
            }}
          />
        </Box>
      </Box>
    </Box>
  );
}
