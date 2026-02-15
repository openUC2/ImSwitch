import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  TextField,
  MenuItem,
  FormControlLabel,
  Switch,
  Box,
  Typography,
} from "@mui/material";
import { Camera } from "@mui/icons-material";

/**
 * This component fetches and updates the detector parameters from:
 *    /SettingsController/getDetectorParameters
 *
 * It calls the known endpoints to update each parameter individually:
 *    setDetectorExposureTime?exposureTime=...
 *    setDetectorGain?gain=...
 *    setDetectorBinning?binning=...
 *    setDetectorBlackLevel?blacklevel=...
 *    setDetectorIsRGB?isRGB=...
 *    setDetectorMode?isAuto=...
 * (Adapt or expand if your API uses a different pattern.)
 *
 * Usage:
 *   <DetectorParameters hostIP={hostIP} hostPort={hostPort} />
 */
export default function DetectorParameters({ hostIP, hostPort }) {
  // Canonical values as last confirmed by the backend
  const [detectorParams, setDetectorParams] = useState({
    exposure: "",
    gain: "",
    pixelSize: "",
    binning: "",
    blacklevel: "",
    isRGB: false,
    mode: "manual",
  });

  // Local string values for text fields to avoid race conditions.
  // These are what the user sees while typing – they are NOT sent to
  // the backend until the user commits (blur / Enter).
  const [localExposure, setLocalExposure] = useState("");
  const [localGain, setLocalGain] = useState("");
  const [localBlacklevel, setLocalBlacklevel] = useState("");

  // Track whether a field is currently being edited so we don't
  // overwrite the user's in-progress typing with a fetch result.
  const editingRef = useRef({ exposure: false, gain: false, blacklevel: false });

  // Fetch existing detector parameters on mount and when connection changes
  useEffect(() => {
    let cancelled = false;
    async function fetchParams() {
      try {
        const resp = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/SettingsController/getDetectorParameters`
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
        setDetectorParams(newParams);
        // Sync local text fields only if user is not currently editing
        if (!editingRef.current.exposure) setLocalExposure(String(newParams.exposure));
        if (!editingRef.current.gain) setLocalGain(String(newParams.gain));
        if (!editingRef.current.blacklevel) setLocalBlacklevel(String(newParams.blacklevel));
      } catch (error) {
        console.error("Error fetching detector parameters:", error);
      }
    }
    fetchParams();
    return () => { cancelled = true; };
  }, [hostIP, hostPort]);

  // Commit a numeric field to the backend.
  // Called on blur or Enter – NOT on every keystroke.
  const commitField = useCallback(async (field, rawValue) => {
    editingRef.current[field] = false;
    const value = Number(rawValue);
    if (rawValue === "" || isNaN(value)) return; // ignore empty / non-numeric
    // Update canonical state
    setDetectorParams((prev) => ({ ...prev, [field]: value }));
    try {
      switch (field) {
        case "exposure":
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorExposureTime?exposureTime=${value}`
          );
          break;
        case "gain":
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorGain?gain=${value}`
          );
          break;
        case "blacklevel":
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorBlackLevel?blacklevel=${value}`
          );
          break;
        default:
          break;
      }
    } catch (error) {
      console.error(`Error updating '${field}' to '${value}':`, error);
    }
  }, [hostIP, hostPort]);

  // Handle non-numeric fields (binning, isRGB, mode) immediately on change
  const handleParamChange = async (field, value) => {
    setDetectorParams((prev) => ({ ...prev, [field]: value }));
    try {
      switch (field) {
        case "binning":
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorBinning?binning=${value}`
          );
          break;
        case "isRGB": {
          const intVal = value ? 1 : 0;
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorIsRGB?isRGB=${intVal}`
          );
          break;
        }
        case "mode": {
          const isAuto = value === "auto";
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorMode?isAuto=${isAuto}`
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
          display: "flex",
          alignItems: "center",
          gap: 0.5,
          px: 1,
        }}
      >
        <Camera fontSize="small" sx={{ color: "text.secondary" }} />
        <Typography variant="subtitle1" sx={{ fontWeight: "medium" }}>
          Detector Parameters
        </Typography>
      </Box>

      <TextField
        label="Exposure"
        type="text"
        inputProps={{ inputMode: "decimal" }}
        value={localExposure}
        onChange={(e) => {
          editingRef.current.exposure = true;
          setLocalExposure(e.target.value);
        }}
        onBlur={() => commitField("exposure", localExposure)}
        onKeyDown={handleKeyDown("exposure", localExposure)}
        size="small"
        margin="dense"
      />
      <TextField
        label="Gain"
        type="text"
        inputProps={{ inputMode: "decimal" }}
        value={localGain}
        onChange={(e) => {
          editingRef.current.gain = true;
          setLocalGain(e.target.value);
        }}
        onBlur={() => commitField("gain", localGain)}
        onKeyDown={handleKeyDown("gain", localGain)}
        size="small"
        margin="dense"
      />

      <TextField
        label="Black Level"
        type="text"
        inputProps={{ inputMode: "decimal" }}
        value={localBlacklevel}
        onChange={(e) => {
          editingRef.current.blacklevel = true;
          setLocalBlacklevel(e.target.value);
        }}
        onBlur={() => commitField("blacklevel", localBlacklevel)}
        onKeyDown={handleKeyDown("blacklevel", localBlacklevel)}
        size="small"
        margin="dense"
      />
      <TextField
        select
        label="Mode"
        value={detectorParams.mode}
        onChange={(e) => handleParamChange("mode", e.target.value)}
        size="small"
        margin="dense"
        sx={{ width: 120 }}
      >
        <MenuItem value="manual">Manual</MenuItem>
        <MenuItem value="auto">Auto</MenuItem>
      </TextField>
    </Box>
  );
}
