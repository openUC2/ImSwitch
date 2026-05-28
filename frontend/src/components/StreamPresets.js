// src/components/StreamPresets.js
//
// Frontend-only "presets" / macros for the live stream + capture pipeline.
//
// A preset bundles:
//  - objective (slot id)
//  - exposure + gain
//  - livestream format (binary | jpeg | webrtc) and parameters
//    (subsampling factor, jpeg quality, etc.)
//
// Presets live in localStorage under STORAGE_KEY. They are independent of the
// backend config and survive page reloads. Users can save the current state
// as a new named preset, apply an existing one (which dispatches all the
// relevant Redux actions and fires the matching backend calls), or delete
// presets they no longer need.
//
import React, { useEffect, useMemo, useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  Box,
  Typography,
  Button,
  IconButton,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Stack,
  Tooltip,
  Alert,
} from "@mui/material";
import {
  Bookmark as BookmarkIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  PlayArrow as ApplyIcon,
  WbSunny as IlluminationIcon,
} from "@mui/icons-material";

import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

import apiLiveViewControllerSetStreamParameters from "../backendapi/apiLiveViewControllerSetStreamParameters";
import apiLiveViewControllerGetStreamParameters from "../backendapi/apiLiveViewControllerGetStreamParameters";
import apiObjectiveControllerMoveToObjective from "../backendapi/apiObjectiveControllerMoveToObjective";

const STORAGE_KEY_V1 = "imswitch.streamPresets.v1";
const STORAGE_KEY = "imswitch.streamPresets.v2";

/** Upgrade a single v1 preset to v2 shape (adds empty illumination block). */
const migrateV1ToV2 = (p) => ({
  ...p,
  illumination: p?.illumination ?? {
    lasers: [],
    leds: [],
    ledMatrix: null,
  },
  schemaVersion: 2,
});

/** Read presets from localStorage, falling back to an empty list. Migrates v1→v2 on read. */
const readPresets = () => {
  try {
    const rawV2 = localStorage.getItem(STORAGE_KEY);
    if (rawV2) {
      const arr = JSON.parse(rawV2);
      return Array.isArray(arr) ? arr.map(migrateV1ToV2) : [];
    }
    // Fall back to v1 if v2 doesn't exist yet — migrate in-memory but don't
    // write v2 until the user next saves, to avoid surprising side-effects.
    const rawV1 = localStorage.getItem(STORAGE_KEY_V1);
    if (!rawV1) return [];
    const arr = JSON.parse(rawV1);
    return Array.isArray(arr) ? arr.map(migrateV1ToV2) : [];
  } catch (_e) {
    return [];
  }
};

const writePresets = (presets) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets || []));
  } catch (_e) { /* quota or disabled — silently ignore */ }
};

const StreamPresets = () => {
  const dispatch = useDispatch();
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const { ip: hostIP, apiPort: hostPort } = useSelector(getConnectionSettingsState);

  const [presets, setPresets] = useState(readPresets);
  const [saveOpen, setSaveOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [selectedId, setSelectedId] = useState("");

  // Keep storage in sync when presets list mutates.
  useEffect(() => { writePresets(presets); }, [presets]);

  // Snapshot of "what the user would save right now."
  const currentSnapshot = useMemo(() => ({
    currentDetector: liveViewState.detectors?.[liveViewState.activeTab] ?? null,
    imageFormat: liveStreamState.imageFormat,
    streamSettings: liveStreamState.streamSettings,
    snapFormat: liveViewState.snapFormat,
    recordFormat: liveViewState.recordFormat,
    objective: {
      currentObjective: objectiveState.currentObjective,
      name: objectiveState.objectivName,
      pixelsize: objectiveState.pixelsize,
    },
  }), [liveStreamState, liveViewState, objectiveState]);

  /** Fetch current exposure/gain from the backend so we can store them too. */
  const fetchExposureGain = async () => {
    try {
      const r = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/SettingsController/getDetectorParameters`,
      );
      if (!r.ok) return { exposure: null, gain: null };
      const data = await r.json();
      return {
        exposure: data?.exposure ?? null,
        gain: data?.gain ?? null,
      };
    } catch (_e) {
      return { exposure: null, gain: null };
    }
  };

  /**
   * Fetch current illumination state (lasers + LEDs) so it can be re-applied
   * when the preset is loaded. Degrades gracefully: any subsystem that is
   * missing / fails just produces an empty list, the rest is still captured.
   */
  const fetchIlluminationState = async () => {
    const result = { lasers: [], leds: [], ledMatrix: null };

    // --- Lasers ---
    try {
      const namesRes = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserNames`,
      );
      if (namesRes.ok) {
        const names = await namesRes.json();
        if (Array.isArray(names)) {
          for (const name of names) {
            const enc = encodeURIComponent(name);
            let power = null;
            let enabled = null;
            try {
              const v = await fetch(
                `${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserValue?laserName=${enc}`,
              );
              if (v.ok) power = await v.json();
            } catch (_e) { /* per-laser non-fatal */ }
            try {
              const a = await fetch(
                `${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserActive?laserName=${enc}`,
              );
              if (a.ok) enabled = await a.json();
            } catch (_e) { /* per-laser non-fatal */ }
            result.lasers.push({ name, power, enabled });
          }
        }
      }
    } catch (_e) { /* no laser controller — leave lasers empty */ }

    // --- LEDs ---
    // LEDController only exposes setLEDValue/setLEDActive — there are no value
    // getters. We still record the LED *names* so handleApply has something to
    // iterate over, but value/enabled stay null until the user edits the
    // preset manually.
    try {
      const namesRes = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/LEDController/getLEDNames`,
      );
      if (namesRes.ok) {
        const names = await namesRes.json();
        if (Array.isArray(names)) {
          result.leds = names.map((name) => ({ name, intensity: null, enabled: null }));
        }
      }
    } catch (_e) { /* no LED controller — leave leds empty */ }

    return result;
  };

  const handleSave = async () => {
    const name = newName.trim();
    if (!name) { setError("Please enter a name for the preset."); return; }
    setError("");

    const { exposure, gain } = await fetchExposureGain();
    const illumination = await fetchIlluminationState();

    const newPreset = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      name,
      createdAt: new Date().toISOString(),
      schemaVersion: 2,
      // Detector
      currentDetector: currentSnapshot.currentDetector,
      // Frontend state snapshot
      imageFormat: currentSnapshot.imageFormat,
      streamSettings: currentSnapshot.streamSettings,
      snapFormat: currentSnapshot.snapFormat,
      recordFormat: currentSnapshot.recordFormat,
      // Detector exposure/gain state
      exposure,
      gain,
      // Objective state
      objective: currentSnapshot.objective,
      // Illumination state (v2)
      illumination,
    };

    setPresets((prev) => [...prev, newPreset]);
    setNewName("");
    setSaveOpen(false);
    setInfo(`Saved preset "${name}".`);
  };

  const handleDelete = (id) => {
    setPresets((prev) => prev.filter((p) => p.id !== id));
    if (selectedId === id) setSelectedId("");
  };

  /** Apply a preset: dispatch Redux updates + fire matching backend calls. */
  const handleApply = async (preset) => {
    if (!preset) return;
    setError("");
    setInfo("");
    try {
      // 0) Switch detector tab if the preset has one
      if (preset.currentDetector) {
        const detectors = liveViewState.detectors || [];
        const idx = detectors.indexOf(preset.currentDetector);
        if (idx >= 0) {
          dispatch(liveViewSlice.setActiveTab(idx));
        }
      }

      // 1) Stream format + settings → Redux + backend
      if (preset.imageFormat) {
        dispatch(liveStreamSlice.setImageFormat(preset.imageFormat));
      }
      if (preset.streamSettings) {
        dispatch(liveStreamSlice.setStreamSettings(preset.streamSettings));
      }
      // Push the protocol-specific params to the backend so the actual stream
      // matches what the preset describes.
      if (preset.streamSettings) {
        const bin = preset.streamSettings.binary;
        const jpg = preset.streamSettings.jpeg;
        if (bin?.enabled) {
          try {
            await apiLiveViewControllerSetStreamParameters("binary", {
              compression_algorithm: bin.compression?.algorithm,
              compression_level: bin.compression?.level,
              subsampling_factor: bin.subsampling?.factor ?? bin.subsampling_factor ?? 4,
              throttle_ms: bin.throttle_ms ?? 100,
            });
          } catch (_e) { /* non-fatal */ }
        }
        if (jpg?.enabled) {
          try {
            await apiLiveViewControllerSetStreamParameters("jpeg", {
              jpeg_quality: jpg.quality,
              subsampling_factor: jpg.subsampling?.factor ?? jpg.subsampling_factor ?? 1,
              throttle_ms: jpg.throttle_ms ?? 100,
            });
          } catch (_e) { /* non-fatal */ }
        }
      }

      // 2) Snap / record format
      if (preset.snapFormat != null) {
        dispatch(liveViewSlice.setSnapFormat(preset.snapFormat));
      }
      if (preset.recordFormat != null) {
        dispatch(liveViewSlice.setRecordFormat(preset.recordFormat));
      }

      // 3) Detector exposure/gain → backend
      if (preset.exposure != null) {
        try {
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorExposureTime?exposureTime=${preset.exposure}`,
          );
        } catch (_e) { /* non-fatal */ }
      }
      if (preset.gain != null) {
        try {
          await fetch(
            `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorGain?gain=${preset.gain}`,
          );
        } catch (_e) { /* non-fatal */ }
      }

      // 4) Objective slot
      if (preset.objective?.currentObjective != null) {
        try {
          await apiObjectiveControllerMoveToObjective(preset.objective.currentObjective, true);
          dispatch(objectiveSlice.setCurrentObjective(preset.objective.currentObjective));
        } catch (_e) { /* non-fatal */ }
      }

      // 5) Illumination (lasers + LEDs). Each call is wrapped individually so
      // one failed device (e.g. safety-interlocked laser) doesn't block the
      // rest of the preset from applying. Failures are collected and surfaced
      // as a single warning at the end.
      const illuErrors = [];
      const illu = preset.illumination || {};
      for (const laser of illu.lasers || []) {
        if (!laser?.name) continue;
        const enc = encodeURIComponent(laser.name);
        if (laser.power != null) {
          try {
            const r = await fetch(
              `${hostIP}:${hostPort}/imswitch/api/LaserController/setLaserValue?laserName=${enc}&value=${laser.power}`,
            );
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
          } catch (e) {
            illuErrors.push(`laser ${laser.name} power: ${e.message || e}`);
          }
        }
        if (laser.enabled != null) {
          try {
            const r = await fetch(
              `${hostIP}:${hostPort}/imswitch/api/LaserController/setLaserActive?laserName=${enc}&active=${laser.enabled}`,
            );
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
          } catch (e) {
            illuErrors.push(`laser ${laser.name} active: ${e.message || e}`);
          }
        }
      }
      for (const led of illu.leds || []) {
        if (!led?.name) continue;
        const enc = encodeURIComponent(led.name);
        if (led.intensity != null) {
          try {
            const r = await fetch(
              `${hostIP}:${hostPort}/imswitch/api/LEDController/setLEDValue?ledName=${enc}&value=${led.intensity}`,
            );
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
          } catch (e) {
            illuErrors.push(`led ${led.name} value: ${e.message || e}`);
          }
        }
        if (led.enabled != null) {
          try {
            const r = await fetch(
              `${hostIP}:${hostPort}/imswitch/api/LEDController/setLEDActive?ledName=${enc}&active=${led.enabled}`,
            );
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
          } catch (e) {
            illuErrors.push(`led ${led.name} active: ${e.message || e}`);
          }
        }
      }

      // 6) Re-confirm backend stream params, so the UI labels match reality.
      try {
        await apiLiveViewControllerGetStreamParameters();
      } catch (_e) { /* ignore */ }

      if (illuErrors.length > 0) {
        setError(
          `Applied preset "${preset.name}", but some illumination devices failed: ${illuErrors.join("; ")}`,
        );
      } else {
        setInfo(`Applied preset "${preset.name}".`);
      }
    } catch (e) {
      setError(`Failed to apply preset: ${e.message || e}`);
    }
  };

  const handleApplySelected = () => {
    const p = presets.find((x) => x.id === selectedId);
    if (p) handleApply(p);
  };

  return (
    <Box
      sx={{
        border: 1,
        borderColor: "divider",
        borderRadius: 1,
        p: 1.5,
        mt: 1,
      }}
    >
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
        <BookmarkIcon fontSize="small" color="action" />
        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
          Stream Presets
        </Typography>
        <Tooltip title="Frontend-only macros for objective, exposure, gain and livestream parameters">
          <Chip label={`${presets.length} saved`} size="small" variant="outlined" />
        </Tooltip>
      </Stack>

      {error && <Alert severity="error" sx={{ mb: 1 }} onClose={() => setError("")}>{error}</Alert>}
      {info && <Alert severity="success" sx={{ mb: 1 }} onClose={() => setInfo("")}>{info}</Alert>}

      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1, flexWrap: "wrap" }}>
        <FormControl size="small" sx={{ minWidth: 200, flexGrow: 1 }}>
          <InputLabel>Saved preset</InputLabel>
          <Select
            label="Saved preset"
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {presets.length === 0 && (
              <MenuItem value="" disabled>
                <em>(none saved yet)</em>
              </MenuItem>
            )}
            {presets.map((p) => {
              const hasIllu =
                (p.illumination?.lasers || []).some((l) => l?.power != null || l?.enabled) ||
                (p.illumination?.leds || []).some((l) => l?.intensity != null || l?.enabled);
              return (
                <MenuItem key={p.id} value={p.id}>
                  {p.name}
                  {p.currentDetector ? ` [${p.currentDetector}]` : ""}
                  {p.imageFormat ? ` · ${p.imageFormat}` : ""}
                  {p.objective?.name ? ` — ${p.objective.name}` : ""}
                  {p.exposure != null ? ` — ${p.exposure}ms` : ""}
                  {hasIllu && (
                    <IlluminationIcon
                      fontSize="inherit"
                      sx={{ ml: 0.75, verticalAlign: "middle", color: "warning.main" }}
                    />
                  )}
                </MenuItem>
              );
            })}
          </Select>
        </FormControl>
        <Tooltip title="Apply the selected preset">
          <span>
            <Button
              size="small"
              variant="contained"
              startIcon={<ApplyIcon />}
              disabled={!selectedId}
              onClick={handleApplySelected}
            >
              Apply
            </Button>
          </span>
        </Tooltip>
        <Tooltip title="Delete the selected preset">
          <span>
            <IconButton
              size="small"
              color="error"
              disabled={!selectedId}
              onClick={() => handleDelete(selectedId)}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Save the current settings as a new preset">
          <Button
            size="small"
            variant="outlined"
            startIcon={<SaveIcon />}
            onClick={() => { setNewName(""); setError(""); setSaveOpen(true); }}
          >
            Save current
          </Button>
        </Tooltip>
      </Stack>

      <Dialog open={saveOpen} onClose={() => setSaveOpen(false)} fullWidth maxWidth="xs">
        <DialogTitle>Save current settings as preset</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Stored locally in your browser only. The following parameters will be saved:
          </Typography>
          <Box sx={{ mb: 2, p: 1, bgcolor: "action.hover", borderRadius: 1 }}>
            <Stack spacing={0.75}>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                <Chip size="small" color="primary" variant="outlined" label={`Detector: ${currentSnapshot.currentDetector || "(none)"}`} />
                <Chip size="small" label={`Protocol: ${currentSnapshot.imageFormat || "?"}`} />
              </Stack>
              {currentSnapshot.streamSettings?.jpeg?.enabled && (
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  <Chip size="small" label={`JPEG quality: ${currentSnapshot.streamSettings.jpeg.quality ?? "?"}`} />
                  <Chip size="small" label={`JPEG subsample: ×${currentSnapshot.streamSettings.jpeg.subsampling?.factor ?? currentSnapshot.streamSettings.jpeg.subsampling_factor ?? "?"}`} />
                  <Chip size="small" label={`JPEG throttle: ${currentSnapshot.streamSettings.jpeg.throttle_ms ?? "?"}ms`} />
                </Stack>
              )}
              {currentSnapshot.streamSettings?.binary?.enabled && (
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  <Chip size="small" label={`Codec: ${currentSnapshot.streamSettings.binary.compression?.algorithm ?? "?"} L${currentSnapshot.streamSettings.binary.compression?.level ?? "?"}`} />
                  <Chip size="small" label={`Bin subsample: ×${currentSnapshot.streamSettings.binary.subsampling?.factor ?? "?"}`} />
                  <Chip size="small" label={`Bin throttle: ${currentSnapshot.streamSettings.binary.throttle_ms ?? "?"}ms`} />
                </Stack>
              )}
              {currentSnapshot.streamSettings?.webrtc?.enabled && (
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  <Chip size="small" label={`WebRTC max-w: ${currentSnapshot.streamSettings.webrtc.max_width ?? "?"}`} />
                  <Chip size="small" label={`WebRTC subsample: ×${currentSnapshot.streamSettings.webrtc.subsampling_factor ?? "?"}`} />
                  <Chip size="small" label={`WebRTC throttle: ${currentSnapshot.streamSettings.webrtc.throttle_ms ?? "?"}ms`} />
                </Stack>
              )}
              <Stack direction="row" spacing={1} flexWrap="wrap">
                <Chip size="small" label={`Objective: ${currentSnapshot.objective?.name || `#${currentSnapshot.objective?.currentObjective ?? "?"}`}`} />
                <Chip size="small" label={`Snap: ${currentSnapshot.snapFormat}`} variant="outlined" />
                <Chip size="small" label={`Rec: ${currentSnapshot.recordFormat}`} variant="outlined" />
                <Chip size="small" label="exposure + gain (fetched on save)" variant="outlined" />
                <Chip
                  size="small"
                  icon={<IlluminationIcon />}
                  label="illumination (fetched on save)"
                  variant="outlined"
                  color="warning"
                />
              </Stack>
            </Stack>
          </Box>
          <TextField
            autoFocus
            fullWidth
            label="Preset name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="e.g. Overview 4× JPEG"
          />
          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleSave} disabled={!newName.trim()}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default StreamPresets;
