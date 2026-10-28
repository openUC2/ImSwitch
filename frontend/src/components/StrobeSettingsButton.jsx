// Strobe timing for the strobed prescan: flash width and delay, the camera
// exposure window, the shortest frame period and the trigger pulse. Values are
// saved to the backend on blur/Enter, where they persist across restarts
// (config/stagemap_strobe.json). "Calibrate" runs the full calibration with the
// stage still and shows whether every camera frame got exactly one flash.
import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Button,
  Checkbox,
  CircularProgress,
  FormControlLabel,
  IconButton,
  Popover,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import TuneIcon from "@mui/icons-material/Tune";

import * as stageMapSlice from "../state/slices/StageMapSlice.js";
import {
  apiStageMapCalibrateStrobe,
  apiStageMapGetParams,
  apiStageMapSetParams,
} from "../backendapi/apiStageMapController";

// auto: the backend value an empty field stands for (the field shows empty).
// scale: shown value = stored value * scale.
const FIELDS = [
  { key: "strobeWidthUs", label: "Flash width", unit: "µs", min: 1, max: 1000,
    help: "How long the LED is on per frame. Sets brightness and motion blur." },
  { key: "strobeDelayUs", label: "Flash delay", unit: "µs", min: 0, auto: -1,
    help: "From camera trigger to flash. Empty: near the end of the window." },
  { key: "strobeWindowMs", label: "Exposure window", unit: "ms", min: 0.1, max: 1000,
    help: "Camera exposure during strobed sweeps. Must be longer than the sensor readout plus the flash, or no delay lights every row; calibration lengthens it when needed." },
  { key: "strobeMinPeriodMs", label: "Shortest frame period", unit: "ms", min: 0, auto: 0,
    help: "Set by calibration when the camera skips triggers. Empty: the camera's own frame period at this window (exposure + readout), or window + 1 ms. The prescan slows down to one field per frame." },
  { key: "strobeTrigUs", label: "Trigger pulse", unit: "µs", min: 1, max: 10000,
    help: "Width of the camera trigger pulse." },
  { key: "strobeTargetLevel", label: "Target brightness", unit: "%", min: 5, max: 95, scale: 100,
    help: "Calibration adjusts the flash width until a lit frame averages this much of full scale." },
];

const toDraft = (params) =>
  Object.fromEntries(
    FIELDS.map((f) => {
      const v = params?.[f.key];
      if (v === undefined || v === null) return [f.key, ""];
      if (f.auto !== undefined && (v === f.auto || v < 0)) return [f.key, ""];
      const shown = f.scale ? v * f.scale : v;
      return [f.key, String(Math.round(shown * 100) / 100)];
    }),
  );

const ms = (us) => `${(us / 1000).toFixed(1)} ms`;

function CalibrationSummary({ result }) {
  if (!result) return null;
  if (!result.success) {
    return (
      <Typography variant="caption" color="error">
        {result.error || "Calibration failed"}
      </Typography>
    );
  }
  const range = result.litRangeUs;
  const hints = result.hints || [];
  return (
    <Stack spacing={0.25}>
      <Typography variant="caption">
        Flash delay {ms(result.bestDelayUs)}
        {range ? ` (all rows lit from ${ms(range[0])} to ${ms(range[1])})` : ""}
      </Typography>
      {result.widthUs !== undefined && (
        <Typography variant="caption">
          Flash width {Math.round(result.widthFromUs)} → {Math.round(result.widthUs)} µs,
          brightness {Math.round((result.level || 0) * 100)} %
        </Typography>
      )}
      {result.windowFromUs !== undefined &&
        result.windowUs !== undefined &&
        Math.abs(result.windowUs - result.windowFromUs) > 1 && (
          <Typography variant="caption">
            Exposure window {ms(result.windowFromUs)} → {ms(result.windowUs)}
            {result.readoutUs ? ` (sensor readout ≈ ${ms(result.readoutUs)})` : ""}
          </Typography>
        )}
      {hints.map((h, i) => (
        <Typography
          key={h}
          variant="caption"
          color={i === 0 && result.matched ? "success.main" : "warning.main"}
        >
          {h}
        </Typography>
      ))}
    </Stack>
  );
}

export default function StrobeSettingsButton({ disabled = false }) {
  const dispatch = useDispatch();
  const lastCalibration = useSelector(
    (state) => stageMapSlice.getStageMapState(state)?.strobeCalibration,
  );
  const [anchor, setAnchor] = useState(null);
  const [params, setParams] = useState(null);
  const [draft, setDraft] = useState({});
  const [adjustWidth, setAdjustWidth] = useState(true);
  const [calibrating, setCalibrating] = useState(false);
  const [error, setError] = useState("");

  const load = () =>
    apiStageMapGetParams()
      .then((p) => {
        setParams(p);
        setDraft(toDraft(p));
        setError("");
      })
      .catch(() => setError("Could not load the strobe settings"));

  const open = (e) => {
    setAnchor(e.currentTarget);
    load();
  };

  const commit = (field) => {
    if (!params) return;
    const text = (draft[field.key] ?? "").trim();
    let value;
    if (text === "") {
      if (field.auto === undefined) {
        setDraft(toDraft(params));
        return;
      }
      value = field.auto;
    } else {
      const n = Number(text);
      if (!Number.isFinite(n)) {
        setDraft(toDraft(params));
        return;
      }
      const clamped = Math.min(field.max ?? Infinity, Math.max(field.min ?? -Infinity, n));
      value = field.scale ? clamped / field.scale : clamped;
    }
    if (value === params[field.key]) {
      setDraft(toDraft(params));
      return;
    }
    const next = { ...params, [field.key]: value };
    apiStageMapSetParams(next)
      .then(() => {
        setParams(next);
        setDraft(toDraft(next));
        setError("");
      })
      .catch(() => setError("Could not save the strobe settings"));
  };

  const calibrate = () => {
    setCalibrating(true);
    apiStageMapCalibrateStrobe({ adjustWidth })
      .then((result) => {
        dispatch(stageMapSlice.setStrobeCalibration(result));
        return load();
      })
      .catch(() =>
        dispatch(
          stageMapSlice.setStrobeCalibration({
            success: false,
            error: "Calibration request failed",
          }),
        ),
      )
      .finally(() => setCalibrating(false));
  };

  return (
    <>
      <Tooltip title="Strobe timing and calibration" arrow>
        <span>
          <IconButton size="small" onClick={open} disabled={disabled}>
            <TuneIcon fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>
      <Popover
        open={Boolean(anchor)}
        anchorEl={anchor}
        onClose={() => setAnchor(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      >
        <Box sx={{ p: 2, width: 330 }}>
          <Typography variant="subtitle2" gutterBottom>
            Strobe timing
          </Typography>
          {!params ? (
            <Typography variant="body2" color="text.secondary">
              {error || "Loading…"}
            </Typography>
          ) : (
            <Stack spacing={1.25}>
              {FIELDS.map((f) => (
                <Tooltip key={f.key} title={f.help} placement="left" arrow>
                  <TextField
                    id={`strobe-${f.key}`}
                    size="small"
                    label={`${f.label} (${f.unit})`}
                    placeholder={f.auto !== undefined ? "auto" : undefined}
                    value={draft[f.key] ?? ""}
                    disabled={calibrating}
                    onChange={(e) => setDraft((d) => ({ ...d, [f.key]: e.target.value }))}
                    onBlur={() => commit(f)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") e.target.blur();
                    }}
                    InputLabelProps={{ shrink: true }}
                  />
                </Tooltip>
              ))}
              <FormControlLabel
                control={
                  <Checkbox
                    size="small"
                    checked={adjustWidth}
                    onChange={(e) => setAdjustWidth(e.target.checked)}
                    disabled={calibrating}
                  />
                }
                label={<Typography variant="body2">Adjust flash width to target brightness</Typography>}
              />
              <Tooltip
                title="The stage stays still. Finds the delay at which one flash lights every sensor row, sets the flash width, then checks that each trigger gives one evenly lit frame at the prescan's frame rate. Results are saved."
                arrow
              >
                <span>
                  <Button
                    variant="outlined"
                    size="small"
                    fullWidth
                    onClick={calibrate}
                    disabled={calibrating}
                    startIcon={calibrating ? <CircularProgress size={16} /> : null}
                  >
                    {calibrating ? "Calibrating…" : "Calibrate"}
                  </Button>
                </span>
              </Tooltip>
              <CalibrationSummary result={lastCalibration} />
              {error && (
                <Typography variant="caption" color="error">
                  {error}
                </Typography>
              )}
            </Stack>
          )}
        </Box>
      </Popover>
    </>
  );
}
