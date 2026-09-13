// Kiosk laser control: one large card per illumination source with an on/off
// switch and an intensity slider. Values are sent on release (not while
// dragging) so the serial link never floods.
//
// Source list + ranges come from ExperimentController/getHardwareParameters
// (ParameterRangeSlice) — NOT LaserController/getLaserNames, which would
// clobber the LED-matrix synthetic channels (see IlluminationController.js).
// Only kind === "default" sources are shown here; the LED matrix has its own
// page.
import { useEffect, useMemo, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Paper,
  Slider,
  Switch,
  Typography,
} from "@mui/material";
import FlareRoundedIcon from "@mui/icons-material/FlareRounded";
import { enqueueSnackbar } from "notistack";

import MobilePage from "../components/MobilePage";
import PlaceholderImage from "../components/PlaceholderImage";
import * as laserSlice from "../../state/slices/LaserSlice";
import { getParameterRangeState } from "../../state/slices/ParameterRangeSlice";
import { getConnectionSettingsState } from "../../state/slices/ConnectionSettingsSlice";
import fetchExperimentControllerGetCurrentExperimentParams from "../../middleware/fetchExperimentControllerGetCurrentExperimentParams";
import fetchLaserRuntimeState from "../../middleware/fetchLaserRuntimeState";
import apiLaserControllerSetLaserValue from "../../backendapi/apiLaserControllerSetLaserValue";
import apiLaserControllerSetLaserActive from "../../backendapi/apiLaserControllerSetLaserActive";

// Rough wavelength-name → display color mapping ("488", "Laser 635", ...).
const laserColor = (name) => {
  const match = String(name).match(/(\d{3})/);
  if (!match) return "#9e9e9e";
  const nm = parseInt(match[1], 10);
  if (nm < 430) return "#8b5cf6";
  if (nm < 500) return "#38bdf8";
  if (nm < 565) return "#4ade80";
  if (nm < 600) return "#facc15";
  if (nm < 700) return "#ef4444";
  return "#f87171";
};

const LaserCard = ({ name, min, max, power, enabled }) => {
  const dispatch = useDispatch();
  const [draft, setDraft] = useState(null); // slider value while dragging

  const commitValue = (value) => {
    setDraft(null);
    dispatch(laserSlice.setLaserPower({ laserName: name, power: value }));
    apiLaserControllerSetLaserValue(name, value).catch(() =>
      enqueueSnackbar(`Could not set ${name}`, { variant: "error" }),
    );
  };

  const toggleActive = (event) => {
    const active = event.target.checked;
    dispatch(laserSlice.setLaserEnabled({ laserName: name, enabled: active }));
    apiLaserControllerSetLaserActive(name, active).catch(() =>
      enqueueSnackbar(`Could not switch ${name}`, { variant: "error" }),
    );
  };

  const value = draft ?? power ?? 0;

  return (
    <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1 }}>
        <Box
          sx={{
            width: 16,
            height: 16,
            borderRadius: "50%",
            bgcolor: laserColor(name),
            boxShadow: enabled ? `0 0 10px ${laserColor(name)}` : "none",
            opacity: enabled ? 1 : 0.4,
            flexShrink: 0,
          }}
        />
        <Typography variant="h6" sx={{ flex: 1 }} noWrap>
          {name}
        </Typography>
        <Typography
          variant="h6"
          sx={{ fontVariantNumeric: "tabular-nums", color: "text.secondary" }}
        >
          {Math.round(value)}
        </Typography>
        <Switch checked={Boolean(enabled)} onChange={toggleActive} />
      </Box>
      <Slider
        value={value}
        min={min}
        max={max}
        step={1}
        disabled={!enabled}
        onChange={(e, v) => setDraft(v)}
        onChangeCommitted={(e, v) => commitValue(v)}
        valueLabelDisplay="auto"
      />
    </Paper>
  );
};

const MobileLaserPage = () => {
  const dispatch = useDispatch();
  const parameterRange = useSelector(getParameterRangeState);
  const lasers = useSelector(laserSlice.getLasers);
  const connectionSettings = useSelector(getConnectionSettingsState);

  const sources = useMemo(() => {
    const kinds = parameterRange.illuSourceKinds || [];
    return (parameterRange.illuSources || [])
      .map((name, i) => ({
        name,
        kind: kinds[i] || "default",
        min: parameterRange.illuSourceMinIntensities?.[i] ?? 0,
        max: parameterRange.illuSourceMaxIntensities?.[i] ?? 1023,
      }))
      .filter((s) => s.kind === "default");
  }, [parameterRange]);

  // Populate the source list if nobody fetched it yet, then sync the current
  // power/enabled state of each laser once.
  useEffect(() => {
    if (!parameterRange.illuSources?.length) {
      fetchExperimentControllerGetCurrentExperimentParams(dispatch);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!sources.length) return;
    let cancelled = false;
    fetchLaserRuntimeState({
      hostIP: connectionSettings.ip,
      hostPort: connectionSettings.apiPort,
      sources: sources.map((s) => s.name),
      kinds: sources.map((s) => s.kind),
    }).then((results) => {
      if (cancelled || !Array.isArray(results)) return;
      const update = {};
      results
        .filter((r) => r.ok)
        .forEach((r) => {
          update[r.laserName] = { power: r.power, enabled: r.enabled };
        });
      if (Object.keys(update).length) {
        dispatch(laserSlice.setLasersState(update));
      }
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sources.length]);

  return (
    <MobilePage title="Lasers" subtitle="Tap the switch, then drag — intensity is sent when you let go">
      {sources.length === 0 ? (
        <Box sx={{ maxWidth: 480 }}>
          <PlaceholderImage
            label="No laser sources reported by this setup"
            height={160}
          />
        </Box>
      ) : (
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
            gap: 2.5,
          }}
        >
          {sources.map((s) => (
            <LaserCard
              key={s.name}
              name={s.name}
              min={s.min}
              max={s.max}
              power={lasers[s.name]?.power}
              enabled={lasers[s.name]?.enabled}
            />
          ))}
        </Box>
      )}
      <Box sx={{ mt: 3, display: "flex", alignItems: "center", gap: 1, color: "text.disabled" }}>
        <FlareRoundedIcon fontSize="small" />
        <Typography variant="caption">
          Laser safety: never look into the beam path while a source is enabled.
        </Typography>
      </Box>
    </MobilePage>
  );
};

export default MobileLaserPage;
