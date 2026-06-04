import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../../backendapi/createAxiosInstance';
import {
  Box,
  Typography,
  Slider,
  TextField,
  Switch,
  FormControlLabel,
  IconButton,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Collapse,
  Chip,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Checkbox,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DragIndicatorIcon from "@mui/icons-material/DragIndicator";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import SettingsIcon from "@mui/icons-material/Settings";
import DeleteIcon from "@mui/icons-material/Delete";
import AddIcon from "@mui/icons-material/Add";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import ScienceIcon from "@mui/icons-material/Science";
import PowerSettingsNewIcon from "@mui/icons-material/PowerSettingsNew";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as parameterRangeSlice from "../../state/slices/ParameterRangeSlice";
import * as connectionSettingsSlice from "../../state/slices/ConnectionSettingsSlice";
import * as laserSlice from "../../state/slices/LaserSlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";
import fetchLaserControllerCurrentValues from "../../middleware/fetchLaserControllerCurrentValues";
import apiLEDMatrixControllerSetRing from "../../backendapi/apiLEDMatrixControllerSetRing";
import apiLEDMatrixControllerSetHalves from "../../backendapi/apiLEDMatrixControllerSetHalves";
import apiLEDMatrixControllerSetAllLED from "../../backendapi/apiLEDMatrixControllerSetAllLED";

/**
 * Single channel block - collapsible card for each illumination source.
 *
 * `kind` drives the controls rendered inside the collapsible body:
 *   - "default" : intensity slider + exposure + gain (legacy laser path)
 *   - "ring"    : radius slider + RGB sliders + exposure + gain (LED matrix)
 *   - "dpc"     : RGB sliders (locked-default 255 on one colour) + exposure + gain
 *
 * `kindParams` is the per-channel params dict from
 * experimentState.parameterValue.illuminationParams[channelName].
 * `onKindParamChange(key, value)` writes one field back to that dict via the
 * setIlluminationParamsForChannel reducer.
 */
const ChannelBlock = ({
  channelName,
  channelIndex,
  intensity,
  exposure,
  gain,
  minIntensity,
  maxIntensity,
  isEnabled,
  isIncludedInExperiment,
  isExpanded,
  onToggleExpand,
  onIntensityChange,
  onExposureChange,
  onGainChange,
  onEnabledChange,
  onIncludeInExperimentChange,
  onRemove,
  kind = "default",
  kindParams = {},
  onKindParamChange = () => {},
  ringMaxRadius = 3,
}) => {
  const theme = useTheme();
  const isSynthetic = kind === "ring" || kind === "dpc";

  // Pick a chip colour per kind so the user can spot LED-matrix channels at a glance.
  const kindChipColor = kind === "ring" ? "secondary" : kind === "dpc" ? "warning" : "primary";
  const kindBadgeLabel = kind === "ring" ? "Ring" : kind === "dpc" ? "DPC" : null;

  // Local string state for exposure so the user can type partial values
  // (e.g. "100." or clear the field) without it snapping to 0.
  const [exposureRaw, setExposureRaw] = useState(String(exposure ?? 100));
  useEffect(() => { setExposureRaw(String(exposure)); }, [exposure]);
  const commitExposure = () => {
    const v = parseFloat(exposureRaw);
    if (!isNaN(v) && v >= 0) onExposureChange(v);
    else setExposureRaw(String(exposure));
  };

  return (
    <Box
      sx={{
        border: `1px solid ${theme.palette.divider}`,
        borderRadius: 1,
        mb: 1,
        overflow: "hidden",
        backgroundColor: alpha(theme.palette.background.paper, 0.5),
      }}
    >
      {/* Channel Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          padding: "8px 12px",
          backgroundColor: alpha(theme.palette.primary.main, 0.05),
          cursor: "pointer",
          "&:hover": {
            backgroundColor: alpha(theme.palette.primary.main, 0.1),
          },
        }}
        onClick={onToggleExpand}
      >
        {/* Drag handle placeholder */}
        <DragIndicatorIcon
          sx={{
            color: theme.palette.text.disabled,
            fontSize: 18,
            mr: 1,
            cursor: "grab",
          }}
        />

        {/* Channel name and badge */}
        <Chip
          label={channelName}
          size="small"
          color={kindChipColor}
          variant="outlined"
          sx={{ mr: 1, fontWeight: 600 }}
        />
        {kindBadgeLabel && (
          <Chip
            label={kindBadgeLabel}
            size="small"
            color={kindChipColor}
            variant="filled"
            sx={{ mr: 1.5, height: 18, fontSize: 10 }}
          />
        )}

        {/* Quick info when collapsed */}
        {!isExpanded && (
          <Typography
            variant="caption"
            sx={{ color: theme.palette.text.secondary, flex: 1 }}
          >
            {kind === "ring" && (
              <>
                r={kindParams.radius ?? 8} · RGB({kindParams.intensityR ?? 0},{" "}
                {kindParams.intensityG ?? 0}, {kindParams.intensityB ?? 0}) · {exposure} ms · Gain {gain}
              </>
            )}
            {kind === "dpc" && (
              <>
                4× halves · RGB({kindParams.intensityR ?? 0},{" "}
                {kindParams.intensityG ?? 255}, {kindParams.intensityB ?? 0}) · {exposure} ms · Gain {gain}
              </>
            )}
            {kind === "default" && (
              <>
                {intensity} mW · {exposure} ms · Gain {gain}
              </>
            )}
            {" · "}
            {isIncludedInExperiment ? "✔ Exp" : "✘ Exp"}
            {!isSynthetic && <> · {isEnabled ? "ON" : "OFF"}</>}
          </Typography>
        )}

        {/* Laser power toggle – physically turns the laser on/off.
            Hidden for LED-matrix synthetic channels: those have no
            stand-alone "on" state — the pattern is driven per-frame by the
            workflow, so the toggle would be misleading. */}
        {!isSynthetic && (
          <Tooltip title="Toggle laser ON/OFF (physically enables the illumination source)" arrow>
            <Checkbox
              checked={isEnabled}
              onChange={(e) => {
                e.stopPropagation();
                onEnabledChange(e.target.checked);
              }}
              onClick={(e) => e.stopPropagation()}
              size="small"
              icon={<PowerSettingsNewIcon sx={{ fontSize: 20, opacity: 0.4 }} />}
              checkedIcon={<PowerSettingsNewIcon sx={{ fontSize: 20, color: theme.palette.success.main }} />}
              sx={{ mr: 0 }}
            />
          </Tooltip>
        )}

        {/* Include in experiment toggle – determines if this channel is used during acquisition */}
        <Tooltip title="Include this channel in the experiment acquisition (does NOT toggle the laser)" arrow>
          <Checkbox
            checked={isIncludedInExperiment}
            onChange={(e) => {
              e.stopPropagation();
              onIncludeInExperimentChange(e.target.checked);
            }}
            onClick={(e) => e.stopPropagation()}
            size="small"
            icon={<ScienceIcon sx={{ fontSize: 20, opacity: 0.4 }} />}
            checkedIcon={<ScienceIcon sx={{ fontSize: 20, color: theme.palette.info.main }} />}
            sx={{ mr: 0.5 }}
          />
        </Tooltip>

        {/* Expand/collapse indicator */}
        <ExpandMoreIcon
          sx={{
            transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
            color: theme.palette.text.secondary,
          }}
        />
      </Box>

      {/* Channel Parameters */}
      <Collapse in={isExpanded}>
        <Box sx={{ p: 2 }}>
          {/* Intensity / kind-specific controls */}
          {kind === "default" && (
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: "flex", alignItems: "center", mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  Intensity
                </Typography>
                <Tooltip title="Illumination power in mW. Higher values give brighter images but may cause photobleaching." arrow>
                  <InfoOutlinedIcon sx={{ fontSize: 14, ml: 0.5, color: "text.disabled", cursor: "help" }} />
                </Tooltip>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Slider
                  value={intensity}
                  min={minIntensity}
                  max={maxIntensity}
                  onChange={(e, val) => onIntensityChange(val)}
                  sx={{ flex: 1 }}
                />
                <Typography
                  variant="body2"
                  sx={{ minWidth: "60px", textAlign: "right", fontWeight: 500 }}
                >
                  {intensity} mW
                </Typography>
              </Box>
            </Box>
          )}

          {/* Ring: radius slider on top of RGB triplet.
              Max radius comes from backend ledMatrixInfo.maxRingRadius
              (clamped to the physical matrix size).  For the common 8×8
              ESP32 matrix this is 3 (rings 0..3). */}
          {kind === "ring" && (
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: "flex", alignItems: "center", mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  Ring Radius (0–{ringMaxRadius})
                </Typography>
                <Tooltip
                  title={`LED-matrix ring radius in LED units (0–${ringMaxRadius}, derived from the physical matrix size). Larger radius = more oblique illumination; r=0 = single centre LED.`}
                  arrow
                >
                  <InfoOutlinedIcon sx={{ fontSize: 14, ml: 0.5, color: "text.disabled", cursor: "help" }} />
                </Tooltip>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Slider
                  value={Math.min(kindParams.radius ?? Math.min(2, ringMaxRadius), ringMaxRadius)}
                  min={0}
                  max={ringMaxRadius}
                  step={1}
                  marks
                  onChange={(e, val) => onKindParamChange("radius", val)}
                  sx={{ flex: 1 }}
                />
                <Typography
                  variant="body2"
                  sx={{ minWidth: "60px", textAlign: "right", fontWeight: 500 }}
                >
                  {Math.min(kindParams.radius ?? Math.min(2, ringMaxRadius), ringMaxRadius)}
                </Typography>
              </Box>
            </Box>
          )}

          {/* RGB sliders for ring + DPC.
              For DPC the default is (0, 255, 0) because anything below 255
              beats with the camera's rolling shutter / PWM duty cycle and
              produces banding in each half-illumination frame. */}
          {isSynthetic && (
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: "flex", alignItems: "center", mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  RGB Intensity (0–255)
                </Typography>
                <Tooltip
                  title={
                    kind === "dpc"
                      ? "Per-channel RGB intensity. Keep the active channel at 255 — lower values beat with the camera's rolling shutter and the LED PWM, producing visible bands across each DPC frame."
                      : "Per-channel RGB intensity for the LED matrix ring (0–255 each)."
                  }
                  arrow
                >
                  <InfoOutlinedIcon sx={{ fontSize: 14, ml: 0.5, color: "text.disabled", cursor: "help" }} />
                </Tooltip>
              </Box>
              {[
                { key: "intensityR", label: "R", color: theme.palette.error.main, defaultVal: kind === "dpc" ? 0 : 0 },
                { key: "intensityG", label: "G", color: theme.palette.success.main, defaultVal: kind === "dpc" ? 255 : 0 },
                { key: "intensityB", label: "B", color: theme.palette.info.main, defaultVal: kind === "dpc" ? 0 : 0 },
              ].map(({ key, label, color, defaultVal }) => (
                <Box key={key} sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 0.5 }}>
                  <Typography sx={{ width: 18, fontWeight: 700, color }}>{label}</Typography>
                  <Slider
                    value={kindParams[key] ?? defaultVal}
                    min={0}
                    max={255}
                    step={1}
                    onChange={(e, val) => onKindParamChange(key, val)}
                    sx={{ flex: 1, color }}
                  />
                  <Typography
                    variant="body2"
                    sx={{ minWidth: "44px", textAlign: "right", fontWeight: 500 }}
                  >
                    {kindParams[key] ?? defaultVal}
                  </Typography>
                </Box>
              ))}
              {kind === "dpc" && (
                <Typography variant="caption" sx={{ color: theme.palette.text.disabled, display: "block", mt: 0.5 }}>
                  DPC captures 4 frames per XY position (top / bottom / left / right halves) saved as channels
                  <code style={{ marginLeft: 4 }}>DPC_top</code>,
                  <code style={{ marginLeft: 4 }}>DPC_bottom</code>,
                  <code style={{ marginLeft: 4 }}>DPC_left</code>,
                  <code style={{ marginLeft: 4 }}>DPC_right</code>.
                </Typography>
              )}
            </Box>
          )}

          {/* Exposure and Gain side by side */}
          <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
            {/* Exposure */}
            <Box sx={{ flex: 1 }}>
              <Box sx={{ display: "flex", alignItems: "center", mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  Exposure
                </Typography>
                <Tooltip title="Camera exposure time per frame. Longer exposure captures more light but slows down acquisition." arrow>
                  <InfoOutlinedIcon sx={{ fontSize: 14, ml: 0.5, color: "text.disabled", cursor: "help" }} />
                </Tooltip>
              </Box>
              <FormControl size="small" fullWidth>
                <TextField
                  type="number"
                  size="small"
                  value={exposureRaw}
                  onChange={(e) => {
                    setExposureRaw(e.target.value);
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v) && v >= 0) onExposureChange(v);
                  }}
                  onBlur={commitExposure}
                  inputProps={{ min: 0, step: 0.1 }}
                  InputProps={{
                    endAdornment: <Typography variant="caption" sx={{ ml: 0.5, whiteSpace: 'nowrap' }}>ms</Typography>,
                  }}
                />
              </FormControl>
            </Box>

            {/* Gain */}
            <Box sx={{ flex: 1 }}>
              <Box sx={{ display: "flex", alignItems: "center", mb: 0.5 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  Gain
                </Typography>
                <Tooltip title="Camera sensor gain (amplification). Higher gain brightens the image but increases noise." arrow>
                  <InfoOutlinedIcon sx={{ fontSize: 14, ml: 0.5, color: "text.disabled", cursor: "help" }} />
                </Tooltip>
              </Box>
              <FormControl size="small" fullWidth>
                <Select
                  value={gain}
                  onChange={(e) => onGainChange(e.target.value)}
                >
                  {[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 23].map((val) => (
                    <MenuItem key={val} value={val}>
                      {val}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          </Box>

          {/* Performance Mode toggle removed - now in OutputDimension */}
        </Box>
      </Collapse>
    </Box>
  );
};

/**
 * ChannelsDimension - Channel configuration interface
 *
 * Contains:
 * - Collapsible channel blocks for each illumination source
 * - Intensity, exposure, gain per channel
 * - Performance mode toggle
 * - Copy settings to all channels action
 * - Advanced camera parameters (hidden by default)
 */
const ChannelsDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const parameterRange = useSelector(parameterRangeSlice.getParameterRangeState);
  const connectionSettings = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  const experimentUI = useSelector(experimentUISlice.getExperimentUIState);
  const laserState = useSelector(laserSlice.getLaserState);
  const lasers = laserState.lasers;

  // Local state for expanded channels
  const [expandedChannels, setExpandedChannels] = useState({});
  
  // Debounce refs for laser value updates to prevent serial overload
  const laserTimeoutRefs = useRef({});
  const LASER_UPDATE_DEBOUNCE_MS = 300;

  // Parameter values from experiment state
  const parameterValue = experimentState.parameterValue;
  // Defensive: every list must be an array. The backend can return null/object
  // when no hardware is connected — slice + UI must both tolerate that.
  const intensities = Array.isArray(parameterValue.illuIntensities) ? parameterValue.illuIntensities : [];
  const exposures = Array.isArray(parameterValue.exposureTimes) ? parameterValue.exposureTimes : [];
  const gains = Array.isArray(parameterValue.gains) ? parameterValue.gains : [];
  const channelEnabledForExperiment = Array.isArray(parameterValue.channelEnabledForExperiment) ? parameterValue.channelEnabledForExperiment : [];

  // Merge synthetic channels AFTER the conventional sources into one flat list
  // so the render loop and all index-based handlers below stay unchanged. The
  // submit payload is split back into `illumination` (default) + a dedicated
  // `syntheticChannels` list in ExperimentDesigner.handleStart — synthetic
  // channels are never mixed into illuIntensities. The default-source count
  // is the split boundary.
  //
  // IMPORTANT: memoize so these are STABLE array references across renders.
  // Otherwise the init effect below (deps: [illuSources]) sees a new array
  // every render and re-dispatches setIllumination on a loop → React
  // "Maximum update depth exceeded".
  const { illuSources, illuSourceKinds, laserMinValues, laserMaxValues } = useMemo(() => {
    const dSources = Array.isArray(parameterRange.illuSources) ? parameterRange.illuSources : [];
    const dKinds = Array.isArray(parameterRange.illuSourceKinds) ? parameterRange.illuSourceKinds : [];
    const dMin = Array.isArray(parameterRange.illuSourceMinIntensities) ? parameterRange.illuSourceMinIntensities : [];
    const dMax = Array.isArray(parameterRange.illuSourceMaxIntensities) ? parameterRange.illuSourceMaxIntensities : [];
    const synth = Array.isArray(parameterRange.syntheticChannels) ? parameterRange.syntheticChannels : [];
    return {
      illuSources: [...dSources, ...synth.map((s) => s.name)],
      illuSourceKinds: [
        ...dSources.map((_, i) => dKinds[i] || "default"),
        ...synth.map((s) => s.kind || "default"),
      ],
      laserMinValues: [...dMin, ...synth.map(() => 0)],
      laserMaxValues: [...dMax, ...synth.map(() => 255)],
    };
  }, [
    parameterRange.illuSources,
    parameterRange.illuSourceKinds,
    parameterRange.illuSourceMinIntensities,
    parameterRange.illuSourceMaxIntensities,
    parameterRange.syntheticChannels,
  ]);
  // Per-channel kind-specific params dict from the experiment slice.  Object
  // shape: { [channelName]: { radius?, intensityR?, intensityG?, intensityB? } }.
  const illuminationParams =
    parameterValue.illuminationParams && typeof parameterValue.illuminationParams === "object"
      ? parameterValue.illuminationParams
      : {};
  // LED matrix hardware bounds.  Backend exposes maxRingRadius from the
  // physical matrix size; we fall back to a safe 3 if absent (works for
  // the common 8×8 matrix).
  const ledMatrixInfo = parameterRange.ledMatrixInfo || null;
  const maxRingRadius =
    ledMatrixInfo && Number.isFinite(ledMatrixInfo.maxRingRadius)
      ? Math.max(0, Math.floor(ledMatrixInfo.maxRingRadius))
      : 3;

  // Initialize timeout refs and cleanup
  useEffect(() => {
    if (illuSources.length > 0) {
      illuSources.forEach(laserName => {
        if (!laserTimeoutRefs.current[laserName]) {
          laserTimeoutRefs.current[laserName] = null;
        }
        // Initialize laser state in Redux if not exists
        if (!lasers[laserName]) {
          dispatch(laserSlice.setLaserState({ laserName, power: 0, enabled: false }));
        }
      });
    }
    
    return () => {
      // Clear all pending timeouts on unmount
      Object.values(laserTimeoutRefs.current).forEach(timeoutRef => {
        if (timeoutRef) clearTimeout(timeoutRef);
      });
    };
  }, [illuSources, dispatch]);

  // Initialize arrays if needed
  useEffect(() => {
    if (illuSources.length > 0) {
      const initIntensities = illuSources.map((_, idx) => intensities[idx] ?? 0);
      const initExposures = illuSources.map((_, idx) => exposures[idx] ?? 100);
      const initGains = illuSources.map((_, idx) => gains[idx] ?? 0);
      // Default: all channels excluded from experiment (user must explicitly enable)
      const initChannelEnabled = illuSources.map((_, idx) => channelEnabledForExperiment[idx] ?? false);

      if (JSON.stringify(intensities) !== JSON.stringify(initIntensities)) {
        dispatch(experimentSlice.setIlluminationIntensities(initIntensities));
      }
      if (JSON.stringify(exposures) !== JSON.stringify(initExposures)) {
        dispatch(experimentSlice.setExposureTimes(initExposures));
      }
      if (JSON.stringify(gains) !== JSON.stringify(initGains)) {
        dispatch(experimentSlice.setGains(initGains));
      }
      if (JSON.stringify(channelEnabledForExperiment) !== JSON.stringify(initChannelEnabled)) {
        dispatch(experimentSlice.setChannelEnabledForExperiment(initChannelEnabled));
      }
      dispatch(experimentSlice.setIllumination(illuSources));
    }
  }, [illuSources]);

  // Update summary when channels change – count only enabled channels
  useEffect(() => {
    const enabledCount = (channelEnabledForExperiment || []).filter(Boolean).length;
    const totalCount = illuSources.length;
    const summary =
      totalCount === 0
        ? "No channels available"
        : enabledCount === 0
        ? `0/${totalCount} channels`
        : enabledCount === 1
        ? `1/${totalCount} channel`
        : `${enabledCount}/${totalCount} channels`;

    dispatch(
      experimentUISlice.setDimensionSummary({
        dimension: DIMENSIONS.CHANNELS,
        summary,
      })
    );
    dispatch(
      experimentUISlice.setDimensionConfigured({
        dimension: DIMENSIONS.CHANNELS,
        configured: enabledCount > 0,
      })
    );
  }, [illuSources, channelEnabledForExperiment, dispatch]);

  // Debounced laser intensity update (copied from IlluminationController)
  const debouncedSetLaserValue = useCallback((laserName, index, val) => {
    // Update Redux state immediately for UI responsiveness
    dispatch(laserSlice.setLaserPower({ laserName, power: val }));
    
    const arr = [...intensities];
    arr[index] = val;
    dispatch(experimentSlice.setIlluminationIntensities(arr));
    
    // Clear existing timeout for this laser
    if (laserTimeoutRefs.current[laserName]) {
      clearTimeout(laserTimeoutRefs.current[laserName]);
    }
    
    // Set new timeout to send to backend after user stops adjusting
    laserTimeoutRefs.current[laserName] = setTimeout(async () => {
      if (connectionSettings.ip && connectionSettings.apiPort) {
        try {
          const api = createAxiosInstance();
          const encodedLaserName = encodeURIComponent(laserName);
          await api.get(`/LaserController/setLaserValue?laserName=${encodedLaserName}&value=${val}`);
          console.log(`${laserName} intensity updated to: ${val}`);
        } catch (error) {
          console.error("Failed to set laser value:", error);
        }
      }
    }, LASER_UPDATE_DEBOUNCE_MS);
  }, [dispatch, connectionSettings, intensities]);

  // Update laser active state (copied from IlluminationController)
  const setLaserActive = useCallback(async (laserName, active) => {
    // Update Redux state immediately
    dispatch(laserSlice.setLaserEnabled({ laserName, enabled: active }));
    
    // Update backend
    if (connectionSettings.ip && connectionSettings.apiPort) {
      try {
        const api = createAxiosInstance();
        const encodedLaserName = encodeURIComponent(laserName);
        await api.get(`/LaserController/setLaserActive?laserName=${encodedLaserName}&active=${active}`);
        console.log(`${laserName} active state updated to: ${active}`);
      } catch (error) {
        console.error("Failed to set laser active state:", error);
      }
    }
  }, [dispatch, connectionSettings]);

  // Handler for intensity change - uses debounced update
  const handleIntensityChange = (index, value) => {
    const laserName = illuSources[index];
    if (laserName) {
      debouncedSetLaserValue(laserName, index, value);
    }
  };

  // Handler for enabled/disabled change (physical laser toggle only)
  const handleEnabledChange = (index, enabled) => {
    const laserName = illuSources[index];
    if (laserName) {
      setLaserActive(laserName, enabled);
    }
  };

  // Handler for toggling channel inclusion in experiment.
  // For synthetic (ring/dpc) channels we ALSO mirror the current max(R,G,B)
  // into illuIntensities[index] when enabling, and zero it when disabling.
  // Rationale: the intensity slider is hidden for synthetic channels so the
  // user can't directly set illuIntensities — without this mirror, every
  // synth channel ships with intensity=0, which collapses passthrough mode
  // on the backend and the active-channel gate silently drops them.  The
  // mirrored value is what the backend uses as `illu_intensity` for the
  // workflow gate; the actual RGB pattern comes from illuminationParams.
  const handleIncludeInExperimentChange = (index, included) => {
    dispatch(experimentSlice.toggleChannelForExperiment(index));
    const channelName = illuSources[index];
    if (!channelName) return;
    const kind = illuSourceKinds[index] || "default";
    if (kind !== "ring" && kind !== "dpc") return;
    // toggleChannelForExperiment inverts the existing flag.  Compute the
    // post-toggle state ourselves rather than reading Redux (the dispatch
    // above is async).
    const willBeEnabled = !(channelEnabledForExperiment[index] === true);
    const params = illuminationParams[channelName] || {};
    const rgbMax = Math.max(
      Number(params.intensityR ?? 0),
      Number(params.intensityG ?? 0),
      Number(params.intensityB ?? 0)
    );
    const newIntensity = willBeEnabled ? (rgbMax > 0 ? rgbMax : 0) : 0;
    if ((intensities[index] ?? 0) !== newIntensity) {
      const arr = [...intensities];
      arr[index] = newIntensity;
      dispatch(experimentSlice.setIlluminationIntensities(arr));
    }
  };

  // Debounced LED-matrix live-preview firing.
  // Whenever the user tweaks a radius / RGB slider for a synthetic channel,
  // push the same pattern to the LED matrix so they can see what their
  // experiment will look like.  The debounce coalesces the rapid Slider
  // onChange stream into ~one HTTP request per pause so the ESP32 serial
  // link doesn't get flooded.
  const ledMatrixPreviewTimeoutRef = useRef(null);
  const LED_MATRIX_PREVIEW_DEBOUNCE_MS = 120;
  const previewLedMatrix = useCallback((kind, params) => {
    if (ledMatrixPreviewTimeoutRef.current) {
      clearTimeout(ledMatrixPreviewTimeoutRef.current);
    }
    ledMatrixPreviewTimeoutRef.current = setTimeout(async () => {
      try {
        if (kind === "ring") {
          await apiLEDMatrixControllerSetRing({
            ringRadius: Number(params.radius ?? 2),
            intensity: Math.max(
              Number(params.intensityR ?? 0),
              Number(params.intensityG ?? 0),
              Number(params.intensityB ?? 0)
            ),
            intensity_r: Number(params.intensityR ?? 0),
            intensity_g: Number(params.intensityG ?? 0),
            intensity_b: Number(params.intensityB ?? 0),
          });
        } else if (kind === "dpc") {
          // Show just the "top" half as a representative preview — the user
          // can tell whether RGB looks right from any single direction, and
          // cycling all four would flash distractingly during slider drags.
          await apiLEDMatrixControllerSetHalves({
            intensity: Math.max(
              Number(params.intensityR ?? 0),
              Number(params.intensityG ?? 255),
              Number(params.intensityB ?? 0)
            ),
            direction: "top",
            intensity_r: Number(params.intensityR ?? 0),
            intensity_g: Number(params.intensityG ?? 255),
            intensity_b: Number(params.intensityB ?? 0),
          });
        }
      } catch (err) {
        // setRing / setHalves errors are already logged inside the helper
        // — swallow here so a flaky preview doesn't break the dialog.
      }
    }, LED_MATRIX_PREVIEW_DEBOUNCE_MS);
  }, []);

  // Cleanup pending LED-matrix preview timeout on unmount.
  useEffect(() => {
    return () => {
      if (ledMatrixPreviewTimeoutRef.current) {
        clearTimeout(ledMatrixPreviewTimeoutRef.current);
      }
    };
  }, []);

  // Handler for kind-specific param changes (radius, intensityR/G/B).
  // Stores the new value under parameterValue.illuminationParams[channelName]
  // AND (for synthetic channels) fires a debounced LED-matrix live preview
  // so the user can see the pattern on the hardware as they tune sliders.
  // When the channel is currently enabled for the experiment, also keeps
  // illuIntensities[i] in sync with max(R,G,B) so the backend's active-gate
  // and passthrough heuristics stay accurate.
  const handleKindParamChange = useCallback(
    (channelName, key, value) => {
      dispatch(
        experimentSlice.setIlluminationParamsForChannel({
          channelName,
          params: { [key]: value },
        })
      );
      // Resolve the channel's kind to pick the right LED-matrix endpoint.
      const idx = illuSources.indexOf(channelName);
      const kind = idx >= 0 ? illuSourceKinds[idx] || "default" : "default";
      if (kind === "ring" || kind === "dpc") {
        // Merge the new value into the current params for an accurate preview
        // (the slice update is async — read-after-write on Redux state would
        // miss this change).
        const currentParams = illuminationParams[channelName] || {};
        const mergedParams = { ...currentParams, [key]: value };
        previewLedMatrix(kind, mergedParams);
        // Mirror max(R,G,B) into illuIntensities when this channel is enabled
        // for the experiment — radius doesn't affect intensity so we only
        // recompute when an RGB component changed.
        if (channelEnabledForExperiment[idx] === true && key !== "radius") {
          const rgbMax = Math.max(
            Number(mergedParams.intensityR ?? 0),
            Number(mergedParams.intensityG ?? 0),
            Number(mergedParams.intensityB ?? 0)
          );
          if ((intensities[idx] ?? 0) !== rgbMax) {
            const arr = [...intensities];
            arr[idx] = rgbMax;
            dispatch(experimentSlice.setIlluminationIntensities(arr));
          }
        }
      }
    },
    [
      dispatch,
      illuSources,
      illuSourceKinds,
      illuminationParams,
      previewLedMatrix,
      channelEnabledForExperiment,
      intensities,
    ]
  );

  // Handler for exposure change
  const handleExposureChange = (index, value) => {
    const arr = [...exposures];
    arr[index] = Number(value);
    dispatch(experimentSlice.setExposureTimes(arr));
    // update backend immediately for real-time feedback
    if ( connectionSettings.ip && connectionSettings.apiPort) {
      const api = createAxiosInstance();
      api.get(
        `/SettingsController/setDetectorExposureTime?exposureTime=${value}`
      ).catch((error) => {
        console.error("Failed to update detector exposure time:", error);
      });
    }
  };

  // Handler for gain change
  const handleGainChange = (index, value) => {
    const arr = [...gains];
    arr[index] = Number(value);
    dispatch(experimentSlice.setGains(arr));
    // update backend immediately for real-time feedback
    if (connectionSettings.ip && connectionSettings.apiPort) {
      const api = createAxiosInstance();
      api.get(
        `/SettingsController/setDetectorGain?gain=${value}`
      ).catch((error) => {
        console.error("Failed to update detector gain:", error);
      });
    }
  };

  // Copy settings from first channel to all
  const handleCopyToAll = () => {
    if (illuSources.length <= 1) return;

    const firstIntensity = intensities[0] ?? 0;
    const firstExposure = exposures[0] ?? 100;
    const firstGain = gains[0] ?? 0;

    const newIntensities = illuSources.map(() => firstIntensity);
    const newExposures = illuSources.map(() => firstExposure);
    const newGains = illuSources.map(() => firstGain);

    dispatch(experimentSlice.setIlluminationIntensities(newIntensities));
    dispatch(experimentSlice.setExposureTimes(newExposures));
    dispatch(experimentSlice.setGains(newGains));
  };

  // Fetch the live exposure + gain from the detector and write them into the
  // experiment parameters for every channel. Useful after tuning exposure
  // interactively in the Live View — you don't want to type the same values
  // back into every channel by hand.
  const handleStoreCurrentSettings = async () => {
    try {
      const hostIP = connectionSettings.ip;
      const hostPort = connectionSettings.apiPort;
      if (!hostIP || !hostPort) return;
      const r = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/SettingsController/getDetectorParameters`
      );
      if (!r.ok) return;
      const data = await r.json();
      const expVal = Number(data?.exposure);
      const gainVal = Number(data?.gain);
      if (Number.isFinite(expVal) && illuSources.length > 0) {
        dispatch(experimentSlice.setExposureTimes(illuSources.map(() => expVal)));
      }
      if (Number.isFinite(gainVal) && illuSources.length > 0) {
        dispatch(experimentSlice.setGains(illuSources.map(() => gainVal)));
      }
    } catch (e) {
      console.warn("Failed to fetch current detector settings:", e);
    }
  };

  // Toggle channel expansion
  const toggleChannelExpand = (index) => {
    setExpandedChannels((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      {/* Header with actions */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2,
        }}
      >
        <Typography variant="body2" color="textSecondary">
          {illuSources.length} {illuSources.length === 1 ? "channel" : "channels"} selected
          <Tooltip
            title={
              "Each channel has two toggles:\n" +
              "⏻ Power – physically turns the laser on/off for live preview.\n" +
              "🔬 Experiment – includes this channel in the automated acquisition.\n" +
              "You can preview with a laser ON but exclude it from the experiment."
            }
            arrow
          >
            <InfoOutlinedIcon sx={{ fontSize: 14, ml: 0.5, mb: -0.3, color: "text.disabled", cursor: "help" }} />
          </Tooltip>
        </Typography>

        <Box sx={{ display: "flex", gap: 1 }}>
          <Tooltip title="Read current exposure & gain from the detector and apply to all channels">
            <span>
              <Button
                size="small"
                variant="outlined"
                color="secondary"
                startIcon={<SettingsIcon />}
                onClick={handleStoreCurrentSettings}
                disabled={illuSources.length === 0}
              >
                Read and Apply current EXP/GAIN settings
              </Button>
            </span>
          </Tooltip>
          <Tooltip title="Copy settings from first channel to all others">
            <span>
              <Button
                size="small"
                variant="outlined"
                startIcon={<ContentCopyIcon />}
                onClick={handleCopyToAll}
                disabled={illuSources.length <= 1}
              >
                Copy settings to all channels
              </Button>
            </span>
          </Tooltip>
        </Box>
      </Box>

      {/* Channel Blocks */}
      {illuSources.length === 0 ? (
        <Box
          sx={{
            textAlign: "center",
            py: 4,
            color: theme.palette.text.secondary,
          }}
        >
          <Typography variant="body2">
            No illumination sources available.
          </Typography>
          <Typography variant="caption">
            Connect to a microscope to see available channels.
          </Typography>
        </Box>
      ) : (
        illuSources.map((source, idx) => {
          // Get laser state from Redux (updated via WebSocket)
          const laserData = lasers[source] || { power: intensities[idx] ?? 0, enabled: false };
          // Per-source kind ("default" | "ring" | "dpc"); fall back if backend
          // omits the parallel array (older versions).
          const channelKind = illuSourceKinds[idx] || "default";
          const channelKindParams = illuminationParams[source] || {};

          return (
            <ChannelBlock
              key={`channel-${source}-${idx}`}
              channelName={source}
              channelIndex={idx}
              intensity={intensities[idx] ?? 0}
              exposure={exposures[idx] ?? 100}
              gain={gains[idx] ?? 0}
              minIntensity={laserMinValues[idx] ?? 0}
              maxIntensity={laserMaxValues[idx] ?? 1023}
              isEnabled={laserData.enabled}
              isIncludedInExperiment={channelEnabledForExperiment[idx] ?? true}
              isExpanded={expandedChannels[idx] ?? idx === 0}
              onToggleExpand={() => toggleChannelExpand(idx)}
              onIntensityChange={(val) => handleIntensityChange(idx, val)}
              onExposureChange={(val) => handleExposureChange(idx, val)}
              onGainChange={(val) => handleGainChange(idx, val)}
              onEnabledChange={(enabled) => handleEnabledChange(idx, enabled)}
              onIncludeInExperimentChange={(included) => handleIncludeInExperimentChange(idx, included)}
              kind={channelKind}
              kindParams={channelKindParams}
              onKindParamChange={(key, value) => handleKindParamChange(source, key, value)}
              ringMaxRadius={maxRingRadius}
            />
          );
        })
      )}

      {/* Advanced Settings */}
      <Accordion
        disableGutters
        sx={{
          mt: 2,
          boxShadow: "none",
          border: `1px solid ${theme.palette.divider}`,
          "&:before": { display: "none" },
        }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <SettingsIcon sx={{ mr: 1, fontSize: 18, color: theme.palette.text.secondary }} />
          <Typography variant="body2">Advanced Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          {/* Keep Illumination On mode */}
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center", mb: 0.5 }}>
              <Typography variant="caption" sx={{ fontWeight: 500 }}>
                Illumination Mode
              </Typography>
              <Tooltip
                title={
                  "Controls when illumination is toggled during acquisition:\n" +
                  "• Auto – keeps illumination on for the entire scan when only one channel is active; toggles per-frame when multiple channels are used.\n" +
                  "• Always On – illumination stays on from start to finish (faster, but may cause bleaching with multiple channels).\n" +
                  "• Per-Frame – illumination is turned on/off around every single frame (safest for multi-channel, slowest)."
                }
                arrow
              >
                <InfoOutlinedIcon sx={{ fontSize: 14, ml: 0.5, color: "text.disabled", cursor: "help" }} />
              </Tooltip>
            </Box>
            <FormControl size="small" fullWidth>
              <Select
                value={parameterValue.keepIlluminationOn || "auto"}
                onChange={(e) => dispatch(experimentSlice.setKeepIlluminationOn(e.target.value))}
              >
                <MenuItem value="auto">Auto (single channel → on, multi → per-frame)</MenuItem>
                <MenuItem value="on">Always On</MenuItem>
                <MenuItem value="off">Per-Frame Toggle</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <Typography variant="caption" color="textSecondary">
            Advanced camera and illumination parameters will be shown here.
            These settings are typically configured once and rarely changed.
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default ChannelsDimension;
