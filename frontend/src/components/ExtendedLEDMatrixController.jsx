// ./components/ExtendedLEDMatrixController.jsx
import {
  Alert,
  Box,
  Button,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Slider,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import { useState } from "react";
import { useDispatch, useSelector } from "react-redux";

import {
  getLEDMatrixState,
  setCircleRadius,
  setDirection,
  setIntensity,
  setIsOn,
  setMode,
  setRingRadius,
} from "../state/slices/LEDMatrixSlice";

import apiLEDMatrixControllerSetAllLED from "../backendapi/apiLEDMatrixControllerSetAllLED";
import apiLEDMatrixControllerSetCircle from "../backendapi/apiLEDMatrixControllerSetCircle";
import apiLEDMatrixControllerSetHalves from "../backendapi/apiLEDMatrixControllerSetHalves";
import apiLEDMatrixControllerSetRing from "../backendapi/apiLEDMatrixControllerSetRing";

// Full-scale per-channel LED value and the overheating guard threshold.
// Above 70 % of full power the matrix firmware auto-switches off after 60 s
// to protect the LEDs, so we surface a warning as soon as any channel crosses
// this line.
const LED_FULL_SCALE = 255;
const OVERHEAT_FRACTION = 0.7;
const OVERHEAT_THRESHOLD = Math.round(LED_FULL_SCALE * OVERHEAT_FRACTION); // 179

/**
 * Reusable RGB illumination controls: a master "white" slider that drives all
 * three channels together, plus individual R / G / B sliders — mirroring the
 * LED-matrix Ring / DPC controls in the Experiment Designer. Also renders the
 * overheating warning when any channel exceeds 70 % of full power.
 */
const RgbIntensityControls = ({ rgb, onChange }) => {
  const { r, g, b } = rgb;
  const maxChannel = Math.max(r, g, b);
  const overheating = maxChannel > OVERHEAT_THRESHOLD;
  const percent = Math.round((maxChannel / LED_FULL_SCALE) * 100);

  const channels = [
    { key: "r", label: "R", color: "#e53935" },
    { key: "g", label: "G", color: "#43a047" },
    { key: "b", label: "B", color: "#1e88e5" },
  ];

  return (
    <Box sx={{ mb: 1 }}>
      {/* Master white / brightness slider — moves R, G and B together */}
      <Box sx={{ display: "flex", alignItems: "center", mb: 0.5 }}>
        <Typography variant="caption" sx={{ fontWeight: 500 }}>
          Brightness (white) — {percent}%
        </Typography>
        <Tooltip
          title="Drives all three channels together for neutral white illumination. Use the individual R/G/B sliders below to tint the illumination."
          arrow
        >
          <InfoOutlinedIcon
            sx={{ fontSize: 14, ml: 0.5, color: "text.disabled", cursor: "help" }}
          />
        </Tooltip>
      </Box>
      <Slider
        value={maxChannel}
        min={0}
        max={LED_FULL_SCALE}
        step={1}
        onChange={(e, val) => onChange({ r: val, g: val, b: val })}
        sx={{ mb: 1.5 }}
      />

      {/* Per-channel RGB sliders */}
      <Typography variant="caption" sx={{ fontWeight: 500, display: "block", mb: 0.5 }}>
        RGB Intensity (0–{LED_FULL_SCALE})
      </Typography>
      {channels.map(({ key, label, color }) => (
        <Box
          key={key}
          sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 0.5 }}
        >
          <Typography sx={{ width: 18, fontWeight: 700, color }}>{label}</Typography>
          <Slider
            value={rgb[key]}
            min={0}
            max={LED_FULL_SCALE}
            step={1}
            onChange={(e, val) => onChange({ ...rgb, [key]: val })}
            sx={{ flex: 1, color }}
          />
          <Typography
            variant="body2"
            sx={{ minWidth: "44px", textAlign: "right", fontWeight: 500 }}
          >
            {rgb[key]}
          </Typography>
        </Box>
      ))}

      {overheating && (
        <Alert severity="warning" sx={{ mt: 1.5 }}>
          Above {Math.round(OVERHEAT_FRACTION * 100)}% of full power ({OVERHEAT_THRESHOLD}
          /{LED_FULL_SCALE}) the LED matrix automatically switches off after 60&nbsp;s to
          prevent overheating. Lower the intensity for continuous illumination.
        </Alert>
      )}
    </Box>
  );
};

const ExtendedLEDMatrixController = () => {
  const dispatch = useDispatch();
  const LEDMatrixState = useSelector(getLEDMatrixState);

  const [activeTab, setActiveTab] = useState(0);
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Shared RGB illumination colour (the matrix emits one physical colour at a
  // time, so a single triplet is reused across every pattern). Seeded from the
  // legacy scalar intensity so existing behaviour (neutral white) is preserved.
  const initialLevel = LEDMatrixState.intensity ?? LED_FULL_SCALE;
  const [rgb, setRgb] = useState({
    r: initialLevel,
    g: initialLevel,
    b: initialLevel,
  });

  // Pattern-specific parameters
  const [halvesDirection, setHalvesDirection] = useState(LEDMatrixState.direction);
  const [ringRadius, setLocalRingRadius] = useState(LEDMatrixState.ringRadius);
  const [circleRadius, setLocalCircleRadius] = useState(LEDMatrixState.circleRadius);
  const [allState, setAllState] = useState(LEDMatrixState.isOn ? 1 : 0);

  // Effective scalar intensity (kept in sync with Redux for compatibility with
  // consumers that still read the single-value intensity).
  const scalarIntensity = Math.max(rgb.r, rgb.g, rgb.b);

  const handleRgbChange = (next) => {
    setRgb(next);
    dispatch(setIntensity(Math.max(next.r, next.g, next.b)));
  };

  // Common RGB payload forwarded to every pattern endpoint.
  const rgbPayload = () => ({
    intensity: scalarIntensity,
    intensity_r: rgb.r,
    intensity_g: rgb.g,
    intensity_b: rgb.b,
  });

  // Button handlers
  const handleSetHalves = () => {
    dispatch(setMode("halves"));
    dispatch(setDirection(halvesDirection));

    apiLEDMatrixControllerSetHalves({ ...rgbPayload(), direction: halvesDirection })
      .then((data) => console.log("Halves set", data))
      .catch((error) => console.error("Error setting halves:", error));
  };

  const handleSetRing = () => {
    dispatch(setMode("ring"));
    dispatch(setRingRadius(ringRadius));

    apiLEDMatrixControllerSetRing({ ...rgbPayload(), ringRadius })
      .then((data) => console.log("Ring set", data))
      .catch((error) => console.error("Error setting ring:", error));
  };

  const handleSetCircle = () => {
    dispatch(setMode("circle"));
    dispatch(setCircleRadius(circleRadius));

    apiLEDMatrixControllerSetCircle({ ...rgbPayload(), circleRadius })
      .then((data) => console.log("Circle set", data))
      .catch((error) => console.error("Error setting circle:", error));
  };

  const handleSetAll = () => {
    dispatch(setMode("all"));
    dispatch(setIsOn(allState === 1));

    apiLEDMatrixControllerSetAllLED({
      ...(allState === 1 ? rgbPayload() : { intensity: 0, intensity_r: 0, intensity_g: 0, intensity_b: 0 }),
      state: allState,
      getReturn: true,
    })
      .then((data) => console.log("All LED set", data))
      .catch((error) => console.error("Error setting all LED:", error));
  };

  return (
    <Paper style={{ padding: "20px", marginTop: "20px" }}>
      <Typography variant="h6">LED Matrix Controller</Typography>
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        style={{ marginBottom: "20px" }}
      >
        <Tab label="Halves" />
        <Tab label="Ring" />
        <Tab label="Circle" />
        <Tab label="All" />
      </Tabs>

      {activeTab === 0 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <img
              src={`${process.env.PUBLIC_URL}/assets/illumination_halves.png`}
              alt="Halves"
              style={{ maxHeight: "180px" }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Direction</InputLabel>
              <Select
                value={halvesDirection}
                label="Direction"
                onChange={(e) => setHalvesDirection(e.target.value)}
              >
                <MenuItem value="top">Top</MenuItem>
                <MenuItem value="bottom">Bottom</MenuItem>
                <MenuItem value="left">Left</MenuItem>
                <MenuItem value="right">Right</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12}>
            <RgbIntensityControls rgb={rgb} onChange={handleRgbChange} />
          </Grid>
          <Grid item xs={12}>
            <Button variant="contained" onClick={handleSetHalves} fullWidth>
              Set Halves
            </Button>
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <img
              src={`${process.env.PUBLIC_URL}/assets/illumination_ring.png`}
              alt="Ring"
              style={{ maxHeight: "180px" }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="caption" sx={{ fontWeight: 500 }}>
              Ring Radius
            </Typography>
            <Slider
              value={ringRadius}
              min={0}
              max={8}
              step={1}
              marks
              valueLabelDisplay="auto"
              onChange={(e, val) => setLocalRingRadius(val)}
            />
          </Grid>
          <Grid item xs={12}>
            <RgbIntensityControls rgb={rgb} onChange={handleRgbChange} />
          </Grid>
          <Grid item xs={12}>
            <Button variant="contained" onClick={handleSetRing} fullWidth>
              Set Ring
            </Button>
          </Grid>
        </Grid>
      )}

      {activeTab === 2 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <img
              src={`${process.env.PUBLIC_URL}/assets/illumination_circle.png`}
              alt="Circle"
              style={{ maxHeight: "180px" }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="caption" sx={{ fontWeight: 500 }}>
              Circle Radius
            </Typography>
            <Slider
              value={circleRadius}
              min={0}
              max={8}
              step={1}
              marks
              valueLabelDisplay="auto"
              onChange={(e, val) => setLocalCircleRadius(val)}
            />
          </Grid>
          <Grid item xs={12}>
            <RgbIntensityControls rgb={rgb} onChange={handleRgbChange} />
          </Grid>
          <Grid item xs={12}>
            <Button variant="contained" onClick={handleSetCircle} fullWidth>
              Set Circle
            </Button>
          </Grid>
        </Grid>
      )}

      {activeTab === 3 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <img
              src={`${process.env.PUBLIC_URL}/assets/illumination_circle.png`}
              alt="All"
              style={{ maxHeight: "180px" }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>State</InputLabel>
              <Select
                value={allState}
                label="State"
                onChange={(e) => setAllState(e.target.value)}
              >
                <MenuItem value={1}>On</MenuItem>
                <MenuItem value={0}>Off</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          {allState === 1 && (
            <Grid item xs={12}>
              <RgbIntensityControls rgb={rgb} onChange={handleRgbChange} />
            </Grid>
          )}
          <Grid item xs={12}>
            <Button variant="contained" onClick={handleSetAll} fullWidth>
              Set All
            </Button>
          </Grid>
        </Grid>
      )}
    </Paper>
  );
};

export default ExtendedLEDMatrixController;
