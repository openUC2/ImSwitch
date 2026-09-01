// Kiosk LED-matrix control: big pattern buttons (all / ring / circle /
// halves), one intensity slider, and a master off. Mirrors its choices into
// LEDMatrixSlice so the desktop UI stays in sync.
import { useMemo, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Button,
  Paper,
  Slider,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import LightModeRoundedIcon from "@mui/icons-material/LightModeRounded";
import RadioButtonUncheckedRoundedIcon from "@mui/icons-material/RadioButtonUncheckedRounded";
import CircleRoundedIcon from "@mui/icons-material/CircleRounded";
import ContrastRoundedIcon from "@mui/icons-material/ContrastRounded";
import PowerSettingsNewRoundedIcon from "@mui/icons-material/PowerSettingsNewRounded";
import { enqueueSnackbar } from "notistack";

import MobilePage from "../components/MobilePage";
import SectionLabel from "../components/SectionLabel";
import TouchButton from "../components/TouchButton";
import * as ledMatrixSlice from "../../state/slices/LEDMatrixSlice";
import { getParameterRangeState } from "../../state/slices/ParameterRangeSlice";
import apiLEDMatrixControllerSetAllLED from "../../backendapi/apiLEDMatrixControllerSetAllLED";
import apiLEDMatrixControllerSetRing from "../../backendapi/apiLEDMatrixControllerSetRing";
import apiLEDMatrixControllerSetCircle from "../../backendapi/apiLEDMatrixControllerSetCircle";
import apiLEDMatrixControllerSetHalves from "../../backendapi/apiLEDMatrixControllerSetHalves";

const DIRECTIONS = ["top", "bottom", "left", "right"];

const MobileLedPage = () => {
  const dispatch = useDispatch();
  const ledState = useSelector(ledMatrixSlice.getLEDMatrixState);
  const parameterRange = useSelector(getParameterRangeState);

  const maxRadius = useMemo(
    () => parameterRange.ledMatrixInfo?.maxRingRadius ?? 8,
    [parameterRange.ledMatrixInfo],
  );

  const [intensityDraft, setIntensityDraft] = useState(null);
  const intensity = intensityDraft ?? ledState.intensity ?? 255;

  const reportError = () =>
    enqueueSnackbar("LED matrix command failed", { variant: "error" });

  // Send the given pattern to the hardware and mirror it into Redux.
  const applyPattern = (mode, overrides = {}) => {
    const next = {
      intensity,
      direction: ledState.direction,
      ringRadius: ledState.ringRadius,
      circleRadius: ledState.circleRadius,
      ...overrides,
    };

    let request;
    if (mode === "off") {
      request = apiLEDMatrixControllerSetAllLED({ state: 0, intensity: 0 });
    } else if (mode === "all") {
      request = apiLEDMatrixControllerSetAllLED({ state: 1, intensity: next.intensity });
    } else if (mode === "ring") {
      request = apiLEDMatrixControllerSetRing({
        ringRadius: next.ringRadius,
        intensity: next.intensity,
      });
    } else if (mode === "circle") {
      request = apiLEDMatrixControllerSetCircle({
        circleRadius: next.circleRadius,
        intensity: next.intensity,
      });
    } else if (mode === "halves") {
      request = apiLEDMatrixControllerSetHalves({
        intensity: next.intensity,
        direction: next.direction,
      });
    } else {
      return;
    }
    request.catch(reportError);

    dispatch(ledMatrixSlice.setIsOn(mode !== "off"));
    if (mode !== "off") dispatch(ledMatrixSlice.setMode(mode));
    dispatch(ledMatrixSlice.setIntensity(next.intensity));
    dispatch(ledMatrixSlice.setDirection(next.direction));
    dispatch(ledMatrixSlice.setRingRadius(next.ringRadius));
    dispatch(ledMatrixSlice.setCircleRadius(next.circleRadius));
  };

  const activeMode = ledState.isOn ? ledState.mode : "off";

  const patternButton = (mode, icon, label) => (
    <TouchButton
      icon={icon}
      label={label}
      variant={activeMode === mode ? "contained" : "outlined"}
      onClick={() => applyPattern(mode)}
      sx={{ flex: 1, minWidth: 120 }}
    />
  );

  return (
    <MobilePage title="LED Matrix" subtitle="Pick a pattern, set the brightness">
      <Box sx={{ display: "flex", gap: 2.5, flexWrap: "wrap" }}>
        <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3, flex: 2, minWidth: 320 }}>
          <SectionLabel>Pattern</SectionLabel>
          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 2.5 }}>
            {patternButton("all", <LightModeRoundedIcon />, "All on")}
            {patternButton("ring", <RadioButtonUncheckedRoundedIcon />, "Ring")}
            {patternButton("circle", <CircleRoundedIcon />, "Circle")}
            {patternButton("halves", <ContrastRoundedIcon />, "Half")}
          </Box>

          {activeMode === "halves" && (
            <>
              <SectionLabel>Direction</SectionLabel>
              <ToggleButtonGroup
                exclusive
                fullWidth
                value={ledState.direction}
                onChange={(e, v) => v !== null && applyPattern("halves", { direction: v })}
                sx={{ mb: 2.5 }}
              >
                {DIRECTIONS.map((d) => (
                  <ToggleButton key={d} value={d}>
                    {d}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </>
          )}

          {activeMode === "ring" && (
            <>
              <SectionLabel>Ring radius</SectionLabel>
              <Slider
                value={ledState.ringRadius}
                min={1}
                max={maxRadius}
                step={1}
                marks
                valueLabelDisplay="auto"
                onChangeCommitted={(e, v) => applyPattern("ring", { ringRadius: v })}
                sx={{ mb: 1 }}
              />
            </>
          )}

          {activeMode === "circle" && (
            <>
              <SectionLabel>Circle radius</SectionLabel>
              <Slider
                value={ledState.circleRadius}
                min={1}
                max={maxRadius}
                step={1}
                marks
                valueLabelDisplay="auto"
                onChangeCommitted={(e, v) => applyPattern("circle", { circleRadius: v })}
                sx={{ mb: 1 }}
              />
            </>
          )}
        </Paper>

        <Box sx={{ flex: 1, minWidth: 260, display: "flex", flexDirection: "column", gap: 2.5 }}>
          <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3 }}>
            <SectionLabel>Brightness</SectionLabel>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Slider
                value={intensity}
                min={0}
                max={255}
                step={1}
                onChange={(e, v) => setIntensityDraft(v)}
                onChangeCommitted={(e, v) => {
                  setIntensityDraft(null);
                  // Re-apply the active pattern with the new brightness.
                  applyPattern(activeMode === "off" ? "all" : activeMode, {
                    intensity: v,
                  });
                }}
              />
              <Typography
                variant="h6"
                sx={{ width: 52, textAlign: "right", fontVariantNumeric: "tabular-nums" }}
              >
                {intensity}
              </Typography>
            </Box>
          </Paper>

          <Button
            size="large"
            color="error"
            variant={activeMode === "off" ? "contained" : "outlined"}
            startIcon={<PowerSettingsNewRoundedIcon />}
            onClick={() => applyPattern("off")}
            sx={{ minHeight: 76 }}
          >
            All off
          </Button>
        </Box>
      </Box>
    </MobilePage>
  );
};

export default MobileLedPage;
