// src/components/touch/TouchStagePanel.jsx
// Thumb-sized stage jog pad for the compact / touch Live View: an XY D-pad
// with STOP in the centre, a Z column, live position readout and step sizes.
//
// XY: tap = one step, hold = constant-velocity travel until release.
// Z:  tap = one step, hold = repeated single steps. Never a free-running Z
//     move — an unstopped focus move drives the objective into the sample.
import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Box, Button, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import KeyboardArrowUpRoundedIcon from "@mui/icons-material/KeyboardArrowUpRounded";
import KeyboardArrowDownRoundedIcon from "@mui/icons-material/KeyboardArrowDownRounded";
import KeyboardArrowLeftRoundedIcon from "@mui/icons-material/KeyboardArrowLeftRounded";
import KeyboardArrowRightRoundedIcon from "@mui/icons-material/KeyboardArrowRightRounded";
import StopRoundedIcon from "@mui/icons-material/StopRounded";
import * as positionSlice from "../../state/slices/PositionSlice.js";
import useStageJog from "../../hooks/useStageJog";
import apiPositionerControllerStopAllAxes from "../../backendapi/apiPositionerControllerStopAllAxes";

const XY_STEPS = [10, 100, 1000];
const Z_STEPS = [1, 10, 100];
const STORAGE_KEY = "imswitch-touch-stage-steps";
// Same constant-velocity speed as the on-stream pad (PositionControllerComponent).
const CONTINUOUS_SPEED = 5000;

const readSteps = () => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return {
      xy: XY_STEPS.includes(saved.xy) ? saved.xy : 100,
      z: Z_STEPS.includes(saved.z) ? saved.z : 10,
    };
  } catch {
    return { xy: 100, z: 10 };
  }
};

const fmt = (v) =>
  Number.isFinite(Number(v)) ? Number(v).toFixed(1) : "—";

function StepSelector({ label, values, value, onChange }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
      <Typography
        variant="caption"
        sx={{ width: 52, flexShrink: 0, color: "text.secondary", fontWeight: 600 }}
      >
        {label}
      </Typography>
      <ToggleButtonGroup
        exclusive
        size="small"
        value={value}
        onChange={(_, v) => v !== null && onChange(v)}
        sx={{
          flex: 1,
          "& .MuiToggleButton-root": {
            flex: 1,
            py: 0.5,
            textTransform: "none",
            fontVariantNumeric: "tabular-nums",
          },
        }}
      >
        {values.map((v) => (
          <ToggleButton key={v} value={v}>
            {v}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </Box>
  );
}

export default function TouchStagePanel({ buttonSize = 60, showA = false }) {
  const position = useSelector(positionSlice.getPositionState);
  const [steps, setSteps] = useState(readSteps);
  const { jogHandlers } = useStageJog({ holdDelayMs: 700 });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(steps));
    } catch {
      // storage unavailable (private mode) — the steps just are not remembered
    }
  }, [steps]);

  const stopAll = () =>
    apiPositionerControllerStopAllAxes().catch((e) =>
      console.error("Stop all axes failed:", e),
    );

  const padButton = (axis, sign, icon, label) => (
    <Button
      variant="contained"
      aria-label={label}
      {...jogHandlers(
        axis,
        axis === "Z"
          ? { dist: sign * steps.z, mode: "repeat" }
          : { dist: sign * steps.xy, speed: sign * CONTINUOUS_SPEED, mode: "continuous" },
      )}
      sx={{
        minWidth: 0,
        p: 0,
        width: buttonSize,
        height: buttonSize,
        borderRadius: 2,
        touchAction: "none",
        userSelect: "none",
        "& svg": { fontSize: buttonSize * 0.6 },
      }}
    >
      {icon}
    </Button>
  );

  const axes = showA ? ["x", "y", "z", "a"] : ["x", "y", "z"];

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.25 }}>
      {/* live position */}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: `repeat(${axes.length}, 1fr)`,
          gap: 0.5,
          p: 0.75,
          borderRadius: 1,
          bgcolor: "action.hover",
        }}
      >
        {axes.map((a) => (
          <Box key={a} sx={{ textAlign: "center", minWidth: 0 }}>
            <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 700 }}>
              {a.toUpperCase()}
            </Typography>
            <Typography
              variant="body2"
              sx={{ fontFamily: "monospace", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis" }}
            >
              {fmt(position[a])}
            </Typography>
          </Box>
        ))}
      </Box>

      {/* XY D-pad + Z column */}
      <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 2 }}>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: `repeat(3, ${buttonSize}px)`,
            gridTemplateRows: `repeat(3, ${buttonSize}px)`,
            gap: 0.75,
          }}
        >
          <span />
          {padButton("Y", -1, <KeyboardArrowUpRoundedIcon />, "Move Y up")}
          <span />
          {padButton("X", -1, <KeyboardArrowLeftRoundedIcon />, "Move X left")}
          <Button
            variant="contained"
            color="error"
            aria-label="Stop all axes"
            onClick={stopAll}
            sx={{ minWidth: 0, p: 0, width: buttonSize, height: buttonSize, borderRadius: 2 }}
          >
            <StopRoundedIcon sx={{ fontSize: buttonSize * 0.55 }} />
          </Button>
          {padButton("X", 1, <KeyboardArrowRightRoundedIcon />, "Move X right")}
          <span />
          {padButton("Y", 1, <KeyboardArrowDownRoundedIcon />, "Move Y down")}
          <span />
        </Box>

        <Box
          sx={{
            display: "grid",
            gridTemplateRows: `repeat(3, ${buttonSize}px)`,
            gap: 0.75,
            justifyItems: "center",
            alignItems: "center",
          }}
        >
          {padButton("Z", 1, <KeyboardArrowUpRoundedIcon />, "Move Z up")}
          <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "text.secondary" }}>
            Z
          </Typography>
          {padButton("Z", -1, <KeyboardArrowDownRoundedIcon />, "Move Z down")}
        </Box>
      </Box>

      <StepSelector
        label="XY µm"
        values={XY_STEPS}
        value={steps.xy}
        onChange={(xy) => setSteps((s) => ({ ...s, xy }))}
      />
      <StepSelector
        label="Z µm"
        values={Z_STEPS}
        value={steps.z}
        onChange={(z) => setSteps((s) => ({ ...s, z }))}
      />
      <Typography variant="caption" sx={{ color: "text.secondary", lineHeight: 1.3 }}>
        Tap to step. Hold an XY arrow to drive until you let go; holding Z
        repeats single steps.
      </Typography>
    </Box>
  );
}
