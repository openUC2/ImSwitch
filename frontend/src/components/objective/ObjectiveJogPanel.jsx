// src/components/objective/ObjectiveJogPanel.jsx
// Step the turret (A) and the focus (Z) by a chosen distance — replaces the
// two rows of six fixed "←← (100)" buttons. One step per tap, no hold-to-run:
// Z drives the objective towards the sample.
import React, { useState } from "react";
import { Box, Button, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import RemoveRoundedIcon from "@mui/icons-material/RemoveRounded";
import AddRoundedIcon from "@mui/icons-material/AddRounded";
import { fmt } from "./objectiveSlots";

const A_STEPS = [10, 100, 1000];
const Z_STEPS = [1, 10, 100, 1000];

function JogRow({ label, hint, value, steps, step, onStep, onJog, minusLabel, plusLabel }) {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
      <Box sx={{ display: "flex", alignItems: "baseline", gap: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          {label}
        </Typography>
        <Typography variant="caption" sx={{ color: "text.secondary", flex: 1 }}>
          {hint}
        </Typography>
        <Typography variant="body2" sx={{ fontFamily: "monospace", fontWeight: 600 }}>
          {fmt(value)} µm
        </Typography>
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Button
          variant="outlined"
          aria-label={minusLabel}
          onClick={() => onJog(-step)}
          sx={{ minWidth: 52, height: 44, flexShrink: 0 }}
        >
          <RemoveRoundedIcon />
        </Button>
        <ToggleButtonGroup
          exclusive
          size="small"
          value={step}
          onChange={(_, v) => v !== null && onStep(v)}
          sx={{ flex: 1, "& .MuiToggleButton-root": { flex: 1, height: 44, fontVariantNumeric: "tabular-nums" } }}
        >
          {steps.map((s) => (
            <ToggleButton key={s} value={s}>
              {s}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <Button
          variant="outlined"
          aria-label={plusLabel}
          onClick={() => onJog(step)}
          sx={{ minWidth: 52, height: 44, flexShrink: 0 }}
        >
          <AddRoundedIcon />
        </Button>
      </Box>
    </Box>
  );
}

export default function ObjectiveJogPanel({ position, hasMotor, onJog }) {
  const [aStep, setAStep] = useState(100);
  const [zStep, setZStep] = useState(10);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
      {hasMotor && (
        <JogRow
          label="Turret (A)"
          hint="slides the revolver"
          value={position.a}
          steps={A_STEPS}
          step={aStep}
          onStep={setAStep}
          onJog={(d) => onJog("A", d)}
          minusLabel={`Move turret by -${aStep} µm`}
          plusLabel={`Move turret by +${aStep} µm`}
        />
      )}
      <JogRow
        label="Focus (Z)"
        hint="+ moves towards the sample"
        value={position.z}
        steps={Z_STEPS}
        step={zStep}
        onStep={setZStep}
        onJog={(d) => onJog("Z", d)}
        minusLabel={`Move focus by -${zStep} µm`}
        plusLabel={`Move focus by +${zStep} µm`}
      />
    </Box>
  );
}
