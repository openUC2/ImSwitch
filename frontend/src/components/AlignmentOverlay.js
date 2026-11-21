// Alignment aid over the live view: a heat map of the field's sharpness
// (utils/sharpnessMap.js), its left/right and top/bottom balance, and the three
// screw sets of a FRAME to work through. Unsigned: it shows where the sharp
// part is, so turn a screw and watch the map even out.
import React, { useState } from "react";
import { Box, Button, Typography } from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import {
  getLiveStreamState,
  setShowSharpnessMap,
} from "../state/slices/LiveStreamSlice";
import { sharpnessBalance } from "../utils/sharpnessMap";

const STEPS = [
  {
    title: "1 · Sample holder",
    text: "Put the calibration grid (or any flat, structured sample) in the holder and focus on the centre. Turn the holder's three screws until the map is even.",
  },
  {
    title: "2 · Objective",
    text: "Refocus on the centre. Turn the objective's three screws until the map is even.",
  },
  {
    title: "3 · Objective mount",
    text: "Refocus on the centre. Turn the mount's three screws; finish when the balance stays within ±0.1.",
  },
];

// 0 → dark blue, 1 → yellow
const heat = (t) => `hsl(${240 - 180 * t}, 85%, ${30 + 30 * t}%)`;

const sideHint = (v, a, b) => {
  if (Math.abs(v) < 0.1) return "even";
  return `${v > 0 ? b : a} sharper (${v.toFixed(2)})`;
};

export default function AlignmentOverlay() {
  const dispatch = useDispatch();
  const { showSharpnessMap, sharpnessGrid, imageFormat } = useSelector(getLiveStreamState);
  const [step, setStep] = useState(0);
  if (!showSharpnessMap) return null;

  const { cols = 8, rows = 6, values = [] } = sharpnessGrid || {};
  const max = Math.max(...values, 1e-9);
  const { lr, tb } = sharpnessBalance(values, cols, rows);
  const stop = (e) => e.stopPropagation(); // keep clicks off the image below

  return (
    <Box
      onPointerDown={stop}
      onMouseDown={stop}
      onClick={stop}
      onWheel={stop}
      sx={{
        position: "absolute",
        top: 10,
        right: 10,
        width: 260,
        bgcolor: "rgba(0,0,0,0.75)",
        color: "#fff",
        zIndex: 5,
        p: 1,
        borderRadius: 1,
      }}
    >
      <Typography variant="caption" sx={{ fontWeight: 600 }}>
        Alignment · sharpness across the field
      </Typography>
      {imageFormat !== "jpeg" ? (
        <Typography variant="caption" component="div" sx={{ color: "#fc9", my: 1 }}>
          Needs the JPEG stream (Settings → Live stream).
        </Typography>
      ) : !values.length ? (
        <Typography variant="caption" component="div" sx={{ color: "#bbb", my: 1 }}>
          Waiting for frames…
        </Typography>
      ) : (
        <>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: `repeat(${cols}, 1fr)`,
              gap: "1px",
              my: 0.5,
              aspectRatio: "4 / 3",
            }}
          >
            {values.map((v, i) => (
              <Box key={i} title={v.toFixed(3)} sx={{ bgcolor: heat(v / max) }} />
            ))}
          </Box>
          <Typography variant="caption" component="div">
            Left/right: {sideHint(lr, "left", "right")}
          </Typography>
          <Typography variant="caption" component="div">
            Top/bottom: {sideHint(tb, "top", "bottom")}
          </Typography>
        </>
      )}
      <Typography variant="caption" component="div" sx={{ fontWeight: 600, mt: 1 }}>
        {STEPS[step].title}
      </Typography>
      <Typography variant="caption" component="div" sx={{ color: "#ddd" }}>
        {STEPS[step].text}
      </Typography>
      <Box sx={{ display: "flex", mt: 0.5 }}>
        <Button size="small" disabled={step === 0} onClick={() => setStep(step - 1)} sx={{ color: "#fff" }}>
          Back
        </Button>
        {step < STEPS.length - 1 ? (
          <Button size="small" onClick={() => setStep(step + 1)} sx={{ color: "#fff", ml: "auto" }}>
            Next
          </Button>
        ) : (
          <Button
            size="small"
            onClick={() => dispatch(setShowSharpnessMap(false))}
            sx={{ color: "#fff", ml: "auto" }}
          >
            Done
          </Button>
        )}
      </Box>
    </Box>
  );
}
