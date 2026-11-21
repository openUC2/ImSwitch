import React, { useEffect, useState } from "react";
import { Box, Button, Slider, TextField, Typography } from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import {
  getLiveStreamState,
  setMinVal,
  setMaxVal,
} from "../state/slices/LiveStreamSlice";
import { getDetectorParameters } from "../state/slices/DetectorParametersSlice";

// Highest value a frame can hold: 255 for JPEG; for raw frames 2^bitDepth - 1 as
// the backend reports it (getDetectorParameters.bitDepth). Cameras that do not
// report one get the smallest 2^n - 1 that holds the brightest histogram bin.
export function rangeMaxFor(isJpeg, histX = [], histY = [], bitDepth = null) {
  if (isJpeg) return 255;
  if (bitDepth >= 8 && bitDepth <= 16) return 2 ** bitDepth - 1;
  let top = 0;
  for (let i = histY.length - 1; i >= 0; i--) {
    if (histY[i] > 0) {
      top = Number(histX[i]) || 0;
      break;
    }
  }
  if (top <= 0) return 65535;
  return Math.min(65535, Math.max(255, 2 ** Math.ceil(Math.log2(top + 1)) - 1));
}

// Auto range: the values below which lo and hi of all pixels lie.
export function autoRange(histX = [], histY = [], lo = 0.001, hi = 0.999) {
  const total = histY.reduce((a, b) => a + b, 0);
  if (!total) return null;
  let acc = 0;
  let min = null;
  let max = null;
  for (let i = 0; i < histY.length; i++) {
    acc += histY[i];
    if (min === null && acc >= lo * total) min = Number(histX[i]);
    if (max === null && acc >= hi * total) max = Number(histX[i]);
  }
  return min < max ? [min, max] : null;
}

/**
 * Display range (min/max) of the live image: one range slider, two number
 * fields and Auto from the histogram. Changes only what is shown, not what the
 * camera records. onDark: white text for the histogram overlay on the image.
 */
export default function DisplayRangeControl({ onDark = false }) {
  const dispatch = useDispatch();
  const {
    minVal = 0,
    maxVal = 255,
    imageFormat,
    histogramX = [],
    histogramY = [],
  } = useSelector(getLiveStreamState);
  const { bitDepth } = useSelector(getDetectorParameters);
  const isJpeg = imageFormat === "jpeg";
  const top = rangeMaxFor(isJpeg, histogramX, histogramY, bitDepth);
  const lo = Math.min(minVal ?? 0, top);
  const hi = Math.min(maxVal ?? top, top);

  const [draft, setDraft] = useState({ min: String(lo), max: String(hi) });
  useEffect(() => setDraft({ min: String(lo), max: String(hi) }), [lo, hi]);

  const commit = (key) => {
    const v = Math.round(Number(draft[key]));
    if (!Number.isFinite(v)) return setDraft({ min: String(lo), max: String(hi) });
    if (key === "min") dispatch(setMinVal(Math.max(0, Math.min(v, hi - 1))));
    else dispatch(setMaxVal(Math.min(top, Math.max(v, lo + 1))));
  };
  const auto = () => {
    const r = autoRange(histogramX, histogramY);
    dispatch(setMinVal(r ? r[0] : 0));
    dispatch(setMaxVal(r ? r[1] : top));
  };

  const color = onDark ? "#fff" : undefined;
  const field = (key, label) => (
    <TextField
      size="small"
      label={label}
      value={draft[key]}
      onChange={(e) => setDraft({ ...draft, [key]: e.target.value })}
      onBlur={() => commit(key)}
      onKeyDown={(e) => e.key === "Enter" && commit(key)}
      inputProps={{ inputMode: "numeric", style: { padding: "2px 6px", color } }}
      InputLabelProps={{ shrink: true, style: { color } }}
      sx={{
        width: 76,
        "& fieldset": onDark ? { borderColor: "rgba(255,255,255,0.4)" } : {},
      }}
    />
  );

  // Keep clicks and drags off the image below (click-to-move, pan, zoom).
  const stop = (e) => e.stopPropagation();
  return (
    <Box
      onPointerDown={stop}
      onMouseDown={stop}
      onClick={stop}
      onWheel={stop}
      onDoubleClick={stop}
    >
      <Slider
        value={[lo, hi]}
        min={0}
        max={top}
        step={1}
        disableSwap
        size="small"
        valueLabelDisplay="auto"
        aria-label="Display range"
        onChange={(_, [a, b]) => {
          dispatch(setMinVal(a));
          dispatch(setMaxVal(b));
        }}
        sx={{ py: 1, color: onDark ? "#9fe" : undefined }}
      />
      <Box sx={{ display: "flex", gap: 1, alignItems: "center", mt: 0.5 }}>
        {field("min", "Min")}
        {field("max", "Max")}
        <Typography variant="caption" sx={{ color: onDark ? "#bbb" : "text.secondary" }}>
          of {top}
        </Typography>
        <Button size="small" onClick={auto} sx={{ ml: "auto", minWidth: 0, color }}>
          Auto
        </Button>
      </Box>
    </Box>
  );
}
