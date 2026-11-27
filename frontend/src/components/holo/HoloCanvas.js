// src/components/holo/HoloCanvas.js
// One float image on a canvas, scaled to its box with crisp pixels. Clicks are
// reported in image pixels; `overlay` is SVG drawn in image-pixel coordinates.
import React, { useEffect, useRef } from "react";
import { Box, IconButton, Stack, Tooltip, Typography } from "@mui/material";
import { Download as DownloadIcon } from "@mui/icons-material";

import { toRGBA } from "./holoMath";
import { useT } from "../../i18n";

const HoloCanvas = ({
  title,
  values,
  w,
  h,
  colormap = "gray",
  min,
  max,
  gamma,
  onPick,
  overlay,
  downloadName,
  caption,
  maxHeight = 360,
}) => {
  const t = useT();
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values || !w || !h) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return; // jsdom
    ctx.putImageData(new ImageData(toRGBA(values, { min, max, colormap, gamma }), w, h), 0, 0);
  }, [values, w, h, colormap, min, max, gamma]);

  const handleClick = (e) => {
    if (!onPick) return;
    const r = e.currentTarget.getBoundingClientRect();
    onPick(((e.clientX - r.left) / r.width) * w, ((e.clientY - r.top) / r.height) * h);
  };

  const download = () => {
    const a = document.createElement("a");
    a.href = ref.current.toDataURL("image/png");
    a.download = `${downloadName}.png`;
    a.click();
  };

  return (
    <Box sx={{ minWidth: 0 }}>
      <Stack direction="row" alignItems="center" sx={{ minHeight: 28 }}>
        <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
          {title}
        </Typography>
        {downloadName && values && (
          <Tooltip title={t("Save as PNG")}>
            <IconButton size="small" onClick={download}>
              <DownloadIcon fontSize="inherit" />
            </IconButton>
          </Tooltip>
        )}
      </Stack>
      <Box
        onClick={handleClick}
        sx={{
          position: "relative",
          lineHeight: 0,
          bgcolor: "#000",
          cursor: onPick ? "crosshair" : "default",
          aspectRatio: w && h ? `${w} / ${h}` : "1 / 1",
          maxHeight,
          mx: "auto",
          maxWidth: w && h ? `${(maxHeight * w) / h}px` : undefined,
        }}
      >
        <canvas
          ref={ref}
          style={{ width: "100%", height: "100%", imageRendering: "pixelated" }}
        />
        {overlay && w && h && (
          <svg
            viewBox={`0 0 ${w} ${h}`}
            preserveAspectRatio="none"
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          >
            {overlay}
          </svg>
        )}
      </Box>
      {caption && (
        <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 0.5 }}>
          {caption}
        </Typography>
      )}
    </Box>
  );
};

export default HoloCanvas;
