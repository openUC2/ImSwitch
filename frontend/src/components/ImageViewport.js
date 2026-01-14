import React from "react";
import { Box, Slider, Typography } from "@mui/material";
import HistogramOverlay from "./HistogramOverlay";
import StageNudgePad from "./StageNudgePad";

export default function ImageViewport({
  detectors,
  activeTab,
  imageUrls,
  pollImageUrl,
  showHistogram,
  histogramActive,
  histogramX,
  histogramY,
  histogramData,
  chartOptions,
  pixelSize,
  minVal,
  maxVal,
  onRangeChange,
  onRangeCommit,
  onMove,
}) {
  const scaleBarPx = 50;
  const scaleBarMicrons = pixelSize ? (scaleBarPx * pixelSize).toFixed(2) : null;

  return (
    <Box sx={{ position: "relative", width: "100%", height: "100%" }}>
      {/* Histogram */}
      <HistogramOverlay
        active={histogramActive}
        visible={showHistogram}
        x={histogramX}
        y={histogramY}
        options={chartOptions}
        dataObj={histogramData}
      />

      {/* Scale bar */}
      {scaleBarMicrons && (
        <Box
          sx={{
            position: "absolute",
            bottom: 100,
            left: "60%",
            transform: "translateX(-50%)",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            zIndex: 4,
          }}
        >
          <Box
            sx={{ width: `${scaleBarPx}px`, height: "10px", backgroundColor: "white", mr: 2 }}
          />
          <Typography variant="body2">{scaleBarMicrons} µm</Typography>
        </Box>
      )}

      {/* Stage pad */}
      <Box sx={{ position: "absolute", bottom: 100, left: 10, zIndex: 3 }}>
        <StageNudgePad onMove={onMove} />
      </Box>

      {/* Images */}
      {detectors.map((d, idx) => (
        <Box
          key={d}
          sx={{
            display: activeTab === idx ? "block" : "none",
            width: "100%",
            height: "100%",
          }}
        >
          {idx === 0 ? (
            imageUrls[d] ? (
              <img src={imageUrls[d]} alt={d} style={{ width: "100%", height: "100%", objectFit: "contain" }} />
            ) : (
              <Typography>No image</Typography>
            )
          ) : pollImageUrl ? (
            <img
              src={pollImageUrl}
              alt="Polled"
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
            />
          ) : (
            <Typography>No image</Typography>
          )}
        </Box>
      ))}

      {/* Gray‑range slider */}
      <Box
        sx={{
          position: "absolute",
          top: 10,
          right: 0,
          height: "100%",
          display: "flex",
          alignItems: "center",
          zIndex: 6,
        }}
      >
        <Slider
          orientation="vertical"
          value={[minVal, maxVal]}
          onChange={onRangeChange}
          onChangeCommitted={onRangeCommit}
          min={0}
          max={255}
          valueLabelDisplay="on"
          valueLabelFormat={(v, i) => (i ? `Max: ${v}` : `Min: ${v}`)}
          sx={{ height: "60%", mr: 1 }}
        />
        <Typography
          variant="body2"
          sx={{ color: "#fff", writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          Gray Range
        </Typography>
      </Box>
    </Box>
  );
}
