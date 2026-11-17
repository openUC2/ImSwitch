// Spectrum plot: counts against pixel, wavelength or Raman shift, with vertical marker
// lines (known lines, alignment target, peaks). Chart.js (registered in utils/chartSetup).
import { useMemo } from "react";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, LogarithmicScale } from "chart.js";
import { useTheme } from "@mui/material/styles";
import "../../utils/chartSetup";

ChartJS.register(LogarithmicScale);

export const AXES = {
  px: { key: "px", label: "pixel (column)", digits: 0 },
  nm: { key: "nm", label: "wavelength (nm)", digits: 2 },
  cm1: { key: "ramanShiftCm1", label: "Raman shift (cm⁻¹)", digits: 1 },
};

// markers: [{x (in the chart's axis units), label, color, dash}]
const markerPlugin = {
  id: "spectrumMarkers",
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx, chartArea, scales } = chart;
    const markers = opts?.markers || [];
    ctx.save();
    markers.forEach((m) => {
      if (m.x == null || !Number.isFinite(m.x)) return;
      const px = scales.x.getPixelForValue(m.x);
      if (px < chartArea.left || px > chartArea.right) return;
      ctx.strokeStyle = m.color || "#e91e63";
      ctx.lineWidth = m.width || 1.5;
      ctx.setLineDash(m.dash || [6, 4]);
      ctx.beginPath();
      ctx.moveTo(px, chartArea.top);
      ctx.lineTo(px, chartArea.bottom);
      ctx.stroke();
      if (m.label) {
        ctx.setLineDash([]);
        ctx.fillStyle = m.color || "#e91e63";
        ctx.font = "11px sans-serif";
        const w = ctx.measureText(m.label).width;
        const x = Math.min(px + 4, chartArea.right - w - 2);
        ctx.fillText(m.label, x, chartArea.top + 12 + (m.row || 0) * 13);
      }
    });
    ctx.restore();
  },
};

export default function SpectrumChart({
  spectrum,
  axis = "px",
  markers = [],
  height = 340,
  logY = false,
  overlays = [],
}) {
  const theme = useTheme();
  const ax = AXES[axis] || AXES.px;

  const data = useMemo(() => {
    const series = [];
    const toPoints = (s) => {
      const xs = s?.[ax.key];
      if (!s || !xs || !s.counts) return [];
      const pts = [];
      for (let i = 0; i < s.counts.length; i += 1) {
        const x = xs[i];
        const y = s.counts[i];
        if (x != null && y != null) pts.push({ x, y: logY ? Math.max(y, 0.1) : y });
      }
      // wavelength falls with the column on this instrument: draw left to right
      if (pts.length > 1 && pts[0].x > pts[pts.length - 1].x) pts.reverse();
      return pts;
    };
    if (spectrum) {
      series.push({
        label: spectrum.which === "acquired" ? "acquired" : "live",
        data: toPoints(spectrum),
        borderColor: theme.palette.primary.main,
        borderWidth: 1.2,
        pointRadius: 0,
        tension: 0,
      });
    }
    overlays.forEach((o, i) =>
      series.push({
        label: o.label,
        data: toPoints(o.spectrum),
        borderColor: o.color || ["#ff9800", "#4caf50", "#9c27b0", "#f44336"][i % 4],
        borderWidth: 1,
        pointRadius: 0,
        tension: 0,
      }),
    );
    return { datasets: series };
  }, [spectrum, overlays, ax.key, logY, theme.palette.primary.main]);

  const options = useMemo(
    () => ({
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: { display: overlays.length > 0, labels: { boxWidth: 12 } },
        tooltip: {
          callbacks: {
            title: (items) =>
              items.length ? `${items[0].parsed.x.toFixed(ax.digits)} ${axis === "px" ? "px" : axis === "nm" ? "nm" : "cm⁻¹"}` : "",
            label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(1)} counts`,
          },
        },
        spectrumMarkers: { markers },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: ax.label },
          ticks: { maxTicksLimit: 12 },
          grid: { color: theme.palette.divider },
        },
        y: {
          type: logY ? "logarithmic" : "linear",
          title: { display: true, text: "counts" },
          grid: { color: theme.palette.divider },
        },
      },
    }),
    [ax, axis, markers, logY, overlays.length, theme.palette.divider],
  );

  return (
    <div style={{ position: "relative", height, width: "100%" }}>
      <Line data={data} options={options} plugins={[markerPlugin]} />
    </div>
  );
}
