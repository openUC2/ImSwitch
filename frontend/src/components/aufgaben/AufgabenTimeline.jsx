// src/components/aufgaben/AufgabenTimeline.jsx
// Gantt view of the current (or last) run: one row per job step in outline
// order, one bar per execution of that step, coloured by task category.
// Wheel zooms, drag pans (plotly), double click fits the whole run.
// Steps shorter than a pixel at the current zoom are merged into one bar
// (hover shows how many), so a long run stays light to draw.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSelector } from "react-redux";
import Plot from "react-plotly.js";
import { Box, Typography, useTheme } from "@mui/material";

import * as A from "../../state/slices/AufgabenSlice";
import { apiAufgabenGetTimeline } from "../../backendapi/apiAufgabenController";
import { catalogById } from "../../utils/aufgabenJob";
import { useT } from "../../i18n";

const POLL_MS = 1000;
const PIXELS = 1500; // bars per row at most, at any zoom
const PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#edc948", "#76b7b2",
  "#e15759", "#9c755f", "#bab0ac", "#86bcb6", "#d4a6c8", "#8cd17d"];

const seconds = (s) => (s < 1 ? `${Math.round(s * 1000)} ms` : s < 120 ? `${s.toFixed(2)} s` : `${(s / 60).toFixed(1)} min`);

// [t, node, 1|0] marks -> { node: [[start, end|null], ...] }
export const spansFromMarks = (marks) => {
  const spans = {};
  const open = {};
  for (const [t, node, start] of marks) {
    if (start) {
      (open[node] = open[node] || []).push((spans[node] = spans[node] || []).length);
      spans[node].push([t, null]);
    } else if (open[node]?.length) {
      spans[node][open[node].pop()][1] = t;
    }
  }
  return spans;
};

// Merge bars closer than `gap` (and inside [lo, hi]) into one: [start, end, count, running].
export const mergeSpans = (list, now, gap, lo, hi) => {
  const out = [];
  for (const [s, e] of list) {
    const end = e ?? now;
    if (end < lo || s > hi) continue;
    const last = out[out.length - 1];
    if (last && s - last[1] < gap && !last[3]) {
      last[1] = Math.max(last[1], end);
      last[2] += 1;
      last[3] = e == null;
    } else {
      out.push([s, end, 1, e == null]);
    }
  }
  return out;
};

const AufgabenTimeline = () => {
  const t = useT();
  const theme = useTheme();
  const { status, catalog } = useSelector(A.getAufgabenState);
  const active = A.isActive(status);
  const byId = useMemo(() => catalogById(catalog), [catalog]);
  const [data, setData] = useState({ runId: null, rows: [], marks: [], now: 0, truncated: false });
  const [range, setRange] = useState(null); // visible x range, null = whole run
  const next = useRef(0);
  const runId = useRef(null);

  const fetchMore = useCallback(async () => {
    const since = runId.current ? next.current : 0;
    try {
      const r = await apiAufgabenGetTimeline(since);
      if (r.run_id !== runId.current && since !== 0) {
        runId.current = null; // a new run started since the last poll: fetch it from the start
        return fetchMore();
      }
      if (since === 0) {
        runId.current = r.run_id;
        setRange(null);
        setData({ runId: r.run_id, rows: r.rows || [], marks: r.events, now: r.now, truncated: r.truncated });
      } else {
        setData((d) => ({ ...d, marks: d.marks.concat(r.events), now: r.now, truncated: r.truncated }));
      }
      next.current = r.next;
    } catch (e) {
      // the controller is missing or the server is restarting; the next poll retries
    }
    return undefined;
  }, []);

  useEffect(() => {
    fetchMore();
    if (!active) return undefined;
    const timer = setInterval(fetchMore, POLL_MS);
    return () => clearInterval(timer);
  }, [active, status.run_id, fetchMore]);

  const labels = useMemo(
    () => data.rows.map((row) => `${"\u00a0\u00a0".repeat(row.depth)}${row.label || t(byId[row.task]?.title || row.task)} (${row.id})`),
    [data.rows, byId, t],
  );

  const traces = useMemo(() => {
    const spans = spansFromMarks(data.marks);
    const [lo, hi] = range || [0, Math.max(data.now, 1e-3)];
    const gap = (hi - lo) / PIXELS;
    const byCategory = {};
    data.rows.forEach((row, index) => {
      const entry = byId[row.task];
      const category = entry?.category || t("Control");
      const label = labels[index];
      const trace = (byCategory[category] = byCategory[category] || { name: category, y: [], base: [], x: [], text: [], running: [] });
      for (const [s, e, n, running] of mergeSpans(spans[row.id] || [], data.now, gap, lo, hi)) {
        trace.y.push(index);
        trace.base.push(s);
        trace.x.push(Math.max(e - s, gap));
        trace.running.push(running);
        trace.text.push(
          n > 1
            ? `${label.trim()}<br>${n}× ${t("between")} ${seconds(s)} – ${seconds(e)}`
            : `${label.trim()}<br>${seconds(e - s)} ${t("from")} ${seconds(s)}${running ? ` (${t("running")})` : ""}`,
        );
      }
    });
    return Object.values(byCategory).map((c, i) => ({
      type: "bar",
      orientation: "h",
      name: c.name,
      y: c.y,
      base: c.base,
      x: c.x,
      hovertext: c.text,
      hoverinfo: "text",
      marker: {
        color: PALETTE[i % PALETTE.length],
        line: { width: c.running.map((r) => (r ? 2 : 0)), color: theme.palette.success.main },
      },
    }));
  }, [data, range, labels, byId, t, theme]);

  const onRelayout = (e) => {
    if (e["xaxis.autorange"]) setRange(null);
    else if (e["xaxis.range[0]"] != null) setRange([e["xaxis.range[0]"], e["xaxis.range[1]"]]);
    else if (Array.isArray(e["xaxis.range"])) setRange(e["xaxis.range"]);
  };

  if (!data.runId || !data.rows.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        {t("The timeline of the next run shows here.")}
      </Typography>
    );
  }
  const text = theme.palette.text.secondary;
  return (
    <Box>
      <Plot
        data={traces}
        layout={{
          barmode: "overlay",
          bargap: 0.3,
          height: Math.min(700, 70 + 22 * data.rows.length),
          margin: { t: 8, r: 8, b: 36, l: 8 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: text, size: 11 },
          showlegend: traces.length > 1,
          legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0 },
          uirevision: data.runId, // keep the user's zoom while the run grows
          dragmode: "pan",
          xaxis: {
            title: { text: t("Run time (s)"), standoff: 4 },
            gridcolor: theme.palette.divider,
            zeroline: false,
            rangemode: "tozero",
          },
          yaxis: {
            tickmode: "array",
            tickvals: labels.map((_, i) => i),
            ticktext: labels,
            range: [labels.length - 0.5, -0.5], // every step, also those not run yet
            automargin: true,
            fixedrange: true,
            gridcolor: theme.palette.divider,
          },
        }}
        config={{ displayModeBar: false, scrollZoom: true, responsive: true }}
        useResizeHandler
        style={{ width: "100%" }}
        onRelayout={onRelayout}
      />
      <Typography variant="caption" color="text.secondary">
        {t("Scroll to zoom, drag to pan, double click to fit the run.")}
        {data.truncated ? ` ${t("This run has more steps than the timeline keeps; later steps are not shown.")}` : ""}
      </Typography>
    </Box>
  );
};

export default AufgabenTimeline;
