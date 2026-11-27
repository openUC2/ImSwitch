// src/components/holo/HoloTutorial.js
// Build guides for the HoloBox inside the app: in-line holography, the
// Michelson interferometer and the Mach-Zehnder for off-axis holography.
// Each guide: cube layout (top view, one square = one 50 mm cube), parts,
// steps with photos from the openUC2 docs, and what to look for.
import React, { useState } from "react";
import {
  Alert,
  Box,
  Button,
  Dialog,
  Grid,
  Paper,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { MenuBook as MenuBookIcon } from "@mui/icons-material";

import { useT } from "../../i18n";
import { GUIDES, LAYOUTS } from "./holoGuides";

const img = (name) => `${process.env.PUBLIC_URL}/assets/holobox/${name}`;

function CubeLayout({ layout, t }) {
  const { cols, rows, cells, beams } = LAYOUTS[layout];
  const s = 90;
  return (
    <svg viewBox={`-4 -4 ${cols * s + 8} ${rows * s + 8}`} width="100%" style={{ maxWidth: cols * s + 8, maxHeight: rows * s + 8 }}>
      {Array.from({ length: cols * rows }).map((_, i) => (
        <rect key={i} x={(i % cols) * s} y={Math.floor(i / cols) * s} width={s} height={s}
          fill="none" stroke="currentColor" strokeOpacity="0.15" />
      ))}
      {cells.map(([c, r, label]) => (
        <g key={`${c}-${r}`}>
          <rect x={c * s + 4} y={r * s + 4} width={s - 8} height={s - 8} rx="4"
            fill="currentColor" fillOpacity={label === "empty" ? 0.03 : 0.1} stroke="currentColor" strokeOpacity="0.6" />
          {/* "mirror (kinematic)" -> two lines so it fits the square */}
          {t(label).split(/ (?=\()/).map((line, k, all) => (
            <text key={k} x={c * s + s / 2} y={r * s + s - 12 - (all.length - 1 - k) * 12}
              fontSize="10" textAnchor="middle" fill="currentColor">
              {line}
            </text>
          ))}
        </g>
      ))}
      {beams.map((pts, i) => (
        <polyline key={i} points={pts.map(([x, y]) => `${x * s},${y * s - 8}`).join(" ")}
          fill="none" stroke="#d32f2f" strokeWidth="2.5" />
      ))}
    </svg>
  );
}

const HoloTutorial = ({ initialTopic = "inline", docsUrl }) => {
  const t = useT();
  const [topic, setTopic] = useState(initialTopic);
  const [zoom, setZoom] = useState(null);
  const g = GUIDES[topic];

  return (
    <Box>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
        <ToggleButtonGroup size="small" exclusive value={topic} onChange={(e, v) => v && setTopic(v)}>
          <ToggleButton value="inline">{t("In-line hologram")}</ToggleButton>
          <ToggleButton value="michelson">{t("Michelson")}</ToggleButton>
          <ToggleButton value="machzehnder">{t("Mach-Zehnder (off-axis)")}</ToggleButton>
        </ToggleButtonGroup>
        {docsUrl && (
          <Button size="small" startIcon={<MenuBookIcon />} href={`${docsUrl}usage/disc/holobox/${g.docs}`} target="_blank" rel="noopener noreferrer">
            {t("Full guide in the documentation")}
          </Button>
        )}
      </Stack>

      <Typography variant="h6">{t(g.title)}</Typography>
      <Typography variant="body1" sx={{ mb: 2, maxWidth: 900 }}>{t(g.intro)}</Typography>

      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 1.5, height: "100%" }}>
            <Typography variant="subtitle2" gutterBottom>{t("Cube layout (top view)")}</Typography>
            <CubeLayout layout={topic} t={t} />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 1.5, height: "100%" }}>
            <Typography variant="subtitle2" gutterBottom>{t("You need")}</Typography>
            <Box component="ul" sx={{ m: 0, pl: 2.5 }}>
              {g.parts.map((p) => <li key={p}><Typography variant="body2">{t(p)}</Typography></li>)}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        {g.steps.map(([file, text], i) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={file}>
            <Paper variant="outlined" sx={{ overflow: "hidden", height: "100%" }}>
              <Box component="img" src={img(file)} alt="" loading="lazy" onClick={() => setZoom(file)}
                sx={{ width: "100%", aspectRatio: "4 / 3", objectFit: "cover", display: "block", cursor: "zoom-in" }} />
              <Box sx={{ p: 1.25 }}>
                <Typography variant="body2">
                  <strong>{i + 1}.</strong> {t(text)}
                </Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Alert severity="info" sx={{ mt: 2 }}>
        <Box component="ul" sx={{ m: 0, pl: 2 }}>
          {g.tips.map((tip) => <li key={tip}>{t(tip)}</li>)}
        </Box>
      </Alert>

      <Dialog open={!!zoom} onClose={() => setZoom(null)} maxWidth="md">
        {zoom && <Box component="img" src={img(zoom)} alt="" onClick={() => setZoom(null)} sx={{ display: "block", maxWidth: "100%" }} />}
      </Dialog>
    </Box>
  );
};

export default HoloTutorial;
