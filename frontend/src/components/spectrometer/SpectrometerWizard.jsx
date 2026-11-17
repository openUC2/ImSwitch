// Calibration wizard: pick known sources (lasers, NeoPixel R/G/B with reference spectra,
// sources switched by hand such as the Q-Box 520 nm laser or a neon lamp), run them one
// after the other, review the fit and its residuals, then apply or discard.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  LinearProgress,
  MenuItem,
  Stack,
  Step,
  StepLabel,
  Stepper,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import * as api from "../../backendapi/apiSpectrometerController.js";
import SpectrumChart from "./SpectrumChart.jsx";

const GROUPS = [
  { title: "Lasers in ImSwitch", test: (s) => s.control === "laser" },
  { title: "NeoPixel ring (reference spectra)", test: (s) => s.control === "ledmatrix" },
  { title: "Switched by hand", test: (s) => s.control === "manual" },
];

const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(d));

function parseReferenceCsv(text) {
  // header: wavelength column first, then one column per channel (r, g, b or red, green, blue)
  const rows = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));
  const sep = rows[0].includes(";") ? ";" : rows[0].includes("\t") ? "\t" : ",";
  const head = rows[0].split(sep).map((h) => h.trim().toLowerCase());
  const body = rows.slice(1).map((r) => r.split(sep).map(Number));
  const key = (h) => ({ r: "r", red: "r", g: "g", green: "g", b: "b", blue: "b" })[h] || h;
  return head.slice(1).map((h, j) => ({
    key: key(h),
    wavelengthNm: body.map((r) => r[0]),
    intensity: body.map((r) => r[j + 1]),
    label: `reference ${h}`,
  }));
}

export default function SpectrometerWizard({ open, onClose, sources, onChanged }) {
  const calSources = useMemo(
    () => (sources || []).filter((s) => s.forCalibration),
    [sources],
  );
  const [selected, setSelected] = useState([]);
  const [wavelengths, setWavelengths] = useState({});
  const [sigmas, setSigmas] = useState({});
  const [opts, setOpts] = useState({
    model: "auto",
    order: 2,
    nAverage: 5,
    nDark: 10,
    takeDark: true,
    measureSmile: true,
  });
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [refMsg, setRefMsg] = useState(null);
  const fileRef = useRef(null);

  // default choice: lasers in ImSwitch and the hand-switched lasers (no lamps)
  useEffect(() => {
    if (!open) return;
    setSelected((cur) =>
      cur.length
        ? cur
        : calSources
            .filter((s) => s.kind === "laser")
            .map((s) => s.id),
    );
  }, [open, calSources]);

  const refresh = useCallback(async () => {
    try {
      const st = await api.apiSpectrometerGetWizard();
      setStatus(st);
      return st;
    } catch (e) {
      setError(String(e));
      return null;
    }
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    refresh();
    const id = setInterval(refresh, 500);
    return () => clearInterval(id);
  }, [open, refresh]);

  const state = status?.state || "idle";
  const running = state === "running" || state === "waiting";
  const reviewing = state === "review";

  const toggle = (id) =>
    setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));

  const start = async () => {
    setError(null);
    const ordered = calSources.map((s) => s.id).filter((id) => selected.includes(id));
    const r = await api.apiSpectrometerStartWizard({
      sources: ordered,
      ...opts,
      wavelengths: Object.fromEntries(
        Object.entries(wavelengths).filter(([, v]) => Number.isFinite(parseFloat(v))).map(([k, v]) => [k, parseFloat(v)]),
      ),
      sigmas: Object.fromEntries(
        Object.entries(sigmas).filter(([, v]) => Number.isFinite(parseFloat(v))).map(([k, v]) => [k, parseFloat(v)]),
      ),
    });
    if (!r.success) setError(r.error);
    refresh();
  };

  const apply = async () => {
    const r = await api.apiSpectrometerApplyWizard(true);
    if (!r.success) setError(r.error);
    refresh();
    onChanged?.();
  };

  const uploadReferences = async (file) => {
    setRefMsg(null);
    try {
      const specs = parseReferenceCsv(await file.text());
      const out = [];
      for (const s of specs) {
        const r = await api.apiSpectrometerSetReference(s);
        out.push(r.success ? `${s.key}: centre ${fmt(r.centerNm, 1)} nm, FWHM ${fmt(r.fwhmNm, 1)} nm` : `${s.key}: ${r.error}`);
      }
      setRefMsg(out.join(" · "));
      onChanged?.();
    } catch (e) {
      setRefMsg(`Could not read ${file.name}: ${e}`);
    }
  };

  const current = status?.steps?.[status?.stepIndex];
  const currentSpectrum = current ? status?.spectra?.[current.id] : null;
  const fit = status?.fit;

  return (
    <Dialog open={open} onClose={running ? undefined : onClose} maxWidth="lg" fullWidth>
      <DialogTitle>Wavelength calibration</DialogTitle>
      <DialogContent dividers>
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 1 }}>
            {error}
          </Alert>
        )}

        {!running && !reviewing && (
          <Stack spacing={2}>
            <Typography variant="body2" color="text.secondary">
              Each source is switched on alone, a few frames are averaged and its line (or
              band centre) is found. Lasers give the best points; check their real wavelength
              (a "520 nm" diode can be a few nm off) and enter it if it was measured. The
              NeoPixel points are only as good as their reference spectra.
            </Typography>
            {GROUPS.map((g) => {
              const items = calSources.filter(g.test);
              if (!items.length) return null;
              return (
                <Box key={g.title}>
                  <Typography variant="subtitle2">{g.title}</Typography>
                  {items.map((s) => (
                    <Stack key={s.id} direction="row" spacing={1} alignItems="center" sx={{ flexWrap: "wrap" }}>
                      <FormControlLabel
                        control={<Checkbox checked={selected.includes(s.id)} onChange={() => toggle(s.id)} />}
                        label={s.label}
                        sx={{ minWidth: 280 }}
                      />
                      {s.wavelengthNm != null && (
                        <>
                          <TextField
                            size="small"
                            label="line (nm)"
                            sx={{ width: 110 }}
                            value={wavelengths[s.id] ?? s.wavelengthNm}
                            onChange={(e) => setWavelengths({ ...wavelengths, [s.id]: e.target.value })}
                          />
                          <TextField
                            size="small"
                            label="± nm"
                            sx={{ width: 80 }}
                            value={sigmas[s.id] ?? s.sigmaNm}
                            onChange={(e) => setSigmas({ ...sigmas, [s.id]: e.target.value })}
                          />
                        </>
                      )}
                      {s.reference && (
                        <Chip
                          size="small"
                          color={s.reference.placeholder ? "warning" : "success"}
                          label={`${s.reference.placeholder ? "placeholder" : "measured"} · ${fmt(s.reference.centerNm, 1)} nm`}
                        />
                      )}
                      {s.lines && <Chip size="small" label={`${s.lines.length} known lines`} />}
                    </Stack>
                  ))}
                </Box>
              );
            })}
            <Stack direction="row" spacing={1} alignItems="center">
              <Button size="small" variant="outlined" onClick={() => fileRef.current?.click()}>
                Load ring reference spectra (CSV)
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.txt"
                hidden
                onChange={(e) => e.target.files?.[0] && uploadReferences(e.target.files[0])}
              />
              <Typography variant="caption" color="text.secondary">
                header: wavelength_nm,r,g,b (measured on another spectrometer)
              </Typography>
            </Stack>
            {refMsg && <Alert severity="info">{refMsg}</Alert>}
            <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
              <TextField select size="small" label="model" value={opts.model} sx={{ width: 150 }}
                onChange={(e) => setOpts({ ...opts, model: e.target.value })}>
                <MenuItem value="auto">auto</MenuItem>
                <MenuItem value="grating">grating model</MenuItem>
                <MenuItem value="polynomial">polynomial</MenuItem>
              </TextField>
              <TextField size="small" type="number" label="poly order" value={opts.order} sx={{ width: 100 }}
                onChange={(e) => setOpts({ ...opts, order: parseInt(e.target.value, 10) || 2 })} />
              <TextField size="small" type="number" label="frames / source" value={opts.nAverage} sx={{ width: 130 }}
                onChange={(e) => setOpts({ ...opts, nAverage: parseInt(e.target.value, 10) || 1 })} />
              <FormControlLabel control={<Checkbox checked={opts.takeDark}
                onChange={(e) => setOpts({ ...opts, takeDark: e.target.checked })} />} label="dark first" />
              <FormControlLabel control={<Checkbox checked={opts.measureSmile}
                onChange={(e) => setOpts({ ...opts, measureSmile: e.target.checked })} />} label="measure smile" />
            </Stack>
            {state !== "idle" && (
              <Alert severity={state === "saved" ? "success" : state === "error" ? "error" : "info"}>
                Last run: {state} {status?.message ? `– ${status.message}` : ""}
              </Alert>
            )}
          </Stack>
        )}

        {(running || reviewing) && status?.steps && (
          <Stack spacing={2}>
            <Stepper activeStep={status.stepIndex} alternativeLabel>
              {status.steps.map((s) => (
                <Step key={s.id} completed={s.status === "done"}>
                  <StepLabel error={s.status === "failed"} optional={<Typography variant="caption">{s.status}</Typography>}>
                    {s.label}
                  </StepLabel>
                </Step>
              ))}
            </Stepper>
            {running && <LinearProgress />}
            {state === "waiting" && (
              <Alert
                severity="warning"
                action={
                  <Stack direction="row" spacing={1}>
                    <Button color="inherit" variant="outlined" onClick={api.apiSpectrometerSkipWizardStep}>Skip</Button>
                    <Button color="inherit" variant="contained" onClick={api.apiSpectrometerContinueWizard}>Continue</Button>
                  </Stack>
                }
              >
                {status.message}
              </Alert>
            )}
            {state === "running" && <Typography variant="body2">{status.message}</Typography>}
            {currentSpectrum && (
              <SpectrumChart
                spectrum={currentSpectrum}
                axis="px"
                height={200}
                markers={(currentSpectrum.peaks || []).map((p) => ({ x: p.px, label: `${fmt(p.px, 1)} px`, color: "#e91e63" }))}
              />
            )}
            {status.steps.map((s) => (
              <Typography key={s.id} variant="caption" display="block">
                <b>{s.label}</b>: {s.detail || s.status}
              </Typography>
            ))}
          </Stack>
        )}

        {reviewing && fit && (
          <Stack spacing={1} sx={{ mt: 2 }}>
            <Typography variant="subtitle1">
              Fit: {fit.model} · rms {fmt(fit.rmsNm, 3)} nm · {fmt(fit.summary?.nmAtFirstPx, 1)}–
              {fmt(fit.summary?.nmAtLastPx, 1)} nm · {fmt(fit.summary?.nmPerPxAtCenter, 4)} nm/px
              {status.smile ? ` · smile ${fmt(status.smile.bowPx, 1)} px` : ""}
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>source</TableCell>
                  <TableCell align="right">column (px)</TableCell>
                  <TableCell align="right">line (nm)</TableCell>
                  <TableCell align="right">± (nm)</TableCell>
                  <TableCell align="right">residual (nm)</TableCell>
                  <TableCell align="right">FWHM (px)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {fit.points.map((p, i) => (
                  <TableRow key={i}>
                    <TableCell>{p.label}</TableCell>
                    <TableCell align="right">{fmt(p.px, 1)}</TableCell>
                    <TableCell align="right">{fmt(p.nm, 2)}</TableCell>
                    <TableCell align="right">{fmt(p.sigmaNm, 2)}</TableCell>
                    <TableCell align="right" sx={{ color: Math.abs(p.residualNm) > 3 * p.sigmaNm ? "error.main" : undefined }}>
                      {fmt(p.residualNm, 3)}
                    </TableCell>
                    <TableCell align="right">{fmt(p.fwhmPx, 1)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {status.resolution?.length > 0 && (
              <Typography variant="body2">
                Line widths:{" "}
                {status.resolution
                  .map((r) => `${r.source} ${fmt(r.fwhmPx, 1)} px = ${fmt(r.fwhmNm, 3)} nm (slit image ${fmt(r.slitImagePx, 1)} px)`)
                  .join(" · ")}
              </Typography>
            )}
            {(status.hints || []).map((h, i) => (
              <Alert key={i} severity="info" sx={{ py: 0 }}>
                {h}
              </Alert>
            ))}
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        {!running && !reviewing && (
          <>
            <Button onClick={onClose}>Close</Button>
            <Button variant="contained" disabled={!selected.length} onClick={start}>
              Start ({selected.length} source{selected.length === 1 ? "" : "s"})
            </Button>
          </>
        )}
        {running && (
          <Button color="error" onClick={api.apiSpectrometerCancelWizard}>
            Cancel (all sources off)
          </Button>
        )}
        {reviewing && (
          <>
            <Button color="error" onClick={async () => { await api.apiSpectrometerCancelWizard(); refresh(); }}>
              Discard
            </Button>
            <Button variant="contained" onClick={apply}>
              Apply and save
            </Button>
          </>
        )}
      </DialogActions>
    </Dialog>
  );
}
