// src/components/aufgaben/AufgabenLibrary.jsx
// Jobs saved on the microscope and the bundled examples; JSON and useq-schema
// import / export; the recent runs with their final state.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";

import * as A from "../../state/slices/AufgabenSlice";
import { aufgabenErrorText } from "../../utils/aufgabenJob";
import {
  apiAufgabenDeleteJob,
  apiAufgabenExportUseq,
  apiAufgabenGetJob,
  apiAufgabenImportUseq,
  apiAufgabenListJobs,
  apiAufgabenListRuns,
} from "../../backendapi/apiAufgabenController";
import { JsonField, SectionTitle } from "./AufgabenFields";
import { useT } from "../../i18n";

const download = (name, data) => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
};

const fileName = (name) => (name || "job").replace(/[^A-Za-z0-9_.-]+/g, "_");
const baseName = (path) => (path || "").split(/[\\/]/).pop();

const UseqDialog = ({ open, onClose, onImport }) => {
  const t = useT();
  const [sequence, setSequence] = useState(undefined);
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{t("Import a useq-schema MDASequence")}</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          {t("Paste the sequence as JSON, e.g. from pymmcore-plus or napari-micromanager.")}
        </Typography>
        <JsonField value={sequence} minRows={8} onChange={setSequence} />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t("Cancel")}</Button>
        <Button variant="contained" disabled={!sequence} onClick={() => onImport(sequence)}>
          {t("Import")}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const AufgabenLibrary = ({ onOpened }) => {
  const t = useT();
  const dispatch = useDispatch();
  const { job, jobs, runs, dirty, status } = useSelector(A.getAufgabenState);
  const [error, setError] = useState("");
  const [useqOpen, setUseqOpen] = useState(false);
  const fileInput = useRef(null);

  const refresh = useCallback(async () => {
    try {
      dispatch(A.setJobs(await apiAufgabenListJobs()));
      dispatch(A.setRuns(await apiAufgabenListRuns(30)));
    } catch (e) {
      setError(aufgabenErrorText(e));
    }
  }, [dispatch]);
  useEffect(() => {
    refresh();
  }, [refresh, status.state]);

  const guard = () => !dirty || window.confirm(t("Discard the unsaved changes of {name}?", { name: job.name }));
  const open = (data) => {
    dispatch(A.loadJob(data));
    setError("");
    onOpened?.();
  };
  const run = (fn) => async (...args) => {
    setError("");
    try {
      await fn(...args);
    } catch (e) {
      setError(aufgabenErrorText(e));
    }
  };
  const openJob = run(async (name) => guard() && open(await apiAufgabenGetJob(name)));
  const deleteJob = run(async (name) => {
    if (!window.confirm(t("Delete the saved job {name}?", { name }))) return;
    await apiAufgabenDeleteJob(name);
    await refresh();
  });
  const exportUseq = run(async () => download(`${fileName(job.name)}.useq.json`, await apiAufgabenExportUseq(job)));
  const importUseq = run(async (sequence) => {
    if (!guard()) return;
    open(await apiAufgabenImportUseq(sequence));
    setUseqOpen(false);
  });
  const readFile = (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file || !guard()) return;
    file
      .text()
      .then((text) => open(JSON.parse(text)))
      .catch((err) => setError(t("Could not read {file}: {error}", { file: file.name, error: err.message })));
  };

  return (
    <Box>
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        <Button variant="outlined" onClick={() => guard() && dispatch(A.newJob())}>
          {t("New job")}
        </Button>
        <Button variant="outlined" onClick={() => fileInput.current?.click()}>
          {t("Open JSON file")}
        </Button>
        <Button variant="outlined" onClick={() => download(`${fileName(job.name)}.json`, job)}>
          {t("Download JSON")}
        </Button>
        <Button variant="outlined" onClick={() => setUseqOpen(true)}>
          {t("Import useq")}
        </Button>
        <Button variant="outlined" onClick={exportUseq}>
          {t("Export useq")}
        </Button>
        <input ref={fileInput} type="file" accept=".json,application/json" hidden onChange={readFile} />
      </Stack>
      {error && (
        <Alert severity="error" sx={{ mt: 1 }} onClose={() => setError("")}>
          {error}
        </Alert>
      )}
      <SectionTitle>{t("Jobs on this microscope")}</SectionTitle>
      <Table size="small">
        <TableBody>
          {jobs.map((entry) => (
            <TableRow key={`${entry.source}-${entry.name}`} hover>
              <TableCell>{entry.name}</TableCell>
              <TableCell sx={{ color: "text.secondary" }}>{t(entry.source)}</TableCell>
              <TableCell align="right">
                <Button size="small" onClick={() => openJob(entry.name)}>
                  {t("Open")}
                </Button>
                {entry.source === "saved" && (
                  <Button size="small" color="error" onClick={() => deleteJob(entry.name)}>
                    {t("Delete")}
                  </Button>
                )}
              </TableCell>
            </TableRow>
          ))}
          {!jobs.length && (
            <TableRow>
              <TableCell colSpan={3} sx={{ color: "text.secondary" }}>
                {t("No saved jobs yet.")}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      <SectionTitle>{t("Recent runs")}</SectionTitle>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>{t("Folder")}</TableCell>
            <TableCell>{t("Job")}</TableCell>
            <TableCell>{t("State")}</TableCell>
            <TableCell align="right">{t("Frames")}</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {runs.map((r) => (
            <TableRow key={r.folder}>
              <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }} title={r.folder}>
                {baseName(r.folder)}
              </TableCell>
              <TableCell>{r.job}</TableCell>
              <TableCell>{t(r.state || "")}</TableCell>
              <TableCell align="right">{r.frames ?? ""}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <UseqDialog open={useqOpen} onClose={() => setUseqOpen(false)} onImport={importUseq} />
    </Box>
  );
};

export default AufgabenLibrary;
