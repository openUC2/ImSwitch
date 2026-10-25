// src/components/aufgaben/AufgabenController.jsx
// Aufgaben app: build acquisition jobs from the steps this microscope offers
// (getAufgabenCatalog), validate them, start them with the ask-first steps
// confirmed, follow the run (sigAufgabenEvent + getAufgabenStatus) and keep a
// library of jobs. The format and the engine are described in
// docs/AUFGABEN_ROADMAP.md.
import React, { useCallback, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Alert, Box, Button, Paper, Stack, Tab, Tabs, Typography } from "@mui/material";
import EditOutlined from "@mui/icons-material/EditOutlined";
import FactCheckOutlined from "@mui/icons-material/FactCheckOutlined";
import LibraryBooksOutlined from "@mui/icons-material/LibraryBooksOutlined";
import MonitorHeartOutlined from "@mui/icons-material/MonitorHeartOutlined";
import PlayArrowOutlined from "@mui/icons-material/PlayArrowOutlined";
import SaveOutlined from "@mui/icons-material/SaveOutlined";

import * as A from "../../state/slices/AufgabenSlice";
import { aufgabenErrorText } from "../../utils/aufgabenJob";
import {
  apiAufgabenGetCatalog,
  apiAufgabenGetStatus,
  apiAufgabenSaveJob,
  apiAufgabenValidateJob,
} from "../../backendapi/apiAufgabenController";
import AufgabenPalette from "./AufgabenPalette";
import AufgabenOutline from "./AufgabenOutline";
import AufgabenInspector from "./AufgabenInspector";
import AufgabenLibrary from "./AufgabenLibrary";
import { QuestionDialog, RunMonitor, StartDialog } from "./AufgabenRun";
import { useT } from "../../i18n";

const POLL_MS = 1000;
const PANEL_HEIGHT = "calc(100vh - 220px)";

const AufgabenController = () => {
  const t = useT();
  const dispatch = useDispatch();
  const { job, catalog, catalogError, validation, dirty, status } = useSelector(A.getAufgabenState);
  const [tab, setTab] = useState("edit");
  const [message, setMessage] = useState(null);   // {severity, text}
  const [startReport, setStartReport] = useState(null);
  const active = A.isActive(status);

  const loadCatalog = useCallback(async () => {
    try {
      dispatch(A.setCatalog(await apiAufgabenGetCatalog()));
    } catch (e) {
      const missing = e?.response?.status === 404;
      dispatch(
        A.setCatalogError(
          missing ? t("This microscope does not run the AufgabenController.") : aufgabenErrorText(e),
        ),
      );
    }
  }, [dispatch, t]);

  const pollStatus = useCallback(async () => {
    try {
      dispatch(A.setStatus(await apiAufgabenGetStatus()));
    } catch (e) {
      // the catalogue error already explains a missing controller
    }
  }, [dispatch]);

  useEffect(() => {
    loadCatalog();
    pollStatus();
  }, [loadCatalog, pollStatus]);

  useEffect(() => {
    if (!active) return undefined;
    const timer = setInterval(pollStatus, POLL_MS);
    return () => clearInterval(timer);
  }, [active, pollStatus]);

  const validate = async () => {
    setMessage(null);
    try {
      const report = await apiAufgabenValidateJob(job);
      dispatch(A.setValidation(report));
      const errors = report.issues.filter((i) => i.level === "error").length;
      const warnings = report.issues.length - errors;
      setMessage(
        errors
          ? { severity: "error", text: t("{n} errors; they are marked in the outline.", { n: errors }) }
          : { severity: warnings ? "warning" : "success", text: warnings ? t("Valid, with {n} warnings.", { n: warnings }) : t("The job is valid.") },
      );
      return report;
    } catch (e) {
      setMessage({ severity: "error", text: aufgabenErrorText(e) });
      return null;
    }
  };

  const start = async () => {
    const report = await validate();
    if (!report) return;
    if (!report.ok) {
      setTab("edit");
      return;
    }
    setStartReport(report);
  };

  const save = async () => {
    setMessage(null);
    try {
      const result = await apiAufgabenSaveJob(job);
      dispatch(A.markSaved());
      setMessage({ severity: "success", text: t("Saved as {name}.", { name: result.name }) });
    } catch (e) {
      setMessage({ severity: "error", text: aufgabenErrorText(e) });
    }
  };

  if (catalogError) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info" action={<Button onClick={loadCatalog}>{t("Retry")}</Button>}>
          {catalogError}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }} flexWrap="wrap" useFlexGap>
        <Typography variant="h6" sx={{ mr: 1 }}>
          {job.name}
          {dirty && (
            <Typography component="span" color="text.secondary" sx={{ ml: 1, fontSize: 13 }}>
              {t("unsaved")}
            </Typography>
          )}
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Button onClick={validate} startIcon={<FactCheckOutlined />}>
          {t("Validate")}
        </Button>
        <Button onClick={save} startIcon={<SaveOutlined />}>
          {t("Save")}
        </Button>
        <Button
          variant="contained"
          onClick={start}
          disabled={active || !catalog.length}
          startIcon={<PlayArrowOutlined />}
        >
          {t("Start")}
        </Button>
      </Stack>
      {message && (
        <Alert severity={message.severity} sx={{ mb: 1, py: 0 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}
      {validation && validation.ask_first?.length > 0 && !message && (
        <Alert severity="warning" sx={{ mb: 1, py: 0 }}>
          {t("Asks for confirmation at start: {steps}", { steps: validation.ask_first.join(", ") })}
        </Alert>
      )}
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ minHeight: 36, mb: 1 }}>
        <Tab value="edit" label={t("Edit")} icon={<EditOutlined fontSize="small" />} iconPosition="start" sx={{ minHeight: 36 }} />
        <Tab
          value="run"
          label={active ? t("Run ({state})", { state: t(status.state) }) : t("Run")}
          icon={<MonitorHeartOutlined fontSize="small" />}
          iconPosition="start"
          sx={{ minHeight: 36 }}
        />
        <Tab
          value="library"
          label={t("Library")}
          icon={<LibraryBooksOutlined fontSize="small" />}
          iconPosition="start"
          sx={{ minHeight: 36 }}
        />
      </Tabs>
      {tab === "edit" && (
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "220px minmax(0, 1fr) 380px" },
            gap: 1,
            height: { md: PANEL_HEIGHT },
            minHeight: 480,
          }}
        >
          <Paper variant="outlined" sx={{ p: 1, minHeight: 0 }}>
            <AufgabenPalette catalog={catalog} />
          </Paper>
          <Paper variant="outlined" sx={{ p: 1, minHeight: 0 }}>
            <AufgabenOutline />
          </Paper>
          <Paper variant="outlined" sx={{ minHeight: 0 }}>
            <AufgabenInspector />
          </Paper>
        </Box>
      )}
      {tab === "run" && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <RunMonitor />
        </Paper>
      )}
      {tab === "library" && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <AufgabenLibrary onOpened={() => setTab("edit")} />
        </Paper>
      )}
      <StartDialog
        open={!!startReport}
        report={startReport}
        onClose={() => setStartReport(null)}
        onStarted={() => {
          setStartReport(null);
          dispatch(A.clearEvents());
          setTab("run");
          pollStatus();
        }}
      />
      <QuestionDialog />
    </Box>
  );
};

export default AufgabenController;
