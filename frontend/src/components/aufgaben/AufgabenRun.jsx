// src/components/aufgaben/AufgabenRun.jsx
// Starting and following a run: the start dialog (what the job will do, the
// ask-first steps to confirm, run parameters), the operator question dialog,
// and the monitor with pause / resume / finish / abort and the event log.
import React, { useEffect, useMemo, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  LinearProgress,
  MenuItem,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";

import * as A from "../../state/slices/AufgabenSlice";
import {
  apiAufgabenAbortJob,
  apiAufgabenAnswerQuestion,
  apiAufgabenFinishJob,
  apiAufgabenPauseJob,
  apiAufgabenResumeJob,
  apiAufgabenStartJob,
} from "../../backendapi/apiAufgabenController";
import { aufgabenErrorText, catalogById, findNode } from "../../utils/aufgabenJob";
import { CommitField } from "./AufgabenFields";
import { useT } from "../../i18n";

const STATE_COLORS = {
  idle: "text.secondary",
  running: "success.main",
  paused: "warning.main",
  finished: "text.primary",
  aborted: "warning.main",
  failed: "error.main",
};

const clock = (seconds) => {
  const s = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(s / 3600);
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
};

// Parameters the operator sets when the job starts.
const runtimeParameters = (job) =>
  Object.entries(job.parameters || {}).filter(([, p]) => (p.ui || "runtime") !== "hidden" && p.ui !== "wizard");

const ParameterInput = ({ name, spec, value, onChange }) => {
  const label = `${spec.title || name}${spec.unit ? ` (${spec.unit})` : ""}`;
  if (spec.type === "boolean") {
    return (
      <FormControlLabel
        control={<Switch size="small" checked={!!value} onChange={(e) => onChange(e.target.checked)} />}
        label={label}
      />
    );
  }
  if (spec.type === "choice") {
    return (
      <TextField select size="small" fullWidth label={label} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {(spec.choices || []).map((c) => (
          <MenuItem key={String(c)} value={c}>
            {String(c)}
          </MenuItem>
        ))}
      </TextField>
    );
  }
  const numeric = spec.type === "integer" || spec.type === "number";
  return (
    <CommitField
      label={label}
      value={value}
      helperText={spec.description || undefined}
      onCommit={(text) => {
        if (!numeric) return onChange(text);
        const n = spec.type === "integer" ? parseInt(text, 10) : parseFloat(text);
        onChange(Number.isNaN(n) ? undefined : n);
      }}
    />
  );
};

export const StartDialog = ({ open, report, onClose, onStarted }) => {
  const t = useT();
  const { job } = useSelector(A.getAufgabenState);
  const params = useMemo(() => runtimeParameters(job), [job]);
  const [values, setValues] = useState({});
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (open) {
      setValues(Object.fromEntries(params.map(([name, spec]) => [name, spec.default])));
      setConfirmed(false);
      setError("");
    }
  }, [open, params]);
  const askFirst = report?.ask_first || [];
  const effects = [
    report?.moves_xy && t("moves the stage in X and Y"),
    report?.moves_z && t("moves Z (focus)"),
    report?.emits_light && t("switches illumination on"),
  ].filter(Boolean);
  const start = async () => {
    setBusy(true);
    setError("");
    try {
      const result = await apiAufgabenStartJob(job, values, askFirst.length > 0 && confirmed);
      onStarted(result);
    } catch (e) {
      setError(aufgabenErrorText(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{t("Start {name}", { name: job.name })}</DialogTitle>
      <DialogContent>
        <Typography variant="body2">
          {effects.length
            ? t("This job {effects}.", { effects: effects.join(", ") })
            : t("This job neither moves the stage nor switches light on.")}
        </Typography>
        {(report?.issues || []).filter((i) => i.level !== "error").map((issue, i) => (
          <Alert key={i} severity="warning" sx={{ mt: 1, py: 0 }}>
            {issue.node ? `${issue.node}: ` : ""}
            {issue.message}
          </Alert>
        ))}
        {askFirst.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="warning.main">
              {t("These steps drive to end stops or swap objectives:")}
            </Typography>
            <ul style={{ margin: "4px 0" }}>
              {askFirst.map((id) => (
                <li key={id}>
                  <Typography variant="body2" sx={{ fontFamily: "monospace" }}>
                    {id}
                  </Typography>
                </li>
              ))}
            </ul>
            <FormControlLabel
              control={<Checkbox checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />}
              label={t("The objective and the sample are clear; run these steps")}
            />
          </Box>
        )}
        {params.length > 0 && (
          <Stack spacing={1.5} sx={{ mt: 2 }}>
            <Typography variant="overline" color="text.secondary">
              {t("Run parameters")}
            </Typography>
            {params.map(([name, spec]) => (
              <ParameterInput
                key={name}
                name={name}
                spec={spec}
                value={values[name]}
                onChange={(v) => setValues({ ...values, [name]: v })}
              />
            ))}
          </Stack>
        )}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t("Cancel")}</Button>
        <Button variant="contained" onClick={start} disabled={busy || (askFirst.length > 0 && !confirmed)}>
          {t("Start")}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export const QuestionDialog = () => {
  const t = useT();
  const dispatch = useDispatch();
  const { question } = useSelector(A.getAufgabenState);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    setText("");
    setError("");
  }, [question]);
  const answer = async (value) => {
    try {
      await apiAufgabenAnswerQuestion(value);
      dispatch(A.answered());
    } catch (e) {
      setError(aufgabenErrorText(e));
    }
  };
  const buttons = question?.buttons || [];
  return (
    <Dialog open={!!question} maxWidth="xs" fullWidth>
      <DialogTitle>{t("The job asks")}</DialogTitle>
      <DialogContent>
        <Typography sx={{ whiteSpace: "pre-line" }}>{question?.text}</Typography>
        {buttons.length === 0 && (
          <TextField
            autoFocus
            fullWidth
            size="small"
            sx={{ mt: 2 }}
            label={t("Answer")}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && text && answer(text)}
          />
        )}
        {error && (
          <Alert severity="error" sx={{ mt: 1 }}>
            {error}
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        {buttons.length ? (
          buttons.map((b, i) => (
            <Button key={b} variant={i === 0 ? "contained" : "text"} onClick={() => answer(b)}>
              {b}
            </Button>
          ))
        ) : (
          <Button variant="contained" disabled={!text} onClick={() => answer(text)}>
            {t("Send")}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

const Readout = ({ label, children }) => (
  <Box sx={{ minWidth: 90 }}>
    <Typography variant="overline" color="text.secondary" sx={{ lineHeight: 1.2, display: "block" }}>
      {label}
    </Typography>
    <Typography variant="body1" sx={{ fontVariantNumeric: "tabular-nums" }}>
      {children}
    </Typography>
  </Box>
);

export const RunMonitor = () => {
  const t = useT();
  const dispatch = useDispatch();
  const { status, progress, events, catalog, job } = useSelector(A.getAufgabenState);
  const [error, setError] = useState("");
  const active = A.isActive(status);
  const byId = useMemo(() => catalogById(catalog), [catalog]);
  const call = (fn) => async () => {
    setError("");
    try {
      const next = await fn();
      if (next?.state) dispatch(A.setStatus(next));
    } catch (e) {
      setError(aufgabenErrorText(e));
    }
  };
  const forceStop = () => {
    if (window.confirm(t("Abort now, stop the stage and switch all light off?"))) {
      call(() => apiAufgabenAbortJob(true))();
    }
  };
  const node = status.node && findNode(job, status.node)?.node;
  const nodeTitle = node ? `${t(byId[node.task]?.title || node.task)} (${node.id})` : status.node || "–";
  return (
    <Box>
      <Stack direction="row" spacing={4} alignItems="flex-end" flexWrap="wrap" useFlexGap>
        <Readout label={t("State")}>
          <Box component="span" sx={{ color: STATE_COLORS[status.state] || "text.primary" }}>
            {t(status.state || "idle")}
          </Box>
        </Readout>
        <Readout label={t("Job")}>{status.job || "–"}</Readout>
        <Readout label={t("Step")}>{active ? nodeTitle : "–"}</Readout>
        <Readout label={t("Frames")}>{status.frames ?? "–"}</Readout>
        <Readout label={t("Elapsed")}>{status.elapsed_s != null ? clock(status.elapsed_s) : "–"}</Readout>
      </Stack>
      {active && status.path?.length > 0 && (
        <Typography variant="body2" sx={{ mt: 1, fontFamily: "monospace" }}>
          {status.path.join("  ›  ")}
        </Typography>
      )}
      {status.message && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {status.message}
        </Typography>
      )}
      {active && progress && (
        <Box sx={{ mt: 1 }}>
          <LinearProgress variant="determinate" value={Math.min(100, progress.percentage || 0)} />
          <Typography variant="caption" color="text.secondary">
            {progress.message}
          </Typography>
        </Box>
      )}
      <Stack direction="row" spacing={1} sx={{ mt: 2 }}>
        {status.state === "paused" ? (
          <Button variant="outlined" onClick={call(apiAufgabenResumeJob)}>
            {t("Resume")}
          </Button>
        ) : (
          <Button variant="outlined" disabled={!active} onClick={call(apiAufgabenPauseJob)}>
            {t("Pause")}
          </Button>
        )}
        <Button variant="outlined" disabled={!active} onClick={call(apiAufgabenFinishJob)}>
          {t("Finish")}
        </Button>
        <Button variant="outlined" color="error" disabled={!active} onClick={call(() => apiAufgabenAbortJob(false))}>
          {t("Abort")}
        </Button>
        <Button color="error" disabled={!active} onClick={forceStop}>
          {t("Abort and stop hardware")}
        </Button>
      </Stack>
      {error && (
        <Alert severity="error" sx={{ mt: 1 }}>
          {error}
        </Alert>
      )}
      <Stack direction="row" alignItems="center" sx={{ mt: 3 }}>
        <Typography variant="overline" color="text.secondary">
          {t("Events")}
        </Typography>
        <Button size="small" sx={{ ml: "auto" }} onClick={() => dispatch(A.clearEvents())} disabled={!events.length}>
          {t("Clear")}
        </Button>
      </Stack>
      <Box sx={{ maxHeight: 320, overflowY: "auto", border: 1, borderColor: "divider" }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>{t("Time")}</TableCell>
              <TableCell>{t("Kind")}</TableCell>
              <TableCell>{t("Step")}</TableCell>
              <TableCell>{t("Message")}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {[...events].reverse().map((event, i) => (
              <TableRow key={`${event.time}-${i}`}>
                <TableCell sx={{ whiteSpace: "nowrap" }}>{new Date(event.time).toLocaleTimeString()}</TableCell>
                <TableCell
                  sx={{
                    color: event.level === "error" ? "error.main" : event.level === "warning" ? "warning.main" : undefined,
                  }}
                >
                  {event.type === "log" ? event.level : event.type}
                </TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{event.node || ""}</TableCell>
                <TableCell>{event.type === "state" ? `${event.state}${event.message ? `: ${event.message}` : ""}` : event.message || event.text || ""}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>
    </Box>
  );
};
