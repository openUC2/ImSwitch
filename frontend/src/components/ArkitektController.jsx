// src/components/ArkitektController.jsx
// Arkitekt panel: bind this microscope to an Arkitekt server (device-code
// login: the server shows a code, someone approves it in a browser), unbind
// it again, and see what Arkitekt may call, what it called and which images
// the microscope sent to the server.
import React, { useCallback, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Alert,
  Box,
  Button,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControlLabel,
  Grid,
  IconButton,
  LinearProgress,
  Link,
  Paper,
  Step,
  StepLabel,
  Stepper,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";

import * as arkitektSlice from "../state/slices/ArkitektSlice";
import {
  apiArkitektBind,
  apiArkitektCancel,
  apiArkitektClearActivity,
  apiArkitektGetStatus,
  apiArkitektGetUploads,
  apiArkitektSetSettings,
  apiArkitektUnbind,
} from "../backendapi/apiArkitektController";
import { useT } from "../i18n";

const POLL_MS = 2000;
const INSTALL_COMMAND = 'pip install "arkitekt[rekuest,mikro]"';

// state -> [label, palette colour] for the status line
const STATE_LABELS = {
  unavailable: ["Not installed", "text.disabled"],
  disabled: ["Disabled in the setup", "text.disabled"],
  unbound: ["Not bound", "text.secondary"],
  connecting: ["Connecting…", "info.main"],
  awaiting_login: ["Waiting for approval", "warning.main"],
  connected: ["Connected", "success.main"],
  error: ["Error", "error.main"],
};

// Bind steps: the server is found, someone approves, the agent provides.
const STEPS = ["Find server", "Approve in a browser", "Provide actions"];
const ACTIVE_STEP = { connecting: 0, awaiting_login: 1, connected: 3 };

const time = (iso) => (iso ? new Date(iso).toLocaleTimeString() : "–");

const elapsed = (iso) => {
  if (!iso) return "";
  const s = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

const copy = (text) => {
  try {
    navigator.clipboard?.writeText(text);
  } catch (_) {
    // clipboard needs a secure context; the text stays selectable
  }
};

// label / value rows, as an instrument readout
const Facts = ({ rows }) => (
  <Box
    component="dl"
    sx={{ m: 0, display: "grid", gridTemplateColumns: "max-content 1fr", columnGap: 2, rowGap: 0.5 }}
  >
    {rows.filter(Boolean).map(([label, value]) => (
      <React.Fragment key={label}>
        <Typography component="dt" variant="caption" color="text.secondary"
          sx={{ textTransform: "uppercase", letterSpacing: 0.5, pt: 0.25 }}>
          {label}
        </Typography>
        <Typography component="dd" variant="body2" sx={{ m: 0, fontFamily: "monospace", wordBreak: "break-all" }}>
          {value}
        </Typography>
      </React.Fragment>
    ))}
  </Box>
);

const Section = ({ title, action, children }) => (
  <Paper variant="outlined" sx={{ p: 2, height: "100%" }}>
    <Box sx={{ display: "flex", alignItems: "center", mb: 1.5 }}>
      <Typography variant="subtitle2" sx={{ textTransform: "uppercase", letterSpacing: 0.8, flex: 1 }}>
        {title}
      </Typography>
      {action}
    </Box>
    {children}
  </Paper>
);

const ArkitektController = () => {
  const t = useT();
  const dispatch = useDispatch();
  const { status, activity, uploads, pending, requestError } = useSelector(arkitektSlice.getArkitektState);
  const [url, setUrl] = useState(null); // null = show the configured url
  const [appName, setAppName] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [confirmUnbind, setConfirmUnbind] = useState(false);
  const [, setTick] = useState(0);

  const state = status?.state;
  const busy = state === "connecting" || state === "awaiting_login";

  const refresh = useCallback(async () => {
    try {
      dispatch(arkitektSlice.setStatus(await apiArkitektGetStatus()));
    } catch (error) {
      dispatch(arkitektSlice.setRequestError(t("The ImSwitch backend did not answer: {message}", { message: error.message })));
    }
  }, [dispatch, t]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  // uploads arrive by socket; reload them when the count says some were missed
  const uploadCount = status?.uploadCount;
  useEffect(() => {
    if (uploadCount === undefined) return;
    apiArkitektGetUploads()
      .then((list) => dispatch(arkitektSlice.setUploads(list)))
      .catch(() => {});
  }, [uploadCount, dispatch]);

  // the login timer
  useEffect(() => {
    if (state !== "awaiting_login") return undefined;
    const id = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [state]);

  const run = async (name, request) => {
    dispatch(arkitektSlice.setPending(name));
    dispatch(arkitektSlice.setRequestError(null));
    try {
      const result = await request();
      if (result) dispatch(arkitektSlice.setStatus(result));
      if (result?.status === "error" || result?.status === "warning") {
        dispatch(arkitektSlice.setRequestError(result.message));
      }
    } catch (error) {
      dispatch(arkitektSlice.setRequestError(error.message));
    } finally {
      dispatch(arkitektSlice.setPending(null));
    }
  };

  const bind = () => run("bind", () => apiArkitektBind((url ?? status?.url ?? "").trim()));
  const cancel = () => run("cancel", apiArkitektCancel);
  const unbind = () => {
    setConfirmUnbind(false);
    run("unbind", apiArkitektUnbind);
  };
  const saveSetting = (settings) => run("settings", () => apiArkitektSetSettings(settings));
  const clearLog = async () => {
    await apiArkitektClearActivity();
    dispatch(arkitektSlice.clearActivity());
  };

  if (!status) {
    return (
      <Box sx={{ p: 2 }}>
        {requestError ? <Alert severity="error">{requestError}</Alert> : <LinearProgress />}
      </Box>
    );
  }

  const [stateLabel, stateColor] = STATE_LABELS[state] || [state, "text.secondary"];
  const shownUrl = url ?? status.url ?? "";
  const storedLogin = status.hasStoredLogin;

  // ── connection ────────────────────────────────────────────────────────────
  const connection = (
    <Section title={t("Connection")}>
      {state === "unavailable" && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {t("The arkitekt Python package is not installed in this ImSwitch environment. Install it and restart ImSwitch:")}
          <Box component="pre" sx={{ m: 0, mt: 1, fontFamily: "monospace", whiteSpace: "pre-wrap" }}>{INSTALL_COMMAND}</Box>
        </Alert>
      )}
      {state === "disabled" && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {t('Arkitekt is disabled in the setup file ("arkitekt": {"enabled": false}).')}
        </Alert>
      )}

      <Box sx={{ display: "flex", gap: 1, alignItems: "flex-start", flexWrap: "wrap" }}>
        <TextField
          size="small"
          label={t("Arkitekt server")}
          placeholder="https://go.arkitekt.live"
          value={shownUrl}
          onChange={(e) => setUrl(e.target.value)}
          disabled={busy || state === "connected"}
          sx={{ flex: 1, minWidth: 260 }}
          helperText={status.insecureUrl && !status.allowInsecureTransport
            ? t("Plain http to a network host: allow the insecure login under Settings, or use https.")
            : t("The public server, or a local one, e.g. http://nas.local")}
          error={status.insecureUrl && !status.allowInsecureTransport}
        />
        {state === "connected" || busy ? (
          <Button variant="outlined" onClick={cancel} disabled={pending === "cancel"}>
            {state === "connected" ? t("Disconnect") : t("Cancel")}
          </Button>
        ) : (
          <Button
            variant="contained"
            onClick={bind}
            disabled={!status.available || !status.enabled || pending === "bind" || !shownUrl.trim()}
          >
            {storedLogin ? t("Connect") : t("Bind microscope")}
          </Button>
        )}
        <Button
          color="error"
          onClick={() => setConfirmUnbind(true)}
          disabled={!status.available || pending === "unbind" || (!storedLogin && state !== "connected" && !busy)}
        >
          {t("Unbind")}
        </Button>
      </Box>

      {(busy || state === "connected") && (
        <Stepper activeStep={ACTIVE_STEP[state] ?? 0} sx={{ my: 2 }} alternativeLabel>
          {STEPS.map((label) => (
            <Step key={label} completed={state === "connected" ? true : undefined}>
              <StepLabel>{t(label)}</StepLabel>
            </Step>
          ))}
        </Stepper>
      )}
      {state === "connecting" && <LinearProgress sx={{ mb: 1 }} />}

      {state === "awaiting_login" && status.userCode && (
        <Paper variant="outlined" sx={{ p: 2, mt: 1, borderColor: "warning.main" }}>
          <Typography variant="body2" gutterBottom>
            {t("Open the approval page, sign in to {server} and approve this microscope. The code must match:", {
              server: status.serverName || status.url,
            })}
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, my: 1.5, flexWrap: "wrap" }}>
            <Typography
              component="span"
              sx={{ fontFamily: "monospace", fontSize: "2rem", letterSpacing: "0.25em", fontWeight: 600 }}
            >
              {status.userCode}
            </Typography>
            <Tooltip title={t("Copy code")}>
              <IconButton size="small" onClick={() => copy(status.userCode)}><ContentCopyIcon fontSize="small" /></IconButton>
            </Tooltip>
          </Box>
          <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
            <Button
              variant="contained"
              color="warning"
              href={status.approveUrl}
              target="_blank"
              rel="noopener noreferrer"
              endIcon={<OpenInNewIcon />}
            >
              {t("Open approval page")}
            </Button>
            <Typography variant="caption" color="text.secondary">
              {t("Waiting {elapsed}", { elapsed: elapsed(status.loginStartedAt) })}
            </Typography>
          </Box>
          <Link href={status.approveUrl} target="_blank" rel="noopener noreferrer" variant="caption"
            sx={{ display: "block", mt: 1, fontFamily: "monospace", wordBreak: "break-all" }}>
            {status.approveUrl}
          </Link>
        </Paper>
      )}

      {state === "error" && status.message && <Alert severity="error" sx={{ mt: 2 }}>{status.message}</Alert>}
      {state !== "error" && state !== "awaiting_login" && status.message && state !== "unavailable" && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>{status.message}</Typography>
      )}
      {requestError && requestError !== status.message && (
        <Alert severity="warning" sx={{ mt: 2 }} onClose={() => dispatch(arkitektSlice.setRequestError(null))}>
          {requestError}
        </Alert>
      )}

      <Box sx={{ mt: 2 }}>
        <Facts rows={[
          [t("App"), `${status.appName} ${status.appVersion || ""}`],
          state === "connected" && [t("Bound since"), time(status.boundSince)],
          state === "connected" && status.deviceId && [t("Device id"), status.deviceId],
          status.services?.length > 0 && [t("Services"), status.services.join(", ")],
          [t("Stored login"), storedLogin ? t("yes, reconnects without approval") : t("none")],
        ]} />
      </Box>

      <Button size="small" sx={{ mt: 1.5, px: 0 }} onClick={() => setShowSettings((v) => !v)}
        endIcon={showSettings ? <ExpandLessIcon /> : <ExpandMoreIcon />}>
        {t("Settings")}
      </Button>
      <Collapse in={showSettings}>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mt: 1 }}>
          <FormControlLabel
            control={<Switch size="small" checked={!!status.autoConnect}
              onChange={(e) => saveSetting({ autoConnect: e.target.checked })} />}
            label={<Typography variant="body2">{t("Reconnect at startup (only with a stored login)")}</Typography>}
          />
          <FormControlLabel
            control={<Switch size="small" checked={!!status.useMikro}
              onChange={(e) => saveSetting({ useMikro: e.target.checked })} />}
            label={<Typography variant="body2">{t("Offer image actions (stores images in the server's mikro)")}</Typography>}
          />
          <FormControlLabel
            control={<Switch size="small" color="warning" checked={!!status.allowInsecureTransport}
              onChange={(e) => saveSetting({ allowInsecureTransport: e.target.checked })} />}
            label={<Typography variant="body2">{t("Allow insecure (http) login to a network server")}</Typography>}
          />
          {status.allowInsecureTransport && (
            <Typography variant="caption" color="warning.main">
              {t("The session token travels unencrypted: use this only on a trusted local network, e.g. for a NAS without TLS.")}
            </Typography>
          )}
          <TextField
            size="small"
            label={t("App name")}
            value={appName ?? status.appName ?? ""}
            onChange={(e) => setAppName(e.target.value)}
            onBlur={() => appName !== null && appName !== status.appName && saveSetting({ appName })}
            disabled={busy || state === "connected"}
            helperText={t("Another name is another registration and another login.")}
            sx={{ mt: 1, maxWidth: 320 }}
          />
          <Typography variant="caption" color="text.secondary">
            {t("Settings are saved to the setup file and apply at the next bind.")}
          </Typography>
        </Box>
      </Collapse>
    </Section>
  );

  // ── what the server sees ──────────────────────────────────────────────────
  const live = status.publishedState;
  const offered = (
    <Section title={t("Offered to Arkitekt")}>
      {state === "connected" && live && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary">{t("Live state, as published")}</Typography>
          <Facts rows={[
            [t("Stage"), `X ${live.x_um.toFixed(1)} · Y ${live.y_um.toFixed(1)} · Z ${live.z_um.toFixed(1)} µm`],
            [t("Light on"), live.illumination_on || "–"],
            [t("Exposure"), live.exposure_ms ? `${live.exposure_ms} ms` : "–"],
            [t("Running"), live.running_action || "–"],
          ]} />
        </Box>
      )}
      {(status.actions || []).map((action) => (
        <Box key={action.name} sx={{ py: 0.75, borderTop: 1, borderColor: "divider" }}>
          <Box sx={{ display: "flex", gap: 1, alignItems: "baseline", flexWrap: "wrap" }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>{action.title}</Typography>
            {action.moves && <Typography variant="caption" color="warning.main">{t("moves hardware")}</Typography>}
            {action.images && <Typography variant="caption" color="info.main">{t("stores images")}</Typography>}
          </Box>
          {action.description && (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>{action.description}</Typography>
          )}
        </Box>
      ))}
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
        {t("Remote calls are refused while an experiment, a recording or a workflow runs in ImSwitch.")}
      </Typography>
    </Section>
  );

  // ── remote calls ──────────────────────────────────────────────────────────
  const log = (
    <Section
      title={t("Remote calls")}
      action={activity.length > 0 && <Button size="small" onClick={clearLog}>{t("Clear")}</Button>}
    >
      {activity.length === 0 ? (
        <Typography variant="body2" color="text.secondary">{t("No calls from Arkitekt yet.")}</Typography>
      ) : (
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>{t("Time")}</TableCell>
                <TableCell>{t("Action")}</TableCell>
                <TableCell>{t("Arguments")}</TableCell>
                <TableCell>{t("Result")}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {activity.map((entry) => (
                <TableRow key={entry.id}>
                  <TableCell sx={{ whiteSpace: "nowrap", fontFamily: "monospace" }}>{time(entry.startedAt)}</TableCell>
                  <TableCell>{entry.action}</TableCell>
                  <TableCell sx={{ fontFamily: "monospace", fontSize: "0.75rem" }}>
                    {Object.entries(entry.arguments || {}).map(([k, v]) => `${k}=${v}`).join("  ") || "–"}
                  </TableCell>
                  <TableCell sx={{ minWidth: 140 }}>
                    {entry.status === "running" ? (
                      <Box>
                        <Typography variant="caption">
                          {entry.results ? t("running · {n} images", { n: entry.results }) : t("running")}
                        </Typography>
                        <LinearProgress />
                      </Box>
                    ) : (
                      <Typography
                        variant="caption"
                        color={entry.status === "failed" ? "error.main" : entry.status === "cancelled" ? "warning.main" : "text.primary"}
                      >
                        {entry.status === "failed" ? entry.error : t(entry.status)}
                        {entry.results ? ` · ${entry.results}` : ""}
                        {entry.durationS != null ? ` · ${entry.durationS} s` : ""}
                      </Typography>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      )}
    </Section>
  );

  // ── images sent ───────────────────────────────────────────────────────────
  const gallery = uploads.length > 0 && (
    <Section title={t("Sent to Arkitekt")}>
      <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 1.5 }}>
        {uploads.map((upload) => (
          <Box key={upload.id}>
            <Box sx={{ aspectRatio: "4 / 3", bgcolor: "action.hover", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
              {upload.thumbnail ? (
                <img src={upload.thumbnail} alt={upload.name} style={{ width: "100%", height: "100%", objectFit: "contain" }} />
              ) : (
                <Typography variant="caption" color="text.secondary">{t("no preview")}</Typography>
              )}
            </Box>
            <Typography variant="caption" sx={{ display: "block", fontWeight: 600, mt: 0.5 }} noWrap title={upload.name}>
              {upload.name}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", fontFamily: "monospace" }}>
              {`${upload.positionUm?.x?.toFixed(0)}, ${upload.positionUm?.y?.toFixed(0)} µm · ${(upload.shape || []).join("×")}`}
            </Typography>
          </Box>
        ))}
      </Box>
    </Section>
  );

  return (
    <Box sx={{ p: 2, maxWidth: 1400 }}>
      <Box sx={{ display: "flex", alignItems: "baseline", gap: 2, mb: 2, flexWrap: "wrap" }}>
        <Typography variant="h5">Arkitekt</Typography>
        <Typography variant="body2" sx={{ color: stateColor, fontWeight: 600 }}>
          <Box component="span" sx={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", bgcolor: stateColor, mr: 1, verticalAlign: "middle" }} />
          {t(stateLabel)}
          {state === "connected" && ` · ${status.serverName || status.url}`}
        </Typography>
      </Box>

      <Grid container spacing={2}>
        <Grid item xs={12} md={7}>{connection}</Grid>
        <Grid item xs={12} md={5}>{offered}</Grid>
        <Grid item xs={12}>{log}</Grid>
        {gallery && <Grid item xs={12}>{gallery}</Grid>}
      </Grid>

      <Dialog open={confirmUnbind} onClose={() => setConfirmUnbind(false)}>
        <DialogTitle>{t("Unbind this microscope?")}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {t("This disconnects from {server} and removes the stored login on this microscope, so binding again needs a new approval. The server keeps the app's registration until it is removed in the Arkitekt web interface.", {
              server: status.serverName || status.url,
            })}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmUnbind(false)}>{t("Keep")}</Button>
          <Button color="error" onClick={unbind}>{t("Unbind")}</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ArkitektController;
