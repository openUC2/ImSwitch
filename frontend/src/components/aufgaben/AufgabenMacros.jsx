// src/components/aufgaben/AufgabenMacros.jsx
// Live View macro buttons: one per saved Aufgaben job marked as a macro
// (metadata.macro, the "Macro button in Live View" switch in the job panel).
// The arrow next to a button opens its run parameters. A running macro's
// button stops it. "From template" copies a bundled macro (autofocus, shutter,
// ...) under a name the operator picks; adapt the copy in the Aufgaben app.
import React, { useCallback, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box, Button, ButtonGroup, Menu, MenuItem, ListItemText, Popover, Stack, Typography } from "@mui/material";

import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import PlayArrowOutlined from "@mui/icons-material/PlayArrowOutlined";
import StopOutlined from "@mui/icons-material/StopOutlined";
import AddOutlined from "@mui/icons-material/AddOutlined";

import {
  apiAufgabenAbortJob,
  apiAufgabenGetJob,
  apiAufgabenGetStatus,
  apiAufgabenListMacroTemplates,
  apiAufgabenListMacros,
  apiAufgabenRunMacro,
  apiAufgabenSaveJob,
} from "../../backendapi/apiAufgabenController";
import { selectHasController } from "../../state/slices/BackendCapabilitiesSlice";
import { setNotification } from "../../state/slices/NotificationSlice";
import { aufgabenErrorText } from "../../utils/aufgabenJob";
import { ParameterInput, runtimeParameters } from "./AufgabenRun";
import { useT } from "../../i18n";

const ACTIVE = ["running", "paused"];
const POLL_MS = 400;

const AufgabenMacros = ({ header = null, sx }) => {
  const t = useT();
  const dispatch = useDispatch();
  const available = useSelector(selectHasController("AufgabenController"));
  const [macros, setMacros] = useState(null); // null until the list arrived
  const [values, setValues] = useState({}); // macro name -> parameter values the operator set
  const [running, setRunning] = useState(null); // { name, title, runId }
  const [menu, setMenu] = useState(null); // { name, anchor }

  const notify = useCallback(
    (message, type = "error") => dispatch(setNotification({ message, type, autoHideDuration: 5000 })),
    [dispatch]
  );

  const [templates, setTemplates] = useState([]);
  const [templateMenu, setTemplateMenu] = useState(null); // anchor element

  const loadMacros = useCallback(
    () =>
      apiAufgabenListMacros()
        .then((list) => setMacros(Array.isArray(list) ? list : []))
        .catch(() => setMacros([])),
    []
  );
  useEffect(() => {
    if (!available) return;
    loadMacros();
    apiAufgabenListMacroTemplates()
      .then((list) => setTemplates(Array.isArray(list) ? list : []))
      .catch(() => setTemplates([]));
  }, [available, loadMacros]);

  // Copy a bundled template under the operator's name; it then shows as a button.
  const fromTemplate = async (template) => {
    setTemplateMenu(null);
    const title = window.prompt(t("Name of your macro"), template.title)?.trim();
    if (!title) return;
    if (macros.some((m) => m.title === title) && !window.confirm(t("A macro with this name exists. Replace it?"))) return;
    try {
      const job = await apiAufgabenGetJob(template.name);
      await apiAufgabenSaveJob({ ...job, name: title, metadata: { ...job.metadata, macro: true } });
      await loadMacros();
      notify(t("Saved. Adapt its steps in the Aufgaben app."), "success");
    } catch (e) {
      notify(`${title}: ${aufgabenErrorText(e)}`);
    }
  };
  const templateButton = templates.length > 0 && (
    <>
      <Button
        size="small"
        startIcon={<AddOutlined />}
        onClick={(e) => setTemplateMenu(e.currentTarget)}
        sx={{ textTransform: "none" }}
      >
        {t("From template")}
      </Button>
      <Menu anchorEl={templateMenu} open={!!templateMenu} onClose={() => setTemplateMenu(null)}>
        {templates.map((tpl) => (
          <MenuItem key={tpl.name} onClick={() => fromTemplate(tpl)} sx={{ maxWidth: 360 }}>
            <ListItemText
              primary={tpl.title}
              secondary={tpl.description}
              secondaryTypographyProps={{ sx: { whiteSpace: "normal" } }}
            />
          </MenuItem>
        ))}
      </Menu>
    </>
  );

  // Follow the run until it ends; say so when it did not finish cleanly.
  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(async () => {
      const status = await apiAufgabenGetStatus().catch(() => null);
      if (!status || (status.run_id === running.runId && ACTIVE.includes(status.state))) return;
      setRunning(null);
      if (status.run_id === running.runId && status.state !== "finished") {
        notify(`${running.title}: ${status.state}${status.message ? ` (${status.message})` : ""}`, "warning");
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [running, notify]);

  const run = async (macro, confirmed = false) => {
    setMenu(null);
    try {
      const result = await apiAufgabenRunMacro(macro.name, values[macro.name] || {}, confirmed);
      setRunning({ name: macro.name, title: macro.title, runId: result.run_id });
    } catch (e) {
      const text = `${macro.title}: ${aufgabenErrorText(e)}`;
      if (e?.response?.status === 403 && !confirmed) {
        if (window.confirm(`${text}\n\n${t("Run anyway?")}`)) run(macro, true);
      } else {
        notify(text);
      }
    }
  };

  const stop = () => apiAufgabenAbortJob(false).catch((e) => notify(aufgabenErrorText(e)));

  if (!available || macros === null) return null;
  if (macros.length === 0) {
    return (
      <Box sx={sx}>
        {header}
        <Typography variant="body2" color="text.secondary">
          {t("No macros yet. Start from a template, or in Aufgaben switch on \"Macro button in Live View\" in a job's Job tab and save it; it shows here as a button.")}
        </Typography>
        {templateButton}
      </Box>
    );
  }
  const open = menu ? macros.find((m) => m.name === menu.name) : null;
  const setValue = (macro, name) => (value) =>
    setValues((all) => ({ ...all, [macro]: { ...all[macro], [name]: value } }));
  return (
    <Box sx={sx}>
      {header}
      <Stack direction="row" flexWrap="wrap" gap={1}>
        {macros.map((macro) => {
          const isRunning = running?.name === macro.name;
          return (
            <ButtonGroup
              key={macro.name}
              size="small"
              variant={isRunning ? "contained" : "outlined"}
              disabled={!!running && !isRunning}
            >
              <Button
                title={macro.description || undefined}
                startIcon={isRunning ? <StopOutlined /> : <PlayArrowOutlined />}
                onClick={() => (isRunning ? stop() : run(macro))}
                sx={{ textTransform: "none" }}
              >
                {macro.title}
              </Button>
              {runtimeParameters(macro).length > 0 && (
                <Button
                  aria-label={t("Parameters")}
                  disabled={isRunning}
                  onClick={(e) => setMenu({ name: macro.name, anchor: e.currentTarget })}
                  sx={{ px: 0.25, minWidth: 0 }}
                >
                  <ArrowDropDownIcon fontSize="small" />
                </Button>
              )}
            </ButtonGroup>
          );
        })}
        {templateButton}
      </Stack>
      <Popover
        open={!!open}
        anchorEl={menu?.anchor}
        onClose={() => setMenu(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
      >
        {open && (
          <Stack spacing={1.5} sx={{ p: 2, width: 260 }}>
            {runtimeParameters(open).map(([name, spec]) => (
              <ParameterInput
                key={name}
                name={name}
                spec={spec}
                value={values[open.name]?.[name] ?? spec.default}
                onChange={setValue(open.name, name)}
              />
            ))}
            <Button variant="contained" size="small" startIcon={<PlayArrowOutlined />} onClick={() => run(open)}>
              {t("Run")}
            </Button>
          </Stack>
        )}
      </Popover>
    </Box>
  );
};

export default AufgabenMacros;
