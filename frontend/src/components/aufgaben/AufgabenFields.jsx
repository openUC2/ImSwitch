// src/components/aufgaben/AufgabenFields.jsx
// Form fields for Aufgaben parameters, generated from the catalogue schema
// (utils/aufgabenJob.paramFields). Numbers also take "${expression}" and
// "@resource" values; an empty field removes the parameter so the task's
// default applies.
import React, { useEffect, useState } from "react";
import {
  Box,
  Button,
  FormControlLabel,
  InputAdornment,
  MenuItem,
  Switch,
  TextField,
  Typography,
} from "@mui/material";

import { parseNumberInput, resourcesOfKind } from "../../utils/aufgabenJob";
import { useT } from "../../i18n";

const mono = { fontFamily: "monospace", fontSize: 13 };
const INLINE = "__inline__";

const show = (value) => (value === undefined || value === null ? "" : String(value));
const formatJson = (value) => (value === undefined ? "" : JSON.stringify(value, null, 1));

// Text that is committed on blur or Enter, so typing "1." does not jump.
export const CommitField = ({ value, onCommit, ...props }) => {
  const [text, setText] = useState(show(value));
  useEffect(() => setText(show(value)), [value]);
  return (
    <TextField
      size="small"
      fullWidth
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => text !== show(value) && onCommit(text)}
      onKeyDown={(e) => {
        if (e.key === "Enter" && !props.multiline) e.target.blur();
      }}
      {...props}
    />
  );
};

// JSON value: kept as text while it does not parse, committed when it does.
export const JsonField = ({ label, value, onChange, minRows = 3, helperText }) => {
  const t = useT();
  const [text, setText] = useState(formatJson(value));
  const [error, setError] = useState("");
  useEffect(() => {
    setText(formatJson(value));
    setError("");
  }, [value]);
  const commit = () => {
    if (text.trim() === "") return onChange(undefined);
    try {
      onChange(JSON.parse(text));
      setError("");
    } catch (e) {
      setError(t("Not valid JSON: {error}", { error: e.message }));
    }
  };
  return (
    <TextField
      size="small"
      fullWidth
      multiline
      minRows={minRows}
      maxRows={16}
      label={label}
      value={text}
      error={!!error}
      helperText={error || helperText}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      InputProps={{ sx: mono }}
    />
  );
};

const unitAdornment = (unit) =>
  unit ? { endAdornment: <InputAdornment position="end">{unit}</InputAdornment> } : undefined;

const label = (field) => `${field.title}${field.required ? " *" : ""}`;

// A resource reference ("@name") of the right kind, or the value inline.
const ResourceField = ({ field, value, job, onChange, onCreateResource }) => {
  const t = useT();
  const names = resourcesOfKind(job, field.resourceKinds);
  const ref = typeof value === "string" && value.startsWith("@") ? value.slice(1) : null;
  const selected = ref ?? (value === undefined ? "" : INLINE);
  return (
    <Box>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <TextField
          select
          size="small"
          fullWidth
          label={label(field)}
          value={selected}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "") onChange(undefined);
            else if (v === INLINE) onChange(ref ? job.resources[ref] : {});
            else onChange(`@${v}`);
          }}
          helperText={t("Resource of kind {kinds}", { kinds: field.resourceKinds.join(", ") })}
        >
          <MenuItem value="">
            <em>{field.required ? t("(choose)") : t("(none)")}</em>
          </MenuItem>
          {names.map((name) => (
            <MenuItem key={name} value={name}>
              @{name}
            </MenuItem>
          ))}
          <MenuItem value={INLINE}>{t("Inline value (JSON)")}</MenuItem>
        </TextField>
        {onCreateResource && (
          <Button
            size="small"
            sx={{ whiteSpace: "nowrap", mb: 2.5 }}
            aria-label={t("New {kind} resource", { kind: field.resourceKinds[0] })}
            onClick={() => onCreateResource(field.resourceKinds[0], field.name)}
          >
            {t("New")}
          </Button>
        )}
      </Box>
      {selected === INLINE && (
        <JsonField label={t("{name} (inline)", { name: field.title })} value={value} onChange={onChange} />
      )}
    </Box>
  );
};

export const ParamField = ({ field, value, job, onChange, onCreateResource }) => {
  const t = useT();
  const help = field.description || undefined;
  switch (field.kind) {
    case "boolean":
      if (typeof value === "string") break;   // an expression: edit as text below
      return (
        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={value ?? field.default ?? false}
              onChange={(e) => onChange(e.target.checked)}
            />
          }
          label={field.title}
        />
      );
    case "choice":
      return (
        <TextField
          select
          size="small"
          fullWidth
          label={label(field)}
          value={value ?? ""}
          helperText={help}
          onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
        >
          {!field.required && (
            <MenuItem value="">
              <em>{t("default")}{field.default != null ? ` (${field.default})` : ""}</em>
            </MenuItem>
          )}
          {field.choices.map((choice) => (
            <MenuItem key={String(choice)} value={choice}>
              {String(choice)}
            </MenuItem>
          ))}
        </TextField>
      );
    case "number":
    case "integer":
      return (
        <CommitField
          label={label(field)}
          value={value}
          placeholder={field.default != null ? String(field.default) : ""}
          helperText={help}
          InputProps={unitAdornment(field.unit)}
          onCommit={(text) => onChange(parseNumberInput(text, field.kind === "integer"))}
        />
      );
    case "expression":
      return (
        <CommitField
          label={label(field)}
          value={value}
          helperText={help || t("Expression, e.g. loop.tile.index % 4 == 0")}
          InputProps={{ sx: mono }}
          onCommit={(text) => onChange(text === "" ? undefined : text)}
        />
      );
    case "strings":
      return (
        <CommitField
          label={label(field)}
          value={Array.isArray(value) ? value.join(", ") : value}
          helperText={help || t("Comma-separated")}
          onCommit={(text) =>
            onChange(text.trim() === "" ? undefined : text.split(",").map((s) => s.trim()).filter(Boolean))
          }
        />
      );
    case "resource":
      return (
        <ResourceField
          field={field}
          value={value}
          job={job}
          onChange={onChange}
          onCreateResource={onCreateResource}
        />
      );
    case "json":
      return <JsonField label={label(field)} value={value} onChange={onChange} helperText={help} />;
    default:
      break;
  }
  return (
    <CommitField
      label={label(field)}
      value={value}
      helperText={help}
      InputProps={unitAdornment(field.unit)}
      onCommit={(text) => onChange(text === "" ? undefined : text)}
    />
  );
};

export const SectionTitle = ({ children }) => (
  <Typography
    variant="overline"
    color="text.secondary"
    sx={{ display: "block", mt: 2, mb: 0.5, lineHeight: 1.5 }}
  >
    {children}
  </Typography>
);
