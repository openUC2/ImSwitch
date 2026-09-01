// src/components/FreeNumberField.js
// A free-typing numeric TextField: stores the literal string the user types
// (incl. empty / partially-typed values like "" or "0.") so the input never
// snaps back to a default mid-edit. Commits a parsed number on blur or Enter.
import React, { useState, useEffect, useCallback, useRef } from "react";
import { TextField, Tooltip, Box } from "@mui/material";

const FreeNumberField = ({
  label,
  value,
  onCommit,
  unitFactor = 1, // value-in-state = displayed-value * unitFactor
  fixedDecimals = null,
  helperText,
  tooltip,
  step,
  min,
  max,
  fullWidth = true,
  ...textFieldProps
}) => {
  const formatDisplay = useCallback(
    (v) => {
      if (v === null || v === undefined || Number.isNaN(v)) return "";
      const display = v / unitFactor;
      if (fixedDecimals !== null) {
        return Number(display).toFixed(fixedDecimals);
      }
      return String(display);
    },
    [unitFactor, fixedDecimals]
  );

  const [draft, setDraft] = useState(() => formatDisplay(value));
  const editingRef = useRef(false);

  // Sync from outside (e.g. backend pushed new value) ONLY while the user
  // is not editing — prevents the field from jumping while typing.
  useEffect(() => {
    if (!editingRef.current) {
      setDraft(formatDisplay(value));
    }
  }, [value, formatDisplay]);

  const handleChange = (e) => {
    editingRef.current = true;
    // Accept any string; don't reject empty / "-" / "." mid-edit.
    setDraft(e.target.value);
  };

  const commit = () => {
    editingRef.current = false;
    const trimmed = draft.trim();
    if (trimmed === "" || trimmed === "-" || trimmed === ".") {
      // Nothing meaningful entered → restore last known value.
      setDraft(formatDisplay(value));
      return;
    }
    const parsed = Number(trimmed);
    if (Number.isNaN(parsed)) {
      setDraft(formatDisplay(value));
      return;
    }
    let next = parsed * unitFactor;
    if (min !== undefined && next < min) next = min;
    if (max !== undefined && next > max) next = max;
    setDraft(formatDisplay(next));
    if (next !== value) onCommit(next);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.currentTarget.blur();
    }
  };

  const field = (
    <TextField
      label={label}
      type="text"
      inputProps={{ inputMode: "decimal", step, min, max }}
      value={draft}
      onFocus={() => {
        editingRef.current = true;
      }}
      onChange={handleChange}
      onBlur={commit}
      onKeyDown={handleKeyDown}
      helperText={helperText}
      fullWidth={fullWidth}
      size="small"
      {...textFieldProps}
    />
  );

  if (tooltip) {
    return (
      <Tooltip title={tooltip} arrow placement="top-start">
        <Box>{field}</Box>
      </Tooltip>
    );
  }
  return field;
};

export default FreeNumberField;
