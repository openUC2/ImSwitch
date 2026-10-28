// src/components/objective/ObjectiveSlotCard.jsx
// Configuration of one turret slot: its optics (name, magnification, NA,
// pixel size) and the two stored positions that make switching work — the
// turret A position that puts it in the beam and its par-focal Z.
import React, { useState } from "react";
import {
  Box,
  Button,
  Chip,
  Divider,
  IconButton,
  Paper,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import EditRoundedIcon from "@mui/icons-material/EditRounded";
import CheckRoundedIcon from "@mui/icons-material/CheckRounded";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import MyLocationRoundedIcon from "@mui/icons-material/MyLocationRounded";
import { fmt } from "./objectiveSlots";

const Stat = ({ label, value }) => (
  <Box sx={{ minWidth: 0 }}>
    <Typography
      variant="caption"
      sx={{ display: "block", color: "text.secondary", textTransform: "uppercase", letterSpacing: "0.05em" }}
    >
      {label}
    </Typography>
    <Typography variant="body1" sx={{ fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
      {value}
    </Typography>
  </Box>
);

const SectionLabel = ({ children }) => (
  <Typography variant="overline" sx={{ color: "text.secondary", lineHeight: 1.6 }}>
    {children}
  </Typography>
);

/** A stored position: value, "use current" and inline edit. */
function PositionRow({ label, help, value, onUseCurrent, onSave, disabled }) {
  const [draft, setDraft] = useState(null); // null = not editing

  if (draft !== null) {
    const parsed = Number(draft);
    const valid = draft.trim() !== "" && Number.isFinite(parsed);
    return (
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <TextField
          label={label}
          type="number"
          size="small"
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && valid) {
              onSave(parsed);
              setDraft(null);
            } else if (e.key === "Escape") setDraft(null);
          }}
          InputProps={{ endAdornment: <Typography variant="caption">µm</Typography> }}
          sx={{ flex: 1 }}
        />
        <IconButton
          color="primary"
          aria-label={`Save ${label}`}
          disabled={!valid}
          onClick={() => {
            onSave(parsed);
            setDraft(null);
          }}
        >
          <CheckRoundedIcon />
        </IconButton>
        <IconButton aria-label="Cancel" onClick={() => setDraft(null)}>
          <CloseRoundedIcon />
        </IconButton>
      </Box>
    );
  }

  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <Tooltip title={help} enterDelay={500}>
        <Box sx={{ flex: "1 1 140px", minWidth: 0 }}>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            {label}
          </Typography>
          <Typography variant="body1" sx={{ fontFamily: "monospace", fontWeight: 600 }}>
            {fmt(value)} µm
          </Typography>
        </Box>
      </Tooltip>
      <Button
        size="small"
        variant="outlined"
        startIcon={<MyLocationRoundedIcon />}
        onClick={onUseCurrent}
        disabled={disabled}
      >
        Use current
      </Button>
      <IconButton
        size="small"
        aria-label={`Edit ${label}`}
        onClick={() => setDraft(Number.isFinite(value) ? String(value) : "")}
        disabled={disabled}
      >
        <EditRoundedIcon fontSize="small" />
      </IconButton>
    </Box>
  );
}

export default function ObjectiveSlotCard({
  slot,
  inBeam,
  hasMotor,
  onSaveOptics,
  onStorePosition,
  onUseCurrent,
}) {
  const [optics, setOptics] = useState(null); // null = not editing
  const [saving, setSaving] = useState(false);

  const startEditing = () =>
    setOptics({
      name: slot.name,
      magnification: slot.magnification > 0 ? String(slot.magnification) : "",
      NA: slot.na > 0 ? String(slot.na) : "",
      pixelsize: slot.pixelSize > 0 ? String(slot.pixelSize) : "",
    });

  const save = async () => {
    setSaving(true);
    const ok = await onSaveOptics(optics);
    setSaving(false);
    if (ok) setOptics(null);
  };

  const field = (key, label, extra = {}) => (
    <TextField
      label={label}
      size="small"
      value={optics[key]}
      onChange={(e) => setOptics((o) => ({ ...o, [key]: e.target.value }))}
      fullWidth
      {...extra}
    />
  );

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 2,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: 1,
        borderRadius: 2,
        borderWidth: inBeam ? 2 : 1,
        borderColor: inBeam ? "primary.main" : "divider",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {slot.label}
        </Typography>
        <Typography variant="body1" sx={{ color: "text.secondary", flex: 1, minWidth: 0 }} noWrap>
          {slot.name}
        </Typography>
        {inBeam && <Chip size="small" color="primary" label="In beam" />}
        {!slot.configured && (
          <Chip size="small" color="warning" variant="outlined" label="Not configured" />
        )}
      </Box>

      {/* optics */}
      <Box sx={{ display: "flex", alignItems: "center" }}>
        <SectionLabel>Optics</SectionLabel>
        <Box sx={{ flex: 1 }} />
        {optics === null && (
          <Button size="small" startIcon={<EditRoundedIcon />} onClick={startEditing}>
            {slot.configured ? "Edit" : "Set up"}
          </Button>
        )}
      </Box>
      {optics === null ? (
        <Box sx={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 1 }}>
          <Stat label="Magnification" value={slot.magnification > 0 ? `${fmt(slot.magnification, 0)}×` : "—"} />
          <Stat label="NA" value={slot.na > 0 ? fmt(slot.na, 2) : "—"} />
          <Stat label="Pixel size" value={slot.pixelSize > 0 ? `${fmt(slot.pixelSize, 3)} µm` : "—"} />
        </Box>
      ) : (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.25, pt: 0.5 }}>
          {field("name", "Name")}
          <Box sx={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 1 }}>
            {field("magnification", "Magnification", {
              type: "number",
              inputProps: { min: 1, step: 1, "data-keypad-presets": "4,10,20,40,60,100" },
            })}
            {field("NA", "NA", { type: "number", inputProps: { min: 0, max: 1.7, step: 0.05 } })}
            {field("pixelsize", "Pixel size", {
              type: "number",
              inputProps: { min: 0, step: 0.01 },
              InputProps: { endAdornment: <Typography variant="caption">µm</Typography> },
            })}
          </Box>
          <Typography variant="caption" sx={{ color: "text.secondary" }}>
            The pixel size drives stitching, tiling and click-to-move — keep it in
            sync with the mounted objective.
          </Typography>
          <Box sx={{ display: "flex", gap: 1, justifyContent: "flex-end" }}>
            <Button onClick={() => setOptics(null)} disabled={saving}>
              Cancel
            </Button>
            <Button variant="contained" onClick={save} disabled={saving}>
              {saving ? "Saving…" : "Save optics"}
            </Button>
          </Box>
        </Box>
      )}

      <Divider sx={{ my: 0.5 }} />

      {/* stored positions */}
      <SectionLabel>Stored positions</SectionLabel>
      {hasMotor && (
        <PositionRow
          label="Turret position (A)"
          help="Revolver A position that puts this objective under the optical axis."
          value={slot.turretPosition}
          onUseCurrent={() => onUseCurrent("a")}
          onSave={(v) => onStorePosition("a", v)}
        />
      )}
      <PositionRow
        label="Par-focal focus (Z)"
        help="Z at which this objective is in focus. With Z leveling on, switching moves Z by the difference between the two stored foci."
        value={slot.focusPosition}
        onUseCurrent={() => onUseCurrent("z")}
        onSave={(v) => onStorePosition("z", v)}
      />
    </Paper>
  );
}
