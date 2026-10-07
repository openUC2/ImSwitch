// src/components/objective/ObjectiveSlotButtons.jsx
// Two large "put this objective in the beam" buttons plus the keep-focus
// (Z leveling) switch. Shared by the Objective page and the Live View panel.
import React from "react";
import {
  Box,
  ButtonBase,
  CircularProgress,
  FormControlLabel,
  Switch,
  Typography,
} from "@mui/material";
import { alpha } from "@mui/material/styles";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import { fmt, opticsLabel } from "./objectiveSlots";

export default function ObjectiveSlotButtons({
  slots,
  currentSlot,
  pendingSlot,
  onSelect,
  withZ,
  onWithZChange,
  compact = false,
}) {
  const busy = pendingSlot !== null && pendingSlot !== undefined;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}>
        {slots.map((slot) => {
          const inBeam = currentSlot === slot.index && !busy;
          const pending = pendingSlot === slot.index;
          const disabled = !slot.configured || busy;
          return (
            <ButtonBase
              key={slot.index}
              disabled={disabled && !inBeam}
              onClick={() => !inBeam && onSelect(slot.index)}
              aria-pressed={inBeam}
              sx={(theme) => ({
                position: "relative",
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                textAlign: "left",
                gap: 0.25,
                p: compact ? 1 : 1.5,
                minHeight: compact ? 64 : 84,
                borderRadius: 2,
                border: 2,
                borderColor: inBeam ? "primary.main" : "divider",
                bgcolor: inBeam
                  ? alpha(theme.palette.primary.main, 0.14)
                  : "transparent",
                opacity: slot.configured ? 1 : 0.5,
                transition: "border-color 150ms, background-color 150ms",
                "&:hover": {
                  bgcolor: inBeam
                    ? alpha(theme.palette.primary.main, 0.18)
                    : alpha(theme.palette.text.primary, 0.05),
                },
              })}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, width: "100%" }}>
                <Typography
                  variant={compact ? "subtitle1" : "h6"}
                  sx={{ fontWeight: 700, lineHeight: 1.2, flex: 1, fontVariantNumeric: "tabular-nums" }}
                >
                  {opticsLabel(slot)}
                </Typography>
                {inBeam && <CheckCircleRoundedIcon color="primary" fontSize="small" />}
                {pending && <CircularProgress size={18} />}
              </Box>
              <Typography variant="caption" sx={{ color: "text.secondary", lineHeight: 1.3 }}>
                {slot.label}
                {slot.magnification > 0 && slot.name ? ` · ${slot.name}` : ""}
              </Typography>
              <Typography
                variant="caption"
                sx={{ fontWeight: 600, color: inBeam ? "primary.main" : "text.secondary" }}
              >
                {!slot.configured
                  ? "Not configured"
                  : pending
                    ? "Switching…"
                    : inBeam
                      ? "In beam"
                      : slot.pixelSize > 0
                        ? `${fmt(slot.pixelSize, 3)} µm/px · tap to switch`
                        : "Tap to switch"}
              </Typography>
            </ButtonBase>
          );
        })}
      </Box>
      <FormControlLabel
        control={
          <Switch
            checked={withZ}
            onChange={(e) => onWithZChange(e.target.checked)}
            size={compact ? "small" : "medium"}
          />
        }
        label={
          <Typography variant="body2">
            Keep focus (Z leveling)
            <Typography component="span" variant="caption" sx={{ color: "text.secondary", ml: 0.75 }}>
              {withZ ? "moves Z to the stored focus" : "Z stays where it is"}
            </Typography>
          </Typography>
        }
        sx={{ ml: 0 }}
      />
    </Box>
  );
}
