// src/components/ObjectiveSwitcher.js
// Compact objective switcher for the Live View panel/dock. Configuration
// lives on the Objective page; switching logic is shared with it through
// objective/useObjectiveActions.
import React, { useEffect, useState } from "react";
import { Box, Paper, Typography } from "@mui/material";
import ObjectiveSlotButtons from "./objective/ObjectiveSlotButtons.jsx";
import useObjectiveActions from "./objective/useObjectiveActions.js";
import { getSlots, fmt } from "./objective/objectiveSlots.js";

export default function ObjectiveSwitcher({ hostIP, hostPort }) {
  const { objectiveState: obj, pendingSlot, switchTo, refresh } =
    useObjectiveActions();
  const [withZ, setWithZ] = useState(true);

  useEffect(() => {
    refresh();
  }, [hostIP, hostPort, refresh]);

  const current =
    obj.currentObjective === 0 || obj.currentObjective === 1
      ? obj.currentObjective
      : null;

  const details = [
    obj.objectivName,
    obj.magnification ? `${fmt(obj.magnification, 0)}×` : null,
    obj.NA ? `NA ${fmt(obj.NA, 2)}` : null,
    obj.pixelsize ? `${fmt(obj.pixelsize, 3)} µm/px` : null,
  ].filter(Boolean);

  return (
    <Paper variant="outlined" sx={{ p: 1.5 }}>
      <Box sx={{ display: "flex", alignItems: "baseline", gap: 1, mb: 1, flexWrap: "wrap" }}>
        <Typography variant="body2" sx={{ color: "text.secondary" }}>
          In beam:
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          {pendingSlot !== null
            ? "switching…"
            : details.length > 0
              ? details.join(" · ")
              : "unknown"}
        </Typography>
      </Box>
      <ObjectiveSlotButtons
        compact
        slots={getSlots(obj)}
        currentSlot={current}
        pendingSlot={pendingSlot}
        onSelect={(slot) => switchTo(slot, { withZ })}
        withZ={withZ}
        onWithZChange={setWithZ}
      />
    </Paper>
  );
}
