// One line under the jog pad: the enabled travel limits and, for a while, the
// last move the backend refused because it left them.
import React, { useEffect, useState } from "react";
import { Typography } from "@mui/material";
import useTravelLimits from "../utils/useTravelLimits";

const SHOW_REFUSAL_MS = 15000;
const um = (v) => (v === null || v === undefined ? "∞" : Math.round(v).toLocaleString());

export default function TravelLimitsNote() {
  const { limits, refusal } = useTravelLimits();
  const [, tick] = useState(0);
  useEffect(() => {
    if (!refusal) return undefined;
    const t = setTimeout(() => tick((n) => n + 1), SHOW_REFUSAL_MS);
    return () => clearTimeout(t);
  }, [refusal]);

  const enabled = Object.entries(limits).filter(([, l]) => l.enabled);
  const showRefusal = refusal && Date.now() - refusal.at < SHOW_REFUSAL_MS;
  if (!enabled.length && !showRefusal) return null;
  return (
    <>
      {enabled.length > 0 && (
        <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 1 }}>
          Travel limits:{" "}
          {enabled.map(([ax, l]) => `${ax} ${um(l.min)} … ${um(l.max)}`).join(" · ")} µm
        </Typography>
      )}
      {showRefusal && (
        <Typography variant="caption" color="error" component="div">
          {refusal.message}
        </Typography>
      )}
    </>
  );
}
