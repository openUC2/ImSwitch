// src/components/touch/TouchControlDock.jsx
// Tabbed control dock for the compact layouts. Sits beside the live image
// (landscape) or below it (portrait), so moving the stage or changing the
// light never covers the stream — the old Live View drawer did exactly that
// on phones.
//
//   sections: [{ key, label, icon, render: () => node }]
//   placement: "side" | "bottom"
import React, { useEffect, useState } from "react";
import { Box, Paper, Tab, Tabs } from "@mui/material";

export default function TouchControlDock({
  sections,
  placement = "side",
  width = 340,
  storageKey,
}) {
  const [active, setActive] = useState(() => {
    try {
      return (storageKey && localStorage.getItem(storageKey)) || sections[0]?.key;
    } catch {
      return sections[0]?.key;
    }
  });

  useEffect(() => {
    if (!storageKey || !active) return;
    try {
      localStorage.setItem(storageKey, active);
    } catch {
      // not remembered — fine
    }
  }, [active, storageKey]);

  const current = sections.find((s) => s.key === active) || sections[0];
  if (!current) return null;

  return (
    <Paper
      variant="outlined"
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        minWidth: 0,
        overflow: "hidden",
        ...(placement === "side"
          ? { width, flexShrink: 0, alignSelf: "stretch" }
          : { flex: 1, width: "100%" }),
      }}
    >
      <Tabs
        value={current.key}
        onChange={(_, v) => setActive(v)}
        // Up to six tabs share the width: a scrolling tab strip hides tabs
        // behind tiny arrows, which is hard to hit with a finger.
        variant={sections.length > 6 ? "scrollable" : "fullWidth"}
        scrollButtons="auto"
        allowScrollButtonsMobile
        sx={{
          flexShrink: 0,
          minHeight: 56,
          borderBottom: 1,
          borderColor: "divider",
          "& .MuiTab-root": {
            minHeight: 56,
            minWidth: 0,
            px: 0.25,
            py: 0.5,
            fontSize: "0.7rem",
            textTransform: "none",
          },
          "& .MuiTab-iconWrapper": { mb: 0.25 },
        }}
      >
        {sections.map((s) => (
          <Tab key={s.key} value={s.key} icon={s.icon} label={s.label} />
        ))}
      </Tabs>
      <Box
        sx={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          overflowX: "hidden",
          overscrollBehavior: "contain",
          p: 1.25,
        }}
      >
        {current.render()}
      </Box>
    </Paper>
  );
}
