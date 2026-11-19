// One "System" page for everything about the instrument itself: backend
// connection, ImSwitch service and host, setup file, board firmware, network
// and logs. These used to be six look-alike entries in the settings menu. Each
// tab is still its own page key (selectedPlugin), so existing navigation to
// e.g. "SystemUpdate" or "Connections" lands on the matching tab.
import React from "react";
import { useSelector } from "react-redux";
import { Box, Tab, Tabs } from "@mui/material";

import ConnectionSettings from "./ConnectionSettings.jsx";
import SystemSettings from "./SystemSettings.js";
import UC2ConfigurationController from "./UC2ConfigurationController.jsx";
import SystemUpdateController from "./SystemUpdateController.jsx";
import WiFiController from "./WiFiController.jsx";
import LoggingController from "./LoggingController.jsx";
import * as uc2Slice from "../state/slices/UC2Slice.js";

const PAGES = [
  { key: "Connections", label: "Connection", Page: ConnectionSettings },
  { key: "SystemSettings", label: "ImSwitch & host", Page: SystemSettings },
  { key: "UC2", label: "Setup file", Page: UC2ConfigurationController },
  { key: "SystemUpdate", label: "Board & firmware", Page: SystemUpdateController },
  { key: "WiFi", label: "WiFi", Page: WiFiController },
  { key: "Logging", label: "Logs", Page: LoggingController },
];

export const SYSTEM_PAGE_KEYS = PAGES.map((p) => p.key);

export default function SystemPage({ page, onNavigate }) {
  const { backendConnected } = useSelector(uc2Slice.getUc2State);
  // Same override the settings menu honours (see useDeveloperMode).
  const developerMode = localStorage.getItem("imswitch_developer_mode") === "true";
  const current = PAGES.find((p) => p.key === page) || PAGES[0];
  const { Page } = current;

  return (
    <Box>
      <Tabs
        value={current.key}
        onChange={(_, key) => onNavigate(key)}
        variant="scrollable"
        scrollButtons="auto"
        allowScrollButtonsMobile
        sx={{ borderBottom: 1, borderColor: "divider", mb: 1 }}
      >
        {PAGES.map((p) => (
          <Tab
            key={p.key}
            value={p.key}
            label={p.label}
            // Every page but the connection itself talks to the backend.
            disabled={p.key !== "Connections" && !backendConnected && !developerMode}
          />
        ))}
      </Tabs>
      <Page />
    </Box>
  );
}
