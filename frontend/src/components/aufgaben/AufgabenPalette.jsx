// src/components/aufgaben/AufgabenPalette.jsx
// The steps this microscope can run (getAufgabenCatalog), by category; steps
// whose hardware is missing (available: false) are left out.
// Click adds a step at the selection; drag drops it anywhere in the outline.
import React, { useMemo, useState } from "react";
import { useDispatch } from "react-redux";
import {
  Box,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  ListSubheader,
  TextField,
  Tooltip,
} from "@mui/material";

import { addNode } from "../../state/slices/AufgabenSlice";
import { stepIcon } from "./aufgabenIcons";
import { useT } from "../../i18n";

export const DRAG_TASK = "application/x-aufgaben-task";

const AufgabenPalette = ({ catalog }) => {
  const t = useT();
  const dispatch = useDispatch();
  const [filter, setFilter] = useState("");
  const groups = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const out = {};
    for (const entry of catalog) {
      if (entry.available === false) continue;
      const text = `${entry.title} ${entry.id} ${entry.category}`.toLowerCase();
      if (needle && !text.includes(needle)) continue;
      (out[entry.category] ||= []).push(entry);
    }
    return out;
  }, [catalog, filter]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: 0, height: "100%" }}>
      <TextField
        size="small"
        placeholder={t("Find a step")}
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        sx={{ mb: 1 }}
      />
      <List dense disablePadding sx={{ overflowY: "auto", flex: 1 }}>
        {Object.entries(groups).map(([category, entries]) => (
          <li key={category}>
            <ul style={{ padding: 0 }}>
              <ListSubheader sx={{ lineHeight: "28px", fontSize: 12 }}>{t(category)}</ListSubheader>
              {entries.map((entry) => (
                <Tooltip
                  key={entry.id}
                  title={entry.description || entry.id}
                  placement="right"
                  enterDelay={600}
                >
                  <ListItemButton
                    draggable
                    onDragStart={(e) => e.dataTransfer.setData(DRAG_TASK, entry.id)}
                    onClick={() => dispatch(addNode(entry.id))}
                    sx={{ py: 0.25 }}
                  >
                    <ListItemIcon sx={{ minWidth: 28, color: "text.secondary" }}>
                      {React.createElement(stepIcon(entry.id, entry.category), { sx: { fontSize: 18 } })}
                    </ListItemIcon>
                    <ListItemText
                      primary={t(entry.title)}
                      secondary={entry.flags?.ask_first ? t("asks first") : null}
                      primaryTypographyProps={{ fontSize: 13 }}
                      secondaryTypographyProps={{ fontSize: 11, color: "warning.main" }}
                    />
                  </ListItemButton>
                </Tooltip>
              ))}
            </ul>
          </li>
        ))}
      </List>
    </Box>
  );
};

export default AufgabenPalette;
