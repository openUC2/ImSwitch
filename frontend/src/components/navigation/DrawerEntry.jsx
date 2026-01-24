import React from "react";
import {
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Avatar,
  Collapse,
  List,
} from "@mui/material";
import { ExpandLess, ExpandMore } from "@mui/icons-material";

/**
 * ImSwitch DrawerEntry Component
 * Reusable sidebar menu entry
 * Focused component for navigation entries with consistent styling
 */
const DrawerEntry = ({
  icon,
  label,
  selected = false,
  onClick,
  tooltip = "",
  color,

  // Layout props
  collapsed = false,
  nested = false,

  // Avatar props for microscopy plugins
  avatar = null,
  avatarText = "",

  // Collapsible group props
  collapsible = false,
  expanded = false,
  children = null,
}) => {
  // Determine icon to display
  const displayIcon = avatar ? (
    <Avatar
      sx={{
        bgcolor: color,
        width: 24,
        height: 24,
        fontSize: 14,
      }}
    >
      {avatarText}
    </Avatar>
  ) : (
    icon
  );

  // Build the entry component with consistent ImSwitch styling
  const entry = (
    <ListItem disablePadding={nested}>
      <ListItemButton
        selected={selected}
        onClick={onClick}
        sx={{
          justifyContent: collapsed ? "center" : "flex-start",
          minHeight: 48,
          pl: nested ? 4 : 2, // Left padding: nested=32px, normal=16px
          pr: 2, // Consistent right padding
          // Special styling for collapsible folders - reduced to prevent overflow
          ...(collapsible && {
            backgroundColor: (theme) =>
              theme.palette.mode === "dark"
                ? "rgba(255, 255, 255, 0.03)"
                : "rgba(0, 0, 0, 0.02)",
            borderLeft: (theme) =>
              `2px solid ${color || theme.palette.primary.main}`, // Thinner border
            "&:hover": {
              backgroundColor: (theme) =>
                theme.palette.mode === "dark"
                  ? "rgba(255, 255, 255, 0.06)"
                  : "rgba(0, 0, 0, 0.04)",
            },
          }),
        }}
      >
        <ListItemIcon
          sx={{
            color: color || "inherit",
            minWidth: collapsed ? 0 : 40, // Consistent icon space
            mr: collapsed ? "auto" : 2, // Consistent margin
            justifyContent: "center",
            // Special styling for folder icons - more conservative approach
            ...(collapsible && {
              backgroundColor: (theme) =>
                `${color || theme.palette.primary.main}15`, // 15% opacity (reduced)
              borderRadius: "4px", // Smaller radius
              padding: "2px", // Reduced padding
              // Removed transform to prevent overflow
            }),
          }}
        >
          {displayIcon}
        </ListItemIcon>

        {!collapsed && (
          <ListItemText
            primary={label}
            sx={{
              opacity: 1,
              // Special typography for folders
              ...(collapsible && {
                "& .MuiListItemText-primary": {
                  fontWeight: 600,
                  fontSize: "0.875rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                  color: (theme) => color || theme.palette.text.primary,
                },
              }),
            }}
          />
        )}

        {/* Collapsible indicator for ImSwitch groups */}
        {collapsible && !collapsed && (
          <ListItemIcon sx={{ minWidth: "auto", ml: 1 }}>
            {expanded ? <ExpandLess /> : <ExpandMore />}
          </ListItemIcon>
        )}
      </ListItemButton>
    </ListItem>
  );

  // Wrap with tooltip if collapsed
  const wrappedEntry =
    tooltip && collapsed ? (
      <Tooltip
        title={tooltip || label}
        placement="right"
        disableHoverListener={!collapsed}
      >
        {entry}
      </Tooltip>
    ) : (
      entry
    );

  // Return with optional collapsible children
  return (
    <>
      {wrappedEntry}
      {collapsible && children && (
        <Collapse in={expanded} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            {children}
          </List>
        </Collapse>
      )}
    </>
  );
};

export default DrawerEntry;
