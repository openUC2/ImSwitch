import React from "react";
import { useDispatch, useSelector } from "react-redux";
import { Box, Typography, Chip, IconButton, Tooltip } from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import LocationOnIcon from "@mui/icons-material/LocationOn";
import TuneIcon from "@mui/icons-material/Tune";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import GridOnIcon from "@mui/icons-material/GridOn";
import SaveAltIcon from "@mui/icons-material/SaveAlt";

import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

/**
 * Dimension configuration with icons and labels
 */
const DIMENSION_CONFIG = {
  [DIMENSIONS.POSITIONS]: {
    label: "Positions",
    icon: LocationOnIcon,
    alwaysEnabled: true,
  },
  [DIMENSIONS.CHANNELS]: {
    label: "Channels",
    icon: TuneIcon,
    alwaysEnabled: false,
  },
  [DIMENSIONS.Z_FOCUS]: {
    label: "Z / Focus",
    icon: CenterFocusStrongIcon,
    alwaysEnabled: false,
  },
  [DIMENSIONS.TIME]: {
    label: "Time",
    icon: AccessTimeIcon,
    alwaysEnabled: false,
  },
  [DIMENSIONS.TILING]: {
    label: "Tiling",
    icon: GridOnIcon,
    alwaysEnabled: false,
  },
  [DIMENSIONS.OUTPUT]: {
    label: "Output",
    icon: SaveAltIcon,
    alwaysEnabled: true,
  },
};

/**
 * Single dimension tab item
 */
const DimensionTab = ({ 
  dimension, 
  config, 
  isExpanded, 
  isEnabled, 
  isConfigured,
  summary,
  onToggleEnabled, 
  onExpand 
}) => {
  const theme = useTheme();
  const IconComponent = config.icon;
  
  // Determine visual states
  const isActive = isExpanded;
  const showCheckmark = isEnabled && !config.alwaysEnabled;
  
  // Calculate styles based on state
  const getBackgroundColor = () => {
    if (isActive) {
      return alpha(theme.palette.primary.main, 0.15);
    }
    if (isEnabled) {
      return alpha(theme.palette.success.main, 0.08);
    }
    return "transparent";
  };
  
  const getBorderColor = () => {
    if (isActive) {
      return theme.palette.primary.main;
    }
    if (isEnabled) {
      return alpha(theme.palette.success.main, 0.4);
    }
    return alpha(theme.palette.divider, 0.5);
  };
  
  const getTextColor = () => {
    if (isActive) {
      return theme.palette.primary.main;
    }
    if (isEnabled) {
      return theme.palette.text.primary;
    }
    return theme.palette.text.secondary;
  };

  return (
    <Box
      onClick={onExpand}
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "8px 16px",
        minWidth: "90px",
        cursor: "pointer",
        borderRadius: "8px 8px 0 0",
        borderBottom: isActive ? `2px solid ${theme.palette.primary.main}` : "2px solid transparent",
        backgroundColor: getBackgroundColor(),
        transition: "all 0.2s ease-in-out",
        position: "relative",
        "&:hover": {
          backgroundColor: alpha(theme.palette.action.hover, 0.1),
        },
      }}
    >
      {/* Enable/disable toggle - only for non-always-enabled dimensions */}
      {!config.alwaysEnabled && (
        <Tooltip title={isEnabled ? "Disable dimension" : "Enable dimension"}>
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              onToggleEnabled();
            }}
            sx={{
              position: "absolute",
              top: 2,
              right: 2,
              padding: "2px",
              opacity: 0.7,
              "&:hover": { opacity: 1 },
            }}
          >
            {isEnabled ? (
              <CheckCircleIcon 
                sx={{ fontSize: 14, color: theme.palette.success.main }} 
              />
            ) : (
              <RadioButtonUncheckedIcon 
                sx={{ fontSize: 14, color: theme.palette.text.disabled }} 
              />
            )}
          </IconButton>
        </Tooltip>
      )}
      
      {/* Icon */}
      <IconComponent 
        sx={{ 
          fontSize: 20, 
          color: getTextColor(),
          mb: 0.5,
        }} 
      />
      
      {/* Label */}
      <Typography
        variant="caption"
        sx={{
          fontWeight: isActive ? 600 : 500,
          color: getTextColor(),
          fontSize: "0.75rem",
          lineHeight: 1.2,
        }}
      >
        {config.label}
      </Typography>
      
      {/* Summary - shown below when enabled */}
      {isEnabled && summary && (
        <Typography
          variant="caption"
          sx={{
            fontSize: "0.65rem",
            color: theme.palette.text.secondary,
            maxWidth: "80px",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            mt: 0.25,
          }}
          title={summary}
        >
          {summary}
        </Typography>
      )}
    </Box>
  );
};

/**
 * DimensionBar - Top-level navigation for experiment dimensions
 * 
 * Shows tabs for: Positions | Channels | Z/Focus | Time | Tiling | Output
 * Each tab can be enabled/disabled and shows configuration status
 */
const DimensionBar = () => {
  const dispatch = useDispatch();
  const theme = useTheme();
  
  // Get UI state
  const experimentUI = useSelector(experimentUISlice.getExperimentUIState);
  const expandedDimension = experimentUI.expandedDimension;
  const dimensions = experimentUI.dimensions;
  
  // Dimension order for display
  const dimensionOrder = [
    DIMENSIONS.POSITIONS,
    DIMENSIONS.CHANNELS,
    DIMENSIONS.Z_FOCUS,
    DIMENSIONS.TIME,
    DIMENSIONS.TILING,
    DIMENSIONS.OUTPUT,
  ];

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "flex-end",
        gap: 0.5,
        borderBottom: `1px solid ${theme.palette.divider}`,
        backgroundColor: alpha(theme.palette.background.paper, 0.5),
        paddingX: 1,
        paddingTop: 1,
        overflowX: "auto",
        "&::-webkit-scrollbar": {
          height: "4px",
        },
        "&::-webkit-scrollbar-thumb": {
          backgroundColor: alpha(theme.palette.text.secondary, 0.3),
          borderRadius: "2px",
        },
      }}
    >
      {dimensionOrder.map((dimension) => {
        const config = DIMENSION_CONFIG[dimension];
        const dimState = dimensions[dimension];
        
        return (
          <DimensionTab
            key={dimension}
            dimension={dimension}
            config={config}
            isExpanded={expandedDimension === dimension}
            isEnabled={dimState?.enabled ?? false}
            isConfigured={dimState?.configured ?? false}
            summary={dimState?.summary}
            onToggleEnabled={() => {
              dispatch(experimentUISlice.toggleDimensionEnabled(dimension));
            }}
            onExpand={() => {
              // Only expand enabled dimensions or always-enabled ones
              if (dimState?.enabled || config.alwaysEnabled) {
                dispatch(experimentUISlice.setExpandedDimension(dimension));
              } else {
                // If clicking a disabled dimension, enable it first
                dispatch(experimentUISlice.setDimensionEnabled({ dimension, enabled: true }));
              }
            }}
          />
        );
      })}
    </Box>
  );
};

export default DimensionBar;
