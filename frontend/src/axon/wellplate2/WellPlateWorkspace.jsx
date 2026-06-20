import React, { useState, useEffect } from "react";
import {
  Box,
  Tabs,
  Tab,
  Tooltip,
  Typography,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";

// Viewport tab icons
import GridViewIcon from "@mui/icons-material/GridView";
import VideocamIcon from "@mui/icons-material/Videocam";
import MapIcon from "@mui/icons-material/Map";
import ViewInArIcon from "@mui/icons-material/ViewInAr";
import FormatListBulletedIcon from "@mui/icons-material/FormatListBulleted";
import LockIcon from "@mui/icons-material/Lock";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import LinkIcon from "@mui/icons-material/Link";

// Existing components — re-hosted verbatim (no logic / design changes)
import WellSelectorComponent from "../WellSelectorComponent";
import LiveViewControlWrapper from "../LiveViewControlWrapper";
import OverviewScanTab from "../OverviewScanTab";
import Frame3DViewerPanel from "../Frame3DViewerPanel.jsx";
import PointListEditorComponent from "../PointListEditorComponent";
import WebSocketComponent from "../WebSocketComponent";
import PositionViewComponent from "../PositionViewComponent";
import ParameterEditorWrapper from "../ParameterEditorWrapper";
import DetectorToggle from "../DetectorToggle";
import FocusLockMiniController from "../../components/FocusLockMiniController";

const VIEWPORT_STORAGE_KEY = "wp2-activeViewport";

/**
 * Live View viewport: camera selector + the existing live view controls,
 * relocated unchanged from the legacy left/right "Live View" tabs.
 */
const LiveViewPanel = () => (
  <Box>
    <DetectorToggle />
    <LiveViewControlWrapper />
  </Box>
);

/**
 * Stage state viewport: the existing connection / camera-position readouts,
 * relocated unchanged from the legacy "State" tab.
 */
const StageStatePanel = () => (
  <Box sx={{ display: "flex", flexWrap: "wrap" }}>
    <WebSocketComponent />
    <PositionViewComponent />
  </Box>
);

/**
 * Tab that renders a real MUI <Tab> wrapped in a <Tooltip>. The forwardRef +
 * prop spread is required so the parent <Tabs> can still inject `selected`,
 * `value`, `onChange` etc. into the underlying Tab (wrapping a Tab directly in
 * a Tooltip breaks selection).
 */
const TooltipTab = React.forwardRef(function TooltipTab({ help, ...tabProps }, ref) {
  return (
    <Tooltip title={help} arrow placement="bottom" enterDelay={400}>
      <Tab ref={ref} {...tabProps} />
    </Tooltip>
  );
});

/**
 * WellPlateWorkspace — the renovated WellPlate screen.
 *
 * Replaces the two competing tab bars with a single "Linked Workspace":
 *   • LEFT  — a viewport tab strip (Plate Map / Live View / Overview / 3D Twin
 *             plus the Points / Focus Lock / Stage utility panels) that swaps
 *             only the viewport.
 *   • RIGHT — the existing experiment inspector (ExperimentDesigner via
 *             ParameterEditorWrapper) which stays mounted at all times.
 *
 * Because the map (left) and the inspector (right) are siblings that are both
 * always present in the DOM, selecting positions on the Plate Map updates the
 * inspector's Positions list live — no tab round-trips. Every panel below is an
 * existing component hosted as-is; this file only re-arranges them.
 */
const WellPlateWorkspace = () => {
  const theme = useTheme();

  // Which viewport is showing on the left. Purely UI state, persisted so the
  // view is sticky across remounts (mirrors GenericTabBar's localStorage).
  const [viewport, setViewport] = useState(() => {
    const saved = localStorage.getItem(VIEWPORT_STORAGE_KEY);
    return saved !== null ? parseInt(saved, 10) : 0;
  });

  useEffect(() => {
    localStorage.setItem(VIEWPORT_STORAGE_KEY, String(viewport));
  }, [viewport]);

  // Primary viewports first, then the utility panels (kept so no functionality
  // from the old tab bars is lost). `divider` marks where the utility group
  // starts so we can render a subtle separator before it.
  const viewports = [
    {
      label: "Plate Map",
      icon: <GridViewIcon />,
      help: "Pan/zoom plate map. Select positions, draw scan areas, move the camera — selections flow straight into the experiment on the right.",
      render: () => <WellSelectorComponent />,
    },
    {
      label: "Live View",
      icon: <VideocamIcon />,
      help: "Live camera stream and capture controls. Use the camera selector to switch detectors.",
      render: () => <LiveViewPanel />,
    },
    {
      label: "Overview",
      icon: <MapIcon />,
      help: "Stitched overview overlay, manual registration wizard and autonomous overview scan.",
      render: () => <OverviewScanTab />,
    },
    {
      label: "3D Twin",
      icon: <ViewInArIcon />,
      help: "FRAME digital twin: live XYZA stage readout, layer toggles and axis mapping.",
      render: () => <Frame3DViewerPanel />,
    },
    {
      label: "Points",
      icon: <FormatListBulletedIcon />,
      help: "Detailed point-list editor: reorder, rename, edit shapes and jump the stage to a point.",
      divider: true,
      render: () => <PointListEditorComponent />,
    },
    {
      label: "Focus Lock",
      icon: <LockIcon />,
      help: "Focus lock mini-controller (hold focus during long acquisitions).",
      render: () => <FocusLockMiniController />,
    },
    {
      label: "Stage",
      icon: <MyLocationIcon />,
      help: "Connection settings and live camera/stage position readout.",
      render: () => <StageStatePanel />,
    },
  ];

  const active = viewports[viewport] || viewports[0];

  return (
    <Box sx={{ width: "100%", px: 1 }}>
      {/* Linked-workspace hint */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 0.75,
          color: theme.palette.text.secondary,
          mb: 0.5,
          px: 0.5,
        }}
      >
        <LinkIcon sx={{ fontSize: 16 }} />
        <Typography variant="caption">
          Pick a viewport on the left — the experiment is always on the right.
          Map selections fill the Positions list live.
        </Typography>
      </Box>

      <Box
        sx={{
          display: "flex",
          gap: 1.5,
          alignItems: "flex-start",
          width: "100%",
        }}
      >
        {/* LEFT — viewport tab strip + content */}
        <Box sx={{ flex: 3, minWidth: 0 }}>
          <Tabs
            value={viewport}
            onChange={(_e, v) => setViewport(v)}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            sx={{
              minHeight: 48,
              borderBottom: 1,
              borderColor: "divider",
              "& .MuiTab-root": {
                minHeight: 48,
                textTransform: "none",
                fontWeight: 500,
              },
            }}
          >
            {viewports.map((v, i) => (
              <TooltipTab
                key={v.label}
                value={i}
                help={v.help}
                icon={v.icon}
                iconPosition="start"
                label={v.label}
                // Subtle separator before the utility-panel group, without
                // injecting non-Tab children into <Tabs>.
                sx={
                  v.divider
                    ? {
                        ml: 1,
                        borderLeft: `1px solid ${alpha(theme.palette.divider, 0.9)}`,
                      }
                    : undefined
                }
              />
            ))}
          </Tabs>

          <Box sx={{ mt: 1 }}>{active.render()}</Box>
        </Box>

        {/* RIGHT — experiment inspector, always mounted (existing design) */}
        <Box
          sx={{
            flex: 2,
            minWidth: 380,
            alignSelf: "stretch",
            borderLeft: `1px solid ${theme.palette.divider}`,
            pl: 1,
          }}
        >
          <ParameterEditorWrapper />
        </Box>
      </Box>
    </Box>
  );
};

export default WellPlateWorkspace;
