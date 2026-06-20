import React, { useState, useEffect, useRef, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Tabs,
  Tab,
  Tooltip,
  Typography,
  IconButton,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import introJs from "intro.js";
import "intro.js/introjs.css";

// Viewport tab icons
import GridViewIcon from "@mui/icons-material/GridView";
import VideocamIcon from "@mui/icons-material/Videocam";
import MapIcon from "@mui/icons-material/Map";
import ViewInArIcon from "@mui/icons-material/ViewInAr";
import LockIcon from "@mui/icons-material/Lock";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import LinkIcon from "@mui/icons-material/Link";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";

// Existing components — re-hosted verbatim (no logic / design changes)
import WellSelectorComponent from "../WellSelectorComponent";
import LiveViewControlWrapper from "../LiveViewControlWrapper";
import OverviewScanTab from "../OverviewScanTab";
import Frame3DViewerPanel from "../Frame3DViewerPanel.jsx";
import WebSocketComponent from "../WebSocketComponent";
import PositionViewComponent from "../PositionViewComponent";
import ParameterEditorWrapper from "../ParameterEditorWrapper";
import DetectorToggle from "../DetectorToggle";
import FocusLockMiniController from "../../components/FocusLockMiniController";

// State + APIs used for per-viewport camera auto-switching
import * as liveViewSlice from "../../state/slices/LiveViewSlice.js";
import * as liveStreamSlice from "../../state/slices/LiveStreamSlice.js";
import * as overviewRegSlice from "../../state/slices/OverviewRegistrationSlice.js";
import { getThemeState } from "../../state/slices/ThemeSlice.js";
import apiLiveViewControllerStartLiveView from "../../backendapi/apiLiveViewControllerStartLiveView";
import apiLiveViewControllerStopLiveView from "../../backendapi/apiLiveViewControllerStopLiveView";

const VIEWPORT_STORAGE_KEY = "wp2-activeViewport";

// Viewport indices that drive camera auto-switching (see VIEWPORTS below).
const VIEW_LIVE = 1;
const VIEW_OVERVIEW = 2;

/**
 * Switch the live-view active detector to `targetIdx`, restarting the stream on
 * the new detector when one is running (mirrors DetectorToggle's switch). No-op
 * when the target is missing or already active.
 */
async function switchToDetector({
  dispatch,
  detectors,
  currentIdx,
  isStreamRunning,
  liveStreamState,
  targetIdx,
}) {
  if (targetIdx == null || targetIdx < 0 || targetIdx === currentIdx) return;
  dispatch(liveViewSlice.setActiveTab(targetIdx));
  if (!isStreamRunning) return; // just remember the selection; nothing streaming
  try {
    await apiLiveViewControllerStopLiveView();
    await new Promise((r) => setTimeout(r, 200));
    const protocol = liveStreamState.imageFormat || "jpeg";
    const name = detectors[targetIdx] || null;
    const saved = name && liveStreamState.perDetectorSettings?.[name];
    const override = saved && saved.protocol === protocol ? saved : null;
    const result = await apiLiveViewControllerStartLiveView(name, protocol, override);
    if (result?.params && name) {
      dispatch(
        liveStreamSlice.updateDetectorSettings({ detectorName: name, settings: result.params }),
      );
    }
  } catch (err) {
    console.error("[WellPlateWorkspace] camera switch failed:", err);
  }
}

/** Resolve the widefield detector index (name match, else first non-overview, else 0). */
const findWidefieldIdx = (detectors, overviewName) => {
  if (!detectors || detectors.length === 0) return -1;
  let idx = detectors.findIndex((n) => /wide/i.test(n));
  if (idx < 0 && overviewName) idx = detectors.findIndex((n) => n !== overviewName);
  return idx < 0 ? 0 : idx;
};

/** Resolve the overview detector index (configured name, else observation/overview match). */
const findOverviewIdx = (detectors, overviewName) => {
  if (!detectors || detectors.length === 0) return -1;
  if (overviewName) {
    const i = detectors.indexOf(overviewName);
    if (i >= 0) return i;
  }
  return detectors.findIndex((n) => /observation|overview/i.test(n));
};

// Shared intro.js options — matches the look of the Live View onboarding tour.
const TOUR_OPTIONS = {
  showProgress: true,
  showBullets: false,
  exitOnOverlayClick: false,
  exitOnEsc: true,
  scrollToElement: true,
  nextLabel: "Next",
  prevLabel: "Back",
  doneLabel: "Done",
  skipLabel: "✕",
  overlayOpacity: 0.6,
  tooltipClass: "imswitch-introjs-tooltip",
  highlightClass: "imswitch-introjs-highlight",
  tooltipRenderAsHtml: true,
};

// Declarative tour steps. `target` is a data-tour-wp selector; missing targets
// are skipped so the tour never breaks. `target: null` => centered floating step.
const TOUR_STEPS = [
  {
    target: null,
    title: "WellPlate workspace",
    intro:
      "The left shows <b>one viewport at a time</b>; the <b>experiment</b> is always on the right, " +
      "so you configure while you watch the sample.",
  },
  {
    target: '[data-tour-wp="viewport-strip"]',
    title: "Viewports",
    intro: "Switch the left side here: plate map, live camera, stitched overview, or the 3D stage twin.",
  },
  {
    target: '[data-tour-wp="tab-plate-map"]',
    title: "Plate Map",
    intro:
      "Select positions, draw scan areas and move the camera. Selections appear in the experiment's " +
      "<b>Positions</b> list instantly — no tab switching.",
  },
  {
    target: '[data-tour-wp="tab-live-view"]',
    title: "Live View",
    intro: "The live <b>widefield</b> camera — auto-selected when you open this tab.",
  },
  {
    target: '[data-tour-wp="tab-overview"]',
    title: "Overview",
    intro: "Stitched overview overlay + registration. Opening it switches to the <b>overview camera</b>.",
  },
  {
    target: '[data-tour-wp="tab-3d-twin"]',
    title: "3D Twin",
    intro: "A live 3D model of the stage position.",
  },
  {
    target: '[data-tour-wp="inspector"]',
    title: "Experiment",
    intro:
      "Channels, Z, time, tiling, output — and <b>Start / Stop</b>. Always visible so the map and the " +
      "parameters stay side by side.",
  },
  {
    target: null,
    title: "You're set",
    intro: "Explore at your own pace — reopen this tour any time from the <b>ⓘ</b> button.",
  },
];

/** Build + launch a self-contained intro.js tour over the workspace. */
const launchTour = (isDarkMode) => {
  const steps = TOUR_STEPS.filter((s) => !s.target || document.querySelector(s.target)).map((s) => ({
    element: s.target ? document.querySelector(s.target) : undefined,
    title: s.title,
    intro: s.intro,
    ...(s.target ? { position: "bottom" } : {}),
  }));
  if (steps.length === 0) return;

  const tour = introJs();
  if (isDarkMode) document.body.classList.add("imswitch-tour-dark");
  tour.setOptions({ ...TOUR_OPTIONS, steps });
  const cleanup = () => document.body.classList.remove("imswitch-tour-dark");
  tour.onComplete(cleanup);
  tour.onExit(cleanup);
  tour.start();
};

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
 *             plus the Focus Lock / Stage utility panels) that swaps only the
 *             viewport.
 *   • RIGHT — the existing experiment inspector (ExperimentDesigner via
 *             ParameterEditorWrapper) which stays mounted at all times.
 *
 * Position editing now lives in the inspector's Positions step, so there is no
 * separate Points tab. Because the map (left) and the inspector (right) are
 * always-mounted siblings, selecting positions on the Plate Map updates the
 * inspector's Positions list live — no tab round-trips.
 */
const WellPlateWorkspace = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const overviewRegState = useSelector(overviewRegSlice.getOverviewRegistrationState);
  const { isDarkMode } = useSelector(getThemeState);

  // Which viewport is showing on the left. Purely UI state, persisted so the
  // view is sticky across remounts (mirrors GenericTabBar's localStorage).
  const [viewport, setViewport] = useState(() => {
    const saved = localStorage.getItem(VIEWPORT_STORAGE_KEY);
    return saved !== null ? parseInt(saved, 10) : 0;
  });

  useEffect(() => {
    localStorage.setItem(VIEWPORT_STORAGE_KEY, String(viewport));
  }, [viewport]);

  // Latest live-view/stream state, read inside the auto-switch effect without
  // making it re-run (and re-switch) on every unrelated stream-settings change.
  const liveRef = useRef({});
  liveRef.current = {
    detectors: liveViewState.detectors || [],
    activeTab: liveViewState.activeTab || 0,
    isStreamRunning: liveViewState.isStreamRunning,
    liveStreamState,
  };

  const overviewName = overviewRegState.cameraName || "";
  const detectorsKey = (liveViewState.detectors || []).join("|");

  // Auto-switch the camera when entering the Live View / Overview viewports.
  useEffect(() => {
    const { detectors, activeTab, isStreamRunning, liveStreamState: ls } = liveRef.current;
    if (!detectors || detectors.length < 2) return; // nothing to switch between
    let targetIdx = -1;
    if (viewport === VIEW_LIVE) targetIdx = findWidefieldIdx(detectors, overviewName);
    else if (viewport === VIEW_OVERVIEW) targetIdx = findOverviewIdx(detectors, overviewName);
    if (targetIdx < 0) return;
    switchToDetector({
      dispatch,
      detectors,
      currentIdx: activeTab,
      isStreamRunning,
      liveStreamState: ls,
      targetIdx,
    });
  }, [viewport, detectorsKey, overviewName, dispatch]);

  const handleTour = useCallback(() => launchTour(isDarkMode), [isDarkMode]);

  // Primary viewports first, then the utility panels. `divider` marks where the
  // utility group starts so we can render a subtle separator before it.
  const viewports = [
    {
      key: "plate-map",
      label: "Plate Map",
      icon: <GridViewIcon />,
      help: "Pan/zoom plate map. Select positions, draw scan areas, move the camera — selections flow straight into the experiment on the right.",
      render: () => <WellSelectorComponent />,
    },
    {
      key: "live-view",
      label: "Live View",
      icon: <VideocamIcon />,
      help: "Live widefield camera (auto-selected here) and capture controls.",
      render: () => <LiveViewPanel />,
    },
    {
      key: "overview",
      label: "Overview",
      icon: <MapIcon />,
      help: "Stitched overview overlay, registration wizard and autonomous scan. Switches to the overview camera.",
      render: () => <OverviewScanTab />,
    },
    {
      key: "3d-twin",
      label: "3D Twin",
      icon: <ViewInArIcon />,
      help: "FRAME digital twin: live XYZA stage readout, layer toggles and axis mapping.",
      render: () => <Frame3DViewerPanel />,
    },
    {
      key: "focus-lock",
      label: "Focus Lock",
      icon: <LockIcon />,
      help: "Focus lock mini-controller (hold focus during long acquisitions).",
      divider: true,
      render: () => <FocusLockMiniController />,
    },
    {
      key: "stage",
      label: "Stage",
      icon: <MyLocationIcon />,
      help: "Connection settings and live camera/stage position readout.",
      render: () => <StageStatePanel />,
    },
  ];

  const active = viewports[viewport] || viewports[0];

  return (
    <Box sx={{ width: "100%", px: 1 }}>
      {/* Linked-workspace hint + tour launcher */}
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
        <Tooltip title="Take a quick tour of the WellPlate workspace">
          <IconButton size="small" onClick={handleTour} sx={{ p: 0.25 }}>
            <HelpOutlineIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>
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
            data-tour-wp="viewport-strip"
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
                key={v.key}
                value={i}
                help={v.help}
                icon={v.icon}
                iconPosition="start"
                label={v.label}
                data-tour-wp={`tab-${v.key}`}
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
          data-tour-wp="inspector"
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
