// Kiosk shell for the Raspberry Pi touchscreen (#/mobile): a Bambu-style
// left icon rail plus one full-height page. Rendered by App.jsx inside the
// existing Redux/WebSocket/Snackbar providers — this component only adds the
// kiosk theme and layout on top.
//
// Kiosk URL: http://<host>:<port>/imswitch/ui/index.html#/mobile
import { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { ThemeProvider } from "@mui/material/styles";
import {
  alpha,
  Box,
  ButtonBase,
  CssBaseline,
  Typography,
  useMediaQuery,
} from "@mui/material";
import HomeRoundedIcon from "@mui/icons-material/HomeRounded";
import OpenWithRoundedIcon from "@mui/icons-material/OpenWithRounded";
import FlareRoundedIcon from "@mui/icons-material/FlareRounded";
import AppsRoundedIcon from "@mui/icons-material/AppsRounded";
import VideocamRoundedIcon from "@mui/icons-material/VideocamRounded";
import CenterFocusStrongRoundedIcon from "@mui/icons-material/CenterFocusStrongRounded";
import WifiRoundedIcon from "@mui/icons-material/WifiRounded";
import MemoryRoundedIcon from "@mui/icons-material/MemoryRounded";
import LogoutRoundedIcon from "@mui/icons-material/LogoutRounded";

import mobileTheme from "./mobileTheme";
import { MOBILE_PAGES, exitMobileUI, useMobileRoute } from "./mobileRoutes";
import { getBackendCapabilitiesState } from "../state/slices/BackendCapabilitiesSlice";
import fetchExperimentControllerGetCurrentExperimentParams from "../middleware/fetchExperimentControllerGetCurrentExperimentParams";
import fetchObjectiveControllerGetStatus from "../middleware/fetchObjectiveControllerGetStatus";
import logoUrl from "../assets/ouc2_logo_quadratic.png";

import MobileHomePage from "./pages/MobileHomePage";
import MobileStagePage from "./pages/MobileStagePage";
import MobileLaserPage from "./pages/MobileLaserPage";
import MobileLedPage from "./pages/MobileLedPage";
import MobileCameraPage from "./pages/MobileCameraPage";
import MobileObjectivePage from "./pages/MobileObjectivePage";
import MobileWifiPage from "./pages/MobileWifiPage";
import MobileSystemPage from "./pages/MobileSystemPage";

const NAV_ITEMS = [
  { id: MOBILE_PAGES.HOME, label: "Home", icon: <HomeRoundedIcon /> },
  {
    id: MOBILE_PAGES.STAGE,
    label: "Stage",
    icon: <OpenWithRoundedIcon />,
    controller: "PositionerController",
  },
  {
    id: MOBILE_PAGES.LASERS,
    label: "Lasers",
    icon: <FlareRoundedIcon />,
    controller: "LaserController",
  },
  {
    id: MOBILE_PAGES.LEDS,
    label: "LED",
    icon: <AppsRoundedIcon />,
    controller: "LEDMatrixController",
  },
  {
    id: MOBILE_PAGES.CAMERA,
    label: "Camera",
    icon: <VideocamRoundedIcon />,
    controller: "LiveViewController",
  },
  {
    id: MOBILE_PAGES.OBJECTIVE,
    label: "Lens",
    icon: <CenterFocusStrongRoundedIcon />,
    controller: "ObjectiveController",
  },
  { id: MOBILE_PAGES.WIFI, label: "WiFi", icon: <WifiRoundedIcon /> },
  { id: MOBILE_PAGES.SYSTEM, label: "System", icon: <MemoryRoundedIcon /> },
];

const RailButton = ({ icon, label, selected, compact, onClick }) => (
  <ButtonBase
    onClick={onClick}
    sx={{
      width: 68,
      minHeight: compact ? 42 : 60,
      flexShrink: 0,
      borderRadius: 3,
      display: "flex",
      flexDirection: "column",
      gap: 0.25,
      py: compact ? 0.25 : 0.75,
      color: selected ? "primary.main" : "text.secondary",
      bgcolor: (theme) =>
        selected ? alpha(theme.palette.primary.main, 0.14) : "transparent",
      "& svg": { fontSize: 26 },
    }}
  >
    {icon}
    {!compact && (
      <Typography variant="caption" sx={{ fontSize: 10, fontWeight: 700 }}>
        {label}
      </Typography>
    )}
  </ButtonBase>
);

const MobileApp = () => {
  const dispatch = useDispatch();
  const { page, navigate } = useMobileRoute();
  const capabilities = useSelector(getBackendCapabilitiesState);
  // 7" 800x480 panels: shrink rail items (icons only) so all pages stay
  // reachable without scrolling the rail.
  const compactRail = useMediaQuery("(max-height: 560px)");

  // Populate the slices that several kiosk pages read (illumination sources,
  // objective status). Both are cheap one-shot GETs.
  useEffect(() => {
    fetchExperimentControllerGetCurrentExperimentParams(dispatch);
    fetchObjectiveControllerGetStatus(dispatch);
  }, [dispatch]);

  // Hide pages whose backend controller is missing in this setup — but only
  // once the capability list has actually been fetched (otherwise the rail
  // would flash empty while the backend is still answering).
  const visibleItems = NAV_ITEMS.filter(
    (item) =>
      !item.controller ||
      !capabilities.lastUpdated ||
      capabilities.availableControllers.includes(item.controller),
  );

  const activePage = page || MOBILE_PAGES.HOME;

  return (
    <ThemeProvider theme={mobileTheme}>
      <CssBaseline />
      <Box
        sx={{
          height: "100dvh",
          display: "flex",
          bgcolor: "background.default",
          overflow: "hidden",
          userSelect: "none",
          WebkitTouchCallout: "none",
        }}
      >
        {/* Left icon rail */}
        <Box
          sx={{
            width: 84,
            flexShrink: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            py: compactRail ? 0.75 : 1.5,
            gap: compactRail ? 0.25 : 0.5,
            borderRight: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
          }}
        >
          <Box
            component="img"
            src={logoUrl}
            alt="openUC2"
            sx={{
              width: compactRail ? 32 : 40,
              height: compactRail ? 32 : 40,
              borderRadius: 1.5,
              mb: compactRail ? 0.5 : 1,
            }}
          />
          <Box
            sx={{
              flex: 1,
              minHeight: 0,
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: compactRail ? 0.25 : 0.5,
              width: "100%",
              scrollbarWidth: "none",
              "&::-webkit-scrollbar": { width: 0 },
            }}
          >
            {visibleItems.map((item) => (
              <RailButton
                key={item.id}
                icon={item.icon}
                label={item.label}
                selected={activePage === item.id}
                compact={compactRail}
                onClick={() => navigate(item.id)}
              />
            ))}
          </Box>
          <RailButton
            icon={<LogoutRoundedIcon />}
            label="Full UI"
            selected={false}
            compact={compactRail}
            onClick={exitMobileUI}
          />
        </Box>

        {/* Active page */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          {activePage === MOBILE_PAGES.HOME && <MobileHomePage navigate={navigate} />}
          {activePage === MOBILE_PAGES.STAGE && <MobileStagePage />}
          {activePage === MOBILE_PAGES.LASERS && <MobileLaserPage />}
          {activePage === MOBILE_PAGES.LEDS && <MobileLedPage />}
          {activePage === MOBILE_PAGES.CAMERA && <MobileCameraPage />}
          {activePage === MOBILE_PAGES.OBJECTIVE && <MobileObjectivePage />}
          {activePage === MOBILE_PAGES.WIFI && <MobileWifiPage />}
          {activePage === MOBILE_PAGES.SYSTEM && <MobileSystemPage />}
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default MobileApp;
