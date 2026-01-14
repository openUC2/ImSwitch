import {
  Apps as AppsIcon,
  Code as CodeIcon,
  Computer as ComputerIcon,
  Dashboard as DashboardIcon,
  Folder as FolderIcon,
  Star as StarIcon,
  GridView as GridViewIcon,
  ViewInAr as ViewInArIcon,
} from "@mui/icons-material";
import { Divider, Drawer, List, useTheme } from "@mui/material";
import { useState } from "react";
import { useSelector } from "react-redux";
import { getSidebarColors } from "../../constants/sidebarColors.js";
import { selectEnabledApps } from "../../state/slices/appManagerSlice.js";
import { APP_REGISTRY, APP_CATEGORIES } from "../../constants/appRegistry.js";
import DrawerEntry from "./DrawerEntry.jsx";
import DrawerHeader from "./DrawerHeader.jsx";

/**
 * ImSwitch Navigation Drawer Component
 * Main navigation sidebar for microscopy control interface
 * Now dynamically shows only enabled applications from app registry
 */
const NavigationDrawer = ({
  // Drawer state
  sidebarVisible,
  setSidebarVisible,
  isMobile,
  drawerWidth,

  // Navigation state
  selectedPlugin,
  handlePluginChange,

  // Dynamic plugins
  plugins = [],
}) => {
  // Get current theme for color adaptation
  const theme = useTheme();
  const SIDEBAR_COLORS = getSidebarColors(theme.palette.mode);

  // Redux state
  const enabledApps = useSelector(selectEnabledApps);

  // Helper function to check if an app is enabled
  const isAppEnabled = (appId) => enabledApps.includes(appId);

  // Helper function to get enabled apps by category
  const getEnabledAppsByCategory = (category) => {
    return Object.values(APP_REGISTRY).filter(
      (app) => app.category === category && isAppEnabled(app.id)
    );
  };

  // Handle App Manager opening (currently handled by plugin change)
  // const handleOpenAppManager = () => {
  //   dispatch(openAppManager());
  // };

  // Internal state management for drawer groups
  const [groupsOpen, setGroupsOpen] = useState(() => {
    // Restore from localStorage if available, otherwise start collapsed
    try {
      const saved = localStorage.getItem("imswitch.groupsOpen");
      if (saved) return JSON.parse(saved);
    } catch (e) {
      // Ignore JSON/localStorage errors
    }
    return {
      essentials: true,
      apps: false,
      coding: false,
      system: false,
      systemSettings: false,
    };
  });

  // Internal toggle function - encapsulated navigation logic
  const toggleGroup = (groupName) => {
    setGroupsOpen((prev) => {
      const next = {
        ...prev,
        [groupName]: !prev[groupName],
      };
      // Persist to localStorage
      try {
        localStorage.setItem("imswitch.groupsOpen", JSON.stringify(next));
      } catch (e) {
        // Ignore localStorage errors
      }
      return next;
    });
  };

  // Render enabled apps for a specific category
  const renderAppsForCategory = (category, color) => {
    const enabledApps = getEnabledAppsByCategory(category);

    return enabledApps.map((app) => {
      const IconComponent = app.icon;
      return (
        <DrawerEntry
          key={app.id}
          icon={<IconComponent />}
          label={app.name}
          selected={selectedPlugin === app.pluginId}
          onClick={() => handlePluginChange(app.pluginId)}
          tooltip={app.description}
          color={color}
          collapsed={!sidebarVisible}
          nested={true}
        />
      );
    });
  };

  return (
    <Drawer
      variant={isMobile ? "temporary" : "persistent"}
      anchor="left"
      open={isMobile ? sidebarVisible : true} // Desktop: always open, Mobile: controlled by sidebarVisible
      onClose={() => setSidebarVisible(false)}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        zIndex: (theme) => theme.zIndex.drawer + 3,
        "& .MuiDrawer-paper": {
          width: drawerWidth,
          boxSizing: "border-box",
          zIndex: (theme) => theme.zIndex.drawer + 3,
          transition: (theme) =>
            theme.transitions.create("width", {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
          display: "flex",
          flexDirection: "column",
          // Remove overflow here - let the List handle scrolling
          overflow: "hidden",
        },
      }}
    >
      <DrawerHeader
        sidebarVisible={sidebarVisible}
        setSidebarVisible={setSidebarVisible}
        isMobile={isMobile}
      />

      <List
        sx={{
          width: "100%",
          boxSizing: "border-box",
          padding: 0,
          flex: 1, // Take remaining space
          // Enable scrolling with hidden scrollbars
          overflowY: "auto",
          overflowX: "hidden",
          "&::-webkit-scrollbar": {
            width: "0px",
            background: "transparent", // Hide scrollbar for webkit
          },
          "&::-webkit-scrollbar-thumb": {
            background: "transparent",
          },
          scrollbarWidth: "none", // Hide scrollbar for Firefox
          msOverflowStyle: "none", // Hide scrollbar for IE/Edge
        }}
      >
        {/* Essentials Group - Core microscopy components */}
        <DrawerEntry
          icon={<StarIcon />}
          label="Essentials"
          onClick={() => toggleGroup("essentials")}
          tooltip="Essential microscopy components"
          color={SIDEBAR_COLORS.essentials}
          collapsed={!sidebarVisible}
          collapsible={true}
          expanded={groupsOpen.essentials}
        >
          {/* LiveView - Main microscopy interface - Always show as it's essential */}
          <DrawerEntry
            icon={<DashboardIcon />}
            label="Live View"
            selected={selectedPlugin === "LiveView"}
            onClick={() => handlePluginChange("LiveView")}
            tooltip="Live View - Main microscopy control"
            color={SIDEBAR_COLORS.essentials}
            collapsed={!sidebarVisible}
            nested={true}
          />

          {/* File Manager - Microscopy data management - Always show as it's essential */}
          <DrawerEntry
            icon={<FolderIcon />}
            label="File Manager"
            selected={selectedPlugin === "FileManager"}
            onClick={() => handlePluginChange("FileManager")}
            tooltip="File Manager - Microscopy data management"
            color={SIDEBAR_COLORS.essentials}
            collapsed={!sidebarVisible}
            nested={true}
          />

          {/* OME-Zarr Viewer - Offline Vizarr viewer for microscopy data */}
          <DrawerEntry
            icon={<ViewInArIcon />}
            label="OME-Zarr Viewer"
            selected={selectedPlugin === "VizarrViewer"}
            onClick={() => handlePluginChange("VizarrViewer")}
            tooltip="OME-Zarr Viewer - Offline viewer for multidimensional data"
            color={SIDEBAR_COLORS.essentials}
            collapsed={!sidebarVisible}
            nested={true}
          />

          {/* App Manager - Customize workspace - Always show as it's essential */}
          <DrawerEntry
            icon={<GridViewIcon />}
            label="App Manager"
            selected={selectedPlugin === "AppManager"}
            onClick={() => handlePluginChange("AppManager")}
            tooltip="App Manager - Customize your workspace"
            color={SIDEBAR_COLORS.essentials}
            collapsed={!sidebarVisible}
            nested={true}
          />
        </DrawerEntry>

        {/* Apps Group - Microscopy Applications */}
        {getEnabledAppsByCategory(APP_CATEGORIES.APPS).length > 0 && (
          <>
            <Divider sx={{ my: 1 }} />
            <DrawerEntry
              icon={<AppsIcon />}
              label="Apps"
              onClick={() => toggleGroup("apps")}
              tooltip="Microscopy Applications"
              color={SIDEBAR_COLORS.apps}
              collapsed={!sidebarVisible}
              collapsible={true}
              expanded={groupsOpen.apps}
            >
              {renderAppsForCategory(APP_CATEGORIES.APPS, SIDEBAR_COLORS.apps)}
            </DrawerEntry>
          </>
        )}

        {/* Coding Group - Development Tools */}
        {getEnabledAppsByCategory(APP_CATEGORIES.CODING).length > 0 && (
          <>
            <Divider sx={{ my: 1 }} />
            <DrawerEntry
              icon={<CodeIcon />}
              label="Coding"
              onClick={() => toggleGroup("coding")}
              tooltip="Development and debugging tools"
              color={SIDEBAR_COLORS.coding}
              collapsed={!sidebarVisible}
              collapsible={true}
              expanded={groupsOpen.coding}
            >
              {renderAppsForCategory(
                APP_CATEGORIES.CODING,
                SIDEBAR_COLORS.coding
              )}
            </DrawerEntry>
          </>
        )}

        {/* System Group - System Configuration */}
        {getEnabledAppsByCategory(APP_CATEGORIES.SYSTEM).length > 0 && (
          <>
            <Divider sx={{ my: 1 }} />
            <DrawerEntry
              icon={<ComputerIcon />}
              label="System"
              onClick={() => toggleGroup("system")}
              tooltip="System configuration and utilities"
              color={SIDEBAR_COLORS.system}
              collapsed={!sidebarVisible}
              collapsible={true}
              expanded={groupsOpen.system}
            >
              {renderAppsForCategory(
                APP_CATEGORIES.SYSTEM,
                SIDEBAR_COLORS.system
              )}
            </DrawerEntry>
          </>
        )}
      </List>
    </Drawer>
  );
};

export default NavigationDrawer;
