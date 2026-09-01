/* global __webpack_init_sharing__, __webpack_share_scopes__ */
import { lazy, Suspense, useEffect, useRef, useState } from "react";

// ImSwitch Themes
import { darkTheme, lightTheme } from "./themes";

import AboutPage from "./components/AboutPage.js";
import BlocklyController from "./components/BlocklyController.js";
import ConnectionSettings from "./components/ConnectionSettings.jsx";
import DesktopAppSettings from "./components/DesktopAppSettings.jsx";
import DetectorTriggerController from "./components/DetectorTriggerController.js";
import ExtendedLEDMatrixController from "./components/ExtendedLEDMatrixController.jsx";
import FlowStopController from "./components/FlowStopController.js";
import FocusLockController from "./components/FocusLockController.js";
import I2CSensorController from "./components/I2CSensorController.jsx";
import HoloController from "./components/HoloController.js";
import OffAxisHoloController from "./components/OffAxisHoloController.js";
import DPCController from "./components/DPCController.js";
import GoniometerController from "./components/GoniometerController.js";
import ImJoyView from "./components/ImJoyView.js";
import JupyterExecutor from "./components/JupyterExecutor.js";
import LepMonController from "./components/LepmonController.js";
import LightsheetController from "./components/LightsheetController.jsx";
import LiveView from "./components/LiveView.js";
import MazeGameController from "./components/MazeGameController.js";
import MotorSettingsController from "./components/MotorSettingsController.jsx";
import ObjectiveController from "./components/ObjectiveController.js";
import LargeFovScanController from "./components/OpenLayers.js";
import SocketView from "./components/SocketView.js";
import MMCoreController from "./components/MMCoreController.jsx";

import TimelapseController from "./components/TimelapseController.js";
import STORMControllerArkitekt from "./components/STORMControllerArkitekt.js";
import FRAMESettingsController from "./components/FRAMESettingsController.js";
import STORMControllerLocal from "./components/STORMControllerLocal.js";
import StresstestController from "./components/StresstestController.js";
import SystemSettings from "./components/SystemSettings.js";
import SystemUpdateController from "./components/SystemUpdateController.jsx";
import UC2ConfigurationController from "./components/UC2ConfigurationController.jsx";
import SerialDebugController from "./components/SerialDebugController.jsx";
import WiFiController from "./components/WiFiController.jsx";
import LoggingController from "./components/LoggingController.jsx";
import VizarrViewer from "./components/VizarrViewer.jsx";
import { JupyterProvider } from "./context/JupyterContext.js";
import { PWAProvider } from "./context/PWAContext.js";
import DemoController from "./components/DemoController.js";
import StageMapController from "./components/StageMapController.jsx";
import AcceptanceTestComponent from "./components/AcceptanceTestComponent.jsx";
import GalvoScannerController from "./components/GalvoScannerController.jsx";
import ShitScopeComponent from "./components/ShitScopeComponent.js";

// ImSwitch Navigation Drawer
import { NavigationDrawer, TopBar } from "./components/navigation";

// Kiosk/touchscreen UI (#/mobile)
import MobileApp from "./mobile/MobileApp";
import { useMobileRoute } from "./mobile/mobileRoutes";
import AppManagerPage from "./components/AppManagerPage.jsx";
import OnboardingTour from "./components/OnboardingTour.jsx";

//axon
import AxonTabComponent from "./axon/AxonTabComponent.js";
import WebSocketHandler from "./middleware/WebSocketHandler.js";
import CompositeAcquisitionComponent from "./axon/CompositeAcquisitionComponent.js";
import CompositeStreamViewer from "./axon/CompositeStreamViewer.js";
import CompositeComponent from "./axon/CompositeComponent";

//redux
import { useDispatch, useSelector } from "react-redux";
import * as connectionSettingsSlice from "./state/slices/ConnectionSettingsSlice.js";
import * as vizarrViewerSlice from "./state/slices/VizarrViewerSlice.js";
import {
  clearNotification,
  getNotificationState,
  setNotification,
} from "./state/slices/NotificationSlice.js";
import { getThemeState } from "./state/slices/ThemeSlice.js";
import { setDynamicApps } from "./state/slices/appManagerSlice.js";
import { makeRegistryEntryFromManifest } from "./constants/appRegistry.js";
import PluginErrorBoundary from "./components/PluginErrorBoundary.jsx";
import { SnackbarProvider, useSnackbar, enqueueSnackbar } from "notistack";
import useBackendControllerCapabilities from "./hooks/useBackendControllerCapabilities";
import apiPositionerControllerGetHomingStatus from "./backendapi/apiPositionerControllerGetHomingStatus";
import apiPositionerControllerDismissHomingRecommendation from "./backendapi/apiPositionerControllerDismissHomingRecommendation";

// Filemanager
import { api } from "./FileManager/api/api.js";
import { createFolderAPI } from "./FileManager/api/createFolderAPI.js";
import { deleteAPI } from "./FileManager/api/deleteAPI.js";
import { downloadFile } from "./FileManager/api/downloadFileAPI.js";
import { copyItemAPI, moveItemAPI } from "./FileManager/api/fileTransferAPI.js";
import { getAllFilesAPI } from "./FileManager/api/getAllFilesAPI.js";
import { renameAPI } from "./FileManager/api/renameAPI.js";
import "./FileManager/App.scss";
import FileManager from "./FileManager/FileManager/FileManager.jsx";

import {
  Box,
  Button,
  CssBaseline,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  TextField,
} from "@mui/material";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import { ThemeProvider } from "@mui/material/styles";

// How long to wait for a plugin's remoteEntry.js to load and register its
// federation scope before giving up. Long enough for a slow Pi over WiFi,
// short enough that a typo does not look like a hang.
const PLUGIN_LOAD_TIMEOUT_MS = 10000;

function ReduxNotificationBridge() {
  const notification = useSelector(getNotificationState);
  const dispatch = useDispatch();
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const enqueuedIdsRef = useRef(new Set());
  const keysByIdRef = useRef(new Map());

  useEffect(() => {
    const notifications = notification.notifications || [];

    notifications.forEach((item) => {
      if (enqueuedIdsRef.current.has(item.id)) {
        return;
      }

      const key = `notification-${item.id}`;
      enqueuedIdsRef.current.add(item.id);
      keysByIdRef.current.set(item.id, key);

      enqueueSnackbar(item.message, {
        key,
        variant: item.type || "info",
        autoHideDuration:
          item.autoHideDuration ?? (item.type === "error" ? 10000 : 6000),
        anchorOrigin: { vertical: "top", horizontal: "center" },
        action: (snackbarKey) => (
          <IconButton
            size="small"
            aria-label="close notification"
            color="inherit"
            onClick={() => closeSnackbar(snackbarKey)}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        ),
        onExited: () => {
          enqueuedIdsRef.current.delete(item.id);
          keysByIdRef.current.delete(item.id);
          dispatch(clearNotification(item.id));
        },
      });
    });
  }, [notification.notifications, enqueueSnackbar, closeSnackbar, dispatch]);

  useEffect(() => {
    const activeIds = new Set(
      (notification.notifications || []).map((n) => n.id),
    );

    for (const [id, key] of keysByIdRef.current.entries()) {
      if (!activeIds.has(id)) {
        closeSnackbar(key);
        keysByIdRef.current.delete(id);
        enqueuedIdsRef.current.delete(id);
      }
    }
  }, [notification.notifications, closeSnackbar]);

  return null;
}

function App() {
  // Notification state
  const dispatch = useDispatch();

  // Access global Redux state
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState,
  );
  const { isDarkMode } = useSelector(getThemeState);

  // Kiosk/touchscreen UI: #/mobile renders the reduced MobileApp shell
  // instead of the full desktop layout (see mobile/mobileRoutes.js).
  const { page: mobileKioskPage } = useMobileRoute();

  // Hook to detect mobile screens
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [sidebarVisible, setSidebarVisible] = useState(window.innerWidth > 768); // Sidebar visibility state - hidden by default on mobile
  const [prevIsMobile, setPrevIsMobile] = useState(window.innerWidth <= 768);

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      const newIsMobile = width <= 768;
      setIsMobile(newIsMobile);

      // Only close sidebar if switching from desktop to mobile
      if (!prevIsMobile && newIsMobile) {
        setSidebarVisible(false);
      }
      setPrevIsMobile(newIsMobile);
    };

    window.addEventListener("resize", handleResize);
    // Set initial state
    handleResize();
    return () => window.removeEventListener("resize", handleResize);
  }, [prevIsMobile]);

  const drawerWidth = sidebarVisible
    ? isMobile
      ? "100%"
      : 240
    : isMobile
      ? 0
      : 90; // Collapsed sidebar width on desktop

  const hostIP = connectionSettingsState.ip;
  const apiPort = connectionSettingsState.apiPort;

  const [selectedPlugin, setSelectedPlugin] = useState("LiveView"); // Control which plugin to show
  const [sharedImage, setSharedImage] = useState(null);
  const [fileManagerInitialPath, setFileManagerInitialPath] = useState("/");
  const [storageRefreshKey, setStorageRefreshKey] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [homingDialogOpen, setHomingDialogOpen] = useState(false);
  const [homingDialogBusy, setHomingDialogBusy] = useState(false);
  const [napariCommandDialog, setNapariCommandDialog] = useState({ open: false, command: "" });

  useBackendControllerCapabilities({
    hostIP,
    apiPort,
    selectedPlugin,
    setSelectedPlugin,
  });

  useEffect(() => {
    let cancelled = false;

    const checkHomingStatus = async () => {
      try {
        const homingStatus = await apiPositionerControllerGetHomingStatus();
        if (
          !cancelled &&
          !homingStatus?.hasHomedSinceStartup &&
          !homingStatus?.homingRecommendationDismissed
        ) {
          setHomingDialogOpen(true);
        }
      } catch (error) {
        console.warn("Failed to fetch homing status on connect", error);
      }
    };

    checkHomingStatus();

    return () => {
      cancelled = true;
    };
  }, [hostIP, apiPort]);

  const handleDismissHomingDialog = async () => {
    if (homingDialogBusy) {
      return;
    }

    setHomingDialogBusy(true);
    try {
      await apiPositionerControllerDismissHomingRecommendation();
      setHomingDialogOpen(false);
    } catch (error) {
      dispatch(
        setNotification({
          message:
            "Could not store homing prompt decision. The prompt may appear again after reload.",
          type: "error",
        }),
      );
    } finally {
      setHomingDialogBusy(false);
    }
  };

  // Route the startup homing recommendation to the FRAME-specific homing
  // procedure (FRAME Settings → "Frame Homing & Transport") rather than a
  // generic X/Y home call. FRAMESettingsController consumes the localStorage
  // key on mount to pre-select that tab.
  const handleOpenFrameHoming = () => {
    localStorage.setItem("frameSettings.initialTab", "frameHoming");
    setSelectedPlugin("FRAMESettings");
    setHomingDialogOpen(false);
  };

  /*
  FileManager
  */
  // Update fileUploadConfig to use hostIP (with protocol)
  const fileUploadConfig = {
    url: `${hostIP}:${apiPort}/imswitch/api/FileManager/upload`,
  };

  // Fetch Files
  const getFiles = async () => {
    setIsLoading(true);
    const response = await getAllFilesAPI();
    setFiles(response.data);
    setIsLoading(false);
  };

  const handleCreateFolder = async (name, parentFolder) => {
    setIsLoading(true);
    const response = await createFolderAPI(name, parentFolder?._id);
    if (response.status === 200 || response.status === 201) {
      setFiles((prev) => [...prev, response.data]);
    }
    setIsLoading(false);
  };

  const handleFileUploading = (file, parentFolder) => ({
    parentId: parentFolder?._id,
  });

  const handleFileUploaded = (response) => {
    const uploadedFile = JSON.parse(response);
    setFiles((prev) => [...prev, uploadedFile]);
  };

  const handleRename = async (file, newName) => {
    setIsLoading(true);
    await renameAPI(file._id, newName);
    getFiles();
    setIsLoading(false);
  };

  const handleDelete = async (files) => {
    setIsLoading(true);
    const idsToDelete = files.map((file) => file._id);
    await deleteAPI(idsToDelete);
    getFiles();
    setIsLoading(false);
  };

  const handlePaste = async (copiedItems, destinationFolder, operationType) => {
    setIsLoading(true);
    try {
      const copiedItemIds = copiedItems.map((item) => item._id);
      if (operationType === "copy") {
        await copyItemAPI(copiedItemIds, destinationFolder?._id);
      } else {
        await moveItemAPI(copiedItemIds, destinationFolder?._id);
      }
      getFiles();
    } catch (error) {
      const detail = error?.response?.data?.detail;
      const message =
        detail || "Copy/move failed. Please refresh and try again.";
      dispatch(setNotification({ message, type: "error" }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = async (files) => {
    await downloadFile(files, hostIP, apiPort);
  };

  const handleRefresh = () => getFiles();

  const handleStorageChange = (newPath) => {
    console.log("App: Storage changed to:", newPath);
    console.log("App: Previous initialPath was:", fileManagerInitialPath);
    // Always use "/" as initialPath - let the backend handle the actual storage location
    setFileManagerInitialPath("/");
    // Increment key to force FileManager remount
    setStorageRefreshKey((prev) => prev + 1);
    // Refresh files after small delay to ensure backend is ready
    setTimeout(() => {
      console.log("App: Refreshing FileManager with new path");
      handleRefresh();
    }, 200);
  };

  const handleOpenWithImJoy = (file) => {
    const fileUrl = `${hostIP}:${apiPort}/imswitch/api/FileManager/download/${file.path}`;
    setSharedImage({
      url: fileUrl,
      name: file.name,
    });
    // Switch to ImJoy tab
    setSelectedPlugin("ImJoy");
  };

  // Copy a `napari --plugin openuc2-processor <url>` command to the clipboard so
  // the user can download/process this dataset in the napari plugin.
  const handleOpenInNapari = (file) => {
    const cleanPath = file.path?.startsWith("/")
      ? file.path.slice(1)
      : file.path;
    const url = `${hostIP}:${apiPort}/imswitch/api/FileManager/download/${cleanPath}`;
    const command = `napari --plugin openuc2-processor "${url}"`;
    const notify = (variant, message) => {
      try {
        enqueueSnackbar(message, { variant });
      } catch (e) {
        /* SnackbarProvider not mounted */
      }
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard
        .writeText(command)
        .then(() => notify("success", "Copied napari command to clipboard"))
        .catch(() => {
          console.log(command);
          setNapariCommandDialog({ open: true, command });
        });
    } else {
      console.log(command);
      setNapariCommandDialog({ open: true, command });
    }
  };

  // Handler to open OME-Zarr files with the integrated Vizarr viewer
  const handleOpenWithVizarr = (file) => {
    console.log("[App] Opening file with Vizarr:", file);
    // Construct the relative path for the Zarr file
    // The path from FileManager is like "/recordings/experiment.ome.zarr"
    const zarrPath = file.path || `/${file.name}`;

    // Open the Vizarr viewer through Redux
    dispatch(
      vizarrViewerSlice.openViewer({
        url: zarrPath,
        fileName: file.name,
      }),
    );

    // Switch to the Vizarr viewer tab
    setSelectedPlugin("VizarrViewer");
  };

  // Get Vizarr viewer state
  const vizarrViewerState = useSelector(vizarrViewerSlice.getVizarrViewerState);

  // Handler to close Vizarr viewer
  const handleCloseVizarr = () => {
    dispatch(vizarrViewerSlice.closeViewer());
    // Optionally switch back to FileManager
    setSelectedPlugin("FileManager");
  };

  // change API url/port and update filelist
  useEffect(() => {
    api.defaults.baseURL = `${hostIP}:${apiPort}/imswitch/api/FileManager`;
    handleRefresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hostIP, apiPort]);

  // handle default filemanager path change
  const handleFileManagerInitialPathChange = (event) => {
    // English comment: store the desired path
    const path = event;
    setFileManagerInitialPath(path);
    setSelectedPlugin("FileManager");
    // Refresh immediately since FileNavigationProvider now handles path changes properly
    handleRefresh();
  };

  // Helper: handle menu click, close drawer on mobile
  const handlePluginChange = (plugin) => {
    setSelectedPlugin(plugin);
    if (isMobile) setSidebarVisible(false);
  };

  // Refresh files whenever the FileManager view is opened
  useEffect(() => {
    if (selectedPlugin === "FileManager") {
      handleRefresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPlugin]);

  /*
  PLUGIN LOADING from ImSwitch
  Plugins are served by the v2 PluginManager. Each plugin's remoteEntry.js
  lives at `remote_entry` (an absolute path already carrying the host's
  root_path, e.g. /imswitch/plugin/goniometer/ui/remoteEntry.js). Never build
  this URL by hand — see docs/plugins/DECISIONS.md.
  */
  function loadRemote({ name, remote_entry, scope, exposed }) {
    const url = `${hostIP}:${apiPort}${remote_entry}`;

    // Every failure path below MUST settle the promise. React's <Suspense>
    // has no timeout of its own: a promise that never settles renders the
    // fallback spinner forever with nothing in the console, which is the
    // worst possible first experience for a plugin author with a typo.
    return new Promise((resolve, reject) => {
      let settled = false;
      const fail = (message, cause) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        const error = new Error(
          `Plugin "${name || scope}" failed to load from ${url}: ${message}`,
        );
        if (cause) error.cause = cause;
        console.error(error);
        reject(error);
      };
      const succeed = (value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(value);
      };

      const timer = setTimeout(
        () =>
          fail(
            `timed out after ${PLUGIN_LOAD_TIMEOUT_MS / 1000}s. Check that the ` +
              `URL is reachable and that the bundle defines the federation ` +
              `scope "${scope}".`,
          ),
        PLUGIN_LOAD_TIMEOUT_MS,
      );

      if (!document.querySelector(`script[data-mf="${scope}"]`)) {
        const el = document.createElement("script");
        el.src = url;
        el.dataset.mf = scope;
        el.onload = init;
        el.onerror = () =>
          fail("the remoteEntry.js script could not be fetched (network/404)");
        document.head.appendChild(el);
      } else {
        // Already injected by an earlier mount — reuse it.
        init();
      }

      async function init() {
        try {
          await __webpack_init_sharing__("default");

          if (!window[scope]) {
            // Was: console.error + bare return, which left the promise pending
            // forever. Reject instead so the error boundary can render it.
            fail(
              `the bundle loaded but did not register federation scope ` +
                `"${scope}" on window. The scope in plugin.toml must match ` +
                `ModuleFederationPlugin.name in the plugin's webpack config.`,
            );
            return;
          }

          const container = window[scope];
          await container.init(__webpack_share_scopes__.default);

          const modulePath = exposed.startsWith("./") ? exposed : `./${exposed}`;
          const factory = await container.get(modulePath);
          if (!factory) {
            fail(
              `federation scope "${scope}" does not expose "${modulePath}". ` +
                `Check the plugin's ModuleFederationPlugin.exposes.`,
            );
            return;
          }

          const module = factory();
          succeed({ default: module.default || module }); // <-- important
        } catch (e) {
          fail(e?.message || String(e), e);
        }
      }
    });
  }

  function usePluginWidgets() {
    const [widgets, setWidgets] = useState([]);

    useEffect(() => {
      const fetchPlugins = async () => {
        try {
          // Construct the API URL dynamically using hostIP and apiPort
          const apiUrl = `${hostIP}:${apiPort}/imswitch/api/plugins`;

          // Fetch the plugin data
          const response = await fetch(apiUrl);
          if (!response.ok) {
            throw new Error(`Failed to fetch plugins: ${response.statusText}`);
          }

          const data = await response.json();
          const manifests = data.plugins || [];
          const backendErrors = data.errors || [];

          // Only plugins the backend actually mounted can be loaded. A
          // "disabled" entry has no remote_entry — it exists so the UI can
          // show it greyed out with a reason instead of hiding it.
          const loadable = manifests.filter(
            (m) => m.status !== "disabled" && m.remote_entry,
          );

          // A plugin can load a backend and ship no frontend bundle at all
          // (nothing built into ui/dist). It has working endpoints but no
          // widget, so it cannot go in the sidebar — report it instead of
          // dropping it, or its absence looks identical to a failed load.
          const backendOnly = manifests.filter(
            (m) => m.status !== "disabled" && !m.remote_entry,
          );

          setWidgets(
            loadable.map((m) => ({
              name: m.name,
              // Hand the plugin the URLs the backend published rather than
              // letting it build them. They already carry the host's
              // root_path, which a plugin has no way to know.
              apiBase: m.api_base,
              socketNs: m.socket_ns,
              Component: lazy(() => loadRemote(m)), // Wrap loadRemote with lazy
            })),
          );

          // Merge into the app registry so plugins appear in the sidebar and
          // participate in the App Manager enable/disable flow like any
          // built-in app.
          dispatch(
            setDynamicApps({
              apps: loadable.map(makeRegistryEntryFromManifest),
              errors: [
                ...backendErrors.map((e) => ({
                  name: e.source,
                  reason: e.error,
                  kind: "error",
                })),
                ...manifests
                  .filter((m) => m.status === "disabled")
                  .map((m) => ({
                    name: m.display_name || m.name,
                    reason: m.reason || "disabled by the setup file",
                    kind: "disabled",
                  })),
                ...backendOnly.map((m) => ({
                  name: m.display_name || m.name,
                  reason:
                    `Loaded, and its API is available at ${m.api_base}, but it ` +
                    `ships no frontend bundle — nothing was found in the ` +
                    `plugin's ui dist_dir, so there is no widget to show.`,
                  kind: "no-ui",
                })),
              ],
            }),
          );
        } catch (error) {
          console.error("Error loading plugins:", error);
          dispatch(
            setDynamicApps({
              apps: [],
              errors: [
                {
                  name: "plugin discovery",
                  reason: `Could not reach ${hostIP}:${apiPort}/imswitch/api/plugins — ${error.message}`,
                  kind: "error",
                },
              ],
            }),
          );
        }
      };

      fetchPlugins();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [hostIP, apiPort]); // Re-run if hostIP or apiPort changes

    return widgets;
  }
  const plugins = usePluginWidgets();

  // Kiosk branch: mount only the cross-cutting providers (single
  // WebSocketHandler!) and the touch shell — no drawer/topbar/file manager.
  if (mobileKioskPage !== null) {
    return (
      <PWAProvider>
        <ThemeProvider theme={darkTheme}>
          <SnackbarProvider maxSnack={6} dense>
            <ReduxNotificationBridge />
            <WebSocketHandler />
            <MobileApp />
          </SnackbarProvider>
        </ThemeProvider>
      </PWAProvider>
    );
  }

  return (
    <PWAProvider>
      <ThemeProvider theme={isDarkMode ? darkTheme : lightTheme}>
        <SnackbarProvider maxSnack={6} dense>
          <ReduxNotificationBridge />
          <WebSocketHandler />
          <OnboardingTour selectedPlugin={selectedPlugin} />
          <CssBaseline />

          <Dialog
            open={napariCommandDialog.open}
            onClose={() => setNapariCommandDialog({ open: false, command: "" })}
            fullWidth
            maxWidth="sm"
          >
            <DialogTitle>Napari Command</DialogTitle>
            <DialogContent>
              <DialogContentText sx={{ mb: 2 }}>
                Install the napari plugin using: "pip install openuc2-processor"
                Clipboard is not available in this context. Copy the command below manually:
              </DialogContentText>
              <TextField
                fullWidth
                multiline
                value={napariCommandDialog.command}
                onChange={(e) =>
                  setNapariCommandDialog((prev) => ({ ...prev, command: e.target.value }))
                }
                inputProps={{ spellCheck: false }}
                variant="outlined"
                size="small"
              />
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setNapariCommandDialog({ open: false, command: "" })}>
                Close
              </Button>
            </DialogActions>
          </Dialog>

          <Dialog
            open={homingDialogOpen}
            onClose={handleDismissHomingDialog}
            disableEscapeKeyDown={homingDialogBusy}
          >
            <DialogTitle>Homing Recommended</DialogTitle>
            <DialogContent>
              <DialogContentText>
                The machine has not been homed since backend startup. Homing is
                recommended before moving the stage.
              </DialogContentText>
            </DialogContent>
            <DialogActions>
              <Button
                onClick={handleDismissHomingDialog}
                disabled={homingDialogBusy}
              >
                Continue without homing
              </Button>
              <Button
                variant="contained"
                onClick={handleOpenFrameHoming}
                disabled={homingDialogBusy}
              >
                Open Frame Homing
              </Button>
            </DialogActions>
          </Dialog>

          <Box sx={{ display: "flex" }}>
            <NavigationDrawer
              sidebarVisible={sidebarVisible}
              setSidebarVisible={setSidebarVisible}
              isMobile={isMobile}
              drawerWidth={drawerWidth}
              selectedPlugin={selectedPlugin}
              handlePluginChange={handlePluginChange}
            />

            <TopBar
              isMobile={isMobile}
              sidebarVisible={sidebarVisible}
              setSidebarVisible={setSidebarVisible}
              selectedPlugin={selectedPlugin}
              onSettingsNavigate={handlePluginChange} // Pass existing navigation handler
              onStorageChange={handleStorageChange}
            />

            <Box
              component="main"
              sx={{
                top: 64,
                flexGrow: 1,
                display: "flex",
                position: "absolute",
                p:
                  selectedPlugin === "JupyterNotebook" ||
                  selectedPlugin === "ImJoy"
                    ? 0
                    : isMobile
                      ? 1
                      : 3,
                left: drawerWidth,
                width: "calc(100% - " + drawerWidth + "px)",
                height: "calc(100vh - 64px)",
                marginLeft: !isMobile && sidebarVisible ? 0 : 0,
                transition: (theme) =>
                  theme.transitions.create(["margin", "padding"], {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.leavingScreen,
                  }),
                minHeight: "calc(100vh - 64px)",
                overflow:
                  selectedPlugin === "JupyterNotebook" ||
                  selectedPlugin === "ImJoy"
                    ? "hidden"
                    : "auto",
              }}
            >
              {selectedPlugin === "LiveView" && (
                <LiveView
                  // pass down a setter or context for the image if needed
                  setFileManagerInitialPath={handleFileManagerInitialPathChange} // pass function
                />
              )}

              {selectedPlugin === "WellPlate" && <AxonTabComponent />}
              {selectedPlugin === "GalvoScannerController" && (
                <GalvoScannerController />
              )}
              {selectedPlugin === "ShitScope" && (
                <ShitScopeComponent
                  onOpenFileManager={handleFileManagerInitialPathChange}
                />
              )}
              {selectedPlugin === "ImJoy" && (
                <ImJoyView sharedImage={sharedImage} />
              )}
              {selectedPlugin === "STORMLocal" && <STORMControllerLocal />}
              {selectedPlugin === "STORMArkitekt" && (
                <STORMControllerArkitekt />
              )}
              {selectedPlugin === "FRAMESettings" && (
                <FRAMESettingsController />
              )}
              {selectedPlugin === "Stresstest" && <StresstestController />}
              {selectedPlugin === "FocusLock" && <FocusLockController />}
              {selectedPlugin === "I2CSensor" && <I2CSensorController />}
              {selectedPlugin === "AcceptanceTest" && (
                <AcceptanceTestComponent />
              )}
              {selectedPlugin === "HoloController" && <HoloController />}
              {selectedPlugin === "OffAxisHoloController" && (
                <OffAxisHoloController />
              )}
              {selectedPlugin === "DPCController" && <DPCController />}
              {selectedPlugin === "JupyterNotebook" && (
                <Box sx={{ width: "100%", height: "100%", minHeight: 0 }}>
                  <JupyterProvider>
                    <JupyterExecutor />
                  </JupyterProvider>
                </Box>
              )}
              {selectedPlugin === "GoniometerController" && (
                <GoniometerController />
              )}
              {selectedPlugin === "Infinity Scanning" && (
                <LargeFovScanController />
              )}
              {selectedPlugin === "Blockly" && <BlocklyController />}
              {selectedPlugin === "Objective" && <ObjectiveController />}
              {selectedPlugin === "About" && <AboutPage />}
              {selectedPlugin === "SystemSettings" && <SystemSettings />}
              {selectedPlugin === "MotorSettings" && (
                <MotorSettingsController />
              )}
              {selectedPlugin === "FileManager" && (
                <div
                  className="app"
                  style={{ width: "100%", maxWidth: "100%" }}
                >
                  <div
                    className="file-manager-container"
                    style={{ width: "100%", maxWidth: "100%" }}
                  >
                    <FileManager
                      key={`fm-${storageRefreshKey}`} // Force remount on storage change
                      baseUrl={`${hostIP}:${apiPort}/imswitch/api`}
                      files={files}
                      fileUploadConfig={fileUploadConfig}
                      isLoading={isLoading}
                      onCreateFolder={handleCreateFolder}
                      onFileUploading={handleFileUploading}
                      onFileUploaded={handleFileUploaded}
                      onPaste={handlePaste}
                      onRename={handleRename}
                      onDownload={handleDownload}
                      onFileOpen={handleOpenWithImJoy}
                      onOpenWithVizarr={handleOpenWithVizarr}
                      onOpenInNapari={handleOpenInNapari}
                      onDelete={handleDelete}
                      onRefresh={handleRefresh}
                      layout="list"
                      enableFilePreview
                      maxFileSize={10485760}
                      filePreviewPath={`${hostIP}:${apiPort}/imswitch/api`}
                      acceptedFileTypes=".txt, .png, .jpg, .jpeg, .pdf, .doc, .docx, .exe, .js, .csv"
                      initialPath={fileManagerInitialPath}
                    />
                  </div>
                </div>
              )}
              {selectedPlugin === "VizarrViewer" && (
                <Box sx={{ width: "100%", height: "calc(100vh - 64px)" }}>
                  <VizarrViewer
                    zarrUrl={vizarrViewerState.currentUrl}
                    onClose={handleCloseVizarr}
                    height="100%"
                    width="100%"
                  />
                </Box>
              )}
              {selectedPlugin === "AppManager" && (
                <AppManagerPage onNavigateToApp={handlePluginChange} />
              )}
              {selectedPlugin === "LightSheet" && <LightsheetController />}
              {selectedPlugin === "Timelapse" && <TimelapseController />}
              {selectedPlugin === "MMCore" && <MMCoreController />}

              {selectedPlugin === "WiFi" && <WiFiController />}
              {/*
                Federated plugin widgets. This block sits inside <Provider>
                (from index.js) and <ThemeProvider> (above), so a plugin gets
                the host's Redux store and MUI theme through context — no props
                and no bridge object — provided react-redux and @mui/material
                are federation singletons. See frontend/shared-deps.js.

                hostIP/hostPort are DEPRECATED and kept only for the existing
                goniometer plugin. New plugins read the backend URL from
                useSelector((s) => s.connectionSettingsState) instead.
              */}
              {plugins.map(
                (p) =>
                  selectedPlugin === p.name && (
                    <PluginErrorBoundary key={p.name} pluginName={p.name}>
                      <Suspense fallback={<div>loading…</div>}>
                        <p.Component
                          apiBase={`${hostIP}:${apiPort}${p.apiBase}`}
                          socketNs={p.socketNs}
                          hostIP={hostIP}
                          hostPort={apiPort}
                        />
                      </Suspense>
                    </PluginErrorBoundary>
                  ),
              )}
              {selectedPlugin === "DemoController" && <DemoController />}
              {selectedPlugin === "StageMap" && <StageMapController />}
              {selectedPlugin === "CompositeAcquisition" && (
                <CompositeAcquisitionComponent />
              )}
              {selectedPlugin === "CompositeStreamViewer" && (
                <CompositeStreamViewer />
              )}
              {selectedPlugin === "CompositeComponent" && (
                <CompositeComponent />
              )}
              {selectedPlugin === "FlowStop" && <FlowStopController />}
              {selectedPlugin === "UC2" && <UC2ConfigurationController />}
              {selectedPlugin === "SerialDebug" && <SerialDebugController />}
              {selectedPlugin === "DetectorTrigger" && (
                <DetectorTriggerController />
              )}
              {selectedPlugin === "ExtendedLEDMatrix" && (
                <ExtendedLEDMatrixController />
              )}
              {selectedPlugin === "Lepmon" && <LepMonController />}
              {selectedPlugin === "MazeGame" && <MazeGameController />}
              {selectedPlugin === "SocketView" && <SocketView />}
              {selectedPlugin === "SystemUpdate" && <SystemUpdateController />}
              {selectedPlugin === "Connections" && <ConnectionSettings />}
              {selectedPlugin === "DesktopApp" && <DesktopAppSettings />}
              {selectedPlugin === "Logging" && <LoggingController />}
            </Box>
          </Box>
        </SnackbarProvider>
      </ThemeProvider>
    </PWAProvider>
  );
}

export default App;
