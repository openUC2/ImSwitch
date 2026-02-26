/* global __webpack_init_sharing__, __webpack_share_scopes__ */
import { lazy, Suspense, useEffect, useState } from "react";

// ImSwitch Themes
import { darkTheme, lightTheme } from "./themes";

import AboutPage from "./components/AboutPage.js";
import BlocklyController from "./components/BlocklyController.js";
import ConnectionSettings from "./components/ConnectionSettings.jsx";
import DetectorTriggerController from "./components/DetectorTriggerController.js";
import ExtendedLEDMatrixController from "./components/ExtendedLEDMatrixController.jsx";
import FlowStopController from "./components/FlowStopController.js";
import FocusLockController from "./components/FocusLockController.js";
import HoloController from "./components/HoloController.js";
import OffAxisHoloController from "./components/OffAxisHoloController.js";
import DPCController from "./components/DPCController.js";
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
import StageOffsetCalibration from "./components/StageOffsetCalibrationController.jsx";
import STORMControllerArkitekt from "./components/STORMControllerArkitekt.js";
import FRAMESettingsController from "./components/FRAMESettingsController.js";
import STORMControllerLocal from "./components/STORMControllerLocal.js";
import StresstestController from "./components/StresstestController.js";
import SystemSettings from "./components/SystemSettings.js";
import SystemUpdateController from "./components/SystemUpdateController.jsx";
import TimelapseController from "./components/TimelapseController.js";
import UC2ConfigurationController from "./components/UC2ConfigurationController.jsx";
import SerialDebugController from "./components/SerialDebugController.jsx";
import WiFiController from "./components/WiFiController.jsx";
import LoggingController from "./components/LoggingController.jsx";
import VizarrViewer from "./components/VizarrViewer.jsx";
import { JupyterProvider } from "./context/JupyterContext.js";
import DemoController from "./components/DemoController.js";
import AcceptanceTestComponent from "./components/AcceptanceTestComponent.jsx";
import StageCenterCalibrationWizard from "./components/StageCenterCalibrationWizard.js";
import GalvoScannerController from "./components/GalvoScannerController.jsx";

// ImSwitch Navigation Drawer
import { NavigationDrawer, TopBar } from "./components/navigation";
import AppManagerPage from "./components/AppManagerPage.jsx";

//axon
import AxonTabComponent from "./axon/AxonTabComponent.js";
import WebSocketHandler from "./middleware/WebSocketHandler.js";
import CompositeAcquisitionComponent from "./axon/CompositeAcquisitionComponent.js";
import CompositeStreamViewer from "./axon/CompositeStreamViewer.js";
import CompositeComponent from "./axon/CompositeComponent";

//redux
import { useDispatch, useSelector } from "react-redux";
import StatusMessage from "./components/StatusMessage.js";
import * as connectionSettingsSlice from "./state/slices/ConnectionSettingsSlice.js";
import * as vizarrViewerSlice from "./state/slices/VizarrViewerSlice.js";
import {
  clearNotification,
  getNotificationState,
  setNotification,
} from "./state/slices/NotificationSlice.js";
import { getThemeState } from "./state/slices/ThemeSlice.js";

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

import { Box, CssBaseline } from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";

function App() {
  // Notification state
  const notification = useSelector(getNotificationState);
  const dispatch = useDispatch();

  // Access global Redux state
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState,
  );
  const { isDarkMode } = useSelector(getThemeState);

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

  /*
  PLUGIN LOADING from ImSwitch
  */
  function loadRemote({ remote, scope, exposed }) {
    const url = `${hostIP}:${apiPort}/imswitch/api${remote}`;

    return new Promise((resolve, reject) => {
      if (!document.querySelector(`script[data-mf="${scope}"]`)) {
        const el = document.createElement("script");
        el.src = url;
        el.dataset.mf = scope;
        el.onload = init;
        el.onerror = reject;
        document.head.appendChild(el);
      } else {
        init();
      }

      async function init() {
        try {
          await __webpack_init_sharing__("default");

          // Check if the scope is loaded
          if (!window[scope]) {
            console.error(
              `Scope '${scope}' not found on window. Make sure the script is loaded correctly.`,
            );
            return;
          }
          const container = window[scope];
          await container.init(__webpack_share_scopes__.default);

          const factory = await container.get(
            exposed.startsWith("./") ? exposed : `./${exposed}`,
          );
          const module = factory();
          resolve({ default: module.default || module }); // <-- important
        } catch (e) {
          reject(e);
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

          // Access the plugins array from the data object
          const data = await response.json();
          const plugins = data.plugins;

          // Map the plugin data to widgets
          const widgetsData = await Promise.all(
            plugins.map(async (m) => ({
              name: m.name,
              Component: lazy(() => loadRemote(m)), // Wrap loadRemote with lazy
            })),
          );

          // Update the widgets state
          setWidgets(widgetsData);
        } catch (error) {
          console.error("Error loading plugins:", error);
        }
      };

      fetchPlugins();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [hostIP, apiPort]); // Re-run if hostIP or apiPort changes

    return widgets;
  }
  const plugins = usePluginWidgets();

  return (
    <ThemeProvider theme={isDarkMode ? darkTheme : lightTheme}>
      <WebSocketHandler />
      <CssBaseline />

      {/* Global Status/Notification Message */}
      <StatusMessage
        message={notification.message}
        type={notification.type}
        onClose={() => dispatch(clearNotification())}
      />

      <Box sx={{ display: "flex" }}>
        <NavigationDrawer
          sidebarVisible={sidebarVisible}
          setSidebarVisible={setSidebarVisible}
          isMobile={isMobile}
          drawerWidth={drawerWidth}
          selectedPlugin={selectedPlugin}
          handlePluginChange={handlePluginChange}
          plugins={plugins}
        />

        <TopBar
          isMobile={isMobile}
          sidebarVisible={sidebarVisible}
          setSidebarVisible={setSidebarVisible}
          selectedPlugin={selectedPlugin}
          onSettingsNavigate={handlePluginChange} // Pass existing navigation handler
          onFileManagerRefresh={handleRefresh}
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
              selectedPlugin === "JupyterNotebook" || selectedPlugin === "ImJoy"
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
              selectedPlugin === "JupyterNotebook" || selectedPlugin === "ImJoy"
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
          {selectedPlugin === "ImJoy" && (
            <ImJoyView sharedImage={sharedImage} />
          )}
          {selectedPlugin === "STORMLocal" && <STORMControllerLocal />}
          {selectedPlugin === "STORMArkitekt" && <STORMControllerArkitekt />}
          {selectedPlugin === "FRAMESettings" && <FRAMESettingsController />}
          {selectedPlugin === "Stresstest" && <StresstestController />}
          {selectedPlugin === "FocusLock" && <FocusLockController />}
          {selectedPlugin === "AcceptanceTest" && <AcceptanceTestComponent />}
          {selectedPlugin === "StageCenterCalibration" && (
            <StageCenterCalibrationWizard />
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
          {selectedPlugin === "Infinity Scanning" && <LargeFovScanController />}
          {selectedPlugin === "Blockly" && <BlocklyController />}
          {selectedPlugin === "Objective" && <ObjectiveController />}
          {selectedPlugin === "About" && <AboutPage />}
          {selectedPlugin === "SystemSettings" && <SystemSettings />}
          {selectedPlugin === "MotorSettings" && <MotorSettingsController />}
          {selectedPlugin === "FileManager" && (
            <div className="app" style={{ width: "100%", maxWidth: "100%" }}>
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
          {selectedPlugin === "WiFi" && <WiFiController />}
          {plugins.map(
            (p) =>
              selectedPlugin === p.name && (
                <Suspense fallback={<div>loading…</div>} key={p.name}>
                  <p.Component hostIP={hostIP} hostPort={apiPort} />
                </Suspense>
              ),
          )}
          {selectedPlugin === "DemoController" && <DemoController />}
          {selectedPlugin === "CompositeAcquisition" && (
            <CompositeAcquisitionComponent />
          )}
          {selectedPlugin === "CompositeStreamViewer" && (
            <CompositeStreamViewer />
          )}
          {selectedPlugin === "CompositeComponent" && <CompositeComponent />}
          {selectedPlugin === "FlowStop" && <FlowStopController />}
          {selectedPlugin === "StageOffsetCalibration" && (
            <StageOffsetCalibration />
          )}
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
          {selectedPlugin === "Logging" && <LoggingController />}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
