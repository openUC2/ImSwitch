// src/constants/appRegistry.js
// Registry of all available applications/components in ImSwitch
// Each app has metadata for display, categorization, and management

import {
  AccessTime as AccessTimeIcon,
  AutoFixHigh as AutoFixHighIcon,
  Cable as CableIcon,
  Dashboard as DashboardIcon,
  Extension as ExtensionIcon,
  Folder as FolderIcon,
  GridOn as GridOnIcon,
  GridView as GridViewIcon,
  Lock as LockIcon,
  SportsEsports as SportsEsportsIcon,
  ThreeDRotation as ThreeDRotationIcon,
  // Additional icons for better differentiation
  ViewModule as ViewModuleIcon,
  CropFree as CropFreeIcon,
  FlashOn as FlashOnIcon,
  Tune as TuneIcon,
  NetworkCheck as NetworkCheckIcon,
  DeviceHub as DeviceHubIcon,
  Speed as SpeedIcon,
  Thermostat as ThermostatIcon,
  Insights as InsightsIcon,
  PhotoCamera as PhotoCameraIcon,
  Visibility as VisibilityIcon,
  ViewInAr as ViewInArIcon,
  // Coding-specific icons
  MenuBook as MenuBookIcon,
  Psychology as PsychologyIcon,
} from "@mui/icons-material";

/**
 * Categories for organizing applications
 */
export const APP_CATEGORIES = {
  ESSENTIALS: "essentials",
  APPS: "apps",
  CODING: "coding",
  SYSTEM: "system",
};

/**
 * Complete registry of all available applications
 * Each app entry contains:
 * - id: unique identifier
 * - name: display name
 * - description: what the app does
 * - category: which category it belongs to
 * - icon: Material-UI icon component
 * - enabled: default enabled state
 * - essential: cannot be disabled (for essentials)
 * - keywords: for search functionality
 * - pluginId: corresponding plugin identifier
 */
export const APP_REGISTRY = {
  // === ESSENTIALS - Always enabled ===
  liveView: {
    id: "liveView",
    name: "Live View",
    description:
      "Main microscopy interface for real-time imaging, intensity controls, and stage movement. Essential for all microscopy operations.",
    category: APP_CATEGORIES.ESSENTIALS,
    icon: DashboardIcon,
    enabled: true,
    essential: true,
    keywords: ["live", "camera", "microscopy", "imaging", "realtime", "stage"],
    pluginId: "LiveView",
  },

  fileManager: {
    id: "fileManager",
    name: "File Manager",
    description:
      "Browse, organize, and manage microscopy data files. Upload, download, and preview images and experimental data.",
    category: APP_CATEGORIES.ESSENTIALS,
    icon: FolderIcon,
    enabled: true,
    essential: true,
    keywords: ["files", "data", "browse", "upload", "download", "organize"],
    pluginId: "FileManager",
  },

  vizarrViewer: {
    id: "VizarrViewer",
    name: "OME-Zarr Viewer",
    description:
      "Offline OME-Zarr viewer for multidimensional microscopy data. View and navigate through large datasets locally without internet connection.",
    category: APP_CATEGORIES.ESSENTIALS,
    icon: ViewInArIcon,
    enabled: true,
    essential: true,
    keywords: [
      "zarr",
      "omezarr",
      "viewer",
      "3d",
      "multidimensional",
      "offline",
      "vizarr",
    ],
    pluginId: "VizarrViewer",
  },

  appManager: {
    id: "appManager",
    name: "App Manager",
    description:
      "Customize your workspace by enabling or disabling apps in the navigation drawer. Tailor ImSwitch to your specific workflow needs.",
    category: APP_CATEGORIES.ESSENTIALS,
    icon: GridViewIcon,
    enabled: true,
    essential: true,
    keywords: [
      "apps",
      "manage",
      "customize",
      "workspace",
      "enable",
      "disable",
      "navigation",
    ],
    pluginId: "AppManager",
  },

  // === APPS - Optional microscopy applications ===
  wellSelector: {
    id: "wellSelector",
    name: "Well Plate",
    description:
      "Multi-well plate navigation and selection. Choose wells for automated experiments and sample screening.",
    category: APP_CATEGORIES.APPS,
    icon: GridOnIcon,
    enabled: false,
    essential: false,
    keywords: ["well", "plate", "multiwell", "selection", "screening"],
    pluginId: "WellPlate",
  },

  stormLocal: {
    id: "stormLocal",
    name: "STORM Local",
    description:
      "Super-resolution STORM microscopy processing. Local reconstruction and analysis of single-molecule data.",
    category: APP_CATEGORIES.APPS,
    icon: AutoFixHighIcon,
    enabled: false,
    essential: false,
    keywords: ["storm", "superresolution", "reconstruction", "singlemolecule"],
    pluginId: "STORMLocal",
  },

  stormArkitekt: {
    id: "stormArkitekt",
    name: "STORM Arkitekt",
    description:
      "Advanced STORM microscopy controller with Arkitekt integration. Cloud-based super-resolution image processing and workflow management.",
    category: APP_CATEGORIES.APPS,
    icon: InsightsIcon,
    enabled: false,
    essential: false,
    keywords: [
      "storm",
      "arkitekt",
      "superresolution",
      "cloud",
      "workflow",
      "processing",
    ],
    pluginId: "STORMArkitekt",
  },

  demoController: {
    id: "demoController",
    name: "Demo Controller",
    description:
      "Simulated microscopy environment for training and testing. Explore ImSwitch features without hardware.",
    category: APP_CATEGORIES.APPS,
    icon: SportsEsportsIcon,
    enabled: false,
    essential: false,
    keywords: ["demo", "simulation", "training", "testing", "features"],
    pluginId: "DemoController",
  },

  frameSettings: {
    id: "frameSettings",
    name: "FRAMESettings",
    description:
      "Comprehensive settings for FRAMEModule including pixel calibration, stage tracking, and laser configuration.",
    category: APP_CATEGORIES.APPS,
    icon: CropFreeIcon,
    enabled: false,
    essential: false,
    keywords: [
      "frame",
      "settings",
      "pixel",
      "calibration",
      "stage",
      "tracking",
      "laser",
      "configuration",
    ],
    pluginId: "FRAMESettings",
  },

  infinityScanning: {
    id: "infinityScanning",
    name: "Infinity Scanning",
    description:
      "Large field-of-view scanning microscopy with OpenLayers integration. Create stitched panoramic images from multiple high-resolution tiles.",
    category: APP_CATEGORIES.APPS,
    icon: CropFreeIcon,
    enabled: false,
    essential: false,
    keywords: [
      "infinity",
      "scanning",
      "largefov",
      "panoramic",
      "stitching",
      "openlayers",
    ],
    pluginId: "Infinity Scanning",
  },

  lightsheet: {
    id: "lightsheet",
    name: "LightSheet",
    description:
      "Light-sheet microscopy control for 3D imaging. Optimized for large sample volumes and reduced phototoxicity.",
    category: APP_CATEGORIES.APPS,
    icon: ThreeDRotationIcon,
    enabled: false,
    essential: false,
    keywords: ["lightsheet", "3d", "volume", "phototoxicity", "large"],
    pluginId: "LightSheet",
  },

  timelapse: {
    id: "timelapse",
    name: "Timelapse",
    description:
      "Time-lapse microscopy with automated image acquisition. Schedule multi-position experiments over time.",
    category: APP_CATEGORIES.APPS,
    icon: AccessTimeIcon,
    enabled: false,
    essential: false,
    keywords: ["timelapse", "time", "automated", "acquisition", "scheduling"],
    pluginId: "Timelapse",
  },

  flowStop: {
    id: "flowStop",
    name: "Flow Stop",
    description:
      "Control flow stop mechanisms for microfluidics experiments. Manage fluid flow and timing.",
    category: APP_CATEGORIES.APPS,
    icon: DeviceHubIcon,
    enabled: false,
    essential: false,
    keywords: ["flow", "stop", "microfluidics", "fluid", "timing"],
    pluginId: "FlowStop",
  },

  lepmon: {
    id: "lepmon",
    name: "Lepmon Controller",
    description:
      "Lepmon thermal imaging and monitoring. Control thermal sensors and imaging.",
    category: APP_CATEGORIES.APPS,
    icon: ThermostatIcon,
    enabled: false,
    essential: false,
    keywords: ["lepmon", "thermal", "imaging", "monitoring", "sensors"],
    pluginId: "Lepmon",
  },

  mazeGame: {
    id: "mazeGame",
    name: "Maze Game",
    description:
      "Interactive game for microscope control training. Learn stage navigation through engaging gameplay.",
    category: APP_CATEGORIES.APPS,
    icon: SportsEsportsIcon,
    enabled: false,
    essential: false,
    keywords: ["game", "training", "interactive", "navigation", "learn"],
    pluginId: "MazeGame",
  },

  // === CODING - Development and debugging tools ===
  blockly: {
    id: "blockly",
    name: "Blockly Controller",
    description:
      "Visual programming interface for creating custom microscopy workflows. Drag-and-drop programming for experiments.",
    category: APP_CATEGORIES.CODING,
    icon: ExtensionIcon,
    enabled: false,
    essential: false,
    keywords: ["blockly", "visual", "programming", "workflow", "experiments"],
    pluginId: "Blockly",
  },

  imjoy: {
    id: "imjoy",
    name: "ImJoy Integration",
    description:
      "Integration with ImJoy platform for advanced image analysis. Run Python and JavaScript plugins in the browser.",
    category: APP_CATEGORIES.CODING,
    icon: PsychologyIcon,
    enabled: false,
    essential: false,
    keywords: ["imjoy", "analysis", "python", "javascript", "plugins"],
    pluginId: "ImJoy",
  },

  jupyterNotebook: {
    id: "jupyterNotebook",
    name: "Jupyter Notebook",
    description:
      "Interactive development environment for Python scripting and data analysis. Execute code cells, visualize data, and create documentation.",
    category: APP_CATEGORIES.CODING,
    icon: MenuBookIcon,
    enabled: false,
    essential: false,
    keywords: [
      "jupyter",
      "notebook",
      "python",
      "interactive",
      "development",
      "analysis",
    ],
    pluginId: "JupyterNotebook",
  },

  // === SYSTEM - Configuration and system tools ===
  focusLock: {
    id: "focusLock",
    name: "Focus Lock",
    description:
      "Hardware-based focus stabilization system. Maintains focus during long-term experiments using feedback control.",
    category: APP_CATEGORIES.SYSTEM,
    icon: LockIcon,
    enabled: false,
    essential: false,
    keywords: ["focus", "lock", "stabilization", "feedback", "longterm"],
    pluginId: "FocusLock",
  },

  acceptanceTest: {
    id: "acceptanceTest",
    name: "Acceptance Test",
    description:
      "Automated acceptance testing suite for system validation. Run predefined tests to verify hardware and software functionality.",
    category: APP_CATEGORIES.SYSTEM,
    icon: PsychologyIcon,
    enabled: false,
    essential: false,
    keywords: [
      "acceptance",
      "test",
      "automation",
      "validation",
      "hardware",
      "software",
    ],
    pluginId: "AcceptanceTest",
  },

  StageCenterCalibration: {
    id: "StageCenterCalibration",
    name: "Stage Center Calibration",
    description:
      "Wizard-guided calibration of stage center position. Interactive tools to accurately define the center of the microscope stage.",
    category: APP_CATEGORIES.SYSTEM,
    icon: TuneIcon,
    enabled: false,
    essential: false,
    keywords: [
      "stage",
      "center",
      "calibration",
      "wizard",
      "interactive",
      "positioning",
    ],
    pluginId: "StageCenterCalibration",
  },

  holoController: {
    id: "holoController",
    name: "Hologram Processing",
    description:
      "Inline hologram processing with Fresnel propagation. Real-time holographic reconstruction for quantitative phase imaging.",
    category: APP_CATEGORIES.APPS,
    icon: VisibilityIcon,
    enabled: false,
    essential: false,
    keywords: [
      "hologram",
      "holography",
      "fresnel",
      "propagation",
      "phase",
      "inline",
      "reconstruction",
    ],
    pluginId: "HoloController",
  },

  dpcController: {
    id: "dpcController",
    name: "DPC Imaging",
    description:
      "Differential Phase Contrast imaging with 4-pattern LED matrix illumination. Real-time phase gradient reconstruction for label-free imaging.",
    category: APP_CATEGORIES.APPS,
    icon: VisibilityIcon,
    enabled: false,
    essential: false,
    keywords: [
      "dpc",
      "differential",
      "phase",
      "contrast",
      "gradient",
      "led",
      "matrix",
      "illumination",
      "label-free",
    ],
    pluginId: "DPCController",
  },

  serialDebug: {
    id: "serialDebug",
    name: "Serial Debug",
    description:
      "Debug and monitor serial communication bridges. Troubleshoot hardware connections and protocol issues.",
    category: APP_CATEGORIES.SYSTEM,
    icon: CableIcon,
    enabled: false,
    essential: false,
    keywords: ["serial", "bridge", "debug", "hardware", "communication"],
    pluginId: "SerialDebug",
  },

  // === MISSING APPS - Apps that exist in App.jsx but were not in registry ===
  objective: {
    id: "objective",
    name: "Objective Controller",
    description:
      "Control motorized objective turret. Switch between objectives during experiments for multi-magnification imaging.",
    category: APP_CATEGORIES.SYSTEM,
    icon: PhotoCameraIcon,
    enabled: false,
    essential: false,
    keywords: ["objective", "lens", "turret", "magnification", "switch"],
    pluginId: "Objective",
  },

  stresstest: {
    id: "stresstest",
    name: "Stress Test",
    description:
      "System stress testing and performance monitoring. Test hardware limits and system stability.",
    category: APP_CATEGORIES.SYSTEM,
    icon: SpeedIcon,
    enabled: false,
    essential: false,
    keywords: ["stress", "test", "performance", "stability", "hardware"],
    pluginId: "Stresstest",
  },

  stageOffsetCalibration: {
    id: "stageOffsetCalibration",
    name: "Stage Offset Calibration",
    description:
      "Calibrate stage positioning offsets for accurate movement. Compensate for mechanical tolerances and improve positioning precision.",
    category: APP_CATEGORIES.SYSTEM,
    icon: TuneIcon,
    enabled: false,
    essential: false,
    keywords: [
      "stage",
      "offset",
      "calibration",
      "positioning",
      "precision",
      "mechanical",
    ],
    pluginId: "StageOffsetCalibration",
  },

  // adding CompositeStreamViewer and CompositeAcquisition to registry
  compositeAcquisition: {
    id: "compositeAcquisition",
    name: "Composite Acquisition",
    description:
      "Control multi-illumination composite image acquisition. Configure LED patterns and capture fused RGB images.",
    category: APP_CATEGORIES.APPS,
    icon: PhotoCameraIcon,
    enabled: false,
    essential: false,
    keywords: [
      "composite",
      "acquisition",
      "multi-illumination",
      "led",
      "rgb",
      "fused",
    ],
    pluginId: "CompositeAcquisition",
  },

  compositeStreamViewer: {
    id: "compositeStreamViewer",
    name: "Composite Stream Viewer",
    description:
      "View live fused RGB composite images from multi-illumination acquisition. Real-time display of MJPEG stream from backend.",
    category: APP_CATEGORIES.APPS,
    icon: ViewModuleIcon,
    enabled: false,
    essential: false,
    keywords: [
      "composite",
      "stream",
      "viewer",
      "live",
      "rgb",
      "mjpeg",
      "backend",
    ],
    pluginId: "CompositeStreamViewer",
  },

  compositeComponent: {
    id: "compositeComponent",
    name: "Composite Component",
    description:
      "Backend component for multi-illumination composite image acquisition and processing. Handles LED control and image fusion.",
    category: APP_CATEGORIES.APPS,
    icon: PhotoCameraIcon,
    enabled: false,
    essential: false,
    keywords: [
      "composite",
      "component",
      "multi-illumination",
      "acquisition",
      "processing",
      "led",
      "fusion",
    ],
    pluginId: "CompositeComponent",
  },

  socketView: {
    id: "socketView",
    name: "Socket View Controller",
    description:
      "Monitor and debug WebSocket connections. View real-time communication between frontend and backend for troubleshooting.",
    category: APP_CATEGORIES.SYSTEM,
    icon: NetworkCheckIcon,
    enabled: false,
    essential: false,
    keywords: [
      "socket",
      "websocket",
      "debug",
      "communication",
      "realtime",
      "troubleshoot",
    ],
    pluginId: "SocketView",
  },

  detectorTrigger: {
    id: "detectorTrigger",
    name: "Detector Trigger",
    description:
      "Control detector triggering mechanisms for synchronized image acquisition. Configure external trigger signals and timing.",
    category: APP_CATEGORIES.SYSTEM,
    icon: FlashOnIcon,
    enabled: false,
    essential: false,
    keywords: [
      "detector",
      "trigger",
      "synchronization",
      "acquisition",
      "timing",
      "external",
    ],
    pluginId: "DetectorTrigger",
  },

  extendedLEDMatrix: {
    id: "extendedLEDMatrix",
    name: "Extended LED Matrix",
    description:
      "Advanced LED matrix controller for structured illumination. Control complex LED patterns for enhanced microscopy techniques.",
    category: APP_CATEGORIES.SYSTEM,
    icon: ViewModuleIcon,
    enabled: false,
    essential: false,
    keywords: [
      "led",
      "matrix",
      "illumination",
      "structured",
      "patterns",
      "enhanced",
    ],
    pluginId: "ExtendedLEDMatrix",
  },
};

/**
 * Get apps by category
 */
export const getAppsByCategory = (category) => {
  return Object.values(APP_REGISTRY).filter((app) => app.category === category);
};

/**
 * Get enabled apps
 */
export const getEnabledApps = (enabledAppIds = []) => {
  return Object.values(APP_REGISTRY).filter(
    (app) => app.essential || enabledAppIds.includes(app.id),
  );
};

/**
 * Search apps by keywords
 */
export const searchApps = (query) => {
  if (!query.trim()) return Object.values(APP_REGISTRY);

  const searchTerm = query.toLowerCase();
  return Object.values(APP_REGISTRY).filter(
    (app) =>
      app.name.toLowerCase().includes(searchTerm) ||
      app.description.toLowerCase().includes(searchTerm) ||
      app.keywords.some((keyword) =>
        keyword.toLowerCase().includes(searchTerm),
      ),
  );
};

export default APP_REGISTRY;
