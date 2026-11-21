// src/components/LightsheetController.jsx
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import ViewInArIcon from "@mui/icons-material/ViewInAr";
import FlipIcon from "@mui/icons-material/Flip";
import VideocamIcon from "@mui/icons-material/Videocam";
import VideocamOffIcon from "@mui/icons-material/VideocamOff";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
  FormControl,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Tab,
  Tabs,
  TextField,
  Typography,
  Alert,
  Chip,
  Slider,
  Switch,
  FormControlLabel,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
  Tooltip,
  Divider,
} from "@mui/material";
import { useCallback, useEffect, useLayoutEffect, useMemo, useState, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as lightsheetSlice from "../state/slices/LightsheetSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import Lightsheet3DViewer from "./Lightsheet3DViewer.jsx";
import LightsheetTileMap from "./LightsheetTileMap.jsx";
import LightsheetVolumeViewer from "./LightsheetVolumeViewer.jsx";
import AxisConfigurationMenu from "./AxisConfigurationMenu.jsx";
import LightsheetPositionControls from "./LightsheetPositionControls.jsx";
import PixelCalibrationTab from "./FRAMESettings/PixelCalibrationTab.js";
import apiPositionerControllerGetPositions from "../backendapi/apiPositionerControllerGetPositions.js";
import apiLaserControllerGetLaserNames from "../backendapi/apiLaserControllerGetLaserNames.js";
import {
  apiStartStepAcquireScan,
  apiStartContinuousScanWithZarr,
  apiGetScanStatus,
  apiGetAvailableScanModes,
  apiGetAvailableStorageFormats,
  apiGetObjectiveFOV,
  apiGetScannedVolumes,
  apiStopLightsheet,
  apiGetVirtualOptics,
  apiSetVirtualIlluminationNA,
} from "../backendapi/apiLightsheetController.js";
import {
  apiLightsheetControllerObservationStreamControl,
  apiLightsheetControllerSetObservationExposure,
  apiLightsheetControllerGetObservationExposure,
  apiLightsheetControllerSetObservationGain,
  apiLightsheetControllerGetObservationGain,
  apiLightsheetControllerSetStreamTransform,
} from "../backendapi/apiLightsheetControllerObservationStream.js";
import {
  apiGetCameraExposureTime,
  apiSetCameraExposureTime,
  apiGetCameraGain,
  apiSetCameraGain,
} from "../backendapi/apiLightsheetCameraSettings.js";

const TABS = ["Scan", "Results & 3D", "Calibration", "Observation Camera", "Galvo Scanner"];

const TabPanel = (props) => {
  const { children, value, index, dense = false, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={dense ? 0 : 2}>{children}</Box>}
    </div>
  );
};

/**
 * A camera value with the range the camera reports: slider (log scale for exposure)
 * plus a field that applies on Enter / leaving the field, not on every keystroke.
 */
const CameraSetting = ({ label, units, value, min, max, log = false, onCommit }) => {
  const [draft, setDraft] = useState(value ?? "");
  useEffect(() => setDraft(value ?? ""), [value]);
  const lo = Math.max(min ?? 0, log ? 1e-3 : -Infinity);
  const hi = max ?? (log ? 1000 : 100);
  const toS = (v) => (log ? Math.log10(Math.max(v, lo)) : v);
  const fromS = (s) => Number((log ? 10 ** s : s).toPrecision(3));
  const commit = (v) => {
    if (Number.isFinite(v)) onCommit(Math.min(hi, Math.max(lo, v)));
  };
  const num = Number.isFinite(parseFloat(draft)) ? parseFloat(draft) : lo;
  return (
    <Box>
      <Typography variant="body2">
        {label}
        <Typography component="span" variant="caption" color="text.secondary">
          {" "}({fromS(toS(lo))}–{fromS(toS(hi))} {units})
        </Typography>
      </Typography>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
        <Slider
          size="small"
          value={toS(num)}
          min={toS(lo)}
          max={toS(hi)}
          step={log ? 0.01 : (hi - lo) / 200}
          onChange={(_, s) => setDraft(fromS(s))}
          onChangeCommitted={(_, s) => commit(fromS(s))}
          valueLabelDisplay="auto"
          valueLabelFormat={(s) => `${fromS(s)} ${units}`}
          disabled={value === null || value === undefined}
        />
        <TextField
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => commit(parseFloat(draft))}
          onKeyDown={(e) => e.key === "Enter" && commit(parseFloat(draft))}
          type="number"
          size="small"
          sx={{ width: 120 }}
          disabled={value === null || value === undefined}
        />
      </Box>
    </Box>
  );
};


/** Collapsible settings block; the summary stays readable when it is closed. */
const Section = ({ title, summary, expanded, onToggle, children }) => (
  <Accordion
    expanded={expanded}
    onChange={(_, v) => onToggle(v)}
    disableGutters
    elevation={0}
    square
    sx={{ bgcolor: "transparent", borderBottom: 1, borderColor: "divider", "&:before": { display: "none" } }}
  >
    <AccordionSummary
      expandIcon={<ExpandMoreIcon fontSize="small" />}
      sx={{ minHeight: 36, px: 0.5, "& .MuiAccordionSummary-content": { my: 0.5, alignItems: "baseline", gap: 1, minWidth: 0 } }}
    >
      <Typography variant="subtitle2" sx={{ flexShrink: 0 }}>{title}</Typography>
      {!expanded && summary && (
        <Typography variant="caption" color="text.secondary" noWrap>{summary}</Typography>
      )}
    </AccordionSummary>
    <AccordionDetails sx={{ px: 0.5, pt: 0, pb: 1.5 }}>{children}</AccordionDetails>
  </Accordion>
);

/** Live view filling the left side (never scrolls away), controls scrolling on the right. */
const SplitWithLiveView = ({ height, children }) => (
  <Box sx={{ display: "flex", flexDirection: { xs: "column", md: "row" }, height: { md: height } }}>
    <Box sx={{ flex: 1, minWidth: 0, height: { xs: 420, md: "100%" }, display: "flex", flexDirection: "column", p: 1 }}>
      <LiveViewControlWrapper fill />
    </Box>
    <Box
      sx={{
        width: { xs: "100%", md: 400, xl: 460 }, flexShrink: 0, overflowY: { md: "auto" }, px: 1.5, py: 0.5,
        borderLeft: { md: 1 }, borderColor: "divider",
      }}
    >
      {children}
    </Box>
  </Box>
);

const SECTIONS_KEY = "lightsheet.openSections";

const PHASE_LABEL = { acquiring: "Acquiring", writing: "Writing to disk", done: "Done", idle: "" };

/**
 * ImSwitch Lightsheet Controller
 *
 * - Continuous scan (stage sweeps, frames spooled to disk) or step-acquire
 *   (go-stop-acquire, optional XY tiles and timelapse), stored as OME-TIFF / OME-Zarr
 * - Scan overview: planned / acquired / earlier tile volumes in perspective
 * - Results: 3D view of any lightsheet volume with editable dz, napari snippet
 * - Calibration: the pixel calibration (camera <-> stage affine) that places the tiles
 */
const LightsheetController = () => {
  const connectionSettingsState = useSelector(connectionSettingsSlice.getConnectionSettingsState);
  const hostIP = connectionSettingsState.ip;
  const hostPort = connectionSettingsState.apiPort;
  const dispatch = useDispatch();
  const lightsheetState = useSelector(lightsheetSlice.getLightsheetState);

  const tabIndex = lightsheetState.tabIndex < TABS.length ? lightsheetState.tabIndex : 0;
  const {
    minPos, maxPos, speed, stepSize, axis, illuSource, illuValue, isRunning, scanMode, storageFormat,
    experimentName, scanStatus, availableScanModes, availableStorageFormats,
    enableTiling, tilesXPositive, tilesXNegative, tilesYPositive, tilesYNegative,
    tileStepSizeX, tileStepSizeY, tileOverlap, timepoints, timeLapsePeriod, objectiveFOV,
  } = lightsheetState;

  const [laserNames, setLaserNames] = useState([]);
  const [pastScans, setPastScans] = useState([]);
  const [frozenPlan, setFrozenPlan] = useState(null); // the plan as started (the stage moves meanwhile)
  const [leftView, setLeftView] = useState("overview");
  const [viewerPath, setViewerPath] = useState(null);
  const [cameraError, setCameraError] = useState("");
  const [virtualOptics, setVirtualOptics] = useState(null); // only on the virtual lightsheet
  const [naDraft, setNaDraft] = useState(0.05);
  const [openSections, setOpenSections] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(SECTIONS_KEY)) || {};
    } catch {
      return {};
    }
  });
  const toggleSection = (id, open) => setOpenSections((o) => {
    const next = { ...o, [id]: open };
    try {
      localStorage.setItem(SECTIONS_KEY, JSON.stringify(next));
    } catch {
      // private window: not remembered
    }
    return next;
  });
  const bodyRef = useRef(null);
  const [paneHeight, setPaneHeight] = useState(600);
  useLayoutEffect(() => {
    const update = () => {
      const top = bodyRef.current?.getBoundingClientRect().top ?? 120;
      setPaneHeight(Math.max(480, window.innerHeight - top - 8));
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [tabIndex, isRunning]);

  // Observation camera stream state
  const [observationStreamActive, setObservationStreamActive] = useState(false);
  const [observationStreamUrl, setObservationStreamUrl] = useState('');
  const [observationExposure, setObservationExposure] = useState(50);
  const [observationGain, setObservationGain] = useState(1);
  const [flipX, setFlipX] = useState(false);
  const [flipY, setFlipY] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [streamError, setStreamError] = useState('');
  const observationImgRef = useRef(null);

  // Lightsheet camera: {exposureTime|gain, min, max} as reported by the camera
  const [exposure, setExposure] = useState({});
  const [gain, setGain] = useState({});

  // Set up observation stream URL
  useEffect(() => {
    if (hostIP && hostPort) {
      setObservationStreamUrl(`${hostIP}:${hostPort}/imswitch/api/LightsheetController/observationStream`);
    }
  }, [hostIP, hostPort]);

  // Fetch initial observation camera settings
  useEffect(() => {
    const fetchCameraSettings = async () => {
      try {
        const exposure = await apiLightsheetControllerGetObservationExposure();
        if (typeof exposure === 'number') {
          setObservationExposure(exposure);
        }
        const gain = await apiLightsheetControllerGetObservationGain();
        if (typeof gain === 'number') {
          setObservationGain(gain);
        }
      } catch (err) {
        console.warn('Failed to fetch observation camera settings:', err);
      }
    };

    if (hostIP && hostPort) {
      fetchCameraSettings();
    }
  }, [hostIP, hostPort]);

  // Lightsheet camera, lasers, scan options, FOV and earlier scans
  const refreshPastScans = useCallback(() => {
    apiGetScannedVolumes().then(setPastScans).catch((e) => console.warn("No scan history:", e));
  }, []);
  useEffect(() => {
    if (!hostIP || !hostPort) return;
    apiGetCameraExposureTime().then((r) => r.success && setExposure(r)).catch(() => {});
    apiGetCameraGain().then((r) => r.success && setGain(r)).catch(() => {});
    apiLaserControllerGetLaserNames().then((n) => Array.isArray(n) && setLaserNames(n)).catch(() => {});
    apiGetAvailableScanModes().then((m) => Array.isArray(m) && dispatch(lightsheetSlice.setAvailableScanModes(m))).catch(() => {});
    apiGetAvailableStorageFormats().then((f) => Array.isArray(f) && dispatch(lightsheetSlice.setAvailableStorageFormats(f))).catch(() => {});
    apiGetObjectiveFOV()
      .then((fov) => {
        if (!fov.success) return;
        dispatch(lightsheetSlice.setObjectiveFOV(fov));
        const keep = 1 - fov.suggestedOverlap;
        dispatch(lightsheetSlice.setTileStepSizeX(Math.round(fov.fovX * keep)));
        dispatch(lightsheetSlice.setTileStepSizeY(Math.round(fov.fovY * keep)));
      })
      .catch(() => {});
    refreshPastScans();
    apiGetVirtualOptics()
      .then((o) => {
        if (o.available) {
          setVirtualOptics(o);
          setNaDraft(o.illuminationNA);
        }
      })
      .catch(() => {});
  }, [hostIP, hostPort, dispatch, refreshPastScans]);

  const commitIlluminationNA = (na) => {
    if (!Number.isFinite(na) || na <= 0) return;
    apiSetVirtualIlluminationNA(na)
      .then((o) => {
        if (o.available) {
          setVirtualOptics(o);
          setNaDraft(Number(o.illuminationNA.toPrecision(3)));
        }
      })
      .catch((e) => setCameraError(`Could not set the illumination NA: ${e.message}`));
  };

  // The FOV follows the calibration: re-read it when coming back from the Calibration tab
  useEffect(() => {
    if (tabIndex === 0 && hostIP) {
      apiGetObjectiveFOV().then((fov) => fov.success && dispatch(lightsheetSlice.setObjectiveFOV(fov))).catch(() => {});
    }
  }, [tabIndex, hostIP, dispatch]);

  const setCamera = async (setter, apply, value) => {
    try {
      const result = await apply(value);
      if (result.success) {
        setter(result);
        setCameraError("");
      } else {
        setCameraError(result.error || "The camera refused the value");
      }
    } catch (err) {
      setCameraError(`Could not set the camera: ${err.message}`);
    }
  };

  // Observation camera handlers
  const handleObservationStreamToggle = async () => {
    try {
      setStreamError('');
      const newState = !observationStreamActive;
      if (newState) {
        setObservationStreamActive(true);
      } else {
        try {
          await apiLightsheetControllerObservationStreamControl(false);
        } catch (err) {
          console.warn('Error stopping stream:', err);
        }
        setObservationStreamActive(false);
      }
    } catch (err) {
      setStreamError(`Failed to ${observationStreamActive ? 'stop' : 'start'} stream: ${err.message}`);
    }
  };
  const handleExposureChange = async (event, value) => setObservationExposure(value);
  const handleExposureChangeCommitted = async (event, value) => {
    try {
      await apiLightsheetControllerSetObservationExposure(value);
    } catch (err) {
      console.error('Failed to set exposure:', err);
    }
  };
  const handleGainChange = async (event, value) => setObservationGain(value);
  const handleGainChangeCommitted = async (event, value) => {
    try {
      await apiLightsheetControllerSetObservationGain(value);
    } catch (err) {
      console.error('Failed to set gain:', err);
    }
  };
  const handleFlipXToggle = async () => {
    const newFlipX = !flipX;
    setFlipX(newFlipX);
    try {
      await apiLightsheetControllerSetStreamTransform({ flipX: newFlipX, flipY, rotation });
    } catch (err) {
      console.error('Failed to set transform:', err);
    }
  };
  const handleFlipYToggle = async () => {
    const newFlipY = !flipY;
    setFlipY(newFlipY);
    try {
      await apiLightsheetControllerSetStreamTransform({ flipX, flipY: newFlipY, rotation });
    } catch (err) {
      console.error('Failed to set transform:', err);
    }
  };
  const handleRotationChange = async (event, newRotation) => {
    if (newRotation !== null) {
      setRotation(newRotation);
      try {
        await apiLightsheetControllerSetStreamTransform({ flipX, flipY, rotation: newRotation });
      } catch (err) {
        console.error('Failed to set rotation:', err);
      }
    }
  };

  // Scan status arrives over the global socket (sigScanStatusUpdate); poll as a fallback
  useEffect(() => {
    if (!hostIP || !hostPort || !isRunning) return undefined;
    const interval = setInterval(() => {
      apiGetScanStatus().then((s) => dispatch(lightsheetSlice.setScanStatus(s))).catch(() => {});
    }, 1000);
    return () => clearInterval(interval);
  }, [hostIP, hostPort, isRunning, dispatch]);

  // When a scan ends: show it among the earlier scans, drop the frozen plan
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !isRunning) {
      refreshPastScans();
      setFrozenPlan(null);
    }
    wasRunning.current = isRunning;
  }, [isRunning, refreshPastScans]);

  // Stage positions for the 3D views: once on mount, then every 20 s
  useEffect(() => {
    if (!hostIP || !hostPort) return undefined;
    const fetchPositions = async () => {
      try {
        const positionsData = await apiPositionerControllerGetPositions();
        const positions = {};
        if (positionsData) {
          ['X', 'Y', 'Z', 'A'].forEach((ax) => {
            if (typeof positionsData[ax] !== 'undefined') positions[ax.toLowerCase()] = positionsData[ax];
          });
          if (Object.keys(positions).length > 0) dispatch(lightsheetSlice.setAllStagePositions(positions));
        }
      } catch (error) {
        console.error("Error fetching stage positions:", error);
      }
    };
    fetchPositions();
    const interval = setInterval(fetchPositions, 20000);
    return () => clearInterval(interval);
  }, [hostIP, hostPort, dispatch]);

  // Mirror the global position state (buttons, websocket) into the lightsheet slice
  const globalPositionState = useSelector(positionSlice.getPositionState);
  useEffect(() => {
    const positions = {};
    ['x', 'y', 'z', 'a'].forEach((ax) => {
      if (typeof globalPositionState[ax] !== 'undefined') positions[ax] = globalPositionState[ax];
    });
    if (Object.keys(positions).length > 0) dispatch(lightsheetSlice.setAllStagePositions(positions));
  }, [globalPositionState.x, globalPositionState.y, globalPositionState.z, globalPositionState.a, dispatch]);

  // What the next scan covers, in stage µm (tile centres around the current XY)
  const stage = lightsheetState.stagePositions;
  const stepTiling = scanMode === "step_acquire" && enableTiling;
  const livePlan = useMemo(() => {
    const scanPos = stage[axis.toLowerCase()] || 0;
    const tiles = [];
    const nx = stepTiling ? [-tilesXNegative, tilesXPositive] : [0, 0];
    const ny = stepTiling ? [-tilesYNegative, tilesYPositive] : [0, 0];
    for (let iy = ny[0]; iy <= ny[1]; iy++) {
      for (let ix = nx[0]; ix <= nx[1]; ix++) {
        tiles.push({ x: stage.x + ix * tileStepSizeX, y: stage.y + iy * tileStepSizeY });
      }
    }
    return { tiles, zRange: [scanPos + parseFloat(minPos || 0), scanPos + parseFloat(maxPos || 0)] };
  }, [stage, axis, stepTiling, tilesXNegative, tilesXPositive, tilesYNegative, tilesYPositive,
    tileStepSizeX, tileStepSizeY, minPos, maxPos]);
  const plan = frozenPlan || livePlan;
  const fov = useMemo(() => [objectiveFOV.fovX, objectiveFOV.fovY], [objectiveFOV.fovX, objectiveFOV.fovY]);
  const stageMarker = useMemo(
    () => ({ x: stage.x, y: stage.y, z: stage[axis.toLowerCase()] }),
    [stage, axis],
  );

  const startScanning = useCallback(async () => {
    try {
      const common = {
        minPos: parseFloat(minPos),
        maxPos: parseFloat(maxPos),
        axis,
        illuSource: laserNames.includes(illuSource) ? illuSource : "",
        illuValue: parseFloat(illuValue),
        storageFormat,
        experimentName,
      };
      const result = scanMode === "step_acquire"
        ? await apiStartStepAcquireScan({
          ...common,
          stepSize: parseFloat(stepSize),
          enableTiling,
          tilesXPositive: parseInt(tilesXPositive),
          tilesXNegative: parseInt(tilesXNegative),
          tilesYPositive: parseInt(tilesYPositive),
          tilesYNegative: parseInt(tilesYNegative),
          tileStepSizeX: parseFloat(tileStepSizeX),
          tileStepSizeY: parseFloat(tileStepSizeY),
          tileOverlap: parseFloat(tileOverlap),
          timepoints: parseInt(timepoints),
          timeLapsePeriod: parseFloat(timeLapsePeriod),
        })
        : await apiStartContinuousScanWithZarr({ ...common, speed: parseFloat(speed) });
      if (result.success) {
        setFrozenPlan(livePlan);
        setLeftView("overview");
        dispatch(lightsheetSlice.setIsRunning(true));
      } else {
        dispatch(lightsheetSlice.setScanStatus({ errorMessage: result.error || "Scan did not start" }));
      }
    } catch (error) {
      dispatch(lightsheetSlice.setScanStatus({ errorMessage: `Could not start the scan: ${error.message}` }));
    }
  }, [scanMode, minPos, maxPos, stepSize, speed, axis, illuSource, illuValue, storageFormat, experimentName,
    enableTiling, tilesXPositive, tilesXNegative, tilesYPositive, tilesYNegative, laserNames, livePlan,
    tileStepSizeX, tileStepSizeY, tileOverlap, timepoints, timeLapsePeriod, dispatch]);

  const stopScanning = () => {
    apiStopLightsheet().catch((e) => console.error("Stop failed:", e));
  };

  const dataPathOf = (abs) => {
    // absolute server path -> data-relative "/LightSheet/..." for the viewer
    const i = abs ? abs.indexOf("/LightSheet/") : -1;
    return i >= 0 ? abs.slice(i) : null;
  };
  const openInViewer = (abs) => {
    setViewerPath(dataPathOf(abs));
    dispatch(lightsheetSlice.setTabIndex(1));
  };

  const handleTabChange = (event, newValue) => dispatch(lightsheetSlice.setTabIndex(newValue));

  const getScanModeLabel = (mode) => ({
    continuous: "Continuous (fast sweep)",
    step_acquire: "Step-acquire (stop at each plane)",
  }[mode] || mode);
  const getStorageFormatLabel = (format) => ({
    tiff: "OME-TIFF",
    ome_zarr: "OME-Zarr",
    both: "OME-TIFF + OME-Zarr",
  }[format] || format);

  const totalTiles = (tilesXNegative + tilesXPositive + 1) * (tilesYNegative + tilesYPositive + 1);
  const nPlanes = scanMode === "step_acquire" && stepSize > 0
    ? Math.floor(Math.abs(maxPos - minPos) / stepSize) + 1 : null;
  const tileMB = objectiveFOV.tileShape ? (objectiveFOV.tileShape[0] * objectiveFOV.tileShape[1] * 2) / 1e6 : null;
  const scanGB = nPlanes && tileMB ? (nPlanes * tileMB * (stepTiling ? totalTiles : 1) * (timepoints || 1)) / 1e3 : null;
  const phase = PHASE_LABEL[scanStatus.phase] || "";

  const modeShort = scanMode === "step_acquire" ? "step" : "continuous";
  const statusLine = isRunning
    ? [phase || "Scanning",
      scanStatus.totalTiles > 1 ? `tile ${scanStatus.currentTile}/${scanStatus.totalTiles}` : null,
      `frame ${scanStatus.currentFrame}${scanStatus.totalPositions ? `/${scanStatus.totalPositions}` : ""}`,
      `${axis} ${Number(scanStatus.currentPosition || 0).toFixed(0)} µm`].filter(Boolean).join(" · ")
    : [scanStatus.phase === "done" ? "Last scan done" : "Ready",
      nPlanes ? `${nPlanes} planes${stepTiling ? ` × ${totalTiles} tiles` : ""}` : modeShort,
      scanGB !== null ? `≈ ${scanGB.toFixed(1)} GB` : null].filter(Boolean).join(" · ");
  const laserLabel = laserNames.includes(illuSource) ? illuSource : laserNames[0] || "default";
  const section = (id, title, summary, children, defaultExpanded = true) => (
    <Section
      title={title}
      summary={summary}
      expanded={openSections[id] ?? defaultExpanded}
      onToggle={(v) => toggleSection(id, v)}
    >
      {children}
    </Section>
  );

  const scanPanel = (
    <>
      {section("camera", "Camera",
        `${exposure.exposureTime ?? "–"} ms · gain ${gain.gain ?? "–"}`,
        <>
          <CameraSetting
            label="Exposure"
            units="ms"
            log
            value={exposure.exposureTime}
            min={exposure.min}
            max={exposure.max}
            onCommit={(v) => setCamera(setExposure, apiSetCameraExposureTime, v)}
          />
          <CameraSetting
            label="Gain"
            units={gain.units || ""}
            value={gain.gain}
            min={gain.min}
            max={gain.max}
            onCommit={(v) => setCamera(setGain, apiSetCameraGain, v)}
          />
          {cameraError && (
            <Alert severity="warning" onClose={() => setCameraError("")} sx={{ mt: 1 }}>{cameraError}</Alert>
          )}
        </>)}

      {virtualOptics && section("virtual", "Virtual optics",
        `illumination NA ${virtualOptics.illuminationNA} · sheet ${virtualOptics.sheetWaistUm.toFixed(1)} µm`,
        <>
          <Typography variant="body2">Illumination NA</Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Slider
              size="small"
              min={0.01}
              max={0.3}
              step={0.005}
              value={Number(naDraft) || 0.01}
              onChange={(_, v) => setNaDraft(v)}
              onChangeCommitted={(_, v) => commitIlluminationNA(v)}
            />
            <TextField
              value={naDraft}
              onChange={(e) => setNaDraft(e.target.value)}
              onBlur={() => commitIlluminationNA(parseFloat(naDraft))}
              onKeyDown={(e) => e.key === "Enter" && commitIlluminationNA(parseFloat(naDraft))}
              type="number"
              size="small"
              sx={{ width: 120 }}
              inputProps={{ step: 0.005, min: 0.005 }}
            />
          </Box>
          <Typography variant="caption" color="text.secondary" component="div" sx={{ fontFamily: "monospace" }}>
            sheet waist {virtualOptics.sheetWaistUm.toFixed(1)} µm · flat over {virtualOptics.sheetUsableFieldUm.toFixed(0)} µm
            (FOV {virtualOptics.fovUm[0].toFixed(0)} µm)
            <br />
            detection NA {virtualOptics.detectionNA} · {virtualOptics.magnification}× · {virtualOptics.pixelSizeUm.toFixed(3)} µm/px
            · resolution {virtualOptics.lateralResolutionUm.toFixed(1)} µm · DOF {virtualOptics.depthOfFieldUm.toFixed(0)} µm
          </Typography>
          <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 0.5 }}>
            Simulated: X/Y and {virtualOptics.sampleScanAxis} move the sample ({virtualOptics.sampleScanAxis} through
            the sheet: use it as scan axis), {virtualOptics.objectiveAxis} moves the detection objective (in focus at 0).
          </Typography>
        </>, false)}

      {section("illumination", "Illumination", `${laserLabel} · ${illuValue}/1024`,
        <Box sx={{ display: "flex", gap: 1 }}>
          <FormControl size="small" sx={{ flex: 1 }}>
            <InputLabel shrink>Laser</InputLabel>
            <Select
              label="Laser"
              notched
              value={laserNames.includes(illuSource) ? illuSource : ""}
              onChange={(e) => dispatch(lightsheetSlice.setIlluSource(e.target.value))}
              displayEmpty
            >
              <MenuItem value="">{laserNames.length ? `Default (${laserNames[0]})` : "Default"}</MenuItem>
              {laserNames.map((n) => <MenuItem key={n} value={n}>{n}</MenuItem>)}
            </Select>
          </FormControl>
          <TextField
            label="Intensity (0–1024)"
            value={illuValue}
            onChange={(e) => dispatch(lightsheetSlice.setIlluValue(e.target.value))}
            size="small"
            type="number"
            inputProps={{ min: 0, max: 1024 }}
            sx={{ width: 140 }}
          />
        </Box>)}

      {section("scan", "Scan",
        `${modeShort} · ${axis} ${minPos}…${maxPos} µm · ${getStorageFormatLabel(storageFormat)}`,
        <Grid container spacing={1}>
          <Grid item xs={12}>
            <ToggleButtonGroup
              size="small"
              exclusive
              fullWidth
              value={scanMode}
              onChange={(_, v) => v && dispatch(lightsheetSlice.setScanMode(v))}
            >
              {availableScanModes.map((m) => (
                <ToggleButton key={m} value={m} sx={{ textTransform: "none", py: 0.25 }}>
                  <Tooltip title={getScanModeLabel(m)}>
                    <span>{m === "step_acquire" ? "Step-acquire" : "Continuous"}</span>
                  </Tooltip>
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="From (µm)"
              value={minPos}
              onChange={(e) => dispatch(lightsheetSlice.setMinPos(e.target.value))}
              fullWidth
              size="small"
              type="number"
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="To (µm)"
              value={maxPos}
              onChange={(e) => dispatch(lightsheetSlice.setMaxPos(e.target.value))}
              fullWidth
              size="small"
              type="number"
            />
          </Grid>
          <Grid item xs={4}>
            {scanMode === "step_acquire" ? (
              <TextField
                label="Step (µm)"
                value={stepSize}
                onChange={(e) => dispatch(lightsheetSlice.setStepSize(e.target.value))}
                fullWidth
                size="small"
                type="number"
                inputProps={{ min: 0.1, step: 0.1 }}
              />
            ) : (
              <Tooltip title="Stage speed; dz ≈ speed / camera fps">
                <TextField
                  label="Speed"
                  value={speed}
                  onChange={(e) => dispatch(lightsheetSlice.setSpeed(e.target.value))}
                  fullWidth
                  size="small"
                  type="number"
                />
              </Tooltip>
            )}
          </Grid>
          <Grid item xs={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Axis</InputLabel>
              <Select value={axis} label="Axis" onChange={(e) => dispatch(lightsheetSlice.setAxis(e.target.value))}>
                {["A", "X", "Y", "Z"].map((a) => <MenuItem key={a} value={a}>{a}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Save as</InputLabel>
              <Select value={storageFormat} label="Save as" onChange={(e) => dispatch(lightsheetSlice.setStorageFormat(e.target.value))}>
                {availableStorageFormats.map((f) => <MenuItem key={f} value={f}>{getStorageFormatLabel(f)}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={5}>
            <TextField
              label="Name"
              value={experimentName}
              onChange={(e) => dispatch(lightsheetSlice.setExperimentName(e.target.value))}
              fullWidth
              size="small"
            />
          </Grid>
          <Grid item xs={12}>
            <Typography variant="caption" color="text.secondary">
              From/To are relative to the current {axis} position. Frames go straight to disk.
            </Typography>
          </Grid>
        </Grid>)}

      {scanMode === "step_acquire" && section("tiles", "Tiles & timelapse",
        `${stepTiling ? `${totalTiles} tiles` : "1 tile"} · ${timepoints > 1 ? `${timepoints} × every ${timeLapsePeriod} s` : "single time point"}`,
        <Grid container spacing={1}>
          <Grid item xs={12}>
            <FormControlLabel
              control={<Switch size="small" checked={enableTiling} onChange={(e) => dispatch(lightsheetSlice.setEnableTiling(e.target.checked))} />}
              label="XY tiles"
            />
            <FormControlLabel
              control={<Switch size="small" checked={timepoints > 1} onChange={(e) => dispatch(lightsheetSlice.setTimepoints(e.target.checked ? 2 : 1))} />}
              label="Timelapse"
            />
          </Grid>
          {enableTiling && (
            <>
              {[
                ["−X", tilesXNegative, lightsheetSlice.setTilesXNegative],
                ["+X", tilesXPositive, lightsheetSlice.setTilesXPositive],
                ["−Y", tilesYNegative, lightsheetSlice.setTilesYNegative],
                ["+Y", tilesYPositive, lightsheetSlice.setTilesYPositive],
              ].map(([label, value, action]) => (
                <Grid item xs={3} key={label}>
                  <TextField
                    label={`Tiles ${label}`}
                    value={value}
                    onChange={(e) => dispatch(action(Math.max(0, parseInt(e.target.value) || 0)))}
                    fullWidth
                    size="small"
                    type="number"
                    inputProps={{ min: 0 }}
                  />
                </Grid>
              ))}
              <Grid item xs={4}>
                <TextField
                  label="Step X (µm)"
                  value={tileStepSizeX}
                  onChange={(e) => dispatch(lightsheetSlice.setTileStepSizeX(parseFloat(e.target.value) || 0))}
                  fullWidth
                  size="small"
                  type="number"
                />
              </Grid>
              <Grid item xs={4}>
                <TextField
                  label="Step Y (µm)"
                  value={tileStepSizeY}
                  onChange={(e) => dispatch(lightsheetSlice.setTileStepSizeY(parseFloat(e.target.value) || 0))}
                  fullWidth
                  size="small"
                  type="number"
                />
              </Grid>
              <Grid item xs={4}>
                <TextField
                  label="Overlap (%)"
                  value={Math.round(tileOverlap * 100)}
                  onChange={(e) => dispatch(lightsheetSlice.setTileOverlap(Math.min(90, Math.max(0, parseFloat(e.target.value) || 0)) / 100))}
                  fullWidth
                  size="small"
                  type="number"
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  size="small"
                  onClick={() => {
                    dispatch(lightsheetSlice.setTileStepSizeX(Math.round(objectiveFOV.fovX * (1 - tileOverlap))));
                    dispatch(lightsheetSlice.setTileStepSizeY(Math.round(objectiveFOV.fovY * (1 - tileOverlap))));
                  }}
                >
                  Steps from FOV ({Math.round(objectiveFOV.fovX)} × {Math.round(objectiveFOV.fovY)} µm) − overlap
                </Button>
              </Grid>
            </>
          )}
          {timepoints > 1 && (
            <>
              <Grid item xs={6}>
                <TextField
                  label="Timepoints"
                  value={timepoints}
                  onChange={(e) => dispatch(lightsheetSlice.setTimepoints(Math.max(1, parseInt(e.target.value) || 1)))}
                  fullWidth
                  size="small"
                  type="number"
                  inputProps={{ min: 1 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Every (s)"
                  value={timeLapsePeriod}
                  onChange={(e) => dispatch(lightsheetSlice.setTimeLapsePeriod(parseFloat(e.target.value) || 60))}
                  fullWidth
                  size="small"
                  type="number"
                  inputProps={{ min: 1 }}
                />
              </Grid>
            </>
          )}
        </Grid>, false)}

      {!isRunning && (scanStatus.zarrPath || scanStatus.tiffPath) && section("result", "Last scan",
        scanStatus.zStepUm ? `dz ${Number(scanStatus.zStepUm.toPrecision(3))} µm · ${Number((scanStatus.pixelSizeUm || 0).toPrecision(3))} µm/px` : "",
        <>
          {[["OME-Zarr", scanStatus.zarrPath], ["OME-TIFF", scanStatus.tiffPath]].filter(([, p]) => p).map(([label, p]) => (
            <Box key={label} sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Tooltip title={p}>
                <Typography variant="body2" noWrap sx={{ fontFamily: "monospace", flex: 1, minWidth: 0 }}>
                  {label}: {p.split("/").slice(-2).join("/")}
                </Typography>
              </Tooltip>
              <Button size="small" startIcon={<ViewInArIcon />} onClick={() => openInViewer(p)}>
                3D
              </Button>
            </Box>
          ))}
        </>)}

      {section("overview", "Overview",
        objectiveFOV.calibrated ? `calibrated · ${Number((objectiveFOV.pixelSize || 0).toPrecision(3))} µm/px` : "not calibrated",
        <>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
            <ToggleButtonGroup size="small" exclusive value={leftView} onChange={(_, v) => v && setLeftView(v)}>
              <ToggleButton value="overview" sx={{ textTransform: "none", py: 0.25 }}>Scanned volumes</ToggleButton>
              <ToggleButton value="assembly" sx={{ textTransform: "none", py: 0.25 }}>Assembly</ToggleButton>
            </ToggleButtonGroup>
            <Box sx={{ flex: 1 }} />
            <Tooltip title={objectiveFOV.calibrated
              ? `Tiles are placed with the pixel calibration of ${objectiveFOV.detectorName} (rotation ${Math.round(objectiveFOV.rotationDeg || 0)}°)`
              : "No pixel calibration for this camera: tiles are placed with the plain pixel size, so flips or a rotated camera are not corrected. Click to calibrate."}>
              <Chip
                size="small"
                color={objectiveFOV.calibrated ? "success" : "warning"}
                label={objectiveFOV.calibrated ? "Calibrated" : "Calibrate"}
                onClick={() => dispatch(lightsheetSlice.setTabIndex(2))}
              />
            </Tooltip>
          </Box>
          {leftView === "overview" ? (
            <LightsheetTileMap
              tiles={plan.tiles}
              fov={fov}
              zRange={plan.zRange}
              currentTile={scanStatus.currentTile || 0}
              isRunning={isRunning}
              pastScans={pastScans}
              stage={stageMarker}
              height={260}
            />
          ) : (
            <Lightsheet3DViewer
              positions={lightsheetState.stagePositions}
              axisConfig={lightsheetState.axisConfig}
              cameraState={lightsheetState.cameraState}
              onCameraChange={(c) => dispatch(lightsheetSlice.setCameraState(c))}
              width={372}
              height={260}
            />
          )}
        </>)}

      {section("stage", "Stage",
        ["x", "y", "z", "a"].map((k) => `${k.toUpperCase()} ${(stage[k] ?? 0).toFixed(0)}`).join(" · "),
        <>
          <LightsheetPositionControls />
          <Box sx={{ mt: 1 }}>
            <AxisConfigurationMenu />
          </Box>
        </>, false)}
    </>
  );

  return (
    <Paper sx={{ display: "flex", flexDirection: "column", flex: 1, width: "100%", minWidth: 0 }}>
      {/* Header: tabs, scan status and start/stop, visible on every tab */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, pr: 1.5, borderBottom: 1, borderColor: "divider" }}>
        <Tabs
          value={tabIndex}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ flex: 1, minHeight: 40, "& .MuiTab-root": { minHeight: 40, py: 0.5, textTransform: "none" } }}
        >
          {TABS.map((label) => <Tab key={label} label={label} />)}
        </Tabs>
        <Box sx={{ width: 280, display: { xs: "none", sm: "block" } }}>
          <Typography variant="caption" color="text.secondary" noWrap component="div" title={statusLine}>
            {statusLine}
          </Typography>
          {isRunning && (
            <LinearProgress
              variant={scanStatus.phase === "writing" ? "indeterminate" : "determinate"}
              value={scanStatus.progress || 0}
              sx={{ height: 4, borderRadius: 2 }}
            />
          )}
        </Box>
        {isRunning ? (
          <Button variant="contained" color="error" size="small" startIcon={<StopIcon />} onClick={stopScanning}>
            Stop
          </Button>
        ) : (
          <Tooltip title={`Start ${scanMode === "step_acquire" ? "step-acquire" : "continuous"} scan`}>
            <Button variant="contained" size="small" startIcon={<PlayArrowIcon />} onClick={startScanning}>
              Start scan
            </Button>
          </Tooltip>
        )}
      </Box>
      {scanStatus.errorMessage && (
        <Alert severity="error" sx={{ borderRadius: 0 }} onClose={() => dispatch(lightsheetSlice.setScanStatus({ errorMessage: null }))}>
          {scanStatus.errorMessage}
        </Alert>
      )}

      <Box ref={bodyRef}>
        <TabPanel value={tabIndex} index={0} dense>
          <SplitWithLiveView height={paneHeight}>{scanPanel}</SplitWithLiveView>
        </TabPanel>

        <TabPanel value={tabIndex} index={1}>
          <LightsheetVolumeViewer initialPath={viewerPath} height={Math.max(420, paneHeight - 90)} />
        </TabPanel>

        <TabPanel value={tabIndex} index={2}>
          <Alert severity="info" sx={{ mb: 2 }}>
            Calibrate the lightsheet camera ({objectiveFOV.detectorName || "first detector"}) against the X/Y stage.
            The result sets the tile FOV and places the tiles of step-acquire scans in the stitched volume (flips,
            a rotated camera and the true overlap are taken into account). The calibration moves the XY stage.
          </Alert>
          <PixelCalibrationTab />
        </TabPanel>

      {/* Observation Camera Tab */}
      <TabPanel value={tabIndex} index={3}>
        <Grid container spacing={3}>
          {/* Stream Error Display */}
          {streamError && (
            <Grid item xs={12}>
              <Alert severity="error" onClose={() => setStreamError('')}>
                {streamError}
              </Alert>
            </Grid>
          )}

          {/* Observation Camera Stream */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Observation Camera Stream
              </Typography>
              
              {/* Stream Controls */}
              <Box sx={{ mb: 2, display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
                <Button 
                  variant="contained" 
                  color={observationStreamActive ? "error" : "primary"}
                  onClick={handleObservationStreamToggle}
                  startIcon={observationStreamActive ? <VideocamOffIcon /> : <VideocamIcon />}
                >
                  {observationStreamActive ? 'Stop Stream' : 'Start Stream'}
                </Button>

                <Divider orientation="vertical" flexItem />

                {/* Flip Controls */}
                <Tooltip title="Flip Horizontal">
                  <IconButton 
                    onClick={handleFlipXToggle}
                    color={flipX ? "primary" : "default"}
                  >
                    <FlipIcon />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title="Flip Vertical">
                  <IconButton 
                    onClick={handleFlipYToggle}
                    color={flipY ? "primary" : "default"}
                    sx={{ transform: 'rotate(90deg)' }}
                  >
                    <FlipIcon />
                  </IconButton>
                </Tooltip>

                <Divider orientation="vertical" flexItem />

                {/* Rotation Controls */}
                <ToggleButtonGroup
                  value={rotation}
                  exclusive
                  onChange={handleRotationChange}
                  size="small"
                >
                  <ToggleButton value={0}>0°</ToggleButton>
                  <ToggleButton value={90}>90°</ToggleButton>
                  <ToggleButton value={180}>180°</ToggleButton>
                  <ToggleButton value={270}>270°</ToggleButton>
                </ToggleButtonGroup>
              </Box>

              {/* MJPEG Stream Display */}
              <Box 
                sx={{ 
                  backgroundColor: 'black', 
                  minHeight: 400,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative',
                  borderRadius: 1,
                  overflow: 'hidden'
                }}
              >
                {observationStreamActive ? (
                  <img
                    ref={observationImgRef}
                    src={observationStreamUrl}
                    alt="Observation Camera"
                    style={{ 
                      display: 'block',
                      margin: 'auto',
                      maxWidth: '100%', 
                      maxHeight: 500,
                      objectFit: 'contain',
                      WebkitUserSelect: 'none',
                      transform: `scaleX(${flipX ? -1 : 1}) scaleY(${flipY ? -1 : 1}) rotate(${rotation}deg)`
                    }}
                    onError={() => setStreamError('Failed to load stream. Check if observation camera is available.')}
                  />
                ) : (
                  <Typography color="white">
                    Stream not active. Click "Start Stream" to begin.
                  </Typography>
                )}
              </Box>
            </Paper>
          </Grid>

          {/* Camera Controls */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Camera Settings
              </Typography>

              {/* Exposure Time Slider */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Exposure Time (ms)
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Slider
                    value={observationExposure}
                    onChange={handleExposureChange}
                    onChangeCommitted={handleExposureChangeCommitted}
                    min={1}
                    max={1000}
                    valueLabelDisplay="auto"
                    disabled={!observationStreamActive}
                  />
                  <TextField
                    value={observationExposure}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val)) {
                        setObservationExposure(val);
                        handleExposureChangeCommitted(null, val);
                      }
                    }}
                    type="number"
                    size="small"
                    sx={{ width: 80 }}
                    disabled={!observationStreamActive}
                  />
                </Box>
              </Box>

              {/* Gain Slider */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Gain
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Slider
                    value={observationGain}
                    onChange={handleGainChange}
                    onChangeCommitted={handleGainChangeCommitted}
                    min={0}
                    max={100}
                    valueLabelDisplay="auto"
                    disabled={!observationStreamActive}
                  />
                  <TextField
                    value={observationGain}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val)) {
                        setObservationGain(val);
                        handleGainChangeCommitted(null, val);
                      }
                    }}
                    type="number"
                    size="small"
                    sx={{ width: 80 }}
                    disabled={!observationStreamActive}
                  />
                </Box>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Transform Status */}
              <Typography variant="subtitle2" gutterBottom>
                Current Transform
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Chip 
                  label={`Flip X: ${flipX ? 'On' : 'Off'}`} 
                  size="small" 
                  color={flipX ? 'primary' : 'default'}
                />
                <Chip 
                  label={`Flip Y: ${flipY ? 'On' : 'Off'}`} 
                  size="small" 
                  color={flipY ? 'primary' : 'default'}
                />
                <Chip 
                  label={`Rotation: ${rotation}°`} 
                  size="small" 
                  color={rotation !== 0 ? 'primary' : 'default'}
                />
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Galvo Scanner Tab */}
      <TabPanel value={tabIndex} index={4} dense>
        <SplitWithLiveView height={paneHeight}>
        <Grid container spacing={1.5} sx={{ py: 1.5 }}>
          <Grid item xs={6}>
            <TextField
              label="Channel"
              value={lightsheetState.galvoChannel || 2}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoChannel(parseInt(e.target.value))
                )
              }
              fullWidth
              type="number"
              size="small"
              inputProps={{ min: 1, max: 2 }}
              helperText="Channel: 1 or 2"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Frequency"
              value={lightsheetState.galvoFrequency || 20}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoFrequency(parseFloat(e.target.value))
                )
              }
              fullWidth
              type="number"
              size="small"
              inputProps={{ min: 0, step: 0.1 }}
              helperText="Frequency (Hz)"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Offset"
              value={lightsheetState.galvoOffset || 0}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoOffset(parseFloat(e.target.value))
                )
              }
              fullWidth
              type="number"
              size="small"
              inputProps={{ step: 0.1 }}
              helperText="Offset value"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Amplitude"
              value={lightsheetState.galvoAmplitude || 2}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoAmplitude(parseFloat(e.target.value))
                )
              }
              fullWidth
              type="number"
              size="small"
              inputProps={{ min: 0, step: 0.1 }}
              helperText="Amplitude value"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Clock Divider"
              value={lightsheetState.galvoClkDiv || 0}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoClkDiv(parseInt(e.target.value))
                )
              }
              fullWidth
              type="number"
              size="small"
              inputProps={{ min: 0 }}
              helperText="Clock divider"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Phase"
              value={lightsheetState.galvoPhase || 0}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoPhase(parseInt(e.target.value))
                )
              }
              fullWidth
              type="number"
              size="small"
              helperText="Phase value"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Invert"
              value={lightsheetState.galvoInvert || 1}
              onChange={(e) =>
                dispatch(
                  lightsheetSlice.setGalvoInvert(parseInt(e.target.value))
                )
              }
              fullWidth
              type="number"
              size="small"
              inputProps={{ min: 0, max: 1 }}
              helperText="Invert: 0 or 1"
            />
          </Grid>
          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => {
                const channel = lightsheetState.galvoChannel || 1;
                const frequency = lightsheetState.galvoFrequency || 10;
                const offset = lightsheetState.galvoOffset || 0;
                const amplitude = lightsheetState.galvoAmplitude || 1;
                const clk_div = lightsheetState.galvoClkDiv || 0;
                const phase = lightsheetState.galvoPhase || 0;
                const invert = lightsheetState.galvoInvert || 1;

                const url = `${hostIP}:${hostPort}/imswitch/api/LightsheetController/setGalvo?channel=${channel}&frequency=${frequency}&offset=${offset}&amplitude=${amplitude}&clk_div=${clk_div}&phase=${phase}&invert=${invert}`;

                fetch(url, { method: "GET" })
                  .then((response) => response.json())
                  .then((data) => {
                    console.log("Galvo parameters set:", data);
                  })
                  .catch((error) =>
                    console.error("Error setting galvo parameters:", error)
                  );
              }}
              size="large"
            >
              Apply Galvo Settings
            </Button>
          </Grid>
        </Grid>
        </SplitWithLiveView>
      </TabPanel>
      </Box>
    </Paper>
  );
};

export default LightsheetController;
