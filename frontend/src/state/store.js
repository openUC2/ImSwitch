// redux/store.js
import { configureStore } from "@reduxjs/toolkit";
import { persistStore, persistReducer } from "redux-persist";
import { combineReducers } from "redux";

// storage types
import storage from "redux-persist/lib/storage";

// Import slices
import connectionSettingsReducer from "./slices/ConnectionSettingsSlice";
import webSocketReducer from "./slices/WebSocketSlice";
import positionReducer from "./slices/PositionSlice";
import wellSelectorReducer from "./slices/WellSelectorSlice";
import experimentReducer from "./slices/ExperimentSlice";
import experimentStatusReducer from "./slices/ExperimentStatusSlice";
import parameterRangeReducer from "./slices/ParameterRangeSlice";
import liveStreamReducer from "./slices/LiveStreamSlice";
import tileStreamReducer from "./slices/TileStreamSlice";
import objectiveReducer from "./slices/ObjectiveSlice";
import LEDMatrixReducer from "./slices/LEDMatrixSlice";
import experimentStateReducer from "./slices/ExperimentStateSlice";
import liveViewReducer from "./slices/LiveViewSlice";
import histoScanReducer from "./slices/HistoScanSlice";
import widgetReducer from "./slices/WidgetSlice";
import lepmonReducer from "./slices/LepmonSlice";
import uc2Reducer from "./slices/UC2Slice";
import stageOffsetCalibrationReducer from "./slices/StageOffsetCalibrationSlice";
import stageCenterCalibrationReducer from "./slices/StageCenterCalibrationSlice";
import flowStopReducer from "./slices/FlowStopSlice";
import lightsheetReducer from "./slices/LightsheetSlice";
import zarrinitialZarrReducer from "./slices/OmeZarrTileStreamSlice";
import stresstestReducer from "./slices/StresstestSlice";
import workflowReducer from "./slices/WorkflowSlice";
import stormReducer from "./slices/STORMSlice";
import focusLockReducer from "./slices/FocusLockSlice";
import demoReducer from "./slices/DemoSlice";
import mazeGameReducer from "./slices/MazeGameSlice";
import themeReducer from "./slices/ThemeSlice";
import notificationReducer from "./slices/NotificationSlice";
import autofocusReducer from "./slices/AutofocusSlice";
import socketDebugReducer from "./slices/SocketDebugSlice";
import appManagerReducer from "./slices/appManagerSlice";
import canOtaReducer from "./slices/canOtaSlice";
import usbFlashReducer from "./slices/usbFlashSlice";
import holoReducer from "./slices/HoloSlice";
import offAxisHoloReducer from "./slices/OffAxisHoloSlice";
import michelsonReducer from "./slices/MichelsonSlice";
import dpcReducer from "./slices/dpcSlice";
import laserReducer from "./slices/LaserSlice";
import vizarrViewerReducer from "./slices/VizarrViewerSlice";
import compositeAcquisitionReducer from "./slices/CompositeAcquisitionSlice";
import experimentUIReducer from "./slices/ExperimentUISlice";
import galvoScannerReducer from "./slices/GalvoScannerSlice";
import focusMapReducer from "./slices/FocusMapSlice";

//#####################################################################################
// Nested persist config for liveStreamState
// Only persist settings, not live data like stats or histogram
const liveStreamPersistConfig = {
  key: "liveStreamState",
  storage,
  whitelist: [
    "minVal",
    "maxVal",
    "gamma",
    "imageFormat", // Persist selected format (binary/jpeg/webrtc)
    "streamSettings", // Persist all stream settings
    "isLegacyBackend",
    "backendCapabilities",
  ],
  // Don't persist: histogramX, histogramY, showHistogram, zoom, translateX, translateY, stats
};

//#####################################################################################
// Nested persist config for lightsheetState
// Persist axis configuration and camera state for 3D viewer
const lightsheetPersistConfig = {
  key: "lightsheet",
  storage,
  whitelist: [
    "axisConfig",      // Persist axis offsets, scales, and invert flags
    "cameraState",     // Persist 3D viewer camera position and zoom
    "experimentName",  // Persist last used experiment name
    "storageFormat",   // Persist last used storage format
    "scanMode",        // Persist last used scan mode
  ],
  // Don't persist: stagePositions, scanStatus, vtkImagePrimary, isRunning
};

//#####################################################################################
// Combine reducers
const rootReducer = combineReducers({
  connectionSettingsState: connectionSettingsReducer,
  webSocketState: webSocketReducer,
  experimentState: experimentReducer,
  experimentStatusState: experimentStatusReducer,
  liveStreamState: persistReducer(liveStreamPersistConfig, liveStreamReducer), // Nested persist
  tileStreamState: tileStreamReducer,
  parameterRangeState: parameterRangeReducer,
  wellSelectorState: wellSelectorReducer,
  objectiveState: objectiveReducer,
  position: positionReducer,
  LEDMatrixState: LEDMatrixReducer,
  experimentWorkflowState: experimentStateReducer,
  liveViewState: liveViewReducer,
  histoScanState: histoScanReducer,
  widgetState: widgetReducer,
  lepmon: lepmonReducer,
  uc2State: uc2Reducer,
  stageOffsetCalibration: stageOffsetCalibrationReducer,
  stageCenterCalibration: stageCenterCalibrationReducer,
  flowStop: flowStopReducer,
  lightsheet: persistReducer(lightsheetPersistConfig, lightsheetReducer), // Nested persist for axis config
  omeZarrState: zarrinitialZarrReducer,
  stresstestState: stresstestReducer,
  workflowState: workflowReducer,
  storm: stormReducer,
  focusLockState: focusLockReducer,
  demoState: demoReducer,
  mazeGameState: mazeGameReducer,
  themeState: themeReducer,
  notification: notificationReducer,
  autofocusState: autofocusReducer,
  socketDebugState: socketDebugReducer,
  appManager: appManagerReducer,
  canOtaState: canOtaReducer,
  usbFlash: usbFlashReducer,
  holoState: holoReducer,
  offAxisHoloState: offAxisHoloReducer,
  michelsonState: michelsonReducer,
  dpc: dpcReducer,
  laserState: laserReducer,
  vizarrViewerState: vizarrViewerReducer,
  compositeAcquisitionState: compositeAcquisitionReducer,
  experimentUI: experimentUIReducer,
  galvoScannerState: galvoScannerReducer,
  focusMap: focusMapReducer,
});

//#####################################################################################
// Persist configuration with whitelist and blacklist
// Note: After changing something in a slice, clear the browser cache.
const persistConfig = {
  key: "root",
  storage, //<--storage type
  whitelist: [
    "connectionSettingsState",
    "parameterRangeState",
    "experimentState",
    "wellSelectorState",
    "position", // Persist step sizes
    "workflowState",
    "themeState",
    "mazeGameState",
    "appManager", // Persist user's app preferences
    "liveViewState", // Persist position controller visibility
    // liveStreamState uses nested persist config above
  ],
  //blacklist: ['webSocketState'],  // Do not persist these
};

//#####################################################################################
// Persisted reducer
const persistedReducer = persistReducer(persistConfig, rootReducer);

//#####################################################################################
// Action creator for syncing the state across tabs
const SYNC_STATE = "SYNC_STATE";

// Action creator for syncing the state across tabs
const syncStateAction = (updatedState) => ({
  type: SYNC_STATE,
  payload: updatedState,
});

// Handle SYNC_STATE action to merge updated state from other tabs
const rootReducerWithSync = (state, action) => {
  // If SYNC_STATE action is dispatched, replace the state with the updated state
  if (action.type === SYNC_STATE) {
    return { ...state, ...action.payload }; // Merge updated state
  }

  // Use the persistedReducer otherwise
  return persistedReducer(state, action);
};

//#####################################################################################
// Sync state across tabs
const syncStateAcrossTabs = () => {
  window.addEventListener("storage", (event) => {
    if (event.key === "persist:root") {
      console.log("State change from another tab detected!");
      const updatedState = JSON.parse(event.newValue);
      // Check if any slice is stringified and needs parsing
      Object.keys(updatedState).forEach((sliceKey) => {
        if (typeof updatedState[sliceKey] === "string") {
          updatedState[sliceKey] = JSON.parse(updatedState[sliceKey]);
        }
      });
      // Dispatch SYNC_STATE action with updated state
      store.dispatch(syncStateAction(updatedState));
    }
  });
};

// Activate the sync function
syncStateAcrossTabs();

//#####################################################################################
// Create the Redux store with the sync logic
// Disable expensive development middlewares for high-frequency WebSocket updates
const store = configureStore({
  reducer: rootReducerWithSync, // Use rootReducerWithSync to manage the state
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      // Disable serializable check - our state contains some non-serializable data
      // and this check is very slow with frequent WebSocket dispatches
      serializableCheck: false,
      // Disable immutable check - this is also slow with large state updates
      immutableCheck: false,
    }),
});

//#####################################################################################
// Persistor setup for Redux-Persist
export const persistor = persistStore(store);

// Export the store
export default store;
