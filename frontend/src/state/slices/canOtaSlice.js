import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // Wizard steps (CAN streaming OTA): 0 server config, 1 device scan,
  // 2 device selection, 3 update progress, 4 completion
  currentStep: 0,
  isWizardOpen: false,

  // Firmware server configuration
  firmwareServerUrl: "",
  defaultFirmwareServerUrl: "",
  availableFirmware: {},
  isLoadingFirmwareServer: false,
  isLoadingFirmwareList: false,

  // CAN device scanning
  scannedDevices: [],
  isScanningDevices: false,
  scanError: null,

  // Device selection
  selectedDeviceIds: [],
  
  // OTA update progress
  isUpdating: false,
  updateProgress: {}, // { canId: { status, message, progress, timestamp, method } }
  activeUpdateCount: 0,
  completedUpdateCount: 0,
  failedUpdateCount: 0,

  // Update results
  updateResults: [], // Array of completed update results
  
  // General states
  error: null,
  successMessage: null,
};

const canOtaSlice = createSlice({
  name: "canOta",
  initialState,
  reducers: {
    // Wizard navigation
    setCurrentStep: (state, action) => {
      state.currentStep = action.payload;
    },
    setIsWizardOpen: (state, action) => {
      state.isWizardOpen = action.payload;
    },
    nextStep: (state) => {
      if (state.currentStep < 4) { // 4 = completion
        state.currentStep += 1;
      }
    },
    previousStep: (state) => {
      if (state.currentStep > 0) {
        state.currentStep -= 1;
      }
    },
    resetWizard: (state) => {
      state.currentStep = 0;
      state.selectedDeviceIds = [];
      state.updateProgress = {};
      state.updateResults = [];
      state.activeUpdateCount = 0;
      state.completedUpdateCount = 0;
      state.failedUpdateCount = 0;
      state.error = null;
      state.successMessage = null;
      state.scanError = null;
    },

    // Firmware server
    setFirmwareServerUrl: (state, action) => {
      state.firmwareServerUrl = action.payload;
    },
    setDefaultFirmwareServerUrl: (state, action) => {
      state.defaultFirmwareServerUrl = action.payload;
      // Only set current value if it's empty
      if (!state.firmwareServerUrl) {
        state.firmwareServerUrl = action.payload;
      }
    },
    setAvailableFirmware: (state, action) => {
      state.availableFirmware = action.payload;
    },
    setIsLoadingFirmwareServer: (state, action) => {
      state.isLoadingFirmwareServer = action.payload;
    },
    setIsLoadingFirmwareList: (state, action) => {
      state.isLoadingFirmwareList = action.payload;
    },

    // CAN device scanning
    setScannedDevices: (state, action) => {
      state.scannedDevices = action.payload;
    },
    setIsScanningDevices: (state, action) => {
      state.isScanningDevices = action.payload;
    },
    setScanError: (state, action) => {
      state.scanError = action.payload;
    },

    // Device selection
    setSelectedDeviceIds: (state, action) => {
      state.selectedDeviceIds = action.payload;
    },
    toggleDeviceSelection: (state, action) => {
      const canId = action.payload;
      const index = state.selectedDeviceIds.indexOf(canId);
      if (index >= 0) {
        state.selectedDeviceIds.splice(index, 1);
      } else {
        state.selectedDeviceIds.push(canId);
      }
    },
    selectAllDevices: (state) => {
      state.selectedDeviceIds = state.scannedDevices.map((device) => device.canId);
    },
    deselectAllDevices: (state) => {
      state.selectedDeviceIds = [];
    },

    // OTA update progress
    setIsUpdating: (state, action) => {
      state.isUpdating = action.payload;
    },
    setUpdateProgress: (state, action) => {
      const { canId, status, message, progress, timestamp } = action.payload;
      const previousStatus = state.updateProgress[canId]?.status;
      state.updateProgress[canId] = {
        status,
        message,
        progress: progress || 0,
        timestamp: timestamp || new Date().toISOString(),
      };
      
      // Only update counters on first transition to a terminal state
      // to avoid double-counting from multiple streaming updates
      const isTerminal = (s) =>
        s === "completed" || s === "success" || s === "failed" || s === "error";
      if (isTerminal(status) && !isTerminal(previousStatus)) {
        if (status === "completed" || status === "success") {
          state.completedUpdateCount += 1;
        } else {
          state.failedUpdateCount += 1;
        }
      }
    },
    clearUpdateProgress: (state) => {
      state.updateProgress = {};
      state.activeUpdateCount = 0;
      state.completedUpdateCount = 0;
      state.failedUpdateCount = 0;
    },
    setActiveUpdateCount: (state, action) => {
      state.activeUpdateCount = action.payload;
    },

    // Update results
    addUpdateResult: (state, action) => {
      state.updateResults.push(action.payload);
    },
    clearUpdateResults: (state) => {
      state.updateResults = [];
    },

    // General states
    setError: (state, action) => {
      state.error = action.payload;
    },
    setSuccessMessage: (state, action) => {
      state.successMessage = action.payload;
    },
    clearMessages: (state) => {
      state.error = null;
      state.successMessage = null;
    },

    // Complete reset
    resetState: (state) => {
      Object.assign(state, initialState);
    },
  },
});

// Export actions
export const {
  setCurrentStep,
  setIsWizardOpen,
  nextStep,
  previousStep,
  resetWizard,
  setFirmwareServerUrl,
  setDefaultFirmwareServerUrl,
  setAvailableFirmware,
  setIsLoadingFirmwareServer,
  setIsLoadingFirmwareList,
  setScannedDevices,
  setIsScanningDevices,
  setScanError,
  setSelectedDeviceIds,
  toggleDeviceSelection,
  selectAllDevices,
  deselectAllDevices,
  setIsUpdating,
  setUpdateProgress,
  clearUpdateProgress,
  setActiveUpdateCount,
  addUpdateResult,
  clearUpdateResults,
  setError,
  setSuccessMessage,
  clearMessages,
  resetState,
} = canOtaSlice.actions;

// Export selector
export const getCanOtaState = (state) => state.canOtaState;

// Export reducer
export default canOtaSlice.reducer;
