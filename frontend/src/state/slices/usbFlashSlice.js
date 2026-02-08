import { createSlice } from "@reduxjs/toolkit";

/**
 * Redux slice for USB firmware flash wizard state management
 * Used for flashing the master CAN HAT via USB/esptool
 */

const initialState = {
  // Wizard step navigation (0: Firmware Server, 1: Select Firmware, 2: Port Selection, 3: Flash Progress, 4: Complete)
  currentStep: 0,
  isWizardOpen: false,

  // Serial port configuration
  availablePorts: [],
  selectedPort: null, // null means auto-detect
  isLoadingPorts: false,
  portMatch: "HAT", // Substring to identify HAT in port metadata

  // Firmware server configuration
  firmwareServerUrl: "",
  defaultFirmwareServerUrl: "",
  isLoadingFirmware: false,

  // Available firmware files (flat list from server)
  firmwareFiles: [], // [{ filename, size, mod_time, url }]
  selectedFirmware: null, // { filename, size, mod_time, url }

  // Flash options
  baudRate: 921600,
  reconnectAfter: true,

  // Flash progress
  isFlashing: false,
  flashStatus: "idle", // idle, disconnecting, downloading, flashing, reconnecting, success, failed
  flashProgress: 0,
  flashMessage: "",
  flashDetails: null,

  // Result
  flashResult: null,

  // General states
  error: null,
  successMessage: null,
};

const usbFlashSlice = createSlice({
  name: "usbFlash",
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
      if (state.currentStep < 4) {
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
      state.flashStatus = "idle";
      state.flashProgress = 0;
      state.flashMessage = "";
      state.flashDetails = null;
      state.flashResult = null;
      state.error = null;
      state.successMessage = null;
      state.selectedFirmware = null;
      state.firmwareFiles = [];
    },

    // Serial ports
    setAvailablePorts: (state, action) => {
      state.availablePorts = action.payload;
    },
    setSelectedPort: (state, action) => {
      state.selectedPort = action.payload;
    },
    setIsLoadingPorts: (state, action) => {
      state.isLoadingPorts = action.payload;
    },
    setPortMatch: (state, action) => {
      state.portMatch = action.payload;
    },

    // Firmware
    setFirmwareServerUrl: (state, action) => {
      state.firmwareServerUrl = action.payload;
    },
    setDefaultFirmwareServerUrl: (state, action) => {
      state.defaultFirmwareServerUrl = action.payload;
    },
    setFirmwareFiles: (state, action) => {
      state.firmwareFiles = action.payload;
    },
    setSelectedFirmware: (state, action) => {
      state.selectedFirmware = action.payload;
    },
    setIsLoadingFirmware: (state, action) => {
      state.isLoadingFirmware = action.payload;
    },

    // Flash options
    setBaudRate: (state, action) => {
      state.baudRate = action.payload;
    },
    setReconnectAfter: (state, action) => {
      state.reconnectAfter = action.payload;
    },

    // Flash progress
    setIsFlashing: (state, action) => {
      state.isFlashing = action.payload;
    },
    setFlashStatus: (state, action) => {
      state.flashStatus = action.payload;
    },
    setFlashProgress: (state, action) => {
      state.flashProgress = action.payload;
    },
    setFlashMessage: (state, action) => {
      state.flashMessage = action.payload;
    },
    setFlashDetails: (state, action) => {
      state.flashDetails = action.payload;
    },
    setFlashResult: (state, action) => {
      state.flashResult = action.payload;
    },

    // Update flash progress from socket event
    updateFlashProgress: (state, action) => {
      const { status, progress, message, details } = action.payload;
      if (status) state.flashStatus = status;
      // Only update progress when a real value is provided (>= 0)
      if (progress !== undefined && progress >= 0) state.flashProgress = progress;
      if (message) state.flashMessage = message;
      if (details) state.flashDetails = details;
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
  },
});

// Export actions
export const {
  setCurrentStep,
  setIsWizardOpen,
  nextStep,
  previousStep,
  resetWizard,
  setAvailablePorts,
  setSelectedPort,
  setIsLoadingPorts,
  setPortMatch,
  setFirmwareServerUrl,
  setDefaultFirmwareServerUrl,
  setFirmwareFiles,
  setSelectedFirmware,
  setIsLoadingFirmware,
  setBaudRate,
  setReconnectAfter,
  setIsFlashing,
  setFlashStatus,
  setFlashProgress,
  setFlashMessage,
  setFlashDetails,
  setFlashResult,
  updateFlashProgress,
  setError,
  setSuccessMessage,
  clearMessages,
} = usbFlashSlice.actions;

// Selectors
export const getUsbFlashState = (state) => state.usbFlash;

export default usbFlashSlice.reducer;
