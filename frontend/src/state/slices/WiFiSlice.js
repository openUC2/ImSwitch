/**
 * @deprecated This slice is no longer used as of November 2025.
 * WiFiController has been refactored to display the device-admin panel
 * via iframe instead of implementing WiFi controls directly.
 *
 * The WiFiController component now only uses ConnectionSettingsSlice
 * to get the hostIP and displays the admin panel at:
 * ${hostIP}/admin/panel/internet
 *
 * This file is kept for reference but can be safely deleted.
 * It has been removed from src/state/store.js
 */

import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for WiFi controller
const initialWiFiState = {
  tabIndex: 0,

  // Available networks from scan
  availableNetworks: [],

  // Current connection status
  currentSSID: null,
  currentIfname: null,
  isConnected: false,

  // Connection form data
  selectedSSID: "",
  password: "",
  ifname: "",
  showPassword: false,

  // Access Point form data
  apSSID: "",
  apPassword: "",
  apIfname: "",
  showApPassword: false,
  apConName: "imswitch-hotspot",
  apBand: "bg",
  apChannel: null,

  // Status states
  isScanning: false,
  isConnecting: false,
  isCreatingAP: false,
  isAPActive: false,

  // Connection info
  connectionInfo: {},
  apInfo: {},

  // Error handling
  lastError: null,
};

// Create slice
const wifiSlice = createSlice({
  name: "wifiState",
  initialState: initialWiFiState,
  reducers: {
    // Tab management
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },

    // Network scanning
    setIsScanning: (state, action) => {
      state.isScanning = action.payload;
    },
    setAvailableNetworks: (state, action) => {
      state.availableNetworks = action.payload;
    },

    // Current connection
    setCurrentSSID: (state, action) => {
      state.currentSSID = action.payload;
    },
    setCurrentIfname: (state, action) => {
      state.currentIfname = action.payload;
    },
    setIsConnected: (state, action) => {
      state.isConnected = action.payload;
    },
    setConnectionInfo: (state, action) => {
      state.connectionInfo = action.payload;
    },

    // Connection form
    setSelectedSSID: (state, action) => {
      state.selectedSSID = action.payload;
    },
    setPassword: (state, action) => {
      state.password = action.payload;
    },
    setShowPassword: (state, action) => {
      state.showPassword = action.payload;
    },
    setIfname: (state, action) => {
      state.ifname = action.payload;
    },
    setIsConnecting: (state, action) => {
      state.isConnecting = action.payload;
    },

    // Access Point form
    setApSSID: (state, action) => {
      state.apSSID = action.payload;
    },
    setApPassword: (state, action) => {
      state.apPassword = action.payload;
    },
    setShowApPassword: (state, action) => {
      state.showApPassword = action.payload;
    },
    setApIfname: (state, action) => {
      state.apIfname = action.payload;
    },
    setApConName: (state, action) => {
      state.apConName = action.payload;
    },
    setApBand: (state, action) => {
      state.apBand = action.payload;
    },
    setApChannel: (state, action) => {
      state.apChannel = action.payload;
    },
    setIsCreatingAP: (state, action) => {
      state.isCreatingAP = action.payload;
    },
    setIsAPActive: (state, action) => {
      state.isAPActive = action.payload;
    },
    setApInfo: (state, action) => {
      state.apInfo = action.payload;
    },

    // Error handling
    setLastError: (state, action) => {
      state.lastError = action.payload;
    },
    clearError: (state) => {
      state.lastError = null;
    },

    // Reset state
    resetState: () => {
      return { ...initialWiFiState };
    },
  },
});

// Export actions
export const {
  setTabIndex,
  setIsScanning,
  setAvailableNetworks,
  setCurrentSSID,
  setCurrentIfname,
  setIsConnected,
  setConnectionInfo,
  setSelectedSSID,
  setPassword,
  setShowPassword,
  setIfname,
  setIsConnecting,
  setApSSID,
  setApPassword,
  setShowApPassword,
  setApIfname,
  setApConName,
  setApBand,
  setApChannel,
  setIsCreatingAP,
  setIsAPActive,
  setApInfo,
  setLastError,
  clearError,
  resetState,
} = wifiSlice.actions;

// Export selector
export const getWiFiState = (state) => state.wifiState;

// Export reducer
export default wifiSlice.reducer;
