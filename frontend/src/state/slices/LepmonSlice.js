import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // System Status
  isRunning: false,
  currentImageCount: 0,
  serverTime: "",
  deviceTime: "",
  freeSpace: "",
  storagePath: "",
  sharpnessValue: null,

  // Temperature/Environment
  insideTemp: null,
  outsideTemp: null,
  humidity: null,

  // Camera Settings
  exposure: 100,
  gain: 0,
  timelapsePeriod: 60,
  timelapseLocked: true,
  rebootLocked: true,

  // GPS/Location
  lat: null,
  lng: null,

  // Time/Date
  time: "",
  date: "",

  // Light Controls (new)
  lightStates: {},
  availableLights: [],

  // Hardware Status (new)
  hardwareStatus: {
    gpio_available: false,
    oled_available: false,
    i2c_available: false,
    simulation_mode: true
  },

  // Images (new)
  latestImage: null,
  imageFormat: "jpeg",

  // Display (new)
  lcdDisplay: {
    line1: "",
    line2: "",
    line3: "",
    line4: ""
  },

  // Button States (new)
  buttonStates: {},
  availableButtons: [],

  // Timing Configuration (new)
  timingConfig: {
    acquisitionInterval: 60,
    stabilizationTime: 5,
    preAcquisitionDelay: 2,
    postAcquisitionDelay: 1
  },

  // Live Sensor Data (new)
  sensorData: {},
  availableSensors: []
};

const lepmonSlice = createSlice({
  name: "lepmon",
  initialState,
  reducers: {
    // System Status actions
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },
    setCurrentImageCount: (state, action) => {
      state.currentImageCount = action.payload;
    },
    setServerTime: (state, action) => {
      state.serverTime = action.payload;
    },
    setDeviceTime: (state, action) => {
      state.deviceTime = action.payload;
    },
    setFreeSpace: (state, action) => {
      state.freeSpace = action.payload;
    },
    setStoragePath: (state, action) => {
      state.storagePath = action.payload;
    },
    setSharpnessValue: (state, action) => {
      state.sharpnessValue = action.payload;
    },

    // Temperature/Environment actions
    setInsideTemp: (state, action) => {
      state.insideTemp = action.payload;
    },
    setOutsideTemp: (state, action) => {
      state.outsideTemp = action.payload;
    },
    setHumidity: (state, action) => {
      state.humidity = action.payload;
    },

    // Camera Settings actions
    setExposure: (state, action) => {
      state.exposure = action.payload;
    },
    setGain: (state, action) => {
      state.gain = action.payload;
    },
    setTimelapsePeriod: (state, action) => {
      state.timelapsePeriod = action.payload;
    },
    setTimelapseLocked: (state, action) => {
      state.timelapseLocked = action.payload;
    },
    setRebootLocked: (state, action) => {
      state.rebootLocked = action.payload;
    },

    // GPS/Location actions
    setLat: (state, action) => {
      state.lat = action.payload;
    },
    setLng: (state, action) => {
      state.lng = action.payload;
    },

    // Time/Date actions
    setTime: (state, action) => {
      state.time = action.payload;
    },
    setDate: (state, action) => {
      state.date = action.payload;
    },

    // Batch update actions for initial data
    setInitialStatus: (state, action) => {
      const { isRunning, currentImageCount, serverTime, freeSpace } = action.payload;
      state.isRunning = isRunning;
      state.currentImageCount = currentImageCount;
      state.serverTime = serverTime;
      state.freeSpace = freeSpace;
    },
    setInitialParams: (state, action) => {
      const { exposureTime, gain, timelapsePeriod, storagePath } = action.payload;
      state.exposure = exposureTime;
      state.gain = gain;
      state.timelapsePeriod = timelapsePeriod;
      state.storagePath = storagePath;
    },
    setTemperatureData: (state, action) => {
      const { innerTemp, outerTemp, humidity } = action.payload;
      state.insideTemp = innerTemp;
      state.outsideTemp = outerTemp;
      state.humidity = humidity;
    },

    // Light Control actions (new)
    setLightStates: (state, action) => {
      state.lightStates = action.payload;
    },
    setLightState: (state, action) => {
      const { lightName, isOn } = action.payload;
      state.lightStates[lightName] = isOn;
    },
    setAvailableLights: (state, action) => {
      state.availableLights = action.payload;
    },

    // Hardware Status actions (new)
    setHardwareStatus: (state, action) => {
      state.hardwareStatus = { ...state.hardwareStatus, ...action.payload };
    },

    // Image actions (new)
    setLatestImage: (state, action) => {
      state.latestImage = action.payload;
    },
    setImageFormat: (state, action) => {
      state.imageFormat = action.payload;
    },

    // Display actions (new)
    setLcdDisplay: (state, action) => {
      state.lcdDisplay = { ...state.lcdDisplay, ...action.payload };
    },

    // Button actions (new)
    setButtonStates: (state, action) => {
      state.buttonStates = action.payload;
    },
    setButtonState: (state, action) => {
      const { buttonName, isPressed } = action.payload;
      state.buttonStates[buttonName] = isPressed;
    },
    setAvailableButtons: (state, action) => {
      state.availableButtons = action.payload;
    },

    // Timing Configuration actions (new)
    setTimingConfig: (state, action) => {
      state.timingConfig = { ...state.timingConfig, ...action.payload };
    },

    // Sensor Data actions (new)
    setSensorData: (state, action) => {
      state.sensorData = { ...state.sensorData, ...action.payload };
    },
    setAvailableSensors: (state, action) => {
      state.availableSensors = action.payload;
    },
  },
});

// Export actions
export const {
  setIsRunning,
  setCurrentImageCount,
  setServerTime,
  setDeviceTime,
  setFreeSpace,
  setStoragePath,
  setSharpnessValue,
  setInsideTemp,
  setOutsideTemp,
  setHumidity,
  setExposure,
  setGain,
  setTimelapsePeriod,
  setTimelapseLocked,
  setRebootLocked,
  setLat,
  setLng,
  setTime,
  setDate,
  setInitialStatus,
  setInitialParams,
  setTemperatureData,
  // New exports
  setLightStates,
  setLightState,
  setAvailableLights,
  setHardwareStatus,
  setLatestImage,
  setImageFormat,
  setLcdDisplay,
  setButtonStates,
  setButtonState,
  setAvailableButtons,
  setTimingConfig,
  setSensorData,
  setAvailableSensors,
} = lepmonSlice.actions;

// Export selector
export const getLepmonState = (state) => state.lepmon;

// Export reducer
export default lepmonSlice.reducer;