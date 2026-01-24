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

  // Light Controls
  lightStates: {},
  availableLights: [],  // Initialize as empty array to prevent map errors
  uvLedActive: false,
  visibleLedActive: false,

  // Hardware Status
  hardwareStatus: {
    gpio_available: false,
    oled_available: false,
    i2c_available: false,
    simulation_mode: true
  },

  // Images
  latestImage: null,
  imageFormat: "jpeg",

  // Display
  lcdDisplay: {
    line1: "",
    line2: "",
    line3: ""
  },

  // Button States
  buttonStates: {},
  availableButtons: [],

  // HMI Menu State
  hmiStatus: {
    hmi_open: false,
    current_menu_state: "main",
    monitoring_active: false
  },

  // Timing Configuration
  timingConfig: {
    acquisitionInterval: 60,
    stabilizationTime: 5,
    preAcquisitionDelay: 2,
    postAcquisitionDelay: 1
  },

  // Live Sensor Data (from modular sensors component)
  sensorData: {
    light: { lux: 0, visible: 0 },
    innerTemp: 0,
    environment: { temperature: 0, humidity: 0, pressure: 0 },
    power: { voltage: 0, current: 0, power: 0 }
  },
  availableSensors: [],

  // Capture Session State (from modular capturing component)
  captureState: "idle",  // idle, starting, waiting, capturing, processing, stopping, error
  captureSession: {
    session_id: null,
    start_time: null,
    expected_images: 0,
    captured_images: 0,
    failed_captures: 0,
    folder_path: "",
    current_exposure: 150,
    current_gain: 7.0
  },

  // Experiment Times (from modular times component)
  experimentTimes: {
    sunrise: "",
    sunset: "",
    start_time: "",
    end_time: "",
    lepiled_end_time: "",
    is_in_capture_window: false
  },

  // Lepmon Configuration (from modular config component)
  lepmonConfig: {
    general: {
      serial_number: "",
      hardware_version: "Pro_Gen_2"
    },
    capture: {
      dusk_threshold: 90,
      interval: 2,
      initial_exposure: 150,
      initial_gain: 7
    },
    image_quality: {
      brightness_reference: 170,
      gamma_correction: true
    },
    gps: {
      latitude: 0,
      longitude: 0
    }
  }
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

    // Display actions
    setLcdDisplay: (state, action) => {
      state.lcdDisplay = { ...state.lcdDisplay, ...action.payload };
    },

    // Button actions
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

    // HMI Status actions
    setHmiStatus: (state, action) => {
      state.hmiStatus = { ...state.hmiStatus, ...action.payload };
    },

    // Timing Configuration actions
    setTimingConfig: (state, action) => {
      state.timingConfig = { ...state.timingConfig, ...action.payload };
    },

    // Sensor Data actions
    setSensorData: (state, action) => {
      state.sensorData = { ...state.sensorData, ...action.payload };
    },
    setAvailableSensors: (state, action) => {
      state.availableSensors = action.payload;
    },

    // Capture State actions
    setCaptureState: (state, action) => {
      state.captureState = action.payload;
    },
    setCaptureSession: (state, action) => {
      state.captureSession = { ...state.captureSession, ...action.payload };
    },

    // Experiment Times actions
    setExperimentTimes: (state, action) => {
      state.experimentTimes = { ...state.experimentTimes, ...action.payload };
    },

    // Lepmon Config actions
    setLepmonConfig: (state, action) => {
      state.lepmonConfig = { ...state.lepmonConfig, ...action.payload };
    },

    // UV and Visible LED state
    setUvLedActive: (state, action) => {
      state.uvLedActive = action.payload;
    },
    setVisibleLedActive: (state, action) => {
      state.visibleLedActive = action.payload;
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
  // Light controls
  setLightStates,
  setLightState,
  setAvailableLights,
  setUvLedActive,
  setVisibleLedActive,
  // Hardware
  setHardwareStatus,
  // Images
  setLatestImage,
  setImageFormat,
  // Display
  setLcdDisplay,
  // Buttons
  setButtonStates,
  setButtonState,
  setAvailableButtons,
  // HMI
  setHmiStatus,
  // Timing
  setTimingConfig,
  // Sensors
  setSensorData,
  setAvailableSensors,
  // Capture
  setCaptureState,
  setCaptureSession,
  // Times
  setExperimentTimes,
  // Config
  setLepmonConfig,
} = lepmonSlice.actions;

// Export selector
export const getLepmonState = (state) => state.lepmon;

// Export reducer
export default lepmonSlice.reducer;