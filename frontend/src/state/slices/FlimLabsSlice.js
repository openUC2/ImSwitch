// FlimLabsSlice.js - Redux slice for the FLIM LABS bridge (dev-mode panel
// inside the galvo scanner UI). Talks directly to a remote flim-imager
// server (Docker), NOT to the ImSwitch backend.
import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  // Remote flim-server connection (Docker container, possibly another host)
  host: 'localhost',
  port: 5249,

  // Connection / card state
  connected: false,
  cardSerial: null,

  // Acquisition state
  running: false,
  step: 'scouting', // 'scouting' | 'imaging'
  currentFrame: 0,
  cps: 0,
  lastDataFile: null,
  firmware: null,
  error: null,

  // Acquisition parameters
  frequencyMhz: 80,
  laserSync: 'in', // 'in' (sync-in) | 'out'
  reconstruction: 'PLF', // 'PLF' | 'LF' | 'F'
  enable100ps: false,
  channels: [true, false, false, false, false, false, false, false],
  maxFrames: 0, // 0 = infinite

  // Bridge behaviour
  syncWithGalvo: true, // derive image size + dwell from galvo scan config
  autoStartGalvo: true, // arm FLIM first, then start the galvo scan
  // Manual values used when syncWithGalvo is off
  manualImageWidth: 256,
  manualImageHeight: 256,
  manualDwellTime: 5,
};

const flimLabsSlice = createSlice({
  name: 'flimLabs',
  initialState,
  reducers: {
    setFlimHost: (state, action) => {
      state.host = action.payload;
      state.connected = false;
      state.cardSerial = null;
    },
    setFlimPort: (state, action) => {
      state.port = action.payload;
      state.connected = false;
      state.cardSerial = null;
    },
    setFlimConnected: (state, action) => {
      state.connected = action.payload.connected;
      state.cardSerial = action.payload.cardSerial ?? null;
    },
    setFlimParam: (state, action) => {
      const { param, value } = action.payload;
      state[param] = value;
    },
    toggleFlimChannel: (state, action) => {
      const idx = action.payload;
      state.channels[idx] = !state.channels[idx];
    },
    setFlimRunning: (state, action) => {
      state.running = action.payload.running;
      if (action.payload.step) state.step = action.payload.step;
      if (action.payload.firmware !== undefined) state.firmware = action.payload.firmware;
      if (!action.payload.running) {
        state.cps = 0;
      } else {
        state.currentFrame = 0;
        state.lastDataFile = null;
      }
    },
    setFlimProgress: (state, action) => {
      if (action.payload.currentFrame !== undefined) state.currentFrame = action.payload.currentFrame;
      if (action.payload.cps !== undefined) state.cps = action.payload.cps;
    },
    setFlimDataFile: (state, action) => {
      state.lastDataFile = action.payload;
    },
    setFlimError: (state, action) => {
      state.error = action.payload;
    },
    clearFlimError: (state) => {
      state.error = null;
    },
  },
});

export const {
  setFlimHost,
  setFlimPort,
  setFlimConnected,
  setFlimParam,
  toggleFlimChannel,
  setFlimRunning,
  setFlimProgress,
  setFlimDataFile,
  setFlimError,
  clearFlimError,
} = flimLabsSlice.actions;

export const getFlimLabsState = (state) => state.flimLabsState;

export default flimLabsSlice.reducer;
