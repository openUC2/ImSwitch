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
  serverHealthy: null, // null = unknown, true/false after /health probe

  // Acquisition state
  // Whether the ImSwitch backend owns the flim-imager /data socket. False on
  // startup: the stream takes one consumer, so ImSwitch stays off the card
  // until the FLIM panel arms it, leaving it free for the FLIM LABS web UI.
  armed: false,
  running: false,
  paused: false, // stopped but image buffer kept (Pause vs Reset)
  step: 'scouting', // 'scouting' | 'imaging' | 'calibration' | 'phasors'
  currentFrame: 0,
  cps: 0,
  lastDataFile: null,
  firmware: null,
  error: null,

  // Acquisition parameters — wizard defaults for the openUC2 FLIM rig:
  // detector on input CH2 (SMA), laser sync IN on SMA at 40 MHz,
  // pixel+line+frame markers from the galvo scanner (PLF).
  frequencyMhz: 40,
  laserSync: 'in', // 'in' (sync-in) | 'out'
  reconstruction: 'PLF', // 'PLF' | 'LF' | 'F'
  enable100ps: false,
  channels: [false, true, false, false, false, false, false, false],
  maxFrames: 0, // 0 = infinite

  // Bridge behaviour
  syncWithGalvo: true, // derive image size + dwell from galvo scan config
  autoStartGalvo: true, // arm FLIM first, then start the galvo scan
  // Manual values used when syncWithGalvo is off
  manualImageWidth: 256,
  manualImageHeight: 256,
  manualDwellTime: 5,

  // Calibration (solid calibrator with known lifetime)
  calibrationTauNs: 4.0, // e.g. FLIM LABS solid calibrator ~4 ns — set to your calibrator
  calibrationHarmonics: 1,
  // Live results of the running/last calibration: [{channel, harmonic, phase, modulation}]
  calibrationResults: [],
  // Server-side path of the calibration reference JSON (used by phasors runs)
  calibrationReferenceFile: null,
  calibrationTimestamp: null,

  // Phasors
  phasorHarmonic: 1,

  // Server-side export (written into the container's ~/.flim-labs volume)
  exportEnabled: false,
  exportFilename: 'imswitch_flim',
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
    setFlimArmed: (state, action) => {
      state.armed = !!action.payload;
      if (!state.armed) {
        state.running = false;
        state.cps = 0;
      }
    },
    setFlimRunning: (state, action) => {
      state.running = action.payload.running;
      if (action.payload.armed !== undefined) state.armed = !!action.payload.armed;
      if (action.payload.step) state.step = action.payload.step;
      if (action.payload.firmware !== undefined) state.firmware = action.payload.firmware;
      if (action.payload.paused !== undefined) state.paused = action.payload.paused;
      if (!action.payload.running) {
        state.cps = 0;
      } else {
        state.paused = false;
        state.currentFrame = 0;
        state.lastDataFile = null;
      }
    },
    setFlimHealth: (state, action) => {
      state.serverHealthy = action.payload;
    },
    addFlimCalibrationResult: (state, action) => {
      const { channel, harmonic, phase, modulation } = action.payload;
      const idx = state.calibrationResults.findIndex(
        (r) => r.channel === channel && r.harmonic === harmonic
      );
      const entry = { channel, harmonic, phase, modulation };
      if (idx >= 0) state.calibrationResults[idx] = entry;
      else state.calibrationResults.push(entry);
    },
    setFlimCalibrationReference: (state, action) => {
      state.calibrationReferenceFile = action.payload.referenceFile ?? null;
      state.calibrationTimestamp = action.payload.timestamp ?? null;
      if (action.payload.clearResults) state.calibrationResults = [];
    },
    hydrateFlimConfig: (state, action) => {
      // Restore persisted settings (subset of state) from the ImSwitch setup file
      const cfg = action.payload || {};
      const allowed = [
        'host', 'port', 'frequencyMhz', 'laserSync', 'reconstruction', 'enable100ps',
        'channels', 'maxFrames', 'syncWithGalvo', 'autoStartGalvo',
        'manualImageWidth', 'manualImageHeight', 'manualDwellTime',
        'calibrationTauNs', 'calibrationHarmonics', 'calibrationReferenceFile',
        'calibrationTimestamp', 'phasorHarmonic', 'exportEnabled', 'exportFilename',
      ];
      for (const key of allowed) {
        if (cfg[key] !== undefined) state[key] = cfg[key];
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
  setFlimArmed,
  setFlimProgress,
  setFlimDataFile,
  setFlimError,
  clearFlimError,
  setFlimHealth,
  addFlimCalibrationResult,
  setFlimCalibrationReference,
  hydrateFlimConfig,
} = flimLabsSlice.actions;

export const getFlimLabsState = (state) => state.flimLabsState;

export default flimLabsSlice.reducer;
