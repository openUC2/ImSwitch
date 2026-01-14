// LaserSlice.js - Redux slice for laser state management via WebSocket callbacks
import { createSlice } from '@reduxjs/toolkit';

/**
 * LaserSlice manages laser state received from WebSocket signals (sigUpdateLaserPower).
 * 
 * Signal format:
 * {
 *   "p0": {
 *     "635": {       // laser name/wavelength
 *       "power": 10000,
 *       "enabled": true
 *     }
 *   }
 * }
 * 
 * State structure:
 * {
 *   lasers: {
 *     "635": { power: 10000, enabled: true },
 *     "488": { power: 5000, enabled: false },
 *     ...
 *   }
 * }
 */

const initialState = {
  // Map of laser name -> { power, enabled }
  lasers: {},
};

const laserSlice = createSlice({
  name: 'laserState',
  initialState,
  reducers: {
    /**
     * Update a single laser's state (power and/or enabled).
     * @param {Object} action.payload - { laserName: string, power?: number, enabled?: boolean }
     */
    setLaserState: (state, action) => {
      const { laserName, power, enabled } = action.payload;
      console.log("LaserSlice.setLaserState:", action.payload);
      
      if (!state.lasers[laserName]) {
        state.lasers[laserName] = { power: 0, enabled: false };
      }
      
      if (power !== undefined) {
        state.lasers[laserName].power = power;
      }
      if (enabled !== undefined) {
        state.lasers[laserName].enabled = enabled;
      }
    },
    
    /**
     * Batch update multiple lasers at once.
     * @param {Object} action.payload - { "laserName": { power, enabled }, ... }
     */
    setLasersState: (state, action) => {
      console.log("LaserSlice.setLasersState:", action.payload);
      const lasersData = action.payload;
      
      Object.entries(lasersData).forEach(([laserName, laserData]) => {
        if (!state.lasers[laserName]) {
          state.lasers[laserName] = { power: 0, enabled: false };
        }
        
        if (laserData.power !== undefined) {
          state.lasers[laserName].power = laserData.power;
        }
        if (laserData.enabled !== undefined) {
          state.lasers[laserName].enabled = laserData.enabled;
        }
      });
    },
    
    /**
     * Set laser power only.
     * @param {Object} action.payload - { laserName: string, power: number }
     */
    setLaserPower: (state, action) => {
      const { laserName, power } = action.payload;
      
      if (!state.lasers[laserName]) {
        state.lasers[laserName] = { power: 0, enabled: false };
      }
      state.lasers[laserName].power = power;
    },
    
    /**
     * Set laser enabled state only.
     * @param {Object} action.payload - { laserName: string, enabled: boolean }
     */
    setLaserEnabled: (state, action) => {
      const { laserName, enabled } = action.payload;
      
      if (!state.lasers[laserName]) {
        state.lasers[laserName] = { power: 0, enabled: false };
      }
      state.lasers[laserName].enabled = enabled;
    },
    
    /**
     * Reset all laser states.
     */
    resetState: () => {
      return { ...initialState };
    },
  },
});

// Export actions
export const {
  setLaserState,
  setLasersState,
  setLaserPower,
  setLaserEnabled,
  resetState,
} = laserSlice.actions;

// Selectors
export const getLaserState = (state) => state.laserState;
export const getLasers = (state) => state.laserState.lasers;
export const getLaserByName = (laserName) => (state) => state.laserState.lasers[laserName];

// Export reducer
export default laserSlice.reducer;
