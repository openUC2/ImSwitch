// src/state/slices/UIPreferencesSlice.js
// Frontend-only UI preferences. Persisted via the root persist whitelist in
// store.js, so a kiosk keeps its choice across reloads.
//
// Both settings are tri-state: "auto" follows the device (see
// hooks/useDeviceProfile.js), "on"/"off" force it — e.g. a Raspberry Pi
// touchscreen with a mouse plugged in reports a fine pointer and would
// otherwise get the desktop UI.

import { createSlice } from "@reduxjs/toolkit";

export const TRI_STATE = ["auto", "on", "off"];

const initialState = {
  // Larger touch targets and the stream-first compact layouts.
  touchMode: "auto",
  // On-screen number pad for numeric fields (there is no OS keyboard on a
  // kiosk). "auto" = touch UI on a device without its own soft keyboard.
  onScreenKeypad: "auto",
};

const uiPreferencesSlice = createSlice({
  name: "uiPreferencesState",
  initialState,
  reducers: {
    setTouchMode: (state, action) => {
      if (TRI_STATE.includes(action.payload)) state.touchMode = action.payload;
    },
    setOnScreenKeypad: (state, action) => {
      if (TRI_STATE.includes(action.payload)) {
        state.onScreenKeypad = action.payload;
      }
    },
  },
});

export const { setTouchMode, setOnScreenKeypad } = uiPreferencesSlice.actions;
export const getUIPreferences = (state) =>
  state.uiPreferencesState || initialState;
export default uiPreferencesSlice.reducer;
