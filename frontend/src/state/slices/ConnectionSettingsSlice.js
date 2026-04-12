import { createSlice } from "@reduxjs/toolkit";

// Smart detection of frontend URL for backend defaults
const getSmartDefaults = () => {
  const location = window.location;
  const protocol = location.protocol.replace(":", ""); // 'http' or 'https'
  const hostname = location.hostname;
  const isHttps = location.protocol === "https:";
  // Use the actual port from the URL; fall back to standard defaults (443/80)
  // for cases where the browser omits the port (e.g. https on 443, http on 80)
  const port = location.port || (isHttps ? "443" : "80");

  return {
    ip: `${protocol}://${hostname}`,
    websocketPort: port,
    apiPort: port,
  };
};

// Define the initial state with smart defaults
const initialState = getSmartDefaults();

// Create webSocketSettingsSlice slice
const connectionSettingsSlice = createSlice({
  name: "connectionSettingsState",
  initialState: initialState,
  reducers: {
    setIp: (state, action) => {
      console.log("setIp", action.payload);
      state.ip = action.payload;
    },
    setWebsocketPort: (state, action) => {
      console.log("setWebsocketPort", action.payload);
      state.websocketPort = action.payload;
    },
    setApiPort: (state, action) => {
      console.log("setApiPort", action.payload);
      state.apiPort = action.payload;
    },

    resetState: (state) => {
      console.log("resetState");
      return { ...initialState }; // Reset to initial state
    },
  },
});

// Export actions from wellSelectorState slice
export const { setIp, setWebsocketPort, setApiPort, resetState } =
  connectionSettingsSlice.actions;

// Selector helper
export const getConnectionSettingsState = (state) =>
  state.connectionSettingsState;

// Export reducer from wellSelectorState slice
export default connectionSettingsSlice.reducer;
