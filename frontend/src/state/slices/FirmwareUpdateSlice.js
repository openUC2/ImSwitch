// Runtime-only (NOT persisted): the result of the backend's firmware check
// after startup, pushed via sigFirmwareUpdatesAvailable (WebSocketHandler) or
// pulled once on connect (FirmwareUpdatePrompt).
import { createSlice } from "@reduxjs/toolkit";

const firmwareUpdateSlice = createSlice({
  name: "firmwareUpdate",
  initialState: { prompt: null },
  reducers: {
    setFirmwarePrompt: (state, action) => {
      state.prompt = action.payload;
    },
  },
});

export const { setFirmwarePrompt } = firmwareUpdateSlice.actions;
export const getFirmwareUpdateState = (state) => state.firmwareUpdate;
export default firmwareUpdateSlice.reducer;
