// Live CAN streaming OTA progress per node, fed by sigOTAStatusUpdate
// (WebSocketHandler). FirmwareUpdateDialog shows it for the board being
// updated; the update itself is tracked by the backend (getFirmwareUpdateStatus).
import { createSlice } from "@reduxjs/toolkit";

const canOtaSlice = createSlice({
  name: "canOta",
  initialState: {
    updateProgress: {}, // { canId: { status, message, progress, timestamp } }
  },
  reducers: {
    setUpdateProgress: (state, action) => {
      const { canId, ...progress } = action.payload;
      state.updateProgress[canId] = progress;
    },
  },
});

export const { setUpdateProgress } = canOtaSlice.actions;
export const getCanOtaState = (state) => state.canOtaState;
export default canOtaSlice.reducer;
