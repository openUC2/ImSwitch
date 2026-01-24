import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  message: "",
  type: "info", // "info", "success", "warning", "error"
};

const notificationSlice = createSlice({
  name: "notification",
  initialState,
  reducers: {
    setNotification: (state, action) => {
      state.message = action.payload.message || "";
      state.type = action.payload.type || "info";
    },
    clearNotification: (state) => {
      state.message = "";
      state.type = "info";
    },
  },
});

export const { setNotification, clearNotification } = notificationSlice.actions;
export const getNotificationState = (state) => state.notification;
export default notificationSlice.reducer;
