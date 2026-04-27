import { createSlice } from "@reduxjs/toolkit";

let nextNotificationId = 1;

const initialState = {
  message: "",
  type: "info", // "info", "success", "warning", "error"
  notifications: [],
};

const syncLegacyFields = (state) => {
  const latest = state.notifications[state.notifications.length - 1];
  state.message = latest?.message || "";
  state.type = latest?.type || "info";
};

const notificationSlice = createSlice({
  name: "notification",
  initialState,
  reducers: {
    setNotification: (state, action) => {
      const message = action.payload?.message || "";
      const type = action.payload?.type || "info";

      if (!message) {
        return;
      }

      state.notifications.push({
        id: nextNotificationId++,
        message,
        type,
      });

      // Keep queue bounded.
      if (state.notifications.length > 30) {
        state.notifications.shift();
      }

      syncLegacyFields(state);
    },
    clearNotification: (state, action) => {
      const notificationId = action.payload;

      if (typeof notificationId === "number") {
        state.notifications = state.notifications.filter(
          (item) => item.id !== notificationId,
        );
      } else {
        // Backward-compatible default behavior for existing callers.
        state.notifications.shift();
      }

      syncLegacyFields(state);
    },
  },
});

export const { setNotification, clearNotification } = notificationSlice.actions;
export const getNotificationState = (state) => state.notification;
export default notificationSlice.reducer;
