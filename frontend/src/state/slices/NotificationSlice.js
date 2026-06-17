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
    setNotification: {
      reducer: (state, action) => {
        const { id, message, type, autoHideDuration } = action.payload;

        if (!message) {
          return;
        }

        state.notifications.push({
          id,
          message,
          type,
          autoHideDuration,
        });

        // Keep queue bounded.
        if (state.notifications.length > 30) {
          state.notifications.shift();
        }

        syncLegacyFields(state);
      },
      prepare: (payload = {}) => {
        const message = payload.message || "";
        const type = payload.type || "info";
        return {
          payload: {
            id: nextNotificationId++,
            message,
            type,
            autoHideDuration: payload.autoHideDuration,
          },
        };
      },
    },
    clearNotification: (state, action) => {
      const notificationId = action.payload;

      if (typeof notificationId === "number") {
        state.notifications = state.notifications.filter(
          (item) => item.id !== notificationId,
        );
      }

      // NOTE: clearNotification() without an ID is intentionally a no-op.
      // Legacy timer-based clear calls would otherwise remove unrelated items
      // from the queue now that multiple notifications can be visible.

      syncLegacyFields(state);
    },
  },
});

export const { setNotification, clearNotification } = notificationSlice.actions;
export const getNotificationState = (state) => state.notification;
export default notificationSlice.reducer;
