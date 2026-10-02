// src/state/slices/ArkitektSlice.js
// Redux slice for the Arkitekt panel: the connection status (which also
// carries the device code during a login), the log of remote calls and the
// previews of images sent to the server. Filled by polling getArkitektStatus
// and by the sigArkitektStatus / sigArkitektActivity / sigArkitektUpload
// socket signals.

import { createSlice } from "@reduxjs/toolkit";

const MAX_ACTIVITY = 50;
const MAX_UPLOADS = 12;

const initialState = {
  status: null, // getArkitektStatus(); null until the first answer
  activity: [], // newest first: [{id, action, arguments, status, startedAt, durationS, results, error}]
  uploads: [], // newest first: [{id, name, datasetId, shape, positionUm, thumbnail, ...}]
  pending: null, // the request in flight: "bind" | "cancel" | "unbind" | "settings"
  requestError: null, // a request that failed before reaching the backend
};

const arkitektSlice = createSlice({
  name: "arkitekt",
  initialState,
  reducers: {
    // The socket signal carries the manager's status only; keep the actions
    // and the activity the last full status brought.
    setStatus: (state, action) => {
      const { activity, ...status } = action.payload || {};
      state.status = { ...(state.status || {}), ...status };
      if (Array.isArray(activity)) {
        state.activity = activity.slice(0, MAX_ACTIVITY);
      }
    },
    upsertActivity: (state, action) => {
      const entry = action.payload;
      const index = state.activity.findIndex((e) => e.id === entry.id);
      if (index >= 0) {
        state.activity[index] = entry;
      } else {
        state.activity.unshift(entry);
        state.activity = state.activity.slice(0, MAX_ACTIVITY);
      }
    },
    clearActivity: (state) => {
      state.activity = [];
      state.uploads = [];
    },
    setUploads: (state, action) => {
      state.uploads = (action.payload || []).slice(0, MAX_UPLOADS);
    },
    addUpload: (state, action) => {
      state.uploads = [action.payload, ...state.uploads.filter((u) => u.id !== action.payload.id)]
        .slice(0, MAX_UPLOADS);
    },
    setPending: (state, action) => {
      state.pending = action.payload;
    },
    setRequestError: (state, action) => {
      state.requestError = action.payload;
    },
  },
});

export const {
  setStatus,
  upsertActivity,
  clearActivity,
  setUploads,
  addUpload,
  setPending,
  setRequestError,
} = arkitektSlice.actions;

export const getArkitektState = (state) => state.arkitektState;

export default arkitektSlice.reducer;
