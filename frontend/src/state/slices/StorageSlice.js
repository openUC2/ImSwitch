import { createSlice } from "@reduxjs/toolkit";

const toBytesFromGigabytes = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }

  return Math.round(value * 1024 ** 3);
};

const normalizeDrive = (drive = {}) => {
  const path = drive.path || drive.mount_point || "";

  return {
    ...drive,
    path,
    mount_point: drive.mount_point || path,
    is_active: Boolean(drive.is_active),
  };
};

export const normalizeStorageSnapshot = (snapshot = {}) => {
  const activePath = snapshot.active_path || snapshot.active_data_path || "";
  const freeBytes =
    snapshot.disk_usage?.free ?? toBytesFromGigabytes(snapshot.free_space_gb);
  const totalBytes =
    snapshot.disk_usage?.total ?? toBytesFromGigabytes(snapshot.total_space_gb);
  const usedBytes =
    snapshot.disk_usage?.used ??
    (typeof freeBytes === "number" && typeof totalBytes === "number"
      ? Math.max(totalBytes - freeBytes, 0)
      : null);
  const availableDrives = (
    snapshot.available_drives ||
    snapshot.drives ||
    []
  ).map(normalizeDrive);

  return {
    ...snapshot,
    active_path: activePath,
    active_data_path: snapshot.active_data_path || activePath,
    disk_usage: {
      ...snapshot.disk_usage,
      free: freeBytes,
      total: totalBytes,
      used: usedBytes,
      percent_used:
        snapshot.disk_usage?.percent_used ?? snapshot.percent_used ?? null,
    },
    available_drives: availableDrives,
    drives: availableDrives,
    updated_at: snapshot.updated_at || null,
  };
};

const initialState = {
  status: normalizeStorageSnapshot(),
  hasReceivedSnapshot: false,
};

const storageSlice = createSlice({
  name: "storageState",
  initialState,
  reducers: {
    setStorageSnapshot: (state, action) => {
      state.status = normalizeStorageSnapshot(action.payload);
      state.hasReceivedSnapshot = true;
    },
    resetStorageState: () => initialState,
  },
});

export const { setStorageSnapshot, resetStorageState } = storageSlice.actions;
export const getStorageState = (state) => state.storageState;
export default storageSlice.reducer;
