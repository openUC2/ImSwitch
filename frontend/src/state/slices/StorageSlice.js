import { createSlice } from "@reduxjs/toolkit";

const toBytesFromGigabytes = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }

  return Math.round(value * 1024 ** 3);
};

const hasNumber = (value) => typeof value === "number" && !Number.isNaN(value);

const normalizeUsage = (usage = {}) => {
  const freeBytes = hasNumber(usage.free)
    ? usage.free
    : toBytesFromGigabytes(usage.free_gb ?? usage.free_space_gb);
  const totalBytes = hasNumber(usage.total)
    ? usage.total
    : toBytesFromGigabytes(usage.total_gb ?? usage.total_space_gb);
  const usedBytes = hasNumber(usage.used)
    ? usage.used
    : typeof freeBytes === "number" && typeof totalBytes === "number"
      ? Math.max(totalBytes - freeBytes, 0)
      : null;
  const percentUsed = hasNumber(usage.percent_used)
    ? usage.percent_used
    : hasNumber(usage.percent)
      ? usage.percent
      : typeof usedBytes === "number" &&
          typeof totalBytes === "number" &&
          totalBytes > 0
        ? Math.round((usedBytes / totalBytes) * 10000) / 100
        : null;

  if (
    freeBytes === null &&
    totalBytes === null &&
    usedBytes === null &&
    percentUsed === null
  ) {
    return null;
  }

  return {
    ...usage,
    free: freeBytes,
    total: totalBytes,
    used: usedBytes,
    percent_used: percentUsed,
  };
};

const normalizeDevice = (device = {}, activePath = "", activeDataPath = "") => {
  const path = device.path || device.mount_point || "";
  const label =
    device.label ||
    device.name ||
    path.split("/").filter(Boolean).pop() ||
    "Storage";
  const isAvailable = device.exists !== false;

  return {
    ...device,
    path,
    mount_point: device.mount_point || path,
    label,
    is_available: isAvailable,
    is_internal: Boolean(device.is_internal),
    is_default: Boolean(device.is_default),
    is_fallback: Boolean(device.is_fallback),
    is_active: Boolean(
      isAvailable &&
        (device.is_active ||
          (activeDataPath && path && activeDataPath.startsWith(path)) ||
          (activePath && path && activePath.startsWith(path))),
    ),
    usage: normalizeUsage(device.usage || device.disk_usage || device),
  };
};

export const normalizeStorageSnapshot = (snapshot = {}) => {
  const rawActivePath = snapshot.active_path || snapshot.active_data_path || "";
  const rawActiveDataPath = snapshot.active_data_path || rawActivePath;

  const normalizedDevices = (snapshot.storage_devices || snapshot.devices || []).map(
    (device) => normalizeDevice(device, rawActivePath, rawActiveDataPath),
  );

  const defaultAvailableDevice =
    normalizedDevices.find((device) => device.is_default && device.is_available) ||
    normalizedDevices.find((device) => device.is_internal && device.is_available) ||
    normalizedDevices.find((device) => device.is_available) ||
    normalizedDevices[0] ||
    null;

  const activeDeviceFromSnapshot =
    normalizedDevices.find((device) => device.is_active && device.is_available) ||
    null;

  const effectiveActiveDevice = activeDeviceFromSnapshot || defaultAvailableDevice;
  const effectiveActivePath = effectiveActiveDevice?.path || rawActivePath;

  const hasValidActivePath = normalizedDevices.some(
    (device) =>
      device.is_available &&
      device.path &&
      rawActiveDataPath &&
      rawActiveDataPath.startsWith(device.path),
  );

  const storageDevices = normalizedDevices.map((device) => ({
    ...device,
    is_active: Boolean(
      effectiveActiveDevice &&
        device.path &&
        effectiveActivePath &&
        effectiveActivePath.startsWith(device.path),
    ),
  }));

  const defaultDevice =
    storageDevices.find((device) => device.is_default) ||
    storageDevices.find((device) => device.is_internal) ||
    storageDevices[0] ||
    null;

  const activeDevice =
    storageDevices.find((device) => device.is_active) || effectiveActiveDevice || null;

  const internalDevice =
    storageDevices.find((device) => device.is_internal) || null;

  const externalDevices = storageDevices.filter(
    (device) => !device.is_internal && device.is_available,
  );

  const activePath = hasValidActivePath ? rawActivePath : effectiveActivePath;
  const activeDataPath = hasValidActivePath ? rawActiveDataPath : activePath;

  return {
    ...snapshot,
    active_path: activePath,
    active_data_path: activeDataPath,
    active_device_path: activeDevice?.path || snapshot.active_device_path || null,
    default_device_path: snapshot.default_device_path || defaultDevice?.path || null,
    fallback_path: snapshot.fallback_path || defaultDevice?.path || null,
    storage_devices: storageDevices,
    devices: storageDevices,
    active_device: activeDevice,
    default_device: defaultDevice,
    internal_device: internalDevice,
    external_devices: externalDevices,
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
