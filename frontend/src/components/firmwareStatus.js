// update_status values from UC2ConfigController.checkFirmwareUpdates -> label + colour
export const FIRMWARE_STATUS = {
  up_to_date: { label: "Up to date", color: "success.main" },
  update_available: { label: "Update available", color: "warning.main" },
  device_newer: { label: "Newer than server", color: "info.main" },
  unknown: { label: "Server has no version info", color: "text.secondary" },
  no_firmware: { label: "No image on server", color: "text.secondary" },
  unreachable: { label: "Not reachable", color: "text.secondary" },
};

// Same rule as FirmwareUpdateMixin._firmware_update_status (backend), for
// views that already hold both strings (CAN OTA wizard).
const FW_TIMESTAMP = /-t(\d{14})(?:-|$)/;
export const firmwareUpdateStatus = (installed, available) => {
  if (!available) return "unknown";
  if (installed === available) return "up_to_date";
  const tInstalled = FW_TIMESTAMP.exec(installed || "");
  const tAvailable = FW_TIMESTAMP.exec(available);
  if (tInstalled && tAvailable && tInstalled[1] > tAvailable[1]) return "device_newer";
  return "update_available";
};
