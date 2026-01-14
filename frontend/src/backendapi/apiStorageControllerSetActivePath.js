import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the active storage path (e.g., mount a USB drive)
 * Creates an 'ImSwitchData' subfolder within the selected path
 *
 * @param {string} path - Base path to set as active storage location
 * @param {boolean} persist - Whether to persist this setting to configuration (default: false)
 * @returns {Promise<Object>} Response with success status and active path
 */
export default async function apiStorageControllerSetActivePath(
  path,
  persist = false
) {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.post(
      `/StorageController/set_active_path?path=${encodeURIComponent(
        path
      )}&persist=${persist}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting active path:", error);
    throw error;
  }
}
