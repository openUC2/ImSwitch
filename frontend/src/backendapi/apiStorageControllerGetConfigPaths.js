import createAxiosInstance from "./createAxiosInstance";

/**
 * Get all configuration-related paths
 *
 * @returns {Promise<Object>} Response with configuration paths:
 *   - config_path: Where ImSwitch configuration files are stored
 *   - data_path: Configured fallback data path from config file
 *   - active_data_path: Currently active runtime data path
 */
export default async function apiStorageControllerGetConfigPaths() {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/StorageController/get_config_paths"
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching config paths:", error);
    throw error;
  }
}
