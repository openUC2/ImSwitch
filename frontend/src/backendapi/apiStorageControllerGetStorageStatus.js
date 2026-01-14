import createAxiosInstance from "./createAxiosInstance";

/**
 * Get current storage status including active path and available drives
 *
 * @returns {Promise<Object>} Response with storage status information
 */
export default async function apiStorageControllerGetStorageStatus() {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/StorageController/get_storage_status"
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching storage status:", error);
    throw error;
  }
}
