import createAxiosInstance from "./createAxiosInstance";

/**
 * List all detected external storage drives
 *
 * @returns {Promise<Object>} Response with list of external drives
 */
export default async function apiStorageControllerListExternalDrives() {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/StorageController/list_external_drives"
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching external drives:", error);
    throw error;
  }
}
