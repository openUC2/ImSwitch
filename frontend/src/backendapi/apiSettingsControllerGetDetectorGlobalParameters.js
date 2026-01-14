// src/backendapi/apiSettingsControllerGetDetectorGlobalParameters.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Get global detector parameters including compression settings
 * GET /SettingsController/getDetectorGlobalParameters
 * 
 * Returns object with:
 * - compressionlevel: number (e.g., 80 for JPEG quality)
 * - stream_compression_algorithm: string (e.g., "jpeg", "lz4", "zstd")
 */
const apiSettingsControllerGetDetectorGlobalParameters = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const url = `/SettingsController/getDetectorGlobalParameters`;
    const response = await axiosInstance.get(url);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to get detector global parameters: ${error.message}`);
  }
};

export default apiSettingsControllerGetDetectorGlobalParameters;

/**
 * Example usage:
 * const params = await apiSettingsControllerGetDetectorGlobalParameters();
 * console.log(params);
 * // { compressionlevel: 80, stream_compression_algorithm: "lz4" }
 * 
 * // Determine stream type from algorithm
 * const isBinary = params.stream_compression_algorithm !== "jpeg";
 */
