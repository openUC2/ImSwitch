// src/backendapi/apiLightsheetController.js
// API functions for Lightsheet Controller with Go-Stop-Acquire mode and OME-Zarr support
// Following Copilot Instructions for API communication patterns

import createAxiosInstance from './createAxiosInstance';


/**
 * Start a Go-Stop-Acquire scan with configurable storage format, optional tiling, and timelapse.
 * This mode moves the stage in discrete steps, stops, acquires an image,
 * then moves to the next position. Suitable for high-quality Z-stacks.
 * 
 * @param {Object} params - Scan parameters
 * @param {number} params.minPos - Start position in µm
 * @param {number} params.maxPos - End position in µm
 * @param {number} params.stepSize - Step size between acquisitions in µm
 * @param {string} params.axis - Scan axis (A, X, Y, Z)
 * @param {string} params.illuSource - Illumination source name
 * @param {number} params.illuValue - Illumination intensity
 * @param {string} params.storageFormat - Storage format (tiff, ome_zarr, both)
 * @param {string} params.experimentName - Name for the experiment/files
 * @param {boolean} params.enableTiling - Enable XY tiling
 * @param {number} params.tilesXPositive - Number of tiles in positive X direction
 * @param {number} params.tilesXNegative - Number of tiles in negative X direction
 * @param {number} params.tilesYPositive - Number of tiles in positive Y direction
 * @param {number} params.tilesYNegative - Number of tiles in negative Y direction
 * @param {number} params.tileStepSizeX - Step size in X for tiling (µm)
 * @param {number} params.tileStepSizeY - Step size in Y for tiling (µm)
 * @param {number} params.tileOverlap - Overlap fraction between tiles (0-1)
 * @param {number} params.timepoints - Number of timepoints to acquire
 * @param {number} params.timeLapsePeriod - Period between timepoints (seconds)
 * @returns {Promise<Object>} Status information including file paths
 */
export const apiStartStepAcquireScan = async ({
  minPos = -500,
  maxPos = 500,
  stepSize = 10,
  axis = "A",
  illuSource = "",
  illuValue = 512,
  storageFormat = "ome_zarr",
  experimentName = "lightsheet_scan",
  enableTiling = false,
  tilesXPositive = 0,
  tilesXNegative = 0,
  tilesYPositive = 0,
  tilesYNegative = 0,
  tileStepSizeX = 0,
  tileStepSizeY = 0,
  tileOverlap = 0.1,
  timepoints = 1,
  timeLapsePeriod = 60
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/startStepAcquireScan", {
      params: {
        minPos,
        maxPos,
        stepSize,
        axis,
        illuSource,
        illuValue,
        storageFormat,
        experimentName,
        enableTiling,
        tilesXPositive,
        tilesXNegative,
        tilesYPositive,
        tilesYNegative,
        tileStepSizeX,
        tileStepSizeY,
        tileOverlap,
        timepoints,
        timeLapsePeriod
      }
    });
    return response.data;
  } catch (error) {
    console.error("Failed to start step-acquire scan:", error);
    throw error;
  }
};

/**
 * Start a continuous scan with optional OME-Zarr storage.
 * Enhanced version of original continuous scan with OME-Zarr support.
 * 
 * @param {Object} params - Scan parameters
 * @param {number} params.minPos - Start position in µm
 * @param {number} params.maxPos - End position in µm
 * @param {number} params.speed - Stage movement speed
 * @param {string} params.axis - Scan axis (A, X, Y, Z)
 * @param {string} params.illuSource - Illumination source name
 * @param {number} params.illuValue - Illumination intensity
 * @param {string} params.storageFormat - Storage format (tiff, ome_zarr, both)
 * @param {string} params.experimentName - Name for the experiment/files
 * @returns {Promise<Object>} Status information
 */
export const apiStartContinuousScanWithZarr = async ({
  minPos = -500,
  maxPos = 500,
  speed = 1000,
  axis = "A",
  illuSource = "",
  illuValue = 512,
  storageFormat = "ome_zarr",
  experimentName = "lightsheet_continuous"
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/startContinuousScanWithZarr", {
      params: {
        minPos,
        maxPos,
        speed,
        axis,
        illuSource,
        illuValue,
        storageFormat,
        experimentName
      }
    });
    return response.data;
  } catch (error) {
    console.error("Failed to start continuous scan with Zarr:", error);
    throw error;
  }
};

/**
 * Get current scan status including progress and file paths.
 * @returns {Promise<Object>} Scan status object
 */
export const apiGetScanStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/getScanStatus");
    return response.data;
  } catch (error) {
    console.error("Failed to get scan status:", error);
    throw error;
  }
};

/**
 * Get available scan modes from backend.
 * @returns {Promise<string[]>} Array of available scan modes
 */
export const apiGetAvailableScanModes = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/getAvailableScanModes");
    return response.data;
  } catch (error) {
    console.error("Failed to get available scan modes:", error);
    return ["continuous", "step_acquire"]; // Fallback defaults
  }
};

/**
 * Get available storage formats from backend.
 * @returns {Promise<string[]>} Array of available storage formats
 */
export const apiGetAvailableStorageFormats = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/getAvailableStorageFormats");
    return response.data;
  } catch (error) {
    console.error("Failed to get available storage formats:", error);
    return ["tiff"]; // Fallback to TIFF only
  }
};

/**
 * Get the path to the latest OME-Zarr store for visualization.
 * @returns {Promise<Object>} Object with zarrPath and absolutePath
 */
export const apiGetLatestZarrPath = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/getLatestZarrPath");
    return response.data;
  } catch (error) {
    console.error("Failed to get latest Zarr path:", error);
    return { zarrPath: null, exists: false };
  }
};
/**
 * Get the field of view (FOV) of the current objective.
 * Used as a hint for tiling step size calculations.
 * @returns {Promise<Object>} FOV information including width, height, pixelSize, and suggested overlap
 */
export const apiGetObjectiveFOV = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/getObjectiveFOV");
    return response.data;
  } catch (error) {
    console.error("Failed to get objective FOV:", error);
    return {
      fovX: 1000,
      fovY: 1000,
      pixelSize: 1.0,
      suggestedOverlap: 0.1,
      success: false
    };
  }
};
/**
 * Check if lightsheet scan is currently running (legacy API).
 * @returns {Promise<boolean>} Whether scan is running
 */
export const apiGetIsLightsheetRunning = async () => {
  try {
    const axiosInstance = createAxiosInstance();    
    const response = await axiosInstance.get("/LightsheetController/getIsLightsheetRunning");
    return response.data;
  } catch (error) {
    console.error("Failed to check lightsheet status:", error);
    return false;
  }
};

/**
 * Start legacy continuous scan (for backward compatibility).
 * @param {Object} params - Scan parameters
 * @returns {Promise<Object>} Status information
 */
export const apiPerformScanningRecording = async ({
  minPos = 0,
  maxPos = 1000,
  speed = 1000,
  axis = "A",
  illusource = -1,
  illuvalue = 512
}) => {
  try {
    const axiosInstance = createAxiosInstance();    
    const response = await axiosInstance.get("/LightsheetController/performScanningRecording", {
      params: { minPos, maxPos, speed, axis, illusource, illuvalue }
    });
    return response.data;
  } catch (error) {
    console.error("Failed to start scanning recording:", error);
    throw error;
  }
};

/**
 * Set galvo scanner parameters.
 * @param {Object} params - Galvo parameters
 * @returns {Promise<Object>} Response data
 */
export const apiSetGalvo = async ({
  channel = 1,
  frequency = 10,
  offset = 0,
  amplitude = 1,
  clk_div = 0,
  phase = 0,
  invert = 1
}) => {
  try {
    const axiosInstance = createAxiosInstance();    
    const response = await axiosInstance.get("/LightsheetController/setGalvo", {
      params: { channel, frequency, offset, amplitude, clk_div, phase, invert }
    });
    return response.data;
  } catch (error) {
    console.error("Failed to set galvo parameters:", error);
    throw error;
  }
};

export default {
  apiStartStepAcquireScan,
  apiStartContinuousScanWithZarr,
  apiGetScanStatus,
  apiGetAvailableScanModes,
  apiGetAvailableStorageFormats,
  apiGetLatestZarrPath,
  apiGetIsLightsheetRunning,
  apiPerformScanningRecording,
  apiSetGalvo
};
