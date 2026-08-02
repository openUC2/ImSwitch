// src/backendapi/apiStageCenterCalibrationGetRecommendedScanParameters.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * GET /StageCenterCalibrationController/getRecommendedScanParameters
 *
 * Returns adaptive scan defaults derived from the active detector's pixel
 * size and frame shape (similar to how the wellplate component picks dx/dy).
 */
const apiStageCenterCalibrationGetRecommendedScanParameters = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/StageCenterCalibrationController/getRecommendedScanParameters"
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching recommended scan parameters:", error);
    throw error;
  }
};

export default apiStageCenterCalibrationGetRecommendedScanParameters;
