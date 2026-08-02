// src/backendapi/apiStageCenterCalibrationGetHeatmap.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * GET /StageCenterCalibrationController/getCalibrationHeatmap
 *
 * Returns the raster scan samples plus the brightest spot detected so far.
 */
const apiStageCenterCalibrationGetHeatmap = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/StageCenterCalibrationController/getCalibrationHeatmap"
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching stage-center heatmap:", error);
    throw error;
  }
};

export default apiStageCenterCalibrationGetHeatmap;
