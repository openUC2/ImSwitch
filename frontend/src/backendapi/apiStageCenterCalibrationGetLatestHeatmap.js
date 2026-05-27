// src/backendapi/apiStageCenterCalibrationGetLatestHeatmap.js
//
// GET /StageCenterCalibrationController/getLatestHeatmap
//
// Returns the most recent heatmap (samples + brightest + meta). The backend
// transparently restores it from disk so the frontend can show the previous
// scan after a page reload.
import createAxiosInstance from "./createAxiosInstance";

const apiStageCenterCalibrationGetLatestHeatmap = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/StageCenterCalibrationController/getLatestHeatmap"
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching latest heatmap:", error);
    return null;
  }
};

export default apiStageCenterCalibrationGetLatestHeatmap;
