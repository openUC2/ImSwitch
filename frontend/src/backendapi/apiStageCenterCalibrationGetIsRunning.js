// src/backendapi/apiStageCenterCalibrationGetIsRunning.js
import createAxiosInstance from "./createAxiosInstance";

const apiStageCenterCalibrationGetIsRunning = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/StageCenterCalibrationController/getIsCalibrationRunning"
    );
    return response.data;
  } catch (error) {
    console.error("Error checking stage-center calibration state:", error);
    throw error;
  }
};

export default apiStageCenterCalibrationGetIsRunning;
