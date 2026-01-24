// src/backendapi/apiStageCenterCalibrationGetStatus.js
import createAxiosInstance from "./createAxiosInstance";

const apiStageCenterCalibrationGetStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance();

    // Send GET request to check calibration status
    const response = await axiosInstance.get("/StageCenterCalibrationController/getIsCalibrationRunning");

    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error getting stage center calibration status:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiStageCenterCalibrationGetStatus;

/*
Example:
apiStageCenterCalibrationGetStatus()
  .then(isRunning => {
    console.log("Calibration running:", isRunning);
  })
  .catch(error => {
    console.error("Error:", error);
  });
*/