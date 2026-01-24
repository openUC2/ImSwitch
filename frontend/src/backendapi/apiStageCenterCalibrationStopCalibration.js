// src/backendapi/apiStageCenterCalibrationStopCalibration.js
import createAxiosInstance from "./createAxiosInstance";

const apiStageCenterCalibrationStopCalibration = async () => {
  try {
    const axiosInstance = createAxiosInstance();

    // Send GET request to stop calibration
    const response = await axiosInstance.get("/StageCenterCalibrationController/stopCalibration");

    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error stopping stage center calibration:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiStageCenterCalibrationStopCalibration;

/*
Example:
apiStageCenterCalibrationStopCalibration()
  .then(result => {
    console.log("Calibration stopped:", result);
  })
  .catch(error => {
    console.error("Error:", error);
  });
*/