// src/backendapi/apiStageCenterCalibrationPerformCalibration.js
import createAxiosInstance from "./createAxiosInstance";

const apiStageCenterCalibrationPerformCalibration = async ({
  start_x = 0,
  start_y = 0,
  exposure_time_us = 3000,
  speed = 5000,
  step_um = 50.0,
  max_radius_um = 2000.0,
  brightness_factor = 1.4,
}) => {
  try {
    const axiosInstance = createAxiosInstance();

    // Build the query string dynamically
    let url = "/StageCenterCalibrationController/performCalibration?";
    const queryParams = [];

    queryParams.push(`start_x=${encodeURIComponent(start_x)}`);
    queryParams.push(`start_y=${encodeURIComponent(start_y)}`);
    queryParams.push(`exposure_time_us=${encodeURIComponent(exposure_time_us)}`);
    queryParams.push(`speed=${encodeURIComponent(speed)}`);
    queryParams.push(`step_um=${encodeURIComponent(step_um)}`);
    queryParams.push(`max_radius_um=${encodeURIComponent(max_radius_um)}`);
    queryParams.push(`brightness_factor=${encodeURIComponent(brightness_factor)}`);

    // Join all query parameters with '&'
    url += queryParams.join("&");

    // Send GET request with the constructed URL
    const response = await axiosInstance.get(url);

    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error performing stage center calibration:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiStageCenterCalibrationPerformCalibration;

/*
Example:
apiStageCenterCalibrationPerformCalibration({
  start_x: 0,
  start_y: 0,
  exposure_time_us: 3000,
  speed: 5000,
  step_um: 50.0,
  max_radius_um: 2000.0,
  brightness_factor: 1.4
})
  .then(positions => {
    console.log("Calibration positions:", positions);
  })
  .catch(error => {
    console.error("Error:", error);
  });
*/