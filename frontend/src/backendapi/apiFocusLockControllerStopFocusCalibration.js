// src/backendapi/apiFocusLockControllerStopFocusCalibration.js
// API to stop an ongoing focus calibration in the FocusLock controller
import createAxiosInstance from "./createAxiosInstance";

/**
 * Stop an ongoing focus calibration process.
 * 
 * @returns {Promise<Object>} - Response from the backend
 */
const apiFocusLockControllerStopFocusCalibration = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/FocusLockController/stopFocusCalibration");
    return response.data;
  } catch (error) {
    console.error("Error stopping focus calibration:", error);
    throw error;
  }
};

export default apiFocusLockControllerStopFocusCalibration;
