import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerGetCalibrationStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/getCalibrationStatus');
    return response.data;
  } catch (error) {
    console.error("Error getting calibration status:", error);
    throw error;
  }
};

export default apiFocusLockControllerGetCalibrationStatus;
