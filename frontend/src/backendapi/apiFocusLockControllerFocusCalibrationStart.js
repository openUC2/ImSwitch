import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerFocusCalibrationStart = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/focusCalibrationStart');
    return response.data;
  } catch (error) {
    console.error("Error starting focus calibration:", error);
    throw error;
  }
};

export default apiFocusLockControllerFocusCalibrationStart;