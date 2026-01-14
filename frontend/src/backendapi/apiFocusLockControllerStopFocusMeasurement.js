import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerStopFocusMeasurement = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/stopFocusMeasurement');
    return response.data;
  } catch (error) {
    console.error("Error stopping focus measurement:", error);
    throw error;
  }
};

export default apiFocusLockControllerStopFocusMeasurement;