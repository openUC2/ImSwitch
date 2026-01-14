import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerStartFocusMeasurement = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/startFocusMeasurement');
    return response.data;
  } catch (error) {
    console.error("Error starting focus measurement:", error);
    throw error;
  }
};

export default apiFocusLockControllerStartFocusMeasurement;