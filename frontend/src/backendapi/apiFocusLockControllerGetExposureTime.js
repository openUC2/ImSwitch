import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerGetExposureTime = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/getExposureTime');
    return response.data;
  } catch (error) {
    console.error("Error getting exposure time:", error);
    throw error;
  }
};

export default apiFocusLockControllerGetExposureTime;
