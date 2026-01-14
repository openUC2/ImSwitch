import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerSetExposureTime = async (exposureTime) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/setExposureTime', {
      params: { exposure_time: exposureTime }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting exposure time:", error);
    throw error;
  }
};

export default apiFocusLockControllerSetExposureTime;