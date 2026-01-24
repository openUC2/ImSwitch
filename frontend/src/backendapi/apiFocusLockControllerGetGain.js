import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerGetGain = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/getGain');
    return response.data;
  } catch (error) {
    console.error("Error getting gain:", error);
    throw error;
  }
};

export default apiFocusLockControllerGetGain;
