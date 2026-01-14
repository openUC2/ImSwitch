import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerSetGain = async (gain) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/setGain', {
      params: { gain }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting gain:", error);
    throw error;
  }
};

export default apiFocusLockControllerSetGain;
