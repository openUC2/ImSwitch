import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerGetParamsAstigmatism = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/getParamsAstigmatism');
    return response.data;
  } catch (error) {
    console.error("Error getting astigmatism parameters:", error);
    throw error;
  }
};

export default apiFocusLockControllerGetParamsAstigmatism;