import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerGetPIParameters = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/getPIParameters');
    return response.data;
  } catch (error) {
    console.error("Error getting PI parameters:", error);
    throw error;
  }
};

export default apiFocusLockControllerGetPIParameters;