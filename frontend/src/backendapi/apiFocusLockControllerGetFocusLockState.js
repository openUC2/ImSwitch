import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerGetFocusLockState = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/getFocusLockState');
    return response.data;
  } catch (error) {
    console.error("Error getting focus lock state:", error);
    throw error;
  }
};

export default apiFocusLockControllerGetFocusLockState;