import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerEnableFocusLock = async ({ enable = true } = {}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/enableFocusLock', {
      params: { enable }
    });
    return response.data;
  } catch (error) {
    console.error("Error enabling/disabling focus lock:", error);
    throw error;
  }
};

export default apiFocusLockControllerEnableFocusLock;