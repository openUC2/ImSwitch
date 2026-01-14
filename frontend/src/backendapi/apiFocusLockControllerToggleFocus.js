import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerToggleFocus = async ({ toLock = null } = {}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = toLock !== null ? { toLock } : {};
    const response = await axiosInstance.get('/FocusLockController/toggleFocus', {
      params
    });
    return response.data;
  } catch (error) {
    console.error("Error toggling focus:", error);
    throw error;
  }
};

export default apiFocusLockControllerToggleFocus;