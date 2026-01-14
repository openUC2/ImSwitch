import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerUnlockFocus = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/unlockFocus');
    return response.data;
  } catch (error) {
    console.error("Error unlocking focus:", error);
    throw error;
  }
};

export default apiFocusLockControllerUnlockFocus;