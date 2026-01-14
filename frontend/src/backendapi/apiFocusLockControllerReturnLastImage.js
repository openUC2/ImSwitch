import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerReturnLastImage = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/returnLastImage', {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    console.error("Error getting last image:", error);
    throw error;
  }
};

export default apiFocusLockControllerReturnLastImage;