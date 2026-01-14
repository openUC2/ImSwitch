import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerReturnLastCroppedImage = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get('/FocusLockController/returnLastCroppedImage', {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    console.error("Error getting last cropped image:", error);
    throw error;
  }
};

export default apiFocusLockControllerReturnLastCroppedImage;