import createAxiosInstance from "./createAxiosInstance.js";

const apiFocusLockControllerSetParamsAstigmatism = async ({ gaussian_sigma, background_threshold, crop_size, crop_center }) => {
  try {
    const axiosInstance = createAxiosInstance();
    const config = {
      params: { gaussian_sigma, background_threshold, crop_size }
    };
    
    // Add cropCenter as request body if provided
    if (crop_center) {
      config.data = crop_center;
      config.headers = {
        'Content-Type': 'application/json'
      };
    }
    
    const response = await axiosInstance.get('/FocusLockController/setParamsAstigmatism', config);
    return response.data;
  } catch (error) {
    console.error("Error setting astigmatism parameters:", error);
    throw error;
  }
};

export default apiFocusLockControllerSetParamsAstigmatism;