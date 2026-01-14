// ./backendapi/apiLEDMatrixControllerSetHalves.js
import createAxiosInstance from "./createAxiosInstance";

const apiLEDMatrixControllerSetHalves = async ({ intensity, direction }) => {
  try {
    const axiosInstance = createAxiosInstance();
    // Example: /LEDMatrixController/setHalves?intensity=255&direction=top
    const response = await axiosInstance.get(
      `/LEDMatrixController/setHalves?intensity=${intensity}&direction=${direction}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting halves illumination:", error);
    throw error;
  }
};

export default apiLEDMatrixControllerSetHalves;
