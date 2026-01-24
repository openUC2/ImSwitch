// ./backendapi/apiLEDMatrixControllerSetRing.js
import createAxiosInstance from "./createAxiosInstance";

const apiLEDMatrixControllerSetRing = async ({ ringRadius, intensity }) => {
  try {
    const axiosInstance = createAxiosInstance();
    // Example: /LEDMatrixController/setRing?ringRadius=3&intensity=255
    const response = await axiosInstance.get(
      `/LEDMatrixController/setRing?ringRadius=${ringRadius}&intensity=${intensity}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting ring illumination:", error);
    throw error;
  }
};

export default apiLEDMatrixControllerSetRing;
