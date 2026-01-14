// ./backendapi/apiLEDMatrixControllerSetCircle.js
import createAxiosInstance from "./createAxiosInstance";

const apiLEDMatrixControllerSetCircle = async ({ circleRadius, intensity }) => {
  try {
    const axiosInstance = createAxiosInstance();
    // Example: /LEDMatrixController/setCircle?circleRadius=1&intensity=255
    const response = await axiosInstance.get(
      `/LEDMatrixController/setCircle?circleRadius=${circleRadius}&intensity=${intensity}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting circle illumination:", error);
    throw error;
  }
};

export default apiLEDMatrixControllerSetCircle;
