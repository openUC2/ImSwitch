// ./backendapi/apiLEDMatrixControllerSetAllLED.js
import createAxiosInstance from "./createAxiosInstance";

const apiLEDMatrixControllerSetAllLED = async ({ state, intensity, getReturn = false }) => {
  try {
    const axiosInstance = createAxiosInstance();
    // Example: /LEDMatrixController/setAllLED?state=1&intensity=255&getReturn=true
    const response = await axiosInstance.get(
      `/LEDMatrixController/setAllLED?state=${state}&intensity=${intensity}&getReturn=${getReturn}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting all LED illumination:", error);
    throw error;
  }
};

export default apiLEDMatrixControllerSetAllLED;
