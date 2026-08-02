// ./backendapi/apiLEDMatrixControllerSetAllLED.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Switch every LED of the matrix on/off.
 *
 * Pass either `intensity` (scalar 0..255) or `intensity_r/g/b` (RGB triplet,
 * each 0..255). The backend signature is
 * `setAllLED(state, intensity, intensity_r?, intensity_g?, intensity_b?)`
 * and uses the RGB triplet instead of the scalar when it is supplied.
 */
const apiLEDMatrixControllerSetAllLED = async ({
  state,
  intensity = 255,
  intensity_r,
  intensity_g,
  intensity_b,
  getReturn = false,
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = new URLSearchParams({
      state: String(state),
      intensity: String(intensity),
      getReturn: String(getReturn),
    });
    if (intensity_r !== undefined) params.append("intensity_r", String(intensity_r));
    if (intensity_g !== undefined) params.append("intensity_g", String(intensity_g));
    if (intensity_b !== undefined) params.append("intensity_b", String(intensity_b));
    const response = await axiosInstance.get(
      `/LEDMatrixController/setAllLED?${params.toString()}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting all LED illumination:", error);
    throw error;
  }
};

export default apiLEDMatrixControllerSetAllLED;
