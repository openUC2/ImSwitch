// ./backendapi/apiLEDMatrixControllerSetHalves.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the LED matrix to a "halves" pattern (one quadrant of the matrix lit).
 *
 * `direction` is one of "top" | "bottom" | "left" | "right".
 * Pass either `intensity` (scalar) or `intensity_r/g/b` (RGB triplet).
 */
const apiLEDMatrixControllerSetHalves = async ({
  intensity = 255,
  direction,
  intensity_r,
  intensity_g,
  intensity_b,
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = new URLSearchParams({
      intensity: String(intensity),
      direction: String(direction),
    });
    if (intensity_r !== undefined) params.append("intensity_r", String(intensity_r));
    if (intensity_g !== undefined) params.append("intensity_g", String(intensity_g));
    if (intensity_b !== undefined) params.append("intensity_b", String(intensity_b));
    const response = await axiosInstance.get(
      `/LEDMatrixController/setHalves?${params.toString()}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting halves illumination:", error);
    throw error;
  }
};

export default apiLEDMatrixControllerSetHalves;
