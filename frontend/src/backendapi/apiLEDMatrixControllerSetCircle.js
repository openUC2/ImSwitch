// ./backendapi/apiLEDMatrixControllerSetCircle.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the LED matrix to a filled-circle pattern.
 *
 * Pass either `intensity` (scalar 0..255) or `intensity_r/g/b` (RGB triplet,
 * each 0..255). The backend signature is
 * `setCircle(circleRadius, intensity, intensity_r?, intensity_g?, intensity_b?)`
 * and uses the RGB triplet instead of the scalar when it is supplied.
 */
const apiLEDMatrixControllerSetCircle = async ({
  circleRadius,
  intensity = 255,
  intensity_r,
  intensity_g,
  intensity_b,
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = new URLSearchParams({
      circleRadius: String(circleRadius),
      intensity: String(intensity),
    });
    if (intensity_r !== undefined) params.append("intensity_r", String(intensity_r));
    if (intensity_g !== undefined) params.append("intensity_g", String(intensity_g));
    if (intensity_b !== undefined) params.append("intensity_b", String(intensity_b));
    const response = await axiosInstance.get(
      `/LEDMatrixController/setCircle?${params.toString()}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting circle illumination:", error);
    throw error;
  }
};

export default apiLEDMatrixControllerSetCircle;
