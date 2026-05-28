// ./backendapi/apiLEDMatrixControllerSetRing.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the LED matrix to a ring pattern.
 *
 * Pass either:
 *   - `intensity` (single scalar 0..255), or
 *   - `intensity_r` + `intensity_g` + `intensity_b` (RGB triplet, each 0..255).
 *
 * The backend signature is `setRing(ringRadius, intensity, intensity_r?, intensity_g?, intensity_b?)`;
 * when RGB params are present the backend uses them instead of the scalar
 * intensity, so we forward whichever the caller provided.
 */
const apiLEDMatrixControllerSetRing = async ({
  ringRadius,
  intensity = 255,
  intensity_r,
  intensity_g,
  intensity_b,
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = new URLSearchParams({
      ringRadius: String(ringRadius),
      intensity: String(intensity),
    });
    if (intensity_r !== undefined) params.append("intensity_r", String(intensity_r));
    if (intensity_g !== undefined) params.append("intensity_g", String(intensity_g));
    if (intensity_b !== undefined) params.append("intensity_b", String(intensity_b));
    const response = await axiosInstance.get(
      `/LEDMatrixController/setRing?${params.toString()}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting ring illumination:", error);
    throw error;
  }
};

export default apiLEDMatrixControllerSetRing;
