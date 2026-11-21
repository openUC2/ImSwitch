import createAxiosInstance from "./createAxiosInstance";

/**
 * Rotate the status OLED picture by 180 degrees (stored on the ESP32).
 *
 * @param {boolean} flip - True to rotate, False for the default orientation.
 * @returns {Promise<Object>} { status, flip }
 */
const apiUC2ConfigControllerSetDisplayFlip = async (flip = true) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/setDisplayFlip", {
    params: { flip },
  });
  return response.data;
};

export default apiUC2ConfigControllerSetDisplayFlip;
