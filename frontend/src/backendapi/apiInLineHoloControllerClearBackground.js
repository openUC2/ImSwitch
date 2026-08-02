// src/backendapi/apiInLineHoloControllerClearBackground.js
// Clear the stored hologram background and disable live division

import createAxiosInstance from "./createAxiosInstance";

/**
 * Drop the stored background and disable live division.
 * @returns {Promise<Object>} { success, has_background, bg_enabled }
 */
const apiInLineHoloControllerClearBackground = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(
      "/InLineHoloController/clear_background_inlineholo"
    );
    return response.data;
  } catch (error) {
    console.error("Error clearing hologram background:", error);
    throw error;
  }
};

export default apiInLineHoloControllerClearBackground;
