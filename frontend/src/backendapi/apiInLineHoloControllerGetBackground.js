// src/backendapi/apiInLineHoloControllerGetBackground.js
// Fetch the stored hologram background (base64 preview) from the backend

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get the stored background as a base64 PNG data URL (or null). Used to restore
 * the preview after a page reload.
 * @returns {Promise<Object>} { has_background, bg_enabled, image, ...meta }
 */
const apiInLineHoloControllerGetBackground = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(
      "/InLineHoloController/get_background_inlineholo"
    );
    return response.data;
  } catch (error) {
    console.error("Error getting hologram background:", error);
    throw error;
  }
};

export default apiInLineHoloControllerGetBackground;
