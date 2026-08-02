// src/backendapi/apiInLineHoloControllerSetBgEnabled.js
// Toggle live background division for inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Enable/disable dividing the live frame by the stored background.
 * @param {boolean} enabled - Whether to divide out the background live.
 * @returns {Promise<Object>} { success, bg_enabled }
 */
const apiInLineHoloControllerSetBgEnabled = async (enabled) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(
      `/InLineHoloController/set_bg_enabled_inlineholo?enabled=${enabled}`
    );
    return response.data;
  } catch (error) {
    console.error("Error setting background enabled:", error);
    throw error;
  }
};

export default apiInLineHoloControllerSetBgEnabled;
