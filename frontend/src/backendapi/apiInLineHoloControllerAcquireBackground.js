// src/backendapi/apiInLineHoloControllerAcquireBackground.js
// Acquire a background image for live hologram normalization (divide)

import createAxiosInstance from "./createAxiosInstance";

/**
 * Acquire a background image used to normalize (divide) the live frame.
 * @param {Object} [opts]
 * @param {("median"|"snapshot")} [opts.mode="median"] - "median": temporal
 *   median over a burst (moving objects wash out); "snapshot": mean of a few
 *   frames (static sample moved out of FOV).
 * @param {number} [opts.numFrames=20] - Frames to grab for the burst.
 * @returns {Promise<Object>} { success, has_background, mode, num_frames, width, height, image }
 */
const apiInLineHoloControllerAcquireBackground = async ({
  mode = "median",
  numFrames = 20,
} = {}) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(
      `/InLineHoloController/acquire_background_inlineholo?mode=${mode}&num_frames=${numFrames}`
    );
    return response.data;
  } catch (error) {
    console.error("Error acquiring hologram background:", error);
    throw error;
  }
};

export default apiInLineHoloControllerAcquireBackground;
