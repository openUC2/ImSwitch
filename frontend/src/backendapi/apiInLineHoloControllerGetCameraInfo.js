// src/backendapi/apiInLineHoloControllerGetCameraInfo.js
// Get info about the detector backing the inline-holo controller

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get the camera (detector) name + RGB flag bound to the inline holo controller.
 * @returns {Promise<{camera: string, is_rgb: boolean}>}
 */
const apiInLineHoloControllerGetCameraInfo = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(
      "/InLineHoloController/get_camera_info_inlineholo"
    );
    return response.data;
  } catch (error) {
    console.error("Error getting inline holo camera info:", error);
    throw error;
  }
};

export default apiInLineHoloControllerGetCameraInfo;
