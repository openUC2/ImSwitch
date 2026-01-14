// src/backendapi/apiCompositeControllerCaptureSingle.js
// Capture a single composite image (one-shot mode)

import createAxiosInstance from "./createAxiosInstance";

/**
 * Capture a single composite image
 * Executes one acquisition cycle and returns the composite image as base64 JPEG
 * 
 * @returns {Promise<Object>} Result with status, base64 image, and metadata
 * 
 * @example
 * const result = await apiCompositeControllerCaptureSingle();
 * // Returns:
 * // {
 * //   status: "success",
 * //   image_base64: "...",
 * //   metadata: {
 * //     timestamp: "2024-01-01T12:00:00",
 * //     mapping: { R: "laser635", G: "laser488", B: "" },
 * //     steps: [...],
 * //     image_shape: [512, 512, 3]
 * //   }
 * // }
 */
const apiCompositeControllerCaptureSingle = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/CompositeController/capture_single_composite");
    return response.data;
  } catch (error) {
    console.error("Error capturing single composite:", error);
    throw error;
  }
};

export default apiCompositeControllerCaptureSingle;
