import createAxiosInstance from "./createAxiosInstance.js";


/**
 * Sets crop frame parameters for the focus lock controller.
 * Submits cropSize as query param, and cropCenter/frameSize as JSON body, as required by OpenAPI spec.
 */
const apiFocusLockControllerSetCropFrameParameters = async ({ cropSize, cropCenter, frameSize }) => {
  try {
    const axiosInstance = createAxiosInstance();

    // Always send cropSize as integer in query
    const params = { crop_size: Math.round(cropSize) };

    // Compose JSON body for cropCenter and frameSize
    const body = {};
    if (cropCenter) body.crop_center = cropCenter;
    if (frameSize && Array.isArray(frameSize) && frameSize.length === 2) body.frameSize = frameSize;

    // POST with cropSize as query param, body as JSON
    const response = await axiosInstance.post(
      '/FocusLockController/setCropFrameParameters',
      body,
      {
        params,
        headers: { 'Content-Type': 'application/json' }
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error setting crop frame parameters:", error);
    throw error;
  }
};

export default apiFocusLockControllerSetCropFrameParameters;