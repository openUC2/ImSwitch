// src/backendapi/apiExperimentControllerRecaptureSlot.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * POST /ExperimentController/recaptureSlot
 *
 * Move the stage to the requested slot's center and snap a fresh overview
 * frame, replacing only that single slot's snapshot.
 *
 * @param {Object} params
 * @param {string|number} params.slot_id  - 1-based slot index ("1" .. "4")
 * @param {Object} [params.layout_data]    - Optional full wellLayout dict
 * @param {string} [params.layout_name]    - Fallback layout name
 * @param {string} [params.camera_name]    - Optional explicit overview cam
 */
const apiExperimentControllerRecaptureSlot = async ({
  slot_id,
  layout_data,
  layout_name,
  camera_name,
} = {}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = { slot_id: String(slot_id) };
    if (layout_name) params.layout_name = layout_name;
    if (camera_name) params.camera_name = camera_name;

    const response = await axiosInstance.post(
      "/ExperimentController/recaptureSlot",
      layout_data || {},
      { params }
    );
    return response.data;
  } catch (error) {
    console.error("Error recapturing slot:", error);
    throw error;
  }
};

export default apiExperimentControllerRecaptureSlot;
