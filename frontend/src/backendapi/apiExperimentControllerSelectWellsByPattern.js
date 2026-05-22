// src/backendapi/apiExperimentControllerSelectWellsByPattern.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * POST /ExperimentController/selectWellsByPattern
 *
 * Resolve a WellSelectionPattern (rows / columns / wells / ranges / all) against
 * a labware. Pure function – does NOT mutate experiment state. Use this for
 * preview / interactive selection in the UI.
 *
 * @param {Object} args
 * @param {string} args.loadName
 * @param {Object} args.pattern - { wells?, rows?, columns?, ranges?, all? }
 * @param {number} [args.offsetXUm=0]
 * @param {number} [args.offsetYUm=0]
 * @returns {Promise<{load_name: string, count: number, wells: Array<Object>}>}
 */
const apiExperimentControllerSelectWellsByPattern = async ({
  loadName,
  pattern,
  offsetXUm = 0,
  offsetYUm = 0,
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.post(
      "/ExperimentController/selectWellsByPattern",
      {
        load_name: loadName,
        pattern: pattern || {},
        offset_x_um: offsetXUm,
        offset_y_um: offsetYUm,
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error resolving well selection pattern:", error);
    throw error;
  }
};

export default apiExperimentControllerSelectWellsByPattern;
