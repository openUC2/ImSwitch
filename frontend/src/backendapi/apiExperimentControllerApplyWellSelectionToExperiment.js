// src/backendapi/apiExperimentControllerApplyWellSelectionToExperiment.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * POST /ExperimentController/applyWellSelectionToExperiment
 *
 * Resolve a pattern and return point dicts ready to be appended to the
 * experiment's pointList. Each point carries wellId / wellRow / wellColumn /
 * labwareLoadName / conditionLabel so that the OME-NGFF plate sidecar can be
 * emitted at acquisition time.
 *
 * @param {Object} args
 * @param {string} args.loadName
 * @param {Object} args.pattern
 * @param {number} [args.offsetXUm=0]
 * @param {number} [args.offsetYUm=0]
 * @param {Object<string,string>} [args.conditionLabels] - map well_id -> label
 * @param {string} [args.pointNameTemplate="{well_id}"]
 * @returns {Promise<{load_name: string, count: number, points: Array<Object>}>}
 */
const apiExperimentControllerApplyWellSelectionToExperiment = async ({
  loadName,
  pattern,
  offsetXUm = 0,
  offsetYUm = 0,
  conditionLabels = {},
  pointNameTemplate = "{well_id}",
}) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.post(
      "/ExperimentController/applyWellSelectionToExperiment",
      {
        load_name: loadName,
        pattern: pattern || {},
        offset_x_um: offsetXUm,
        offset_y_um: offsetYUm,
        condition_labels: conditionLabels,
        point_name_template: pointNameTemplate,
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error applying well selection to experiment:", error);
    throw error;
  }
};

export default apiExperimentControllerApplyWellSelectionToExperiment;
