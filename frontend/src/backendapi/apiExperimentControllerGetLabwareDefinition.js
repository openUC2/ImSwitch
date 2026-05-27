// src/backendapi/apiExperimentControllerGetLabwareDefinition.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * GET /ExperimentController/getLabwareDefinition
 *
 * Fetch the full LabwareDefinition (µm) for a single labware. Optional offsets
 * shift every well's x/y from the plate frame to the stage frame.
 *
 * @param {string} loadName - Labware loadName, e.g. "corning_96_wellplate_360ul_flat"
 * @param {number} [offsetXUm=0]
 * @param {number} [offsetYUm=0]
 * @returns {Promise<Object>}
 */
const apiExperimentControllerGetLabwareDefinition = async (
  loadName,
  offsetXUm = 0,
  offsetYUm = 0
) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/ExperimentController/getLabwareDefinition",
      {
        params: {
          load_name: loadName,
          offset_x_um: offsetXUm,
          offset_y_um: offsetYUm,
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error(`Error fetching labware definition '${loadName}':`, error);
    throw error;
  }
};

export default apiExperimentControllerGetLabwareDefinition;
