// src/backendapi/apiExperimentControllerGetLabwareList.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * GET /ExperimentController/getLabwareList
 *
 * Returns an array of summary objects describing every labware definition
 * the LabwareManager has loaded. Each entry includes loadName, displayName,
 * format, brand, dimensions (µm), well counts and tags.
 *
 * @returns {Promise<Array<Object>>}
 */
const apiExperimentControllerGetLabwareList = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/ExperimentController/getLabwareList");
    return Array.isArray(response.data) ? response.data : [];
  } catch (error) {
    console.error("Error fetching labware list:", error);
    throw error;
  }
};

export default apiExperimentControllerGetLabwareList;
