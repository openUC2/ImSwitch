// src/backendapi/apiExperimentControllerGetKnownCalibrationPoint.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * GET /ExperimentController/getKnownCalibrationPoint?layout_name=...
 */
const apiExperimentControllerGetKnownCalibrationPoint = async (layoutName) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/ExperimentController/getKnownCalibrationPoint",
      { params: { layout_name: layoutName } }
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching known calibration point:", error);
    throw error;
  }
};

export default apiExperimentControllerGetKnownCalibrationPoint;
