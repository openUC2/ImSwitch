// src/backendapi/apiExperimentControllerGetKnownCalibrationLayouts.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerGetKnownCalibrationLayouts = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/ExperimentController/getKnownCalibrationLayouts"
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching known calibration layouts:", error);
    throw error;
  }
};

export default apiExperimentControllerGetKnownCalibrationLayouts;
