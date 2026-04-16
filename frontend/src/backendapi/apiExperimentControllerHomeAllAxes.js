// src/api/apiExperimentControllerHomeAllAxes.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerHomeAllAxes = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.post("/ExperimentController/homeAllAxes");
    return response.data;
  } catch (error) {
    console.error("Error homing all axes:", error);
    throw error;
  }
};

export default apiExperimentControllerHomeAllAxes;
