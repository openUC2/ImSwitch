// src/backendapi/apiExperimentControllerMeasureFocusMapFromPoints.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Autofocus-measure Z at each manual point (focusMapConfig.points) and fit.
 *
 * This is a single BLOCKING backend call (the move→autofocus→fit loop runs on
 * the backend), so the browser just awaits it instead of looping per-point.
 *
 * @param {Object} focusMapConfig - FocusMapConfig including `points: [{x,y,z?}]`
 *                                   and the autofocus parameters.
 * @returns {Promise<Object>} { manual: <FocusMapResult> }
 */
const apiExperimentControllerMeasureFocusMapFromPoints = async (
  focusMapConfig = null,
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/ExperimentController/measureFocusMapFromPoints",
    focusMapConfig,
  );
  return response.data;
};

export default apiExperimentControllerMeasureFocusMapFromPoints;
