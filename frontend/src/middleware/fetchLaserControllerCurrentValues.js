import * as experimentSlice from "../state/slices/ExperimentSlice.js";
import fetchLaserRuntimeState from "./fetchLaserRuntimeState.js";

/**
 * Fetches current laser intensity values from the backend and updates Redux state
 * @param {Function} dispatch - Redux dispatch function
 * @param {Object} connectionSettings - Object containing ip and apiPort
 * @param {Array} laserNames - Array of laser names to fetch values for
 */
const fetchLaserControllerCurrentValues = async (dispatch, connectionSettings, laserNames) => {
  if (!connectionSettings.ip || !connectionSettings.apiPort || !laserNames || laserNames.length === 0) {
    console.warn("Cannot fetch laser values: missing connection settings or laser names");
    return;
  }

  try {
    const runtimeStates = await fetchLaserRuntimeState({
      hostIP: connectionSettings.ip,
      hostPort: connectionSettings.apiPort,
      sources: laserNames,
      kinds: laserNames.map(() => "default"),
    });

    runtimeStates.forEach(({ laserName, ok, error }) => {
      if (!ok && error) {
        console.error(`Error fetching value for laser ${laserName}:`, error);
      }
    });

    const laserValues = runtimeStates.map(({ power }) => power);

    // Update Redux state with fetched values
    dispatch(experimentSlice.setIlluminationIntensities(laserValues));
    
    console.log("Successfully fetched laser values:", laserValues);
  } catch (error) {
    console.error("Failed to fetch laser values:", error);
  }
};

export default fetchLaserControllerCurrentValues;
