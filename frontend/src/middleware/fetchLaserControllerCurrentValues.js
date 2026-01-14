import createAxiosInstance from '../backendapi/createAxiosInstance';
import * as experimentSlice from "../state/slices/ExperimentSlice.js";

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
    const api = createAxiosInstance();
    // Fetch current values for each laser
    const laserValues = await Promise.all(
      laserNames.map(async (laserName, index) => {
        try {
          const encodedLaserName = encodeURIComponent(laserName);
          const response = await api.get(
            `/LaserController/getLaserValue?laserName=${encodedLaserName}`
          );
          
          const value = response.data;
          return typeof value === 'number' ? value : 0;
        } catch (error) {
          console.error(`Error fetching value for laser ${laserName}:`, error);
          return 0; // Default value if fetch fails
        }
      })
    );

    // Update Redux state with fetched values
    dispatch(experimentSlice.setIlluminationIntensities(laserValues));
    
    console.log("Successfully fetched laser values:", laserValues);
  } catch (error) {
    console.error("Failed to fetch laser values:", error);
  }
};

export default fetchLaserControllerCurrentValues;
