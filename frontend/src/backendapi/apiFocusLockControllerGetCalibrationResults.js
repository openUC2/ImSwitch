// src/backendapi/apiFocusLockControllerGetCalibrationResults.js
// API to get calibration results from the FocusLock controller
import createAxiosInstance from "./createAxiosInstance";

/**
 * Get calibration results from the FocusLock controller.
 * Returns calibration data including position vs focus data, polynomial coefficients,
 * R-squared value, and sensitivity.
 * 
 * @returns {Promise<Object>} - Calibration results containing:
 *   - signalData: Array of focus signal values
 *   - positionData: Array of Z positions
 *   - poly: Polynomial coefficients for fit
 *   - calibrationResult: Calibration result values
 *   - r_squared: R-squared value of the fit
 *   - sensitivity_nm_per_px: Sensitivity in nm per pixel unit
 *   - calibration_data: Detailed calibration data object
 *   - pid_integration_active: Whether PID integration is active
 */
const apiFocusLockControllerGetCalibrationResults = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/FocusLockController/getCalibrationResults");
    return response.data;
  } catch (error) {
    console.error("Error getting calibration results:", error);
    throw error;
  }
};

export default apiFocusLockControllerGetCalibrationResults;
