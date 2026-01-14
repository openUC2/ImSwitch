// src/backendapi/apiPositionerControllerSetStageOffsetAxis.js
// API to set stage offset axis with known position (backend calculates offset)
import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the stage offset for a specific axis using a known position.
 * The backend will calculate the offset based on the difference between
 * the known position and the current true position.
 * 
 * @param {Object} params - Parameters for setting stage offset
 * @param {string} [params.positionerName] - Name of the positioner (optional)
 * @param {number} params.knownPosition - The known/expected position at the current stage location
 * @param {number} [params.currentPosition] - Current position (optional, backend uses true position if not provided)
 * @param {number} [params.knownOffset] - Direct offset value (alternative to knownPosition)
 * @param {string} [params.axis="X"] - Axis to set offset for ("X" or "Y")
 * @returns {Promise<Object>} - Response from the backend
 */
const apiPositionerControllerSetStageOffsetAxis = async ({
  positionerName,
  knownPosition,
  currentPosition,
  knownOffset,
  axis = "X",
}) => {
  try {
    const axiosInstance = createAxiosInstance();

    // Build the query string dynamically
    let url = "/PositionerController/setStageOffsetAxis?";
    const queryParams = [];

    if (positionerName !== undefined && positionerName !== null) {
      queryParams.push(`positionerName=${encodeURIComponent(positionerName)}`);
    }
    if (knownPosition !== undefined && knownPosition !== null) {
      queryParams.push(`knownPosition=${encodeURIComponent(knownPosition)}`);
    }
    if (currentPosition !== undefined && currentPosition !== null) {
      queryParams.push(`currentPosition=${encodeURIComponent(currentPosition)}`);
    }
    if (knownOffset !== undefined && knownOffset !== null) {
      queryParams.push(`knownOffset=${encodeURIComponent(knownOffset)}`);
    }
    if (axis) {
      queryParams.push(`axis=${encodeURIComponent(axis)}`);
    }

    // Join all query parameters with '&'
    url += queryParams.join("&");

    // Send GET request
    const response = await axiosInstance.get(url);

    return response.data;
  } catch (error) {
    console.error("Error setting stage offset axis:", error);
    throw error;
  }
};

export default apiPositionerControllerSetStageOffsetAxis;
