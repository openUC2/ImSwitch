// src/backendapi/apiPixelCalibrationControllerManualPixelSizeCalibration.js
// Perform manual two-point pixel size calibration

import createAxiosInstance from './createAxiosInstance';

/**
 * Submit a manual pixel-size calibration based on two marked points
 * and a known stage movement distance.
 *
 * @param {Object} params
 * @param {number} params.point1X  - X pixel of marked feature BEFORE movement
 * @param {number} params.point1Y  - Y pixel of marked feature BEFORE movement
 * @param {number} params.point2X  - X pixel of marked feature AFTER movement
 * @param {number} params.point2Y  - Y pixel of marked feature AFTER movement
 * @param {number} params.movementDistanceUm - Known stage movement in µm
 * @param {string} params.movementAxis       - "X" or "Y"
 * @param {string} [params.objectiveId]      - Objective name (optional)
 * @returns {Promise<Object>} Result with pixelSizeUm, displacementPx, etc.
 */
const apiPixelCalibrationControllerManualPixelSizeCalibration = async ({
  point1X,
  point1Y,
  point2X,
  point2Y,
  movementDistanceUm,
  movementAxis,
  detectorName,
  objectiveId,
  previewSubsamplingFactor,
}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    '/PixelCalibrationController/manualPixelSizeCalibration',
    null,
    {
      params: {
        point1X,
        point1Y,
        point2X,
        point2Y,
        movementDistanceUm,
        movementAxis,
        ...(detectorName != null && detectorName !== '' && { detectorName }),
        ...(objectiveId != null && { objectiveId }),
        ...(previewSubsamplingFactor != null && { previewSubsamplingFactor }),
      },
    }
  );
  return response.data;
};

export default apiPixelCalibrationControllerManualPixelSizeCalibration;
