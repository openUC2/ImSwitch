import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerManualFourPointCalibration = async ({
  pointA1X, pointA1Y,
  pointA2X, pointA2Y,
  movementDistanceXUm,
  pointB1X, pointB1Y,
  pointB2X, pointB2Y,
  movementDistanceYUm,
  detectorName,
  objectiveId,
  previewSubsamplingFactor,
}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    '/PixelCalibrationController/manualFourPointCalibration',
    null,
    {
      params: {
        pointA1X, pointA1Y,
        pointA2X, pointA2Y,
        movementDistanceXUm,
        pointB1X, pointB1Y,
        pointB2X, pointB2Y,
        movementDistanceYUm,
        ...(detectorName != null && detectorName !== '' && { detectorName }),
        ...(objectiveId != null && { objectiveId }),
        ...(previewSubsamplingFactor != null && { previewSubsamplingFactor }),
      },
    }
  );
  return response.data;
};

export default apiPixelCalibrationControllerManualFourPointCalibration;
