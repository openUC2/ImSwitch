// Read persisted calibration data for a (detector, objective) pair
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGetCalibrationData = async (
  detectorName,
  objectiveId,
) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (detectorName) params.detectorName = detectorName;
  if (objectiveId !== null && objectiveId !== undefined && objectiveId !== '') {
    params.objectiveId = String(objectiveId);
  }
  const response = await axiosInstance.get(
    '/PixelCalibrationController/getCalibrationData',
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerGetCalibrationData;
