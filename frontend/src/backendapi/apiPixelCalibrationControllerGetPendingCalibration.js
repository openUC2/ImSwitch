// Poll for a pending (un-applied) calibration result
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerGetPendingCalibration = async (detectorName, objectiveId) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (detectorName) params.detectorName = detectorName;
  if (objectiveId !== null && objectiveId !== undefined && objectiveId !== '') {
    params.objectiveId = String(objectiveId);
  }
  const response = await axiosInstance.get(
    '/PixelCalibrationController/getPendingCalibration',
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerGetPendingCalibration;
