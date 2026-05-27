// Discard a pending (un-applied) calibration
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerDiscardPendingCalibration = async (detectorName, objectiveId) => {
  const axiosInstance = createAxiosInstance();
  const params = { detectorName };
  if (objectiveId !== null && objectiveId !== undefined && objectiveId !== '') {
    params.objectiveId = String(objectiveId);
  }
  const response = await axiosInstance.post(
    '/PixelCalibrationController/discardPendingCalibration',
    null,
    { params }
  );
  return response.data;
};

export default apiPixelCalibrationControllerDiscardPendingCalibration;
